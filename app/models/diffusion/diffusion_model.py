# app/models/diffusion_model.py

from __future__ import annotations

from pathlib import Path
import os
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline

from app.utils.image_utils import compose_2d_on_background
from app.utils.device import resolve_device, resolve_dtype, resolve_autocast
from app.utils.util import load_prompt_by_key


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

logger = logging.getLogger(__name__)

def _chunked(items: list, size: int):
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


#"stabilityai/stable-diffusion-3.5-medium"

class DiffusionModel:
    """
    Ranks composed background+figure images by how well a pretrained diffusion
    UNet predicts a fixed noise pattern for a given prompt.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        pipeline_type: str | None = None,
        hf_token: str | None = None,
        text_encoder_only: bool = True,
        image_size: int = 512,
        num_inference_steps: int = 10,
        noise_step_index: int | None = None,
        guidance_scale: float = 1.0,
        batch_size: int = 1,
        noise_seed: int = 1234,
        enable_xformers: bool = True,
    ):
        self.model_id = model_id
        self.pipeline_type = pipeline_type or self._infer_pipeline_type(model_id)
        self.device = resolve_device(device)
        self.device_type = self.device.type
        self.use_autocast = resolve_autocast(self.device)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.text_encoder_only = text_encoder_only

        if dtype is None:
            if self.device_type == "cuda":
                if self.pipeline_type == "sd3" and torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
            else:
                self.dtype = resolve_dtype(self.device)
        else:
            self.dtype = dtype

        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.noise_step_index = (
            noise_step_index
            if noise_step_index is not None
            else max(0, num_inference_steps // 2)
        )
        self.guidance_scale = guidance_scale
        self.batch_size = max(1, batch_size)
        self.noise_seed = noise_seed
        self.enable_xformers = enable_xformers

        self.prompt = self._load_prompt_text()

        self._pipe = None
        self._denoiser = None
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None

        self._prompt_embeds = None
        self._pooled_prompt_embeds = None
        self._uncond_embeds = None
        self._neg_pooled_prompt_embeds = None
        self._noise_timestep = None

        self._load_pipeline()

    def load(self):
        return None

    @staticmethod
    def _infer_pipeline_type(model_id: str) -> str:
        lowered = model_id.lower()
        if "stable-diffusion-3" in lowered or "sd3" in lowered:
            return "sd3"
        return "sd15"

    def _load_pipeline(self):
        if self.pipeline_type == "sd3":
            logger.info("Loading SD3 pipeline: %s", self.model_id)
        kwargs = {"torch_dtype": self.dtype}
        if self.dtype == torch.float16:
            kwargs["variant"] = "fp16"
        if self.hf_token:
            kwargs["token"] = self.hf_token

        def _load_pipe(load_kwargs):
            try:
                if self.pipeline_type == "sd3":
                    return StableDiffusion3Pipeline.from_pretrained(
                        self.model_id,
                        **load_kwargs,
                    )
                return StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    safety_checker=None,
                    requires_safety_checker=False,
                    **load_kwargs,
                )
            except TypeError:
                if self.pipeline_type == "sd3":
                    return StableDiffusion3Pipeline.from_pretrained(
                        self.model_id,
                        **load_kwargs,
                    )
                return StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    safety_checker=None,
                    **load_kwargs,
                )

        try:
            self._pipe = _load_pipe(kwargs)
        except OSError:
            kwargs.pop("variant", None)
            self._pipe = _load_pipe(kwargs)
        if not self.text_encoder_only:
            self._pipe.to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.enable_attention_slicing()
        self._pipe.enable_vae_slicing()

        if self.enable_xformers:
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        if self.pipeline_type == "sd3":
            self._denoiser = self._pipe.transformer.eval()
        else:
            self._denoiser = self._pipe.unet.eval()
        self._vae = self._pipe.vae.eval()
        if self.pipeline_type != "sd3":
            self._text_encoder = self._pipe.text_encoder.eval()
            self._tokenizer = self._pipe.tokenizer
        self._scheduler = self._pipe.scheduler
        self._scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        if self.text_encoder_only:
            self._encode_prompt_with_text_encoder_only()
        else:
            if self.pipeline_type == "sd3":
                (
                    self._prompt_embeds,
                    self._pooled_prompt_embeds,
                    self._uncond_embeds,
                    self._neg_pooled_prompt_embeds,
                ) = self._encode_prompt_sd3(self.prompt)
            else:
                self._prompt_embeds = self._encode_prompt(self.prompt)
                if self.guidance_scale > 1.0:
                    self._uncond_embeds = self._encode_prompt("")

        step_index = min(
            max(0, self.noise_step_index),
            len(self._scheduler.timesteps) - 1,
        )
        noise_timestep = self._scheduler.timesteps[step_index]
        if not torch.is_tensor(noise_timestep):
            noise_timestep = torch.tensor(noise_timestep)
        self._noise_timestep = noise_timestep.to(self.device)
        if self.pipeline_type == "sd3":
            self._log_sd3_devices("after_load")

    def _load_prompt_text(self) -> str:
        prompt_cfg = load_prompt_by_key("diffusion_prompt")

        if isinstance(prompt_cfg, dict):
            prompt = prompt_cfg.get("positive", "")
        elif isinstance(prompt_cfg, str):
            prompt = prompt_cfg
        else:
            raise ValueError("diffusion_prompt must be a string or dict")

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("diffusion_prompt is empty")

        return prompt

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        inputs = self._tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=self._tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.use_autocast,
        ):
            prompt_embeds = self._text_encoder(input_ids)[0]

        return prompt_embeds

    def _encode_prompt_sd3(
        self,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        encode_fn = getattr(self._pipe, "encode_prompt", None)
        if encode_fn is None:
            encode_fn = getattr(self._pipe, "_encode_prompt", None)
        if encode_fn is None:
            raise RuntimeError("SD3 pipeline missing encode_prompt")

        do_cfg = self.guidance_scale > 1.0
        kwargs = {
            "prompt": prompt,
            "prompt_2": None,
            "prompt_3": None,
            "device": self.device,
            "num_images_per_prompt": 1,
            "do_classifier_free_guidance": do_cfg,
            "negative_prompt": negative_prompt or "",
            "negative_prompt_2": None,
            "negative_prompt_3": None,
        }

        try:
            encoded = encode_fn(**kwargs)
        except TypeError:
            kwargs.pop("prompt_2", None)
            kwargs.pop("prompt_3", None)
            kwargs.pop("negative_prompt_2", None)
            kwargs.pop("negative_prompt_3", None)
            encoded = encode_fn(**kwargs)

        if not isinstance(encoded, tuple):
            raise RuntimeError("SD3 encode_prompt returned unexpected output")

        if len(encoded) == 4:
            prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = encoded
        elif len(encoded) == 2:
            prompt_embeds, pooled = encoded
            neg_prompt_embeds = None
            neg_pooled = None
        else:
            raise RuntimeError("SD3 encode_prompt returned unexpected tuple size")

        return prompt_embeds, pooled, neg_prompt_embeds, neg_pooled

    def _encode_prompt_with_text_encoder_only(self):
        if self.pipeline_type == "sd3":
            self._log_sd3_devices("before_text_encode")
        self._move_text_encoders(self.device)

        if self.pipeline_type == "sd3":
            (
                self._prompt_embeds,
                self._pooled_prompt_embeds,
                self._uncond_embeds,
                self._neg_pooled_prompt_embeds,
            ) = self._encode_prompt_sd3(self.prompt)
        else:
            self._prompt_embeds = self._encode_prompt(self.prompt)
            if self.guidance_scale > 1.0:
                self._uncond_embeds = self._encode_prompt("")

        self._move_text_encoders("cpu")
        if self.pipeline_type == "sd3":
            self._log_sd3_devices("after_text_offload")
        self._drop_text_encoders()
        if self.pipeline_type == "sd3":
            self._move_module(self._pipe.transformer, self.device)
        else:
            self._move_module(self._pipe.unet, self.device)
        self._move_module(self._pipe.vae, self.device)
        if self.pipeline_type == "sd3":
            self._log_sd3_devices("after_denoiser_load")

        if self.pipeline_type != "sd3":
            self._text_encoder = None
            self._tokenizer = self._pipe.tokenizer

    def _move_text_encoders(self, device: str | torch.device):
        if self.pipeline_type == "sd3":
            self._move_module(getattr(self._pipe, "text_encoder", None), device)
            self._move_module(getattr(self._pipe, "text_encoder_2", None), device)
            self._move_module(getattr(self._pipe, "text_encoder_3", None), device)
        else:
            self._move_module(getattr(self._pipe, "text_encoder", None), device)

    @staticmethod
    def _move_module(module, device: str | torch.device):
        if module is None:
            return
        module.to(device)

    def _drop_text_encoders(self):
        if self.pipeline_type == "sd3":
            if hasattr(self._pipe, "text_encoder"):
                del self._pipe.text_encoder
            if hasattr(self._pipe, "text_encoder_2"):
                del self._pipe.text_encoder_2
            if hasattr(self._pipe, "text_encoder_3"):
                del self._pipe.text_encoder_3
        else:
            if hasattr(self._pipe, "text_encoder"):
                del self._pipe.text_encoder
        if self._text_encoder is not None:
            self._text_encoder = None
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
            logger.info(
                "CUDA memory allocated after drop: %.2f GB",
                torch.cuda.memory_allocated() / (1024 ** 3),
            )

    @staticmethod
    def _module_device(module) -> str:
        if module is None:
            return "none"
        try:
            param = next(module.parameters())
        except StopIteration:
            return "no-params"
        return str(param.device)

    def _log_sd3_devices(self, stage: str):
        logger.info(
            "SD3 devices @ %s | te1=%s te2=%s te3=%s denoiser=%s vae=%s embeds=%s pooled=%s",
            stage,
            self._module_device(getattr(self._pipe, "text_encoder", None)),
            self._module_device(getattr(self._pipe, "text_encoder_2", None)),
            self._module_device(getattr(self._pipe, "text_encoder_3", None)),
            self._module_device(getattr(self._pipe, "transformer", None)),
            self._module_device(getattr(self._pipe, "vae", None)),
            "none" if self._prompt_embeds is None else str(self._prompt_embeds.device),
            "none" if self._pooled_prompt_embeds is None else str(self._pooled_prompt_embeds.device),
        )

    def _prepare_image_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        if self.image_size is not None:
            size = max(8, self.image_size - (self.image_size % 8))
            image = image.resize((size, size), Image.BICUBIC)
        else:
            width, height = image.size
            width = max(8, width - (width % 8))
            height = max(8, height - (height % 8))
            image = image.resize((width, height), Image.BICUBIC)

        image_array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor

    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        image_tensors = torch.cat(
            [self._prepare_image_tensor(img) for img in images],
            dim=0,
        ).to(self.device, dtype=self.dtype)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.use_autocast,
        ):
            latents = self._vae.encode(image_tensors).latent_dist.mean
            latents = latents * self._vae.config.scaling_factor

        return latents

    def _get_prompt_embeds(self, batch_size: int) -> torch.Tensor:
        if self.pipeline_type == "sd3":
            prompt_embeds = self._prompt_embeds.repeat(batch_size, 1, 1)
            pooled = self._pooled_prompt_embeds.repeat(batch_size, 1)

            if self.guidance_scale > 1.0 and self._uncond_embeds is not None:
                uncond_embeds = self._uncond_embeds.repeat(batch_size, 1, 1)
                uncond_pooled = self._neg_pooled_prompt_embeds.repeat(batch_size, 1)
                prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                pooled = torch.cat([uncond_pooled, pooled], dim=0)

            return prompt_embeds, pooled

        prompt_embeds = self._prompt_embeds.repeat(batch_size, 1, 1)

        if self.guidance_scale > 1.0:
            uncond_embeds = self._uncond_embeds.repeat(batch_size, 1, 1)
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)

        return prompt_embeds

    def _predict_noise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        latent_model_input = latents
        timesteps_input = timesteps

        if self.guidance_scale > 1.0:
            latent_model_input = torch.cat([latents] * 2)
            timesteps_input = timesteps.repeat(2)

        prompt_embeds = self._get_prompt_embeds(batch_size)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.use_autocast,
        ):
            if self.pipeline_type == "sd3":
                text_embeds, pooled = prompt_embeds
                output = self._denoiser(
                    latent_model_input,
                    timesteps_input,
                    encoder_hidden_states=text_embeds,
                    pooled_projections=pooled,
                )
            else:
                output = self._denoiser(
                    latent_model_input,
                    timesteps_input,
                    encoder_hidden_states=prompt_embeds,
                )

            if hasattr(output, "sample"):
                noise_pred = output.sample
            elif isinstance(output, tuple):
                noise_pred = output[0]
            else:
                noise_pred = output

        if self.guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    def _list_figure_files(self, figures_dir: str | Path) -> List[str]:
        figures_dir = Path(figures_dir)
        if not figures_dir.exists():
            raise FileNotFoundError(f"Figures directory not found: {figures_dir}")

        files = [
            path.name
            for path in sorted(figures_dir.iterdir())
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        ]

        if not files:
            raise RuntimeError(f"No images found in {figures_dir}")

        return files

    def compose_candidates(
        self,
        background_path: str | Path,
        figures_dir: str | Path = "data/2d",
    ) -> List[Tuple[str, Image.Image]]:
        figure_files = self._list_figure_files(figures_dir)
        return compose_2d_on_background(
            bg_path=str(background_path),
            fg_dir=str(figures_dir),
            fg_files=figure_files,
            return_format="pil",
        )

    def score_images(
        self,
        items: List[Tuple[str, Image.Image]],
    ) -> List[dict]:
        if not items:
            return []

        results = []
        fixed_noise = None
        generator = torch.Generator(device=self.device).manual_seed(self.noise_seed)
        offset = 0

        for batch in _chunked(items, self.batch_size):
            names, images = zip(*batch)
            latents = self._encode_images(list(images))

            if fixed_noise is None:
                fixed_noise = torch.randn(
                    (1, latents.shape[1], latents.shape[2], latents.shape[3]),
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )

            noise = fixed_noise.repeat(latents.shape[0], 1, 1, 1)
            timesteps = self._noise_timestep.repeat(latents.shape[0])

            noisy_latents = self._scheduler.add_noise(latents, noise, timesteps)
            latent_model_input = self._scheduler.scale_model_input(
                noisy_latents,
                timesteps,
            )

            noise_pred = self._predict_noise(
                latents=latent_model_input,
                timesteps=timesteps,
                batch_size=latents.shape[0],
            )

            mse = (noise_pred.float() - noise.float()).pow(2)
            mse = mse.mean(dim=(1, 2, 3)).tolist()

            for idx, (name, score) in enumerate(zip(names, mse)):
                results.append(
                    {
                        "index": offset + idx,
                        "name": name,
                        "mse": float(score),
                    }
                )

            offset += len(batch)

        results.sort(key=lambda x: x["mse"])
        return results

    def select_best(
        self,
        background_path: str | Path,
        figures_dir: str | Path = "data/2d",
    ) -> tuple[Image.Image, dict]:
        items = self.compose_candidates(background_path, figures_dir)
        scores = self.score_images(items)

        if not scores:
            raise RuntimeError("No scored items to select from")

        best = scores[0]
        best_image = items[best["index"]][1]

        return best_image, best
