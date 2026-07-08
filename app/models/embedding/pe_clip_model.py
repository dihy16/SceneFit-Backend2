# app/models/pe_clip_model.py

import torch
import torch.nn.functional as F
from PIL import Image

import app.services.pe_core.vision_encoder.pe as pe
import app.services.pe_core.vision_encoder.transforms as transforms
from app.models.embedding.clip_subspace import ClipSubspace, orthogonalize_subspaces

from app.utils.device import *
from app.utils.util import load_prompts, load_images_from_folder


class PEClipModel:
    """
    PE-Core CLIP-based image-text similarity model.
    API-compatible with MmEmbModel.
    """

    def __init__(
        self,
        config_name: str = "PE-Core-L14-336",
        device: str | None = None,
        autocast: bool = True,
    ):
        self.device = resolve_device(device)
        self.device_type = self.device.type
        self.use_autocast = autocast and resolve_autocast(self.device)
        self.autocast_dtype = resolve_dtype(self.device)

        # -------------------------
        # Load model
        # -------------------------
        self.model = pe.CLIP.from_config(
            config_name,
            pretrained=True,
        ).to(self.device)
        self.model.eval()

        # -------------------------
        # Transforms
        # -------------------------
        self.image_transform = transforms.get_image_transform(
            self.model.image_size
        )
        self.text_tokenizer = transforms.get_text_tokenizer(
            self.model.context_length
        )

        # # -------------------------
        # # Load prompts & encode once
        # # -------------------------
        # self.prompts = load_prompts("outfit_match")
        # self.clothing_emb = self.encode_text("clothing")
        # self.env_emb = self.encode_text("background")
        # raw_prompt_embeddings = self.encode_text(self.prompts)

        # self.prompt_embeddings = (
        #     raw_prompt_embeddings
        #     - self.clothing_emb.unsqueeze(0)
        #     - self.env_emb.unsqueeze(0)
        # )

        # self.background_images = load_images_from_folder("data/bg")
        # self.clothing_images = load_images_from_folder("data/2d")


    # -------------------------------------------------
    # Text encoding
    # -------------------------------------------------
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(texts).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            _, text_features, _ = self.model(
                image=None,
                text=tokens,
            )

        return self._normalize(text_features)

    # -------------------------------------------------
    # Image encoding
    # -------------------------------------------------
    def encode_image(self, images) -> torch.Tensor:
        """
        images: PIL.Image | list[PIL.Image]
        """
        if isinstance(images, Image.Image):
            images = [images]

        image_tensor = torch.stack(
            [self.image_transform(img) for img in images]
        ).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            image_features, _, _ = self.model(
                image=image_tensor,
                text=None,
            )

        return self._normalize(image_features)

    # -------------------------------------------------
    # Similarity
    # -------------------------------------------------
    @staticmethod
    def similarity(text_embeddings, image_embeddings):
        return text_embeddings @ image_embeddings.T

    @staticmethod
    def _normalize(x):
        return F.normalize(x, p=2, dim=-1)
    
    def reload_prompts(self, prompt_name: str = "outfit_match"):
        self.prompts = load_prompts(prompt_name)
        self.prompt_embeddings = self.encode_text(self.prompts)

    # -------------------------------------------------
    # Scoring helpers (same API as MmEmbModel)
    # -------------------------------------------------
    def score_image(self, image: Image.Image):
        image_emb = self.encode_image(image)

        pos_emb = self.prompt_embeddings[0]
        neg_emb = self.prompt_embeddings[1]

        pos_score = (image_emb @ pos_emb.T).item()
        neg_score = (image_emb @ neg_emb.T).item()

        return {
            "positive_score": pos_score,
            "negative_score": neg_score,
            "confidence": pos_score - neg_score,
        }

    def score_images(self, items: list[tuple[str, Image.Image]]):
        
        self.reload_prompts()
        
        names, images = zip(*items)
        image_embeddings = self.encode_image(list(images))

        image_embeddings = self.remove_bg_and_clothes(image_embeddings)        
        
        pos_emb = self.prompt_embeddings[0]
        neg_emb = self.prompt_embeddings[1]

        results = []

        for idx, img_emb in enumerate(image_embeddings):
            pos_score = (img_emb @ pos_emb.T).item()
            #neg_score = (img_emb @ neg_emb.T).item()

            results.append({
                "name_clothes": names[idx],
                "positive_score": pos_score,
                "negative_score": 0,
                "confidence": pos_score #- neg_score,
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
        
    def remove_bg_and_clothes(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Remove both background and clothing subspaces.
        Keeps only the residual signal.
        """
        # Remove background
        image_embeddings = self.bg_subspace.remove(image_embeddings)

        # Remove clothing (already orthogonal to background)
        image_embeddings = self.clothes_subspace.remove(image_embeddings)

        return F.normalize(image_embeddings, dim=-1)

    def build_image_subspace(
        self,
        images: list[Image.Image],
        num_components: int | None = None,
        variance_ratio: float | None = 0.9,
    ) -> ClipSubspace:
        """
        Build a PCA/SVD subspace from a list of images.
        """

        # 1. Encode images
        with torch.no_grad():
            X = self.encode_image(images)  # [n, d]

        X = X.float().cpu()

        # 2. Center (important!)
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean

        # 3. SVD
        # Xc = U Σ Vᵀ → components are V
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

        # 4. Choose dimensionality
        if num_components is not None:
            k = num_components
        else:
            # variance-based selection
            var = S**2
            cumvar = var.cumsum(0) / var.sum()
            k = int((cumvar <= variance_ratio).sum().item()) + 1

        basis = Vh[:k].T  # [d, k]

        return ClipSubspace(basis=basis)
