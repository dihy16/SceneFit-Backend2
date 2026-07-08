# app/models/negative_pe_model.py

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple

import app.services.pe_core.vision_encoder.pe as pe
import app.services.pe_core.vision_encoder.transforms as transforms
from app.models.embedding.clip_subspace import ClipSubspace, orthogonalize_subspaces

from app.utils.device import (
    resolve_device,
    resolve_autocast,
    resolve_dtype,
)
from app.utils.util import load_images_from_folder, load_prompts


class NegativePEModel:
    """
    PE-Core CLIP with:
    - background subspace removal
    - hard negative text sampling
    - score-level contrastive inference

    Final score:
        score = pos_score - lambda_neg * neg_score
    """

    def __init__(
        self,
        config_name: str = "PE-Core-L14-336",
        device: str | None = None,
        autocast: bool = True,
        lambda_neg: float = 0.4,
    ):
        # -------------------------------------------------
        # Device
        # -------------------------------------------------
        self.device = resolve_device(device)
        self.device_type = self.device.type
        self.use_autocast = autocast and resolve_autocast(self.device)
        self.autocast_dtype = resolve_dtype(self.device)
        self.lambda_neg = lambda_neg

        # -------------------------------------------------
        # Model
        # -------------------------------------------------
        self.model = pe.CLIP.from_config(
            config_name,
            pretrained=True,
        ).to(self.device)
        self.model.eval()

        # -------------------------------------------------
        # Transforms
        # -------------------------------------------------
        self.image_transform = transforms.get_image_transform(
            self.model.image_size
        )
        self.text_tokenizer = transforms.get_text_tokenizer(
            self.model.context_length
        )
        # -------------------------
        # Load prompts & encode once
        # -------------------------
        self.prompts = load_prompts("outfit_match")
        self.clothing_emb = self.encode_text(["clothing"])[0]
        self.env_emb = self.encode_text(["background"])[0]

        raw_prompt_embeddings = self.encode_text(self.prompts)

        self.prompt_embeddings = (
            raw_prompt_embeddings
            - self.clothing_emb.unsqueeze(0)
            - self.env_emb.unsqueeze(0)
        )
        # -------------------------------------------------
        # Background subspace (ONLY background)
        # -------------------------------------------------
        self.background_images = load_images_from_folder("data/bg")
        self.clothing_images = load_images_from_folder("data/2d")
        self.bg_subspace = self._build_image_subspace(
            self.background_images,
            variance_ratio=0.9,
        )
        raw_clothes_subspace = self._build_image_subspace(
            images=self.clothing_images,
            variance_ratio=0.9,
        )

        self.clothes_subspace = orthogonalize_subspaces(
            primary=self.bg_subspace,
            secondary=raw_clothes_subspace,
        )

    # =================================================
    # Encoding
    # =================================================
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(texts).to(self.device)

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            _, text_features, _ = self.model(
                image=None,
                text=tokens,
            )

        return F.normalize(text_features, dim=-1)

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        image_tensor = torch.stack(
            [self.image_transform(img) for img in images]
        ).to(self.device)

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            image_features, _, _ = self.model(
                image=image_tensor,
                text=None,
            )

        return F.normalize(image_features, dim=-1)

    # =================================================
    # Core scoring
    # =================================================
    @torch.no_grad()
    def score_images(
        self,
        items: List[Tuple[str, Image.Image]],
        topk_pos: int = 3,
    ):
        names,images = zip(*items)
        image_embeddings = self.encode_image(list(images))  # (Ni, D)
        image_embeddings = self.bg_subspace.remove(image_embeddings)
        pos_text_emb = self.prompt_embeddings[0]
        neg_text_emb = self.prompt_embeddings[1]


        img_emb = F.normalize(image_embeddings, dim=-1)

        # -------------------------
        # Similarities
        # -------------------------
        pos_sim = img_emb @ pos_text_emb.T    # (Ni, Np)
        neg_sim = img_emb @ neg_text_emb.T    # (Ni, Nn)

        # -------------------------
        # Aggregate
        # -------------------------
        pos_score = pos_sim

        neg_score = neg_sim

        final_score = pos_sim / (pos_sim + self.lambda_neg * neg_sim + 1e-6)

        # -------------------------
        # Pack results
        # -------------------------
        results = []
        for i in range(len(names)):
            results.append({
                "name_clothes": names[i],
                "score": float(final_score[i]),
                "pos_score": float(pos_score[i]),
                "neg_score": float(neg_score[i]),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # =================================================
    # Utilities
    # =================================================
    def _build_image_subspace(
        self,
        images: List[Image.Image],
        variance_ratio: float = 0.9,
    ) -> ClipSubspace:
        """
        Build PCA/SVD subspace from images.
        """

        with torch.no_grad():
            X = self.encode_image(images)

        X = X.float().cpu()
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean

        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

        var = S**2
        cumvar = var.cumsum(0) / var.sum()
        k = int((cumvar <= variance_ratio).sum().item()) + 1

        basis = Vh[:k].T  # (D, k)

        return ClipSubspace(basis=basis)
