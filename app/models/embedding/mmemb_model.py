# app/models/mmemb_model.py 

import torch
import torch.nn.functional as F
from transformers import AutoModel
from app.utils.device import resolve_device, resolve_dtype
from app.utils.util import load_prompts


class MmEmbModel:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_name = model_name

        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )

        self.prompts = load_prompts("outfit_match")
        self.prompt_embeddings = self.encode_text(self.prompts)

        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # Text encoding
    # -------------------------
    def encode_text(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts=texts,
                task="retrieval",
                prompt_name="query",
            )
        return self._normalize(embeddings)

    # -------------------------
    # Image encoding
    # -------------------------
    def encode_image(self, images):
        with torch.no_grad():
            embeddings = self.model.encode_image(
                images=images,
                task="retrieval",
            )
        return self._normalize(embeddings)

    # -------------------------
    # Cosine similarity
    # -------------------------
    def similarity(self, text_embeddings, image_embeddings):
        return text_embeddings @ image_embeddings.T

    @staticmethod
    def _normalize(x):
        return F.normalize(x, p=2, dim=-1)
    
    def score_image(self, image):
        """
        Returns similarity scores for:
        - positive prompt
        - negative prompt
        """
        image_embedding = self.encode_image(image)

        pos_emb = self.prompt_embeddings[0]
        neg_emb = self.prompt_embeddings[1]

        pos_score = (image_embedding @ pos_emb.T).item()
        neg_score = (image_embedding @ neg_emb.T).item()

        return {
            "positive_score": pos_score,
            "negative_score": neg_score,
            "confidence": pos_score - neg_score,
        }
    
    def score_images(self, images):
        """
        Calculate confidence scores for a list of images and return
        results sorted by confidence (descending).

        Args:
            images: list of images (PIL images or model-compatible inputs)

        Returns:
            List[dict]: sorted results with scores
        """
        # Encode all images at once (batch)
        image_embeddings = self.encode_image(images)

        results = []

        pos_emb = self.prompt_embeddings[0]
        neg_emb = self.prompt_embeddings[1]

        for idx, img_emb in enumerate(image_embeddings):
            pos_score = (img_emb @ pos_emb.T).item()
            neg_score = (img_emb @ neg_emb.T).item()
            confidence = pos_score - neg_score

            results.append({
                "index": idx,
                "positive_score": pos_score,
                "negative_score": neg_score,
                "confidence": confidence,
            })

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return results
