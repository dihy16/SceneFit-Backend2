from typing import List, Dict
from PIL import Image
import torch
from app.models.scripts.qwen3_vl_reranker import Qwen3VLReranker


class Qwen3VLRerankerWrapper:
    """
    Thin wrapper around Qwen3-VL-Reranker for clothes reranking.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-Reranker-2B",
        **kwargs,
    ):
        self.model = Qwen3VLReranker(
            model_name_or_path=model_name_or_path,
            **kwargs,
        )
    def release(self):
        #self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()

    def rerank(
        self,
        query_text: str,
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        query_text: scene caption
        candidates: [
            {
                "name_clothes": str,
                "caption": str,
                "image": PIL.Image | None,
                "similarity": float
            }
        ]
        Returns:
            candidates with added field "rerank_score", sorted desc
        """

        documents = []
        for c in candidates:
            doc = {}
            if c.get("caption"):
                doc["text"] = c["caption"]
            if c.get("image") is not None:
                doc["image"] = c["image"]
            documents.append(doc)

        inputs = {
            "instruction": "Select clothing that best matches the background context.",
            "query": {"text": query_text},
            "documents": documents,
            "fps": 1.0,
        }

        scores = self.model.process(inputs)

        # Attach scores back
        for c, s in zip(candidates, scores):
            c["score"] = float(s)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
