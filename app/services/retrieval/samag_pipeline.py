"""
SaMaG-R (VLM-Faiss) Retrieval Pipeline.
Canonical location: app/services/retrieval/samag_pipeline.py

Implements a Builder pattern to orchestrate the VLM-Faiss pipeline.
Allows easy toggling of modules (ablation studies) without editing the core math.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from pathlib import Path
from PIL import Image

from app.models.registry import ModelRegistry


class SaMaGPipeline:
    """
    Builder pattern for the SaMaG-R retrieval pipeline.
    
    Usage:
        results = (
            SaMaGPipeline(use_visual=True, use_semantic=True, ...)
            .extract_visual_features(bg_path)
            .extract_semantic_features(bg_path)
            .formulate_query(w_semantic=0.4, w_scene=0.4, w_img=0.2)
            .execute_faiss_search(top_k=10)
            .rerank_candidates()
            .get_results()
        )
    """

    def __init__(
        self,
        use_visual: bool = True,
        use_semantic: bool = True,
        use_orthogonal_rejection: bool = True,
        use_reranker: bool = True,
        n_good: int = 7,
        domain_suffix: str = ", 3D rendered character model, game asset style"
    ):
        self.config = {
            "use_visual": use_visual,
            "use_semantic": use_semantic,
            "use_orthogonal_rejection": use_orthogonal_rejection,
            "use_reranker": use_reranker,
            "n_good": n_good,
            "domain_suffix": domain_suffix,
        }
        self.context: Dict[str, Any] = {
            "bg_emb": None,
            "scene_emb": None,
            "good_emb": None,
            "bad_emb": None,
            "scene_caption": None,
            "query_emb": None,
            "candidates": [],
        }

    def extract_visual_features(self, bg_image_path: str | Path) -> SaMaGPipeline:
        """Extract visual embeddings using the PE CLIP Image Encoder (Path A)."""
        if not self.config["use_visual"]:
            return self

        matcher = ModelRegistry.get("pe_clip_matcher")
        bg_img = Image.open(bg_image_path).convert("RGB")
        self.context["bg_emb"] = matcher.encode_image([bg_img])  # (1, D)
        return self

    def extract_semantic_features(self, bg_image_path: str | Path) -> SaMaGPipeline:
        """Generate and encode descriptions using the VLM and PE CLIP Text Encoder (Path B)."""
        if not self.config["use_semantic"]:
            return self

        # 1. Generate text signals from VLM
        vlm = ModelRegistry.get("vlm")
        signals = vlm.extract_query_signals(str(bg_image_path))
        ModelRegistry.release("vlm")  # Free memory early
        
        self.context["scene_caption"] = signals["scene_caption"]
        
        # 2. Append domain suffix to clothes descriptions
        suffix = self.config["domain_suffix"]
        color_outfits = [desc + suffix for desc in signals["color_outfits"]]

        # 3. Encode text signals to vectors
        matcher = ModelRegistry.get("pe_clip_matcher")
        semantic_emb = matcher.encode_text(color_outfits)     # (M+N, D)
        scene_emb = matcher.encode_text([signals["scene_caption"]]) # (1, D)

        n_good = self.config["n_good"]
        self.context["scene_emb"] = scene_emb
        self.context["good_emb"] = semantic_emb[:n_good]  # (M, D)
        self.context["bad_emb"] = semantic_emb[n_good:]   # (N, D)
        
        return self

    def formulate_query(
        self,
        w_semantic: float = 0.4,
        w_scene: float = 0.4,
        w_img: float = 0.2
    ) -> SaMaGPipeline:
        """Fuse visual and semantic cues, applying Orthogonal Rejection if enabled."""
        
        # Determine embedding dimension dynamically or default to 512
        emb_dim = 512
        if self.context["bg_emb"] is not None:
            emb_dim = self.context["bg_emb"].shape[-1]
        elif self.context["scene_emb"] is not None:
            emb_dim = self.context["scene_emb"].shape[-1]
            
        # Base query formulation
        query_emb = torch.zeros((self.config["n_good"], emb_dim))
        
        # Add Visual Cue
        if self.config["use_visual"] and self.context["bg_emb"] is not None:
            query_emb = query_emb + (w_img * self.context["bg_emb"])

        # Add Semantic Cues
        if self.config["use_semantic"] and self.context["good_emb"] is not None:
            
            # Apply Orthogonal Rejection (remove 'bad' concept from 'good' clothes)
            if self.config["use_orthogonal_rejection"] and self.context["bad_emb"] is not None:
                bad_vec = self.context["bad_emb"].mean(dim=0, keepdim=True)
                bad_unit = F.normalize(bad_vec, p=2, dim=-1)
                
                # Projection = (good @ bad.T) * bad
                projection = (self.context["good_emb"] @ bad_unit.T) * bad_unit
                clean_good = self.context["good_emb"] - projection
                
                query_emb = query_emb + (w_semantic * clean_good)
            else:
                # Ablated: Just use the raw 'good' embeddings
                query_emb = query_emb + (w_semantic * self.context["good_emb"])
                
            # Add scene background text context
            if self.context["scene_emb"] is not None:
                query_emb = query_emb + (w_scene * self.context["scene_emb"])

        # Normalize the fused queries
        self.context["query_emb"] = F.normalize(query_emb, dim=-1)
        return self

    def execute_faiss_search(self, top_k: int = 10) -> SaMaGPipeline:
        """Perform Maximum Similarity Aggregation across M queries in FAISS."""
        if self.context["query_emb"] is None:
            raise ValueError("Query embedding is missing. Did you forget to call formulate_query()?")
            
        matcher = ModelRegistry.get("pe_clip_matcher")
        self.context["candidates"] = matcher.match_clothes(
            query_emb=self.context["query_emb"],
            top_k=top_k,
        )
        ModelRegistry.release("pe_clip_matcher")
        return self

    def rerank_candidates(self) -> SaMaGPipeline:
        """Fine-grained reranking using Qwen3-VL and scene descriptions."""
        if not self.config["use_reranker"] or not self.context["candidates"]:
            return self

        # 1. Attach images for the reranker
        for c in self.context["candidates"]:
            img_path = Path("data/2d") / f"{c['name_clothes']}"
            c["image"] = Image.open(img_path).convert("RGB")
            
        # 2. Execute Reranker
        reranker = ModelRegistry.get("qwen_reranker")
        
        # Fallback if semantic path was ablated and we have no scene caption
        caption = self.context["scene_caption"] or ""
        
        reranked = reranker.rerank(
            query_text=caption,
            candidates=self.context["candidates"],
        )
        ModelRegistry.release("qwen_reranker")
        
        # 3. Clean up non-serializable fields
        for c in reranked:
            c.pop("image", None)
            
        self.context["candidates"] = reranked
        return self

    def get_results(self) -> List[Dict[str, Any]]:
        """Return the final candidate list."""
        return self.context["candidates"]
