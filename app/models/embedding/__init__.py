"""
app/models/embedding — CLIP, PE-CLIP, and multimodal embedding models.

Exports:
    MmEmbModel          — Jina-v4 multimodal embedding
    PEClipModel         — Perception Encoder CLIP backbone
    PEClipMatcher       — PE-CLIP cosine similarity matcher
    ClipSubspace        — Orthonormal CLIP subspace projection
    orthogonalize_subspaces
"""
from app.models.embedding.mmemb_model import MmEmbModel
from app.models.embedding.pe_clip_model import PEClipModel
from app.models.embedding.pe_clip_matcher import PEClipMatcher
from app.models.embedding.clip_subspace import ClipSubspace, orthogonalize_subspaces

__all__ = [
    "MmEmbModel",
    "PEClipModel",
    "PEClipMatcher",
    "ClipSubspace",
    "orthogonalize_subspaces",
]
