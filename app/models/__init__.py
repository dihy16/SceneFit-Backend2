"""
app/models — domain-grouped ML model packages.

Top-level re-exports for convenience:
    from app.models import ModelRegistry
    from app.models.embedding import MmEmbModel, PEClipModel
    from app.models.vlm import VLModel
    etc.
"""
from app.models.registry import ModelRegistry

__all__ = ["ModelRegistry"]
