"""
app/services/retrieval — retrieval business logic subpackage.

Public API (import from here in new code):
    from app.services.retrieval import REGISTRY, BaseRetrievalStrategy
    from app.services.retrieval import shuffle_retrieval_results
    from app.services.retrieval.adapter import get_clip_results, USE_MOCK_DATA
    from app.services.retrieval.compat import get_aes_results   # backward-compat wrappers
"""
from app.services.retrieval.strategies import (
    REGISTRY,
    BaseRetrievalStrategy,
    CLIPRetrieval,
    ImageEditRetrieval,
    VLMRetrieval,
    AestheticRetrieval,
    RetrievalRegistry,
)
from app.services.retrieval.post_processing import (
    shuffle_retrieval_results,
    rerank_with_soft_penalty,
)

__all__ = [
    "REGISTRY",
    "BaseRetrievalStrategy",
    "CLIPRetrieval",
    "ImageEditRetrieval",
    "VLMRetrieval",
    "AestheticRetrieval",
    "RetrievalRegistry",
    "shuffle_retrieval_results",
    "rerank_with_soft_penalty",
]
