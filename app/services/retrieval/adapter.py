"""
Retrieval adapter — canonical location: app/services/retrieval/adapter.py
(Backward-compat shim remains at app/services/retrieval_adapter.py)

Reads SCENEFIT_USE_MOCK env var and passes mock=True/False into strategies.
"""
import os

from app.services.retrieval.strategies import REGISTRY

USE_MOCK_DATA: bool = os.getenv("SCENEFIT_USE_MOCK", "false").lower() in ("true", "1", "yes")

if USE_MOCK_DATA:
    print("[retrieval_adapter] Mock mode enabled (no remote API calls)")
else:
    print("[retrieval_adapter] Live mode enabled (remote API calls)")


def get_clip_results(image_content: bytes, filename: str, content_type: str, top_k: int):
    return REGISTRY.get("clip").retrieve(image_content, filename, content_type, top_k, mock=USE_MOCK_DATA)


def get_image_edit_results(image_content: bytes, filename: str, content_type: str, top_k: int):
    return REGISTRY.get("image_edit").retrieve(image_content, filename, content_type, top_k, mock=USE_MOCK_DATA)


def get_vlm_results(image_content: bytes, filename: str, content_type: str, top_k: int):
    return REGISTRY.get("vlm").retrieve(image_content, filename, content_type, top_k, mock=USE_MOCK_DATA)


def get_aes_results(image_content: bytes, filename: str, content_type: str, top_k: int):
    return REGISTRY.get("aesthetic").retrieve(image_content, filename, content_type, top_k, mock=USE_MOCK_DATA)


def get_all_results(image_content: bytes, filename: str, content_type: str, top_k: int = 10) -> dict:
    """Run all registered strategies and return results keyed by strategy name."""
    return {
        name: REGISTRY.get(name).retrieve(image_content, filename, content_type, top_k, mock=USE_MOCK_DATA)
        for name in REGISTRY.names()
    }


__all__ = [
    "get_clip_results",
    "get_image_edit_results",
    "get_vlm_results",
    "get_aes_results",
    "get_all_results",
    "USE_MOCK_DATA",
]
