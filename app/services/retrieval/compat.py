"""
Backward-compatible retrieval wrappers — canonical location: app/services/retrieval/compat.py

Consolidates the old all_methods.py and all_methods_safe.py shims into one place.
All functions here delegate to the RetrievalRegistry.

Usage (new code — prefer the registry directly):
    from app.services.retrieval import REGISTRY
    results = REGISTRY.get("clip").retrieve(...)

Usage (legacy code — still supported):
    from app.services.retrieval.compat import get_clip_results, generate_mock_results
    from app.services.retrieval.compat import get_clip_results_safe
"""

from app.services.retrieval.strategies import (
    REGISTRY,
    _normalize_and_shuffle,
    _normalize_item,
    _get_outfit_names,
    _request_top_k,
    _OVERSAMPLE_FACTOR,
    _MAX_REQUEST_TOP_K,
)


# ---------------------------------------------------------------------------
# Mock generation helper (kept for test_mock_data.py compatibility)
# ---------------------------------------------------------------------------

def generate_mock_results(top_k: int, method_name: str = "mock") -> list:
    """Generate randomized mock results matching outfit names in app/data/2d/."""
    return REGISTRY.get("clip")._generate_mock(top_k)


# ---------------------------------------------------------------------------
# Standard wrappers (previously in all_methods.py)
# ---------------------------------------------------------------------------

def get_clip_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return REGISTRY.get("clip").retrieve(image_content, filename, content_type, top_k, mock)


def get_image_edit_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return REGISTRY.get("image_edit").retrieve(image_content, filename, content_type, top_k, mock)


def get_vlm_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return REGISTRY.get("vlm").retrieve(image_content, filename, content_type, top_k, mock)


def get_aes_results(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return REGISTRY.get("aesthetic").retrieve(image_content, filename, content_type, top_k, mock)


# ---------------------------------------------------------------------------
# Safe wrappers (previously in all_methods_safe.py) — return error dicts, never raise
# ---------------------------------------------------------------------------

def _safe(name: str, image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    try:
        return REGISTRY.get(name).retrieve(image_content, filename, content_type, top_k, mock)
    except Exception as e:
        msg = str(e)
        print(f"[{name.upper()}] Service error: {msg[:100]}")
        return {"error": True, "message": msg}


def get_clip_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return _safe("clip", image_content, filename, content_type, top_k, mock)


def get_image_edit_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return _safe("image_edit", image_content, filename, content_type, top_k, mock)


def get_vlm_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return _safe("vlm", image_content, filename, content_type, top_k, mock)


def get_aes_results_safe(image_content: bytes, filename: str, content_type: str, top_k: int, mock: bool = True):
    return _safe("aesthetic", image_content, filename, content_type, top_k, mock)
