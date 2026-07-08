"""
Retrieval Strategy Pattern
--------------------------
Canonical location: app/services/retrieval/strategies.py
(Backward-compat shim remains at app/services/retrieval_strategies.py)

Each retrieval method is encapsulated as a concrete strategy that extends
`BaseRetrievalStrategy`. Common concerns (HTTP request/retry, mock fallback,
timing, result normalization, and post-processing shuffle) are handled once
inside the base class `retrieve()` template method.

Adding a new method:
  1. Subclass `BaseRetrievalStrategy` and override `config_key` and `display_name`.
  2. Instantiate and register it: `REGISTRY.register(MyStrategy())`.
  3. Done — it is immediately available in the dynamic endpoint and all-methods aggregator.
"""

from __future__ import annotations

import io
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import HTTPException

from app.services.retrieval.post_processing import shuffle_retrieval_results
from app.utils.util import convert_filename_to_url

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "retrieval_methods.yaml"

with open(_CONFIG_PATH, "r") as _f:
    _raw_config = yaml.safe_load(_f)

_RETRIEVAL_CONFIG: dict[str, Any] = _raw_config["retrieval_methods"]
_TIMEOUT: int = _raw_config.get("timeout", 60)
_RETRY: dict = _raw_config.get("retry", {"max_attempts": 3, "delay_seconds": 1})

# Oversampling factors (can be tuned via env vars)
_OVERSAMPLE_FACTOR = int(os.getenv("SCENEFIT_OVERSAMPLE_FACTOR", "20"))
_MAX_REQUEST_TOP_K = int(os.getenv("SCENEFIT_MAX_REQUEST_TOP_K", "100"))

# Cache for outfit names so we only scan the directory once per process
_OUTFIT_NAMES_CACHE: list[str] | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_png(name: str) -> str:
    return name[:-4] if isinstance(name, str) and name.lower().endswith(".png") else name


def _ensure_png(name: str) -> str:
    if not isinstance(name, str):
        return name
    return name if name.lower().endswith(".png") else f"{name}.png"


def _get_outfit_names() -> list[str]:
    """Lazily load and cache outfit names from app/data/2d/."""
    global _OUTFIT_NAMES_CACHE
    if _OUTFIT_NAMES_CACHE is not None:
        return _OUTFIT_NAMES_CACHE

    data_dir = Path(__file__).parent.parent.parent / "data" / "2d"
    names: list[str] = []
    if data_dir.exists():
        names = [p.stem for p in data_dir.iterdir() if p.is_file() and p.suffix == ".png"]

    _OUTFIT_NAMES_CACHE = names
    print(f"[RetrievalStrategy] Cached {len(names)} outfit names from {data_dir}")
    return names


def _normalize_item(item: dict) -> dict | None:
    """Coerce a raw result dict into the canonical {name, score, image_url} shape."""
    if not isinstance(item, dict):
        return None
    raw_name = item.get("name") or item.get("image_name") or item.get("file_name")
    if not raw_name:
        return None

    with_ext = _ensure_png(raw_name)
    score_val = item.get("score")
    try:
        score = float(score_val) if score_val is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0

    existing_url = item.get("image_url")
    image_url = (
        existing_url
        if isinstance(existing_url, str) and existing_url.lower().endswith(".png")
        else convert_filename_to_url(with_ext)
    )

    return {"name": _strip_png(raw_name), "score": score, "image_url": image_url}


def _normalize_and_shuffle(raw: Any, top_k: int) -> list[dict]:
    """Normalize raw upstream/mock payload and apply post-processing shuffle."""
    if isinstance(raw, dict) and raw.get("error"):
        return raw  # propagate error objects as-is

    results_list = (
        raw.get("results")
        if isinstance(raw, dict) and "results" in raw
        else raw
    )
    if not isinstance(results_list, list):
        return raw

    normalized = [n for item in results_list if (n := _normalize_item(item)) is not None]
    return shuffle_retrieval_results(normalized, top_k)


def _request_top_k(top_k: int) -> int:
    """Inflate top_k for upstream so post-processing has more items to shuffle."""
    if top_k <= 0:
        return 0
    return min(_MAX_REQUEST_TOP_K, max(top_k, top_k * max(1, _OVERSAMPLE_FACTOR)))


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class BaseRetrievalStrategy(ABC):
    """
    Template method base class for all retrieval strategies.

    Concrete subclasses only need to set `config_key` (and optionally
    `display_name`); the `retrieve()` method handles everything else.
    """

    config_key: str = ""
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.config_key.upper()

    def retrieve(
        self,
        image_content: bytes,
        filename: str,
        content_type: str,
        top_k: int,
        mock: bool = True,
    ) -> list[dict] | dict:
        """
        Run retrieval. Uses mock data when `mock=True` or when the real
        upstream request fails (automatic fallback).
        """
        tag = self.display_name or self.config_key.upper()
        request_top_k = _request_top_k(top_k)
        start = time.perf_counter()
        fell_back = False

        if mock:
            results = self._generate_mock(request_top_k)
        else:
            try:
                results = self._call_upstream(image_content, filename, content_type, request_top_k)
            except Exception as exc:
                print(f"[{tag}] Falling back to mock: {str(exc)[:120]}")
                results = self._generate_mock(request_top_k)
                fell_back = True

        elapsed = time.perf_counter() - start
        mode = "mock" if (mock or fell_back) else "real"
        print(f"[{tag}] Completed in {elapsed:.3f}s (mode={mode}, request_top_k={request_top_k}, return_top_k={top_k})")

        return _normalize_and_shuffle(results, top_k)

    def _generate_mock(self, top_k: int) -> list[dict]:
        """Return randomized mock results drawn from real outfit names."""
        tag = self.display_name or self.config_key.upper()
        names = _get_outfit_names()
        if not names:
            print(f"[{tag}] WARNING - no outfit names found, using placeholders")
            names = [f"outfit_{i}" for i in range(100)]

        selected = random.sample(names, min(top_k, len(names)))
        results = []
        for i, name in enumerate(selected):
            score = 0.95 - (i * 0.45 / max(1, top_k - 1))
            with_ext = _ensure_png(name)
            results.append({
                "name": _strip_png(name),
                "score": round(score, 4),
                "image_url": convert_filename_to_url(with_ext),
            })

        print(f"[{tag}] Generated {len(results)} mock results")
        return results

    def _call_upstream(
        self,
        image_content: bytes,
        filename: str,
        content_type: str,
        top_k: int,
    ) -> Any:
        """
        Make an HTTP POST to the configured upstream endpoint with retry logic.
        Raises HTTPException on all-attempts failure.
        """
        tag = self.display_name or self.config_key.upper()

        if self.config_key not in _RETRIEVAL_CONFIG:
            raise HTTPException(
                status_code=500,
                detail=f"Strategy '{self.config_key}' not found in retrieval_methods.yaml",
            )

        cfg = _RETRIEVAL_CONFIG[self.config_key]
        base_url = cfg['url'].rstrip('/')
        endpoint_path = cfg['endpoint'].lstrip('/')
        url = f"{base_url}/{endpoint_path}"
        print(f"[{tag}] -> {url}  (top_k={top_k})")

        for attempt in range(_RETRY["max_attempts"]):
            try:
                print(f"[{tag}] Attempt {attempt + 1}/{_RETRY['max_attempts']}")
                img_file = io.BytesIO(image_content)
                with httpx.Client(timeout=_TIMEOUT) as client:
                    response = client.post(
                        url,
                        files={"image": (filename, img_file, content_type)},
                        data={"top_k": top_k},
                    )
                    response.raise_for_status()
                    data = response.json()
                    count = len(data) if isinstance(data, list) else len(data.get("results", []))
                    print(f"[{tag}] OK {count} results (HTTP {response.status_code})")
                    return data

            except httpx.HTTPStatusError as e:
                print(f"[{tag}] HTTP {e.response.status_code}: {e.response.text[:100]}")
                if attempt == _RETRY["max_attempts"] - 1:
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Error from {tag} service: {e.response.text}",
                    )

            except httpx.RequestError as e:
                print(f"[{tag}] Connection error: {str(e)[:100]}")
                if attempt == _RETRY["max_attempts"] - 1:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Unable to reach {tag} service at {url}: {e}",
                    )

            if attempt < _RETRY["max_attempts"] - 1:
                print(f"[{tag}] Retrying in {_RETRY['delay_seconds']}s...")
                time.sleep(_RETRY["delay_seconds"])

        raise HTTPException(status_code=503, detail=f"{tag} unreachable after retries")

    @property
    def name(self) -> str:
        return self.config_key

    @property
    def description(self) -> str:
        return _RETRIEVAL_CONFIG.get(self.config_key, {}).get("description", "")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} config_key={self.config_key!r}>"


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class CLIPRetrieval(BaseRetrievalStrategy):
    """Naive CLIP embeddings + FAISS vector database retrieval."""
    config_key = "clip"
    display_name = "CLIP"


class ImageEditRetrieval(BaseRetrievalStrategy):
    """GPT/Flux generative image edit -> cropped outfit -> vector DB search."""
    config_key = "image_edit"
    display_name = "IMAGE_EDIT"


class VLMRetrieval(BaseRetrievalStrategy):
    """VLM scene analysis -> fused CLIP+text query -> FAISS coarse recall -> Qwen3-VL rerank."""
    config_key = "vlm"
    display_name = "VLM"


class AestheticRetrieval(BaseRetrievalStrategy):
    """Aesthetic-predictor scoring of composed outfit+background candidates."""
    config_key = "aesthetic"
    display_name = "AESTHETIC"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class RetrievalRegistry:
    """
    Central registry mapping strategy names to `BaseRetrievalStrategy` instances.

    Usage:
        strategy = REGISTRY.get("clip")
        results  = strategy.retrieve(image_content, filename, content_type, top_k)
    """

    def __init__(self) -> None:
        self._strategies: dict[str, BaseRetrievalStrategy] = {}

    def register(self, strategy: BaseRetrievalStrategy) -> None:
        if not strategy.config_key:
            raise ValueError(f"Strategy {strategy!r} has no config_key set")
        self._strategies[strategy.config_key] = strategy
        print(f"[RetrievalRegistry] Registered strategy: '{strategy.config_key}'")

    def get(self, name: str) -> BaseRetrievalStrategy:
        strategy = self._strategies.get(name)
        if strategy is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown retrieval method '{name}'. Available: {sorted(self._strategies)}",
            )
        return strategy

    def all(self) -> dict[str, BaseRetrievalStrategy]:
        return dict(self._strategies)

    def names(self) -> list[str]:
        return sorted(self._strategies)


# ---------------------------------------------------------------------------
# Singleton — pre-loaded with the four main strategies
# ---------------------------------------------------------------------------

REGISTRY = RetrievalRegistry()
REGISTRY.register(CLIPRetrieval())
REGISTRY.register(ImageEditRetrieval())
REGISTRY.register(VLMRetrieval())
REGISTRY.register(AestheticRetrieval())
