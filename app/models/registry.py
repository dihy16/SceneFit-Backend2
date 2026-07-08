"""
Central model registry — canonical location: app/models/registry.py
(Previously app/services/model_registry.py — a backward-compat shim remains there.)

Lazy-loads heavy ML models by name and caches them for the process lifetime.
"""

from typing import Any, Dict

from app.models.embedding.mmemb_model import MmEmbModel
from app.models.embedding.pe_clip_model import PEClipModel
from app.models.vlm.qwen_reranker import Qwen3VLRerankerWrapper
from app.models.vlm.vl_model import VLModel
from app.models.embedding.pe_clip_matcher import PEClipMatcher
from app.models.aesthetic.aesthetic_predictor import AestheticPredictor
from app.models.diffusion.diffusion_model import DiffusionModel
from app.models.embedding.negative_generator import NegativePEModel
from app.models.asr.asr_model import ASRModel
from app.models.text.text_macher_model import TextMatcherModel
from app.models.diffusion.image_edit_model import ImageEditFlux


class ModelRegistry:
    """
    Singleton-style class registry for heavy ML models.

    Usage:
        vlm = ModelRegistry.get("vlm")
        ModelRegistry.release("vlm")    # unload and free VRAM
    """

    _models: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, model: Any) -> None:
        """Manually register a pre-loaded model instance."""
        cls._models[name] = model

    @classmethod
    def get(cls, name: str) -> Any:
        """Return a model, loading it lazily on first access."""
        if name not in cls._models:
            cls._models[name] = cls._load(name)
        return cls._models[name]

    @classmethod
    def release(cls, name: str) -> None:
        """Unload a model and free its resources."""
        model = cls._models.get(name)
        if model is None:
            return
        if hasattr(model, "release"):
            model.release()
        del cls._models[name]

    @staticmethod
    def _load(name: str) -> Any:
        """Instantiate and optionally call `.load()` on the model."""
        loaders = {
            "jina-v4":        lambda: MmEmbModel(),
            "pe":             lambda: PEClipModel(device="cpu"),
            "vlm":            lambda: VLModel(),
            "pe_clip_matcher":lambda: PEClipMatcher(),
            "aesthetic":      lambda: AestheticPredictor(),
            "diffusion":      lambda: DiffusionModel(
                                  model_id="stabilityai/stable-diffusion-3.5-medium",
                                  pipeline_type="sd3",
                                  text_encoder_only=True,
                              ),
            "qwen_reranker":  lambda: Qwen3VLRerankerWrapper(),
            "asr":            lambda: ASRModel(),
            "text_matcher":   lambda: TextMatcherModel(),
            "image_edit_flux":lambda: ImageEditFlux(),
        }

        factory = loaders.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown model: '{name}'. Available: {sorted(loaders)}"
            )

        model = factory()
        load_fn = getattr(model, "load", None)
        if callable(load_fn):
            load_fn()
        return model
