# app/core/lifecycle.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch

from app.core.vector_db import VectorDatabase
from app.models.diffusion.image_edit_model import ImageEditFlux
from app.models.registry import ModelRegistry
from app.models.embedding.mmemb_model import MmEmbModel
from app.models.embedding.pe_clip_model import PEClipModel
from app.models.vlm.vl_model import VLModel
from app.models.embedding.pe_clip_matcher import PEClipMatcher
from app.models.diffusion.diffusion_model import DiffusionModel
# from app.models.vqvae.vqvae_model import VQVAEModel
from app.core.vector_db import VectorDatabase

# Expose vector DB for downstream endpoints
vector_db: VectorDatabase | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # ---------- Startup ----------
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[START] Backend started")

    print("[START] Loading models...")

    # Try to load model weights with retry logic
    model_registry = ModelRegistry()
    loaded_models = 0

    for _ in range(3):
        try:
            # Try to load each required model
            if vector_db is None:
                vector_db = VectorDatabase()

            model_registry.fl_model = ImageEditFlux()
            model_registry.mmemb_model = MmEmbModel()
            model_registry.pe_clip_model = PEClipModel()
            model_registry.vl_model = VLModel()
            model_registry.pe_matcher = PEClipMatcher()
            model_registry.diffusion_model = DiffusionModel()

            # Validate that models loaded successfully
            if all([
                model_registry.fl_model,
                model_registry.mmemb_model,
                model_registry.pe_clip_model,
                model_registry.vl_model,
                model_registry.pe_matcher,
                model_registry.diffusion_model
            ]):
                loaded_models = 1
                break
        except Exception as e:
            print(f"[START] Model load failed: {e}")
            import time
            time.sleep(10)

    # Check if models loaded successfully
    if loaded_models == 0:
        print("[ERROR] Failed to load all models after retries")
        
    print("[START] Models loaded")
    print("[START] Backend started")

    yield  # Application runs here

    # ---------- Shutdown ----------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] Backend shutdown")
