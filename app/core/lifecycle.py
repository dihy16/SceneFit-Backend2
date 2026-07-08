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

    print("[START] Models loaded")
    print("[START] Backend started")

    yield  # Application runs here

    # ---------- Shutdown ----------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] Backend shutdown")
