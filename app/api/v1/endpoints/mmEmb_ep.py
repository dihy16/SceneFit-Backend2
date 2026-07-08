# app/api/v1/endpoints/clip_ep.py

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form

from app.utils.image_utils import compose_2d_on_background
from app.models.registry import ModelRegistry

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/mmEmb")
def retrieve_best_matched_figures(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    scale: float = Form(1.0),
):
    """
    Given a background/figure image, compose it with all 2D clothes,
    score them using MmEmbModel, and return the best matches.
    """

    # ------------------------------------------------------------------
    # 1. Save uploaded image temporarily
    # ------------------------------------------------------------------
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename

    with open(bg_path, "wb") as f:
        f.write(image.file.read())

    # ------------------------------------------------------------------
    # 2. Compose background with all foreground candidates
    # ------------------------------------------------------------------
    images = compose_2d_on_background(
        bg_path=bg_path,
        fg_dir="app/clothes/2d",
        scale=scale,
        return_format="pil",
    )

    # ------------------------------------------------------------------
    # 3. Score images with MM Embedding model
    # ------------------------------------------------------------------
    model = ModelRegistry.get("mmEmb")
    scores = model.score_images(images)

    # ------------------------------------------------------------------
    # 4. Select top-K results
    # ------------------------------------------------------------------
    top_results = scores[:top_k]

    return {
        "count": len(top_results),
        "results": top_results,
    }