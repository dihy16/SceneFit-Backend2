# app/api/v1/endpoints/neg_pe_clip_ep.py

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form

from app.utils.image_utils import compose_2d_on_background
from app.models.registry import ModelRegistry

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/negative_pe")
def retrieve_best_matched_figures_neg_pe(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Negative-sampling PE-Core image retrieval.

    Uses prompt:
      positive: "clothing that complement the background."
      negative: "clothing that clash with the background."
    """

    # -------------------------------------------------
    # 1. Save uploaded background
    # -------------------------------------------------
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename

    with open(bg_path, "wb") as f:
        f.write(image.file.read())

    # -------------------------------------------------
    # 2. Compose candidate clothing images
    # -------------------------------------------------
    items = compose_2d_on_background(
        bg_path=bg_path,
        fg_dir="data/2d",
        return_format="pil",
    )

    # -------------------------------------------------
    # 3. Score using NegativePEModel
    # -------------------------------------------------
    model = ModelRegistry.get("negative_pe")
    scores = model.score_images(items)

    # -------------------------------------------------
    # 4. Top-K
    # -------------------------------------------------
    return {
        "count": min(top_k, len(scores)),
        "results": scores[:top_k],
    }
