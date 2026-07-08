# app/api/v1/endpoints/pe_clip_ep.py

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, Request, UploadFile, File, Form

from app.utils.image_utils import compose_2d_on_background
from app.models.registry import ModelRegistry

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)

def get_vector_db(request: Request):
    return request.app.state.vector_db

@router.post("/pe")
def retrieve_best_matched_figures_pe(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    PE-Core version of image retrieval.
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
    # 2. Compose candidate images
    # -------------------------------------------------
    items = compose_2d_on_background(
        bg_path=bg_path,
        fg_dir="data/2d",
        return_format="pil",
    )

    # -------------------------------------------------
    # 3. Score using PE-Core model
    # -------------------------------------------------
    model = ModelRegistry.get("pe")
    scores = model.score_images(items)

    # -------------------------------------------------
    # 4. Top-K
    # -------------------------------------------------
    return {
        "count": min(top_k, len(scores)),
        "results": scores[:top_k],
    }

@router.post("/clip")
def retrieve_scene_outfit(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    vector_db = Depends(get_vector_db),
):
    """
    Retrieval using naive CLIP embeddings and vector database.
    """
    # -------------------------------------------------
    # 1. Save uploaded background
    # -------------------------------------------------
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename
    
    if not bg_path.exists():
        with open(bg_path, "wb") as f:
            f.write(image.file.read())

    # -------------------------------------------------
    # 2. Top-K
    # -------------------------------------------------
    scores = vector_db.search_by_image(bg_path, top_k=top_k)

    # -------------------------------------------------
    # 3. Format results
    # -------------------------------------------------
    results = []
    for item in scores:
        name = Path(item["metadata"]).name if item.get("metadata") else f"image_{item['id']}"
        results.append({
            "name": name,
            "score": item["score"]
        })
    
    return results
