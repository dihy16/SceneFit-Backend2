# app/api/v1/endpoints/diffusion_ep.py

import base64
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form

from app.models.registry import ModelRegistry

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
OUTPUT_DIR = Path("app/outputs/diffusion")
BG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _save_bg_upload(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename
    with open(bg_path, "wb") as f:
        f.write(image.file.read())
    return bg_path


def _encode_image(image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@router.post("/diffusion")
def retrieve_best_matched_figures_diffusion(
    image: UploadFile = File(...),
    top_k: int = Form(1),
    include_image: bool = Form(False),
    save_best: bool = Form(False),
):
    """
    Compose all 2D figures on the background, score via diffusion noise
    prediction error, and return the best candidates.
    """
    bg_path = _save_bg_upload(image)

    model = ModelRegistry.get("diffusion")
    items = model.compose_candidates(
        background_path=bg_path,
        figures_dir="data/2d",
    )
    scores = model.score_images(items)

    if not scores:
        return {"count": 0, "results": []}

    top_k = max(1, min(top_k, len(scores)))
    top_results = scores[:top_k]

    response = {
        "count": len(top_results),
        "results": top_results,
    }

    if include_image or save_best:
        best = top_results[0]
        best_image = items[best["index"]][1]

        if include_image:
            response["best_image_base64"] = _encode_image(best_image)

        if save_best:
            out_name = f"{Path(best['name']).stem}_{uuid.uuid4().hex}.png"
            out_path = OUTPUT_DIR / out_name
            best_image.convert("RGB").save(out_path)
            response["best_image_path"] = str(out_path)

    return response
