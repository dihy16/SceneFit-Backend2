from fastapi import APIRouter, UploadFile, File, Form
import uuid
from pathlib import Path
from app.utils.image_utils import compose_2d_on_background
from app.models.registry import ModelRegistry
from app.utils.candidates import parse_candidate_names, resolve_candidate_filenames
import time

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)

def _save_bg_upload(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{time.time_ns()}{suffix}"
    bg_path = BG_DIR / bg_filename
    
    if not bg_path.exists():
        with open(bg_path, "wb") as f:
            f.write(image.file.read())
    return bg_path

def score_outfits(
    rg_head,
    bg_path: Path,
    top_k: int = 5,
    batch_size: int = 300,
    candidate_names: list[str] | None = None,
):
    all_scores = []
    offset = 0
    candidate_files = (
        resolve_candidate_filenames(candidate_names, Path("data/2d"))
        if candidate_names is not None
        else None
    )
    
    while True:
        print(f"[AESTHETIC] Preparing batch {offset//batch_size + 1} ...")
        # Load and compose batch
        items = compose_2d_on_background(
            bg_path=bg_path,
            fg_dir="data/2d",
            fg_files=candidate_files,
            return_format="pil",
            offset=offset,
            limit=batch_size,
        )
        
        if not items:
            break  # No more items
        
        print(f"[AESTHETIC] Batch {offset//batch_size + 1}: Scoring {len(items)} outfits ...")
        
        # Score this batch
        batch_scores = rg_head.score_images(items)
        all_scores.extend(batch_scores)
        
        # Clear batch from memory
        del items
        del batch_scores
        
        offset += batch_size
    
    print(f"[AESTHETIC] Total scored: {len(all_scores)} outfits")
    
    # Sort all scores and return top_k
    all_scores.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "count": min(top_k, len(all_scores)),
        "results": all_scores[:top_k],
    }

@router.post("/aesthetic")
def retrieve_best_fit_aesthetic(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    batch_size: int = Form(100),
    candidate_names: str | None = Form(None),
):
    bg_path = _save_bg_upload(image)
    model = ModelRegistry.get("aesthetic")
    candidates = parse_candidate_names(candidate_names)
    results = score_outfits(model, bg_path, top_k, batch_size, candidates)
    return results["results"]
