# app/api/v1/endpoints/vlm_ep.py

import json
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Body, Form
from PIL import Image
import time
import torch
import torch.nn.functional as F
import gc

from app.models.registry import ModelRegistry
from app.utils.util import load_str_images_from_folder
from app.services.clothes_captions import generate_clothes_captions_json
from app.api.v1.endpoints.aesthetic_ep import score_outfits
router = APIRouter()

BG_DIR = Path("app/uploads/bg")
CLOTHES_DIR = Path("data/2d")
CLOTHES_CAPTION = Path("data/clothes_captions.json")
RESULTS_DIR = Path("app/uploads/results")
BG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _save_bg_upload(image: UploadFile) -> Path:
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{time.time_ns()}{suffix}"
    bg_path = BG_DIR / bg_filename
    with open(bg_path, "wb") as f:
        f.write(image.file.read())
    return bg_path

def _get_clothes_captions() -> dict:
    if CLOTHES_CAPTION.exists():
        with open(CLOTHES_CAPTION, "r", encoding="utf-8") as f:
            return json.load(f)
    return generate_clothes_captions_json()


def _write_result(payload: dict) -> Path:
    filename = f"{time.time_ns()}_{uuid.uuid4().hex}.json"
    output_path = RESULTS_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path

def _select_clothes_via_tournament(vlm, background_caption: str, clothes_captions: dict) -> str | None:
    if not clothes_captions:
        return None
    candidates = sorted(clothes_captions.items())
    while len(candidates) > 1:
        next_round: list[tuple[str, str]] = []
        for i in range(0, len(candidates), 10):
            batch = candidates[i:i + 10]
            if len(batch) == 1:
                next_round.append(batch[0])
                continue
            best_name = vlm.choose_best_clothes(background_caption, batch)
            best_caption = dict(batch).get(best_name, batch[0][1])
            next_round.append((best_name, best_caption))
        candidates = next_round
    return candidates[0][0]

def _method_image_match(vlm, bg_path: Path):
    descriptions = _generate_clothes_descriptions(vlm, bg_path)
    results = _rank_clothes_by_image(descriptions, top_k=1)
    return descriptions, (results[0] if results else None)

def _method_caption_match(vlm, descriptions):
    results = _rank_clothes_by_caption(descriptions, top_k=1)
    return descriptions, (results[0] if results else None)

def _generate_clothes_descriptions(vlm, bg_path: Path) -> list[str]:
    return vlm.generate_clothing_from_image(bg_path)

def _rank_clothes_by_image(descriptions: list[str], top_k: int = 10):
    clothes_images = load_str_images_from_folder(CLOTHES_DIR)
    clothes = [
        (img_path.stem, Image.open(img_path).convert("RGB"))
        for img_path in clothes_images
    ]
    matcher = ModelRegistry.get("pe_clip_matcher")
    return matcher.match_clothes(
        descriptions=descriptions,
        query_emb=None,
        clothes=clothes,
        top_k=top_k,
    )

def _rank_clothes_by_caption(descriptions: list[str],
                            matcher_name: str = "text_matcher",
                            top_k: int = 10):
    clothes_captions = _get_clothes_captions()
    matcher = ModelRegistry.get(matcher_name)
    return matcher.match_clothes_captions(
        descriptions=descriptions,
        clothes_captions=clothes_captions,
        top_k=top_k,
    )

# _build_query_embedding logic moved to app/services/retrieval/samag_pipeline.py


def _rank_clothes_by_feedback(descriptions: list[str],
                            matcher_name: str = "text_matcher",
                            top_k: int = 10,
                            fb_text: str | None = None):
    clothes_captions = _get_clothes_captions()
    matcher = ModelRegistry.get(matcher_name)
    return matcher.match_clothes_captions(
        descriptions=descriptions,
        clothes_captions=clothes_captions,
        fb_text=fb_text,
        top_k=top_k,
    )

@router.post("/vlm-generated-clothes-captions")
def get_vlm_descriptions(
    image: UploadFile = File(...),
):
    bg_path = _save_bg_upload(image)

    model = ModelRegistry.get("vlm")
    res = model.generate_clothing_from_image(bg_path)

    return {
        "res": res,
    }
    
@router.post("/vlm-clip-images-matching")
def get_clothes_by_image_match(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates clothing descriptions
    3. PE-CLIP ranks clothes by similarity
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    bg_path = _save_bg_upload(image)

    # -------------------------
    # Generate clothing text (VLM)
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    descriptions = _generate_clothes_descriptions(vlm, bg_path)
    results = _rank_clothes_by_image(descriptions, top_k=10)

    # -------------------------
    # Response
    # -------------------------
    return {
        "query": descriptions,
        "results": results,
    }

@router.post("/vlm-clip-caption-matching")
def get_clothes_by_image_match_captions(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates clothing descriptions
    3. Matcher ranks clothes by similarity using captions JSON
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    bg_path = _save_bg_upload(image)

    # -------------------------
    # Generate clothing text (VLM)
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    descriptions = _generate_clothes_descriptions(vlm, bg_path)

    # -------------------------
    # Load clothes captions + match
    # -------------------------
    results = _rank_clothes_by_caption(descriptions, top_k=10)

    # -------------------------
    # Response
    # -------------------------
    return {
        "query": descriptions,
        "results": results,
    }
    
@router.get("/clothes-captions")
def get_clothes_captions():
    """
    Return clothes captions JSON.
    If it doesn't exist, generate it first.
    """

    # -------------------------
    # Load cached JSON if exists
    # -------------------------
    return _get_clothes_captions()


@router.post("/vlm-tournament-selection")
def get_best_clothes_by_tournament(image: UploadFile = File(...)):
    """
    1. Upload image
    2. VLM generates background caption
    3. VLM selects best clothes from captions in batches of 10
    4. Repeat until a single winner remains
    """

    # -------------------------
    # Save uploaded image
    # -------------------------
    bg_path = _save_bg_upload(image)

    # -------------------------
    # Background caption
    # -------------------------
    vlm = ModelRegistry.get("vlm")
    background_caption = vlm.generate_clothes_caption(
        str(bg_path),
        vlm.bg_caption,
    )

    # -------------------------
    # Load clothes captions
    # -------------------------
    clothes_captions = _get_clothes_captions()
    best_clothes = _select_clothes_via_tournament(vlm, background_caption, clothes_captions)
    if best_clothes is None:
        return {
            "background_caption": background_caption,
            "best_clothes": None,
        }
    print(f"[VLM] Best clothes: {best_clothes}", flush=True)

    return {
        "background_caption": background_caption,
        "best_clothes": best_clothes,
    }

from app.services.retrieval.samag_pipeline import SaMaGPipeline

@router.post("/vlm-faiss-composed-retrieval")
def composed_retrieval(image: UploadFile = File(...), top_k: int = 10):
    bg_path = _save_bg_upload(image)

    # -------------------------
    # Ablation Configuration
    # Toggle any of these flags to False to ablate a module.
    # -------------------------
    pipeline = SaMaGPipeline(
        use_visual=True, 
        use_semantic=True, 
        use_orthogonal_rejection=True,
        use_reranker=True
    )
    
    results = (
        pipeline
        .extract_visual_features(bg_path)
        .extract_semantic_features(bg_path)
        .formulate_query(w_semantic=0.4, w_scene=0.4, w_img=0.2)
        .execute_faiss_search(top_k=top_k)
        .rerank_candidates()
        .get_results()
    )
    
    return results


@router.post("/vlm-caption-feedback")
def get_clothes_by_feedback(payload: dict = Body(...)):

    descriptions = payload["descriptions"]
    fb_text = payload.get("fb_text")

    results = _rank_clothes_by_feedback(
        descriptions=descriptions,
        top_k=10,
        fb_text=fb_text,
    )

    return {
        "query": descriptions,
        "feedback": fb_text,
        "results": results,
    }


@router.post("/vlm-suggest-outfit")
def suggest_outfit(
    bg_image: UploadFile = File(...),
    preference_text: str | None = Form(None),
    feedback_text: str | None = Form(None)
):
    bg_path = _save_bg_upload(bg_image)

    vlm = ModelRegistry.get("vlm")

    outfit_desc = vlm.suggest_outfit_from_bg(str(bg_path), preference_text=preference_text, feedback_text=feedback_text)
    print(f"[vlm_ep] Outfit suggestion: {outfit_desc}")

    return {
        "bg_image": bg_image.filename,
        "outfit_description": outfit_desc,
    }