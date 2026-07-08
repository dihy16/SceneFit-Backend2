import os
import uuid
import io
import mimetypes
from pathlib import Path
from typing import Optional
import numpy as np
import requests
from fastapi import APIRouter, UploadFile, File, Form, Request, Depends, HTTPException

from app.services.image_edit.service import edit_image_scene_img, edit_image_outfit_desc, get_outfit_suggestion_remote
from app.models.registry import ModelRegistry
from app.services.asr.service import load_audio_from_upload, convert_speech_to_text

router = APIRouter()

BG_DIR = Path("app/uploads/bg")
BG_DIR.mkdir(parents=True, exist_ok=True)
ALL_BG_DIR = Path("data/bg")
RETRIEVAL_RESULTS_DIR = Path("results/image_edit")
RETRIEVAL_SESSIONS: dict[str, dict[str, str | None]] = {}


def _as_optional_str(value: Path | str | None) -> str | None:
    if value is None:
        return None
    return str(value)

def get_vector_db(request: Request):
    return request.app.state.vector_db

def _extract_outfit_name(meta: str | None) -> str | None:
    if not meta:
        return None
    return Path(meta).stem  # filename without extension

def _save_upload(file: UploadFile, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".png"
    filename = f"{uuid.uuid4().hex}{suffix}"
    out_path = directory / filename
    with open(out_path, "wb") as f:
        f.write(file.file.read())
    return out_path

@router.post("/image-edit")
def retrieve_clothes_image_edit(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    gender: str = Form("male"),
    crop_clothes: bool = Form(True),
    return_metadata: bool = Form(True),
    preference_text: str | None = Form(None),
    preference_audio: UploadFile | None = File(None),
    vector_db = Depends(get_vector_db),
):
    # -------------------------------------------------
    # 1. Save uploaded background
    # -------------------------------------------------
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename

    with open(bg_path, "wb") as f:
        f.write(image.file.read())

    # -------------------------------------------------
    # 2. Get GPT edited images
    # -------------------------------------------------
    pref_text = preference_text or convert_speech_to_text(preference_audio) if preference_audio else ""
    print(f"[image_edit_ep] Preference text: {pref_text}")

    edit_result = None
    print("[image_edit_ep] Editing image via GPT...")
    edit_result = edit_image_scene_img(
        bg_path,
        save_result=True,
        gender=gender,
        crop_clothes=crop_clothes,
        preference_text=pref_text,
        ref_image_path=None
    )
    processed_image_path = edit_result.get("cropped_path") if crop_clothes else edit_result.get("edited_path")
    ref_image_path = edit_result.get("ref_path")
    print(f"[image_edit_ep] Reference image used: {ref_image_path}")
    print(f"[image_edit_ep] Processed image saved to: {processed_image_path}")

    # -------------------------------------------------
    # 3. Score using PE-Core model
    # -------------------------------------------------
    print("[image_edit_ep] Retrieving best matched clothes via vector DB...")
    scores = vector_db.search_by_image(processed_image_path, top_k=top_k)

    session_id = uuid.uuid4().hex
    edited_path_raw = edit_result.get("edited_path") if edit_result else processed_image_path
    cropped_path_raw = edit_result.get("cropped_path") if edit_result else None
    RETRIEVAL_SESSIONS[session_id] = {
        "bg_path": str(bg_path),
        "edited_path": _as_optional_str(edited_path_raw),
        "cropped_path": _as_optional_str(cropped_path_raw),
        "ref_path": _as_optional_str(ref_image_path),
        "gender": gender,
        "preference_text": pref_text,
    }

    print("[image_edit_ep] Returning results...")
    return [
        {
            "name": _extract_outfit_name(s.get("metadata")),
            "score": s["score"],
        }
        for s in scores[:top_k]
    ]

@router.post("/image-edit-flux")
def retrieve_clothes_image_edit_flux(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    gender: str = Form("male"),
    crop_clothes: bool = Form(True),
    return_metadata: bool = Form(True),
    preference_text: str | None = Form(None),
    preference_audio: UploadFile | None = File(None),
    vector_db = Depends(get_vector_db),
):
    # -------------------------------------------------
    # 1. Save uploaded background
    # -------------------------------------------------
    suffix = Path(image.filename).suffix or ".png"
    bg_filename = f"{uuid.uuid4().hex}{suffix}"
    bg_path = BG_DIR / bg_filename

    with open(bg_path, "wb") as f:
        f.write(image.file.read())

    # -------------------------------------------------
    # 2. Get outfit suggestion from remote VLM service
    # -------------------------------------------------
    pref_text = preference_text or convert_speech_to_text(preference_audio) if preference_audio else ""
    print(f"[image_edit_ep] Preference text: {pref_text}")
    
    outfit_desc = get_outfit_suggestion_remote(bg_path, preference_text=pref_text)
    print(f"[image_edit_ep] Outfit suggestion: {outfit_desc}")

    # -------------------------------------------------
    # 3. Image Edit with Flux
    # -------------------------------------------------
    edit_result = None
    print("[image_edit_ep] Editing image via Flux...")
    edit_result = edit_image_outfit_desc(
        outfit_description=outfit_desc,
        gender=gender,
        crop_clothes=crop_clothes,
        ref_image_path=None
    )
    processed_image_path = edit_result.get("cropped_path") if crop_clothes else edit_result.get("edited_path")
    ref_image_path = edit_result.get("ref_path")
    print(f"[image_edit_ep] Reference image used: {ref_image_path}")
    print(f"[image_edit_ep] Processed image saved to: {processed_image_path}")

    # -------------------------------------------------
    # 4. Score using PE-Core model
    # -------------------------------------------------
    print("[image_edit_ep] Retrieving best matched clothes via vector DB...")
    scores = vector_db.search_by_image(processed_image_path, top_k=top_k)

    session_id = uuid.uuid4().hex
    edited_path_raw = edit_result.get("edited_path") if edit_result else processed_image_path
    cropped_path_raw = edit_result.get("cropped_path") if edit_result else None
    RETRIEVAL_SESSIONS[session_id] = {
        "bg_path": str(bg_path),
        "edited_path": _as_optional_str(edited_path_raw),
        "cropped_path": _as_optional_str(cropped_path_raw),
        "ref_path": _as_optional_str(ref_image_path),
        "gender": gender,
        "preference_text": pref_text,
    }

    print("[image_edit_ep] Returning results...")
    return [
        {
            "name": _extract_outfit_name(s.get("metadata")),
            "score": s["score"],
        }
        for s in scores[:top_k]
    ]

@router.post("/image-edit-flux/apply-feedback")
def apply_feedback_image_edit_flux(
    session_id: str = Form(...),
    top_k: int = Form(5),
    crop_clothes: bool = Form(True),
    feedback_text: str | None = Form(None),
    feedback_audio: UploadFile | None = File(None),
    vector_db = Depends(get_vector_db),
):
    session = RETRIEVAL_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new retrieval first.")

    # -------------------------------------------------
    # 2. Get outfit suggestion from remote VLM service
    # -------------------------------------------------
    outfit_desc = get_outfit_suggestion_remote(
        session["bg_path"],
        preference_text=session.get("preference_text", ""),
        feedback_text=feedback_text or convert_speech_to_text(feedback_audio)
    )
    
    print(f"[image_edit_ep] Outfit suggestion: {outfit_desc}")

    pref_text = session.get("preference_text", "")
    fb_text = feedback_text or convert_speech_to_text(feedback_audio)
    print(f"[image_edit_ep] Preference text: {pref_text}")
    print(f"[image_edit_ep] Feedback text: {fb_text}")

    ref_image_path = session.get("edited_path")
    if ref_image_path is None:
        raise HTTPException(status_code=400, detail="Session is missing edited image.")

    print("[image_edit_ep] Applying feedback via Flux on existing edit...")
    edit_result = edit_image_outfit_desc(
        outfit_description=outfit_desc,
        gender=session.get("gender", "male"),
        crop_clothes=crop_clothes,
        ref_image_path=ref_image_path
    )

    processed_image_path = edit_result.get("cropped_path") if crop_clothes else edit_result.get("edited_path")
    print(f"[image_edit_ep] Processed image saved to: {processed_image_path}")

    print("[image_edit_ep] Retrieving best matched clothes via vector DB (feedback)...")
    scores = vector_db.search_by_image(processed_image_path, top_k=top_k)

    # Update session with latest edit
    session["edited_path"] = _as_optional_str(edit_result.get("edited_path"))
    session["cropped_path"] = _as_optional_str(edit_result.get("cropped_path"))
    session["preference_text"] = pref_text
    RETRIEVAL_SESSIONS[session_id] = session

    print("[image_edit_ep] Returning results...")
    return [
        {
            "name": _extract_outfit_name(s.get("metadata")),
            "score": s["score"],
        }
        for s in scores[:top_k]
    ]

@router.post("/image-edit/apply-feedback")
def apply_feedback_image_edit(
    session_id: str = Form(...),
    top_k: int = Form(5),
    crop_clothes: bool = Form(True),
    feedback_text: str | None = Form(None),
    feedback_audio: UploadFile | None = File(None),
    vector_db = Depends(get_vector_db),
):
    session = RETRIEVAL_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new retrieval first.")

    pref_text = session.get("preference_text", "")
    fb_text = feedback_text or convert_speech_to_text(feedback_audio)
    print(f"[image_edit_ep] Preference text: {pref_text}")
    print(f"[image_edit_ep] Feedback text: {fb_text}")

    scene_input = session.get("bg_path")
    if scene_input is None:
        raise HTTPException(status_code=400, detail="Session is missing scene image.")

    ref_image_path = session.get("edited_path")
    if ref_image_path is None:
        raise HTTPException(status_code=400, detail="Session is missing edited image.")

    print("[image_edit_ep] Applying feedback via GPT on existing edit...")
    edit_result = edit_image_scene_img(
        scene_input,
        save_result=False,
        gender=session.get("gender", "male"),
        crop_clothes=crop_clothes,
        preference_text=pref_text,
        feedback_text=fb_text,
        ref_image_path=ref_image_path
    )

    processed_image_path = edit_result.get("cropped_path") if crop_clothes else edit_result.get("edited_path")
    print(f"[image_edit_ep] Processed image saved to: {processed_image_path}")

    print("[image_edit_ep] Retrieving best matched clothes via vector DB (feedback)...")
    scores = vector_db.search_by_image(processed_image_path, top_k=top_k)

    # Update session with latest edit
    session["edited_path"] = _as_optional_str(edit_result.get("edited_path"))
    session["cropped_path"] = _as_optional_str(edit_result.get("cropped_path"))
    session["preference_text"] = pref_text
    RETRIEVAL_SESSIONS[session_id] = session

    print("[image_edit_ep] Returning results...")
    return [
        {
            "name": _extract_outfit_name(s.get("metadata")),
            "score": s["score"],
        }
        for s in scores[:top_k]
    ]

@router.post("/image-edit/retrieve-all")
def retrieve_all_backgrounds(
    top_k: int = Form(5),
    crop_clothes: bool = Form(True),
    return_metadata: bool = Form(True),
    vector_db = Depends(get_vector_db),
):
    if not RETRIEVAL_RESULTS_DIR.exists():
        os.makedirs(RETRIEVAL_RESULTS_DIR)

    results = []
    for bg_file in ALL_BG_DIR.glob("*"):
        for gender in ['male', 'female']:
            # ------------------------------
            # 1. Edit image via GPT
            # ------------------------------

            print(f"[image_edit_ep] Editing image via GPT for background: {bg_file} with gender: {gender}...")
            edit_result = edit_image_scene_img(bg_file, save_result=False, gender=gender, crop_clothes=crop_clothes)
            processed_image_path = edit_result.get("cropped_path") if crop_clothes else edit_result.get("edited_path")
            print(f"[image_edit_ep] Processed image saved to: {processed_image_path}")

            # ------------------------------
            # 2. Score using PE-Core model
            # ------------------------------

            print("[image_edit_ep] Retrieving best matched clothes via vector DB...")
            scores = vector_db.search_by_image(processed_image_path, top_k=top_k)

            # ------------------------------
            # 3. Top-K
            # ------------------------------
            print("[image_edit_ep] Returning results...")
            entry = {
                "method": "image-edit",
                "gender": gender,
                "bg_path": str(bg_file),
                "edited_image_path": str(processed_image_path),
                "count": min(top_k, len(scores)),
                "results": [
                    {
                        "name": _extract_outfit_name(s.get("metadata")),
                        "score": s["score"],
                    }
                    for s in scores[:top_k]
                ],
            }

            if not return_metadata:
                for item in entry["results"]:
                    item.pop("clothes_path", None)

            results.append(entry)

    with open(RETRIEVAL_RESULTS_DIR / "retrieval_results.json", "w") as f:
        import json
        json.dump(results, f, indent=2)

    return results