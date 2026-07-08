"""
Image editing orchestration service.
Canonical location: app/services/image_edit/service.py
(Backward-compat shim remains at app/services/image_edit.py)
"""
from io import BytesIO
import sys
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import base64
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from fastapi import HTTPException
import mimetypes
import ssl

from app.models.registry import ModelRegistry
from app.utils.util import crop_clothes_region

load_dotenv()

SAVE_DIR = Path('results/image_edit/edited_image/')
CROPPED_SAVE_DIR = Path('results/image_edit/cropped_clothes/')
REF_IMAGE_PATH_MAN = Path('data/ref_images/man.png')
REF_IMAGE_PATH_WOMAN = Path('data/ref_images/woman.png')

API_KEY = os.getenv("IMAGEROUTER_API_KEY")
URL = "https://api.imagerouter.io/v1/openai/images/edits"
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "https://nondepressed-semipneumatically-eveline.ngrok-free.dev")
VLM_VERIFY_SSL = os.getenv("VLM_VERIFY_SSL", "true").lower() != "false"
VECTOR_DB_BASE_URL = os.getenv("VECTOR_DB_BASE_URL", "https://nondepressed-semipneumatically-eveline.ngrok-free.dev")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
MODEL = "openai/gpt-image-1.5:free"


def save_to_file(result, save_result, crop_clothes=True):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    img_id = sum(1 for _ in SAVE_DIR.glob("*.jpg")) + 1
    if save_result:
        res_path = SAVE_DIR / f"{img_id}_response.json"
        with open(res_path, 'w') as f:
            json.dump(result, f, indent=2)
    filepath = SAVE_DIR / f"{img_id}.jpg"
    cropped_filepath = None
    image_data = requests.get(result['data'][0]['url']).content
    with open(filepath, 'wb') as f:
        f.write(image_data)
    if crop_clothes:
        with Image.open(BytesIO(image_data)) as img:
            cropped_img = crop_clothes_region(img)
            CROPPED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            cropped_filepath = CROPPED_SAVE_DIR / f"{img_id}_cropped.jpg"
            cropped_img.save(cropped_filepath)
    return {"edited_path": str(filepath), "cropped_path": str(cropped_filepath) if cropped_filepath else None}


def save_edited_image(edited_image: Image.Image, cropped_image: Image.Image | None = None) -> dict:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    img_id = sum(1 for _ in SAVE_DIR.glob("*.jpg")) + 1
    filepath = SAVE_DIR / f"{img_id}.jpg"
    edited_image.save(filepath)
    result = {"edited_path": str(filepath)}
    if cropped_image:
        CROPPED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        cropped_filepath = CROPPED_SAVE_DIR / f"{img_id}_cropped.jpg"
        cropped_image.save(cropped_filepath)
        result["cropped_path"] = str(cropped_filepath)
    return result


def format_prompt(scene_description):
    return (
        "Change the outfit of this person into an outfit that is suitable "
        f"for a travel to {scene_description}. "
        "Do not change the background, just change the outfit."
    )


def edit_image_scene_desc(scene_description, save_result=True, gender='male', crop_clothes=True):
    prompt = format_prompt(scene_description)
    REF_IMAGE_PATH = REF_IMAGE_PATH_MAN if gender == 'male' else (
        REF_IMAGE_PATH_WOMAN if gender == 'female' else (_ for _ in ()).throw(ValueError("Gender must be 'male' or 'female'"))
    )
    if not REF_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Reference image not found: {REF_IMAGE_PATH}")
    data = {"model": MODEL, "prompt": prompt, "output_format": "jpg", "quality": "auto", "size": "auto"}
    with REF_IMAGE_PATH.open("rb") as ref_image:
        files = {"image": (REF_IMAGE_PATH.name, ref_image, f"image/{REF_IMAGE_PATH.suffix.lstrip('.')}")}
        response = requests.post(URL, headers=HEADERS, files=files, data=data)
        result = response.json()
        result['prompt'] = prompt
    if result.get('error'):
        raise RuntimeError(f"API Error: {result['error']}")
    paths = save_to_file(result=result, save_result=save_result, crop_clothes=crop_clothes)
    return {"result": result, "edited_path": paths["edited_path"], "cropped_path": paths["cropped_path"]}


def edit_image_scene_img(
    scene_path,
    save_result=True,
    gender='male',
    crop_clothes=True,
    preference_text: str | None = None,
    feedback_text: str | None = None,
    ref_image_path: Path | None = None,
):
    prompt_parts = [
        "Change the outfit of the given person into an outfit that matches the scene.",
        "Return the image of such person in the original background of that person.",
        "Do not add the person into the scene image.",
    ]
    if preference_text:
        prompt_parts.append(f"User preference: {preference_text}.")
    if feedback_text:
        prompt_parts.append(f"Additional feedback: {feedback_text}.")
    prompt = " ".join(prompt_parts)
    if not ref_image_path:
        REF = REF_IMAGE_PATH_MAN if gender == 'male' else REF_IMAGE_PATH_WOMAN
    else:
        REF = Path(ref_image_path)
    if not REF.is_file():
        raise FileNotFoundError(f"Reference image not found: {REF}")
    if not Path(scene_path).is_file():
        raise FileNotFoundError(f"Scene image not found: {scene_path}")
    data = {"model": MODEL, "prompt": prompt, "output_format": "jpg", "quality": "auto", "size": "auto"}
    files = [
        ("image", (REF.name, open(REF, "rb"), f"image/{REF.suffix.lstrip('.')}")),
        ("image", (Path(scene_path).name, open(scene_path, "rb"), f"image/{Path(scene_path).suffix.lstrip('.')}")),
    ]
    response = requests.post(URL, headers=HEADERS, files=files, data=data)
    result = response.json()
    if result.get("error"):
        raise RuntimeError(f"API Error: {result['error']}")
    result["prompt"] = prompt
    paths = save_to_file(result=result, save_result=save_result, crop_clothes=crop_clothes)
    return {"result": result, "edited_path": paths["edited_path"], "cropped_path": paths["cropped_path"], "ref_path": str(REF)}


def edit_image_outfit_desc(
    outfit_description: str,
    save_result=True,
    gender='male',
    crop_clothes=True,
    ref_image_path: Path | None = None,
    model_name='image_edit_flux',
):
    model = ModelRegistry.get(f"{model_name}")
    used_ref_path = (
        Path(ref_image_path) if ref_image_path
        else (REF_IMAGE_PATH_MAN if gender == 'male' else REF_IMAGE_PATH_WOMAN)
    )
    edited_image = model.edit_outfit_desc(
        outfit_description=outfit_description,
        gender=gender,
        ref_image_path=used_ref_path,
    )
    cropped_image = crop_clothes_region(edited_image) if crop_clothes else None
    paths = save_edited_image(edited_image=edited_image, cropped_image=cropped_image)
    paths["ref_path"] = str(used_ref_path)
    return paths


def get_outfit_suggestion_remote(
    bg_path: Path | str,
    preference_text: str | None = None,
    feedback_text: str | None = None,
    verify_ssl: bool | None = None,
) -> str:
    bg_path = Path(bg_path) if isinstance(bg_path, str) else bg_path
    url = f"{VLM_BASE_URL}/api/v1/retrieval/vlm-suggest-outfit"
    mime_type, _ = mimetypes.guess_type(bg_path)
    mime_type = mime_type or "image/png"
    verify = VLM_VERIFY_SSL if verify_ssl is None else verify_ssl

    headers = {"ngrok-skip-browser-warning": "true"}

    class SSLContextAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ctx = create_urllib3_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            kwargs['ssl_context'] = ctx
            return super().init_poolmanager(*args, **kwargs)

    def _send(verify_flag: bool):
        session = requests.Session()
        if not verify_flag:
            session.mount('https://', SSLContextAdapter())
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        with open(bg_path, "rb") as fp:
            files = {"bg_image": (bg_path.name, fp, mime_type)}
            data = {}
            if preference_text:
                data["preference_text"] = preference_text
            if feedback_text:
                data["feedback_text"] = feedback_text
            return session.post(url, files=files, data=data, headers=headers, timeout=60, verify=verify_flag)

    try:
        resp = _send(verify)
    except requests.exceptions.SSLError as ssl_err:
        if verify:
            try:
                resp = _send(False)
            except Exception as retry_err:
                raise HTTPException(status_code=502, detail=f"VLM SSL error: {ssl_err}; retry failed: {retry_err}") from ssl_err
        else:
            raise HTTPException(status_code=502, detail=f"VLM SSL protocol error: {ssl_err}") from ssl_err
    except requests.exceptions.RequestException as req_err:
        raise HTTPException(status_code=502, detail=f"VLM connection error: {req_err}") from req_err

    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"VLM service error {resp.status_code}: {resp.text}")
    try:
        data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid VLM response: {resp.text}") from exc
    outfit_desc = data.get("outfit_description")
    if not outfit_desc:
        raise HTTPException(status_code=502, detail="VLM response missing outfit_description")
    return outfit_desc


def retrieve_from_edited_image_remote(processed_image: Image.Image, top_k: int = 5):
    img_byte_arr = BytesIO()
    processed_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    url = f"{VECTOR_DB_BASE_URL}/api/v1/retrieval/vector_db"
    files = {"image": ("cropped_image.png", img_byte_arr, "image/png")}
    data = {"top_k": top_k}
    try:
        response = requests.post(url, files=files, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Vector DB service error: {e}") from e


# Aliases for backward compat
generate_outfit_image = edit_image_scene_img
generate_outfit_image_from_text = edit_image_outfit_desc
search_similar_clothes = retrieve_from_edited_image_remote
