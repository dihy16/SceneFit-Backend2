# app/utils/util.py

import yaml
from pathlib import Path
from functools import lru_cache
from typing import List
import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
MAIN_PREFIX = "https://wifelier-melita-soapiest.ngrok-free.dev/"

_PROMPT_CONFIG_PATH = Path("config/prompts.yaml")
_CLOTHES_JSON = Path("data/clothes.json")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def crop_clothes_region(
    image: Image.Image,
    bottom_fraction: float = 0.65,
    horizontal_fraction: float = 0.50,
    keep_square: bool = False,
) -> Image.Image:
    """
    Heuristic crop for clothing on front-facing, full-body renders.

    Args:
        image: PIL image of a full-body character on solid background.
        bottom_fraction: Keep this fraction of the image height from the bottom (e.g., 0.70 keeps bottom 70%).
        horizontal_fraction: Fraction of image width to keep, centered horizontally (e.g., 0.60 keeps centered 60%).
        keep_square: If True, expand the crop to a square around the center.

    Returns:
        Cropped PIL image focused on the clothing region.
    """

    w, h = image.size

    # Vertical crop: keep the bottom N% (default 70%)
    bottom = h
    top = max(0, int(h * (1.0 - bottom_fraction)))

    # Horizontal crop: keep a centered fraction (default 60%)
    horizontal_fraction = max(0.0, min(1.0, horizontal_fraction))
    crop_width = int(w * horizontal_fraction)
    crop_width = max(1, min(w, crop_width))
    center_x = w // 2
    half_width = crop_width // 2
    left = max(0, center_x - half_width)
    right = min(w, center_x + half_width)

    # Optional square adjustment to keep downstream resize behavior predictable
    if keep_square:
        crop_w = right - left
        crop_h = bottom - top
        side = max(crop_w, crop_h)
        half_side = side // 2
        left = max(0, center_x - half_side)
        right = min(w, center_x + half_side)
        mid_y = (top + bottom) // 2
        top = max(0, mid_y - half_side)
        bottom = min(h, mid_y + half_side)

    # Guard against degenerate boxes
    if right - left <= 1 or bottom - top <= 1:
        return image

    return image.crop((left, top, right, bottom))

def load_prompts(task: str) -> list[str]:
    with open(_PROMPT_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    if task not in data:
        raise KeyError(f"Prompt task '{task}' not found in prompts.yaml")

    prompts = data[task]
    return [prompts["positive"], prompts["negative"]]

def load_prompt_by_key(key: str) -> str:
    with open(_PROMPT_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    if key not in data:
        raise KeyError(f"Prompt key '{key}' not found in prompts.yaml")

    return data[key]

@lru_cache(maxsize=1)
def get_clothes_list() -> List[str]:
    with open(_CLOTHES_JSON, "r") as f:
        return json.load(f)

def load_images_from_folder(
    folder: str | Path,
    recursive: bool = True,
    max_images: int | None = None,
) -> List[Image.Image]:
    """
    Load all images from a folder into a list of PIL Images.

    Args:
        folder: Path to folder containing images
        recursive: Whether to search subfolders
        max_images: Optional cap to limit number of images loaded

    Returns:
        List[PIL.Image.Image]
    """
    folder = BASE_DIR / folder
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    pattern = "**/*" if recursive else "*"

    images: List[Image.Image] = []
    for path in sorted(folder.glob(pattern)):
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            # Skip corrupted images but don’t crash
            print(f"[WARN] Failed to load image {path}: {e}")

        if max_images is not None and len(images) >= max_images:
            break

    if not images:
        raise RuntimeError(f"No valid images found in {folder}")

    return images

def load_str_images_from_folder(folder: str | Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)

    return [
        p for p in folder.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    

def convert_filename_to_url(filename: str) -> str:
    """
    Convert a local filename to a URL path for accessing via StaticFiles.

    Args:
        filename: Local filename (e.g., "data/2d/outfit1.png")

    Returns:
        URL path (e.g., "/images/outfit1.png")
    """
    
    url = MAIN_PREFIX + f"images/{filename}"
    return url

if __name__ == "__main__":
    # Test cropping function
    # test_image_path = BASE_DIR / "data/edited_image/17.jpg"
    # with Image.open(test_image_path) as img:
    #     cropped = crop_clothes_region(img, keep_square=True)
    #     cropped.show()
    image_name = 'avatars_0ea0469c16cd4cf1be30b26b37b4f6e7.png'
    print(convert_filename_to_url(image_name))