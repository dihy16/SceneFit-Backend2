"""
Image processing utilities.
Canonical location: app/utils/image_utils.py

Merges:
  - app/services/compose.py   (compose two images)
  - app/services/img_processor.py  (background removal, paste, batch compose)

Backward-compat shims remain at both original paths.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from rembg import remove


# ---------------------------------------------------------------------------
# Simple compositing (from compose.py)
# ---------------------------------------------------------------------------

def compose(bg_img: Image.Image, cloth_img: Image.Image) -> Image.Image:
    """Paste `cloth_img` centred on `bg_img` using its alpha channel."""
    bg = bg_img.copy().convert("RGBA")
    fg = cloth_img.convert("RGBA")
    x = (bg.width - fg.width) // 2
    y = (bg.height - fg.height) // 2
    bg.paste(fg, (x, y), fg)
    return bg


# ---------------------------------------------------------------------------
# Background removal (from img_processor.py)
# ---------------------------------------------------------------------------

def remove_background(input_path: str, output_path: str) -> None:
    """Remove the background from an image file and write the result."""
    with open(input_path, "rb") as f:
        input_image = f.read()
    output_image = remove(input_image)
    with open(output_path, "wb") as f:
        f.write(output_image)


def paste_centered(
    fg_path: str,
    bg_path: str,
    output_path: str,
    scale: float = 1.0,
) -> None:
    """
    Paste a foreground image (with transparency) centered on a background image.

    Args:
        fg_path: Path to foreground image (RGBA, background removed).
        bg_path: Path to background image.
        output_path: Where to save the result.
        scale: Optional scale factor for the foreground image.
    """
    fg = Image.open(fg_path).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")

    if scale != 1.0:
        fg_w, fg_h = fg.size
        fg = fg.resize((int(fg_w * scale), int(fg_h * scale)), Image.LANCZOS)

    bg_w, bg_h = bg.size
    fg_w, fg_h = fg.size
    x = (bg_w - fg_w) // 2
    y = (bg_h - fg_h) // 2
    bg.paste(fg, (x, y), fg)
    bg.save(output_path)


def compose_2d_on_background(
    bg_path: str,
    fg_dir: str = "data/2d",
    fg_files: List[str] | None = None,
    clothes_json: str = "data/clothes.json",
    scale: float = 1.0,
    return_format: str = "pil",  # "pil" | "numpy"
    output_dir: str = "app/outputs/composed",
    offset: int = 0,
    limit: int | None = None,
) -> List[Tuple[str, Image.Image]]:
    """
    Paste foreground images centered on a background image.

    Uses `fg_files` when provided; otherwise loads from `fg_dir` (falling back
    to `clothes_json` if the directory is empty).

    Args:
        bg_path: Path to background image.
        fg_dir: Directory of foreground PNG files.
        fg_files: Explicit list of filenames (overrides fg_dir).
        clothes_json: Fallback JSON list of filenames.
        scale: Foreground scale factor.
        return_format: "pil" returns PIL Images; "numpy" returns RGB arrays.
        output_dir: Where to save composed images (currently unused).
        offset: Skip first N files (for batching).
        limit: Process at most N files (for batching).

    Returns:
        List of (filename, image) tuples.
    """
    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")
    bg_original = Image.open(bg_path).convert("RGBA")
    os.makedirs(output_dir, exist_ok=True)

    if fg_files is None:
        if not os.path.isdir(fg_dir):
            raise RuntimeError(f"fg_dir does not exist: {fg_dir}")
        all_files = sorted(os.listdir(fg_dir))
        fg_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTS)]
        if not fg_files:
            if os.path.isfile(clothes_json):
                with open(clothes_json, "r") as f:
                    fg_files = json.load(f)
            if not fg_files:
                raise RuntimeError(f"No image files found in {fg_dir} and clothes.json is empty")
    else:
        fg_files = [str(p) for p in fg_files]
        if not fg_files:
            raise RuntimeError("fg_files is empty")

    total = len(fg_files)
    end_idx = offset + limit if limit is not None else total
    fg_files = fg_files[offset:end_idx]

    if not fg_files:
        return []

    results: List[Tuple[str, Image.Image]] = []
    for fg_file in fg_files:
        fg_path = os.path.join(fg_dir, fg_file)
        if not os.path.isfile(fg_path):
            continue
        fg = Image.open(fg_path).convert("RGBA")
        bg = bg_original.copy()
        if scale != 1.0:
            fg_w, fg_h = fg.size
            fg = fg.resize((int(fg_w * scale), int(fg_h * scale)), Image.LANCZOS)
        bg_w, bg_h = bg.size
        fg_w, fg_h = fg.size
        x = (bg_w - fg_w) // 2
        y = (bg_h - fg_h) // 2
        bg.paste(fg, (x, y), fg)
        if return_format == "pil":
            results.append((fg_file, bg))
        elif return_format == "numpy":
            results.append((fg_file, np.array(bg.convert("RGB"))))
        else:
            raise ValueError("return_format must be 'pil' or 'numpy'")

    if not results:
        raise RuntimeError("No valid foreground images found (all files missing?)")

    return results
