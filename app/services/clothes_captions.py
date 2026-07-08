# app/scripts/generate_clothes_captions.py

import json
from pathlib import Path
from tqdm import tqdm

from app.models.registry import ModelRegistry


def generate_clothes_captions_json(
    clothes_dir: Path = Path("data/2d"),
    output_path: Path = Path("data/clothes_captions.json"),
):
    """
    Use VLM to generate captions for all clothes images in a folder.

    Output format:
    {
        "image_name.png": "caption text"
    }
    """

    vlm = ModelRegistry.get("vlm")

    captions: dict[str, str] = {}

    image_files = sorted(
        p for p in clothes_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

    for img_path in tqdm(
        image_files,
        desc="Generating clothes captions",
        unit="image",
    ):
        # generate_clothes_caption returns a single string
        caption = vlm.generate_clothes_caption(str(img_path), vlm.clothes_caption)
        captions[img_path.name] = caption
        print(f"{img_path.name} : {caption}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved captions to: {output_path}")
    return captions

