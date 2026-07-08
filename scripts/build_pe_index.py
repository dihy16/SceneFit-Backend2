# build_peclip_faiss.py

import faiss
import torch
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import trange

from app.models.embedding.pe_clip_matcher import PEClipMatcher


@torch.no_grad()
def main():
    device = "cuda"
    batch_size = 32   # reduce to 16 if GPU is small

    clothes_dir = Path("data/2d")
    output_dir = Path("data/faiss")
    output_dir.mkdir(parents=True, exist_ok=True)

    matcher = PEClipMatcher(device=device, load_faiss=False)

    image_paths = sorted(
        p for p in clothes_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    assert len(image_paths) > 0, "No clothing images found."

    # -------- BATCHED IMAGE ENCODING --------
    all_embs = []

    for i in trange(0, len(image_paths), batch_size, desc="Encoding images"):
        batch_paths = image_paths[i : i + batch_size]

        images = []
        for p in batch_paths:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))

        embs = matcher.encode_image(images)   # (B, D)
        all_embs.append(embs.cpu())

    image_embs = torch.cat(all_embs, dim=0).numpy().astype("float32")

    # -------- NORMALIZE FOR COSINE SIMILARITY --------
    image_embs /= np.linalg.norm(image_embs, axis=1, keepdims=True)

    dim = image_embs.shape[1]

    # -------- FAISS INDEX --------
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(image_embs)

    faiss.write_index(index, str(output_dir / "clothes_image.index"))

    # -------- METADATA --------
    with open(output_dir / "clothes_image_meta.pkl", "wb") as f:
        pickle.dump(
            {
                "filenames": [p.name for p in image_paths]
            },
            f,
        )

    print(f"[OK] Indexed {len(image_paths)} clothing images.")


if __name__ == "__main__":
    main()