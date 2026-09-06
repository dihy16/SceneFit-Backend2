"""Build the PE-CLIP FAISS image index, optionally from a benchmark manifest."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clothes-dir", type=Path, default=Path("data/2d"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/faiss"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser


def _image_paths(clothes_dir: Path, manifest_path: Path | None) -> list[Path]:
    if manifest_path is None:
        return sorted(
            path.resolve()
            for path in clothes_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [Path(entry["path"]).resolve() for entry in manifest["outfits"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Benchmark manifest references {len(missing)} missing outfits; "
            f"first missing path: {missing[0]}"
        )
    return paths


def _index_matches(output_dir: Path, image_paths: list[Path]) -> bool:
    import faiss

    index_path = output_dir / "clothes_image.index"
    metadata_path = output_dir / "clothes_image_meta.pkl"
    if not index_path.is_file() or not metadata_path.is_file():
        return False
    try:
        index = faiss.read_index(str(index_path))
        with metadata_path.open("rb") as stream:
            metadata = pickle.load(stream)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
        return False
    expected = [path.name for path in image_paths]
    return (
        index.ntotal == len(expected)
        and isinstance(metadata, dict)
        and metadata.get("filenames") == expected
    )


def main() -> int:
    import faiss
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import trange

    from app.models.embedding.pe_clip_matcher import PEClipMatcher

    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    image_paths = _image_paths(args.clothes_dir, args.manifest)
    if not image_paths:
        raise ValueError("No clothing images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if _index_matches(args.output_dir, image_paths):
        print(f"[PE-FAISS] Reusing index for {len(image_paths)} outfits")
        return 0

    matcher = PEClipMatcher(device=args.device, load_faiss=False)
    all_embeddings = []
    for offset in trange(
        0,
        len(image_paths),
        args.batch_size,
        desc="Encoding PE-CLIP outfits",
    ):
        batch_paths = image_paths[offset : offset + args.batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        all_embeddings.append(matcher.encode_image(images).cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy().astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.clip(norms, a_min=1e-12, a_max=None)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = args.output_dir / "clothes_image.index"
    metadata_path = args.output_dir / "clothes_image_meta.pkl"
    temporary_index = index_path.with_suffix(".index.tmp")
    temporary_metadata = metadata_path.with_suffix(".pkl.tmp")
    faiss.write_index(index, str(temporary_index))
    with temporary_metadata.open("wb") as stream:
        pickle.dump({"filenames": [path.name for path in image_paths]}, stream)
    temporary_index.replace(index_path)
    temporary_metadata.replace(metadata_path)

    print(f"[PE-FAISS] Indexed {len(image_paths)} clothing images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
