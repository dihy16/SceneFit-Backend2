"""Create a compact Colab dataset archive for the retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.benchmark.core import create_manifest


def _path(value: str) -> Path:
    return Path(value).resolve()


def _write_file(archive: zipfile.ZipFile, path: Path, root: Path) -> None:
    archive.write(path, path.relative_to(root).as_posix())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=_path, default=REPO_ROOT / "benchmark_data.zip")
    parser.add_argument("--scenes-dir", type=_path, default=REPO_ROOT / "data" / "bg")
    parser.add_argument("--outfits-dir", type=_path, default=REPO_ROOT / "data" / "2d")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--num-outfits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = create_manifest(
        root=REPO_ROOT,
        scenes_dir=args.scenes_dir,
        outfits_dir=args.outfits_dir,
        num_scenes=args.num_scenes,
        num_outfits=args.num_outfits,
        seed=args.seed,
    )
    selected = [*manifest["scenes"], *manifest["outfits"]]
    extra_paths = [
        REPO_ROOT / "data" / "clothes.json",
        REPO_ROOT / "data" / "clothes_captions.json",
    ]
    reference_dir = REPO_ROOT / "data" / "ref_images"
    if reference_dir.is_dir():
        extra_paths.extend(path for path in reference_dir.rglob("*") if path.is_file())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in selected:
            _write_file(archive, REPO_ROOT / entry["path"], REPO_ROOT)
        for path in extra_paths:
            if path.is_file():
                _write_file(archive, path, REPO_ROOT)
        archive.writestr(
            "results/benchmark/latest/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    print(
        f"Wrote {args.output} with {len(manifest['scenes'])} scenes and "
        f"{len(manifest['outfits'])} outfits"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
