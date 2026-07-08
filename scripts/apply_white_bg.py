"""Add a solid white background to images with transparency.

Usage:
	python app/scripts/apply_white_bg.py
	python app/scripts/apply_white_bg.py --src <input_dir> --dst <output_dir>

The default input is `app/data/data/2d` and the output is `app/data/data/2d_white_bg`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image


def iter_images(src_dir: Path) -> Iterable[Path]:
	"""Yield image files from the source directory."""

	for path in src_dir.iterdir():
		if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
			yield path


def add_white_background(image_path: Path, output_path: Path) -> None:
	"""Apply a white background to an image and save it to output_path."""

	with Image.open(image_path) as img:
		if img.mode == "RGBA":
			background = Image.new("RGBA", img.size, (255, 255, 255, 255))
			background.paste(img, mask=img.split()[3])
			result = background.convert("RGB")
		else:
			result = img.convert("RGB")

		output_path.parent.mkdir(parents=True, exist_ok=True)
		result.save(output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Apply white background to images.")

	default_src = Path(__file__).resolve().parent.parent / "data" / "data" / "2d"
	default_dst = default_src.parent / "2d_white_bg"

	parser.add_argument(
		"--src",
		type=Path,
		default=default_src,
		help="Directory containing source images (default: app/data/data/2d)",
	)
	parser.add_argument(
		"--dst",
		type=Path,
		default=default_dst,
		help="Directory to write processed images (default: app/data/data/2d_white_bg)",
	)

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	src_dir: Path = args.src
	dst_dir: Path = args.dst

	if not src_dir.exists():
		raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

	print(f"Source directory: {src_dir}")
	print(f"Destination directory: {dst_dir}")

	processed = 0
	skipped = 0

	for image_path in iter_images(src_dir):
		output_path = dst_dir / image_path.name
		if output_path.exists():
			skipped += 1
			print(f"Skipping existing file: {output_path.name}")
			continue

		print(f"Processing: {image_path.name}")
		add_white_background(image_path, output_path)
		processed += 1

	print(f"Done. Processed: {processed}, Skipped: {skipped}, Output: {dst_dir}")


if __name__ == "__main__":
	main()
