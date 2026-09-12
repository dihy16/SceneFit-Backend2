"""Serve a Gradio UI for comparing benchmark rankings.

Example:
    python scripts/visualize_benchmark_rankings.py \
        --run-dir results/benchmark/all-methods-evaluation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import textwrap
import threading
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "benchmark" / "all-methods-evaluation"
METHOD_ORDER = ("aesthetic", "clip", "vlm", "image_edit")
RELEVANCE_COLORS = {
    5: "#1b5e20",
    4: "#2e7d32",
    3: "#8a6d00",
    2: "#b45309",
    1: "#b91c1c",
}


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _display_name(method: str) -> str:
    names = {
        "aesthetic": "Aesthetic",
        "clip": "CLIP",
        "vlm": "VLM",
        "image_edit": "ImageEdit",
    }
    return names.get(method, method.replace("_", " ").title())


@dataclass(frozen=True)
class RankedOutfit:
    outfit_id: str
    rank: int
    raw_score: float | None
    relevance: float
    image_path: Path


@dataclass(frozen=True)
class BenchmarkData:
    scenes: list[dict[str, str]]
    methods: list[str]
    rankings: dict[str, dict[str, list[RankedOutfit]]]
    ground_truth: dict[str, list[RankedOutfit]]
    repo_root: Path
    data_archive: Path | None
    score_label: str


def _load_judgments(path: Path) -> dict[str, dict[str, int]]:
    judgments: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                scene_id = str(row["scene_id"])
                outfit_id = str(row["outfit_id"])
                score = int(row["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid judgment at line {line_number}"
                ) from exc
            if not 1 <= score <= 5:
                raise ValueError(
                    f"Judgment score must be 1-5 at line {line_number}"
                )
            judgments.setdefault(scene_id, {})[outfit_id] = score
    return judgments


def load_benchmark(
    run_dir: Path,
    repo_root: Path,
    data_archive: Path | None = None,
    judge_dir: Path | None = None,
) -> BenchmarkData:
    """Load a completed benchmark run and validate its rank references."""
    manifest_path = run_dir / "manifest.json"
    judge_source = judge_dir or run_dir
    judgments_path = judge_source / "judgments.jsonl"
    ratings_path = judge_source / "ratings.json"
    rankings_dir = run_dir / "rankings"
    if (
        not manifest_path.is_file()
        or (not judgments_path.is_file() and not ratings_path.is_file())
        or not rankings_dir.is_dir()
    ):
        raise FileNotFoundError(
            "Run directory must contain manifest.json, either judgments.jsonl or ratings.json, "
            "and rankings/"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes", [])
    outfits = manifest.get("outfits", [])
    if not scenes or not outfits:
        raise ValueError("Manifest must contain non-empty scenes and outfits")

    outfit_images = {
        str(outfit["id"]): repo_root / str(outfit["path"])
        for outfit in outfits
    }
    pairwise = ratings_path.is_file()
    judgments = {} if pairwise else _load_judgments(judgments_path)
    rating_matrix: dict[str, dict[str, float]] = {}
    if pairwise:
        ratings_payload = json.loads(ratings_path.read_text(encoding="utf-8"))
        for scene_id, scene_data in ratings_payload.get("scenes", {}).items():
            overall = scene_data.get("criteria", {}).get("overall", [])
            rating_matrix[str(scene_id)] = {
                str(item["outfit_id"]): float(item["rating"]) for item in overall
            }
    ground_truth: dict[str, list[RankedOutfit]] = {}
    for scene in scenes:
        scene_id = str(scene["id"])
        try:
            scene_judgments = rating_matrix[scene_id] if pairwise else judgments[scene_id]
        except KeyError as exc:
            raise ValueError(f"No judgments found for scene '{scene_id}'") from exc
        missing = set(outfit_images) - set(scene_judgments)
        if missing:
            raise ValueError(
                f"Scene '{scene_id}' is missing {len(missing)} outfit judgments"
            )
        ordered_ids = sorted(
            outfit_images,
            key=lambda outfit_id: (-scene_judgments[outfit_id], outfit_id),
        )
        ground_truth[scene_id] = [
            RankedOutfit(
                outfit_id=outfit_id,
                rank=rank,
                raw_score=None,
                relevance=float(scene_judgments[outfit_id]),
                image_path=outfit_images[outfit_id],
            )
            for rank, outfit_id in enumerate(ordered_ids, 1)
        ]

    rankings: dict[str, dict[str, list[RankedOutfit]]] = {}
    for ranking_path in sorted(rankings_dir.glob("*.json")):
        payload = json.loads(ranking_path.read_text(encoding="utf-8"))
        method = str(payload.get("method", ranking_path.stem))
        method_scenes: dict[str, list[RankedOutfit]] = {}
        for scene in scenes:
            scene_id = str(scene["id"])
            rows = payload.get("scenes", {}).get(scene_id)
            if not isinstance(rows, list):
                raise ValueError(
                    f"{method} has no ranking for scene '{scene_id}'"
                )
            ranked: list[RankedOutfit] = []
            for row in sorted(rows, key=lambda item: int(item["rank"])):
                outfit_id = str(row["outfit_id"])
                if outfit_id not in outfit_images:
                    raise ValueError(
                        f"Unknown outfit '{outfit_id}' in {method} ranking"
                    )
                try:
                    relevance = (
                        rating_matrix[scene_id][outfit_id]
                        if pairwise else judgments[scene_id][outfit_id]
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"No judgment for '{scene_id}' / '{outfit_id}'"
                    ) from exc
                raw_score = row.get("raw_score")
                ranked.append(
                    RankedOutfit(
                        outfit_id=outfit_id,
                        rank=int(row["rank"]),
                        raw_score=(
                            float(raw_score) if raw_score is not None else None
                        ),
                        relevance=relevance,
                        image_path=outfit_images[outfit_id],
                    )
                )
            method_scenes[scene_id] = ranked
        rankings[method] = method_scenes

    if not rankings:
        raise ValueError(f"No ranking JSON files found in {rankings_dir}")
    methods = [method for method in METHOD_ORDER if method in rankings]
    methods.extend(sorted(set(rankings) - set(methods)))
    return BenchmarkData(
        scenes=list(scenes),
        methods=methods,
        rankings=rankings,
        ground_truth=ground_truth,
        repo_root=repo_root,
        data_archive=(
            data_archive if data_archive and data_archive.is_file() else None
        ),
        score_label="overall Elo" if pairwise else "relevance",
    )


class ImageStore:
    """Load and cache unpacked images or members of the data archive."""

    def __init__(self, repo_root: Path, data_archive: Path | None) -> None:
        self.repo_root = repo_root.resolve()
        self.data_archive = data_archive
        self._temp_dir = tempfile.TemporaryDirectory(prefix="scenefit-gradio-")
        self._cache: dict[tuple[Path, int], str] = {}
        self._sheet_cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def load(self, path: Path, max_size: int = 512) -> str:
        resolved = path.resolve()
        cache_key = (resolved, max_size)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            if resolved.is_file():
                with Image.open(resolved) as image:
                    loaded = image.convert("RGB").copy()
            elif self.data_archive is not None:
                member = resolved.relative_to(self.repo_root).as_posix()
                with zipfile.ZipFile(self.data_archive) as archive:
                    with archive.open(member) as stream:
                        loaded = Image.open(
                            BytesIO(stream.read())
                        ).convert("RGB")
            else:
                raise FileNotFoundError(resolved)
        except (FileNotFoundError, KeyError, OSError, ValueError, zipfile.BadZipFile):
            loaded = Image.new("RGB", (512, 512), "#eeeeee")
            draw = ImageDraw.Draw(loaded)
            draw.multiline_text(
                (24, 24),
                f"Missing image\n{path.name}",
                fill="#333333",
                spacing=8,
            )

        loaded.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        digest = hashlib.sha1(
            f"{resolved}:{max_size}".encode("utf-8")
        ).hexdigest()
        display_path = Path(self._temp_dir.name) / f"{digest}.jpg"
        loaded.save(display_path, format="JPEG", quality=85)

        with self._lock:
            self._cache[cache_key] = str(display_path)
        return str(display_path)

    def save_sheet(self, key: str, image: Image.Image) -> str:
        """Cache a rendered ranking sheet and return its JPEG path."""
        with self._lock:
            cached = self._sheet_cache.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        path = Path(self._temp_dir.name) / f"sheet-{digest}.jpg"
        image.save(path, format="JPEG", quality=88)
        with self._lock:
            self._sheet_cache[key] = str(path)
        return str(path)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _ranking_sheet(
    outfits: list[RankedOutfit],
    image_store: ImageStore,
    method: str | None,
    cache_key: str,
    score_label: str,
) -> str:
    """Render ranked outfits as one efficient, top-to-bottom image."""
    width = 720
    row_height = 160
    margin = 10
    thumbnail_size = row_height - 2 * margin
    sheet = Image.new("RGB", (width, row_height * len(outfits)), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(20)
    detail_font = _font(17)

    for index, outfit in enumerate(outfits):
        y = index * row_height
        with Image.open(
            image_store.load(outfit.image_path, max_size=thumbnail_size)
        ) as thumbnail:
            copied = thumbnail.convert("RGB")
            image_x = margin + (thumbnail_size - copied.width) // 2
            image_y = y + margin + (thumbnail_size - copied.height) // 2
            sheet.paste(copied, (image_x, image_y))
        text_x = thumbnail_size + 2 * margin
        if method is None:
            title = (
                f"Ground truth #{outfit.rank}  |  "
                f"{score_label} {outfit.relevance:.1f}"
            )
            score_line = f"Oracle: {score_label} descending"
        else:
            title = (
                f"{_display_name(method)} #{outfit.rank}  |  "
                f"{score_label} {outfit.relevance:.1f}"
            )
            raw = (
                "n/a"
                if outfit.raw_score is None
                else f"{outfit.raw_score:.4f}"
            )
            score_line = f"Raw retrieval score: {raw}"
        draw.text(
            (text_x, y + 18),
            title,
            fill=RELEVANCE_COLORS.get(int(round(outfit.relevance)), "#1b5e20"),
            font=title_font,
        )
        wrapped_id = textwrap.wrap(outfit.outfit_id, width=47) or [""]
        draw.text(
            (text_x, y + 53),
            f"Outfit: {wrapped_id[0]}",
            fill="#111111",
            font=detail_font,
        )
        if len(wrapped_id) > 1:
            draw.text(
                (text_x + 55, y + 78),
                wrapped_id[1],
                fill="#111111",
                font=detail_font,
            )
        draw.text(
            (text_x, y + 113),
            score_line,
            fill="#444444",
            font=detail_font,
        )
        if index:
            draw.line((0, y, width, y), fill="#dddddd", width=1)
    return image_store.save_sheet(cache_key, sheet)


def render_rank_window(
    data: BenchmarkData,
    image_store: ImageStore,
    scene_number: float,
    method_label: str,
    rank_start: float,
    top_k: int,
    rows_per_page: int,
) -> tuple[str, str, str, str]:
    """Render one lightweight page of the two complete ranking lists."""
    scene_index = min(max(int(scene_number) - 1, 0), len(data.scenes) - 1)
    scene = data.scenes[scene_index]
    scene_id = str(scene["id"])
    method_lookup = {_display_name(method): method for method in data.methods}
    if method_label not in method_lookup:
        raise ValueError(f"Unknown method: {method_label}")
    method = method_lookup[method_label]

    maximum_start = max(0, top_k - rows_per_page)
    start = min(max(int(rank_start) - 1, 0), maximum_start)
    end = min(start + rows_per_page, top_k)
    ground_rows = data.ground_truth[scene_id][start:end]
    method_rows = data.rankings[method][scene_id][:top_k][start:end]
    scene_path = data.repo_root / str(scene["path"])
    status = (
        f"### {scene_id}\n"
        f"Showing ranks **{start + 1}-{end} of {top_k}** — "
        f"ground truth vs. **{_display_name(method)}**"
    )
    ground_sheet = _ranking_sheet(
        ground_rows,
        image_store,
        None,
        f"ground:{scene_id}:{start}:{end}",
        data.score_label,
    )
    method_sheet = _ranking_sheet(
        method_rows,
        image_store,
        method,
        f"method:{method}:{scene_id}:{start}:{end}",
        data.score_label,
    )
    return (
        image_store.load(scene_path, max_size=900),
        status,
        ground_sheet,
        method_sheet,
    )


def build_app(
    data: BenchmarkData,
    top_k: int,
    rows_per_page: int,
) -> Any:
    """Build the Gradio Blocks application."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Run: python -m pip install -r "
            "requirements.txt"
        ) from exc

    image_store = ImageStore(data.repo_root, data.data_archive)
    method_labels = [_display_name(method) for method in data.methods]
    maximum_start = max(1, top_k - rows_per_page + 1)

    def render(
        scene_number: float,
        method_label: str,
        rank_start: float,
    ) -> tuple[str, str, str, str]:
        return render_rank_window(
            data,
            image_store,
            scene_number,
            method_label,
            rank_start,
            top_k,
            rows_per_page,
        )

    def move_page(
        offset: int,
        scene_number: float,
        method_label: str,
        rank_start: float,
    ) -> tuple[float, str, str, str, str]:
        next_start = min(
            max(int(rank_start) + offset, 1),
            maximum_start,
        )
        return next_start, *render(scene_number, method_label, next_start)

    with gr.Blocks(title="SceneFit benchmark rankings") as demo:
        gr.Markdown("# SceneFit benchmark ranking comparison")
        with gr.Row():
            scene_slider = gr.Slider(
                minimum=1,
                maximum=len(data.scenes),
                value=1,
                step=1,
                label="Scene",
            )
            method_radio = gr.Radio(
                choices=method_labels,
                value=method_labels[0],
                label="Method",
            )
            rank_slider = gr.Slider(
                minimum=1,
                maximum=maximum_start,
                value=1,
                step=1,
                label="First visible rank",
            )
        with gr.Row():
            previous_button = gr.Button(
                f"Previous {rows_per_page}",
                variant="secondary",
            )
            next_button = gr.Button(
                f"Next {rows_per_page}",
                variant="secondary",
            )

        status = gr.Markdown()
        scene_image = gr.Image(
            label="Selected scene",
            type="pil",
            interactive=False,
            height=320,
        )
        with gr.Row(equal_height=True):
            ground_gallery = gr.Image(
                label="Ground truth — relevance descending",
                type="filepath",
                interactive=False,
                height=1100,
            )
            method_gallery = gr.Image(
                label="Selected method — retrieval order",
                type="filepath",
                interactive=False,
                height=1100,
            )

        inputs = [scene_slider, method_radio, rank_slider]
        outputs = [scene_image, status, ground_gallery, method_gallery]
        demo.load(render, inputs=inputs, outputs=outputs)
        scene_slider.release(render, inputs=inputs, outputs=outputs)
        method_radio.change(render, inputs=inputs, outputs=outputs)
        rank_slider.release(render, inputs=inputs, outputs=outputs)
        previous_button.click(
            lambda scene, method, rank: move_page(
                -rows_per_page, scene, method, rank
            ),
            inputs=inputs,
            outputs=[rank_slider, *outputs],
        )
        next_button.click(
            lambda scene, method, rank: move_page(
                rows_per_page, scene, method, rank
            ),
            inputs=inputs,
            outputs=[rank_slider, *outputs],
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=_path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--repo-root", type=_path, default=REPO_ROOT)
    parser.add_argument(
        "--judge-dir",
        type=_path,
        help="optional separate directory containing pairwise ratings or legacy judgments",
    )
    parser.add_argument(
        "--data-archive",
        type=_path,
        default=REPO_ROOT / "data" / "data.zip",
        help="optional data ZIP used when images are not unpacked locally",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="number of ranks available in both columns (default: 100)",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=10,
        help="number of ranks rendered per column (default: 10)",
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1 or args.top_k > 100:
        raise ValueError("--top-k must be between 1 and 100")
    if args.rows_per_page < 1 or args.rows_per_page > args.top_k:
        raise ValueError("--rows-per-page must be between 1 and --top-k")
    data = load_benchmark(args.run_dir, args.repo_root, args.data_archive, args.judge_dir)
    available = min(
        len(rows)
        for method_scenes in data.rankings.values()
        for rows in method_scenes.values()
    )
    if args.top_k > available:
        raise ValueError(
            f"--top-k is {args.top_k}, but the shortest ranking has "
            f"only {available} outfits"
        )
    app = build_app(data, args.top_k, args.rows_per_page)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
