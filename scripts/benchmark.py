"""Run the Gemini-judged SceneFit retrieval benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from app.services.benchmark.collector import collect_rankings
from app.services.benchmark.core import (
    create_manifest,
    load_manifest,
    manifest_fingerprint,
    run_judging,
    write_json,
)
from app.services.benchmark.gemini_judge import GeminiJudge
from app.services.benchmark.metrics import evaluate_benchmark


DEFAULT_RUN_DIR = REPO_ROOT / "results" / "benchmark" / "latest"
DEFAULT_SMOKE_DIR = REPO_ROOT / "results" / "benchmark" / "smoke"


class SmokeJudge:
    """Deterministic local judge used only to exercise the benchmark pipeline."""

    model_name = "smoke-fixture"
    prompt_version = "smoke-v1"

    def score_batch(self, scene_path: Path, outfits: list[tuple[str, Path]]) -> list[dict]:
        return [
            {
                "outfit_id": outfit_id,
                "score": 1
                + int(hashlib.sha256(f"{scene_path.stem}:{outfit_id}".encode()).hexdigest()[:8], 16) % 5,
                "reason": "Deterministic smoke-test fixture",
            }
            for outfit_id, _ in outfits
        ]


def _path(value: str) -> Path:
    return Path(value).resolve()


def _add_manifest_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenes-dir", type=_path, default=REPO_ROOT / "data" / "bg")
    parser.add_argument("--outfits-dir", type=_path, default=REPO_ROOT / "data" / "2d")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--num-outfits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)


def _create_manifest(args: argparse.Namespace, path: Path) -> dict:
    manifest = create_manifest(
        root=REPO_ROOT,
        scenes_dir=args.scenes_dir,
        outfits_dir=args.outfits_dir,
        num_scenes=args.num_scenes,
        num_outfits=args.num_outfits,
        seed=args.seed,
    )
    write_json(path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="select and fingerprint benchmark data")
    _add_manifest_inputs(manifest_parser)
    manifest_parser.add_argument("--output", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")

    judge_parser = subparsers.add_parser("judge", help="create or resume Gemini relevance judgments")
    judge_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    judge_parser.add_argument("--output", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    judge_parser.add_argument("--model", default="gemini-3.7-flash")
    judge_parser.add_argument("--batch-size", type=int, default=10)
    judge_parser.add_argument("--max-attempts", type=int, default=3)

    collect_parser = subparsers.add_parser("collect", help="collect or resume live worker rankings")
    collect_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    collect_parser.add_argument("--output-dir", type=_path, default=DEFAULT_RUN_DIR / "rankings")
    collect_parser.add_argument("--config", type=_path, default=REPO_ROOT / "config" / "retrieval_methods.yaml")
    collect_parser.add_argument("--methods", nargs="+", default=None)

    scene_parser = subparsers.add_parser(
        "scene", help="collect and judge one scene as a resumable checkpoint"
    )
    scene_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    scene_parser.add_argument("--scene-index", type=int, required=True)
    scene_parser.add_argument("--judgments", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    scene_parser.add_argument("--rankings-dir", type=_path, default=DEFAULT_RUN_DIR / "rankings")
    scene_parser.add_argument("--config", type=_path, default=REPO_ROOT / "config" / "retrieval_methods.yaml")
    scene_parser.add_argument("--methods", nargs="+", default=None)
    scene_parser.add_argument("--model", default="gemini-3.7-flash")
    scene_parser.add_argument("--batch-size", type=int, default=10)
    scene_parser.add_argument("--max-attempts", type=int, default=3)
    scene_parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="collect rankings without creating judgments (for a later benchmark phase)",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate cached rankings")
    evaluate_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    evaluate_parser.add_argument("--judgments", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    evaluate_parser.add_argument("--rankings-dir", type=_path, default=DEFAULT_RUN_DIR / "rankings")
    evaluate_parser.add_argument("--output-dir", type=_path, default=DEFAULT_RUN_DIR)

    smoke_parser = subparsers.add_parser("smoke", help="run a tiny offline end-to-end smoke test")
    smoke_parser.add_argument("--run-dir", type=_path, default=DEFAULT_SMOKE_DIR)
    smoke_parser.add_argument("--scenes-dir", type=_path, default=REPO_ROOT / "data" / "bg")
    smoke_parser.add_argument("--outfits-dir", type=_path, default=REPO_ROOT / "data" / "2d")
    smoke_parser.add_argument("--num-scenes", type=int, default=2)
    smoke_parser.add_argument("--num-outfits", type=int, default=5)
    smoke_parser.add_argument("--seed", type=int, default=42)

    run_parser = subparsers.add_parser("run", help="run manifest, judge, collect, and evaluate")
    _add_manifest_inputs(run_parser)
    run_parser.add_argument("--run-dir", type=_path, default=DEFAULT_RUN_DIR)
    run_parser.add_argument("--model", default="gemini-3.7-flash")
    run_parser.add_argument("--batch-size", type=int, default=10)
    run_parser.add_argument("--max-attempts", type=int, default=3)
    run_parser.add_argument("--config", type=_path, default=REPO_ROOT / "config" / "retrieval_methods.yaml")
    run_parser.add_argument("--methods", nargs="+", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "manifest":
        manifest = _create_manifest(args, args.output)
        print(f"Wrote {len(manifest['scenes'])} scenes and {len(manifest['outfits'])} outfits to {args.output}")
        return 0

    if args.command == "judge":
        manifest = load_manifest(args.manifest)
        judge = GeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
        count = run_judging(manifest, REPO_ROOT, args.output, judge, args.batch_size)
        print(f"Appended {count} judgments to {args.output}")
        return 0

    if args.command == "collect":
        manifest = load_manifest(args.manifest)
        paths = collect_rankings(manifest, REPO_ROOT, args.config, args.output_dir, args.methods)
        print(f"Rankings ready: {', '.join(str(path) for path in paths)}")
        return 0

    if args.command == "scene":
        manifest = load_manifest(args.manifest)
        if args.scene_index < 0 or args.scene_index >= len(manifest["scenes"]):
            raise ValueError(
                f"scene-index must be between 0 and {len(manifest['scenes']) - 1}"
            )
        scene_id = manifest["scenes"][args.scene_index]["id"]
        collect_rankings(
            manifest,
            REPO_ROOT,
            args.config,
            args.rankings_dir,
            args.methods,
            scene_ids=[scene_id],
        )
        count = 0
        if not args.skip_judge:
            judge = GeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
            count = run_judging(
                manifest,
                REPO_ROOT,
                args.judgments,
                judge,
                args.batch_size,
                scene_ids=[scene_id],
            )
        print(
            f"Scene {args.scene_index + 1}/{len(manifest['scenes'])} complete: "
            f"{scene_id} ({count} new judgments)"
        )
        return 0

    if args.command == "evaluate":
        result = evaluate_benchmark(
            load_manifest(args.manifest),
            args.judgments,
            args.rankings_dir,
            args.output_dir,
        )
        print(json.dumps(result["metrics"], indent=2))
        return 0

    if args.command == "smoke":
        run_dir: Path = args.run_dir
        manifest_path = run_dir / "manifest.json"
        manifest = load_manifest(manifest_path) if manifest_path.exists() else _create_manifest(args, manifest_path)
        judgments_path = run_dir / "judgments.jsonl"
        run_judging(manifest, REPO_ROOT, judgments_path, SmokeJudge(), batch_size=args.num_outfits)

        matrix: dict[str, dict[str, int]] = {}
        for line in judgments_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            matrix.setdefault(row["scene_id"], {})[row["outfit_id"]] = int(row["score"])

        rankings_dir = run_dir / "rankings"
        fingerprint = manifest_fingerprint(manifest)
        for method, reverse in (("smoke_ideal", True), ("smoke_reverse", False)):
            scenes = {}
            for scene in manifest["scenes"]:
                ordered = sorted(
                    matrix[scene["id"]],
                    key=lambda outfit_id: (matrix[scene["id"]][outfit_id], outfit_id),
                    reverse=reverse,
                )
                scenes[scene["id"]] = [
                    {"outfit_id": outfit_id, "rank": rank, "raw_score": None}
                    for rank, outfit_id in enumerate(ordered, 1)
                ]
            write_json(
                rankings_dir / f"{method}.json",
                {
                    "version": 1,
                    "method": method,
                    "manifest_fingerprint": fingerprint,
                    "scenes": scenes,
                },
            )
        result = evaluate_benchmark(manifest, judgments_path, rankings_dir, run_dir)
        ideal = result["metrics"]["smoke_ideal"]["ndcg@5"]
        reversed_score = result["metrics"]["smoke_reverse"]["ndcg@5"]
        if not math.isclose(ideal, 1.0) or reversed_score >= ideal:
            raise RuntimeError(
                f"Smoke metric invariant failed: ideal={ideal}, reverse={reversed_score}"
            )
        print(f"Smoke benchmark passed; artifacts: {run_dir}")
        print(json.dumps(result["metrics"], indent=2))
        return 0


    run_dir: Path = args.run_dir
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else _create_manifest(args, manifest_path)
    collect_rankings(manifest, REPO_ROOT, args.config, run_dir / "rankings", args.methods)
    judge = GeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
    run_judging(manifest, REPO_ROOT, run_dir / "judgments.jsonl", judge, args.batch_size)
    result = evaluate_benchmark(
        manifest,
        run_dir / "judgments.jsonl",
        run_dir / "rankings",
        run_dir,
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
