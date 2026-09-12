"""Run the VLM-judged SceneFit retrieval benchmark."""

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
    validate_judgments,
    write_json,
)
from app.services.benchmark.gemini_judge import (
    DEFAULT_JUDGE_MODEL,
    PROMPT_VERSION,
    GeminiJudge,
)
from app.services.benchmark.metrics import evaluate_benchmark, evaluate_pairwise_benchmark
from app.services.benchmark.pairwise import (
    DEFAULT_PAIRS_PER_REQUEST,
    DEFAULT_ROUNDS,
    PROTOCOL as PAIRWISE_PROTOCOL,
    run_pairwise_judging,
    validate_pairwise_judge,
)
from app.services.benchmark.pairwise_judge import PairwiseGeminiJudge


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

    judge_parser = subparsers.add_parser("judge", help="create or resume VLM relevance judgments")
    judge_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    judge_parser.add_argument("--output", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    judge_parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    judge_parser.add_argument("--batch-size", type=int, default=10)
    judge_parser.add_argument("--max-attempts", type=int, default=3)
    judge_parser.add_argument("--scene-index", type=int)
    judge_parser.add_argument("--judge-dir", type=_path, help="directory for pairwise judge artifacts")
    judge_parser.add_argument("--protocol", choices=[PAIRWISE_PROTOCOL, "absolute-1to5"], default=PAIRWISE_PROTOCOL)
    judge_parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    judge_parser.add_argument("--pairs-per-request", type=int, default=DEFAULT_PAIRS_PER_REQUEST)

    pilot_parser = subparsers.add_parser(
        "pilot", help="score one pair without writing benchmark judgments"
    )
    pilot_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    pilot_parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    pilot_parser.add_argument("--scene-index", type=int, default=0)
    pilot_parser.add_argument("--outfit-index", type=int, default=0)
    pilot_parser.add_argument("--max-attempts", type=int, default=3)
    pilot_parser.add_argument("--protocol", choices=[PAIRWISE_PROTOCOL, "absolute-1to5"], default=PAIRWISE_PROTOCOL)

    validation_parser = subparsers.add_parser(
        "validate-judgments",
        help="require a complete compatible relevance matrix",
    )
    validation_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    validation_parser.add_argument("--judgments", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    validation_parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    validation_parser.add_argument("--batch-size", type=int, default=10)

    pairwise_validation_parser = subparsers.add_parser(
        "validate-judge", help="require a complete compatible pairwise judge run"
    )
    pairwise_validation_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    pairwise_validation_parser.add_argument("--judge-dir", type=_path, default=DEFAULT_RUN_DIR)
    pairwise_validation_parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    pairwise_validation_parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    pairwise_validation_parser.add_argument("--pairs-per-request", type=int, default=DEFAULT_PAIRS_PER_REQUEST)

    collect_parser = subparsers.add_parser("collect", help="collect or resume live worker rankings")
    collect_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    collect_parser.add_argument("--judgments", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    collect_parser.add_argument("--output-dir", type=_path, default=DEFAULT_RUN_DIR / "rankings")
    collect_parser.add_argument("--config", type=_path, default=REPO_ROOT / "config" / "retrieval_methods.yaml")
    collect_parser.add_argument("--methods", nargs="+", default=None)
    collect_parser.add_argument("--scene-index", type=int)
    collect_parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    collect_parser.add_argument("--batch-size", type=int, default=10)
    collect_parser.add_argument("--protocol", choices=[PAIRWISE_PROTOCOL, "absolute-1to5"], default=PAIRWISE_PROTOCOL)
    collect_parser.add_argument("--judge-dir", type=_path, default=DEFAULT_RUN_DIR)
    collect_parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    collect_parser.add_argument("--pairs-per-request", type=int, default=DEFAULT_PAIRS_PER_REQUEST)

    scene_parser = subparsers.add_parser(
        "scene", help="judge then collect one scene as a resumable checkpoint"
    )
    scene_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    scene_parser.add_argument("--scene-index", type=int, required=True)
    scene_parser.add_argument("--judgments", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    scene_parser.add_argument("--rankings-dir", type=_path, default=DEFAULT_RUN_DIR / "rankings")
    scene_parser.add_argument("--config", type=_path, default=REPO_ROOT / "config" / "retrieval_methods.yaml")
    scene_parser.add_argument("--methods", nargs="+", default=None)
    scene_parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    scene_parser.add_argument("--batch-size", type=int, default=10)
    scene_parser.add_argument("--max-attempts", type=int, default=3)
    scene_parser.add_argument("--protocol", choices=[PAIRWISE_PROTOCOL, "absolute-1to5"], default=PAIRWISE_PROTOCOL)
    scene_parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    scene_parser.add_argument("--pairs-per-request", type=int, default=DEFAULT_PAIRS_PER_REQUEST)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate cached rankings")
    evaluate_parser.add_argument("--manifest", type=_path, default=DEFAULT_RUN_DIR / "manifest.json")
    evaluate_parser.add_argument("--judgments", type=_path, default=DEFAULT_RUN_DIR / "judgments.jsonl")
    evaluate_parser.add_argument("--rankings-dir", type=_path, default=DEFAULT_RUN_DIR / "rankings")
    evaluate_parser.add_argument("--output-dir", type=_path, default=DEFAULT_RUN_DIR)
    evaluate_parser.add_argument("--protocol", choices=[PAIRWISE_PROTOCOL, "absolute-1to5"], default=PAIRWISE_PROTOCOL)
    evaluate_parser.add_argument("--judge-dir", type=_path, default=DEFAULT_RUN_DIR)

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
    run_parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    run_parser.add_argument("--batch-size", type=int, default=10)
    run_parser.add_argument("--max-attempts", type=int, default=3)
    run_parser.add_argument("--config", type=_path, default=REPO_ROOT / "config" / "retrieval_methods.yaml")
    run_parser.add_argument("--methods", nargs="+", default=None)
    run_parser.add_argument("--protocol", choices=[PAIRWISE_PROTOCOL, "absolute-1to5"], default=PAIRWISE_PROTOCOL)
    run_parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    run_parser.add_argument("--pairs-per-request", type=int, default=DEFAULT_PAIRS_PER_REQUEST)
    return parser


def _scene_id(manifest: dict, scene_index: int) -> str:
    if scene_index < 0 or scene_index >= len(manifest["scenes"]):
        raise ValueError(
            f"scene-index must be between 0 and {len(manifest['scenes']) - 1}"
        )
    return str(manifest["scenes"][scene_index]["id"])


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "manifest":
        manifest = _create_manifest(args, args.output)
        print(f"Wrote {len(manifest['scenes'])} scenes and {len(manifest['outfits'])} outfits to {args.output}")
        return 0

    if args.command == "judge":
        manifest = load_manifest(args.manifest)
        scene_ids = (
            [_scene_id(manifest, args.scene_index)]
            if args.scene_index is not None
            else None
        )
        if args.protocol == PAIRWISE_PROTOCOL:
            judge = PairwiseGeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
            judge_dir = args.judge_dir or args.output.parent
            count = run_pairwise_judging(
                manifest, REPO_ROOT, judge_dir, judge,
                args.rounds, args.pairs_per_request, scene_ids,
            )
            print(f"Appended {count} pairwise comparisons to {judge_dir}")
        else:
            judge = GeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
            count = run_judging(
                manifest, REPO_ROOT, args.output, judge, args.batch_size, scene_ids
            )
            print(f"Appended {count} judgments to {args.output}")
        return 0

    if args.command == "pilot":
        manifest = load_manifest(args.manifest)
        scene_id = _scene_id(manifest, args.scene_index)
        if args.outfit_index < 0 or args.outfit_index >= len(manifest["outfits"]):
            raise ValueError(
                f"outfit-index must be between 0 and {len(manifest['outfits']) - 1}"
            )
        scene = manifest["scenes"][args.scene_index]
        outfit = manifest["outfits"][args.outfit_index]
        if args.protocol == PAIRWISE_PROTOCOL:
            other = manifest["outfits"][(args.outfit_index + 1) % len(manifest["outfits"])]
            judge = PairwiseGeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
            result = judge.compare_batch(
                REPO_ROOT / scene["path"],
                [("pilot", REPO_ROOT / outfit["path"], REPO_ROOT / other["path"])],
            )
            print(json.dumps({"scene_id": scene_id, "comparison": result[0]}, indent=2))
        else:
            judge = GeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
            result = judge.score_batch(
                REPO_ROOT / scene["path"],
                [(outfit["id"], REPO_ROOT / outfit["path"])],
            )
            print(json.dumps({"scene_id": scene_id, "judgment": result[0]}, indent=2))
        return 0

    if args.command == "validate-judgments":
        manifest = load_manifest(args.manifest)
        count = validate_judgments(
            manifest,
            args.judgments,
            args.model,
            PROMPT_VERSION,
            args.batch_size,
        )
        print(f"Judgment matrix complete: {count} pairs")
        return 0

    if args.command == "validate-judge":
        manifest = load_manifest(args.manifest)
        count = validate_pairwise_judge(
            manifest, args.judge_dir, args.model, args.rounds, args.pairs_per_request
        )
        print(f"Pairwise judge complete: {count} comparisons")
        return 0

    if args.command == "collect":
        manifest = load_manifest(args.manifest)
        if args.protocol == PAIRWISE_PROTOCOL:
            validate_pairwise_judge(
                manifest, args.judge_dir, args.model, args.rounds, args.pairs_per_request
            )
        else:
            validate_judgments(
                manifest, args.judgments, args.model, PROMPT_VERSION, args.batch_size
            )
        scene_ids = (
            [_scene_id(manifest, args.scene_index)]
            if args.scene_index is not None
            else None
        )
        paths = collect_rankings(
            manifest,
            REPO_ROOT,
            args.config,
            args.output_dir,
            args.methods,
            scene_ids=scene_ids,
        )
        print(f"Rankings ready: {', '.join(str(path) for path in paths)}")
        return 0

    if args.command == "scene":
        manifest = load_manifest(args.manifest)
        scene_id = _scene_id(manifest, args.scene_index)
        if args.protocol == PAIRWISE_PROTOCOL:
            judge = PairwiseGeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
            count = run_pairwise_judging(
                manifest, REPO_ROOT, args.judgments.parent, judge,
                args.rounds, args.pairs_per_request, [scene_id],
            )
            validate_pairwise_judge(
                manifest, args.judgments.parent, args.model, args.rounds, args.pairs_per_request
            )
        else:
            judge = GeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
            count = run_judging(
                manifest, REPO_ROOT, args.judgments, judge, args.batch_size, scene_ids=[scene_id]
            )
            validate_judgments(
                manifest, args.judgments, args.model, PROMPT_VERSION, args.batch_size
            )
        collect_rankings(
            manifest,
            REPO_ROOT,
            args.config,
            args.rankings_dir,
            args.methods,
            scene_ids=[scene_id],
        )
        print(
            f"Scene {args.scene_index + 1}/{len(manifest['scenes'])} complete: "
            f"{scene_id} ({count} new judgments)"
        )
        return 0

    if args.command == "evaluate":
        manifest = load_manifest(args.manifest)
        if args.protocol == PAIRWISE_PROTOCOL:
            result = evaluate_pairwise_benchmark(
                manifest,
                args.judge_dir / "comparisons.jsonl",
                args.judge_dir / "ratings.json",
                args.rankings_dir,
                args.output_dir,
            )
        else:
            result = evaluate_benchmark(
                manifest, args.judgments, args.rankings_dir, args.output_dir
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
    if args.protocol == PAIRWISE_PROTOCOL:
        judge = PairwiseGeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
        run_pairwise_judging(
            manifest, REPO_ROOT, run_dir, judge, args.rounds, args.pairs_per_request
        )
        validate_pairwise_judge(
            manifest, run_dir, args.model, args.rounds, args.pairs_per_request
        )
    else:
        judge = GeminiJudge(model_name=args.model, max_attempts=args.max_attempts)
        run_judging(manifest, REPO_ROOT, run_dir / "judgments.jsonl", judge, args.batch_size)
        validate_judgments(
            manifest, run_dir / "judgments.jsonl", args.model, PROMPT_VERSION, args.batch_size
        )
    collect_rankings(manifest, REPO_ROOT, args.config, run_dir / "rankings", args.methods)
    result = (
        evaluate_pairwise_benchmark(
            manifest, run_dir / "comparisons.jsonl", run_dir / "ratings.json", run_dir / "rankings", run_dir
        )
        if args.protocol == PAIRWISE_PROTOCOL
        else evaluate_benchmark(manifest, run_dir / "judgments.jsonl", run_dir / "rankings", run_dir)
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
