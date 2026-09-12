"""Run resumable benchmark stages directly inside a GPU runtime or Colab."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.benchmark.gemini_judge import DEFAULT_JUDGE_MODEL
from app.services.benchmark.pairwise import (
    DEFAULT_PAIRS_PER_REQUEST,
    DEFAULT_ROUNDS,
    PROTOCOL as PAIRWISE_PROTOCOL,
)


DEFAULT_RUN_DIR = REPO_ROOT / "results" / "benchmark" / "latest"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-dir", type=_path, required=True)
    parser.add_argument("--run-dir", type=_path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--num-outfits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--protocol", choices=[PAIRWISE_PROTOCOL, "absolute-1to5"], default=PAIRWISE_PROTOCOL)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--pairs-per-request", type=int, default=DEFAULT_PAIRS_PER_REQUEST)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=["clip", "aesthetic"])
    parser.add_argument(
        "--config",
        type=_path,
        default=REPO_ROOT / "config" / "retrieval_methods.yaml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "judge", "retrieve"):
        _add_common_arguments(subparsers.add_parser(stage))
    return parser


def _run(command: list[str]) -> None:
    print(f"[benchmark] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _archive_path(checkpoint_dir: Path, name: str) -> Path:
    return checkpoint_dir / f"{name}.zip"


def _load_state(archive: Path) -> dict[str, Any]:
    if not archive.is_file():
        return {}
    try:
        with zipfile.ZipFile(archive) as stream:
            return json.loads(stream.read("checkpoint.json"))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return {}


def _restore(checkpoint_dir: Path, run_dir: Path) -> None:
    latest = _archive_path(checkpoint_dir, "latest")
    if latest.is_file() and not (run_dir / "manifest.json").is_file():
        print(f"[benchmark] Restoring checkpoint: {latest}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(latest, run_dir)


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_scenes": args.num_scenes,
        "num_outfits": args.num_outfits,
        "seed": args.seed,
        "model": args.model,
        "batch_size": args.batch_size,
        "protocol": args.protocol,
        "rounds": args.rounds,
        "pairs_per_request": args.pairs_per_request,
        "methods": args.methods,
    }


def _validate_state(state: dict[str, Any], configuration: dict[str, Any]) -> None:
    if not state:
        return
    if state.get("version") != 3:
        raise ValueError(
            "latest.zip uses an unsupported checkpoint format; choose a fresh "
            "--checkpoint-dir"
        )
    if state.get("configuration") != configuration:
        raise ValueError(
            "latest.zip belongs to a different benchmark configuration; choose "
            "a fresh --checkpoint-dir or restore the original arguments"
        )


def _state(configuration: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 3,
        "configuration": configuration,
        "pilot_completed": bool(previous.get("pilot_completed", False)),
        "judged_scene_indices": sorted(
            {int(index) for index in previous.get("judged_scene_indices", [])}
        ),
        "retrieved_scene_indices": sorted(
            {int(index) for index in previous.get("retrieved_scene_indices", [])}
        ),
    }


def _save_checkpoint(
    run_dir: Path,
    checkpoint_dir: Path,
    name: str,
    state: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    destination = _archive_path(checkpoint_dir, name)
    temporary = destination.with_suffix(".tmp.zip")
    temporary.unlink(missing_ok=True)
    shutil.make_archive(str(temporary.with_suffix("")), "zip", root_dir=run_dir)
    temporary.replace(destination)
    shutil.copy2(destination, _archive_path(checkpoint_dir, "latest"))
    print(f"[benchmark] Saved checkpoint: {destination}", flush=True)


def _manifest_path(run_dir: Path) -> Path:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Run the prepare stage before this stage")
    return path


def _prepare(args: argparse.Namespace, state: dict[str, Any], configuration: dict[str, Any]) -> None:
    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.is_file():
        _run(
            [
                "python",
                "scripts/benchmark.py",
                "manifest",
                "--output",
                str(manifest_path),
                "--num-scenes",
                str(args.num_scenes),
                "--num-outfits",
                str(args.num_outfits),
                "--seed",
                str(args.seed),
            ]
        )
    _save_checkpoint(args.run_dir, args.checkpoint_dir, "prepared", _state(configuration, state))


def _judge(args: argparse.Namespace, state: dict[str, Any], configuration: dict[str, Any]) -> None:
    manifest_path = _manifest_path(args.run_dir)
    current = _state(configuration, state)
    if not current["pilot_completed"]:
        _run(
            [
                "python",
                "scripts/benchmark.py",
                "pilot",
                "--manifest",
                str(manifest_path),
                "--model",
                args.model,
                "--max-attempts",
                str(args.max_attempts),
                "--protocol",
                args.protocol,
            ]
        )
        current["pilot_completed"] = True
        _save_checkpoint(args.run_dir, args.checkpoint_dir, "pilot", current)

    completed = set(current["judged_scene_indices"])
    for scene_index in range(args.num_scenes):
        if scene_index in completed:
            print(f"[benchmark] Judgments already contain scene {scene_index + 1}; skipping")
            continue
        _run(
            [
                "python",
                "scripts/benchmark.py",
                "judge",
                "--manifest",
                str(manifest_path),
                "--scene-index",
                str(scene_index),
                "--model",
                args.model,
                "--batch-size",
                str(args.batch_size),
                "--max-attempts",
                str(args.max_attempts),
                "--protocol",
                args.protocol,
                "--rounds",
                str(args.rounds),
                "--pairs-per-request",
                str(args.pairs_per_request),
            ]
        )
        completed.add(scene_index)
        current["judged_scene_indices"] = sorted(completed)
        _save_checkpoint(
            args.run_dir,
            args.checkpoint_dir,
            f"judge-scene-{scene_index + 1:03d}",
            current,
        )

    validation = ["python", "scripts/benchmark.py"]
    if args.protocol == PAIRWISE_PROTOCOL:
        validation.extend([
            "validate-judge", "--manifest", str(manifest_path), "--judge-dir", str(args.run_dir),
            "--model", args.model, "--rounds", str(args.rounds),
            "--pairs-per-request", str(args.pairs_per_request),
        ])
    else:
        validation.extend([
            "validate-judgments", "--manifest", str(manifest_path), "--model", args.model,
            "--batch-size", str(args.batch_size),
        ])
    _run(validation)


def _verify_image_edit_assets() -> None:
    missing = [
        path
        for path in (REPO_ROOT / "data" / "ref_images" / "man.png", REPO_ROOT / "data" / "ref_images" / "woman.png")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "ImageEdit requires reference images: " + ", ".join(map(str, missing))
        )


def _retrieve(args: argparse.Namespace, state: dict[str, Any], configuration: dict[str, Any]) -> None:
    manifest_path = _manifest_path(args.run_dir)
    validation = ["python", "scripts/benchmark.py"]
    if args.protocol == PAIRWISE_PROTOCOL:
        validation.extend([
            "validate-judge", "--manifest", str(manifest_path), "--judge-dir", str(args.run_dir),
            "--model", args.model, "--rounds", str(args.rounds),
            "--pairs-per-request", str(args.pairs_per_request),
        ])
    else:
        validation.extend([
            "validate-judgments", "--manifest", str(manifest_path), "--model", args.model,
            "--batch-size", str(args.batch_size),
        ])
    _run(validation)
    if "image_edit" in args.methods:
        _verify_image_edit_assets()
    if "vlm" in args.methods:
        _run(
            [
                "python",
                "-m",
                "scripts.build_pe_index",
                "--manifest",
                str(manifest_path),
            ]
        )

    current = _state(configuration, state)
    completed = set(current["retrieved_scene_indices"])
    for scene_index in range(args.num_scenes):
        if scene_index in completed:
            print(f"[benchmark] Rankings already contain scene {scene_index + 1}; skipping")
            continue
        _run(
            [
                "python",
                "scripts/benchmark.py",
                "collect",
                "--manifest",
                str(manifest_path),
                "--judgments",
                str(args.run_dir / "judgments.jsonl"),
                "--output-dir",
                str(args.run_dir / "rankings"),
                "--config",
                str(args.config),
                "--scene-index",
                str(scene_index),
                "--model",
                args.model,
                "--batch-size",
                str(args.batch_size),
                "--protocol",
                args.protocol,
                "--judge-dir",
                str(args.run_dir),
                "--rounds",
                str(args.rounds),
                "--pairs-per-request",
                str(args.pairs_per_request),
                "--methods",
                *args.methods,
            ]
        )
        completed.add(scene_index)
        current["retrieved_scene_indices"] = sorted(completed)
        _save_checkpoint(
            args.run_dir,
            args.checkpoint_dir,
            f"scene-{scene_index + 1:03d}",
            current,
        )

    _run(
        [
            "python",
            "scripts/benchmark.py",
            "evaluate",
            "--manifest",
            str(manifest_path),
            "--judgments",
            str(args.run_dir / "judgments.jsonl"),
            "--rankings-dir",
            str(args.run_dir / "rankings"),
            "--output-dir",
            str(args.run_dir),
            "--protocol",
            args.protocol,
            "--judge-dir",
            str(args.run_dir),
        ]
    )
    _save_checkpoint(args.run_dir, args.checkpoint_dir, "final", current)


def main() -> int:
    args = build_parser().parse_args()
    if args.num_scenes < 1 or args.num_outfits < 1 or args.batch_size < 1:
        raise ValueError("num-scenes, num-outfits, and batch-size must be positive")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _restore(args.checkpoint_dir, args.run_dir)
    configuration = _configuration(args)
    state = _load_state(_archive_path(args.checkpoint_dir, "latest"))
    _validate_state(state, configuration)

    if args.stage == "prepare":
        _prepare(args, state, configuration)
    elif args.stage == "judge":
        _judge(args, state, configuration)
    else:
        _retrieve(args, state, configuration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
