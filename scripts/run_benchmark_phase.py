"""Run a resumable benchmark phase on the current machine.

This runner is intended for the heavy Vast.ai phase. It restores the completed
light-phase archive, preserves its manifest and judgments, and collects methods
in method-major order to avoid keeping unrelated GPU models resident.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "benchmark" / "latest"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "benchmark" / "phase-checkpoints"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.benchmark.core import (
    load_manifest,
    manifest_fingerprint,
    verify_manifest_files,
    write_json,
)


def _path(value: str) -> Path:
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--run-dir", type=_path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume-archive", type=_path)
    parser.add_argument(
        "--config",
        type=_path,
        default=REPO_ROOT / "config" / "retrieval_methods.yaml",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def _restore_archive(archive: Path, run_dir: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"Resume archive does not exist: {archive}")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(run_dir))


def _verify_complete_judgments(manifest: dict, judgments_path: Path) -> None:
    if not judgments_path.is_file():
        raise FileNotFoundError(
            "The previous phase did not provide judgments.jsonl; finish the T4 phase first"
        )
    expected = {
        (scene["id"], outfit["id"])
        for scene in manifest["scenes"]
        for outfit in manifest["outfits"]
    }
    actual: set[tuple[str, str]] = set()
    with judgments_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair = (row["scene_id"], row["outfit_id"])
            if pair in actual:
                raise ValueError(f"Duplicate judgment at line {line_number}")
            actual.add(pair)
    if actual != expected:
        raise ValueError(
            "Judgments are incompatible or incomplete: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )


def _prepare(run_dir: Path, methods: list[str], timeout: int) -> dict:
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    verify_manifest_files(manifest, REPO_ROOT)
    _verify_complete_judgments(manifest, run_dir / "judgments.jsonl")
    if "vlm" in methods:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.build_pe_index",
                "--manifest",
                str(manifest_path),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=timeout,
        )
    return manifest


def _archive(run_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.make_archive(
        str(destination.with_suffix("")),
        "zip",
        root_dir=run_dir,
    )


def _precompute_image_edit_prompts(
    manifest: dict,
    run_dir: Path,
    config_path: Path,
    timeout: int,
) -> None:
    import httpx

    fingerprint = manifest_fingerprint(manifest)
    output_path = run_dir / "image_edit_prompts.json"
    payload = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.is_file()
        else {"version": 1, "manifest_fingerprint": fingerprint, "scenes": {}}
    )
    if payload.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Cached ImageEdit prompts use another manifest")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    worker = config["retrieval_methods"]["image_edit"]
    base_url = str(worker["url"]).rstrip("/")
    endpoint = str(
        worker.get("suggestion_endpoint", "api/v1/workers/vlm-suggest-outfit")
    ).lstrip("/")
    url = f"{base_url}/{endpoint}"

    for scene in manifest["scenes"]:
        if payload["scenes"].get(scene["id"]):
            continue
        scene_path = REPO_ROOT / scene["path"]
        with scene_path.open("rb") as image_stream:
            response = httpx.post(
                url,
                files={
                    "bg_image": (
                        scene_path.name,
                        image_stream,
                        "application/octet-stream",
                    )
                },
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=timeout,
            )
        response.raise_for_status()
        outfit_description = response.json().get("outfit_description")
        if not outfit_description:
            raise ValueError(f"Suggestion worker returned no prompt for {scene['id']}")
        payload["scenes"][scene["id"]] = outfit_description
        write_json(output_path, payload)
        print(f"Cached ImageEdit prompt: {scene['id']}")


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout < 1:
        raise ValueError("timeout must be positive")
    if args.resume_archive:
        _restore_archive(args.resume_archive, args.run_dir)

    manifest = _prepare(args.run_dir, args.methods, args.timeout)
    print(
        f"Phase ready: {len(manifest['scenes'])} scenes, "
        f"{len(manifest['outfits'])} outfits, methods={args.methods}"
    )
    if args.prepare_only:
        return 0

    checkpoint_path = args.run_dir / "phase-checkpoint.json"
    expected_configuration = {
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "methods": args.methods,
    }
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {"configuration": expected_configuration, "completed": {}}
    )
    if checkpoint.get("configuration") != expected_configuration:
        raise ValueError(
            "Existing phase checkpoint uses another manifest or method order; "
            "choose a different --run-dir"
        )
    completed = {
        method: {int(index) for index in indices}
        for method, indices in checkpoint.get("completed", {}).items()
    }

    for method in args.methods:
        if method == "image_edit":
            _precompute_image_edit_prompts(
                manifest,
                args.run_dir,
                args.config,
                args.timeout,
            )
        method_completed = completed.setdefault(method, set())
        for scene_index in range(len(manifest["scenes"])):
            if scene_index in method_completed:
                print(f"{method}: scene {scene_index + 1} already complete; skipping")
                continue
            subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark.py",
                    "scene",
                    "--scene-index",
                    str(scene_index),
                    "--methods",
                    method,
                    "--skip-judge",
                    "--config",
                    str(args.config),
                    "--manifest",
                    str(args.run_dir / "manifest.json"),
                    "--judgments",
                    str(args.run_dir / "judgments.jsonl"),
                    "--rankings-dir",
                    str(args.run_dir / "rankings"),
                ],
                cwd=REPO_ROOT,
                check=True,
                timeout=args.timeout,
            )
            method_completed.add(scene_index)
            serializable = {
                "configuration": expected_configuration,
                "completed": {
                    name: sorted(indices) for name, indices in completed.items()
                },
            }
            write_json(checkpoint_path, serializable)
            checkpoint_archive = (
                args.output_dir / f"{method}-scene-{scene_index + 1:03d}.zip"
            )
            _archive(args.run_dir, checkpoint_archive)
            shutil.copy2(checkpoint_archive, args.output_dir / "latest.zip")
            print(f"Saved checkpoint: {checkpoint_archive}")

    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "evaluate",
            "--manifest",
            str(args.run_dir / "manifest.json"),
            "--judgments",
            str(args.run_dir / "judgments.jsonl"),
            "--rankings-dir",
            str(args.run_dir / "rankings"),
            "--output-dir",
            str(args.run_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        timeout=args.timeout,
    )
    final_archive = args.output_dir / "final.zip"
    _archive(args.run_dir, final_archive)
    shutil.copy2(final_archive, args.output_dir / "latest.zip")
    print(f"Phase complete: {final_archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
