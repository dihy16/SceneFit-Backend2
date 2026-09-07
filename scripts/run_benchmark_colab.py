"""Run the benchmark scene-by-scene on Colab and download every checkpoint.

Run this script locally from Linux, macOS, or WSL after preparing a named
Google Colab CLI session as described in docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/content")
REMOTE_RUN_DIR = REMOTE_ROOT / "results" / "benchmark" / "latest"
REMOTE_ARCHIVE = REMOTE_ROOT / "benchmark-checkpoint.zip"
REMOTE_STATUS = REMOTE_ROOT / ".scenefit-benchmark-command-status.json"


def _run_colab(
    arguments: list[str],
    *,
    code: str | None = None,
    timeout: int | None = None,
) -> None:
    command = ["colab", *arguments]
    subprocess.run(command, input=code, text=True, check=True, timeout=timeout)


def _remote_exec(session: str, code: str, timeout: int) -> None:
    wrapped_code = (
        "from pathlib import Path\n"
        "import json, traceback\n"
        f"status_path = Path({str(REMOTE_STATUS)!r})\n"
        "status_path.unlink(missing_ok=True)\n"
        "try:\n"
        f"    exec({code!r}, globals(), globals())\n"
        "except BaseException:\n"
        "    status_path.write_text(json.dumps({'ok': False, 'error': traceback.format_exc()}), encoding='utf-8')\n"
        "    raise\n"
        "else:\n"
        "    status_path.write_text(json.dumps({'ok': True}), encoding='utf-8')\n"
    )
    finished = threading.Event()

    def report_heartbeat() -> None:
        while not finished.wait(30):
            print("[benchmark] Remote Colab command is still running...", flush=True)

    print("[benchmark] Starting remote Colab command...", flush=True)
    heartbeat = threading.Thread(target=report_heartbeat, daemon=True)
    heartbeat.start()
    try:
        _run_colab(
            ["exec", "-s", session, "--timeout", str(timeout)],
            code=wrapped_code,
            timeout=timeout + 60,
        )
    finally:
        finished.set()
        heartbeat.join()
    with tempfile.TemporaryDirectory() as directory:
        status_file = Path(directory) / "remote-status.json"
        _run_colab(["download", "-s", session, str(REMOTE_STATUS), str(status_file)])
        status = json.loads(status_file.read_text(encoding="utf-8"))
    if not status.get("ok"):
        raise RuntimeError(f"Remote Colab command failed:\n{status.get('error', 'unknown error')}")


def _remote_command(session: str, command: list[str], timeout: int) -> None:
    code = (
        "import subprocess\n"
        f"result = subprocess.run({json.dumps(command)}, cwd={str(REMOTE_ROOT)!r}, "
        "text=True, capture_output=True)\n"
        "print(result.stdout, end='')\n"
        "if result.returncode:\n"
        "    raise RuntimeError(\n"
        "        'Remote process failed with exit code ' + str(result.returncode) + "
        "        ':\\n' + result.stdout + result.stderr\n"
        "    )\n"
    )
    _remote_exec(session, code, timeout)


def _checkpoint_state(archive: Path) -> dict:
    if not archive.exists():
        return {}
    try:
        with zipfile.ZipFile(archive) as stream:
            return json.loads(stream.read("checkpoint.json"))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return {}


def _download_checkpoint(session: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    _run_colab(["download", "-s", session, str(REMOTE_ARCHIVE), str(partial)])
    partial.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="scenefit-benchmark")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--num-outfits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare or restore the manifest, then exit before calling workers",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed per scene")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "benchmark" / "colab-checkpoints",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_scenes < 1 or args.num_outfits < 1:
        raise ValueError("num-scenes and num-outfits must be positive")

    output_dir = args.output_dir.resolve()
    latest_archive = output_dir / "latest.zip"
    output_dir.mkdir(parents=True, exist_ok=True)

    if latest_archive.exists():
        print(f"[benchmark] Restoring checkpoint: {latest_archive}", flush=True)
        remote_resume = REMOTE_ROOT / "benchmark-resume.zip"
        _run_colab(["upload", "-s", args.session, str(latest_archive), str(remote_resume)])
        _remote_exec(
            args.session,
            "from pathlib import Path\n"
            "import shutil\n"
            f"target = Path({str(REMOTE_RUN_DIR)!r})\n"
            f"archive = Path({str(remote_resume)!r})\n"
            "if archive.exists() and not (target / 'manifest.json').exists():\n"
            "    target.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.unpack_archive(archive, target)\n",
            300,
        )

    manifest_command = [
        "python",
        "scripts/benchmark.py",
        "manifest",
        "--num-scenes",
        str(args.num_scenes),
        "--num-outfits",
        str(args.num_outfits),
        "--seed",
        str(args.seed),
    ]
    _remote_exec(
        args.session,
        "from pathlib import Path\n"
        "import json, subprocess\n"
        f"manifest_path = Path({str(REMOTE_RUN_DIR / 'manifest.json')!r})\n"
        f"command = {json.dumps(manifest_command)}\n"
        "if not manifest_path.exists():\n"
        f"    result = subprocess.run(command, cwd={str(REMOTE_ROOT)!r}, text=True, capture_output=True)\n"
        "    if result.returncode:\n"
        "        raise RuntimeError('Manifest creation failed:\\n' + result.stdout + result.stderr)\n"
        "    print(result.stdout, end='')\n"
        "manifest = json.loads(manifest_path.read_text(encoding='utf-8'))\n"
        f"assert len(manifest['scenes']) == {args.num_scenes}, 'Existing manifest has a different scene count'\n"
        f"assert len(manifest['outfits']) == {args.num_outfits}, 'Existing manifest has a different outfit count'\n"
        f"assert manifest['seed'] == {args.seed}, 'Existing manifest has a different seed'\n",
        300,
    )
    print(
        f"[benchmark] Manifest ready: {args.num_scenes} scenes, "
        f"{args.num_outfits} outfits",
        flush=True,
    )

    if args.methods is None or "vlm" in args.methods:
        _remote_command(
            args.session,
            [
                "python",
                "-m",
                "scripts.build_pe_index",
                "--manifest",
                str(REMOTE_RUN_DIR / "manifest.json"),
            ],
            args.timeout,
        )

    checkpoint = _checkpoint_state(latest_archive)
    expected_configuration = {
        "num_scenes": args.num_scenes,
        "num_outfits": args.num_outfits,
        "seed": args.seed,
        "model": args.model,
        "batch_size": args.batch_size,
        "methods": args.methods,
    }
    if checkpoint and checkpoint.get("configuration") != expected_configuration:
        raise ValueError(
            "latest.zip belongs to a different benchmark configuration; "
            "use another --output-dir or restore the original arguments"
        )
    completed = {int(index) for index in checkpoint.get("completed_scene_indices", [])}
    if args.prepare_only:
        print(
            "Benchmark manifest is ready. Start Uvicorn with "
            "BENCHMARK_MANIFEST=/content/results/benchmark/latest/manifest.json"
        )
        return 0

    for scene_index in range(args.num_scenes):
        if scene_index in completed:
            print(f"Checkpoint already contains scene {scene_index + 1}; skipping")
            continue
        print(
            f"[benchmark] Running scene {scene_index + 1}/{args.num_scenes} "
            f"with methods: {', '.join(args.methods or ['all configured'])}",
            flush=True,
        )
        command = [
            "python",
            "scripts/benchmark.py",
            "scene",
            "--scene-index",
            str(scene_index),
            "--model",
            args.model,
            "--batch-size",
            str(args.batch_size),
            "--max-attempts",
            str(args.max_attempts),
        ]
        if args.methods:
            command.extend(["--methods", *args.methods])
        state = {
            "completed_scene_indices": sorted(completed | {scene_index}),
            "configuration": expected_configuration,
        }
        code = (
            "from pathlib import Path\n"
            "import json, shutil, subprocess\n"
            f"command = {json.dumps(command)}\n"
            f"result = subprocess.run(command, cwd={str(REMOTE_ROOT)!r}, text=True, capture_output=True)\n"
            "print(result.stdout, end='')\n"
            "if result.returncode:\n"
            "    raise RuntimeError(\n"
            "        'Benchmark scene failed with exit code ' + str(result.returncode) + "
            "        ':\\n' + result.stdout + result.stderr\n"
            "    )\n"
            f"run_dir = Path({str(REMOTE_RUN_DIR)!r})\n"
            f"state = {json.dumps(state)}\n"
            "(run_dir / 'checkpoint.json').write_text(json.dumps(state, indent=2) + '\\n', encoding='utf-8')\n"
            f"archive = Path({str(REMOTE_ARCHIVE)!r})\n"
            "archive.unlink(missing_ok=True)\n"
            "shutil.make_archive(str(archive.with_suffix('')), 'zip', root_dir=run_dir)\n"
        )
        _remote_exec(args.session, code, args.timeout)
        scene_archive = output_dir / f"scene-{scene_index + 1:03d}.zip"
        _download_checkpoint(args.session, scene_archive)
        shutil.copy2(scene_archive, latest_archive)
        completed.add(scene_index)
        print(f"Downloaded completed scene {scene_index + 1} to {scene_archive}")

    _remote_command(args.session, ["python", "scripts/benchmark.py", "evaluate"], args.timeout)
    _remote_exec(
        args.session,
        "from pathlib import Path\n"
        "import shutil\n"
        f"archive = Path({str(REMOTE_ARCHIVE)!r})\n"
        "archive.unlink(missing_ok=True)\n"
        f"shutil.make_archive(str(archive.with_suffix('')), 'zip', root_dir={str(REMOTE_RUN_DIR)!r})\n",
        300,
    )
    final_archive = output_dir / "final.zip"
    _download_checkpoint(args.session, final_archive)
    shutil.copy2(final_archive, latest_archive)
    print(f"Benchmark complete; final results: {final_archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
