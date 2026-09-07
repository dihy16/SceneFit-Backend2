from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, Sequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MANIFEST_VERSION = 1


class Judge(Protocol):
    model_name: str
    prompt_version: str

    def score_batch(
        self,
        scene_path: Path,
        outfits: Sequence[tuple[str, Path]],
    ) -> list[dict[str, Any]]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def _select(paths: list[Path], count: int, rng: random.Random, label: str) -> list[Path]:
    if count < 1:
        raise ValueError(f"{label} count must be positive")
    if len(paths) < count:
        raise ValueError(f"Requested {count} {label}, but only {len(paths)} are available")
    return rng.sample(paths, count)


def _entries(paths: Sequence[Path], root: Path, label: str) -> list[dict[str, str]]:
    ids = [path.stem for path in paths]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs derived from filenames: {duplicates}")
    return [
        {
            "id": path.stem,
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def create_manifest(
    root: Path,
    scenes_dir: Path,
    outfits_dir: Path,
    num_scenes: int = 10,
    num_outfits: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Select and fingerprint a stable benchmark candidate set."""
    root = root.resolve()
    rng = random.Random(seed)
    scenes = _select(_images(scenes_dir), num_scenes, rng, "scenes")
    outfits = _select(_images(outfits_dir), num_outfits, rng, "outfits")
    return {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "num_scenes": num_scenes,
        "num_outfits": num_outfits,
        "scenes": _entries(scenes, root, "scene"),
        "outfits": _entries(outfits, root, "outfit"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    stable = {
        "version": manifest.get("version"),
        "seed": manifest.get("seed"),
        "scenes": manifest.get("scenes"),
        "outfits": manifest.get("outfits"),
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    for key in ("scenes", "outfits"):
        entries = manifest.get(key)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Manifest field '{key}' must be a non-empty list")
        ids = [entry.get("id") for entry in entries]
        if None in ids or len(ids) != len(set(ids)):
            raise ValueError(f"Manifest field '{key}' contains missing or duplicate IDs")
    return manifest


def verify_manifest_files(manifest: dict[str, Any], root: Path) -> None:
    for group in ("scenes", "outfits"):
        for entry in manifest[group]:
            path = root / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Manifest file is missing: {path}")
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"Manifest file changed after selection: {path}")


def _completed_pairs(
    path: Path,
    manifest: dict[str, Any],
    judge: Judge,
) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    scene_hashes = {item["id"]: item["sha256"] for item in manifest["scenes"]}
    outfit_hashes = {item["id"]: item["sha256"] for item in manifest["outfits"]}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                scene_id, outfit_id = row["scene_id"], row["outfit_id"]
                if (
                    row.get("scene_sha256") != scene_hashes.get(scene_id)
                    or row.get("outfit_sha256") != outfit_hashes.get(outfit_id)
                    or row.get("judge_model") != judge.model_name
                    or row.get("prompt_version") != judge.prompt_version
                ):
                    raise ValueError(
                        f"Judgment cache is incompatible with the manifest or evaluator at line {line_number}"
                    )
                pair = (scene_id, outfit_id)
                if pair in completed:
                    raise ValueError(
                        f"Duplicate judgment for scene {scene_id} and outfit "
                        f"{outfit_id} at line {line_number}"
                    )
                completed.add(pair)
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid judgment JSONL at line {line_number}") from exc
    return completed


def validate_judgments(
    manifest: dict[str, Any],
    output_path: Path,
    judge_model: str,
    prompt_version: str,
    batch_size: int,
) -> int:
    """Require a complete, compatible judgment matrix before retrieval."""
    metadata_path = output_path.with_suffix(".meta.json")
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Judgment artifacts are incomplete: {output_path} and {metadata_path} "
            "must both exist"
        )
    expected_metadata = {
        "version": 1,
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "judge_model": judge_model,
        "prompt_version": prompt_version,
        "batch_size": batch_size,
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != expected_metadata:
        raise ValueError(
            f"Judgment cache metadata does not match this run: {metadata_path}"
        )
    identity = SimpleNamespace(
        model_name=judge_model,
        prompt_version=prompt_version,
    )
    completed = _completed_pairs(output_path, manifest, identity)
    expected_count = len(manifest["scenes"]) * len(manifest["outfits"])
    if len(completed) != expected_count:
        raise ValueError(
            f"Judgments are incomplete: {len(completed)}/{expected_count} pairs finished"
        )
    return len(completed)


def run_judging(
    manifest: dict[str, Any],
    root: Path,
    output_path: Path,
    judge: Judge,
    batch_size: int = 10,
    scene_ids: Sequence[str] | None = None,
) -> int:
    """Judge outstanding pairs and append each completed result immediately."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    verify_manifest_files(manifest, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".meta.json")
    metadata = {
        "version": 1,
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "judge_model": judge.model_name,
        "prompt_version": judge.prompt_version,
        "batch_size": batch_size,
    }
    if metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != metadata:
            raise ValueError(f"Judgment cache metadata does not match this run: {metadata_path}")
    else:
        write_json(metadata_path, metadata)
    completed = _completed_pairs(output_path, manifest, judge)
    scene_hashes = {item["id"]: item["sha256"] for item in manifest["scenes"]}
    outfit_hashes = {item["id"]: item["sha256"] for item in manifest["outfits"]}
    selected_scene_ids = set(scene_ids) if scene_ids is not None else set(scene_hashes)
    unknown_scene_ids = selected_scene_ids - set(scene_hashes)
    if unknown_scene_ids:
        raise ValueError(f"Unknown scene IDs: {sorted(unknown_scene_ids)}")
    written = 0
    selected_pair_count = len(selected_scene_ids) * len(manifest["outfits"])
    completed_selected = sum(
        1 for scene_id, _ in completed if scene_id in selected_scene_ids
    )
    print(
        f"[JUDGE] Starting {judge.model_name}: "
        f"{completed_selected}/{selected_pair_count} selected pairs cached",
        flush=True,
    )

    with output_path.open("a", encoding="utf-8", buffering=1) as stream:
        for scene_index, scene in enumerate(manifest["scenes"], 1):
            if scene["id"] not in selected_scene_ids:
                continue
            pending = [
                outfit
                for outfit in manifest["outfits"]
                if (scene["id"], outfit["id"]) not in completed
            ]
            if not pending:
                print(
                    f"[JUDGE] Scene {scene_index}/{len(manifest['scenes'])} "
                    f"{scene['id']}: all {len(manifest['outfits'])} judgments cached",
                    flush=True,
                )
                continue
            batch_count = math.ceil(len(pending) / batch_size)
            print(
                f"[JUDGE] Scene {scene_index}/{len(manifest['scenes'])} "
                f"{scene['id']}: {len(pending)} pairs pending in {batch_count} batches",
                flush=True,
            )
            random.Random(f"{manifest['seed']}:{scene['id']}").shuffle(pending)
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                requested = [(item["id"], root / item["path"]) for item in batch]
                batch_index = offset // batch_size + 1
                print(
                    f"[JUDGE] Scene {scene['id']}: sending batch "
                    f"{batch_index}/{batch_count} ({len(batch)} outfits)",
                    flush=True,
                )
                results = judge.score_batch(root / scene["path"], requested)
                returned_ids = [str(item.get("outfit_id")) for item in results]
                expected_ids = [item[0] for item in requested]
                if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(expected_ids):
                    raise ValueError(
                        f"Judge returned mismatched outfit IDs for scene {scene['id']}: "
                        f"expected {expected_ids}, got {returned_ids}"
                    )
                result_by_id = {item["outfit_id"]: item for item in results}
                for outfit_id in expected_ids:
                    result = result_by_id[outfit_id]
                    score = int(result["score"])
                    if not 1 <= score <= 5:
                        raise ValueError(f"Judge score must be between 1 and 5, got {score}")
                    row = {
                        "scene_id": scene["id"],
                        "outfit_id": outfit_id,
                        "scene_sha256": scene_hashes[scene["id"]],
                        "outfit_sha256": outfit_hashes[outfit_id],
                        "score": score,
                        "reason": str(result.get("reason", "")).strip(),
                        "judge_model": judge.model_name,
                        "prompt_version": judge.prompt_version,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                print(
                    f"[JUDGE] Scene {scene['id']}: saved batch "
                    f"{batch_index}/{batch_count}; {written} new judgments this run",
                    flush=True,
                )
    print(f"[JUDGE] Complete: wrote {written} new judgments", flush=True)
    return written
