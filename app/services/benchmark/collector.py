from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from app.services.benchmark.core import manifest_fingerprint, verify_manifest_files, write_json


NAME_KEYS = ("outfit_id", "name", "outfit_name", "name_clothes", "image_name", "file_name", "metadata")


def _outfit_id(item: dict[str, Any]) -> str:
    value = next((item.get(key) for key in NAME_KEYS if item.get(key) is not None), None)
    if value is None:
        raise ValueError(f"Ranking item has no outfit identifier: {item}")
    return Path(str(value)).stem


def normalize_ranking(payload: Any, expected_ids: Sequence[str]) -> list[dict[str, Any]]:
    items = payload.get("results") if isinstance(payload, dict) and "results" in payload else payload
    if not isinstance(items, list):
        raise ValueError("Worker response must be a list or an object containing 'results'")
    normalized = []
    for rank, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError("Every worker result must be an object")
        normalized.append(
            {
                "outfit_id": _outfit_id(item),
                "rank": rank,
                "raw_score": item.get("score", item.get("similarity")),
            }
        )
    ids = [item["outfit_id"] for item in normalized]
    expected = list(expected_ids)
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    missing = sorted(set(expected) - set(ids))
    extra = sorted(set(ids) - set(expected))
    if duplicates or missing or extra or len(ids) != len(expected):
        raise ValueError(
            "Worker did not rank the exact manifest pool: "
            f"duplicates={duplicates}, missing={missing}, extra={extra}"
        )
    return normalized


def _response_items(response: Any) -> Any:
    response.raise_for_status()
    return response.json()


def collect_rankings(
    manifest: dict[str, Any],
    root: Path,
    config_path: Path,
    output_dir: Path,
    methods: Sequence[str] | None = None,
    request: Callable[..., Any] | None = None,
    scene_ids: Sequence[str] | None = None,
) -> list[Path]:
    """Collect strict rankings directly from configured model workers."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_manifest_files(manifest, root)
    fingerprint = manifest_fingerprint(manifest)
    worker_config = config["retrieval_methods"]
    selected_methods = list(methods or sorted(worker_config))
    retry = config.get("retry", {"max_attempts": 3, "delay_seconds": 1})
    timeout = config.get("timeout", 300)
    if request is None:
        import httpx

        request = httpx.post

    expected_ids = [item["id"] for item in manifest["outfits"]]
    all_scene_ids = {item["id"] for item in manifest["scenes"]}
    selected_scene_ids = set(scene_ids) if scene_ids is not None else all_scene_ids
    unknown_scene_ids = selected_scene_ids - all_scene_ids
    if unknown_scene_ids:
        raise ValueError(f"Unknown scene IDs: {sorted(unknown_scene_ids)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for method in selected_methods:
        if method not in worker_config:
            raise ValueError(f"Unknown retrieval method: {method}")
        cfg = worker_config[method]
        base_url = str(cfg.get("url", "")).rstrip("/")
        if not base_url:
            raise ValueError(f"Worker URL is empty for method '{method}'")
        url = f"{base_url}/{str(cfg['endpoint']).lstrip('/')}"
        output_path = output_dir / f"{method}.json"
        cached = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
        if cached and cached.get("manifest_fingerprint") != fingerprint:
            raise ValueError(f"Cached ranking does not match the current manifest: {output_path}")
        scenes = cached.get("scenes", {})

        for scene in manifest["scenes"]:
            if scene["id"] not in selected_scene_ids:
                continue
            if scene["id"] in scenes:
                normalize_ranking(scenes[scene["id"]], expected_ids)
                continue
            last_error: Exception | None = None
            for attempt in range(int(retry["max_attempts"])):
                try:
                    scene_path = root / scene["path"]
                    with scene_path.open("rb") as image_stream:
                        response = request(
                            url,
                            files={"image": (scene_path.name, image_stream, "application/octet-stream")},
                            data={
                                "top_k": len(expected_ids),
                                "candidate_names": json.dumps(expected_ids),
                            },
                            headers={"ngrok-skip-browser-warning": "true"},
                            timeout=timeout,
                        )
                    scenes[scene["id"]] = normalize_ranking(_response_items(response), expected_ids)
                    write_json(
                        output_path,
                        {
                            "version": 1,
                            "method": method,
                            "manifest_fingerprint": fingerprint,
                            "worker_endpoint": url,
                            "scenes": scenes,
                        },
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < int(retry["max_attempts"]):
                        time.sleep(float(retry["delay_seconds"]) * (2**attempt))
            else:
                raise RuntimeError(f"Failed collecting {method} for scene {scene['id']}") from last_error
        written_paths.append(output_path)
    return written_paths
