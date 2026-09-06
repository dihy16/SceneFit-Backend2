from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.services.benchmark.collector import normalize_ranking
from app.services.benchmark.core import manifest_fingerprint, write_json


def dcg(relevances: Sequence[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], 1))


def ndcg(ranked_relevances: Sequence[int], all_relevances: Iterable[int], k: int) -> float:
    ideal = dcg(sorted(all_relevances, reverse=True), k)
    return 0.0 if ideal == 0 else dcg(ranked_relevances, k) / ideal


def _judgments(path: Path) -> tuple[dict[str, dict[str, int]], set[str], set[str]]:
    matrix: dict[str, dict[str, int]] = {}
    models: set[str] = set()
    prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            scene_scores = matrix.setdefault(row["scene_id"], {})
            if row["outfit_id"] in scene_scores:
                raise ValueError(f"Duplicate judgment at line {line_number}")
            scene_scores[row["outfit_id"]] = int(row["score"])
            models.add(str(row.get("judge_model", "unknown")))
            prompts.add(str(row.get("prompt_version", "unknown")))
    return matrix, models, prompts


def evaluate_benchmark(
    manifest: dict[str, Any],
    judgments_path: Path,
    rankings_dir: Path,
    output_dir: Path,
    ks: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    expected_ids = [item["id"] for item in manifest["outfits"]]
    expected_scenes = [item["id"] for item in manifest["scenes"]]
    matrix, judge_models, prompt_versions = _judgments(judgments_path)
    expected_pairs = {(scene, outfit) for scene in expected_scenes for outfit in expected_ids}
    actual_pairs = {(scene, outfit) for scene, scores in matrix.items() for outfit in scores}
    if actual_pairs != expected_pairs:
        missing = len(expected_pairs - actual_pairs)
        extra = len(actual_pairs - expected_pairs)
        raise ValueError(f"Judgment matrix is incomplete or foreign: missing={missing}, extra={extra}")

    ranking_paths = sorted(rankings_dir.glob("*.json"))
    if not ranking_paths:
        raise ValueError(f"No ranking JSON files found in {rankings_dir}")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for ranking_path in ranking_paths:
        payload = json.loads(ranking_path.read_text(encoding="utf-8"))
        method = payload.get("method", ranking_path.stem)
        method_rows = []
        for scene_id in expected_scenes:
            if scene_id not in payload.get("scenes", {}):
                raise ValueError(f"Method {method} has no ranking for scene {scene_id}")
            ranking = normalize_ranking(payload["scenes"][scene_id], expected_ids)
            relevances = [matrix[scene_id][item["outfit_id"]] for item in ranking]
            row: dict[str, Any] = {
                "method": method,
                "scene_id": scene_id,
                "coverage": len(ranking) / len(expected_ids),
            }
            for k in ks:
                row[f"ndcg@{k}"] = ndcg(relevances, matrix[scene_id].values(), k)
                row[f"mean_relevance@{k}"] = statistics.fmean(relevances[:k])
            rows.append(row)
            method_rows.append(row)

        method_summary: dict[str, Any] = {"scene_count": len(method_rows), "coverage": 1.0}
        for metric in [f"{name}@{k}" for k in ks for name in ("ndcg", "mean_relevance")]:
            values = [float(row[metric]) for row in method_rows]
            method_summary[metric] = statistics.fmean(values)
            method_summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[method] = method_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_scene.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "version": 1,
        "scene_count": len(expected_scenes),
        "outfit_count": len(expected_ids),
        "seed": manifest.get("seed"),
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "cutoffs": list(ks),
        "judge_models": sorted(judge_models),
        "prompt_versions": sorted(prompt_versions),
        "metrics": summary,
    }
    write_json(output_dir / "summary.json", result)
    return result
