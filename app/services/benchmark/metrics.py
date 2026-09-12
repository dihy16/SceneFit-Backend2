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


def _pairwise_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid comparison JSONL at line {line_number}") from exc
    return rows


def _kendall_tau(order: Sequence[str], reference: Sequence[str]) -> float:
    """Kendall tau for two complete deterministic orderings."""
    positions = {item: index for index, item in enumerate(reference)}
    concordant = 0
    discordant = 0
    for left_index, left in enumerate(order):
        for right in order[left_index + 1 :]:
            if positions[left] < positions[right]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return 0.0 if total == 0 else (concordant - discordant) / total


def evaluate_pairwise_benchmark(
    manifest: dict[str, Any],
    comparisons_path: Path,
    ratings_path: Path,
    rankings_dir: Path,
    output_dir: Path,
    ks: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    """Evaluate retrieval order against reconciled pairwise VLM preferences."""
    comparisons = _pairwise_rows(comparisons_path)
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    expected_ids = [str(item["id"]) for item in manifest["outfits"]]
    expected_scenes = [str(item["id"]) for item in manifest["scenes"]]
    by_scene: dict[str, list[dict[str, Any]]] = {scene_id: [] for scene_id in expected_scenes}
    for row in comparisons:
        scene_id = str(row["scene_id"])
        if scene_id not in by_scene:
            raise ValueError(f"Unknown scene in comparisons: {scene_id}")
        by_scene[scene_id].append(row)
    ranking_paths = sorted(rankings_dir.glob("*.json"))
    if not ranking_paths:
        raise ValueError(f"No ranking JSON files found in {rankings_dir}")

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for ranking_path in ranking_paths:
        payload = json.loads(ranking_path.read_text(encoding="utf-8"))
        method = str(payload.get("method", ranking_path.stem))
        method_rows: list[dict[str, Any]] = []
        for scene_id in expected_scenes:
            ranking = normalize_ranking(payload.get("scenes", {}).get(scene_id, []), expected_ids)
            ordered = [str(item["outfit_id"]) for item in ranking]
            positions = {outfit_id: index for index, outfit_id in enumerate(ordered)}
            decisive = 0
            decisive_correct = 0.0
            agreement_total = 0.0
            for comparison in by_scene[scene_id]:
                left = str(comparison["left_outfit_id"])
                right = str(comparison["right_outfit_id"])
                winner = comparison["outcomes"]["overall"]["winner"]
                if winner is None:
                    agreement_total += 0.5
                    continue
                loser = right if winner == left else left
                correct = positions[winner] < positions[loser]
                decisive += 1
                decisive_correct += float(correct)
                agreement_total += float(correct)
            overall_ratings = ratings["scenes"][scene_id]["criteria"]["overall"]
            elo_order = [str(item["outfit_id"]) for item in overall_ratings]
            elo_by_id = {str(item["outfit_id"]): float(item["rating"]) for item in overall_ratings}
            row: dict[str, Any] = {
                "method": method,
                "scene_id": scene_id,
                "coverage": len(ordered) / len(expected_ids),
                "pairwise_agreement": agreement_total / len(by_scene[scene_id]),
                "decisive_pairwise_accuracy": (
                    decisive_correct / decisive if decisive else 0.5
                ),
                "kendall_tau": _kendall_tau(ordered, elo_order),
            }
            for k in ks:
                row[f"top{k}_overlap"] = len(set(ordered[:k]) & set(elo_order[:k])) / k
                row[f"mean_elo@{k}"] = statistics.fmean(elo_by_id[item] for item in ordered[:k])
            rows.append(row)
            method_rows.append(row)
        method_summary: dict[str, Any] = {"scene_count": len(method_rows), "coverage": 1.0}
        metric_names = [
            "pairwise_agreement",
            "decisive_pairwise_accuracy",
            "kendall_tau",
            *[f"top{k}_overlap" for k in ks],
            *[f"mean_elo@{k}" for k in ks],
        ]
        for metric in metric_names:
            values = [float(row[metric]) for row in method_rows]
            method_summary[metric] = statistics.fmean(values)
            method_summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[method] = method_summary
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "version": 2,
        "evaluation_protocol": "pairwise-elo-v1",
        "scene_count": len(expected_scenes),
        "outfit_count": len(expected_ids),
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "cutoffs": list(ks),
        "primary_metric": "pairwise_agreement",
        "metrics": summary,
    }
    write_json(output_dir / "summary.json", result)
    return result
