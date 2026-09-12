"""Resumable GPTEval3D-style pairwise judging for SceneFit."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.services.benchmark.core import (
    manifest_fingerprint,
    verify_manifest_files,
    write_json,
)
from app.services.benchmark.pairwise_judge import CRITERIA, PROMPT_VERSION


PROTOCOL = "pairwise-elo-v1"
DEFAULT_ROUNDS = 7
DEFAULT_PAIRS_PER_REQUEST = 5
ELO_INITIAL = 1000.0
ELO_K = 32.0


class PairwiseJudge(Protocol):
    model_name: str
    prompt_version: str

    def compare_batch(
        self,
        scene_path: Path,
        pairs: Sequence[tuple[str, Path, Path]],
    ) -> list[dict[str, Any]]: ...


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metadata(
    manifest: dict[str, Any],
    judge: PairwiseJudge,
    rounds: int,
    pairs_per_request: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "judge_model": judge.model_name,
        "prompt_version": judge.prompt_version,
        "rounds": rounds,
        "pairs_per_request": pairs_per_request,
        "mirrored": True,
        "elo_initial": ELO_INITIAL,
        "elo_k": ELO_K,
        "seed": manifest["seed"],
    }


def _winner(decisions: Sequence[dict[str, Any]], criterion: str) -> str | None:
    for decision in decisions:
        if decision["criterion"] == criterion:
            return decision["winner"]
    raise ValueError(f"Response omitted criterion '{criterion}'")


def _outcome_points(comparison: dict[str, Any], outfit_id: str) -> float:
    winner = comparison["outcomes"]["overall"]["winner"]
    if winner is None:
        return 0.5
    return 1.0 if winner == outfit_id else 0.0


def _history(comparisons: Sequence[dict[str, Any]], scene_id: str) -> set[frozenset[str]]:
    return {
        frozenset((str(row["left_outfit_id"]), str(row["right_outfit_id"])))
        for row in comparisons
        if row["scene_id"] == scene_id
    }


def _points(comparisons: Sequence[dict[str, Any]], scene_id: str) -> dict[str, float]:
    points: dict[str, float] = defaultdict(float)
    for row in comparisons:
        if row["scene_id"] != scene_id:
            continue
        points[row["left_outfit_id"]] += _outcome_points(row, row["left_outfit_id"])
        points[row["right_outfit_id"]] += _outcome_points(row, row["right_outfit_id"])
    return points


def _pair_round(
    outfit_ids: Sequence[str],
    history: set[frozenset[str]],
    points: dict[str, float],
    seed: str,
    round_number: int,
) -> list[tuple[str, str]]:
    """Create one deterministic Swiss round without prior pair repeats."""
    if len(outfit_ids) % 2:
        raise ValueError("Pairwise tournament requires an even number of outfits")
    ordered = list(outfit_ids)
    if round_number == 1:
        random.Random(seed).shuffle(ordered)
    else:
        ordered.sort(key=lambda outfit_id: (-points.get(outfit_id, 0.0), outfit_id))
    remaining = ordered[:]
    pairs: list[tuple[str, str]] = []
    while remaining:
        left = remaining.pop(0)
        candidate_index = next(
            (
                index
                for index, right in enumerate(remaining)
                if frozenset((left, right)) not in history
            ),
            None,
        )
        if candidate_index is None:
            # At seven rounds with 100 candidates this should not occur. A swap
            # with the latest pair retains a perfect matching without a rematch.
            if not pairs:
                raise ValueError("Unable to schedule a non-repeated comparison")
            previous_left, previous_right = pairs.pop()
            options = (
                ((left, previous_left), (previous_right,)),
                ((left, previous_right), (previous_left,)),
            )
            for candidate_pair, replacement in options:
                if frozenset(candidate_pair) not in history:
                    pairs.append(candidate_pair)
                    remaining = list(replacement) + remaining
                    break
            else:
                raise ValueError("Unable to repair pairwise tournament matching")
            continue
        right = remaining.pop(candidate_index)
        pairs.append((left, right))
    return pairs


def _response_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["comparison_id"]), str(row["orientation"])


def _reconcile(
    original: dict[str, Any],
    mirrored: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for criterion in CRITERIA:
        first = _winner(original["decisions"], criterion)
        second = _winner(mirrored["decisions"], criterion)
        first_winner = (
            original["left_outfit_id"] if first == "left"
            else original["right_outfit_id"] if first == "right" else None
        )
        second_winner = (
            mirrored["left_outfit_id"] if second == "left"
            else mirrored["right_outfit_id"] if second == "right" else None
        )
        winner = first_winner if first_winner is not None and first_winner == second_winner else None
        outcomes[criterion] = {
            "winner": winner,
            "reason": (
                next(item["reason"] for item in original["decisions"] if item["criterion"] == criterion)
                if winner is not None else "Mirrored presentations disagreed or tied"
            ),
        }
    return outcomes


def _ratings(
    manifest: dict[str, Any], comparisons: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {"version": 1, "protocol": PROTOCOL, "scenes": {}}
    outfit_ids = [str(item["id"]) for item in manifest["outfits"]]
    for scene in manifest["scenes"]:
        scene_id = str(scene["id"])
        scene_comparisons = sorted(
            (row for row in comparisons if row["scene_id"] == scene_id),
            key=lambda row: (int(row["round"]), str(row["comparison_id"])),
        )
        criteria: dict[str, Any] = {}
        for criterion in CRITERIA:
            ratings = {outfit_id: ELO_INITIAL for outfit_id in outfit_ids}
            records = {outfit_id: {"wins": 0, "losses": 0, "ties": 0} for outfit_id in outfit_ids}
            for row in scene_comparisons:
                left, right = row["left_outfit_id"], row["right_outfit_id"]
                winner = row["outcomes"][criterion]["winner"]
                left_score = 0.5 if winner is None else float(winner == left)
                right_score = 1.0 - left_score
                expected_left = 1 / (1 + 10 ** ((ratings[right] - ratings[left]) / 400))
                ratings[left] += ELO_K * (left_score - expected_left)
                ratings[right] += ELO_K * (right_score - (1 - expected_left))
                if winner is None:
                    records[left]["ties"] += 1
                    records[right]["ties"] += 1
                else:
                    loser = right if winner == left else left
                    records[winner]["wins"] += 1
                    records[loser]["losses"] += 1
            ordered = sorted(outfit_ids, key=lambda item: (-ratings[item], item))
            criteria[criterion] = [
                {
                    "outfit_id": outfit_id,
                    "rank": rank,
                    "rating": round(ratings[outfit_id], 6),
                    **records[outfit_id],
                }
                for rank, outfit_id in enumerate(ordered, 1)
            ]
        result["scenes"][scene_id] = {"criteria": criteria}
    return result


def run_pairwise_judging(
    manifest: dict[str, Any],
    root: Path,
    output_dir: Path,
    judge: PairwiseJudge,
    rounds: int = DEFAULT_ROUNDS,
    pairs_per_request: int = DEFAULT_PAIRS_PER_REQUEST,
    scene_ids: Sequence[str] | None = None,
) -> int:
    """Run or resume the mirrored pairwise tournament and write Elo ratings."""
    if rounds < 1 or pairs_per_request < 1:
        raise ValueError("rounds and pairs_per-request must be positive")
    verify_manifest_files(manifest, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "judge.meta.json"
    metadata = _metadata(manifest, judge, rounds, pairs_per_request)
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Pairwise judge metadata does not match this run")
    if not meta_path.exists():
        write_json(meta_path, metadata)
    responses_path = output_dir / "judge_responses.jsonl"
    comparisons_path = output_dir / "comparisons.jsonl"
    responses = _read_jsonl(responses_path)
    comparisons = _read_jsonl(comparisons_path)
    response_map = {_response_key(row): row for row in responses}
    comparison_ids = {str(row["comparison_id"]) for row in comparisons}
    selected = set(scene_ids) if scene_ids is not None else {item["id"] for item in manifest["scenes"]}
    unknown = selected - {item["id"] for item in manifest["scenes"]}
    if unknown:
        raise ValueError(f"Unknown scene IDs: {sorted(unknown)}")
    outfit_paths = {item["id"]: root / item["path"] for item in manifest["outfits"]}
    scene_hashes = {item["id"]: item["sha256"] for item in manifest["scenes"]}
    outfit_hashes = {item["id"]: item["sha256"] for item in manifest["outfits"]}
    written = 0

    for scene in manifest["scenes"]:
        scene_id = str(scene["id"])
        if scene_id not in selected:
            continue
        for round_number in range(1, rounds + 1):
            existing_round = [
                row for row in comparisons
                if row["scene_id"] == scene_id and int(row["round"]) == round_number
            ]
            if len(existing_round) == len(manifest["outfits"]) // 2:
                continue
            if existing_round:
                raise ValueError(f"Scene {scene_id} round {round_number} is partially reconciled")
            schedule = _pair_round(
                [str(item["id"]) for item in manifest["outfits"]],
                _history(comparisons, scene_id),
                _points(comparisons, scene_id),
                f"{manifest['seed']}:{scene_id}:round-{round_number}",
                round_number,
            )
            round_pairs = [
                {
                    "comparison_id": f"{scene_id}:r{round_number}:p{index:03d}",
                    "left_outfit_id": left,
                    "right_outfit_id": right,
                }
                for index, (left, right) in enumerate(schedule, 1)
            ]
            for offset in range(0, len(round_pairs), pairs_per_request):
                batch = round_pairs[offset : offset + pairs_per_request]
                for orientation in ("original", "mirrored"):
                    missing = [
                        pair for pair in batch
                        if (pair["comparison_id"], orientation) not in response_map
                    ]
                    if not missing:
                        continue
                    request_pairs = [
                        (
                            pair["comparison_id"],
                            outfit_paths[
                                pair["left_outfit_id"] if orientation == "original" else pair["right_outfit_id"]
                            ],
                            outfit_paths[
                                pair["right_outfit_id"] if orientation == "original" else pair["left_outfit_id"]
                            ],
                        )
                        for pair in missing
                    ]
                    decisions = judge.compare_batch(root / scene["path"], request_pairs)
                    returned = {item["pair_id"]: item for item in decisions}
                    rows = []
                    for pair in missing:
                        left = pair["left_outfit_id"] if orientation == "original" else pair["right_outfit_id"]
                        right = pair["right_outfit_id"] if orientation == "original" else pair["left_outfit_id"]
                        rows.append(
                            {
                                "comparison_id": pair["comparison_id"],
                                "scene_id": scene_id,
                                "round": round_number,
                                "orientation": orientation,
                                "left_outfit_id": left,
                                "right_outfit_id": right,
                                "scene_sha256": scene_hashes[scene_id],
                                "left_outfit_sha256": outfit_hashes[left],
                                "right_outfit_sha256": outfit_hashes[right],
                                "decisions": returned[pair["comparison_id"]]["decisions"],
                                "judge_model": judge.model_name,
                                "prompt_version": judge.prompt_version,
                            }
                        )
                    _append_jsonl(responses_path, rows)
                    responses.extend(rows)
                    response_map.update({_response_key(row): row for row in rows})
                reconciled = []
                for pair in batch:
                    if pair["comparison_id"] in comparison_ids:
                        continue
                    original = response_map[(pair["comparison_id"], "original")]
                    mirrored = response_map[(pair["comparison_id"], "mirrored")]
                    reconciled.append(
                        {
                            "comparison_id": pair["comparison_id"],
                            "scene_id": scene_id,
                            "round": round_number,
                            "left_outfit_id": pair["left_outfit_id"],
                            "right_outfit_id": pair["right_outfit_id"],
                            "scene_sha256": scene_hashes[scene_id],
                            "left_outfit_sha256": outfit_hashes[pair["left_outfit_id"]],
                            "right_outfit_sha256": outfit_hashes[pair["right_outfit_id"]],
                            "outcomes": _reconcile(original, mirrored),
                            "judge_model": judge.model_name,
                            "prompt_version": judge.prompt_version,
                        }
                    )
                _append_jsonl(comparisons_path, reconciled)
                comparisons.extend(reconciled)
                comparison_ids.update(row["comparison_id"] for row in reconciled)
            write_json(output_dir / "ratings.json", _ratings(manifest, comparisons))
            written += len(round_pairs) - len(existing_round)
    write_json(output_dir / "ratings.json", _ratings(manifest, comparisons))
    return written


def validate_pairwise_judge(
    manifest: dict[str, Any],
    output_dir: Path,
    model_name: str,
    rounds: int = DEFAULT_ROUNDS,
    pairs_per_request: int = DEFAULT_PAIRS_PER_REQUEST,
) -> int:
    """Require a complete compatible mirrored tournament and rating artifact."""
    meta_path = output_dir / "judge.meta.json"
    comparisons_path = output_dir / "comparisons.jsonl"
    responses_path = output_dir / "judge_responses.jsonl"
    ratings_path = output_dir / "ratings.json"
    if not all(path.is_file() for path in (meta_path, comparisons_path, responses_path, ratings_path)):
        raise FileNotFoundError("Pairwise judge artifacts are incomplete")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "judge_model": model_name,
        "prompt_version": PROMPT_VERSION,
        "rounds": rounds,
        "pairs_per_request": pairs_per_request,
        "mirrored": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Pairwise judge metadata mismatch for {key}")
    comparisons = _read_jsonl(comparisons_path)
    responses = _read_jsonl(responses_path)
    expected_per_scene = len(manifest["outfits"]) // 2 * rounds
    scene_ids = {item["id"] for item in manifest["scenes"]}
    if len(comparisons) != expected_per_scene * len(scene_ids):
        raise ValueError("Pairwise comparison count is incomplete")
    keys = {_response_key(row) for row in responses}
    if len(keys) != len(responses):
        raise ValueError("Duplicate pairwise response orientation")
    seen_pairs: set[tuple[str, frozenset[str]]] = set()
    appearances: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in comparisons:
        scene_id = row["scene_id"]
        if scene_id not in scene_ids:
            raise ValueError("Unknown scene in pairwise comparison")
        pair = frozenset((row["left_outfit_id"], row["right_outfit_id"]))
        pair_key = (scene_id, pair)
        if pair_key in seen_pairs:
            raise ValueError("Repeated outfit pairing")
        seen_pairs.add(pair_key)
        for outfit_id in pair:
            appearances[(scene_id, int(row["round"]), outfit_id)] += 1
        for criterion in CRITERIA:
            if criterion not in row["outcomes"]:
                raise ValueError("Missing pairwise criterion")
        comparison_id = str(row["comparison_id"])
        if (comparison_id, "original") not in keys or (comparison_id, "mirrored") not in keys:
            raise ValueError("Each comparison requires original and mirrored responses")
    for scene_id in scene_ids:
        for round_number in range(1, rounds + 1):
            for outfit in manifest["outfits"]:
                if appearances[(scene_id, round_number, outfit["id"])] != 1:
                    raise ValueError("Every outfit must appear once per round")
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    if ratings.get("protocol") != PROTOCOL or set(ratings.get("scenes", {})) != scene_ids:
        raise ValueError("Pairwise Elo ratings are incompatible")
    return len(comparisons)
