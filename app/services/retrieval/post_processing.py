"""
Post-processing for retrieval results.
Canonical location: app/services/retrieval/post_processing.py
(Backward-compat shim remains at app/services/post_processing.py)
"""
import random
from difflib import SequenceMatcher


def _get_distributions(num_results: int, top_k: int) -> list[int]:
    """
    Distribute num_results into top_k size-increasing bins.

    Guarantees all bins are at least 1 (when top_k <= num_results) to avoid
    downstream zero-sized buckets.
    """
    if top_k <= 0 or num_results <= 0:
        return []

    top_k = min(top_k, num_results)

    if top_k == 1:
        return [num_results]

    weights = [(i + 1) ** 2 for i in range(top_k)]
    total_weight = sum(weights)

    remaining = num_results - top_k
    distributions = [1] * top_k
    cumulative = 0

    for idx, weight in enumerate(weights):
        extra = int((weight / total_weight) * remaining)
        distributions[idx] += extra
        cumulative += extra

    distributions[-1] += remaining - cumulative
    return distributions


def _default_name_similarity(a: str | None, b: str | None) -> float:
    """Lightweight string similarity used when no embedding similarity is available."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def rerank_with_soft_penalty(
    results: list[dict],
    alpha: float = 0.25,
    similarity_threshold: float = 0.85,
    similarity_fn=None,
) -> list[dict]:
    """
    Diversity-aware re-ranking with a soft penalty on near-duplicate items.

    adjusted_score = relevance - alpha * max_similarity_to_already_picked

    Args:
        results: List of {name, score, ...} dicts, any order.
        alpha: Penalty weight for similarity.
        similarity_threshold: Early-exit threshold; items above this are
            treated as already penalised to the max.
        similarity_fn: Optional (a, b) -> float. Defaults to string ratio on names.
    """
    if not results:
        return []

    sim_fn = similarity_fn or _default_name_similarity
    remaining = sorted(results, key=lambda x: x.get("score", 0.0) or 0.0, reverse=True)
    picked: list[dict] = []

    while remaining:
        best_idx = None
        best_adjusted = None

        for idx, candidate in enumerate(remaining):
            name = str(candidate.get("name", ""))
            base_score = float(candidate.get("score", 0.0) or 0.0)
            max_sim = 0.0

            for chosen in picked:
                sim = sim_fn(name, str(chosen.get("name", "")))
                if sim > max_sim:
                    max_sim = sim
                if max_sim >= similarity_threshold:
                    break

            adjusted = base_score - alpha * max_sim
            if best_adjusted is None or adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = idx

        picked.append(remaining.pop(best_idx))

    return picked


def shuffle_retrieval_results(
    results: list[dict],
    k: int,
    apply_soft_penalty: bool = False,
    penalty_alpha: float = 0.25,
    penalty_similarity_threshold: float = 0.85,
    similarity_fn=None,
) -> list[dict]:
    """
    Post-process retrieval results with optional diversity re-ranking and
    bucket-shuffle for mild randomness while keeping high-scoring items near
    the top.

    Steps:
      1. (Optional) Apply soft diversity penalty via `rerank_with_soft_penalty`.
      2. Partition into size-increasing buckets and shuffle within each bucket.
      3. Return the top-k items.
    """
    if not results:
        return []

    if k >= len(results):
        return results[:k]

    if apply_soft_penalty:
        results = rerank_with_soft_penalty(
            results,
            alpha=penalty_alpha,
            similarity_threshold=penalty_similarity_threshold,
            similarity_fn=similarity_fn,
        )

    num_results = len(results)
    distributions = _get_distributions(num_results, min(k, num_results))

    shuffled: list[dict] = []
    start = 0
    for bucket_size in distributions:
        bucket = results[start : start + bucket_size]
        random.shuffle(bucket)
        shuffled.append(bucket[0])
        start += bucket_size

    return shuffled[:k]
