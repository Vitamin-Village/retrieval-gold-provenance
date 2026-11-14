"""Retrieval metrics for Manufacturing GraphRAG Paper V3 diagnostics."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Optional


def source_metrics(predicted_ids: list[str], gold_ids: list[str], k: int) -> dict[str, object]:
    """Return source retrieval metrics at rank ``k``.

    ``matched_rank`` is the 1-based rank of the first matching prediction within
    the raw top-k ranking. It is ``None`` when there is no match.
    """
    top_predictions = _top_k(predicted_ids, k)
    gold_set = set(gold_ids)

    matched_rank = _first_match_rank(top_predictions, gold_set)
    matched_gold = {predicted_id for predicted_id in top_predictions if predicted_id in gold_set}
    recall = len(matched_gold) / len(gold_set) if gold_set else 0.0

    return {
        "hit_at_k": 1.0 if matched_rank is not None else 0.0,
        "mrr": (1.0 / matched_rank) if matched_rank is not None else 0.0,
        "recall_at_k": recall,
        "matched_rank": matched_rank,
    }


def entity_metrics(predicted_ids: list[str], gold_ids: list[str], k: int) -> dict[str, float]:
    """Return entity retrieval precision, recall, F1, and hit at rank ``k``."""
    return _deduped_retrieval_metrics(predicted_ids, gold_ids, k)


def edge_metrics(predicted_ids: list[str], gold_ids: list[str], k: int) -> dict[str, float]:
    """Return edge retrieval precision, recall, F1, and hit at rank ``k``."""
    return _deduped_retrieval_metrics(predicted_ids, gold_ids, k)


def path_exact_match(predicted_paths: list[list[str]], gold_paths: list[list[str]]) -> float:
    """Return 1.0 when any predicted path matches a gold path on endpoints.

    P4 fix: gold paths in qa_v4_from_kg.jsonl use the 2-node shape
    ``[head_entity_id, tail_entity_id]`` while predicted paths returned by
    ``_reconstruct_retrieved_paths`` use the 3+ element shape
    ``[src, edge_id, tgt, edge_id, ...]``. The historical strict tuple
    comparison therefore always returned 0. We now canonicalise both
    representations to their ``(head, tail)`` endpoint tuple before
    comparing so the metric is non-degenerate while remaining stricter than
    ``path_endpoint_recall`` (which counts per-gold matches rather than a
    binary any-match indicator).
    """
    if not predicted_paths or not gold_paths:
        return 0.0

    gold_endpoints = {_canonical_path_endpoints(path) for path in gold_paths}
    gold_endpoints.discard(("", ""))
    if not gold_endpoints:
        return 0.0
    for predicted_path in predicted_paths:
        endpoints = _canonical_path_endpoints(predicted_path)
        if endpoints == ("", ""):
            continue
        if endpoints in gold_endpoints:
            return 1.0
    return 0.0


def path_endpoint_recall(
    predicted_paths: list[list[str]], gold_paths: list[list[str]]
) -> float:
    """Fraction of gold paths whose endpoint pair appears in any predicted path.

    Gold paths and predicted paths are normalised to their ``(head, tail)``
    endpoint tuple via ``_canonical_path_endpoints`` so the 2-node gold
    shape and the 3+ element predicted shape are comparable structurally.
    Returns 0 when there are no gold paths.
    """
    if not gold_paths:
        return 0.0
    gold_endpoints = [_canonical_path_endpoints(path) for path in gold_paths]
    gold_endpoints = [pair for pair in gold_endpoints if pair != ("", "")]
    if not gold_endpoints:
        return 0.0
    predicted_endpoint_set = {
        endpoints
        for endpoints in (
            _canonical_path_endpoints(path) for path in predicted_paths
        )
        if endpoints != ("", "")
    }
    if not predicted_endpoint_set:
        return 0.0
    matched = sum(
        1 for pair in gold_endpoints if pair in predicted_endpoint_set
    )
    return matched / len(gold_endpoints)


def path_edge_sequence_recall(
    predicted_paths: list[list[str]], gold_paths: list[list[str]]
) -> float:
    """Orderless recall of gold edges across predicted multi-hop paths.

    For each gold path the function derives its edge set (consecutive
    ``(head, tail)`` pairs); for 2-node gold paths this is a single pair
    ``{(head, tail)}``. Predicted paths contribute their consecutive node
    pairs ``(node_i, node_{i+2})`` (skipping edge-id slots) because
    ``_reconstruct_retrieved_paths`` returns the ``[src, edge_id, tgt, ...]``
    shape. We average the per-gold-path coverage so multi-hop golds with a
    partial predicted overlap can still contribute fractional credit.
    Returns 0 when no gold paths have any edges.
    """
    if not gold_paths or not predicted_paths:
        return 0.0
    gold_edge_lists = [_path_edge_pairs(path) for path in gold_paths]
    gold_edge_lists = [edges for edges in gold_edge_lists if edges]
    if not gold_edge_lists:
        return 0.0
    predicted_edges: set[tuple[str, str]] = set()
    for path in predicted_paths:
        predicted_edges.update(_path_edge_pairs(path))
    if not predicted_edges:
        return 0.0

    coverages: list[float] = []
    for gold_edges in gold_edge_lists:
        matched = sum(1 for pair in gold_edges if pair in predicted_edges)
        coverages.append(matched / len(gold_edges))
    return sum(coverages) / len(coverages)


def _canonical_path_endpoints(path: Sequence[str]) -> tuple[str, str]:
    """Reduce any path representation to ``(head_node_id, tail_node_id)``.

    Accepts both the 2-node gold shape ``[head, tail]`` and the
    ``[src, edge_id, tgt, edge_id, ...]`` predicted shape. Returns
    ``("", "")`` for empty / malformed input so callers can filter the
    sentinel out before set comparisons.
    """
    if not path:
        return ("", "")
    head = str(path[0]).strip()
    tail = str(path[-1]).strip()
    if not head or not tail:
        return ("", "")
    return (head, tail)


def _path_edge_pairs(path: Sequence[str]) -> set[tuple[str, str]]:
    """Extract orderless ``(head, tail)`` pairs from a path.

    The 2-node gold shape ``[head, tail]`` yields ``{(head, tail)}``. The
    ``[src, edge_id, tgt, edge_id, ...]`` predicted shape yields pairs
    ``(src, tgt), (tgt, next_tgt), ...`` by stepping over edge-id slots.
    Returns an empty set for paths shorter than 2 nodes.
    """
    parts = [str(item).strip() for item in path if str(item).strip()]
    if len(parts) < 2:
        return set()
    if len(parts) == 2:
        return {(parts[0], parts[1])}
    # Predicted shape: nodes at even indices (0, 2, 4, ...), edge ids in odd.
    nodes = parts[::2]
    pairs: set[tuple[str, str]] = set()
    for index in range(len(nodes) - 1):
        pairs.add((nodes[index], nodes[index + 1]))
    return pairs


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> dict[str, float | int]:
    """Return a percentile bootstrap confidence interval for the sample mean."""
    scores = _float_sequence(values, name="values")
    _validate_resample_count(n_bootstrap, name="n_bootstrap")
    _validate_confidence(confidence)

    rng = random.Random(seed)
    sample_size = len(scores)
    bootstrapped_means = [
        sum(scores[rng.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(n_bootstrap)
    ]
    bootstrapped_means.sort()

    alpha = (1.0 - confidence) / 2.0
    return {
        "n": sample_size,
        "mean": _mean(scores),
        "ci_low": _percentile(bootstrapped_means, alpha),
        "ci_high": _percentile(bootstrapped_means, 1.0 - alpha),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def paired_permutation_test(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    n_permutations: int = 10_000,
    alternative: str = "two-sided",
    seed: int | None = 42,
) -> dict[str, float | int | str | bool]:
    """Return a paired sign-flip permutation test for candidate-baseline mean delta."""
    baseline = _float_sequence(baseline_scores, name="baseline_scores")
    candidate = _float_sequence(candidate_scores, name="candidate_scores")
    if len(baseline) != len(candidate):
        raise ValueError("paired scores must have the same non-zero length")
    _validate_resample_count(n_permutations, name="n_permutations")
    _validate_alternative(alternative)

    differences = [candidate_score - baseline_score for baseline_score, candidate_score in zip(baseline, candidate)]
    observed_delta = _mean(differences)
    exact_assignment_count = 1 << len(differences)
    if exact_assignment_count <= n_permutations:
        p_value = _exact_sign_flip_p_value(differences, observed_delta, alternative)
        permutations_used = exact_assignment_count
        exact = True
    else:
        p_value = _random_sign_flip_p_value(
            differences,
            observed_delta,
            alternative,
            n_permutations=n_permutations,
            seed=seed,
        )
        permutations_used = n_permutations
        exact = False

    return {
        "n": len(differences),
        "mean_delta": observed_delta,
        "p_value": p_value,
        "alternative": alternative,
        "n_permutations": permutations_used,
        "exact": exact,
    }


def _deduped_retrieval_metrics(
    predicted_ids: list[str],
    gold_ids: list[str],
    k: int,
) -> dict[str, float]:
    top_predictions = _top_k(predicted_ids, k)
    unique_predictions = _unique_in_order(top_predictions)
    gold_set = set(gold_ids)
    matched_count = sum(1 for predicted_id in unique_predictions if predicted_id in gold_set)

    precision = matched_count / len(unique_predictions) if unique_predictions else 0.0
    recall = matched_count / len(gold_set) if gold_set else 0.0
    f1 = _f1(precision, recall)

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "hit_at_k": 1.0 if matched_count else 0.0,
    }


def _top_k(ids: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    return ids[:k]


def _unique_in_order(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    for item_id in ids:
        if item_id not in seen:
            seen.add(item_id)
            unique_ids.append(item_id)
    return unique_ids


def _first_match_rank(predicted_ids: list[str], gold_set: set[str]) -> Optional[int]:
    if not gold_set:
        return None
    for index, predicted_id in enumerate(predicted_ids, start=1):
        if predicted_id in gold_set:
            return index
    return None


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _float_sequence(values: Sequence[float], *, name: str) -> list[float]:
    scores: list[float] = []
    for value in values:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"{name} must contain only finite values")
        scores.append(score)
    if not scores:
        raise ValueError(f"{name} must contain at least one value")
    return scores


def _validate_resample_count(count: int, *, name: str) -> None:
    if count <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


def _validate_alternative(alternative: str) -> None:
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * fraction


def _exact_sign_flip_p_value(
    differences: Sequence[float],
    observed_delta: float,
    alternative: str,
) -> float:
    total = 1 << len(differences)
    extreme_count = 0
    for mask in range(total):
        permuted_delta = _mean(
            [
                difference if mask & (1 << index) else -difference
                for index, difference in enumerate(differences)
            ]
        )
        if _is_extreme(permuted_delta, observed_delta, alternative):
            extreme_count += 1
    return extreme_count / total


def _random_sign_flip_p_value(
    differences: Sequence[float],
    observed_delta: float,
    alternative: str,
    *,
    n_permutations: int,
    seed: int | None,
) -> float:
    rng = random.Random(seed)
    extreme_count = 1
    for _ in range(n_permutations):
        permuted_delta = _mean(
            [
                difference if rng.random() < 0.5 else -difference
                for difference in differences
            ]
        )
        if _is_extreme(permuted_delta, observed_delta, alternative):
            extreme_count += 1
    return extreme_count / (n_permutations + 1)


def _is_extreme(permuted_delta: float, observed_delta: float, alternative: str) -> bool:
    tolerance = 1e-12
    if alternative == "greater":
        return permuted_delta >= observed_delta - tolerance
    if alternative == "less":
        return permuted_delta <= observed_delta + tolerance
    return abs(permuted_delta) >= abs(observed_delta) - tolerance


__all__ = [
    "source_metrics",
    "entity_metrics",
    "edge_metrics",
    "path_exact_match",
    "path_endpoint_recall",
    "path_edge_sequence_recall",
    "bootstrap_mean_ci",
    "paired_permutation_test",
]
