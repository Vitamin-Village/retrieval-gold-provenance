"""Consolidated retrieval and KG-typing metrics for the GraphRAG paper.

Contracts are derived from prototype/eval/metrics_v3.py and
prototype/eval/run_retrieval_eval_v3.py.  This module is the canonical
paper-side version — do NOT modify the prototype originals.

Exported functions:
    node_typing_prf1      — precision / recall / F1 over KGNode type labels
    edge_typing_prf1      — precision / recall / F1 over edge relation_type labels
    gold_coverage         — fraction of gold entity names found in predicted nodes
    hit_at_k              — 0/1 indicator: gold_id in top-k of predicted_rank
    mrr                   — reciprocal rank of gold_id in predicted_rank

KGNode is a lightweight dataclass mirroring paper.method.entity_alignment_5d.KGNode
so this module stays self-contained (no circular imports required).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KGNode:
    """Minimal node record used for typing metrics."""

    node_id: str
    name: str
    node_type: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    source_docs: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Node-typing metrics
# ---------------------------------------------------------------------------


def node_typing_prf1(
    predicted: list[KGNode],
    gold: list[KGNode],
) -> dict[str, float]:
    """Precision / recall / F1 on *node_type* labels.

    Matching is done by (node_id, node_type) pairs.  A predicted node is
    correct when its node_id appears in gold **with the same node_type**.

    Args:
        predicted: Predicted list of KGNodes (may contain duplicates by id).
        gold: Gold list of KGNodes.

    Returns:
        dict with keys: precision, recall, f1, n_predicted, n_gold, n_correct.
    """
    gold_pairs: set[tuple[str, str]] = {
        (n.node_id, n.node_type) for n in gold
    }
    pred_pairs: set[tuple[str, str]] = {
        (n.node_id, n.node_type) for n in predicted
    }

    n_correct = len(pred_pairs & gold_pairs)
    precision = n_correct / len(pred_pairs) if pred_pairs else 0.0
    recall = n_correct / len(gold_pairs) if gold_pairs else 0.0
    f1 = _f1(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": len(pred_pairs),
        "n_gold": len(gold_pairs),
        "n_correct": n_correct,
    }


# ---------------------------------------------------------------------------
# Edge-typing metrics
# ---------------------------------------------------------------------------


def edge_typing_prf1(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> dict[str, float]:
    """Precision / recall / F1 on *relation_type* labels for edges.

    Each edge dict must contain at minimum: source_id (or source/head/from),
    target_id (or target/tail/to), and relation_type (or relation).

    A predicted edge is correct when its (source_id, target_id, relation_type)
    triple matches a gold triple exactly.

    Args:
        predicted: List of predicted edge dicts.
        gold: List of gold edge dicts.

    Returns:
        dict with keys: precision, recall, f1, n_predicted, n_gold, n_correct.
    """
    gold_triples: set[tuple[str, str, str]] = {_edge_triple(e) for e in gold}
    pred_triples: set[tuple[str, str, str]] = {_edge_triple(e) for e in predicted}

    n_correct = len(pred_triples & gold_triples)
    precision = n_correct / len(pred_triples) if pred_triples else 0.0
    recall = n_correct / len(gold_triples) if gold_triples else 0.0
    f1 = _f1(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": len(pred_triples),
        "n_gold": len(gold_triples),
        "n_correct": n_correct,
    }


# ---------------------------------------------------------------------------
# Gold coverage
# ---------------------------------------------------------------------------


def gold_coverage(
    predicted_nodes: list[KGNode],
    gold_entities: list[str],
) -> float:
    """Fraction of gold entity *names* present in predicted node names/aliases.

    Coverage is case-insensitive.  An entity is covered when any predicted
    node's name or alias matches the gold entity name (after lowercasing).

    Args:
        predicted_nodes: Predicted KGNode list.
        gold_entities: List of gold entity name strings.

    Returns:
        Coverage in [0, 1]; 0.0 when gold_entities is empty.
    """
    if not gold_entities:
        return 0.0

    covered_names: set[str] = set()
    for node in predicted_nodes:
        covered_names.add(node.name.lower())
        for alias in node.aliases:
            covered_names.add(alias.lower())

    covered = sum(1 for e in gold_entities if e.lower() in covered_names)
    return covered / len(gold_entities)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def hit_at_k(
    predicted_rank: list[str],
    gold_id: str,
    k: int = 5,
) -> int:
    """Binary hit indicator: 1 if gold_id appears in top-k, else 0.

    Args:
        predicted_rank: Ordered list of predicted IDs (best first).
        gold_id: The target ID to look for.
        k: Cutoff rank (inclusive).

    Returns:
        1 or 0.
    """
    if k <= 0:
        return 0
    return int(gold_id in predicted_rank[:k])


def mrr(predicted_rank: list[str], gold_id: str) -> float:
    """Mean Reciprocal Rank contribution for a single query.

    Returns 1/rank if gold_id is found, else 0.0.  Rank is 1-based.

    Args:
        predicted_rank: Ordered list of predicted IDs (best first).
        gold_id: The target ID to look for.
    """
    for rank, pid in enumerate(predicted_rank, start=1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _edge_triple(edge: dict[str, Any]) -> tuple[str, str, str]:
    source = str(
        edge.get("source_id") or edge.get("source")
        or edge.get("head") or edge.get("from") or ""
    ).strip()
    target = str(
        edge.get("target_id") or edge.get("target")
        or edge.get("tail") or edge.get("to") or ""
    ).strip()
    relation = str(
        edge.get("relation_type") or edge.get("relation") or ""
    ).strip()
    return source, target, relation


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


__all__ = [
    "KGNode",
    "node_typing_prf1",
    "edge_typing_prf1",
    "gold_coverage",
    "hit_at_k",
    "mrr",
]
