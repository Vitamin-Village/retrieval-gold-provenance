"""C6 4-axis evidence-grounded trust (D35 / H10).

The current retriever already combines three of the four project-design axes
inside ``ProposedV3Retriever._trust_signal`` (source diversity ->
Triangulation, reliability -> partial Authority, context tokens ->
Context Sufficiency). This module:

    - exposes the four axes as independent functions so each can be unit
      tested in isolation,
    - adds the missing fourth axis ("Query-Graph Alignment") as a
      Jaccard between query-mention entity IDs and the candidate's KG
      subgraph nodes, and
    - replaces the ad-hoc weighted sum with the formal four-axis design
      definition (uniform weights, structural -- not tuned).

The OFF flag (cell C11, ``single_source_weight``) collapses back to the
existing ``_trust_signal`` behaviour so the ablation isolates the delta.

Reference: project design; MASTER_PLAN W7; DECISIONS.md D35.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from prototype.baselines.bm25_retriever import tokenize


CONTEXT_TOKEN_TARGET = 80
"""Soft cap used by Axis 4 (Context Sufficiency). Same value as the legacy
``_trust_signal`` so the cell C11 OFF baseline reproduces today's number.
"""

UNIFORM_WEIGHTS: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
"""The four-axis design prescribes uniform aggregation. The weights
are STRUCTURAL (a rule), not tuned, so the cell C11 vs full delta isolates
the axis-set change without confounding it with a weight search.
"""


@dataclass(frozen=True)
class Trust4Score:
    """Per-axis breakdown for a single candidate.

    The aggregate field is the uniform mean of the four axes, exposed
    separately so the runner can write all four sub-scores to the
    per-record diagnostics in addition to the scalar trust signal.
    """

    a1_source_triangulation: float
    a2_query_graph_alignment: float
    a3_entity_authority: float
    a4_context_sufficiency: float
    aggregate: float


def score(
    *,
    candidate_sources: set[str],
    candidate_entity_ids: set[str],
    query_entity_ids: set[str],
    paper_text: str,
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]] | None = None,
    edge_reliability: float = 0.0,
    has_direct_match: bool = False,
    weights: tuple[float, float, float, float] = UNIFORM_WEIGHTS,
) -> Trust4Score:
    """Compute the 4-axis trust score for a single candidate.

    Parameters
    ----------
    candidate_sources:
        Channels that contributed this candidate (``"bm25"``, ``"dense"``,
        ``"kg"``). Used by Axis 1.
    candidate_entity_ids:
        KG node IDs attached to the candidate (matched + neighbour). Used
        by Axes 2 and 3.
    query_entity_ids:
        KG node IDs the query mentioned (the matched-nodes set). Used by
        Axis 2.
    paper_text:
        Title + abstract concatenated. Used by Axis 4.
    adjacency:
        KG adjacency map. Used by Axis 3 (node degree).
    edge_reliability:
        Maximum reliability score from incident edges. Used as a fallback
        in Axis 3 when the adjacency is empty.
    has_direct_match:
        Whether the candidate has at least one direct (non-alias)
        node hit. Mirrors the legacy fallback used by ``_trust_signal``.
    weights:
        Per-axis weights. Defaults to the project-design uniform vector.
    """
    a1 = source_triangulation(candidate_sources)
    a2 = query_graph_alignment(query_entity_ids, candidate_entity_ids)
    a3 = entity_authority(
        candidate_entity_ids,
        adjacency or {},
        fallback=edge_reliability,
        has_direct_match=has_direct_match,
    )
    a4 = context_sufficiency(paper_text)

    aggregate = _weighted_sum((a1, a2, a3, a4), weights)
    return Trust4Score(
        a1_source_triangulation=a1,
        a2_query_graph_alignment=a2,
        a3_entity_authority=a3,
        a4_context_sufficiency=a4,
        aggregate=aggregate,
    )


# ---------------------------------------------------------------------------
# Individual axes.
# ---------------------------------------------------------------------------


def source_triangulation(candidate_sources: Iterable[str]) -> float:
    """Axis 1: how many independent retrieval channels backed the candidate.

    Bounded to [0, 1] by dividing by 3 (the maximum number of channels:
    ``bm25``, ``dense``, ``kg``). A single-channel candidate scores 0.33,
    two channels 0.67, all three 1.0.
    """
    sources = {str(item) for item in candidate_sources}
    return min(len(sources) / 3.0, 1.0)


def query_graph_alignment(
    query_entity_ids: Iterable[str],
    candidate_entity_ids: Iterable[str],
) -> float:
    """Axis 2: Jaccard between query-mention entities and candidate entities.

    Returns 0 when either set is empty so a candidate without KG
    grounding cannot inherit alignment credit from a noisy query.
    """
    query_set = {str(item) for item in query_entity_ids}
    candidate_set = {str(item) for item in candidate_entity_ids}
    if not query_set or not candidate_set:
        return 0.0
    union = query_set | candidate_set
    return len(query_set & candidate_set) / len(union)


def entity_authority(
    candidate_entity_ids: Iterable[str],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    *,
    fallback: float = 0.0,
    has_direct_match: bool = False,
) -> float:
    """Axis 3: normalised mean in-degree of the candidate's KG entities.

    Computes ``mean(deg(node)) / max_degree`` over the candidate's
    entity set so the score is bounded to [0, 1]. When the adjacency is
    empty (e.g., during an ablation that strips the KG), the function
    falls back to the legacy reliability proxy used by ``_trust_signal``:
    return ``fallback`` clamped to [0, 1], promoted to 0.5 when a direct
    match exists but reliability is unset.
    """
    candidate_set = {str(item) for item in candidate_entity_ids}
    if not candidate_set or not adjacency:
        proxy = max(0.0, min(fallback, 1.0))
        if has_direct_match and proxy == 0.0:
            return 0.5
        return proxy

    degrees = {node_id: len(adjacency.get(node_id, [])) for node_id in adjacency}
    max_degree = max(degrees.values(), default=0)
    if max_degree <= 0:
        # All isolated nodes -> defer to the legacy fallback path.
        proxy = max(0.0, min(fallback, 1.0))
        if has_direct_match and proxy == 0.0:
            return 0.5
        return proxy

    candidate_degrees = [
        degrees.get(node_id, 0) for node_id in candidate_set
    ]
    if not candidate_degrees:
        return 0.0
    mean_degree = sum(candidate_degrees) / len(candidate_degrees)
    return min(mean_degree / max_degree, 1.0)


def context_sufficiency(paper_text: str) -> float:
    """Axis 4: how much textual context surrounds the candidate.

    Reproduces the legacy ``min(len(tokens) / 80, 1.0)`` measure so the
    cell C11 OFF baseline collapses cleanly to today's number.
    """
    tokens = tokenize(paper_text or "")
    return min(len(tokens) / CONTEXT_TOKEN_TARGET, 1.0)


# ---------------------------------------------------------------------------
# Legacy fallback for the OFF flag.
# ---------------------------------------------------------------------------


def legacy_single_source_weight(
    *,
    candidate_sources: Iterable[str],
    paper_text: str,
    edge_reliability: float,
    has_direct_match: bool,
) -> float:
    """Reproduce ``ProposedV3Retriever._trust_signal`` exactly.

    Used by cell C11 (``c6_trust=single_source_weight``). Mirrors the
    three-term weighted sum at proposed_v3_retriever.py lines 601-615
    bit-for-bit so a numerical regression test can pin the parity.
    """
    sources = {str(item) for item in candidate_sources}
    source_diversity = min(len(sources) / 3.0, 1.0)
    tokens = tokenize(paper_text or "")
    sufficiency = min(len(tokens) / 80.0, 1.0)
    reliability = min(edge_reliability, 1.0)
    if has_direct_match and reliability == 0.0:
        reliability = 0.5
    return min(
        1.0,
        0.40 * source_diversity
        + 0.25 * sufficiency
        + 0.35 * reliability,
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _weighted_sum(
    values: tuple[float, float, float, float],
    weights: tuple[float, float, float, float],
) -> float:
    if len(weights) != 4:
        raise ValueError(f"weights must have length 4, got {len(weights)}")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weight sum must be > 0")
    return min(
        1.0,
        sum(value * weight for value, weight in zip(values, weights)) / weight_sum,
    )
