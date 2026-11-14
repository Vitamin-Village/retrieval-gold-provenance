"""C5 query-type-adaptive multi-hop traversal (D34 / H9).

The retriever already classifies queries as causal / compare / define /
factual via ``ProposedV3Retriever._classify_query``. This module adds the
*traversal* delta documented in DECISIONS.md D34:

    causal  -> BFS along CAUSES_ISSUE / RESOLVED_BY / leads_to edges first
               (causal relations get priority over generic neighbours)
    compare -> enumerate common-vs-distinct attributes via
               SUBPROCESS_OF / IS_A / alternative_to edges between seed
               nodes
    define  -> 1-hop only (existing default for define)
    factual -> uniform BFS to depth 4

The retriever's ``_iter_neighbor_paths`` (line 322 in
proposed_v3_retriever.py) is replaced by ``traverse(...)`` when the
``adaptive`` strategy is selected. The OFF flag (uniform_BFS) is a thin
adapter that calls the existing depth-2 BFS.

Reference: project design; MASTER_PLAN W7; DECISIONS.md D34.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


QueryType = str  # "causal" | "compare" | "define" | "factual"


CAUSAL_RELATIONS: frozenset[str] = frozenset({
    "cause", "causes", "caused_by", "causes_by",
    "leads_to", "results_in",
    "resolves", "mitigates", "prevents",
    "causes_issue", "resolved_by",
})

COMPARE_RELATIONS: frozenset[str] = frozenset({
    "alternative_to", "compares_with", "contrasts_with", "substitutes",
    "is_a", "subprocess_of", "subclass_of",
})

# D45 / H9 typed-factual BFS. The factual branch previously degenerated
# into the same plain BFS as the OFF flag, so C10 produced an identical
# candidate set to C0. The typed-factual variant prioritises schema-bearing
# neighbours (Process / Material / Equipment / Issue) in a first pass and
# only then layers in remaining 1-hop neighbours, so the toggle changes the
# candidate pool composition (structural change, not a weight change).
FACTUAL_TYPED_PRIORITY: frozenset[str] = frozenset({
    "process",
    "material",
    "equipment",
    "issue",
})


_CAUSAL_RE = re.compile(r"\b(cause|causes|caused|why|leads?\s+to|due\s+to)\b")
_COMPARE_RE = re.compile(r"\b(compare|versus|vs\.?|alternative|difference)\b")
_DEFINE_RE = re.compile(r"\b(what\s+is|define|definition|describe)\b")


@dataclass(frozen=True)
class TraversalResult:
    """The neighbour set produced for a single query.

    Mirrors the shape of ``_iter_neighbor_paths`` (list of
    ``(neighbor_id, edge)`` tuples) so the retriever can use it as a
    drop-in replacement.
    """

    neighbors: list[tuple[str, Mapping[str, Any]]]
    query_type: QueryType
    notes: dict[str, Any]


def classify(question: str) -> QueryType:
    """Classify the question into causal/compare/define/factual.

    Wraps the same regex set used by the retriever so the two stay in
    lockstep. Pulled out into a function so unit tests can pin the
    expected behaviour independently of the retriever.
    """
    lowered = question.lower()
    if _CAUSAL_RE.search(lowered):
        return "causal"
    if _COMPARE_RE.search(lowered):
        return "compare"
    if _DEFINE_RE.search(lowered):
        return "define"
    return "factual"


def traverse(
    seed_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    query_type: QueryType,
    *,
    depth: int = 4,
    node_types: Mapping[str, str] | None = None,
) -> TraversalResult:
    """Adaptive multi-hop traversal keyed off ``query_type``.

    Parameters
    ----------
    seed_ids:
        Initial KG node IDs to expand from (the matched-node set).
    adjacency:
        Standard undirected adjacency map: ``node_id -> [(neighbor_id,
        edge_dict), ...]``.
    query_type:
        One of ``causal`` / ``compare`` / ``define`` / ``factual``.
    depth:
        Maximum BFS depth for ``factual``; causal uses ``min(depth, 4)``
        and define caps at 1 hop.
    node_types:
        Optional ``node_id -> schema_type`` map (lowercase strings such as
        ``"process"`` / ``"material"`` / ``"equipment"`` / ``"issue"`` /
        ``"source"``). When supplied, the factual branch runs a typed
        two-pass BFS (D45 / H9) that prioritises schema-bearing
        neighbours so the C10 OFF toggle (uniform_BFS) becomes a true
        structural baseline. When omitted, factual falls back to the
        legacy uniform BFS for backward compatibility.

    Returns
    -------
    A ``TraversalResult`` whose ``neighbors`` list mimics the return type
    of ``_iter_neighbor_paths`` so the retriever can splice it in.
    """
    if query_type == "causal":
        return _traverse_causal(seed_ids, adjacency, depth=min(depth, 4))
    if query_type == "compare":
        return _traverse_compare(seed_ids, adjacency)
    if query_type == "define":
        return _traverse_uniform_bfs(seed_ids, adjacency, depth=1, label="define")
    # factual: typed two-pass BFS when node_types are available (D45);
    # otherwise legacy uniform BFS so older callers keep working.
    if node_types is not None:
        return _traverse_factual_typed(
            seed_ids, adjacency, node_types, depth=depth,
        )
    return _traverse_uniform_bfs(seed_ids, adjacency, depth=depth, label="factual")


def uniform_bfs(
    seed_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    *,
    depth: int = 2,
) -> TraversalResult:
    """OFF-flag adapter: ignore query type and walk a uniform BFS.

    This reproduces the existing ``_iter_neighbor_paths(max_depth=2)``
    behaviour so cell C10 (query_strategy=uniform_BFS) is a true OFF
    baseline.
    """
    return _traverse_uniform_bfs(seed_ids, adjacency, depth=depth, label="uniform")


# ---------------------------------------------------------------------------
# Per-type traversals.
# ---------------------------------------------------------------------------


def _traverse_causal(
    seed_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    *,
    depth: int,
) -> TraversalResult:
    """BFS that prefers causal-typed edges first, then falls back to others.

    Two-pass over the adjacency from each seed: first emit every
    causal-relation neighbour, then layer in remaining 1-hop neighbours.
    This biases the candidate set toward the relation classes that
    underwrite causal claims without dropping the rest of the local
    graph.
    """
    visited: set[str] = set(seed_ids)
    neighbors: list[tuple[str, Mapping[str, Any]]] = []
    causal_hits = 0
    fallback_hits = 0

    # Phase 1: causal-edge BFS.
    frontier: deque[tuple[str, int]] = deque((s, 0) for s in seed_ids)
    while frontier:
        node_id, dist = frontier.popleft()
        if dist >= depth:
            continue
        for neighbor_id, edge in adjacency.get(node_id, []):
            if neighbor_id in visited:
                continue
            relation = str(edge.get("relation_type") or edge.get("relation") or "").lower()
            if relation not in CAUSAL_RELATIONS:
                continue
            visited.add(neighbor_id)
            neighbors.append((neighbor_id, edge))
            causal_hits += 1
            frontier.append((neighbor_id, dist + 1))

    # Phase 2: 1-hop fallback over remaining neighbours of the seed set.
    for seed in seed_ids:
        for neighbor_id, edge in adjacency.get(seed, []):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            neighbors.append((neighbor_id, edge))
            fallback_hits += 1

    return TraversalResult(
        neighbors=neighbors,
        query_type="causal",
        notes={"causal_hits": causal_hits, "fallback_hits": fallback_hits},
    )


def _traverse_compare(
    seed_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
) -> TraversalResult:
    """For each seed pair, emit the common / distinct 1-hop attributes.

    For seeds ``A`` and ``B``:
        - ``common``   = neighbours of A intersected with neighbours of B
        - ``distinct`` = symmetric difference of those neighbour sets
    Both are surfaced as ``(neighbor_id, edge)`` tuples, tagged with a
    note so downstream consumers can attribute the lift.
    """
    seed_neighbors: dict[str, dict[str, Mapping[str, Any]]] = {}
    for seed in seed_ids:
        nb_map: dict[str, Mapping[str, Any]] = {}
        for neighbor_id, edge in adjacency.get(seed, []):
            nb_map.setdefault(neighbor_id, edge)
        seed_neighbors[seed] = nb_map

    visited: set[str] = set(seed_ids)
    neighbors: list[tuple[str, Mapping[str, Any]]] = []
    common_hits = 0
    distinct_hits = 0

    seeds_list = list(seed_ids)
    for i in range(len(seeds_list)):
        for j in range(i + 1, len(seeds_list)):
            a, b = seeds_list[i], seeds_list[j]
            a_neighbors = seed_neighbors.get(a, {})
            b_neighbors = seed_neighbors.get(b, {})
            common = set(a_neighbors) & set(b_neighbors)
            distinct = (set(a_neighbors) ^ set(b_neighbors))
            for neighbor_id in common:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                neighbors.append((neighbor_id, a_neighbors[neighbor_id]))
                common_hits += 1
            for neighbor_id in distinct:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                edge = a_neighbors.get(neighbor_id) or b_neighbors.get(neighbor_id)
                if edge is None:
                    continue
                neighbors.append((neighbor_id, edge))
                distinct_hits += 1

    if len(seeds_list) == 1:
        # Degenerate single-seed compare: fall back to 1-hop (so the result
        # is still usable for downstream scoring).
        for neighbor_id, edge in adjacency.get(seeds_list[0], []):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            neighbors.append((neighbor_id, edge))

    return TraversalResult(
        neighbors=neighbors,
        query_type="compare",
        notes={"common_hits": common_hits, "distinct_hits": distinct_hits},
    )


def _traverse_factual_typed(
    seed_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    node_types: Mapping[str, str],
    *,
    depth: int,
) -> TraversalResult:
    """Two-pass typed BFS for factual queries (D45 / H9).

    Phase 1: BFS expansion that only enqueues neighbours whose schema type
    is in ``FACTUAL_TYPED_PRIORITY`` (Process / Material / Equipment /
    Issue). This biases the candidate pool toward the typed entity classes
    that actually carry factual answers, instead of letting Source nodes
    and untyped neighbours flood the pool at depth>=2.

    Phase 2: 1-hop fallback over any remaining neighbours of the seed set
    so the result remains a superset of a depth-1 plain BFS. This keeps
    recall safe even when a seed's local graph carries few priority-typed
    neighbours.

    Returns a ``TraversalResult`` with the same shape as
    ``_traverse_uniform_bfs``; the ``query_type`` label is
    ``factual_typed`` so downstream diagnostics can attribute the lift.
    """
    visited: set[str] = set(seed_ids)
    neighbors: list[tuple[str, Mapping[str, Any]]] = []
    typed_hits = 0
    fallback_hits = 0

    # Phase 1: typed BFS. Only schema-priority neighbours propagate.
    frontier: deque[tuple[str, int]] = deque((s, 0) for s in seed_ids)
    while frontier:
        node_id, dist = frontier.popleft()
        if dist >= depth:
            continue
        for neighbor_id, edge in adjacency.get(node_id, []):
            if neighbor_id in visited:
                continue
            nb_type = str(node_types.get(neighbor_id, "")).lower()
            if nb_type not in FACTUAL_TYPED_PRIORITY:
                continue
            visited.add(neighbor_id)
            neighbors.append((neighbor_id, edge))
            typed_hits += 1
            frontier.append((neighbor_id, dist + 1))

    # Phase 2: 1-hop fallback so untyped / Source / Parameter neighbours
    # of the seed set are still recoverable.
    for seed in seed_ids:
        for neighbor_id, edge in adjacency.get(seed, []):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            neighbors.append((neighbor_id, edge))
            fallback_hits += 1

    return TraversalResult(
        neighbors=neighbors,
        query_type="factual_typed",
        notes={
            "typed_hits": typed_hits,
            "fallback_hits": fallback_hits,
            "depth": depth,
            "priority_types": sorted(FACTUAL_TYPED_PRIORITY),
        },
    )


def _traverse_uniform_bfs(
    seed_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    *,
    depth: int,
    label: str,
) -> TraversalResult:
    """Plain BFS to a fixed depth - the OFF and factual/define behaviour."""
    visited: set[str] = set(seed_ids)
    neighbors: list[tuple[str, Mapping[str, Any]]] = []
    frontier: deque[tuple[str, int]] = deque((s, 0) for s in seed_ids)
    while frontier:
        node_id, dist = frontier.popleft()
        if dist >= depth:
            continue
        for neighbor_id, edge in adjacency.get(node_id, []):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            neighbors.append((neighbor_id, edge))
            frontier.append((neighbor_id, dist + 1))

    return TraversalResult(
        neighbors=neighbors,
        query_type=label,
        notes={"visited": len(visited), "depth": depth},
    )
