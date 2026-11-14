"""C4 postprocess saturation pipeline (D33 / H8).

This module implements the 7-step keyword saturation pipeline described in
the project-design post-processing saturation rule. Given a ranked list of retrieval
candidates plus the KG nodes/edges and the original query, each step extends
the candidate's ``coverage_terms`` set with additional keywords drawn from a
different evidence channel:

    Step 1 - source-term injection (paper title + abstract tokens)
    Step 2 - alias-term injection (matched node aliases)
    Step 3 - related-entity-term injection (1-hop KG neighbours)
    Step 4 - full-name expansion (canonical node names for matched IDs)
    Step 5 - conceptual translation (caller-supplied lexicon dictionary)
    Step 6 - edge-context concept (relation_type tokens of incident edges)
    Step 7 - source-doc concept (paper-level descriptor tokens)

Each step is a pure function that returns the updated candidate list and a
per-candidate ``coverage_gain`` increment so callers can attribute the lift
in §6.4-style analysis. After step 7 the saturator recomputes a Jaccard
``coverage_score`` against the *query-relevant* term set and produces
``final_score' = final_score + lambda * coverage_score`` (lambda = 0.10 by
default; configurable).

All logic lives in this module - the retriever just toggles the call.

Reference: project design; MASTER_PLAN W7; DECISIONS.md D33.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from prototype.baselines.bm25_retriever import tokenize


DEFAULT_LAMBDA = 0.10
"""Mixing coefficient for the re-rank bonus added on top of the V3 score."""

DEFAULT_STEPS = 7
"""Number of saturation steps in the retrieval pipeline."""

_GENERIC_TOKENS = frozenset({
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for",
    "by", "as", "with", "is", "are", "be", "this", "that", "from",
    "what", "which", "how", "why", "when", "where", "who",
    "do", "does", "did", "can", "will", "may", "would", "should",
    "process", "method", "system", "result", "effect", "value",
})


@dataclass
class SaturatedCandidate:
    """A single candidate enriched by the 7-step pipeline.

    Mirrors the dict shape produced by ``ProposedV3Retriever.query`` so we
    can hand the result straight back to the runner without conversion.
    """

    candidate: dict[str, Any]
    coverage_terms: set[str] = field(default_factory=set)
    coverage_gain_per_step: list[int] = field(default_factory=list)
    final_score: float = 0.0
    coverage_score: float = 0.0
    bonus: float = 0.0


def saturate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    query: str,
    nodes: Sequence[Mapping[str, Any]] | None = None,
    edges: Sequence[Mapping[str, Any]] | None = None,
    papers_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    lexicon: Mapping[str, Sequence[str]] | None = None,
    nodes_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]] | None = None,
    steps: int = DEFAULT_STEPS,
    lambda_bonus: float = DEFAULT_LAMBDA,
) -> list[dict[str, Any]]:
    """Run the 7-step saturation pipeline and re-rank.

    Parameters
    ----------
    candidates:
        Ranked list of candidate dicts emitted by ``ProposedV3Retriever``.
        Must expose ``"score"``, ``"paper"``, and ``"diagnostics"`` keys.
    query:
        Original natural-language question (used to derive query-relevant
        terms).
    nodes / edges:
        Full KG node and edge lists. Edges are used in step 6 only.
    papers_by_id:
        Lookup table for source-doc tokens (step 7).
    lexicon:
        Optional concept translation dictionary (step 5). ``None`` skips
        the step but still records a coverage_gain entry of ``0`` so the
        diagnostics array length stays stable.
    nodes_by_id / adjacency:
        Optional pre-built lookups; computed locally if absent. Letting
        callers pass these avoids redundant work when the saturator is
        invoked once per query.
    steps:
        How many of the 7 steps to apply. The default and only fully
        supported value is 7; lowering it disables the trailing steps and
        is intended for unit tests.
    lambda_bonus:
        Mixing weight for the coverage bonus added to the V3 score.

    Returns
    -------
    A new list of candidate dicts with ``"score"`` updated and a fresh
    ``"diagnostics.c4_postprocess"`` block describing the gain per step.
    The order is sorted by the bonus-adjusted score, then by paper id for
    deterministic ties.
    """
    if not candidates:
        return []
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if steps > DEFAULT_STEPS:
        raise ValueError(f"steps must be <= {DEFAULT_STEPS}, got {steps}")

    nodes_by_id = nodes_by_id or _build_nodes_by_id(nodes or [])
    adjacency = adjacency or _build_adjacency(edges or [])

    query_terms = _query_relevant_terms(query)
    enriched: list[SaturatedCandidate] = []
    for original in candidates:
        item = SaturatedCandidate(
            candidate=dict(original),
            final_score=float(original.get("score") or 0.0),
        )
        enriched.append(item)

    pipeline = (
        _step1_source_terms,
        _step2_alias_terms,
        _step3_related_entity_terms,
        _step4_full_name_expansion,
        _step5_conceptual_translation,
        _step6_edge_context_concept,
        _step7_source_doc_concept,
    )

    for step_index in range(steps):
        step_fn = pipeline[step_index]
        for item in enriched:
            before = len(item.coverage_terms)
            step_fn(
                item,
                nodes_by_id=nodes_by_id,
                adjacency=adjacency,
                papers_by_id=papers_by_id or {},
                lexicon=lexicon or {},
            )
            item.coverage_gain_per_step.append(len(item.coverage_terms) - before)

    # Re-rank using Jaccard coverage between query-relevant terms and the
    # accumulated coverage set.
    for item in enriched:
        item.coverage_score = _jaccard(query_terms, item.coverage_terms)
        item.bonus = lambda_bonus * item.coverage_score

    enriched.sort(
        key=lambda i: (
            -(i.final_score + i.bonus),
            i.candidate.get("paper", {}).get("id")
            or i.candidate.get("paper", {}).get("paper_id")
            or "",
        )
    )

    out: list[dict[str, Any]] = []
    for rank, item in enumerate(enriched, start=1):
        new_score = item.final_score + item.bonus
        new_candidate = dict(item.candidate)
        new_candidate["score"] = float(new_score)
        new_candidate["rank"] = rank
        diag = dict(new_candidate.get("diagnostics") or {})
        diag["c4_postprocess"] = {
            "coverage_terms": sorted(item.coverage_terms),
            "coverage_gain_per_step": list(item.coverage_gain_per_step),
            "coverage_score": item.coverage_score,
            "bonus": item.bonus,
            "lambda": lambda_bonus,
            "steps": steps,
        }
        new_candidate["diagnostics"] = diag
        out.append(new_candidate)
    return out


# ---------------------------------------------------------------------------
# Saturation steps (pure functions on a SaturatedCandidate).
# ---------------------------------------------------------------------------


def _step1_source_terms(
    item: SaturatedCandidate,
    **_: Any,
) -> None:
    """Step 1: inject source-document title and abstract tokens."""
    paper = item.candidate.get("paper") or {}
    blob = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
    item.coverage_terms.update(_content_tokens(blob))


def _step2_alias_terms(
    item: SaturatedCandidate,
    *,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    **_: Any,
) -> None:
    """Step 2: inject matched-node aliases as additional surface forms."""
    diag = item.candidate.get("diagnostics") or {}
    alias_node_ids = diag.get("alias_nodes") or diag.get("matched_nodes") or []
    for node_id in alias_node_ids:
        node = nodes_by_id.get(str(node_id))
        if node is None:
            continue
        for alias in node.get("aliases") or []:
            item.coverage_terms.update(_content_tokens(str(alias)))


def _step3_related_entity_terms(
    item: SaturatedCandidate,
    *,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    **_: Any,
) -> None:
    """Step 3: inject 1-hop KG neighbour names (related entities)."""
    diag = item.candidate.get("diagnostics") or {}
    seed_ids = list(diag.get("matched_nodes") or []) + list(diag.get("alias_nodes") or [])
    seen: set[str] = set()
    for seed in seed_ids:
        for neighbor_id, _edge in adjacency.get(str(seed), []):
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            neighbor = nodes_by_id.get(neighbor_id)
            if neighbor is None:
                continue
            item.coverage_terms.update(_content_tokens(str(neighbor.get("name") or "")))


def _step4_full_name_expansion(
    item: SaturatedCandidate,
    *,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    **_: Any,
) -> None:
    """Step 4: expand matched-node IDs to their canonical full names."""
    diag = item.candidate.get("diagnostics") or {}
    for node_id in diag.get("matched_nodes") or []:
        node = nodes_by_id.get(str(node_id))
        if node is None:
            continue
        item.coverage_terms.update(_content_tokens(str(node.get("name") or "")))


def _step5_conceptual_translation(
    item: SaturatedCandidate,
    *,
    lexicon: Mapping[str, Sequence[str]],
    **_: Any,
) -> None:
    """Step 5: translate existing coverage terms via the concept lexicon.

    The lexicon maps a surface token to a tuple of related concept tokens
    (e.g., ``"plasma" -> ("ion", "discharge", "etching")``). We extend the
    coverage set with all translations for tokens already covered by an
    earlier step. No model call is made.
    """
    new_terms: set[str] = set()
    for term in list(item.coverage_terms):
        translations = lexicon.get(term)
        if not translations:
            continue
        for translation in translations:
            new_terms.update(_content_tokens(str(translation)))
    item.coverage_terms.update(new_terms)


def _step6_edge_context_concept(
    item: SaturatedCandidate,
    **_: Any,
) -> None:
    """Step 6: ingest the diagnostics' edge_relations as concept tokens."""
    diag = item.candidate.get("diagnostics") or {}
    for relation in diag.get("edge_relations") or []:
        item.coverage_terms.update(_content_tokens(str(relation).replace("_", " ")))


def _step7_source_doc_concept(
    item: SaturatedCandidate,
    *,
    papers_by_id: Mapping[str, Mapping[str, Any]],
    **_: Any,
) -> None:
    """Step 7: pull paper-level descriptor tokens (keywords / domain tags)."""
    paper = item.candidate.get("paper") or {}
    paper_id = paper.get("id") or paper.get("paper_id")
    descriptor_source = papers_by_id.get(str(paper_id)) if paper_id else None
    descriptor_source = descriptor_source or paper

    for key in ("keywords", "tags", "domain", "subject", "categories"):
        value = descriptor_source.get(key)
        if isinstance(value, str):
            item.coverage_terms.update(_content_tokens(value))
        elif isinstance(value, Iterable):
            for entry in value:
                item.coverage_terms.update(_content_tokens(str(entry)))


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in _GENERIC_TOKENS}


def _query_relevant_terms(query: str) -> set[str]:
    return _content_tokens(query)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _build_nodes_by_id(
    nodes: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        raw_id = node.get("node_id") or node.get("id")
        if raw_id:
            out[str(raw_id)] = node
    return out


def _build_adjacency(
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[str, Mapping[str, Any]]]]:
    adjacency: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for edge in edges:
        source = edge.get("source") or edge.get("head") or edge.get("from")
        target = edge.get("target") or edge.get("tail") or edge.get("to")
        if not source or not target:
            continue
        adjacency.setdefault(str(source), []).append((str(target), edge))
        adjacency.setdefault(str(target), []).append((str(source), edge))
    return adjacency
