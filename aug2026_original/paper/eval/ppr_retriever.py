# -*- coding: utf-8 -*-
"""Standalone Personalized PageRank (PPR) retriever for KG-swap experiments.

Mechanistically independent from ProposedV3Retriever:
- No BM25 / dense fusion.
- No imports from prototype.baselines.proposed_v3_retriever.
- Stdlib only (no third-party deps beyond what Python ships with).

Interface matches eval_independent.retrieve_top_ids expectation:
    retriever.query(question, top_k) -> list[{"paper": paper_dict, "score": float}]
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """casefold + collapse whitespace."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


_WIKIDATA_RE = re.compile(r"^q\d+$")


def _is_wikidata_id(s: str) -> bool:
    return bool(_WIKIDATA_RE.match(s))


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("id", "paper_id", "openalex_id", "source_id", "doc_id", "arxiv_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


# ---------------------------------------------------------------------------
# PPRRetriever
# ---------------------------------------------------------------------------

class PPRRetriever:
    """Rank papers via Personalized PageRank on the entity KG.

    Parameters
    ----------
    papers:
        List of paper dicts (same list passed to other retrievers).
    nodes:
        List of KG node dicts from LoadedKg.nodes.
    edges:
        List of KG edge dicts from LoadedKg.edges.
    damping:
        PPR damping factor (alpha). Default 0.85.
    max_iter:
        Maximum power-iteration steps. Default 50.
    tol:
        L1 convergence tolerance. Default 1e-8.
    """

    def __init__(
        self,
        papers: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        *,
        damping: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> None:
        self._damping = damping
        self._max_iter = max_iter
        self._tol = tol

        # Build paper id set for filtering output
        self._papers: list[dict[str, Any]] = papers
        self._paper_ids: set[str] = {_record_id(p) for p in papers if _record_id(p)}

        # Index nodes by node_id
        self._nodes: list[dict[str, Any]] = nodes
        self._node_ids: list[str] = [n.get("node_id", "") for n in nodes]
        self._node_index: dict[str, int] = {
            nid: i for i, nid in enumerate(self._node_ids) if nid
        }
        n = len(self._node_ids)

        # Build undirected adjacency list (uniform transition)
        adj: list[list[int]] = [[] for _ in range(n)]
        for edge in edges:
            src = edge.get("source_id") or edge.get("source", "")
            tgt = edge.get("target_id") or edge.get("target", "")
            si = self._node_index.get(src, -1)
            ti = self._node_index.get(tgt, -1)
            if si >= 0 and ti >= 0 and si != ti:
                adj[si].append(ti)
                adj[ti].append(si)

        # Deduplicate adjacency entries
        self._adj: list[list[int]] = [list(dict.fromkeys(lst)) for lst in adj]

        # Precompute transition matrix rows as (neighbor_index, weight) pairs.
        # Dangling nodes (no out-edges in undirected sense) contribute mass
        # back to the personalization vector uniformly.
        self._trans: list[list[tuple[int, float]]] = []
        for i in range(n):
            neighbors = self._adj[i]
            if neighbors:
                w = 1.0 / len(neighbors)
                self._trans.append([(j, w) for j in neighbors])
            else:
                self._trans.append([])

        # Precompute: for each paper_id, which node indices have it in source_docs
        self._paper_to_nodes: dict[str, list[int]] = {}
        for i, node in enumerate(nodes):
            for doc_id in node.get("source_docs", []):
                if doc_id:
                    self._paper_to_nodes.setdefault(doc_id, []).append(i)

        # Precompute normalized surface forms for entity seeding
        # Each entry: (node_index, set_of_normalized_surfaces)
        self._node_surfaces: list[tuple[int, list[str]]] = []
        for i, node in enumerate(nodes):
            surfaces: list[str] = []
            name_norm = _norm(node.get("name", ""))
            if name_norm:
                surfaces.append(name_norm)
            for alias in node.get("aliases", []):
                alias_norm = _norm(alias)
                # Skip bare Wikidata ids and very short strings
                if len(alias_norm) >= 3 and not _is_wikidata_id(alias_norm):
                    surfaces.append(alias_norm)
            if surfaces:
                self._node_surfaces.append((i, surfaces))

    # ------------------------------------------------------------------
    # Seed matching
    # ------------------------------------------------------------------

    def _seed_indices(self, question_norm: str) -> list[int]:
        """Return node indices whose name/alias appears as whole-word in question."""
        seeds: list[int] = []
        seen: set[int] = set()
        for idx, surfaces in self._node_surfaces:
            for surface in surfaces:
                # Build regex: word-boundary on both sides, escape surface
                escaped = re.escape(surface)
                pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
                if re.search(pattern, question_norm):
                    if idx not in seen:
                        seen.add(idx)
                        seeds.append(idx)
                    break
        return seeds

    # ------------------------------------------------------------------
    # PPR power iteration
    # ------------------------------------------------------------------

    def _ppr(self, seed_indices: list[int]) -> list[float]:
        n = len(self._node_ids)
        if n == 0:
            return []

        # Personalization vector: uniform over seeds. Defensive dedup so a
        # caller-supplied duplicate seed cannot shrink the vector's total
        # mass below 1 (audit F2).
        seed_indices = list(dict.fromkeys(seed_indices))
        pers = [0.0] * n
        seed_weight = 1.0 / len(seed_indices)
        for i in seed_indices:
            pers[i] = seed_weight

        # Initial distribution = personalization
        p = list(pers)

        alpha = self._damping  # probability of following an edge
        teleport = 1.0 - alpha

        for _ in range(self._max_iter):
            p_next = [0.0] * n

            # Accumulate dangling mass (nodes with no out-edges)
            dangling_mass = 0.0
            for i in range(n):
                if not self._trans[i]:
                    dangling_mass += p[i]

            # Distribute dangling mass back via personalization
            dangling_contrib = dangling_mass * alpha

            for i in range(n):
                # Teleportation + dangling redistribution
                p_next[i] += teleport * pers[i] + dangling_contrib * pers[i]

            # Random walk contributions
            for i in range(n):
                if self._trans[i]:
                    for j, w in self._trans[i]:
                        p_next[j] += alpha * p[i] * w

            # Check L1 convergence
            l1 = sum(abs(p_next[i] - p[i]) for i in range(n))
            p = p_next
            if l1 < self._tol:
                break

        return p

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top_k papers ranked by PPR mass.

        Parameters
        ----------
        question:
            Natural language question.
        top_k:
            Number of results to return.

        Returns
        -------
        list of {"paper": paper_dict, "score": float}, sorted by score desc.
        """
        question_norm = _norm(question)
        seeds = self._seed_indices(question_norm)
        if not seeds:
            return []

        ppr_scores = self._ppr(seeds)

        # Aggregate PPR mass per paper_id
        paper_scores: dict[str, float] = {}
        for pid, node_indices in self._paper_to_nodes.items():
            if pid not in self._paper_ids:
                continue
            mass = sum(ppr_scores[i] for i in node_indices if i < len(ppr_scores))
            if mass > 0.0:
                paper_scores[pid] = mass

        if not paper_scores:
            return []

        # Sort by score desc, tie-break by paper_id asc
        ranked_ids = sorted(
            paper_scores.keys(),
            key=lambda pid: (-paper_scores[pid], pid),
        )

        # Build result list from papers list (preserve original paper dict)
        pid_to_paper: dict[str, dict[str, Any]] = {
            _record_id(p): p for p in self._papers if _record_id(p)
        }

        results: list[dict[str, Any]] = []
        for pid in ranked_ids[:top_k]:
            paper = pid_to_paper.get(pid)
            if paper is None:
                continue
            results.append({"paper": paper, "score": paper_scores[pid]})

        return results
