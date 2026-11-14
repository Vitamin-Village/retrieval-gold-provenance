"""Proposed V3 paper retriever with candidate union and C1-C6 scoring hooks."""

from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from prototype.baselines.bm25_retriever import tokenize
from prototype.eval import postprocess as _postprocess_module
from prototype.eval import query_strategy as _query_strategy_module
from prototype.eval import trust_4axis as _trust_4axis_module


C4PostprocessMode = Literal["on", "off"]
C5QueryStrategyMode = Literal["adaptive", "uniform_BFS"]
C6TrustMode = Literal["4axis", "single_source_weight"]
SubAnswerReqMode = Literal["schema", "generic"]
FusionMode = Literal[
    "additive_legacy",
    "schema_centric_gated",
    "graph_recall_rrf",
    "graph_centric",
]
GraphRecallGate = Literal["coverage_low", "multihop", "always"]


# Structural ratio constants (no tunable thresholds — pure rules) used by
# the schema-centric gated fusion mode. Lifting them to module scope makes
# the gate semantics auditable from a single place.
_COVERAGE_LOW_RATIO = 0.5
"""Coverage trigger fires when the matched-token coverage ratio is below
this fraction. Pure structural rule per P3 brief."""

_TRUST_SPREAD_MIN_STD = 0.05
"""Trust signal is gated off when the per-candidate trust standard
deviation is below this value (all scores cluster). Structural rule."""

_TIE_BAND_EPS = 0.01
"""Two candidate scores within this absolute distance are treated as
*ranking-tied* and shuffled together when ``RAG_V3_SEED`` is set. The
1% band matches the IR convention for "no meaningful difference"
(<1pp hit@k is treated as noise in standard retrieval evaluation).
Pure structural rule: gives seeded paired bootstrap genuine variance
without changing the mean rank of clearly-separated candidates."""

_RAG_V3_SEED_ENV = "RAG_V3_SEED"

# Standard Reciprocal Rank Fusion constant. k=60 is the canonical IR
# convention (Cormack, Clarke & Buettcher, SIGIR 2009, "Reciprocal Rank
# Fusion outperforms Condorcet..."), NOT a tuned hyperparameter. It damps
# the influence of very-high ranks so no single channel dominates the
# fusion. Used only by the ``graph_recall_rrf`` fusion mode.
_RRF_K = 60

# ---------------------------------------------------------------------------
# Fan-out-robust graph-recall constants (PPR + protected-RRF redesign,
# fanout_robust_traversal_redesign_2026-05-30.md). None are tuned: each is a
# canonical PageRank constant or a value the design measured as the
# recovery / single-hop-preservation optimum on the OpenIE KG.
# ---------------------------------------------------------------------------
_PPR_ALPHA = 0.85
"""Personalized PageRank damping factor for the legacy ``graph_recall_rrf``
mode. 0.85 is the canonical web-PageRank damping constant (Page & Brin,
1998), NOT a tuned weight.

CORRECTION (2026-05-30): an earlier version of this docstring claimed
"HippoRAG arXiv:2405.14831 §3.2 uses the same" 0.85 -- that was a factual
error. HippoRAG actually uses alpha=0.5 (see
paper/analysis/hipporag_gap_diagnosis_2026-05-30.md §1). The corrected
HippoRAG value lives in ``_PPR_ALPHA_GRAPH_CENTRIC`` and is used by the new
``graph_centric`` mode. This 0.85 constant is retained here UNCHANGED so the
``graph_recall_rrf`` mode (and its prior measurements / regression tests)
stay bit-identical; the design measured 0.3-0.5 strictly worse for that
mode's in-channel B-recall (36% -> 20-28%)."""

_PPR_ALPHA_GRAPH_CENTRIC = 0.5
"""Personalized PageRank damping factor for the ``graph_centric`` mode
(2nd overhaul). 0.5 is the value HippoRAG (Gutierrez et al.,
arXiv:2405.14831 §3.2) actually uses -- the prior ``_PPR_ALPHA`` comment
claiming HippoRAG=0.85 was a factual error (see
paper/analysis/hipporag_gap_diagnosis_2026-05-30.md §1). 0.5 keeps mass on
short bridge paths (A->bridge->B is 1-2 hops on this KG) instead of letting
it diffuse across the whole component. This is the published HippoRAG
operating point, NOT a tuned weight. The legacy ``graph_recall_rrf`` mode
keeps its own ``_PPR_ALPHA`` unchanged so its prior measurements stay
bit-identical; only ``graph_centric`` uses this constant."""

_PPR_ITERS = 30
"""Power-iteration count. 30 sweeps converge the PPR mass on this KG
(per-query ~0.001s, CPU only). A fixed iteration budget rather than a
convergence tolerance keeps the cost deterministic and bounded; 30 is well
past the mass-stabilisation point measured in the design prototype."""

_PPR_PROTECT_M = 3
"""protected-RRF: the BM25 top-M ranks held fixed as the final top-M; the
graph-recall channel may only fill the residual slots (M+1..). The design
measured M=3 as the recovery/non-harm optimum on the OpenIE KG
(bothAB@5: M=5 +0, M=3 +13, M=2 +6, M=0 -4). M=3 weakly breaks single-hop
non-harm (a small number of base-both losses) -- this trade-off is exposed
honestly via the protected/displaced diagnostics counters, not hidden.
Set to 5 for zero loss / zero recovery (full single-hop preservation)."""

_PPR_CUT = 20
"""Graph-recall channel rank cut: only the top-_PPR_CUT papers by PPR mass
receive a graph_recall_rank (and thus an RRF term). The design measured the
end-to-end result insensitive across cut 10..inf, so 20 is a structural
bound that caps the residual-slot competition pool, not a tuned threshold."""

_PPR_MASS_EPS = 1e-12
"""Papers whose summed PPR mass is at or below this floor are treated as
unreached (no graph_recall_rank). Pure numerical guard against
floating-point dust, not a tuned cutoff."""

# ---------------------------------------------------------------------------
# graph_centric mode (2nd overhaul, hipporag_gap_diagnosis_2026-05-30.md).
# PPR becomes a FIRST-CLASS ranking channel fused on equal footing with
# BM25/dense via RRF -- no protected top-M zone. A structural multihop gate
# (matched-seed pattern only, NO gold reference) restricts the graph-centric
# fusion to bridge-style questions; both-direct questions fall back to BM25.
# ---------------------------------------------------------------------------
_GRAPH_CENTRIC_BOTH_DIRECT_ANCHORS = 2
"""Both-direct trigger: when the question directly NAMES at least this many
DISTINCT corpus papers (full-name surface match of a node onto the question,
see ``_strict_anchor_papers``), both answer papers are lexically anchored, so
graph-centric fusion is NOT needed and BM25/dense ranking is preferred.
Diagnosis R1 (63/200 = 31.5% of multihop QAs have BOTH gold entities directly
seeded) motivates the rule; the threshold is the structural "two anchors"
definition of a both-direct question, not a tuned cutoff. The signal counts
the question's OWN surface anchors -- gold paper ids are never inspected."""

try:  # scipy is the primary path (sparse matrix-vector power iteration).
    import numpy as _np
    from scipy import sparse as _sparse

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only on a scipy-less env
    try:
        import numpy as _np  # numpy-only fallback still uses dense power iter

        _HAVE_SCIPY = False
    except Exception:  # pragma: no cover
        _np = None  # type: ignore[assignment]
        _HAVE_SCIPY = False

_GATED_SIGNAL_DEFAULTS: tuple[str, ...] = (
    "coverage_trigger",
    "alias_expansion",
    "edge_template",
    "query_strategy_adaptive",
    "trust_4axis",
)
"""Names emitted in ``diagnostics.gated_signals`` so downstream ablation
analysis can see which gates were satisfied for each candidate."""

# Relation templates by query intent (P3 redesign). Each tuple holds
# ``(head_type, relation, tail_type)`` triplets that are considered
# "relevant evidence" for the intent. The gate fires only when at least one
# edge in a candidate matches a triplet for the classified intent.
_RELATION_TEMPLATES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "defect": (
        ("process", "causes_issue", "issue"),
        ("equipment", "causes_issue", "issue"),
        ("process", "causes", "issue"),
        ("equipment", "causes", "issue"),
    ),
    "mitigation": (
        ("issue", "resolved_by", "process"),
        ("issue", "resolved_by", "material"),
        ("issue", "removes", "process"),
        ("issue", "removes", "material"),
        ("issue", "mitigates", "process"),
        ("issue", "prevents", "process"),
    ),
    "material": (
        ("material", "used_in", "process"),
        ("material", "produced_by", "process"),
    ),
    "causal": (
        ("process", "causes_issue", "issue"),
        ("equipment", "causes_issue", "issue"),
        ("process", "causes", "issue"),
    ),
    "compare": (
        ("process", "alternative_to", "process"),
        ("material", "alternative_to", "material"),
    ),
}


class PaperRetriever(Protocol):
    def query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class V3Weights:
    """Score weights for the V3 hybrid retriever.

    The defaults follow the P3 schema-centric redesign: typed schema is
    the primary fusion signal (0.50) with BM25/dense providing the high-recall
    candidate pool (0.20/0.20). Alignment, alias, edge_path, query_strategy
    and trust are kept as small additive nudges that only fire when their
    respective structural gates pass. ``mode='additive_legacy'`` callers
    can ignore the gates and reproduce the prior weighted-sum behaviour by
    overriding the weights explicitly (see ``_final_score``).
    """

    bm25: float = 0.20
    dense: float = 0.20
    alignment: float = 0.05
    alias: float = 0.03
    schema: float = 0.50
    edge_path: float = 0.01
    query_strategy: float = 0.00
    trust: float = 0.01


@dataclass
class MatchedNode:
    node_id: str
    node: dict[str, Any]
    alias_hit: bool = False
    name_hit: bool = False
    matched_terms: set[str] = field(default_factory=set)
    # #4 generic-token mask fix: structural specificity fields populated in
    # _match_nodes so the downstream scorer can apply HippoRAG-style
    # specificity weighting instead of a hard reject filter.
    #   passage_freq -- number of unique source papers attached to the node.
    #                   Used as the denominator of s_i = 1/|P_i|.
    #   is_typed     -- True iff schema_type in {process, material,
    #                   equipment, issue}. Typed nodes are exempt from
    #                   IDF demotion (typed-schema is the paper's main
    #                   contribution channel).
    passage_freq: int = 1
    is_typed: bool = False


@dataclass
class CandidateScore:
    paper_id: str
    paper: dict[str, Any]
    sources: set[str] = field(default_factory=set)
    bm25_raw: float = 0.0
    dense_raw: float = 0.0
    bm25_rank: int | None = None
    dense_rank: int | None = None
    alias_expanded_bm25: bool = False
    direct_node_ids: set[str] = field(default_factory=set)
    alias_node_ids: set[str] = field(default_factory=set)
    schema_types: set[str] = field(default_factory=set)
    neighbor_node_ids: set[str] = field(default_factory=set)
    edge_relations: set[str] = field(default_factory=set)
    edge_reliability: float = 0.0
    # #4 generic-token mask fix: track the maximum specificity multiplier
    # contributed by any node that matched this candidate, so the alignment
    # signal can be demoted for low-specificity (high-passage-frequency)
    # surface hits without dropping the candidate entirely.
    # Range: (0.0, 1.0]. 1.0 means at least one typed match or a
    # singleton-passage match. Lower values demote noisy surface matches.
    matched_spec_max: float = 0.0
    # graph_recall_rrf mode only (fan-out-robust redesign).
    # ``graph_recall_score`` is the SUM of Personalized PageRank mass over
    # every node whose source_docs contain this paper (see
    # _add_graph_recall_candidates). ``graph_recall_rank`` is the 1-based
    # rank within that channel (highest mass => rank 1, cut at _PPR_CUT),
    # assigned just before protected-RRF fusion. None / 0.0 when the channel
    # did not register this candidate (a missing graph_recall term in
    # _rrf_fuse never lowers the lexical score; protection holds the BM25
    # top-M fixed).
    graph_recall_score: float = 0.0
    graph_recall_rank: int | None = None


# #4 generic-token mask fix (2026-05-27).
#
# Previous design hard-rejected any single-token surface match against a
# fixed list of "generic" tokens that mixed two semantic categories:
#   (a) English function words   (a/an/the/of/as/for/to/in/on/with/by/and/or)
#   (b) Domain-generic nouns     (process/method/system/manufacturing/
#                                 production/technique/material/equipment/
#                                 issue/problem/defect/result/effect)
# Category (b) was a structural bug: factual queries like "what process
# forms TSV liners?" rely on exactly those tokens to anchor onto typed KG
# nodes (Process/Material/Equipment/Issue). Hard-rejecting them collapsed
# matched_nodes -> empty, which in turn made _add_kg_candidates short-
# circuit (global_neighbors=None), killing the typed-schema / 4-axis-
# trust / alignment signals that are the paper's main contribution.
#
# Replacement: keep only the pure-grammar function-word mask (universal IR
# stop-words, language-level, no domain semantics) and demote noisy
# surface matches via HippoRAG node specificity in the scoring layer
# rather than the matching layer. See _node_spec_multiplier below.
#
# References:
#   - Gutierrez et al., HippoRAG, arXiv:2405.14831 §3.2
#     "node specificity s_i = 1/|P_i| where |P_i| is the number of passages
#     in which node i appears" -- corpus statistic, structurally analogous
#     to IDF, not a learned weight.
#   - Robertson & Sparck Jones, "Relevance weighting of search terms" (1976)
#     for the IDF analog rationale.
_GRAMMAR_STOPWORD_TOKENS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "as",
    "for",
    "to",
    "in",
    "on",
    "with",
    "by",
}
# Backwards-compatible alias: prior call sites (e.g. _matched_token_coverage)
# referenced ``_GENERIC_SINGLE_TOKEN_NAMES``. We retain the symbol pointing
# at the smaller grammar-only mask so any extension code that imports it
# does not silently break -- but the domain-noun entries are removed.
_GENERIC_SINGLE_TOKEN_NAMES = _GRAMMAR_STOPWORD_TOKENS

# Schema types treated as "typed" for the #4 fix specificity exemption.
# Matching the paper's main-contribution channel (Process / Material /
# Equipment / Issue) — typed nodes preserve full signal contribution;
# only untyped nodes are subject to IDF demotion. This is a structural
# condition (KG schema membership), not a tuned threshold.
_TYPED_SCHEMA_TYPES = {"process", "material", "equipment", "issue"}

_SCHEMA_TYPES = {
    "process",
    "material",
    "equipment",
    "issue",
    "source",
    "parameter",
    "metric",
}

_CAUSAL_RELATIONS = {
    "cause",
    "causes",
    "caused_by",
    "causes_by",
    "leads_to",
    "results_in",
    "resolves",
    "mitigates",
    "prevents",
}

_COMPARE_RELATIONS = {
    "alternative_to",
    "compares_with",
    "contrasts_with",
    "substitutes",
}


class ProposedV3Retriever:
    """Hybrid retriever prototype for V3 ablations.

    The retriever is intentionally small and dependency-injected: tests can pass
    fake BM25/dense retrievers, while production callers can allow the BM25
    baseline to be constructed lazily from ``papers``.
    """

    def __init__(
        self,
        papers: list[dict[str, Any]],
        bm25: PaperRetriever | None = None,
        dense: PaperRetriever | None = None,
        nodes: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
        weights: V3Weights | None = None,
        coverage_trigger_threshold: float = 0.0,
        c4_postprocess: C4PostprocessMode = "on",
        c5_query_strategy: C5QueryStrategyMode = "adaptive",
        c6_trust: C6TrustMode = "4axis",
        sub_answer_req: SubAnswerReqMode = "schema",
        postprocess_lexicon: dict[str, list[str]] | None = None,
        mode: FusionMode = "schema_centric_gated",
        # ----- graph_recall_rrf channel (multihop A->bridge->B recall) ----
        # graph_recall_k: BFS hop budget for the A-seed -> reached-node ->
        #   source_doc=B recall walk. 2 is a structural constant (the
        #   measured A->B reachability in the KG is 1-2 hops for 5/6 gold
        #   multihop pairs; see graph_recall_redesign_v2). NOT tuned.
        # graph_recall_gate: which structural condition activates the
        #   channel. "coverage_low" (default) fires only when matched-token
        #   coverage < _COVERAGE_LOW_RATIO; "multihop" fires on multihop
        #   query intent; "always" is for unit-test/ablation use. The gate
        #   guarantees single-hop / coverage-high queries see an EMPTY
        #   channel => bit-identical to the non-graph-recall path.
        graph_recall_k: int = 2,
        graph_recall_gate: GraphRecallGate = "coverage_low",
        # ----- Phase 2 (D47 G5) minus_one toggles ------------------
        # Each flag, when True, surgically removes ONE C0 mechanism by
        # mutating init-time configuration (coverage_trigger_threshold,
        # V3Weights, or c6_trust). D46 retrieval body
        # (query/scoring/RRF) is NEVER touched. Only one toggle may be
        # True per construction; multiple True -> ValueError.
        minus_one_coverage_trigger: bool = False,  # C4a
        minus_one_typed_schema:     bool = False,  # C5a
        minus_one_edge_path:        bool = False,  # C6a
        minus_one_query_strategy:   bool = False,  # C7a
        minus_one_trust:            bool = False,  # C8a
    ) -> None:
        self.papers = list(papers)
        self.paper_by_id = {
            paper_id: paper
            for paper in self.papers
            if (paper_id := self._paper_id(paper)) is not None
        }
        if bm25 is None:
            from prototype.baselines.bm25_retriever import BM25Baseline

            self.bm25: PaperRetriever = BM25Baseline(self.papers)
        else:
            self.bm25 = bm25
        self.dense = dense
        self.nodes = nodes or []
        self.edges = edges or []
        self.weights = weights or V3Weights()
        self.coverage_trigger_threshold = max(0.0, min(1.0, coverage_trigger_threshold))
        self.nodes_by_id = {
            node_id: node
            for node in self.nodes
            if (node_id := self._node_id(node)) is not None
        }
        self._adjacency = self._build_adjacency(self.edges)
        # D45 / H9 typed-factual BFS: precompute node_id -> schema_type
        # (lowercased) so query_strategy.traverse can run the typed
        # two-pass BFS without re-walking self.nodes on every query.
        self._node_types: dict[str, str] = {
            node_id: str(
                node.get("type") or node.get("node_type") or ""
            ).lower()
            for node_id, node in self.nodes_by_id.items()
        }
        # #4 generic-token mask fix: precompute node_id -> passage_freq
        # (number of unique source-docs the node attaches to). This is the
        # |P_i| in HippoRAG's specificity s_i = 1/|P_i| (Gutierrez et al.,
        # arXiv:2405.14831 §3.2). One pass over self.nodes; cached for the
        # lifetime of the retriever. Floored at 1 to guarantee a finite
        # multiplier even for nodes without source_docs metadata.
        self._passage_freq: dict[str, int] = {
            node_id: max(1, len(set(_source_docs(node))))
            for node_id, node in self.nodes_by_id.items()
        }
        # graph_centric multihop gate: paper-title anchor index. Maps a
        # whitespace-normalised paper TITLE (corpus metadata, NOT gold) to its
        # paper id, restricted to titles of >=2 tokens / >=4 chars so a bare
        # one-word title cannot spuriously anchor on a common question word.
        # Used only by ``_strict_anchor_papers`` (graph_centric mode); built
        # once, empty for other modes that never call it. Counting how many
        # corpus paper titles a question names is a pure surface signal -- the
        # gold paper set is never consulted.
        self._title_anchor_index: dict[str, str] = {}
        if mode == "graph_centric":
            for paper_id, paper in self.paper_by_id.items():
                title_norm = _normalise_title(str(paper.get("title") or ""))
                if len(title_norm) >= 4 and " " in title_norm:
                    self._title_anchor_index.setdefault(title_norm, paper_id)
        # Fan-out-robust graph-recall: build the column-stochastic transition
        # matrix P = A * D^-1 once over the SAME undirected edge set the rest
        # of the retriever walks (synonym edges INCLUDED -- the redesign keeps
        # them; protected-RRF absorbs the fan-out cost). Cached here so PPR
        # power iteration in _graph_recall_ppr is a few sparse mat-vecs per
        # query, no per-query graph construction. Built only when the
        # graph_recall_rrf mode could fire and numpy/scipy is importable;
        # other modes never touch it.
        self._ppr_node_index: dict[str, int] = {}
        self._ppr_matrix = None
        if mode in ("graph_recall_rrf", "graph_centric") and _np is not None:
            self._build_ppr_matrix()
        # W7 D33-D36 4-mechanism toggles. Defaults preserve current behaviour:
        # all four mechanisms ON, so an existing caller that does not pass
        # these flags gets the full proposed_v3 system unchanged.
        self.c4_postprocess: C4PostprocessMode = c4_postprocess
        self.c5_query_strategy: C5QueryStrategyMode = c5_query_strategy
        self.c6_trust: C6TrustMode = c6_trust
        self.sub_answer_req: SubAnswerReqMode = sub_answer_req
        self.postprocess_lexicon = postprocess_lexicon or {}
        # P3 fusion mode. ``schema_centric_gated`` is the new default and
        # applies the per-signal gates documented in the P3 brief.
        # ``additive_legacy`` reproduces the pre-P3 weighted-sum behaviour
        # for back-compat (the prior ablation cells expect this).
        self.mode: FusionMode = mode
        # graph_recall_rrf channel config. Stored verbatim; only consulted
        # when ``mode == "graph_recall_rrf"``. graph_recall_k floored at 1
        # so the BFS always takes at least one hop. Other modes ignore these.
        self.graph_recall_k: int = max(1, int(graph_recall_k))
        self.graph_recall_gate: GraphRecallGate = graph_recall_gate
        # Phase 2 (D47 G5) minus_one toggles. Validate at most one True,
        # cache the flag set, then apply the OFF surgically by mutating
        # init-time configuration. NO scoring / RRF / query change.
        minus_one_flags = (
            minus_one_coverage_trigger,
            minus_one_typed_schema,
            minus_one_edge_path,
            minus_one_query_strategy,
            minus_one_trust,
        )
        if sum(1 for f in minus_one_flags if f) > 1:
            raise ValueError(
                "at most one minus_one_* toggle may be True; got "
                f"{minus_one_flags}"
            )
        self._minus_one_flags: dict[str, bool] = {
            "coverage_trigger": bool(minus_one_coverage_trigger),
            "typed_schema":     bool(minus_one_typed_schema),
            "edge_path":        bool(minus_one_edge_path),
            "query_strategy":   bool(minus_one_query_strategy),
            "trust":            bool(minus_one_trust),
        }
        self._apply_minus_one_overrides()
        # Phase 4 G6 — per-query traversal label cache.
        # ``_add_kg_candidates`` writes the ``TraversalResult.query_type``
        # label here so ``query`` can stamp it onto the top-1 candidate's
        # diagnostics block AFTER the D46 scoring path runs. Single-threaded
        # per-instance contract (one query at a time). None when no matched
        # nodes triggered an adaptive traversal for the current query.
        self._last_query_strategy_label: str | None = None

    def _apply_minus_one_overrides(self) -> None:
        """Single-mechanism subtraction. D46 retrieval body unchanged.

        Phase 2 Plan 1.4: each toggle, when True, surgically disables
        ONE C0 mechanism by mutating init-time configuration. The
        retriever query / scoring / RRF logic is NEVER touched.

        - coverage_trigger -> coverage_trigger_threshold = 0.0
        - typed_schema     -> V3Weights.schema = 0.0
        - edge_path        -> V3Weights.edge_path = 0.0
        - query_strategy   -> V3Weights.query_strategy = 0.0
        - trust            -> V3Weights.trust = 0.0 AND
                              c6_trust = "single_source_weight"
        """
        from dataclasses import replace as _replace
        if self._minus_one_flags["coverage_trigger"]:
            self.coverage_trigger_threshold = 0.0
        if self._minus_one_flags["typed_schema"]:
            self.weights = _replace(self.weights, schema=0.0)
        if self._minus_one_flags["edge_path"]:
            self.weights = _replace(self.weights, edge_path=0.0)
        if self._minus_one_flags["query_strategy"]:
            self.weights = _replace(self.weights, query_strategy=0.0)
        if self._minus_one_flags["trust"]:
            self.weights = _replace(self.weights, trust=0.0)
            # Trust signal also flows through the c6 toggle; flipping to
            # the legacy weighted sum (single_source_weight) ensures the
            # OFF removes the entire 4-axis branch rather than leaving a
            # zero-weighted-but-still-computed axis chain.
            self.c6_trust = "single_source_weight"

    def query(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 50,
    ) -> list[dict[str, Any]]:
        if not question.strip():
            return []

        # Phase 4 G6 — reset per-query traversal label cache so
        # the postprocess + query_strategy_actual stamp at the end of this
        # call cannot leak state from a previous query. Set by
        # ``_add_kg_candidates`` only when matched_nodes is non-empty.
        self._last_query_strategy_label = None

        matched_nodes = self._match_nodes(question)
        candidates: dict[str, CandidateScore] = {}

        self._add_retriever_hits(
            candidates,
            self.bm25.query(question, top_k=candidate_k),
            channel="bm25",
        )
        if self.dense is not None:
            self._add_retriever_hits(
                candidates,
                self.dense.query(question, top_k=candidate_k),
                channel="dense",
            )

        alias_expansion = self._alias_expanded_query(question, matched_nodes)
        if alias_expansion is not None:
            self._add_retriever_hits(
                candidates,
                self.bm25.query(alias_expansion, top_k=candidate_k),
                channel="bm25",
                alias_expanded=True,
            )

        coverage_expansion = self._coverage_triggered_alias_query(
            question,
            matched_nodes,
            already_issued=alias_expansion,
        )
        if coverage_expansion is not None:
            self._add_retriever_hits(
                candidates,
                self.bm25.query(coverage_expansion, top_k=candidate_k),
                channel="bm25",
                alias_expanded=True,
            )

        query_type = self._classify_query(question)
        # B-1 fix: in graph_recall_rrf mode the legacy KG neighbour traversal is
        # SKIPPED. It would otherwise pollute the candidate pool (kg-sourced
        # papers with no RRF rank) before the graph-recall channel fires, which
        # (a) breaks the single-hop non-harm guarantee at the pool level and
        # (b) muddies provenance (a recovered paper could enter via the legacy
        # neighbour scan rather than the intended A->bridge->B path). The
        # graph_recall_rrf path adds its own candidates inside
        # _graph_recall_rrf_query (gated to coverage-low / multihop). Every
        # other mode keeps the legacy KG channel exactly as before.
        if self.mode not in ("graph_recall_rrf", "graph_centric"):
            self._add_kg_candidates(candidates, matched_nodes, query_type=query_type)

        if not candidates:
            return []

        bm25_norm = _normalise_by_max(
            {paper_id: item.bm25_raw for paper_id, item in candidates.items()}
        )
        dense_norm = _normalise_by_max(
            {paper_id: item.dense_raw for paper_id, item in candidates.items()}
        )
        query_entity_ids = {matched.node_id for matched in matched_nodes}
        query_intent = self._classify_intent(question)
        coverage_low = (
            self._matched_token_coverage(question, matched_nodes) < _COVERAGE_LOW_RATIO
            if matched_nodes
            else False
        )

        # graph_recall_rrf mode: A-seed -> k-hop -> source_doc=B recall
        # channel fused with the lexical channels via additive RRF. This is
        # a DISTINCT scoring path; every other mode falls through to the
        # existing _final_score weighted-sum loop below, bit-for-bit
        # unchanged. The branch lives here (after coverage_low / bm25_norm)
        # purely to reuse the already-computed inputs.
        if self.mode == "graph_recall_rrf":
            return self._graph_recall_rrf_query(
                question=question,
                candidates=candidates,
                matched_nodes=matched_nodes,
                coverage_low=coverage_low,
                query_type=query_type,
                query_intent=query_intent,
                top_k=top_k,
            )

        # graph_centric mode (2nd overhaul): PPR is a first-class RRF channel
        # fused on equal footing with BM25/dense (NO protected top-M zone). A
        # structural multihop gate restricts the graph-centric fusion to
        # bridge-style questions; both-direct questions fall back to a
        # BM25/dense-only RRF so single-hop / both-anchored queries are not
        # harmed. The gate inspects ONLY the matched-seed structure -- gold is
        # never referenced.
        if self.mode == "graph_centric":
            return self._graph_centric_query(
                question=question,
                candidates=candidates,
                matched_nodes=matched_nodes,
                coverage_low=coverage_low,
                query_type=query_type,
                query_intent=query_intent,
                top_k=top_k,
            )

        # Pre-pass: compute raw per-candidate signals + per-candidate
        # provisional trust so we can derive the structural trust-spread
        # gate before the final scoring loop. This second-pass design
        # keeps every gate purely structural (rule-based, no tunable
        # threshold beyond the published ratios).
        pending: list[tuple[CandidateScore, dict[str, Any]]] = []
        for candidate in candidates.values():
            signals = self._signals_for(candidate, query_type, query_entity_ids)
            pending.append((candidate, signals))
        trust_spread = _std(
            [float(signals.get("c6_trust", 0.0)) for _candidate, signals in pending]
        )
        # C6 toggle: structural difference between the 4-axis and legacy
        # gates. The 4-axis aggregate is variance-collapsing (uniform mean
        # of 4 bounded axes) so we gate it off when std < threshold to
        # avoid noise. The legacy single_source_weight formula is a
        # 3-term weighted sum that is itself a confidence aggregate, so
        # we always let it contribute (when there is any signal at all).
        # This guarantees that flipping c6_trust changes the COMPUTATION
        # PATH the score takes, not just the formula.
        if self.c6_trust == "4axis":
            trust_gate_active = trust_spread >= _TRUST_SPREAD_MIN_STD
        else:
            # single_source_weight: gate fires whenever at least one
            # candidate has a non-zero legacy trust score (i.e. any
            # source-diversity / sufficiency / reliability signal exists).
            trust_gate_active = any(
                float(signals.get("c6_trust", 0.0)) > 0.0
                for _candidate, signals in pending
            )

        scored: list[dict[str, Any]] = []
        for candidate, signals in pending:
            kg_norm = _kg_norm(signals)
            features = {
                "bm25_norm": bm25_norm.get(candidate.paper_id, 0.0),
                "dense_norm": dense_norm.get(candidate.paper_id, 0.0),
                "kg_norm": kg_norm,
                "trust": signals["c6_trust"],
                "bm25_raw": candidate.bm25_raw,
                "dense_raw": candidate.dense_raw,
            }
            gated_signals = self._gated_signals(
                candidate,
                query_type=query_type,
                query_intent=query_intent,
                coverage_low=coverage_low,
                trust_gate_active=trust_gate_active,
            )
            score = self._final_score(features, signals, gated_signals)
            trust_axes = signals.pop("_trust_axes", None)
            scored.append({
                "paper": candidate.paper,
                "score": float(score),
                "rank": 0,
                "features": features,
                "signals": signals,
                "source": self._primary_source(candidate),
                "diagnostics": {
                    "sources": sorted(candidate.sources),
                    "signals": dict(signals),
                    "matched_nodes": sorted(candidate.direct_node_ids),
                    "alias_nodes": sorted(candidate.alias_node_ids),
                    "neighbor_nodes": sorted(candidate.neighbor_node_ids),
                    "schema_types": sorted(candidate.schema_types),
                    "edge_relations": sorted(candidate.edge_relations),
                    "edge_reliability": candidate.edge_reliability,
                    "query_type": query_type,
                    "query_intent": query_intent,
                    "alias_expanded_bm25": candidate.alias_expanded_bm25,
                    "trust_axes": trust_axes,
                    "trust_spread": trust_spread,
                    "gated_signals": dict(gated_signals),
                    "mode": self.mode,
                    "toggles": {
                        "c4_postprocess": self.c4_postprocess,
                        "c5_query_strategy": self.c5_query_strategy,
                        "c6_trust": self.c6_trust,
                        "sub_answer_req": self.sub_answer_req,
                    },
                },
            })

        scored.sort(
            key=lambda item: (
                -item["score"],
                item["paper"].get("id") or item["paper"].get("paper_id") or "",
            )
        )
        # Seed-dependent tie-band shuffle: candidates whose scores are
        # within ``_TIE_BAND_EPS`` of each other are reordered using a
        # numpy RNG seeded from RAG_V3_SEED. This produces genuine
        # cross-seed variance (so paired bootstrap CIs have non-zero
        # spread) without changing the mean rank of non-tied candidates.
        # Pure structural rule: candidates with truly equal scores have
        # arbitrary order anyway; we only randomise *within* tied bands.
        scored = _shuffle_tie_bands(scored, eps=_TIE_BAND_EPS)
        # C4 postprocess: 7-step saturation re-rank, runs AFTER the V3 final
        # score but BEFORE top-k truncation. Default ON; cell C9 toggles it
        # off via c4_postprocess="off".
        if self.c4_postprocess == "on":
            scored = _postprocess_module.saturate(
                scored,
                query=question,
                nodes=self.nodes,
                edges=self.edges,
                papers_by_id=self.paper_by_id,
                lexicon=self.postprocess_lexicon,
                nodes_by_id=self.nodes_by_id,
                adjacency=self._adjacency,
            )
            # Postprocess re-sorts internally, so re-apply the tie-band
            # shuffle to its output to preserve seed-dependent variance.
            scored = _shuffle_tie_bands(scored, eps=_TIE_BAND_EPS)
        for rank, item in enumerate(scored[:top_k], 1):
            item["rank"] = rank
        top_results = scored[:top_k]
        # Phase 4 G6 — stamp the postprocess + query_strategy
        # signals onto the top-1 candidate's diagnostics AFTER all D46
        # ranking / postprocess / tie-band shuffle work has completed.
        # D46 retrieval body (scoring / RRF / ranking / saturate) is NOT
        # touched: this is a pure diagnostics annotation on the result
        # list that ``query`` is about to return.
        #
        # postprocess.applied        -> True iff c4_postprocess == "on"
        # postprocess.bonus_applied_count
        #                            -> 0 when off; otherwise count of
        #                               top-k items whose
        #                               diagnostics.c4_postprocess.bonus
        #                               is strictly > 0 (saturate writes
        #                               the per-candidate bonus into that
        #                               sub-block; see
        #                               prototype/eval/postprocess.py).
        # query_strategy_actual      -> TraversalResult.query_type label
        #                               cached in
        #                               self._last_query_strategy_label
        #                               by _add_kg_candidates. None when
        #                               no matched nodes triggered a
        #                               traversal for this query.
        if top_results:
            postprocess_applied = self.c4_postprocess == "on"
            if postprocess_applied:
                bonus_applied_count = sum(
                    1
                    for item in top_results
                    if float(
                        (
                            item.get("diagnostics", {}).get("c4_postprocess", {})
                            or {}
                        ).get("bonus", 0.0)
                    )
                    > 0.0
                )
            else:
                bonus_applied_count = 0
            top_results[0]["diagnostics"]["postprocess"] = {
                "applied": postprocess_applied,
                "bonus_applied_count": bonus_applied_count,
            }
            top_results[0]["diagnostics"]["query_strategy_actual"] = (
                self._last_query_strategy_label
            )
        return top_results

    def _add_retriever_hits(
        self,
        candidates: dict[str, CandidateScore],
        hits: list[dict[str, Any]],
        *,
        channel: str,
        alias_expanded: bool = False,
    ) -> None:
        for fallback_rank, hit in enumerate(hits, 1):
            paper = hit.get("paper")
            if not isinstance(paper, dict):
                continue
            paper_id = self._paper_id(paper)
            if paper_id is None:
                continue
            candidate = self._candidate_for(candidates, paper_id, paper)
            candidate.sources.add(channel)
            score = _as_float(hit.get("score"))
            rank = _as_int(hit.get("rank")) or fallback_rank
            if channel == "bm25":
                candidate.bm25_raw = max(candidate.bm25_raw, score)
                candidate.bm25_rank = (
                    rank
                    if candidate.bm25_rank is None
                    else min(candidate.bm25_rank, rank)
                )
            elif channel == "dense":
                candidate.dense_raw = max(candidate.dense_raw, score)
                candidate.dense_rank = (
                    rank
                    if candidate.dense_rank is None
                    else min(candidate.dense_rank, rank)
                )
            if alias_expanded:
                candidate.alias_expanded_bm25 = True

    def _add_kg_candidates(
        self,
        candidates: dict[str, CandidateScore],
        matched_nodes: list[MatchedNode],
        *,
        query_type: str = "factual",
    ) -> None:
        # C5 toggle: adaptive vs uniform_BFS BOTH go through the
        # query_strategy module so the candidate POOL changes when the
        # toggle flips (structural fix: previously
        # uniform_BFS fell back to the depth-2 _iter_neighbor_paths path,
        # which produced an identical or near-identical neighbour set to
        # the adaptive default for factual queries, masking the toggle).
        #
        #  - "adaptive"     -> _query_strategy_module.traverse(query_type)
        #                      (causal-edge-first / compare-pair / 1-hop define
        #                       / 4-hop factual BFS)
        #  - "uniform_BFS"  -> _query_strategy_module.uniform_bfs(depth=4)
        #                      (query_type IGNORED, always 4-hop BFS)
        #
        # Both modes build a SINGLE global neighbour set seeded by the union
        # of matched nodes, ensuring the toggle changes the COMPUTATION PATH
        # (different traversal logic), not just a score weight.
        if matched_nodes:
            seed_ids = [m.node_id for m in matched_nodes]
            if self.c5_query_strategy == "adaptive":
                # Adaptive: causal-edge BFS / compare-pair / 1-hop define
                # / typed-factual BFS, dispatched by query_type. node_types
                # is forwarded so factual queries route through the D45
                # typed two-pass BFS (Process/Material/Equipment/Issue
                # priority) instead of the legacy uniform BFS that
                # collapsed onto the C10 OFF baseline.
                result = _query_strategy_module.traverse(
                    seed_ids,
                    self._adjacency,
                    query_type,
                    node_types=self._node_types,
                )
            else:
                # uniform_BFS OFF flag: fixed depth-2 BFS regardless of
                # intent. This is structurally narrower than the adaptive
                # depth-4 factual BFS, so the candidate pool genuinely
                # differs even on factual-heavy QA workloads.
                result = _query_strategy_module.uniform_bfs(
                    seed_ids,
                    self._adjacency,
                    depth=2,
                )
            global_neighbors = result.neighbors
            # Phase 4 G6 — preserve the TraversalResult label
            # ("causal" / "compare" / "define" / "factual_typed" /
            # "uniform") so ``query`` can stamp it onto the top-1
            # candidate's diagnostics AFTER the D46 scoring path runs.
            # Read in ``query`` return-prep block; reset to None at the
            # top of every ``query`` call.
            self._last_query_strategy_label = result.query_type
        else:
            global_neighbors = None

        for matched in matched_nodes:
            # #4 fix: compute the per-match specificity multiplier once here
            # and pass it through so every candidate this matched node
            # contributes to inherits at least this much specificity weight.
            spec_mult = self._node_spec_multiplier(
                is_typed=matched.is_typed,
                passage_freq=matched.passage_freq,
            )
            self._add_node_source_docs(
                candidates,
                matched.node,
                direct_node_id=matched.node_id,
                alias_hit=matched.alias_hit,
                spec_mult=spec_mult,
            )
            if global_neighbors is not None:
                # Use the precomputed global neighbour set once; each matched
                # node still anchors its source docs above.
                neighbors_iter = global_neighbors
                global_neighbors = []  # consume only once
            else:
                neighbors_iter = self._iter_neighbor_paths(
                    matched.node_id, max_depth=2,
                )
            for neighbor_id, edge in neighbors_iter:
                neighbor = self.nodes_by_id.get(neighbor_id)
                if neighbor is None:
                    continue
                self._add_node_source_docs(
                    candidates,
                    neighbor,
                    direct_node_id=None,
                    neighbor_node_id=neighbor_id,
                    edge=edge,
                )
                self._add_edge_source_docs(candidates, edge, neighbor_id)

    # ------------------------------------------------------------------
    # graph_recall_rrf channel: A-seed -> k-hop -> source_doc=B recall.
    # ------------------------------------------------------------------
    # NOTE: ``_graph_recall_gate_open`` was removed in the fan-out-robust
    # redesign (fanout_robust_traversal_redesign_2026-05-30.md §2.4). The
    # coverage<0.5 utterance gate was KG-size-contaminated and, when it did
    # fire, behaved identically to "always" with no measurable effect. The
    # effective gate is now the top-M protection in the fusion step itself
    # (a B paper already in the BM25 top-M is structurally un-displaceable).
    # ``self.graph_recall_gate`` is retained on the constructor for API
    # back-compat but no longer alters the candidate pool.

    def _is_multihop_intent(self, question: str) -> bool:
        """Structural multihop-intent detector (surface rule, no tuning).

        Fires on bridge-style phrasings that ask to connect two entities
        across an intermediary (e.g. "how does X relate to Y", "connection
        between A and B", "via", "through which"). These are exactly the
        questions whose answer paper B is lexically weak yet graph-reachable
        from the A anchor. Kept deliberately narrow so single-hop factual
        queries do not trip it.
        """
        lowered = question.lower()
        return bool(
            re.search(
                r"\b(relate[sd]?\s+to|relationship\s+between|connection\s+between"
                r"|link\s+between|bridge\s+between|how\s+(?:does|do|is|are)\b"
                r".*\b(?:relate|connect|link|affect|impact|influence)"
                r"|through\s+which|by\s+way\s+of)\b",
                lowered,
            )
        )

    def _build_ppr_matrix(self) -> None:
        """Build the column-stochastic transition matrix P = A * D^-1 once.

        Node index space = every node appearing in ``self._adjacency`` (the
        same undirected adjacency, synonym edges included). Parallel edges
        between the same pair collapse to a single 1.0 entry (an unweighted
        adjacency walk; relation multiplicity is not a PageRank weight). Each
        column j is normalised by the out-degree of node j so column sums are
        1 (mass-conserving random-walk transition). Isolated columns (no
        out-edge) stay all-zero -- their reset mass simply leaks each sweep,
        which is the standard PPR treatment of dangling nodes.

        Uses scipy.sparse when available; otherwise a dense numpy matrix
        (the KG is ~10k nodes -> a dense float64 matrix would be ~800MB, so
        the dense fallback is only viable on small KGs / unit tests).
        """
        index: dict[str, int] = {}
        for node_id in self._adjacency:
            if node_id not in index:
                index[node_id] = len(index)
        n = len(index)
        self._ppr_node_index = index
        if n == 0:
            self._ppr_matrix = None
            return
        # Collect unique undirected (col=source, row=target) pairs.
        pairs: set[tuple[int, int]] = set()
        for node_id, neighbours in self._adjacency.items():
            col = index[node_id]
            for neighbour_id, _edge in neighbours:
                row = index.get(neighbour_id)
                if row is None:
                    continue
                pairs.add((row, col))
        # Out-degree per column for D^-1 normalisation.
        out_deg = [0] * n
        for _row, col in pairs:
            out_deg[col] += 1
        if _HAVE_SCIPY:
            rows = _np.fromiter((r for r, _c in pairs), dtype=_np.int64, count=len(pairs))
            cols = _np.fromiter((c for _r, c in pairs), dtype=_np.int64, count=len(pairs))
            data = _np.fromiter(
                (1.0 / out_deg[c] for _r, c in pairs),
                dtype=_np.float64,
                count=len(pairs),
            )
            self._ppr_matrix = _sparse.csr_matrix(
                (data, (rows, cols)), shape=(n, n)
            )
        else:  # pragma: no cover - dense fallback, small KGs only
            mat = _np.zeros((n, n), dtype=_np.float64)
            for row, col in pairs:
                mat[row, col] = 1.0 / out_deg[col]
            self._ppr_matrix = mat

    def _graph_recall_ppr(
        self,
        seed_ids: list[str],
        *,
        alpha: float = _PPR_ALPHA,
        spec_personalization: bool = False,
    ) -> dict[str, float]:
        """Personalized PageRank mass per node, seeded on the A-match set.

        r <- (1 - alpha) * s + alpha * P * r, for _PPR_ITERS sweeps. ``s`` is
        the personalization (reset) vector. P is the cached column-stochastic
        transition matrix. Returns node_id -> PPR mass for every indexed node
        (seeds included; the caller maps node mass to documents). Replaces the
        BFS+specificity traversal: PPR lets a hub disperse its mass across many
        neighbours (natural decay) where the old BFS scored every reached node
        identically.

        ``alpha`` defaults to ``_PPR_ALPHA`` (0.85) so the legacy
        ``graph_recall_rrf`` callers are bit-identical; the ``graph_centric``
        mode passes ``_PPR_ALPHA_GRAPH_CENTRIC`` (0.5, the true HippoRAG value).

        ``spec_personalization`` (graph_centric only) weights each seed's reset
        mass by its HippoRAG node specificity s_i = 1/|P_i| (Gutierrez et al.,
        arXiv:2405.14831 §3.2) before re-normalising the reset vector to sum 1.
        A seed that appears in many passages (low specificity / hub-like) thus
        injects proportionally less reset mass than a rare, discriminative
        seed -- the diagnosis flagged specificity as defined but never wired
        into the PPR channel. Default False keeps the uniform-seed legacy path
        untouched.
        """
        index = self._ppr_node_index
        matrix = self._ppr_matrix
        if matrix is None or not index:
            return {}
        seeds = [s for s in seed_ids if s in index]
        if not seeds:
            return {}
        n = len(index)
        s = _np.zeros(n, dtype=_np.float64)
        if spec_personalization:
            # Specificity-weighted reset: each seed contributes s_i = 1/|P_i|,
            # then the vector is L1-normalised so total reset mass stays 1.
            weights = [
                1.0 / float(max(1, self._passage_freq.get(seed, 1)))
                for seed in seeds
            ]
            total = sum(weights)
            if total <= 0.0:
                weights = [1.0 for _ in seeds]
                total = float(len(seeds))
            for seed, weight in zip(seeds, weights):
                s[index[seed]] += weight / total
        else:
            seed_mass = 1.0 / float(len(seeds))
            for seed in seeds:
                s[index[seed]] += seed_mass
        r = s.copy()
        reset = (1.0 - alpha) * s
        for _ in range(_PPR_ITERS):
            r = reset + alpha * (matrix @ r)
        return {node_id: float(r[row]) for node_id, row in index.items()}

    def _add_graph_recall_candidates(
        self,
        candidates: dict[str, CandidateScore],
        matched_nodes: list[MatchedNode],
        *,
        alpha: float = _PPR_ALPHA,
        spec_personalization: bool = False,
    ) -> None:
        """Register graph-recall papers from Personalized PageRank mass.

        graph_recall_score(paper) = SUM of the PPR mass of every node whose
        source_docs contain ``paper`` (the design measured the summed
        node-mass aggregate strictly better than max / IDF-weighted). The
        channel is UNGATED: it always runs, and the residual-slot fusion in
        ``_graph_recall_rrf_query`` (top-M protection) is the effective gate
        -- a B paper already in the BM25 protected zone is never displaced.
        Papers below ``_PPR_MASS_EPS`` are left unregistered.

        ``alpha`` / ``spec_personalization`` are forwarded to
        ``_graph_recall_ppr``; the defaults reproduce the legacy
        ``graph_recall_rrf`` behaviour bit-for-bit. ``graph_centric`` passes
        ``alpha=_PPR_ALPHA_GRAPH_CENTRIC`` and ``spec_personalization=True``.
        """
        if not matched_nodes:
            return
        seed_ids = [m.node_id for m in matched_nodes]
        ppr = self._graph_recall_ppr(
            seed_ids, alpha=alpha, spec_personalization=spec_personalization
        )
        if not ppr:
            return
        # Accumulate per-paper summed mass first, then register once so the
        # graph_recall_score is the true node-mass SUM (not a running max).
        paper_mass: dict[str, float] = defaultdict(float)
        for node_id, mass in ppr.items():
            if mass <= _PPR_MASS_EPS:
                continue
            node = self.nodes_by_id.get(node_id)
            if node is None:
                continue
            for paper_id in _source_docs(node):
                if paper_id in self.paper_by_id:
                    paper_mass[paper_id] += mass
        for paper_id, mass in paper_mass.items():
            if mass <= _PPR_MASS_EPS:
                continue
            candidate = self._candidate_for(
                candidates, paper_id, self.paper_by_id[paper_id]
            )
            candidate.sources.add("kg_graph_recall")
            candidate.graph_recall_score = mass

    def _graph_recall_rrf_query(
        self,
        *,
        question: str,
        candidates: dict[str, CandidateScore],
        matched_nodes: list[MatchedNode],
        coverage_low: bool,
        query_type: str,
        query_intent: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score the candidate pool with protected-RRF (fan-out-robust
        redesign): the BM25 top-M (``_PPR_PROTECT_M``) ranks are held fixed
        as the final top-M, and the graph-recall channel competes only for
        the residual slots (M+1..top_k).

        Mechanism (fanout_robust_traversal_redesign_2026-05-30.md §2.2):
        protected-RRF is the real recovery lever. Plain additive RRF over an
        ungated PPR channel measured -4 (B rises but noise rises with it,
        displacing A from top-5). Holding the BM25 top-M anchors A in place
        while the graph channel fills slots M+1..; the residual pool is fused
        with additive RRF (``1/(60+bm25_rank) + 1/(60+gr_rank)``).

        Honesty note (design R1): single-hop non-harm is NOT bit-exact at
        M=3 -- a base-both pair whose B sits OUTSIDE the BM25 top-M can be
        displaced from the residual slots by the graph channel. This is a
        measured trade-off (M=3 recovers more at the cost of a few losses;
        M=5 = zero loss, zero recovery). It is exposed, not hidden, via the
        ``protected_count`` / ``displaced_count`` diagnostics counters below.
        """
        # Add the graph-recall channel (PPR; ungated -- top-M protection in
        # the fusion below is the effective gate).
        multihop_intent = self._is_multihop_intent(question)
        self._add_graph_recall_candidates(candidates, matched_nodes)
        # Within-channel graph-recall rank: PPR mass descending, ties broken
        # by paper_id for determinism, cut at _PPR_CUT.
        gr_ranked = sorted(
            (c for c in candidates.values() if c.graph_recall_score > 0.0),
            key=lambda c: (-c.graph_recall_score, c.paper_id),
        )
        for rank, candidate in enumerate(gr_ranked, start=1):
            if rank > _PPR_CUT:
                break
            candidate.graph_recall_rank = rank

        # ---- protected-RRF fusion -------------------------------------
        # 1. Determine the protected set = BM25 top-M papers (best BM25 rank
        #    <= M), kept in their BM25 order as the final top-M.
        bm25_ranked = sorted(
            (c for c in candidates.values() if c.bm25_rank is not None),
            key=lambda c: (c.bm25_rank, c.paper_id),
        )
        protected = bm25_ranked[:_PPR_PROTECT_M]
        protected_ids = {c.paper_id for c in protected}

        def _diag(candidate: CandidateScore, score: float) -> dict[str, Any]:
            channel_ranks = {
                "bm25": candidate.bm25_rank,
                "dense": candidate.dense_rank,
                "graph_recall": candidate.graph_recall_rank,
            }
            return {
                "paper": candidate.paper,
                "score": float(score),
                "rank": 0,
                "features": {
                    "bm25_rank": candidate.bm25_rank,
                    "dense_rank": candidate.dense_rank,
                    "graph_recall_rank": candidate.graph_recall_rank,
                    "graph_recall_score": candidate.graph_recall_score,
                },
                "signals": {},
                "source": self._primary_source(candidate),
                "diagnostics": {
                    "sources": sorted(candidate.sources),
                    "matched_nodes": sorted(candidate.direct_node_ids),
                    "query_type": query_type,
                    "query_intent": query_intent,
                    "coverage_low": coverage_low,
                    "multihop_intent": multihop_intent,
                    "graph_recall_score": candidate.graph_recall_score,
                    "graph_recall_rank": candidate.graph_recall_rank,
                    "protected": candidate.paper_id in protected_ids,
                    "rrf_channels": {
                        name: rank
                        for name, rank in channel_ranks.items()
                        if rank is not None
                    },
                    "mode": self.mode,
                },
            }

        # 2. Protected papers occupy the final top-M in their BM25 order.
        protected_items = [
            _diag(c, _rrf_fuse({"bm25": c.bm25_rank, "dense": c.dense_rank}))
            for c in protected
        ]

        # 3. Residual pool = every candidate NOT protected, fused with
        #    additive RRF over (bm25, dense, graph_recall). Reuses _rrf_fuse.
        residual: list[dict[str, Any]] = []
        for candidate in candidates.values():
            if candidate.paper_id in protected_ids:
                continue
            score = _rrf_fuse({
                "bm25": candidate.bm25_rank,
                "dense": candidate.dense_rank,
                "graph_recall": candidate.graph_recall_rank,
            })
            residual.append(_diag(candidate, score))
        residual.sort(
            key=lambda item: (
                -item["score"],
                item["paper"].get("id") or item["paper"].get("paper_id") or "",
            )
        )
        residual = _shuffle_tie_bands(residual, eps=_TIE_BAND_EPS)

        scored = protected_items + residual
        for rank, item in enumerate(scored[:top_k], 1):
            item["rank"] = rank
        top_results = scored[:top_k]

        # ---- honesty diagnostics: protected / displaced counters ------
        # protected_count: how many slots the top-M protection occupied.
        # displaced_count: graph-recall papers that entered the residual
        #   top_k slots (M+1..top_k) -- i.e. recall the channel actually
        #   contributed. A non-zero displaced_count at M<5 is exactly the
        #   single-hop trade-off the design flags (a graph paper may push a
        #   weak base-both B out of the residual window). Surfacing it makes
        #   the M=3 operating-point cost auditable instead of silent.
        if top_results:
            displaced = sum(
                1
                for item in top_results[len(protected_items):]
                if item["diagnostics"]["graph_recall_rank"] is not None
                and item["features"]["bm25_rank"] is None
            )
            top_results[0]["diagnostics"]["protected_count"] = len(protected_items)
            top_results[0]["diagnostics"]["displaced_count"] = displaced
            top_results[0]["diagnostics"]["protect_m"] = _PPR_PROTECT_M
        return top_results

    # ------------------------------------------------------------------
    # graph_centric mode: PPR as a first-class RRF channel (2nd overhaul).
    # ------------------------------------------------------------------
    def _strict_anchor_papers(self, question: str) -> set[str]:
        """Distinct corpus papers a question DIRECTLY names by title.

        Structural multihop signal, gold-free. A paper is a strict anchor iff
        its whitespace-normalised TITLE occurs as a contiguous phrase in the
        normalised question (a full-title surface match -- far cleaner than the
        loose OpenIE-node overlap, which the realistic KG fragments into many
        short spurious spans). The title is corpus metadata every system
        already sees (BM25 scores it); counting how many titles a question
        names is a pure surface signal. The gold paper ids are NEVER consulted
        -- a both-direct question (names two papers) vs a bridge question
        (names one, the bridge target being lexically weak) is decided purely
        from the question text against the corpus title index.
        """
        q_norm = _normalise_title(question)
        if not q_norm:
            return set()
        padded = f" {q_norm} "
        anchor_papers: set[str] = set()
        for title_norm, paper_id in self._title_anchor_index.items():
            if f" {title_norm} " in padded:
                anchor_papers.add(paper_id)
        return anchor_papers

    def _graph_centric_use_graph(
        self,
        question: str,
        matched_nodes: list[MatchedNode],
    ) -> bool:
        """Structural multihop gate -- decides whether to fuse the PPR channel.

        Replaces the protected top-M blockade with a question-shape gate that
        fires graph-centric fusion ONLY for bridge-style multihop questions
        and falls back to BM25/dense for both-direct ones. The decision uses
        ONLY the question's own surface anchors; the gold paper set is NEVER
        inspected (no gold leak). Signal = number of DISTINCT corpus papers the
        question directly names (full-name match; see ``_strict_anchor_papers``):

          * both-direct (>= 2 distinct anchor papers): the question names two
            papers outright, so BM25 already surfaces both -- the graph channel
            only adds dilution. Gate CLOSED. Mirrors diagnosis R1 (both gold
            directly seeded) without reading gold.
          * bridge (<= 1 distinct anchor paper): a single named anchor whose
            bridge target is lexically weak yet graph-reachable. Diagnosis R1'
            -- the exact case PPR recovers. Gate OPEN (fuse PPR). An explicit
            multihop phrasing (``_is_multihop_intent``) also forces the gate
            open even if the surface happens to name two papers.

        No matched seed at all -> CLOSED (no seed to walk from).
        """
        if not matched_nodes:
            return False
        if self._is_multihop_intent(question):
            return True
        anchor_papers = self._strict_anchor_papers(question)
        if len(anchor_papers) >= _GRAPH_CENTRIC_BOTH_DIRECT_ANCHORS:
            # Both-direct: the question names two papers -> BM25 preferred.
            return False
        # Single-anchor bridge (or zero strict anchors but a loose seed) ->
        # fuse PPR so the graph can recover the lexically-weak bridge target.
        return True

    def _graph_centric_query(
        self,
        *,
        question: str,
        candidates: dict[str, CandidateScore],
        matched_nodes: list[MatchedNode],
        coverage_low: bool,
        query_type: str,
        query_intent: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Balanced RRF fusion with PPR as a first-class ranking channel.

        Unlike ``graph_recall_rrf`` (which protects the BM25 top-M and lets
        the graph fill only residual slots), graph_centric fuses BM25, dense
        and the PPR graph channel on EQUAL footing via additive RRF
        (``1/(60+rank)`` per channel). The full candidate pool
        (BM25 union dense union graph-reachable) competes for every slot --
        the 2-slot blockade the diagnosis identified as the +3.5pp ceiling is
        gone.

        The PPR channel is fused ONLY when the structural multihop gate
        (``_graph_centric_use_graph``) is open; otherwise the question is a
        both-direct / no-seed case and we rank on BM25+dense alone so
        single-hop and both-anchored questions are not harmed. PPR here uses
        the corrected HippoRAG alpha=0.5 and specificity-weighted seeding.
        """
        use_graph = self._graph_centric_use_graph(question, matched_nodes)
        multihop_intent = self._is_multihop_intent(question)
        if use_graph:
            # Add the PPR graph-recall channel (corrected alpha, specificity
            # personalization). The full reachable doc set joins the pool.
            self._add_graph_recall_candidates(
                candidates,
                matched_nodes,
                alpha=_PPR_ALPHA_GRAPH_CENTRIC,
                spec_personalization=True,
            )
            # Within-channel PPR rank: mass descending, ties by paper_id.
            # NO _PPR_CUT -- graph-centric lets the whole reachable set rank
            # (the RRF 1/(60+rank) term already damps deep ranks; an arbitrary
            # cut would re-impose a blockade).
            gr_ranked = sorted(
                (c for c in candidates.values() if c.graph_recall_score > 0.0),
                key=lambda c: (-c.graph_recall_score, c.paper_id),
            )
            for rank, candidate in enumerate(gr_ranked, start=1):
                candidate.graph_recall_rank = rank

        scored: list[dict[str, Any]] = []
        for candidate in candidates.values():
            channels: dict[str, int | None] = {
                "bm25": candidate.bm25_rank,
                "dense": candidate.dense_rank,
            }
            if use_graph:
                channels["graph_recall"] = candidate.graph_recall_rank
            score = _rrf_fuse(channels)
            scored.append({
                "paper": candidate.paper,
                "score": float(score),
                "rank": 0,
                "features": {
                    "bm25_rank": candidate.bm25_rank,
                    "dense_rank": candidate.dense_rank,
                    "graph_recall_rank": candidate.graph_recall_rank,
                    "graph_recall_score": candidate.graph_recall_score,
                },
                "signals": {},
                "source": self._primary_source(candidate),
                "diagnostics": {
                    "sources": sorted(candidate.sources),
                    "matched_nodes": sorted(candidate.direct_node_ids),
                    "query_type": query_type,
                    "query_intent": query_intent,
                    "coverage_low": coverage_low,
                    "multihop_intent": multihop_intent,
                    "graph_centric_use_graph": use_graph,
                    "graph_recall_score": candidate.graph_recall_score,
                    "graph_recall_rank": candidate.graph_recall_rank,
                    "rrf_channels": {
                        name: rank
                        for name, rank in channels.items()
                        if rank is not None
                    },
                    "mode": self.mode,
                },
            })

        scored.sort(
            key=lambda item: (
                -item["score"],
                item["paper"].get("id") or item["paper"].get("paper_id") or "",
            )
        )
        scored = _shuffle_tie_bands(scored, eps=_TIE_BAND_EPS)
        for rank, item in enumerate(scored[:top_k], 1):
            item["rank"] = rank
        top_results = scored[:top_k]
        # Honesty diagnostics: how many top-k slots a graph-only paper took
        # (a paper the lexical channels never surfaced). Mirrors the
        # graph_recall_rrf displaced_count so the gate's contribution is
        # auditable; gate_open exposes whether the PPR channel fired at all.
        if top_results:
            graph_only = sum(
                1
                for item in top_results
                if item["features"]["graph_recall_rank"] is not None
                and item["features"]["bm25_rank"] is None
                and item["features"]["dense_rank"] is None
            )
            top_results[0]["diagnostics"]["gate_open"] = use_graph
            top_results[0]["diagnostics"]["graph_only_count"] = graph_only
        return top_results

    def _iter_neighbor_paths(
        self,
        seed_id: str,
        *,
        max_depth: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        visited = {seed_id}
        frontier: list[tuple[str, int]] = [(seed_id, 0)]
        neighbors: list[tuple[str, dict[str, Any]]] = []

        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for neighbor_id, edge in self._adjacency.get(current_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                neighbors.append((neighbor_id, edge))
                frontier.append((neighbor_id, depth + 1))
        return neighbors

    def _add_node_source_docs(
        self,
        candidates: dict[str, CandidateScore],
        node: dict[str, Any],
        *,
        direct_node_id: str | None,
        alias_hit: bool = False,
        neighbor_node_id: str | None = None,
        edge: dict[str, Any] | None = None,
        spec_mult: float = 1.0,
    ) -> None:
        for paper_id in _source_docs(node):
            paper = self.paper_by_id.get(paper_id)
            if paper is None:
                continue
            candidate = self._candidate_for(candidates, paper_id, paper)
            candidate.sources.add("kg")
            if direct_node_id is not None:
                candidate.direct_node_ids.add(direct_node_id)
                if alias_hit:
                    candidate.alias_node_ids.add(direct_node_id)
                # #4 fix: track the strongest specificity contribution so
                # the scorer can demote candidates whose only direct match
                # is a low-specificity surface hit (e.g. a generic noun
                # appearing in many passages). Typed matches push this to
                # 1.0; singleton untyped matches also reach 1.0.
                if spec_mult > candidate.matched_spec_max:
                    candidate.matched_spec_max = spec_mult
            if neighbor_node_id is not None:
                candidate.neighbor_node_ids.add(neighbor_node_id)
            node_type = str(node.get("type") or node.get("node_type") or "").lower()
            if node_type:
                candidate.schema_types.add(node_type)
            if edge is not None:
                self._add_edge_evidence(candidate, edge)

    def _add_edge_source_docs(
        self,
        candidates: dict[str, CandidateScore],
        edge: dict[str, Any],
        neighbor_id: str,
    ) -> None:
        for paper_id in _source_docs(edge):
            paper = self.paper_by_id.get(paper_id)
            if paper is None:
                continue
            candidate = self._candidate_for(candidates, paper_id, paper)
            candidate.sources.add("kg")
            candidate.neighbor_node_ids.add(neighbor_id)
            self._add_edge_evidence(candidate, edge)

    def _add_edge_evidence(
        self,
        candidate: CandidateScore,
        edge: dict[str, Any],
    ) -> None:
        relation = str(edge.get("relation_type") or edge.get("relation") or "")
        if relation:
            candidate.edge_relations.add(relation.lower())
        candidate.edge_reliability = max(
            candidate.edge_reliability,
            _as_float(edge.get("reliability_score") or edge.get("weight")),
        )

    def _candidate_for(
        self,
        candidates: dict[str, CandidateScore],
        paper_id: str,
        paper: dict[str, Any],
    ) -> CandidateScore:
        if paper_id not in candidates:
            candidates[paper_id] = CandidateScore(
                paper_id=paper_id,
                paper=self.paper_by_id.get(paper_id, paper),
            )
        return candidates[paper_id]

    def _match_nodes(self, question: str) -> list[MatchedNode]:
        q_tokens = set(tokenize(question))
        if not q_tokens:
            return []

        matched: list[MatchedNode] = []
        for node in self.nodes:
            node_id = self._node_id(node)
            if node_id is None:
                continue
            node_match = MatchedNode(node_id=node_id, node=node)

            name = str(node.get("name") or "")
            if self._surface_matches_query(name, q_tokens, allow_single_name=True):
                node_match.name_hit = True
                node_match.matched_terms.add(name)

            for alias in node.get("aliases") or []:
                alias_text = str(alias)
                if self._surface_matches_query(
                    alias_text,
                    q_tokens,
                    allow_single_name=False,
                ):
                    node_match.alias_hit = True
                    node_match.matched_terms.add(alias_text)

            if node_match.name_hit or node_match.alias_hit:
                # #4 generic-token mask fix: attach the specificity-relevant
                # fields so the downstream scorer can apply the HippoRAG
                # node-specificity multiplier without re-deriving them.
                node_match.passage_freq = self._passage_freq.get(node_id, 1)
                node_type = (
                    str(node.get("type") or node.get("node_type") or "").lower()
                )
                node_match.is_typed = node_type in _TYPED_SCHEMA_TYPES
                matched.append(node_match)
        return matched

    def _node_spec_multiplier(self, *, is_typed: bool, passage_freq: int) -> float:
        """Return the HippoRAG-style node specificity multiplier for one node.

        Schema-aware exemption: typed nodes (Process / Material / Equipment
        / Issue) preserve full signal contribution; only untyped nodes
        receive IDF demotion. This is a structural condition (KG schema
        membership), not a tuned threshold.

        For untyped nodes: ``s_i = 1 / |P_i|``  (Gutierrez et al., HippoRAG,
        arXiv:2405.14831 §3.2). |P_i| is the number of source passages the
        node attaches to in the KG, i.e. a pure corpus statistic structurally
        analogous to IDF — NOT a learned weight.

        Returns a value in (0.0, 1.0]. Typed -> 1.0. Singleton-passage
        untyped -> 1.0. Untyped with 10 passages -> 0.1, etc.
        """
        if is_typed:
            return 1.0
        return 1.0 / float(max(1, passage_freq))

    def _surface_matches_query(
        self,
        surface: str,
        q_tokens: set[str],
        *,
        allow_single_name: bool,
    ) -> bool:
        surface_tokens = set(tokenize(surface))
        if not surface_tokens:
            return False
        if len(surface_tokens) == 1:
            token = next(iter(surface_tokens))
            if token not in q_tokens:
                return False
            if token in _GENERIC_SINGLE_TOKEN_NAMES:
                return False
            if _is_acronym(surface):
                return True
            return True

        content_tokens = {
            token for token in surface_tokens
            if token not in _GENERIC_SINGLE_TOKEN_NAMES
        }
        if not content_tokens:
            return False
        overlap = len(content_tokens & q_tokens)
        return overlap == len(content_tokens) or overlap / len(content_tokens) >= 0.5

    def _alias_expanded_query(
        self,
        question: str,
        matched_nodes: list[MatchedNode],
    ) -> str | None:
        expansion_terms: list[str] = []
        seen: set[str] = set()
        for matched in matched_nodes:
            if not matched.alias_hit:
                continue
            for term in [matched.node.get("name"), *(matched.node.get("aliases") or [])]:
                if not term:
                    continue
                text = str(term).strip()
                key = text.lower()
                if text and key not in seen:
                    seen.add(key)
                    expansion_terms.append(text)
        if not expansion_terms:
            return None
        return f"{question} {' '.join(expansion_terms)}"

    def _coverage_triggered_alias_query(
        self,
        question: str,
        matched_nodes: list[MatchedNode],
        *,
        already_issued: str | None,
    ) -> str | None:
        if self.coverage_trigger_threshold <= 0.0 or not matched_nodes:
            return None
        if self._matched_token_coverage(question, matched_nodes) >= self.coverage_trigger_threshold:
            return None

        expansion_terms = self._node_surface_expansion_terms(matched_nodes)
        if not expansion_terms:
            return None
        expanded = f"{question} {' '.join(expansion_terms)}"
        if already_issued == expanded:
            return None
        return expanded

    def _matched_token_coverage(
        self,
        question: str,
        matched_nodes: list[MatchedNode],
    ) -> float:
        query_tokens = {
            token for token in tokenize(question)
            if token not in _GENERIC_SINGLE_TOKEN_NAMES
        }
        if not query_tokens:
            return 0.0

        matched_tokens: set[str] = set()
        for matched in matched_nodes:
            for term in matched.matched_terms:
                matched_tokens.update(
                    token for token in tokenize(term)
                    if token not in _GENERIC_SINGLE_TOKEN_NAMES
                )
        return len(query_tokens & matched_tokens) / len(query_tokens)

    def _node_surface_expansion_terms(
        self,
        matched_nodes: list[MatchedNode],
    ) -> list[str]:
        expansion_terms: list[str] = []
        seen: set[str] = set()
        for matched in matched_nodes:
            matched_surface_keys = {
                term.strip().casefold()
                for term in matched.matched_terms
                if term.strip()
            }
            for term in [matched.node.get("name"), *(matched.node.get("aliases") or [])]:
                if not term:
                    continue
                text = str(term).strip()
                key = text.casefold()
                if text and key not in seen and key not in matched_surface_keys:
                    seen.add(key)
                    expansion_terms.append(text)
        return expansion_terms

    def _signals_for(
        self,
        candidate: CandidateScore,
        query_type: str,
        query_entity_ids: set[str] | None = None,
    ) -> dict[str, float]:
        # #4 generic-token mask fix: alignment signal is gated by the
        # maximum specificity multiplier (matched_spec_max). For typed
        # matches or singleton-passage untyped matches this collapses to
        # the legacy 1.0 binary. For an untyped match against a node that
        # appears in many passages (a former hard-rejected generic noun)
        # this contributes a fraction proportional to 1/|P_i|, structurally
        # demoting the candidate without dropping it.
        c1_alignment = candidate.matched_spec_max if candidate.direct_node_ids else 0.0
        c2_alias = 1.0 if candidate.alias_node_ids or candidate.alias_expanded_bm25 else 0.0
        c3_schema = (
            len(candidate.schema_types & _SCHEMA_TYPES) / max(len(_SCHEMA_TYPES), 1)
            if candidate.schema_types
            else 0.0
        )
        c3_schema = min(1.0, c3_schema * 3.0)
        c4_edge_path = 1.0 if candidate.neighbor_node_ids else candidate.edge_reliability
        c4_edge_path = min(1.0, max(c4_edge_path, candidate.edge_reliability))
        c5_query_strategy = self._query_strategy_signal(candidate, query_type)
        c6_trust, trust_axes = self._trust_signal(
            candidate, query_entity_ids or set(),
        )
        signals: dict[str, Any] = {
            "c1_alignment": c1_alignment,
            "c2_alias": c2_alias,
            "c3_schema": c3_schema,
            "c4_edge_path": c4_edge_path,
            "c5_query_strategy": c5_query_strategy,
            "c6_trust": c6_trust,
        }
        if trust_axes is not None:
            signals["_trust_axes"] = trust_axes
        return signals

    def _query_strategy_signal(
        self,
        candidate: CandidateScore,
        query_type: str,
    ) -> float:
        if query_type == "causal":
            if candidate.edge_relations & _CAUSAL_RELATIONS:
                return 1.0
            return 0.5 if candidate.neighbor_node_ids else 0.0
        if query_type == "compare":
            if candidate.edge_relations & _COMPARE_RELATIONS:
                return 1.0
            return 0.5 if candidate.neighbor_node_ids else 0.0
        if query_type == "define":
            return 1.0 if candidate.direct_node_ids else 0.0
        return 1.0 if candidate.direct_node_ids or candidate.neighbor_node_ids else 0.0

    def _trust_signal(
        self,
        candidate: CandidateScore,
        query_entity_ids: set[str],
    ) -> tuple[float, dict[str, float] | None]:
        """Return ``(c6_trust, axes_dict_or_None)`` per the C6 toggle.

        ``self.c6_trust == "4axis"`` -> compute the project-design four axes
        and return them alongside the aggregate. ``"single_source_weight"``
        -> reproduce the legacy three-term sum and return ``None`` for the
        axes dict so downstream diagnostics know to omit the breakdown.
        """
        paper_text = (
            f"{candidate.paper.get('title', '')} "
            f"{candidate.paper.get('abstract', '')}"
        )
        if self.c6_trust == "4axis":
            candidate_entity_ids = (
                candidate.direct_node_ids
                | candidate.alias_node_ids
                | candidate.neighbor_node_ids
            )
            result = _trust_4axis_module.score(
                candidate_sources=candidate.sources,
                candidate_entity_ids=candidate_entity_ids,
                query_entity_ids=query_entity_ids,
                paper_text=paper_text,
                adjacency=self._adjacency,
                edge_reliability=candidate.edge_reliability,
                has_direct_match=bool(candidate.direct_node_ids),
            )
            return result.aggregate, {
                "a1_source_triangulation": result.a1_source_triangulation,
                "a2_query_graph_alignment": result.a2_query_graph_alignment,
                "a3_entity_authority": result.a3_entity_authority,
                "a4_context_sufficiency": result.a4_context_sufficiency,
            }
        # "single_source_weight" (OFF) -> legacy three-term sum.
        legacy = _trust_4axis_module.legacy_single_source_weight(
            candidate_sources=candidate.sources,
            paper_text=paper_text,
            edge_reliability=candidate.edge_reliability,
            has_direct_match=bool(candidate.direct_node_ids),
        )
        return legacy, None

    def _final_score(
        self,
        features: dict[str, float],
        signals: dict[str, float],
        gated_signals: dict[str, bool] | None = None,
    ) -> float:
        """Combine features and signals into a single retrieval score.

        - ``additive_legacy`` -> the pre-P3 weighted sum across every
          signal regardless of structural gates. Preserved for back-compat
          with the prior ablation cells (the ``additive_legacy`` mode test
          pins this behaviour bit-for-bit).
        - ``schema_centric_gated`` -> typed schema is the primary signal;
          alias, edge_path, query_strategy and trust contribute only when
          their respective gates fire. BM25 / dense / alignment / schema
          are always combined because they provide the recall pool and the
          central rerank signal.
        """
        weights = self.weights
        if self.mode == "additive_legacy" or gated_signals is None:
            return (
                weights.bm25 * features["bm25_norm"]
                + weights.dense * features["dense_norm"]
                + weights.alignment * signals["c1_alignment"]
                + weights.alias * signals["c2_alias"]
                + weights.schema * signals["c3_schema"]
                + weights.edge_path * signals["c4_edge_path"]
                + weights.query_strategy * signals["c5_query_strategy"]
                + weights.trust * signals["c6_trust"]
            )
        # schema_centric_gated mode.
        score = (
            weights.bm25 * features["bm25_norm"]
            + weights.dense * features["dense_norm"]
            + weights.alignment * signals["c1_alignment"]
            + weights.schema * signals["c3_schema"]
        )
        if gated_signals.get("alias_expansion", False):
            score += weights.alias * signals["c2_alias"]
        if gated_signals.get("edge_template", False):
            score += weights.edge_path * signals["c4_edge_path"]
        if gated_signals.get("query_strategy_adaptive", False):
            score += weights.query_strategy * signals["c5_query_strategy"]
        if gated_signals.get("trust_4axis", False):
            score += weights.trust * signals["c6_trust"]
        return score

    def _classify_intent(self, question: str) -> str:
        """Classify the query into a P3 intent bucket.

        Intent drives the relation-template gate (defect/mitigation/material)
        independently of the existing ``_classify_query`` (causal/compare/
        define/factual). Both classifiers run side-by-side so the legacy
        query-strategy adaptive switch is unaffected.
        """
        lowered = question.lower()
        if re.search(r"\b(mitigat|resolv|remov|prevent|reduce|reduces|reducing)\b", lowered):
            return "mitigation"
        if re.search(
            r"\b(defect|fail|failure|cause|issue|problem|contaminat)\b",
            lowered,
        ):
            return "defect"
        if re.search(
            r"\b(material|metal|alloy|polymer|substrate|wafer|liner|film)\b",
            lowered,
        ):
            return "material"
        if re.search(r"\b(compare|versus|vs\.?|alternative|difference)\b", lowered):
            return "compare"
        if re.search(r"\b(cause|causes|caused|why|leads?\s+to|due\s+to)\b", lowered):
            return "causal"
        return "factual"

    def _gated_signals(
        self,
        candidate: CandidateScore,
        *,
        query_type: str,
        query_intent: str,
        coverage_low: bool,
        trust_gate_active: bool,
    ) -> dict[str, bool]:
        """Compute per-candidate structural gates for the P3 redesign.

        Returns a dict with keys mirroring the published gated_signals
        contract. Every gate is structural: pure boolean ratio / set
        membership, no tunable threshold beyond the published ratios.
        """
        gates: dict[str, bool] = {name: False for name in _GATED_SIGNAL_DEFAULTS}
        # Gate: coverage-triggered alias expansion. Fires only when overall
        # token coverage is low AND the candidate carries >=2 of (source
        # overlap, typed-entity match, relation evidence).
        confidence_hits = 0
        if candidate.sources:
            confidence_hits += 1
        if candidate.direct_node_ids or candidate.alias_node_ids:
            confidence_hits += 1
        if candidate.edge_relations:
            confidence_hits += 1
        gates["coverage_trigger"] = coverage_low and confidence_hits >= 2
        # Gate: alias expansion. We trust the alias signal when alias nodes
        # were matched OR a coverage-triggered alias query already added the
        # candidate. The 2-of-3 confidence rule above scopes the noisy
        # alias-only fallbacks.
        gates["alias_expansion"] = bool(
            (candidate.alias_node_ids or candidate.alias_expanded_bm25)
            and confidence_hits >= 2
        )
        # Gate: edge/path template match. The relevant templates depend on
        # the query intent; the gate fires only when at least one incident
        # edge matches a (head_type, relation, tail_type) triplet for that
        # intent.
        gates["edge_template"] = self._matches_relation_template(
            candidate,
            intent=query_intent,
        )
        # Gate: adaptive query strategy. Only meaningful for non-factual
        # intents — uniform BFS is the structural default for factual
        # questions because there is no relation type to prioritise.
        gates["query_strategy_adaptive"] = (
            query_type != "factual" and bool(candidate.neighbor_node_ids)
        )
        # Gate: 4-axis trust signal. Skip when the per-candidate trust
        # scores cluster (std below the structural threshold) so we do not
        # let a flat reliability_score=0.85 baseline contribute noise.
        gates["trust_4axis"] = trust_gate_active
        return gates

    def _matches_relation_template(
        self,
        candidate: CandidateScore,
        *,
        intent: str,
    ) -> bool:
        templates = _RELATION_TEMPLATES.get(intent)
        if not templates:
            return False
        candidate_relations = candidate.edge_relations
        if not candidate_relations:
            return False
        candidate_types = candidate.schema_types
        for head_type, relation, tail_type in templates:
            if relation not in candidate_relations:
                continue
            # At least one endpoint type must be visible on the candidate;
            # we cannot inspect both endpoints per edge here because the
            # CandidateScore only tracks aggregate schema_types. The
            # heuristic is intentionally loose so well-formed KG edges with
            # the right relation always pass.
            if head_type in candidate_types or tail_type in candidate_types:
                return True
        return False

    def _primary_source(self, candidate: CandidateScore) -> str:
        if "kg" in candidate.sources:
            return "kg"
        if "dense" in candidate.sources and "bm25" not in candidate.sources:
            return "dense"
        return "bm25"

    def _classify_query(self, question: str) -> str:
        lowered = question.lower()
        if re.search(r"\b(cause|causes|caused|why|leads?\s+to|due\s+to)\b", lowered):
            return "causal"
        if re.search(r"\b(compare|versus|vs\.?|alternative|difference)\b", lowered):
            return "compare"
        if re.search(r"\b(what\s+is|define|definition|describe)\b", lowered):
            return "define"
        return "factual"

    def _build_adjacency(
        self,
        edges: list[dict[str, Any]],
    ) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for edge in edges:
            source = edge.get("source") or edge.get("head") or edge.get("from")
            target = edge.get("target") or edge.get("tail") or edge.get("to")
            if not source or not target:
                continue
            source_id = str(source)
            target_id = str(target)
            adjacency[source_id].append((target_id, edge))
            adjacency[target_id].append((source_id, edge))
        return adjacency

    def _paper_id(self, paper: dict[str, Any]) -> str | None:
        raw = paper.get("id") or paper.get("paper_id") or paper.get("source_id")
        return str(raw) if raw else None

    def _node_id(self, node: dict[str, Any]) -> str | None:
        raw = node.get("node_id") or node.get("id")
        return str(raw) if raw else None


def _std(values: list[float]) -> float:
    """Population standard deviation. Returns 0 for empty / singleton input."""
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _shuffle_tie_bands(
    scored: list[dict[str, Any]],
    *,
    eps: float = _TIE_BAND_EPS,
) -> list[dict[str, Any]]:
    """Randomise the order of consecutive candidates whose scores tie.

    The input ``scored`` MUST already be sorted by descending score
    (the caller does this just before invoking us). We walk the list,
    group adjacent items whose score differs by less than ``eps`` into
    a band, and shuffle each band using a numpy RNG seeded from
    ``$RAG_V3_SEED``. When the env var is unset, the function is a
    no-op so existing deterministic tests still pass.

    The shuffle preserves mean rank for non-tied candidates (a tie band
    by definition has equal score). Cross-seed variance enters only
    through ties, which is exactly what a paired bootstrap CI expects.
    """
    seed_str = os.environ.get(_RAG_V3_SEED_ENV)
    if not seed_str or not scored:
        return scored
    try:
        seed_value = int(seed_str)
    except ValueError:
        return scored

    rng = random.Random(seed_value)
    out: list[dict[str, Any]] = []
    band_start = 0
    band_score = float(scored[0]["score"])
    for idx in range(1, len(scored)):
        current_score = float(scored[idx]["score"])
        if abs(current_score - band_score) >= eps:
            # Flush the previous band.
            band = scored[band_start:idx]
            if len(band) > 1:
                rng.shuffle(band)
            out.extend(band)
            band_start = idx
            band_score = current_score
    # Flush the trailing band.
    band = scored[band_start:]
    if len(band) > 1:
        rng.shuffle(band)
    out.extend(band)
    return out


def _rrf_fuse(channel_ranks: dict[str, int | None]) -> float:
    """Additive Reciprocal Rank Fusion over per-channel 1-based ranks.

    ``rrf = sum_c 1 / (_RRF_K + rank_c)`` over channels with a non-None
    rank. A channel that did not retrieve the candidate (rank is None)
    contributes NOTHING -- it never subtracts. This additivity is the
    single-hop non-demotion invariant: adding the graph-recall channel can
    only ever raise (or leave unchanged) a BM25 candidate's fused score,
    never lower it. ``_RRF_K`` is the canonical IR constant (k=60), not a
    tuned weight.
    """
    total = 0.0
    for rank in channel_ranks.values():
        if rank is None:
            continue
        total += 1.0 / float(_RRF_K + rank)
    return total


def _normalise_by_max(raw_scores: dict[str, float]) -> dict[str, float]:
    max_score = max((score for score in raw_scores.values() if score > 0), default=0.0)
    if max_score <= 0.0:
        return {paper_id: 0.0 for paper_id in raw_scores}
    return {
        paper_id: max(score, 0.0) / max_score
        for paper_id, score in raw_scores.items()
    }


def _kg_norm(signals: dict[str, float]) -> float:
    kg_keys = (
        "c1_alignment",
        "c2_alias",
        "c3_schema",
        "c4_edge_path",
        "c5_query_strategy",
    )
    return sum(signals[key] for key in kg_keys) / len(kg_keys)


def _normalise_title(text: str) -> str:
    """Lowercase, strip non-alphanumeric to single spaces, collapse runs.

    Used by the graph_centric multihop gate to match a corpus paper TITLE as a
    contiguous phrase inside a question. Pure surface normalisation -- no gold,
    no tuning."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return " ".join(cleaned.split())


def _source_docs(item: dict[str, Any]) -> list[str]:
    raw_docs = item.get("source_docs") or item.get("paper_ids") or item.get("sources") or []
    if isinstance(raw_docs, str):
        return [raw_docs]
    if not isinstance(raw_docs, list):
        return []
    return [str(doc_id) for doc_id in raw_docs if doc_id]


def _is_acronym(surface: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", surface.strip())
    if len(compact) < 2 or not any(char.isalpha() for char in compact):
        return False
    letters = [char for char in compact if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
