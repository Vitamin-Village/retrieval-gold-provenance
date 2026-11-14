# -*- coding: utf-8 -*-
"""Independent-benchmark retrieval eval runner (D-D, 2026-05-29).

Measures retrievers against the FROZEN, acyclic independent benchmark
``paper/gold_v2/qa_independent_v1.jsonl`` (118 QA). This module IMPORTS
models, so it lives OUTSIDE ``paper/gold_v2`` -- the gold_v2 acyclicity
CI guard is gold-build-only; an eval using the models is expected and
correct.

What it does
------------
For each QA we run the question through a retriever, take the top-K paper
ids, and score four metrics. Every metric is computed by ONE function over
the SAME ``(top_paper_ids, retrieved_passage_texts)`` input for EVERY
system, so no retriever (in particular proposed_v3) gets a private scoring
advantage. There is NO tie-band fabrication: the per-QA score arrays are
the real, deterministic per-question outcomes, saved for downstream paired
bootstrap over the 118 questions.

Metrics (top-K, default K=5)
----------------------------
hit@5
    Gold-paper containment. Single-hop (one gold id): 1.0 iff that id is in
    the top-K. Multi-hop (two gold ids): 1.0 iff BOTH gold ids are in the
    top-K (all-or-nothing), else 0.0.
MRR
    Reciprocal rank of the gold paper. Single-hop: 1 / rank of the gold id
    in the ranked list (0 if absent). Multi-hop: we average the reciprocal
    ranks of the two gold papers (mean of 1/rank_a and 1/rank_b); a gold
    paper missing from the top-K contributes 0 to that average. This is a
    symmetric, retriever-agnostic definition that rewards ranking both
    bridge papers highly without collapsing to the single-paper case.
answer-span recall@5
    Text-grounded, KG-independent. The gold evidence text (each
    ``evidence_spans[i].span`` sliced out of its passage's text, falling
    back to the QA ``answer`` string) must appear in the concatenated
    passage text of the top-K retrieved papers. Reported as the fraction of
    gold evidence snippets so recovered (1.0 when the QA has a single
    snippet that is found). Identical passage pool and matching function for
    every system.
bridge-coverage@5 (multi-hop only)
    1.0 iff both gold papers are in the top-K, else 0.0. For single-hop QAs
    this metric is ``None`` (excluded from its aggregate).

Systems
-------
bm25         -> prototype.baselines.bm25_retriever.BM25Baseline (rank_bm25)
dense        -> run_retrieval_eval_v3._build_dense_retriever (BGE-M3 or the
                deterministic surrogate, per RAG_V3_DENSE_BACKEND)
proposed_v3  -> prototype.baselines.proposed_v3_retriever.ProposedV3Retriever
                (current full system; KG loaded from
                paper/baselines/proposed_kg_v3_strict_combined_v2/)

HippoRAG / MS-GraphRAG are not wired (no result loader present here). See
``_HOOK_SYSTEMS`` TODO below.

CLI
---
    python paper/eval/eval_independent.py \
        --qa paper/gold_v2/qa_independent_v1.jsonl \
        --systems bm25,dense,proposed_v3 \
        --out paper/results/independent_eval_v1.json \
        --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

# Ensure the repo root is importable when run as a script
# (``python paper/eval/eval_independent.py``); ``python -m`` / pytest already
# put it on sys.path.
_REPO_ROOT_FOR_IMPORT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_IMPORT)

# Reuse existing loaders / retriever wiring -- do NOT reimplement.
from prototype.eval.run_retrieval_eval_v3 import (
    DenseRetrieverConfig,
    _build_dense_retriever,
    _load_kg,
    _load_manifest_records,
    _paper_from_manifest_record,
    _record_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA = REPO_ROOT / "paper" / "gold_v2" / "qa_independent_v1.jsonl"
DEFAULT_PASSAGES = REPO_ROOT / "paper" / "gold_v2" / "passages.jsonl"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "paper"
    / "corpus"
    / "arxiv_fulltext_strict_combined_v1"
    / "manifest.json"
)
DEFAULT_KG_DIR = (
    REPO_ROOT / "paper" / "baselines" / "proposed_kg_v3_strict_combined_v2"
)
DEFAULT_OUT = REPO_ROOT / "paper" / "results" / "independent_eval_v1.json"
DEFAULT_TOP_K = 5

DEFAULT_SYSTEMS = ("bm25", "dense", "proposed_v3")
# Systems we can build here. HippoRAG / MS-GraphRAG would need a result
# loader (their retrievals are produced by external pipelines); no such
# loader exists in this module yet.
_HOOK_SYSTEMS: tuple[str, ...] = ()
# TODO(independent-eval): wire HippoRAG / MS-GraphRAG by loading their
# precomputed per-QA top-K paper-id lists from paper/results/* and feeding
# them through `score_record` exactly like the live retrievers, so the
# metric functions stay system-agnostic.


# ---------------------------------------------------------------------------
# Gold accessors
# ---------------------------------------------------------------------------
def gold_paper_ids(qa: Mapping[str, Any]) -> list[str]:
    """Gold paper ids for a QA.

    Prefers explicit ``gold_paper_ids`` (always present on multi-hop rows and
    on the handful of single-hop rows that carry it); otherwise falls back to
    ``answer_paper_ids`` (the single-hop default field). Order preserved,
    duplicates dropped.
    """
    raw = qa.get("gold_paper_ids") or qa.get("answer_paper_ids") or []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        pid = str(value).strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def is_multihop(qa: Mapping[str, Any]) -> bool:
    """Two gold papers => multi-hop. Uses gold ids, not just hop_count, so a
    mislabeled hop_count cannot desync the metric from the actual gold set."""
    return len(gold_paper_ids(qa)) >= 2


def gold_span_texts(
    qa: Mapping[str, Any], passages_by_id: Mapping[str, str]
) -> list[str]:
    """Evidence snippets to look for in answer-span recall.

    Each ``evidence_spans[i]`` slices ``span = [start, end]`` out of its
    passage's text (passage-local offsets). Snippets that cannot be resolved
    (missing passage / out-of-range span) fall back to nothing for that entry.
    If no usable snippet survives, fall back to the QA ``answer`` string so
    the metric is always defined.
    """
    snippets: list[str] = []
    for span in qa.get("evidence_spans") or []:
        if not isinstance(span, Mapping):
            continue
        passage_id = str(span.get("passage_id") or "")
        bounds = span.get("span")
        text = passages_by_id.get(passage_id)
        if (
            text is not None
            and isinstance(bounds, (list, tuple))
            and len(bounds) == 2
        ):
            start, end = int(bounds[0]), int(bounds[1])
            snippet = text[start:end].strip()
            if snippet:
                snippets.append(snippet)
    if not snippets:
        answer = str(qa.get("answer") or "").strip()
        if answer:
            snippets.append(answer)
    return snippets


# ---------------------------------------------------------------------------
# Metric primitives (system-agnostic: same input, same function, every system)
# ---------------------------------------------------------------------------
def hit_at_k(top_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """1.0 iff ALL gold ids appear in ``top_ids`` (single gold => contains
    that one; multi-hop => contains both)."""
    if not gold_ids:
        return 0.0
    top = set(top_ids)
    return 1.0 if all(gid in top for gid in gold_ids) else 0.0


def reciprocal_rank(top_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """MRR contribution for one QA.

    Single gold: reciprocal of that paper's 1-based rank (0 if absent).
    Multi gold: mean of each gold paper's reciprocal rank (absent => 0 term).
    """
    if not gold_ids:
        return 0.0
    rank_of: dict[str, int] = {}
    for rank, pid in enumerate(top_ids, start=1):
        rank_of.setdefault(pid, rank)
    rrs = [1.0 / rank_of[gid] if gid in rank_of else 0.0 for gid in gold_ids]
    return sum(rrs) / len(rrs)


def answer_span_recall(
    retrieved_text: str, span_texts: Sequence[str]
) -> float:
    """Fraction of gold evidence snippets found (case-insensitive substring,
    whitespace-normalized) in the concatenated top-K passage text."""
    if not span_texts:
        return 0.0
    haystack = _normalize(retrieved_text)
    found = sum(1 for s in span_texts if _normalize(s) and _normalize(s) in haystack)
    return found / len(span_texts)


def bridge_coverage(top_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Multi-hop: 1.0 iff both gold papers are in top-K (same predicate as
    multi-hop hit@k, named separately for the bridge aggregate)."""
    return hit_at_k(top_ids, gold_ids)


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).casefold()


# ---------------------------------------------------------------------------
# Per-QA scoring (the single shared path for every system)
# ---------------------------------------------------------------------------
def score_record(
    qa: Mapping[str, Any],
    top_ids: Sequence[str],
    *,
    passages_by_paper: Mapping[str, str],
    passages_by_id: Mapping[str, str],
) -> dict[str, Any]:
    """Score ONE QA given a retriever's ranked top-K paper ids.

    ``top_ids`` is already truncated to top-K by the caller. The same
    function is used for bm25 / dense / proposed_v3 (and any future hook
    system) so the evaluator is symmetric.
    """
    gold = gold_paper_ids(qa)
    multihop = is_multihop(qa)
    span_texts = gold_span_texts(qa, passages_by_id)
    retrieved_text = "\n".join(
        passages_by_paper.get(pid, "") for pid in top_ids
    )

    metrics: dict[str, Any] = {
        "hit_at_k": hit_at_k(top_ids, gold),
        "mrr": reciprocal_rank(top_ids, gold),
        "answer_span_recall_at_k": answer_span_recall(retrieved_text, span_texts),
        "bridge_coverage_at_k": (
            bridge_coverage(top_ids, gold) if multihop else None
        ),
    }
    return {
        "qa_id": _qa_id(qa),
        "qtype": str(qa.get("qtype") or "unknown"),
        "domain": str(qa.get("domain") or "unknown"),
        "hop_count": int(qa.get("hop_count") or (2 if multihop else 1)),
        "multihop": multihop,
        "gold_paper_ids": gold,
        "retrieved_paper_ids": list(top_ids),
        "metrics": metrics,
    }


def _qa_id(qa: Mapping[str, Any]) -> str:
    explicit = qa.get("qa_id") or qa.get("id")
    if explicit:
        return str(explicit)
    # Stable fallback id from question hash so per-QA arrays stay aligned.
    import hashlib

    digest = hashlib.sha1(
        str(qa.get("question") or "").encode("utf-8")
    ).hexdigest()
    return f"qa_{digest[:12]}"


# ---------------------------------------------------------------------------
# Aggregation -- preserves per-QA arrays for downstream paired bootstrap.
# ---------------------------------------------------------------------------
_NUMERIC_METRICS = ("hit_at_k", "mrr", "answer_span_recall_at_k")


def summarize_system(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall + per-qtype + per-domain means, plus per-QA score arrays.

    ``bridge_coverage_at_k`` is aggregated only over multi-hop QAs (its value
    is ``None`` elsewhere and excluded from the mean / array).
    """
    overall = {metric: _mean([r["metrics"][metric] for r in records]) for metric in _NUMERIC_METRICS}
    bridge_vals = [
        r["metrics"]["bridge_coverage_at_k"]
        for r in records
        if r["metrics"]["bridge_coverage_at_k"] is not None
    ]
    overall["bridge_coverage_at_k"] = _mean(bridge_vals) if bridge_vals else None
    overall["n_multihop"] = len(bridge_vals)

    by_qtype = _grouped(records, "qtype")
    by_domain = _grouped(records, "domain")

    # Per-QA arrays (aligned to qa_id order) for paired bootstrap. These are
    # the REAL per-question outcomes -- no tie-band shuffle, no fabricated
    # variance.
    per_qa = {
        "qa_id": [r["qa_id"] for r in records],
        **{
            metric: [r["metrics"][metric] for r in records]
            for metric in _NUMERIC_METRICS
        },
        "bridge_coverage_at_k": [
            r["metrics"]["bridge_coverage_at_k"] for r in records
        ],
        "multihop": [r["multihop"] for r in records],
    }

    return {
        "n_qa": len(records),
        "overall": overall,
        "by_qtype": by_qtype,
        "by_domain": by_domain,
        "per_qa": per_qa,
    }


def _grouped(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[str(rec.get(key) or "unknown")].append(rec)
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(buckets):
        group = buckets[name]
        block: dict[str, Any] = {
            "n_qa": len(group),
        }
        for metric in _NUMERIC_METRICS:
            block[metric] = _mean([r["metrics"][metric] for r in group])
        bridge_vals = [
            r["metrics"]["bridge_coverage_at_k"]
            for r in group
            if r["metrics"]["bridge_coverage_at_k"] is not None
        ]
        block["bridge_coverage_at_k"] = (
            _mean(bridge_vals) if bridge_vals else None
        )
        out[name] = block
    return out


def _mean(values: Sequence[float]) -> float:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {err}"
                ) from err
            records.append(record)
    return records


def build_passage_indexes(
    passages: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (passages_by_paper, passages_by_id).

    ``passages_by_paper`` concatenates every passage's text per paper (the
    frozen full-text pool every system retrieves over). ``passages_by_id``
    maps each passage_id to its raw text (for evidence-span slicing).
    """
    by_paper: dict[str, list[str]] = defaultdict(list)
    by_id: dict[str, str] = {}
    for passage in passages:
        text = str(passage.get("text") or "")
        paper_id = str(passage.get("paper_id") or "")
        passage_id = str(passage.get("passage_id") or "")
        if passage_id:
            by_id[passage_id] = text
        if paper_id:
            by_paper[paper_id].append(text)
    joined = {pid: "\n".join(parts) for pid, parts in by_paper.items()}
    return joined, by_id


def build_candidate_papers(
    manifest_records: list[dict[str, Any]],
    passages_by_paper: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build the retrieval corpus: one paper dict per manifest record, with
    its retrieval text set to the frozen passage-pool full text (identical
    input for BM25 / dense / proposed_v3).

    BM25Baseline scores ``title + abstract`` and ProposedV3 / dense read the
    paper body, so we overwrite ``abstract``/``content``/``text`` with the
    concatenated passages to guarantee every system sees the same evidence.
    Papers absent from the passage pool keep their manifest abstract.
    """
    papers: list[dict[str, Any]] = []
    for record in manifest_records:
        paper = _paper_from_manifest_record(record)
        pid = _record_id(paper)
        full_text = passages_by_paper.get(pid)
        if full_text:
            paper["abstract"] = full_text
            paper["content"] = full_text
            paper["text"] = full_text
        papers.append(paper)
    return papers


# ---------------------------------------------------------------------------
# Retriever construction (reuses the existing v3 wiring)
# ---------------------------------------------------------------------------
def build_retriever(
    system: str,
    papers: list[dict[str, Any]],
    *,
    kg_dir: Path | None,
    dense_config: DenseRetrieverConfig | None = None,
) -> Any:
    if system == "bm25":
        from prototype.baselines.bm25_retriever import BM25Baseline

        return BM25Baseline(papers)
    if system == "dense":
        return _build_dense_retriever(
            papers, config=dense_config or DenseRetrieverConfig.from_env()
        )
    if system in (
        "proposed_v3",
        "proposed_v4_graphrecall",
        "proposed_v5_graphcentric",
    ):
        from prototype.baselines.proposed_v3_retriever import (
            ProposedV3Retriever,
            V3Weights,
        )
        from prototype.baselines.bm25_retriever import BM25Baseline

        kg = _load_kg(kg_dir)
        bm25 = BM25Baseline(papers)
        weights = V3Weights()
        dense = None
        if weights.dense > 0.0:
            dense = _build_dense_retriever(
                papers, config=dense_config or DenseRetrieverConfig.from_env()
            )
        # Each non-default label flips ONLY the fusion mode; proposed_v3 keeps
        # the default schema_centric_gated mode (bit-identical to the prior
        # path). proposed_v4_graphrecall -> graph_recall_rrf (protected-RRF).
        # proposed_v5_graphcentric -> graph_centric (PPR as a first-class RRF
        # channel + structural multihop gate; 2nd overhaul).
        _MODE_BY_SYSTEM = {
            "proposed_v3": "schema_centric_gated",
            "proposed_v4_graphrecall": "graph_recall_rrf",
            "proposed_v5_graphcentric": "graph_centric",
        }
        mode = _MODE_BY_SYSTEM[system]
        return ProposedV3Retriever(
            papers=papers,
            bm25=bm25,
            dense=dense,
            nodes=kg.nodes,
            edges=kg.edges,
            weights=weights,
            mode=mode,
        )
    if system == "ppr":
        from paper.eval.ppr_retriever import PPRRetriever

        if kg_dir is None:
            raise ValueError("system='ppr' requires kg_dir to be set")
        kg = _load_kg(kg_dir)
        return PPRRetriever(papers, kg.nodes, kg.edges)
    raise ValueError(f"unsupported system: {system!r}")


def retrieve_top_ids(retriever: Any, question: str, top_k: int) -> list[str]:
    """Run a retriever and return its top-K paper ids (deduped, order kept)."""
    hits = retriever.query(question, top_k=top_k)
    ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        paper = hit.get("paper") if isinstance(hit, Mapping) else None
        if not isinstance(paper, Mapping):
            continue
        pid = _record_id(paper)
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)
        if len(ids) >= top_k:
            break
    return ids


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------
def evaluate_system(
    system: str,
    qa_records: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    *,
    passages_by_paper: Mapping[str, str],
    passages_by_id: Mapping[str, str],
    kg_dir: Path | None,
    top_k: int,
    dense_config: DenseRetrieverConfig | None = None,
) -> dict[str, Any]:
    retriever = build_retriever(
        system, papers, kg_dir=kg_dir, dense_config=dense_config
    )
    records: list[dict[str, Any]] = []
    for qa in qa_records:
        question = str(qa.get("question") or qa.get("query") or "")
        top_ids = retrieve_top_ids(retriever, question, top_k)
        records.append(
            score_record(
                qa,
                top_ids,
                passages_by_paper=passages_by_paper,
                passages_by_id=passages_by_id,
            )
        )
    summary = summarize_system(records)
    summary["system"] = system
    return {"summary": summary, "records": records}


def run(
    *,
    qa_path: Path,
    passages_path: Path,
    manifest_path: Path,
    kg_dir: Path | None,
    systems: Sequence[str],
    out_path: Path,
    top_k: int = DEFAULT_TOP_K,
    dense_config: DenseRetrieverConfig | None = None,
) -> dict[str, Any]:
    qa_records = load_jsonl(qa_path)
    passages = load_jsonl(passages_path)
    manifest_records = _load_manifest_records(manifest_path)
    passages_by_paper, passages_by_id = build_passage_indexes(passages)
    papers = build_candidate_papers(manifest_records, passages_by_paper)

    results: dict[str, Any] = {}
    for system in systems:
        if system in _HOOK_SYSTEMS:
            # No live loader -- skip with an explicit marker, never crash.
            results[system] = {
                "summary": {"system": system, "skipped": "no result loader"},
                "records": [],
            }
            continue
        results[system] = evaluate_system(
            system,
            qa_records,
            papers,
            passages_by_paper=passages_by_paper,
            passages_by_id=passages_by_id,
            kg_dir=kg_dir,
            top_k=top_k,
            dense_config=dense_config,
        )

    payload = {
        "benchmark": str(qa_path),
        "n_qa": len(qa_records),
        "top_k": top_k,
        "systems": list(systems),
        "metric_defs": {
            "hit_at_k": "single gold => contained in top-K; multi-hop => BOTH gold contained",
            "mrr": "single => 1/rank of gold; multi-hop => mean of the two golds' reciprocal ranks",
            "answer_span_recall_at_k": "fraction of evidence snippets found in top-K passage text (text-based, KG-independent)",
            "bridge_coverage_at_k": "multi-hop only: 1 iff both gold papers in top-K",
        },
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def _parse_systems(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independent-benchmark retrieval eval (frozen acyclic gold_v2)."
    )
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument(
        "--systems",
        default=",".join(DEFAULT_SYSTEMS),
        help="Comma-separated list. Default: bm25,dense,proposed_v3",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    payload = run(
        qa_path=args.qa,
        passages_path=args.passages,
        manifest_path=args.manifest,
        kg_dir=args.kg_dir,
        systems=_parse_systems(args.systems),
        out_path=args.out,
        top_k=args.top_k,
    )
    for system, result in payload["results"].items():
        overall = result["summary"].get("overall")
        if overall is None:
            print(f"{system}: {result['summary'].get('skipped', 'skipped')}")
            continue
        print(
            f"{system}: hit@{args.top_k}={overall['hit_at_k']:.3f} "
            f"mrr={overall['mrr']:.3f} "
            f"span_recall@{args.top_k}={overall['answer_span_recall_at_k']:.3f} "
            f"bridge@{args.top_k}="
            + (
                f"{overall['bridge_coverage_at_k']:.3f}"
                if overall["bridge_coverage_at_k"] is not None
                else "n/a"
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
