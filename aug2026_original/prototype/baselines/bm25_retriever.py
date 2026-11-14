"""T4.1 BL1 BM25 sparse retrieval baseline (E7).

CLI
---
    python -m prototype.baselines.bm25_retriever \
        --corpus paper/corpus/manifest.json \
        --qa     paper/gold/qa_100.jsonl \
        --output paper/results/baseline_bm25.json \
        [--domain semiconductor] [--partition gold] [--top-k 5]

Output schema (per QA, JSON list under ``records``):
    {run_id, baseline_name, qa_id, hit_at_5, mrr, retrieved_entities, notes}

``retrieved_entities`` for a retrieval-only baseline records the *retrieved
paper titles* — downstream eval (T4.6) matches against
``qa.expected_entities`` either via substring or via the paper's title token
overlap.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .corpus_loader import (
    DEFAULT_MANIFEST,
    DEFAULT_QA,
    load_corpus_with_abstracts,
    load_qa,
)

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]+")


def tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric (incl. Hangul) tokenizer."""
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(text)


class BM25Baseline:
    """Wrap ``rank_bm25.BM25Okapi`` over (title + abstract) per paper."""

    def __init__(self, corpus_papers: List[Dict[str, Any]]):
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 is required. Install with: pip install rank_bm25"
            ) from exc
        self.papers = list(corpus_papers)
        self._tokenized = [
            tokenize(f"{p.get('title', '')} {p.get('abstract', '')}")
            for p in self.papers
        ]
        # rank_bm25 errors on empty docs — give them a sentinel token.
        for i, toks in enumerate(self._tokenized):
            if not toks:
                self._tokenized[i] = ["__empty__"]
        self._bm25 = BM25Okapi(self._tokenized)

    def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not q.strip():
            return []
        q_tokens = tokenize(q)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            {
                "paper": self.papers[i],
                "score": float(scores[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_idx)
        ]


# ---------------------------------------------------------------------------
# Eval helpers (shared with the dense baseline)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def hit_and_mrr(
    retrieved: List[Dict[str, Any]],
    expected_entities: List[str],
    *,
    title_token_threshold: float = 0.34,
) -> Dict[str, float]:
    """Compute hit@k + reciprocal rank for a single QA item.

    A retrieved paper counts as a hit when (a) any expected entity string is a
    substring of its title or abstract, OR (b) >=
    ``title_token_threshold`` of the expected-entity tokens appear in the
    paper's (title + abstract).
    """
    if not retrieved or not expected_entities:
        return {"hit_at_k": 0.0, "rr": 0.0, "matched_rank": 0}

    exp_norm = [_normalise(e) for e in expected_entities if e]
    exp_tokens = [set(tokenize(e)) for e in expected_entities if e]

    matched_rank = 0
    for i, item in enumerate(retrieved, 1):
        paper = item["paper"]
        haystack = _normalise(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        if any(e in haystack for e in exp_norm):
            matched_rank = i
            break
        paper_tokens = set(tokenize(haystack))
        for toks in exp_tokens:
            if not toks:
                continue
            overlap = len(toks & paper_tokens) / max(len(toks), 1)
            if overlap >= title_token_threshold:
                matched_rank = i
                break
        if matched_rank:
            break

    if not matched_rank:
        return {"hit_at_k": 0.0, "rr": 0.0, "matched_rank": 0}
    return {
        "hit_at_k": 1.0,
        "rr": 1.0 / matched_rank,
        "matched_rank": matched_rank,
    }


def build_records(
    baseline_name: str,
    run_id: str,
    qa_items: Iterable[Dict[str, Any]],
    retrieve_fn,
    *,
    top_k: int = 5,
    notes_suffix: str = "",
) -> List[Dict[str, Any]]:
    """Shared loop: run ``retrieve_fn(query, k)`` per QA item and score it."""
    records: List[Dict[str, Any]] = []
    for qa in qa_items:
        retrieved = retrieve_fn(qa["question"], top_k)
        metrics = hit_and_mrr(retrieved, qa.get("expected_entities") or [])
        records.append({
            "run_id": run_id,
            "baseline_name": baseline_name,
            "qa_id": qa.get("qa_id"),
            "domain": qa.get("domain"),
            "difficulty": qa.get("difficulty"),
            "hit_at_5": metrics["hit_at_k"],
            "mrr": metrics["rr"],
            "matched_rank": metrics["matched_rank"],
            "retrieved_entities": [
                {
                    "paper_id": r["paper"].get("id"),
                    "title": r["paper"].get("title"),
                    "score": r["score"],
                    "rank": r["rank"],
                }
                for r in retrieved
            ],
            "notes": notes_suffix,
        })
    return records


def summarise(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(len(records), 1)
    return {
        "n_qa": len(records),
        "hit_at_5_mean": sum(r["hit_at_5"] for r in records) / n,
        "mrr_mean": sum(r["mrr"] for r in records) / n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BM25 baseline (T4.1, BL1)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", default=None,
                        help="Restrict corpus to one of {semiconductor,construction,textile,steel}")
    parser.add_argument("--partition", default="gold",
                        choices=["gold", "eval", "all"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-papers", type=int, default=None)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip abstract fetching — use titles only.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    partition = None if args.partition == "all" else args.partition
    t0 = time.time()
    corpus = load_corpus_with_abstracts(
        args.corpus,
        domain=args.domain,
        partition=partition,
        limit=args.limit_papers,
        fetch=not args.no_fetch,
    )
    t_load = time.time() - t0
    logger.info("Loaded %d papers in %.1fs", len(corpus), t_load)

    bm25 = BM25Baseline(corpus)
    qa_items = load_qa(args.qa)
    if args.domain:
        qa_items = [q for q in qa_items if q.get("domain") == args.domain]
    run_id = f"bm25-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

    records = build_records(
        "BM25",
        run_id,
        qa_items,
        bm25.query,
        top_k=args.top_k,
        notes_suffix=f"domain={args.domain or 'all'};partition={args.partition}",
    )
    summary = summarise(records)
    summary.update({
        "run_id": run_id,
        "baseline_name": "BM25",
        "corpus_size": len(corpus),
        "qa_size": len(qa_items),
        "domain": args.domain,
        "partition": args.partition,
        "top_k": args.top_k,
        "elapsed_seconds": round(time.time() - t0, 2),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": records}
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
