"""QA/corpus text-overlap leakage audit (MASTER_PLAN D19).

Detects whether question/answer text has abnormally high overlap with corpus
papers that are NOT the designated answer_source.  Three overlap metrics:
Jaccard (token n-gram sets), ROUGE-L (longest common subsequence), and
TF-IDF cosine similarity.

CLI usage:
    python -m paper.eval.leakage_audit \\
        --qa paper/gold/qa_v4_from_kg.jsonl \\
        --manifest paper/corpus/arxiv_fulltext_strict_combined_v1/manifest.json \\
        --out paper/audit/leakage_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to",
    "is", "are", "was", "were", "be", "been", "with", "at", "by",
    "from", "as", "this", "that", "it", "its",
})


def _tokenize(text: str, ngram: int = 1) -> list[str]:
    """Lowercase word-tokenize *text*, drop stop-words, return n-gram tokens."""
    raw = re.findall(r"\b[a-z0-9]+\b", text.lower())
    tokens = [t for t in raw if t not in _STOP_WORDS]
    if ngram == 1:
        return tokens
    return [" ".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def _rouge_l(ref: list[str], hyp: list[str]) -> float:
    """ROUGE-L F1 via LCS."""
    m, n = len(ref), len(hyp)
    if m == 0 or n == 0:
        return 0.0
    # Space-efficient LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    p = lcs / n
    r = lcs / m
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _tfidf_cosine(tokens_a: list[str], tokens_b: list[str]) -> float:
    """TF-IDF cosine over 2-document micro-corpus."""
    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 0.0
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)
    idf: dict[str, float] = {}
    for term in vocab:
        df = (1 if term in tf_a else 0) + (1 if term in tf_b else 0)
        idf[term] = math.log(3.0 / (df + 1.0))  # smoothed IDF over 2 docs
    vec_a = {t: tf_a[t] * idf[t] for t in vocab}
    vec_b = {t: tf_b[t] * idf[t] for t in vocab}
    dot = sum(vec_a[t] * vec_b[t] for t in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Manifest / text loading
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Return paper_id -> manifest record dict."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records: list[dict[str, Any]] = []
    if isinstance(raw, list):
        records = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict):
        for key in ("files", "papers", "records", "sources", "documents", "items"):
            val = raw.get(key)
            if isinstance(val, list):
                records = [r for r in val if isinstance(r, dict)]
                break
    return {str(r.get("id") or r.get("arxiv_id") or ""): r for r in records if r}


def _load_paper_text(record: dict[str, Any]) -> str:
    """Best-effort: read text file, fall back to abstract."""
    text_path = record.get("text_path")
    if text_path:
        p = Path(text_path)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")[:20_000]
            except OSError:
                pass
    return str(record.get("abstract") or record.get("title") or "")


# ---------------------------------------------------------------------------
# Per-QA leakage scoring
# ---------------------------------------------------------------------------

def _qa_text(qa: dict[str, Any]) -> str:
    parts = [
        str(qa.get("question") or ""),
        str(qa.get("answer") or ""),
        str(qa.get("crawl_text") or ""),
    ]
    return " ".join(p for p in parts if p)


def _answer_source_ids(qa: dict[str, Any]) -> set[str]:
    src = qa.get("answer_source")
    if isinstance(src, list):
        return {str(s) for s in src}
    if src:
        return {str(src)}
    evid = qa.get("evidence_source_ids")
    if isinstance(evid, list):
        return {str(s) for s in evid}
    return set()


def _score_pair(
    qa_tokens: list[str],
    qa_ngrams: set[str],
    paper_tokens: list[str],
    paper_ngrams: set[str],
) -> dict[str, float]:
    return {
        "jaccard": _jaccard(qa_ngrams, paper_ngrams),
        "rouge_l": _rouge_l(qa_tokens, paper_tokens),
        "tfidf_cosine": _tfidf_cosine(qa_tokens, paper_tokens),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_leakage(
    qa_path: Path,
    manifest_path: Path,
    ngram: int = 3,
) -> dict[str, Any]:
    """Compute per-QA leakage scores against every corpus paper.

    Args:
        qa_path: Path to JSONL file with QA records.
        manifest_path: Path to corpus manifest JSON.
        ngram: N-gram size for overlap computation (default 3 for trigrams).

    Returns:
        Report dict with per_qa list and summary stats.
    """
    manifest = _load_manifest(manifest_path)
    qa_records: list[dict[str, Any]] = []
    for line in qa_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            qa_records.append(json.loads(line))

    # Pre-load all paper texts
    paper_cache: dict[str, tuple[list[str], set[str]]] = {}
    for pid, rec in manifest.items():
        text = _load_paper_text(rec)
        toks = _tokenize(text, ngram=ngram)
        paper_cache[pid] = (toks, set(toks))

    per_qa: list[dict[str, Any]] = []
    for qa in qa_records:
        qa_id = str(qa.get("qa_id") or qa.get("id") or "")
        answer_sources = _answer_source_ids(qa)
        qa_text = _qa_text(qa)
        qa_toks = _tokenize(qa_text, ngram=ngram)
        qa_ngrams = set(qa_toks)

        leak_scores: list[dict[str, Any]] = []
        for pid, (paper_toks, paper_ngrams) in paper_cache.items():
            # Normalise: strip "arxiv:" prefix for comparison
            clean_pid = pid.replace("arxiv:", "")
            is_answer_source = (
                pid in answer_sources
                or clean_pid in answer_sources
                or any(clean_pid in s or pid in s for s in answer_sources)
            )
            scores = _score_pair(qa_toks, qa_ngrams, paper_toks, paper_ngrams)
            leak_scores.append({
                "paper_id": pid,
                "is_answer_source": is_answer_source,
                "jaccard": scores["jaccard"],
                "rouge_l": scores["rouge_l"],
                "tfidf_cosine": scores["tfidf_cosine"],
            })

        # Flag non-answer-source papers with high overlap
        leaks = [
            s for s in leak_scores
            if not s["is_answer_source"]
            and (s["jaccard"] > 0.0 or s["rouge_l"] > 0.0)
        ]
        leaks_sorted = sorted(leaks, key=lambda x: x["jaccard"] + x["rouge_l"], reverse=True)
        worst_leak = leaks_sorted[0] if leaks_sorted else None

        # Per-metric group maxima (additive; consumed by the S5 relative gate).
        # These are computed over ALL papers in each group independently per
        # metric -- NOT the composite (jaccard+rouge_l) sort winner -- so a
        # downstream relative gate can compare answer-vs-nonanswer maxima per
        # metric correctly (incl. tfidf_cosine, which the composite sort and the
        # jaccard/rouge_l-only `leaks` filter both ignore).
        _ans_rows = [s for s in leak_scores if s["is_answer_source"]]
        _non_rows = [s for s in leak_scores if not s["is_answer_source"]]

        def _metric_max(rows: list[dict[str, Any]], metric: str) -> float:
            return max((float(r[metric]) for r in rows), default=0.0)

        _METRIC_KEYS = ("jaccard", "rouge_l", "tfidf_cosine")
        per_qa.append({
            "qa_id": qa_id,
            "answer_sources": list(answer_sources),
            "n_papers": len(leak_scores),
            "worst_non_source_leak": worst_leak,
            "answer_metric_max": {m: _metric_max(_ans_rows, m) for m in _METRIC_KEYS},
            "nonanswer_metric_max": {m: _metric_max(_non_rows, m) for m in _METRIC_KEYS},
            "flagged": (
                worst_leak is not None
                and (worst_leak["jaccard"] > 0.6 or worst_leak["rouge_l"] > 0.6)
            ),
        })

    # Summary statistics
    flagged = [r for r in per_qa if r["flagged"]]
    all_jaccard = [
        r["worst_non_source_leak"]["jaccard"]
        for r in per_qa
        if r["worst_non_source_leak"] is not None
    ]
    avg_worst_jaccard = sum(all_jaccard) / len(all_jaccard) if all_jaccard else 0.0

    # Top-5 worst leaks
    top5 = sorted(
        [r for r in per_qa if r["worst_non_source_leak"] is not None],
        key=lambda x: x["worst_non_source_leak"]["jaccard"]
        + x["worst_non_source_leak"]["rouge_l"],
        reverse=True,
    )[:5]

    return {
        "n_qa": len(qa_records),
        "n_papers": len(manifest),
        "ngram": ngram,
        "n_flagged": len(flagged),
        "avg_worst_non_source_jaccard": avg_worst_jaccard,
        "top5_worst_leaks": [
            {
                "qa_id": r["qa_id"],
                "paper_id": r["worst_non_source_leak"]["paper_id"],
                "jaccard": r["worst_non_source_leak"]["jaccard"],
                "rouge_l": r["worst_non_source_leak"]["rouge_l"],
            }
            for r in top5
        ],
        "per_qa": per_qa,
    }


def write_leakage_report(report: dict[str, Any], out_path: Path) -> None:
    """Write leakage report JSON to *out_path*, creating parent dirs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="QA/corpus text-overlap leakage audit (MASTER_PLAN D19)."
    )
    parser.add_argument("--qa", type=Path, required=True, help="Path to QA JSONL file.")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Path to corpus manifest JSON."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output path for leakage_report.json."
    )
    parser.add_argument("--ngram", type=int, default=3, help="N-gram size (default 3).")
    args = parser.parse_args(argv)

    report = compute_leakage(args.qa, args.manifest, ngram=args.ngram)
    write_leakage_report(report, args.out)

    print(f"n_qa: {report['n_qa']}")
    print(f"n_papers: {report['n_papers']}")
    print(f"n_flagged (jaccard|rouge_l > 0.6): {report['n_flagged']}")
    print(f"avg_worst_non_source_jaccard: {report['avg_worst_non_source_jaccard']:.4f}")
    print("top5 worst leaks (jaccard, qa_id, paper_id):")
    for row in report["top5_worst_leaks"]:
        print(f"  {row['jaccard']:.4f}  {row['qa_id']}  {row['paper_id']}")
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
