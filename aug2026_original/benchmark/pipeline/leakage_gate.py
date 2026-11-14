# -*- coding: utf-8 -*-
"""S5: leakage gate for the independent (acyclic) QA benchmark.

Inputs : the S4-survivor QA (qa_labeled.jsonl after S4 merge, or any QA jsonl)
         + the corpus manifest (paper text). Reuses paper/eval/leakage_audit.py
         verbatim for the Jaccard / ROUGE-L / TF-IDF overlap computation between
         each QA's (question + answer) text and EVERY corpus paper, split into
         answer papers vs NON-answer papers.

Gate    : a QA is "ambiguous / leaked" and DISCARDED when, for any overlap
         metric (jaccard, rouge_l, tfidf_cosine), the MAX overlap with a
         NON-answer paper equals or exceeds the MAX overlap with an answer paper. That
         means a wrong paper looks at least as textually on-topic as the gold
         paper, so the gold label is not text-defensible -> drop it. We ALSO
         drop a QA whose answer surface form is embedded verbatim in the
         question (a final safety net; S2/make_* already block most, this is the
         last line of defence).

Reuse   : the QA rows carry `answer_paper_ids` (gold_v2 schema). leakage_audit
         keys on `answer_source` / `evidence_source_ids`, so we inject
         answer_source = answer_paper_ids into a temp view before calling
         compute_leakage -- no modification to leakage_audit.py.

Output  : paper/audit/leakage_report_v2.json  -- the raw leakage_audit report
         PLUS a `gate` block (per-QA pass/fail + reasons) and the surviving QA
         id list. The driver also writes the passing QA back to a jsonl.

This module makes NO API call (text-only). NOT a hyperparameter knob: the gate
rule is "non-answer max >= answer max fails the QA" (ties fail) -- a structural relative comparison, not a
tuned absolute threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_PACKAGE_ROOT_FOR_IMPORT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT_FOR_IMPORT)

# Reuse the leakage audit machinery verbatim (design S5: leakage_audit reuse).
from paper.eval.leakage_audit import compute_leakage

# ---------------------------------------------------------------------------
# Intentionally NO import of prototype.baselines / proposed_kg / qa_from_kg.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "paper"
    / "corpus"
    / "arxiv_fulltext_strict_combined_v1"
    / "manifest.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "paper" / "audit" / "leakage_report_v2.json"

# Overlap metrics compared by the relative gate.
_METRICS = ("jaccard", "rouge_l", "tfidf_cosine")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _qa_id(qa: dict, fallback_index: int) -> str:
    """Stable id for a QA row (leakage_audit uses qa_id / id)."""
    return str(qa.get("qa_id") or qa.get("id") or "qa{0:05d}".format(fallback_index))


def prepare_audit_view(qa_rows: Sequence[dict], tmp_path: Path) -> Path:
    """Write a temp jsonl where each row carries the fields leakage_audit reads.

    leakage_audit keys the answer source on `answer_source` and scores
    question+answer+crawl_text. gold_v2 rows use `answer_paper_ids`, so we inject
    answer_source = answer_paper_ids and a stable qa_id. No other field changes.
    """
    tmp_path = Path(tmp_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        for i, qa in enumerate(qa_rows):
            view = dict(qa)
            view["qa_id"] = _qa_id(qa, i)
            # Prefer the post-S4 gold_paper_ids when present (single source of
            # truth for the gold label); fall back to answer_paper_ids.
            sources = qa.get("gold_paper_ids") or qa.get("answer_paper_ids") or []
            view["answer_source"] = list(sources)
            f.write(json.dumps(view, ensure_ascii=True))
            f.write("\n")
    return tmp_path


def _answer_surface_in_question(qa: dict) -> bool:
    """True if the answer surface form is embedded verbatim in the question."""
    answer = (qa.get("answer") or "").strip().lower()
    question = (qa.get("question") or "").lower()
    if not answer or not question:
        return False
    return answer in question


def compute_gate(
    qa_rows: Sequence[dict],
    manifest_path: Path,
    *,
    ngram: int = 3,
    tmp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the leakage audit and apply the relative leakage gate.

    Returns a dict:
      {report, gate: {n_qa, n_pass, n_fail, per_qa:[{qa_id, passed, reasons}]},
       passing_ids:[...], passing_qa:[...]}

    Gate rule (structural, relative -- not a tuned absolute threshold):
      A QA FAILS if, for ANY overlap metric, the max overlap with a NON-answer
      paper >= the max overlap with an answer paper (tie counts as fail: the
      gold paper is not strictly more on-topic than the best distractor). A QA
      also fails if its answer surface form is embedded verbatim in the question.

    NOTE on metric source: compute_leakage is called exactly ONCE. Its per-QA
    records expose per-metric group maxima (answer_metric_max /
    nonanswer_metric_max), computed independently per metric over the answer
    and non-answer paper groups; the relative gate compares those two maxima
    per metric. This keeps leakage_audit unmodified. (An earlier two-call
    design is obsolete; this docstring describes the shipped code.)
    """
    tmp_dir = Path(tmp_dir) if tmp_dir else (PROJECT_ROOT / "paper" / "audit")
    view_path = prepare_audit_view(qa_rows, tmp_dir / "_leakage_gate_view.jsonl")

    # SINGLE call: compute_leakage now exposes per-metric group maxima
    # (answer_metric_max / nonanswer_metric_max), computed over ALL papers in
    # each group INDEPENDENTLY PER METRIC -- not the composite (jaccard+rouge_l)
    # sort winner. This is the correct numerator/denominator for the per-metric
    # relative gate, and it covers tfidf_cosine (which the old composite-sort +
    # jaccard/rouge_l-only `worst_non_source_leak` path silently ignored).
    report = compute_leakage(view_path, Path(manifest_path), ngram=ngram)
    by_id = {r["qa_id"]: r for r in report["per_qa"]}

    per_qa_gate: List[dict] = []
    passing_ids: List[str] = []
    passing_qa: List[dict] = []
    for i, qa in enumerate(qa_rows):
        qid = _qa_id(qa, i)
        reasons: List[str] = []

        rec = by_id.get(qid, {})
        ans_max = rec.get("answer_metric_max")
        non_max = rec.get("nonanswer_metric_max")

        # Relative gate (structural, NO absolute threshold): for each metric, the
        # gold answer paper must be STRICTLY more on-topic than the best
        # non-answer paper. A QA FAILS if a non-answer paper's max overlap >= the
        # answer paper's max overlap on ANY metric (tie -> fail). When the
        # non-answer max is 0 on a metric there is nothing to leak, so skip it.
        if ans_max is not None and non_max is not None:
            for metric in _METRICS:
                non_v = float(non_max.get(metric, 0.0))
                ans_v = float(ans_max.get(metric, 0.0))
                if non_v > 0.0 and non_v >= ans_v:
                    reasons.append("nonsource_ge_answer:{0}".format(metric))
        else:
            # per_qa record missing for this id (should not happen); conservative.
            reasons.append("overlap_record_missing")

        # Final safety net: answer surface verbatim in question.
        if _answer_surface_in_question(qa):
            reasons.append("answer_surface_in_question")

        passed = not reasons
        per_qa_gate.append({"qa_id": qid, "passed": passed, "reasons": reasons})
        if passed:
            passing_ids.append(qid)
            passing_qa.append(qa)

    gate = {
        "n_qa": len(qa_rows),
        "n_pass": len(passing_ids),
        "n_fail": len(qa_rows) - len(passing_ids),
        "ngram": ngram,
        "rule": (
            "DISCARD if, for any metric, max overlap with a NON-answer paper "
            ">= max overlap with an answer paper (relative gate), OR the answer "
            "surface form is embedded verbatim in the question."
        ),
        "per_qa": per_qa_gate,
    }
    return {
        "report": report,
        "gate": gate,
        "passing_ids": passing_ids,
        "passing_qa": passing_qa,
    }


def write_report(result: Dict[str, Any], out_path: Path) -> None:
    """Write leakage_report_v2.json (audit report + gate block)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": result["report"],
        "gate": result["gate"],
        "passing_ids": result["passing_ids"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S5 leakage gate")
    parser.add_argument(
        "--pilot-dir",
        required=True,
        help="pilot directory holding qa_labeled.jsonl (post-S4)",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--passing-out",
        default=None,
        help="optional jsonl of surviving QA (default <pilot-dir>/qa_leak_passed.jsonl)",
    )
    parser.add_argument("--ngram", type=int, default=3)
    args = parser.parse_args(argv)

    pilot_dir = Path(args.pilot_dir)
    qa_path = pilot_dir / "qa_labeled.jsonl"
    qa_rows = load_jsonl(qa_path)

    result = compute_gate(qa_rows, Path(args.manifest), ngram=args.ngram)
    write_report(result, Path(args.report_out))

    passing_out = (
        Path(args.passing_out)
        if args.passing_out
        else pilot_dir / "qa_leak_passed.jsonl"
    )
    with open(passing_out, "w", encoding="utf-8", newline="\n") as f:
        for qa in result["passing_qa"]:
            f.write(json.dumps(qa, ensure_ascii=True))
            f.write("\n")

    gate = result["gate"]
    print(
        "S5 done: {0}/{1} QA pass the leakage gate -> {2}".format(
            gate["n_pass"], gate["n_qa"], passing_out
        )
    )
    print("report -> {0}".format(args.report_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
