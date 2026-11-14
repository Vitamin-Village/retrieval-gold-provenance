# -*- coding: utf-8 -*-
"""S4: multihop bridge verification for the independent (acyclic) QA benchmark.

Inputs : the multihop_bridge candidates from S2 (qa_candidates.jsonl) or the
         multihop rows already carried in qa_labeled.jsonl, plus the S1 passage
         pool. Each multihop candidate carries answer_paper_ids=[A, B], a
         claimed `bridge` entity, and evidence_spans into both papers.
Process: a single cross-family VERIFIER backend (Gemini Flash, the SAME panel
         as S3 per design D-A: generator = Claude Sonnet -> verifier = a
         different family) is asked a strict YES/NO question:
           "Does the claimed bridge entity actually appear in BOTH paper A and
            paper B, and does it genuinely link a fact in A to a fact in B as a
            real 2-hop chain? If it appears in only one paper, or is a generic
            word, or the question is answerable from a SINGLE paper, answer NO."
         We reuse the EntityJudgeTask + judge_entity_tasks + ConsensusResult
         machinery from paper/eval/cross_judge.py (a YES/NO choice is just a
         2-option entity task), rather than adding a new task type.

WHY S4 catches weak bridges:
  - Without it, a generic single-token bridge (e.g. bridge="thickness") and
    questions phrased "...measured in THIS study" (single-hop, answerable from one
    paper) can slip through. S4 removes both, with TWO independent gates:
      (1) a deterministic STRUCTURAL pre-gate (no model): bridge must appear in
          both passages AND must not be a generic word AND the question must not
          be single-hop-phrased ("in this study/paper/work"). These mirror the
          S2 gates as a final safety net and are decided WITHOUT an API call, so
          a clearly-weak bridge is downgraded before the verifier is ever asked.
      (2) the cross-family verifier YES/NO judgment for the survivors. The
          prompt explicitly tells the verifier to answer NO when the bridge does
          not meaningfully link the two papers or when one paper alone suffices.
    A candidate is kept as hard-multihop (hop_count=2) ONLY if it passes BOTH
    the structural pre-gate AND gets a YES from the verifier. Anything else is
    DOWNGRADED to single-hop (kept on the strongest single paper) or DISCARDED,
    with the reason recorded.

Outputs: <pilot-dir>/multihop_verified.jsonl  -- one record per multihop
         candidate with the S4 verdict (KEPT_MULTIHOP / DOWNGRADED_SINGLE /
         DISCARDED) and the reason. KEPT_MULTIHOP rows carry gold_paper_ids=[A,B],
         gold_bridge_entities=[bridge], hop_count=2.
         The driver ALSO merges S4 verdicts back into qa_labeled.jsonl in place:
         multihop rows that fail are downgraded/dropped; survivors gain the gold_*
         fields. Non-multihop rows pass through untouched.

IAA hook (optional, design note): export_bridge_spotcheck_csv mirrors S3's
         human spot-check export so a blind human rater can label the same
         YES/NO bridge question for a Cohen-kappa IAA measurement. It is NOT
         required for the gate; compute_bridge_human_kappa reuses S3's
         compute_openai_human_kappa convention (returns None when unmeasured,
         never fabricates a value).

ACYCLICITY (CI guard, test_no_kg_imports): this module must NOT import
prototype.baselines, proposed_kg, or qa_from_kg. The bridge verdict is decided
from passage text + the cross-family verifier ONLY; the KG (its edges, nodes,
source_docs) is never referenced.

NO real API call is made by the unit tests (a stub verifier backend is
injected). Live verification is gated by GEMINI_API_KEY and the google
package via _build_default_verifier. DO NOT add hyperparameter tuning; the
structural gates below are fixed rules, not tunable thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PACKAGE_ROOT_FOR_IMPORT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT_FOR_IMPORT)

# Reuse the cross-judge consensus machinery (design S4: reuse cross_judge.py /
# RelationJudge pattern). A YES/NO bridge judgment is a 2-option entity task, so
# we reuse EntityJudgeTask + judge_entity_tasks verbatim. We also reuse S3's
# human-IAA helpers so the optional bridge-kappa hook stays consistent.
from paper.eval.cross_judge import (
    ABSTAIN_INDEX,
    STATUS_ACCEPTED,
    EntityJudgeTask,
    judge_entity_tasks,
)
from paper.gold_v2.verify_answer_papers import (
    compute_openai_human_kappa,
    index_passages_by_paper,
    load_jsonl,
    representative_excerpt,
)
# Reuse the SAME structural weak-bridge rules as S2 so S4 is a faithful final
# safety net (no divergent re-definition of "weak bridge").
from paper.gold_v2.generate_qa_candidates import (
    answer_in_passage,
    is_valid_bridge,
)

# ---------------------------------------------------------------------------
# Intentionally NO import of:
#   prototype.baselines, proposed_kg, qa_from_kg
# A static CI test (test_no_kg_imports.py) scans all gold_v2/*.py for these.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASSAGES = PROJECT_ROOT / "paper" / "gold_v2" / "passages.jsonl"

# Verdict tags (ASCII).
VERDICT_KEPT = "KEPT_MULTIHOP"
VERDICT_DOWNGRADED = "DOWNGRADED_SINGLE"
VERDICT_DISCARDED = "DISCARDED"

# Structural single-hop tell-tale phrases (design: catch "answerable from one
# paper" questions like "...measured in THIS study"). A genuine 2-hop bridge
# question must reference the cross-paper link, not a single self-contained
# "this study/work/paper". This is a fixed phrase list (structural rule), not a
# tuned threshold.
_SINGLE_HOP_PHRASE_RE = re.compile(
    r"\b(?:in|of|from|for)\s+this\s+(?:study|paper|work|article|research)\b",
    re.IGNORECASE,
)

# Verifier system prompt (ASCII-only; deterministic). Frames the bridge check as
# a 2-option YES/NO EntityJudgeTask so judge_entity_tasks' INDEX:<n> parser
# applies (0 = YES, 1 = NO).
BRIDGE_SYSTEM_PROMPT = (
    "You are a careful annotator verifying a 2-hop bridge question. You are "
    "given a question, a claimed bridge entity, and two paper excerpts (A and "
    "B). Answer YES only if the bridge entity actually appears in BOTH "
    "excerpts AND genuinely links a fact in A to a fact in B so that BOTH "
    "papers are needed to answer the question. Answer NO if the bridge appears "
    "in only one excerpt, if it is a generic word that does not meaningfully "
    "connect the two papers, or if the question can be answered from a SINGLE "
    "paper. The numbered options are 0: YES and 1: NO. Reply with the index of "
    "your choice. Use exactly the format INDEX: <n> on the final line. Do not "
    "output any other commentary."
)

# Option index meanings for the YES/NO task.
YES_INDEX = 0
NO_INDEX = 1


# ---------------------------------------------------------------------------
# Candidate selection + structural pre-gate.
# ---------------------------------------------------------------------------


def select_multihop_candidates(rows: Sequence[dict]) -> List[dict]:
    """Return rows that are multihop candidates (qtype multihop_bridge OR
    hop_count == 2 with two answer papers)."""
    out: List[dict] = []
    for r in rows:
        is_mh = r.get("qtype") == "multihop_bridge" or r.get("hop_count") == 2
        if is_mh and len(r.get("answer_paper_ids", [])) >= 2:
            out.append(r)
    return out


def structural_bridge_reason(
    candidate: dict, passages_by_paper: Dict[str, List[dict]]
) -> Optional[str]:
    """Deterministic weak-bridge pre-gate (NO model call).

    Returns a reason string if the candidate is structurally weak (so it must
    be downgraded/discarded WITHOUT asking the verifier), or None if it passes
    the structural gate and is worth a verifier YES/NO.

    Gates (all structural, mirroring S2 make_multihop_candidate):
      - bridge must be present and a genuine technical link (is_valid_bridge:
        rejects generic single tokens like "thickness").
      - the question must not be single-hop-phrased ("in this study/work").
      - the bridge surface form must appear in BOTH papers' passage text.
    """
    bridge = (candidate.get("bridge") or "").strip()
    question = candidate.get("question") or ""
    if not bridge:
        return "no_bridge_entity"
    if not is_valid_bridge(bridge):
        return "generic_or_single_token_bridge"
    if _SINGLE_HOP_PHRASE_RE.search(question):
        return "single_hop_phrasing"

    paper_ids = candidate.get("answer_paper_ids", [])[:2]
    if len(paper_ids) < 2:
        return "fewer_than_two_papers"
    # Bridge must appear in BOTH papers' concatenated passage text.
    for pid in paper_ids:
        paper_text = " ".join(
            p.get("text", "") for p in passages_by_paper.get(pid, [])
        )
        if not answer_in_passage(bridge, paper_text):
            return "bridge_absent_from_paper:{0}".format(pid)
    return None


# ---------------------------------------------------------------------------
# Build the YES/NO EntityJudgeTask for one multihop candidate.
# ---------------------------------------------------------------------------


def _evidence_passage_id_for(candidate: dict, paper_id: str) -> Optional[str]:
    """Return the passage_id of the evidence span for `paper_id`, or None."""
    for span in candidate.get("evidence_spans") or []:
        if span.get("paper_id") == paper_id:
            return span.get("passage_id")
    return None


def build_bridge_task(
    candidate: dict,
    cand_index: int,
    passages_by_paper: Dict[str, List[dict]],
) -> EntityJudgeTask:
    """Build a 2-option YES/NO EntityJudgeTask for a multihop bridge check.

    The two paper excerpts (preferring each paper's evidence passage) and the
    claimed bridge are placed in the entity_mention field; the candidate
    options are literally ["YES", "NO"] so the verifier returns INDEX: 0 (YES)
    or INDEX: 1 (NO).
    """
    paper_ids = candidate.get("answer_paper_ids", [])[:2]
    pid_a = paper_ids[0] if len(paper_ids) > 0 else ""
    pid_b = paper_ids[1] if len(paper_ids) > 1 else ""

    excerpt_a = representative_excerpt(
        passages_by_paper.get(pid_a, []),
        preferred_passage_id=_evidence_passage_id_for(candidate, pid_a),
    )
    excerpt_b = representative_excerpt(
        passages_by_paper.get(pid_b, []),
        preferred_passage_id=_evidence_passage_id_for(candidate, pid_b),
    )

    bridge = candidate.get("bridge") or ""
    question = candidate.get("question") or ""

    # Pack the full bridge-check context into entity_mention; build_entity_user_
    # prompt renders it verbatim. ASCII-safe (upstream S1/S2 normalised text).
    mention = (
        "Question: {q}\n"
        "Claimed bridge entity: {b}\n"
        "Paper A excerpt: {ea}\n"
        "Paper B excerpt: {eb}\n"
        "Answer YES (index 0) if the bridge is in BOTH excerpts and truly links "
        "A to B; otherwise NO (index 1)."
    ).format(q=question, b=bridge, ea=excerpt_a, eb=excerpt_b)

    task_id = "mh{0:05d}".format(cand_index)
    return EntityJudgeTask(
        task_id=task_id,
        domain=candidate.get("domain", "unknown"),
        paper_excerpt="(bridge context in mention)",
        entity_mention=mention,
        candidate_node_ids=("YES", "NO"),
        candidate_node_descriptions=(
            "the bridge appears in both papers and links A to B",
            "the bridge is one-sided, generic, or single-hop answerable",
        ),
    )


# ---------------------------------------------------------------------------
# Verification driver.
# ---------------------------------------------------------------------------


def verify_bridges(
    candidates: Sequence[dict],
    passages: Sequence[dict],
    verifier_backend,
    *,
    system_prompt: str = BRIDGE_SYSTEM_PROMPT,
) -> List[dict]:
    """Run the S4 bridge check over multihop candidates.

    Returns one verdict-record per candidate:
      {candidate, task_id, domain, verdict, reason, verifier_index}

    Verdict logic:
      - DISCARDED if the structural pre-gate flags a hard defect that means the
        item is not a usable QA at all (no bridge / fewer than two papers).
      - DOWNGRADED_SINGLE if the structural pre-gate flags a weak-but-recoverable
        bridge (generic token, single-hop phrasing, bridge in only one paper):
        the item is not a true 2-hop, so it is demoted to single-hop on the
        first answer paper rather than thrown away.
      - For structurally-clean items, ask the verifier. YES -> KEPT_MULTIHOP;
        NO / abstain -> DOWNGRADED_SINGLE (the verifier judged it not a genuine
        2-hop, but the structural checks passed, so a single-hop demotion is the
        conservative outcome).
    """
    passages_by_paper = index_passages_by_paper(passages)

    # First pass: structural pre-gate. Only structurally-clean items go to the
    # verifier (so a clearly-weak bridge never costs an API call).
    metas: List[dict] = []
    tasks = []
    for i, cand in enumerate(candidates):
        reason = structural_bridge_reason(cand, passages_by_paper)
        meta = {"candidate": cand, "index": i, "structural_reason": reason}
        if reason is None:
            task = build_bridge_task(cand, i, passages_by_paper)
            meta["task_id"] = task.task_id
            tasks.append(task)
        else:
            meta["task_id"] = "mh{0:05d}".format(i)
        metas.append(meta)

    # Single-rater consensus (design D-A: one cross-family verifier; min_agree=1).
    results = judge_entity_tasks(
        tasks, [verifier_backend], min_agree=1, system_prompt=system_prompt
    )
    results_by_id = {r.task_id: r for r in results}

    # Hard structural defects -> DISCARDED; recoverable weakness -> DOWNGRADED.
    _hard_defects = {"no_bridge_entity", "fewer_than_two_papers"}

    records: List[dict] = []
    for meta in metas:
        cand = meta["candidate"]
        s_reason = meta["structural_reason"]
        if s_reason is not None:
            if any(s_reason.startswith(d) for d in _hard_defects):
                verdict, reason = VERDICT_DISCARDED, s_reason
            else:
                verdict, reason = VERDICT_DOWNGRADED, s_reason
            records.append(
                {
                    "candidate": cand,
                    "task_id": meta["task_id"],
                    "domain": cand.get("domain", "unknown"),
                    "verdict": verdict,
                    "reason": reason,
                    "verifier_index": ABSTAIN_INDEX,
                }
            )
            continue

        r = results_by_id.get(meta["task_id"])
        verifier_index = ABSTAIN_INDEX
        if r is not None and r.votes:
            verifier_index = r.votes[0].chosen_index
        is_yes = (
            r is not None
            and r.status == STATUS_ACCEPTED
            and verifier_index == YES_INDEX
        )
        if is_yes:
            verdict, reason = VERDICT_KEPT, "verifier_yes"
        else:
            verdict, reason = VERDICT_DOWNGRADED, "verifier_not_yes"
        records.append(
            {
                "candidate": cand,
                "task_id": meta["task_id"],
                "domain": cand.get("domain", "unknown"),
                "verdict": verdict,
                "reason": reason,
                "verifier_index": verifier_index,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Output: multihop_verified.jsonl + merge back into qa_labeled.jsonl.
# ---------------------------------------------------------------------------


def _kept_multihop_row(candidate: dict) -> dict:
    """Promote a candidate to a KEPT_MULTIHOP gold row (gold_* fields)."""
    row = dict(candidate)
    paper_ids = candidate.get("answer_paper_ids", [])[:2]
    row["gold_paper_ids"] = list(paper_ids)
    bridge = candidate.get("bridge")
    row["gold_bridge_entities"] = [bridge] if bridge else []
    row["hop_count"] = 2
    row["s4_verdict"] = VERDICT_KEPT
    return row


def _downgraded_single_row(candidate: dict) -> dict:
    """Demote a multihop candidate to single-hop on its first answer paper."""
    row = dict(candidate)
    paper_ids = candidate.get("answer_paper_ids", [])
    primary = [paper_ids[0]] if paper_ids else []
    row["answer_paper_ids"] = primary
    row["gold_paper_ids"] = list(primary)
    row["gold_bridge_entities"] = []
    row["hop_count"] = 1
    row["qtype"] = "factual"  # no longer a bridge question
    # Keep only the evidence span(s) belonging to the surviving paper.
    if primary:
        row["evidence_spans"] = [
            s
            for s in candidate.get("evidence_spans", [])
            if s.get("paper_id") == primary[0]
        ]
    row["s4_verdict"] = VERDICT_DOWNGRADED
    return row


def write_multihop_verified_jsonl(
    records: Sequence[dict], out_path: Path
) -> Dict[str, int]:
    """Write one record per multihop candidate with its S4 verdict.

    KEPT_MULTIHOP rows carry gold_paper_ids / gold_bridge_entities / hop_count=2.
    DOWNGRADED_SINGLE rows are demoted to single-hop. DISCARDED rows are written
    with the verdict + reason (so the audit trail is complete) but carry NO gold
    fields. Returns a verdict count summary.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = defaultdict(int)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            verdict = rec["verdict"]
            counts[verdict] += 1
            if verdict == VERDICT_KEPT:
                row = _kept_multihop_row(rec["candidate"])
            elif verdict == VERDICT_DOWNGRADED:
                row = _downgraded_single_row(rec["candidate"])
            else:  # DISCARDED: record the verdict only, no gold promotion.
                row = dict(rec["candidate"])
                row["s4_verdict"] = VERDICT_DISCARDED
            row["s4_reason"] = rec["reason"]
            row["s4_verifier_index"] = rec["verifier_index"]
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")
    return dict(counts)


def _candidate_key(row: dict) -> Tuple[str, str]:
    """Stable identity for a QA row: (question, sorted answer_paper_ids)."""
    return (
        row.get("question", ""),
        "|".join(sorted(row.get("answer_paper_ids", []))),
    )


def merge_into_labeled(
    labeled_rows: Sequence[dict], records: Sequence[dict]
) -> List[dict]:
    """Merge S4 verdicts back into the qa_labeled rows.

    Non-multihop rows pass through untouched. Multihop rows are replaced by the
    promoted (KEPT) / demoted (DOWNGRADED) row, or dropped (DISCARDED). Matching
    is by (question, answer_paper_ids); a multihop record that has no matching
    labeled row is appended (covers the "verify candidates directly" path).
    """
    verdict_by_key: Dict[Tuple[str, str], dict] = {}
    for rec in records:
        verdict_by_key[_candidate_key(rec["candidate"])] = rec

    out: List[dict] = []
    matched_keys = set()
    for row in labeled_rows:
        key = _candidate_key(row)
        rec = verdict_by_key.get(key)
        is_mh = row.get("qtype") == "multihop_bridge" or row.get("hop_count") == 2
        if rec is None or not is_mh:
            out.append(row)
            continue
        matched_keys.add(key)
        verdict = rec["verdict"]
        if verdict == VERDICT_KEPT:
            merged = _kept_multihop_row(row)
        elif verdict == VERDICT_DOWNGRADED:
            merged = _downgraded_single_row(row)
        else:
            continue  # DISCARDED -> drop from labeled set
        merged["s4_reason"] = rec["reason"]
        out.append(merged)

    # Append any verified multihop record that did not match a labeled row.
    for rec in records:
        key = _candidate_key(rec["candidate"])
        if key in matched_keys:
            continue
        verdict = rec["verdict"]
        if verdict == VERDICT_KEPT:
            merged = _kept_multihop_row(rec["candidate"])
        elif verdict == VERDICT_DOWNGRADED:
            merged = _downgraded_single_row(rec["candidate"])
        else:
            continue
        merged["s4_reason"] = rec["reason"]
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# Optional IAA hook: bridge spot-check CSV (mirrors S3; NOT a gate).
# ---------------------------------------------------------------------------

BRIDGE_SPOTCHECK_HEADER = [
    "task_id",
    "domain",
    "question",
    "bridge",
    "verifier_index",  # 0=YES 1=NO -1=abstain
    "human_index",     # BLANK -- human fills 0 (YES) or 1 (NO)
]


def export_bridge_spotcheck_csv(records: Sequence[dict], out_path: Path) -> None:
    """Export a blind YES/NO bridge subset for a human IAA rater (optional)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BRIDGE_SPOTCHECK_HEADER)
        for rec in records:
            cand = rec["candidate"]
            writer.writerow(
                [
                    rec["task_id"],
                    rec["domain"],
                    cand.get("question", ""),
                    cand.get("bridge", ""),
                    str(rec["verifier_index"]),
                    "",
                ]
            )


def compute_bridge_human_kappa(records, human_labels):
    """Cohen's kappa (verifier vs human) for the bridge YES/NO check.

    Thin wrapper over S3's compute_openai_human_kappa so the unmeasured/degenerate
    conventions match exactly (returns (None, n) when unmeasured; never
    fabricates a value). Returns (kappa, n_pairs_used).
    """
    return compute_openai_human_kappa(records, human_labels)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _build_default_verifier():
    """Construct the Gemini Flash verifier (live use only; cross-family vs the
    Sonnet generator -- SAME panel as S3)."""
    from paper.eval.cross_judge_backends import GeminiFlashBackend

    return GeminiFlashBackend()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S4 multihop bridge verifier")
    parser.add_argument(
        "--pilot-dir",
        required=True,
        help="pilot directory holding qa_candidates.jsonl / qa_labeled.jsonl",
    )
    parser.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    parser.add_argument(
        "--source",
        choices=["candidates", "labeled"],
        default="labeled",
        help="which file to read multihop rows from (default labeled)",
    )
    parser.add_argument(
        "--spotcheck-out",
        default=None,
        help="optional bridge YES/NO spot-check CSV path (IAA hook)",
    )
    args = parser.parse_args(argv)

    pilot_dir = Path(args.pilot_dir)
    labeled_path = pilot_dir / "qa_labeled.jsonl"
    candidates_path = pilot_dir / "qa_candidates.jsonl"
    verified_path = pilot_dir / "multihop_verified.jsonl"

    passages = load_jsonl(Path(args.passages))

    source_path = labeled_path if args.source == "labeled" else candidates_path
    rows = load_jsonl(source_path)
    multihop = select_multihop_candidates(rows)

    verifier = _build_default_verifier()  # needs GEMINI_API_KEY + google pkg
    records = verify_bridges(multihop, passages, verifier)

    counts = write_multihop_verified_jsonl(records, verified_path)

    # Merge verdicts back into qa_labeled.jsonl in place (if it exists).
    if labeled_path.exists():
        labeled_rows = load_jsonl(labeled_path)
        merged = merge_into_labeled(labeled_rows, records)
        with open(labeled_path, "w", encoding="utf-8", newline="\n") as f:
            for row in merged:
                f.write(json.dumps(row, ensure_ascii=True))
                f.write("\n")

    if args.spotcheck_out:
        export_bridge_spotcheck_csv(records, Path(args.spotcheck_out))

    print(
        "S4 done: {0} multihop candidates -> {1}".format(
            len(multihop), verified_path
        )
    )
    print(
        "verdicts: kept={0} downgraded={1} discarded={2}".format(
            counts.get(VERDICT_KEPT, 0),
            counts.get(VERDICT_DOWNGRADED, 0),
            counts.get(VERDICT_DISCARDED, 0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
