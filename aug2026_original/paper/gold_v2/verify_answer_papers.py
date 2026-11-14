# -*- coding: utf-8 -*-
"""S3: answer-paper verification for the provenance-independent QA benchmark.

Inputs : S2 qa_candidates.jsonl + the S1 passage pool. For each candidate we
         build a multiple-choice "which paper holds the answer?" task whose
         options are [true answer paper, TRAP_1 .. TRAP_N, NONE]. Traps are
         retriever-agnostic distractor papers (design D-E: BM25-alone is
         forbidden -> diversify with same-domain random + a lexical-overlap
         pick so the trap set is not tied to any single retriever).
Process: a single cross-family VERIFIER backend votes for the option index it
         believes holds the answer (or NONE). The current live default is
         Gemini Flash; the backend identifier is written into every newly
         generated row instead of being hard-coded. We reuse the EntityJudge
         task + judge_entity_tasks + Cohen-kappa machinery from
         paper/eval/cross_judge.py rather than reimplementing it.

Kappa (IAA): the 2nd rater is a HUMAN spot-check subset (design D-E /
         D-A: domain-wise ~10 items). This module provides:
           (a) export_human_spotcheck_csv  -- pick a per-domain subset for a
               human to label,
           (b) load_human_labels_csv       -- import the filled CSV,
           (c) compute_openai_human_kappa  -- Cohen's kappa verifier vs rater
               (legacy function name retained for compatibility).
         If NO human labels exist, kappa is reported as null (status
         "kappa_unmeasured"); we NEVER fabricate a kappa value.

Outputs: paper/gold_v2/qa_labeled.jsonl  (ACCEPTED candidates only)
         paper/gold_v2/agreement.csv      (per-candidate verifier vote)
         paper/gold_v2/kappa.json         (Cohen kappa verifier-vs-rater or null)
         NONE / disagreement candidates are discarded.

Gate    : Cohen kappa >= 0.6 (design S3 gate, reuses D38b threshold). Below the
         gate the driver ABORTS and signals "task redesign needed" -- the gate
         is NEVER lowered to make a run pass.

ACYCLICITY (CI guard, test_no_kg_imports): this module must NOT import
prototype.baselines, proposed_kg, or qa_from_kg. Only the S1 passage pool text
and the cross-family verifier panel are inputs; the KG is never referenced.

NO real API call is made by the unit tests (a stub verifier backend is
injected). Live verification uses the credentials required by the selected
backend; the current default uses GEMINI_API_KEY.
DO NOT add hyperparameter tuning; structural rules only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PACKAGE_ROOT_FOR_IMPORT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT_FOR_IMPORT)

# Reuse the cross-judge consensus + Cohen-kappa machinery (design S3: reuse
# cross_judge.py patterns). EntityJudgeTask is a generic "pick the best
# numbered candidate or NONE" task -- exactly the shape of the answer-paper
# choice, so we reuse it verbatim rather than adding a new task type.
from paper.eval.cross_judge import (
    ABSTAIN_INDEX,
    ENTITY_SYSTEM_PROMPT,
    STATUS_ACCEPTED,
    EntityJudgeTask,
    compute_cohens_kappa,
    judge_entity_tasks,
)

# ---------------------------------------------------------------------------
# Intentionally NO import of:
#   prototype.baselines, proposed_kg, qa_from_kg
# A static CI test (test_no_kg_imports.py) scans all gold_v2/*.py for these.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The DEFAULT_* paths below reference the development-repo layout; the
# generation-stage inputs they point to are not part of the released
# archives (see README, 'Path layout'). Pass explicit --candidates/--passages
# when running outside the dev repo.
DEFAULT_CANDIDATES = PROJECT_ROOT / "paper" / "gold_v2" / "qa_candidates.jsonl"
DEFAULT_PASSAGES = PROJECT_ROOT / "paper" / "gold_v2" / "passages.jsonl"
DEFAULT_LABELED = PROJECT_ROOT / "paper" / "gold_v2" / "qa_labeled.jsonl"
DEFAULT_AGREEMENT = PROJECT_ROOT / "paper" / "gold_v2" / "agreement.csv"
DEFAULT_KAPPA = PROJECT_ROOT / "paper" / "gold_v2" / "kappa.json"
DEFAULT_SPOTCHECK = PROJECT_ROOT / "paper" / "gold_v2" / "human_spotcheck.csv"

# ---------------------------------------------------------------------------
# Structural constants (NOT tuned hyperparameters -- each justified inline).
# ---------------------------------------------------------------------------

# Number of TRAP (distractor) papers per candidate. The option list is
# [true_paper] + N_TRAP traps + [NONE], so a candidate has N_TRAP+2 choices.
# Rationale: 4 distractors give a 1-in-5 random-guess baseline (true paper vs
# 4 traps), enough for a discriminative "which paper holds the answer?" test
# without exploding verifier prompt length. This is a fixed task-structure
# constant from the design ("candidate1 + trap N"), not a tuned knob.
N_TRAP = 4

# Of the N_TRAP traps, this many come from a lexical-overlap (BM25-like) pick
# and the rest are same-domain RANDOM. Design D-E forbids BM25-ALONE; mixing a
# random majority with a single lexical trap keeps the trap set
# retriever-agnostic (a lexical-only system and a random baseline both face
# hard distractors). Structural split, not tuned.
N_TRAP_LEXICAL = 1

# Cohen-kappa acceptance gate (design S3 / D38b). NEVER lower this to pass.
KAPPA_GATE = 0.6

# Deterministic trap sampling seed (reproducibility; design section 6 records
# model + seed in the dataset card). Fixed value, not tuned.
TRAP_SEED = 0

# A short passage excerpt cap (chars) keeps the verifier prompt bounded. This
# is an I/O capacity bound (prompt length), not a quality threshold.
EXCERPT_CHARS = 1200

# The NONE option label appended as the last choice.
NONE_OPTION = "NONE: none of the above papers holds the answer"

# Verifier system prompt (ASCII-only; deterministic). Frames the answer-paper
# choice as an EntityJudgeTask so judge_entity_tasks' INDEX:<n> parser applies.
# NOTE (release audit 2026-07-19): this prompt was drafted during development
# but was NOT forwarded to the judge in the frozen v1 verification run --
# verify_candidates() called judge_entity_tasks() without a system_prompt
# kwarg, so the judge applied its default ENTITY_SYSTEM_PROMPT (the
# entity-linking framing) with the question in the Mention slot. The
# signature below now makes that default explicit; this constant is kept
# for reference only. Paper Appendix D.2 documents the applied prompt.
VERIFIER_SYSTEM_PROMPT = (
    "You are a careful annotator verifying which research paper holds the "
    "answer to a question. Read the question and the numbered candidate paper "
    "excerpts, then choose the single excerpt that actually states the answer. "
    "If none of the excerpts states the answer, choose NONE. Reply with the "
    "index of the best candidate or NONE. Use exactly the format INDEX: <n> on "
    "the final line. Do not output any other commentary."
)


# ---------------------------------------------------------------------------
# Passage pool helpers.
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[dict]:
    """Load a .jsonl file into a list of dicts (UTF-8; ASCII output guaranteed
    by the upstream S1/S2 ascii_normalize gate, so reading UTF-8 is safe)."""
    rows: List[dict] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_passages_by_paper(passages: Sequence[dict]) -> Dict[str, List[dict]]:
    """Map paper_id -> list of its passages (input order preserved)."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for p in passages:
        out[p["paper_id"]].append(p)
    return dict(out)


def representative_excerpt(
    passages_of_paper: Sequence[dict],
    *,
    preferred_passage_id: Optional[str] = None,
) -> str:
    """Return a bounded excerpt for a paper.

    Fix B-3: if `preferred_passage_id` is given and that passage exists in the
    list, use it instead of the first passage. This ensures the answer evidence
    passage (from evidence_spans[0]) is shown to the verifier, preventing
    accept-rate undercount when the answer evidence appears in a non-first
    passage. Traps and NONE still call this without preferred_passage_id so
    they use the first passage (no change in trap behaviour).
    """
    if not passages_of_paper:
        return ""
    # Prefer the passage whose passage_id matches the answer evidence.
    if preferred_passage_id:
        for p in passages_of_paper:
            if p.get("passage_id") == preferred_passage_id:
                return p.get("text", "")[:EXCERPT_CHARS]
    # Fallback: first passage in pool order.
    return passages_of_paper[0].get("text", "")[:EXCERPT_CHARS]


def _lexical_overlap(question: str, text: str) -> int:
    """Count shared lowercased word tokens (a cheap BM25-like proxy).

    Used ONLY to pick the single lexical trap; not a scoring hyperparameter.
    """
    q_tokens = set(re.findall(r"\b[a-z0-9]+\b", question.lower()))
    t_tokens = set(re.findall(r"\b[a-z0-9]+\b", text.lower()))
    return len(q_tokens & t_tokens)


# ---------------------------------------------------------------------------
# Trap selection (retriever-agnostic; design D-E).
# ---------------------------------------------------------------------------


def select_trap_papers(
    candidate: dict,
    passages_by_paper: Dict[str, List[dict]],
    domain_of_paper: Dict[str, str],
    rng: random.Random,
    *,
    n_trap: int = N_TRAP,
    n_trap_lexical: int = N_TRAP_LEXICAL,
) -> List[str]:
    """Pick trap paper_ids for a candidate (retriever-agnostic mix).

    Design D-E: BM25-alone is forbidden. We diversify:
      - n_trap_lexical traps chosen by highest lexical (BM25-like) overlap with
        the question, from the SAME domain (a lexical-only system's hardest
        distractors);
      - the remaining (n_trap - n_trap_lexical) traps chosen RANDOMLY from the
        same domain (random/dense-agnostic distractors).
    Traps never include the candidate's own answer paper(s). If the domain has
    too few papers, fewer traps are returned (the task still works).
    """
    answer_papers = set(candidate.get("answer_paper_ids", []))
    # Determine candidate domain via its first answer paper.
    cand_domain = None
    for ap in candidate.get("answer_paper_ids", []):
        if ap in domain_of_paper:
            cand_domain = domain_of_paper[ap]
            break
    same_domain_papers = [
        pid
        for pid in passages_by_paper
        if pid not in answer_papers and domain_of_paper.get(pid) == cand_domain
    ]
    if not same_domain_papers:
        return []

    question = candidate.get("question", "")
    # Lexical traps: rank same-domain papers by question overlap (desc), then
    # paper_id (asc) for determinism.
    ranked = sorted(
        same_domain_papers,
        key=lambda pid: (
            -_lexical_overlap(question, representative_excerpt(passages_by_paper[pid])),
            pid,
        ),
    )
    lexical_traps = ranked[: min(n_trap_lexical, len(ranked))]

    # Random traps from the remaining pool (deterministic via injected rng).
    remaining = [pid for pid in same_domain_papers if pid not in lexical_traps]
    rng.shuffle(remaining)
    n_random = max(0, n_trap - len(lexical_traps))
    random_traps = remaining[:n_random]

    return lexical_traps + random_traps


# ---------------------------------------------------------------------------
# Build EntityJudgeTask choices for one candidate.
# ---------------------------------------------------------------------------


def _evidence_passage_id(candidate: dict) -> Optional[str]:
    """Return the passage_id of the first evidence span, or None."""
    spans = candidate.get("evidence_spans") or []
    if spans:
        return spans[0].get("passage_id")
    return None


def _shuffle_options(
    paper_ids: List[str],
    descriptions: List[str],
    true_index_before: int,
    task_id: str,
) -> Tuple[List[str], List[str], int]:
    """Deterministically shuffle option order using a hash of the task_id.

    Fix N2: always placing the true answer at index 0 lets LLM first-option
    bias inflate accept rate. We permute using a stable hashlib digest of the
    task_id as the seed so the order is reproducible ACROSS PROCESSES (built-in
    hash() is salted by PYTHONHASHSEED and differs per run, which would break
    benchmark reproducibility). NONE is always re-appended last
    after the shuffle (its position is structural, not a fairness concern).

    Returns (shuffled_paper_ids, shuffled_descriptions, new_true_index).
    """
    # Separate NONE (last element) from the content options.
    content_ids = paper_ids[:-1]
    content_descs = descriptions[:-1]
    none_id = paper_ids[-1]     # always ""
    none_desc = descriptions[-1]  # always NONE_OPTION

    # Build index list and shuffle deterministically by task_id hash.
    indices = list(range(len(content_ids)))
    # Stable cross-process seed: md5 digest of the ASCII task_id (built-in
    # hash() is PYTHONHASHSEED-salted and non-reproducible across runs).
    seed = int(hashlib.md5(task_id.encode("ascii")).hexdigest(), 16) & 0x7FFFFFFF
    rng = random.Random(seed)
    rng.shuffle(indices)

    shuffled_ids = [content_ids[i] for i in indices]
    shuffled_descs = [content_descs[i] for i in indices]
    new_true = indices.index(true_index_before)

    # Re-append NONE at the end.
    shuffled_ids.append(none_id)
    shuffled_descs.append(none_desc)

    return shuffled_ids, shuffled_descs, new_true


def build_verification_task(
    candidate: dict,
    cand_index: int,
    passages_by_paper: Dict[str, List[dict]],
    domain_of_paper: Dict[str, str],
    trap_paper_ids: Sequence[str],
) -> Tuple[EntityJudgeTask, List[str], int]:
    """Build an EntityJudgeTask for a candidate's answer-paper choice.

    Options are [true_paper] + traps + NONE, then deterministically shuffled
    (fix N2: true answer is no longer always index 0 -- that causes LLM
    first-option bias). NONE is always last after the shuffle (structural).

    Fix B-3: the true paper's excerpt is taken from the candidate's evidence
    passage (evidence_spans[0].passage_id) rather than always the first passage,
    so the verifier sees the passage that actually contains the answer span.

    Returns (task, option_paper_ids, true_index). option_paper_ids[i] is the
    paper backing option i; the NONE option maps to "" (empty string).
    true_index tracks the shuffled position of the primary answer paper.
    """
    answer_papers = list(candidate.get("answer_paper_ids", []))
    primary = answer_papers[0] if answer_papers else ""

    # B-3: use the evidence passage for the true paper excerpt; trap excerpts
    # use the first passage of each trap paper (representative_excerpt default).
    ev_pid = _evidence_passage_id(candidate)
    true_excerpt = representative_excerpt(
        passages_by_paper.get(primary, []),
        preferred_passage_id=ev_pid,
    )

    # Build pre-shuffle list: [true_paper] + traps + [NONE].
    option_paper_ids: List[str] = [primary]
    descriptions: List[str] = [true_excerpt]
    pre_shuffle_true = 0

    for tp in trap_paper_ids:
        option_paper_ids.append(tp)
        descriptions.append(representative_excerpt(passages_by_paper.get(tp, [])))

    # Append NONE last (will stay last through _shuffle_options).
    option_paper_ids.append("")
    descriptions.append(NONE_OPTION)

    task_id = "qa{0:05d}".format(cand_index)

    # N2: shuffle content options (everything except NONE) deterministically.
    option_paper_ids, descriptions, true_index = _shuffle_options(
        option_paper_ids, descriptions, pre_shuffle_true, task_id
    )

    domain = domain_of_paper.get(primary, "unknown")
    task = EntityJudgeTask(
        task_id=task_id,
        domain=domain,
        paper_excerpt="(see candidate excerpts below)",
        entity_mention=candidate.get("question", ""),
        candidate_node_ids=tuple(
            (pid if pid else "NONE") for pid in option_paper_ids
        ),
        candidate_node_descriptions=tuple(descriptions),
    )
    return task, option_paper_ids, true_index


# ---------------------------------------------------------------------------
# Verification driver (single OpenAI verifier; backend injectable).
# ---------------------------------------------------------------------------


def verify_candidates(
    candidates: Sequence[dict],
    passages: Sequence[dict],
    verifier_backend,
    *,
    n_trap: int = N_TRAP,
    n_trap_lexical: int = N_TRAP_LEXICAL,
    trap_seed: int = TRAP_SEED,
    system_prompt: str = ENTITY_SYSTEM_PROMPT,
) -> List[dict]:
    """Run the single-verifier answer-paper check over all candidates.

    Returns one verification-record dict per candidate:
      {candidate, task_id, domain, option_paper_ids, true_index,
       verifier_index, accepted}
    `accepted` is True iff the verifier picked the TRUE answer option (not a
    trap and not NONE). NONE / trap / abstain picks are rejected.

    The verifier panel is a SINGLE backend (design D-A; the frozen v1 run
    used Gemini Flash). We reuse
    judge_entity_tasks with min_agree=1 (single rater); the human spot-check is
    the 2nd kappa rater handled separately, NOT a second live model.
    """
    rng = random.Random(trap_seed)
    passages_by_paper = index_passages_by_paper(passages)
    domain_of_paper = {
        p["paper_id"]: p["domain"] for p in passages
    }

    tasks = []
    metas: List[dict] = []
    for i, cand in enumerate(candidates):
        traps = select_trap_papers(
            cand,
            passages_by_paper,
            domain_of_paper,
            rng,
            n_trap=n_trap,
            n_trap_lexical=n_trap_lexical,
        )
        task, option_paper_ids, true_index = build_verification_task(
            cand, i, passages_by_paper, domain_of_paper, traps
        )
        tasks.append(task)
        metas.append(
            {
                "candidate": cand,
                "task_id": task.task_id,
                "domain": task.domain,
                "option_paper_ids": option_paper_ids,
                "true_index": true_index,
            }
        )

    # Single-rater consensus: min_agree=1 so one vote suffices to "accept" the
    # vote; we still compare it to the TRUE index ourselves below.
    results = judge_entity_tasks(
        tasks, [verifier_backend], min_agree=1, system_prompt=system_prompt
    )
    results_by_id = {r.task_id: r for r in results}

    records: List[dict] = []
    for meta in metas:
        r = results_by_id.get(meta["task_id"])
        verifier_index = ABSTAIN_INDEX
        if r is not None and r.votes:
            verifier_index = r.votes[0].chosen_index
        accepted = (
            r is not None
            and r.status == STATUS_ACCEPTED
            and verifier_index == meta["true_index"]
        )
        records.append(
            {
                "candidate": meta["candidate"],
                "task_id": meta["task_id"],
                "domain": meta["domain"],
                "option_paper_ids": meta["option_paper_ids"],
                "true_index": meta["true_index"],
                "verifier_index": verifier_index,
                "accepted": bool(accepted),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Human spot-check: CSV export / import + Cohen kappa (OpenAI vs human).
# ---------------------------------------------------------------------------

SPOTCHECK_CSV_HEADER = [
    "task_id",
    "domain",
    "question",
    "options",          # human reference: paper_ids joined by '|' (NONE last)
    "openai_index",     # verifier vote (filled by us)
    "human_index",      # BLANK -- the human fills this (option index or NONE)
]


def export_human_spotcheck_csv(
    records: Sequence[dict],
    out_path: Path,
    *,
    per_domain: int = 10,
    seed: int = TRAP_SEED,
) -> List[dict]:
    """Export a per-domain subset for a human to label (kappa 2nd rater).

    Picks up to `per_domain` records per domain (deterministic shuffle via
    `seed`). The CSV has a blank human_index column the human fills with the
    chosen option index, or the literal NONE. Returns the exported subset.

    `per_domain` default 10 follows design D-A ("domain-wise ~10"); it is a
    sampling-size constant for the human IAA subset, not a model knob.
    """
    rng = random.Random(seed)
    by_domain: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_domain[rec["domain"]].append(rec)

    subset: List[dict] = []
    for dom in sorted(by_domain.keys()):
        pool = list(by_domain[dom])
        rng.shuffle(pool)
        subset.extend(pool[:per_domain])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SPOTCHECK_CSV_HEADER)
        for rec in subset:
            opts = "|".join(
                (pid if pid else "NONE") for pid in rec["option_paper_ids"]
            )
            writer.writerow(
                [
                    rec["task_id"],
                    rec["domain"],
                    rec["candidate"].get("question", ""),
                    opts,
                    str(rec["verifier_index"]),
                    "",  # human_index -- left blank for the human to fill
                ]
            )
    return subset


def _parse_human_cell(raw: str, none_index: int) -> int:
    """Parse a human-filled cell into an option index (NONE -> none_index)."""
    s = (raw or "").strip()
    if not s:
        return ABSTAIN_INDEX  # unfilled -> treated as missing
    if re.search(r"\bNONE\b", s, re.IGNORECASE):
        return none_index
    try:
        return int(s)
    except ValueError:
        return ABSTAIN_INDEX


def load_human_labels_csv(path: Path) -> Dict[str, int]:
    """Load a filled spot-check CSV -> {task_id: human_index}.

    Rows with a blank human_index are skipped (no human label yet). The NONE
    literal maps to the last option index (derived from the 'options' column
    length so it matches what the verifier saw).
    """
    path = Path(path)
    out: Dict[str, int] = {}
    if not path.exists():
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_raw = (row.get("human_index") or "").strip()
            if not human_raw:
                continue
            opts = (row.get("options") or "").split("|")
            none_index = max(0, len(opts) - 1)  # NONE is the last option
            idx = _parse_human_cell(human_raw, none_index)
            if idx != ABSTAIN_INDEX:
                out[row["task_id"]] = idx
    return out


def compute_openai_human_kappa(
    records: Sequence[dict], human_labels: Dict[str, int]
) -> Tuple[Optional[float], int]:
    """Cohen's kappa between the OpenAI verifier and human labels.

    Returns (kappa, n_pairs_used).

    kappa is None when:
      - No human labels exist (never fabricate a value), OR
      - All surviving pairs fall in a single category (degenerate: P_e = 1
        makes Cohen's kappa undefined; returning None prevents a spurious
        1.0 from the cross_judge formula from reaching the gate).
        NOTE: cross_judge.compute_cohens_kappa is NOT modified (regression
        risk); we detect degeneracy HERE before calling it.

    n_pairs_used is the number of (verifier, human) pairs that actually
    entered the kappa computation (abstain / unmatched pairs are excluded).
    This is reported separately from n_human (total human labels loaded)
    so kappa.json shows the real denominator -- fix M-2.
    """
    if not human_labels:
        return None, 0
    rec_by_id = {rec["task_id"]: rec for rec in records}
    # Category space = union of non-abstain indices seen; re-encode to dense
    # 0..k-1 so compute_cohens_kappa's range check passes.
    pairs_raw: List[Tuple[int, int]] = []
    for task_id, human_idx in human_labels.items():
        rec = rec_by_id.get(task_id)
        if rec is None:
            continue
        pairs_raw.append((rec["verifier_index"], human_idx))
    if not pairs_raw:
        return None, 0

    # Filter out abstain on either side (consistent with cross_judge convention).
    valid_pairs = [
        (a, b) for a, b in pairs_raw
        if a != ABSTAIN_INDEX and b != ABSTAIN_INDEX
    ]
    n_pairs_used = len(valid_pairs)
    if n_pairs_used == 0:
        return None, 0

    seen = sorted({v for a, b in valid_pairs for v in (a, b)})
    if not seen:
        return None, 0

    # B-1 degeneracy guard: if ALL surviving pairs use the same single category,
    # Cohen's kappa is undefined (P_e = 1 -> division by zero; the cross_judge
    # implementation returns 1.0 in this branch, which is mathematically
    # incorrect -- there is no inter-rater information when only one label
    # appears). We return None (unmeasured) instead of calling compute_cohens_kappa.
    # cross_judge.py is NOT modified to avoid regression in auto_label_pipeline.
    if len(seen) == 1:
        return None, n_pairs_used

    remap = {c: i for i, c in enumerate(seen)}

    def _enc(v: int) -> int:
        return remap.get(v, ABSTAIN_INDEX)

    per_task = [(_enc(a), _enc(b)) for a, b in valid_pairs]
    kappa = compute_cohens_kappa(per_task, len(seen))
    return kappa, n_pairs_used


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------

AGREEMENT_CSV_HEADER = [
    "task_id",
    "domain",
    "true_index",
    "verifier_index",
    "accepted",
]


def write_agreement_csv(records: Sequence[dict], out_path: Path) -> None:
    """Write one row per verification record (verifier vote vs true index)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(AGREEMENT_CSV_HEADER)
        for rec in records:
            writer.writerow(
                [
                    rec["task_id"],
                    rec["domain"],
                    str(rec["true_index"]),
                    str(rec["verifier_index"]),
                    "1" if rec["accepted"] else "0",
                ]
            )


def write_labeled_jsonl(
    records: Sequence[dict], out_path: Path, *, verifier_id: str
) -> int:
    """Write ACCEPTED candidates only (NONE / disagreement discarded).

    Each output row is the candidate dict merged with the S3 verification
    fields so downstream consumers have full context in one record:
      verifier, s3_status  -- provenance
      s3_true_index        -- which option index was the ground-truth paper
      s3_verifier_index    -- which option index the verifier chose
      s3_accepted          -- True (always, since we only write accepted rows)
      domain               -- from rec["domain"] (C2 fix: was missing because
                              rec["candidate"] never carried domain; now pulled
                              from the verification record which has it from
                              the passage pool via domain_of_paper in
                              verify_candidates).

    Returns the number of accepted candidates written.
    """
    if not isinstance(verifier_id, str) or not verifier_id.strip():
        raise ValueError("verifier_id must be a non-empty string")
    verifier_id = verifier_id.strip()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            if not rec["accepted"]:
                continue
            row = dict(rec["candidate"])
            # C2: merge verification fields that were missing from candidate.
            row["verifier"] = verifier_id
            row["s3_status"] = STATUS_ACCEPTED
            row["s3_true_index"] = rec["true_index"]
            row["s3_verifier_index"] = rec["verifier_index"]
            row["s3_accepted"] = True
            # domain: prefer candidate's own domain field (set by S2 C1 fix);
            # fall back to the verification-record domain (from passage pool)
            # so rows from both old and new S2 output are handled correctly.
            if "domain" not in row or row["domain"] is None:
                row["domain"] = rec.get("domain")
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")
            n += 1
    return n


def write_kappa_json(
    kappa: Optional[float],
    n_human: int,
    out_path: Path,
    *,
    n_pairs_used: int = 0,
    gate: float = KAPPA_GATE,
) -> dict:
    """Write the kappa report. kappa=None -> status 'kappa_unmeasured'.

    Fields:
      cohens_kappa   : float or null (null = unmeasured / degenerate).
      status         : "measured" | "kappa_unmeasured".
      n_human_labels : total number of human-filled labels loaded (raw count).
      n_pairs_used   : pairs that entered kappa computation after dropping
                       abstains and unmatched rows (fix M-2: real denominator).
      gate           : KAPPA_GATE threshold.
      gate_passed    : True only when status=="measured" AND kappa >= gate.

    kappa=None covers two cases:
      1. No human labels at all.
      2. Degenerate single-category agreement (B-1: P_e=1, kappa undefined).
    In both cases gate_passed=False (unmeasured cannot pass the gate).

    json.dump uses ensure_ascii=True; kappa is a float or Python None (which
    serialises to JSON null -- never NaN, which is not valid JSON).
    """
    if kappa is None:
        payload = {
            "cohens_kappa": None,
            "status": "kappa_unmeasured",
            "n_human_labels": n_human,
            "n_pairs_used": n_pairs_used,
            "gate": gate,
            "gate_passed": False,
            "note": (
                "kappa is unmeasured: either no human labels were provided, "
                "or all surviving pairs fell in a single category (degenerate). "
                "Provide human labels covering at least 2 outcome categories to "
                "measure OpenAI-vs-human IAA. A fabricated kappa is NEVER written."
            ),
        }
    else:
        payload = {
            "cohens_kappa": kappa,
            "status": "measured",
            "n_human_labels": n_human,
            "n_pairs_used": n_pairs_used,
            "gate": gate,
            "gate_passed": bool(kappa >= gate),
        }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return payload


# ---------------------------------------------------------------------------
# CLI (live verification; OpenAI backend constructed lazily).
# ---------------------------------------------------------------------------


def _build_default_verifier():
    """Construct the Gemini verifier backend (live use only).

    Verification uses Gemini Flash (GEMINI_API_KEY), cross-family relative
    to the generator model. OpenAIBackend remains a documented alternative
    backend.
    """
    from paper.eval.cross_judge_backends import GeminiFlashBackend

    return GeminiFlashBackend()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S3 answer-paper verifier")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    parser.add_argument("--labeled-out", default=str(DEFAULT_LABELED))
    parser.add_argument("--agreement-out", default=str(DEFAULT_AGREEMENT))
    parser.add_argument("--kappa-out", default=str(DEFAULT_KAPPA))
    parser.add_argument("--spotcheck-out", default=str(DEFAULT_SPOTCHECK))
    parser.add_argument(
        "--human-labels",
        default=None,
        help="filled spot-check CSV path (omit to leave kappa unmeasured)",
    )
    parser.add_argument(
        "--spotcheck-per-domain",
        type=int,
        default=10,
        help="human spot-check subset size per domain (design D-A ~10)",
    )
    args = parser.parse_args(argv)

    candidates = load_jsonl(Path(args.candidates))
    passages = load_jsonl(Path(args.passages))
    verifier = _build_default_verifier()  # needs GEMINI_API_KEY + provider pkg

    records = verify_candidates(candidates, passages, verifier)
    write_agreement_csv(records, Path(args.agreement_out))
    export_human_spotcheck_csv(
        records, Path(args.spotcheck_out), per_domain=args.spotcheck_per_domain
    )

    human_labels = (
        load_human_labels_csv(Path(args.human_labels))
        if args.human_labels
        else {}
    )
    kappa, n_pairs_used = compute_openai_human_kappa(records, human_labels)
    payload = write_kappa_json(
        kappa,
        len(human_labels),
        Path(args.kappa_out),
        n_pairs_used=n_pairs_used,
    )

    verifier_id = getattr(verifier, "name", verifier.__class__.__name__)
    n_accepted = write_labeled_jsonl(
        records, Path(args.labeled_out), verifier_id=verifier_id
    )

    print(
        "S3 done: {0}/{1} accepted -> {2}".format(
            n_accepted, len(records), args.labeled_out
        )
    )
    print(
        "spot-check CSV -> {0} ; agreement -> {1} ; kappa -> {2} (status={3})".format(
            args.spotcheck_out, args.agreement_out, args.kappa_out, payload["status"]
        )
    )
    # Gate 1: measured kappa below threshold -> task redesign needed.
    if payload["status"] == "measured" and not payload["gate_passed"]:
        print(
            "ABORT (exit 2): Cohen kappa {0:.3f} < gate {1} -- "
            "TASK REDESIGN NEEDED (do NOT lower the gate).".format(
                payload["cohens_kappa"], KAPPA_GATE
            )
        )
        return 2
    # Gate 2 (B-2): kappa unmeasured is NOT a pass.
    if payload["status"] == "kappa_unmeasured":
        print(
            "WARNING (exit 3): kappa unmeasured -- no human labels provided "
            "or degenerate single-category agreement. "
            "Fill human_spotcheck.csv and re-run with --human-labels. "
            "kappa_unmeasured is NOT a pass."
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
