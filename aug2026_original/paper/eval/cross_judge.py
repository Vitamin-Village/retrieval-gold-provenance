"""4-model cross-judge consensus + Fleiss kappa / Krippendorff alpha reporting.

Inputs : EntityJudgeTask / RelationJudgeTask / PathJudgeTask lists + JudgeBackend panel.
Outputs: ConsensusResult lists, KappaReport per (domain, task_type), CSV + JSON files.
Decisions: D32 (4-model cross-family Fleiss kappa), D38 (W2 auto-label), D38b (kappa >= 0.6 gate).

DO NOT EDIT without reading paper/MASTER_PLAN.md D33-D41.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (ASCII only).
# ---------------------------------------------------------------------------

ABSTAIN_INDEX = -1
# Default consensus floor for the historical 4-model panel (D32). The
# 2-model panel (D42, 2026-05-23) overrides this via the `min_agree`
# parameter on judge_*_tasks (auto_label_pipeline forwards
# cfg.consensus_min through). Keep the module-level constant at 3 so the
# legacy 4-model unit tests in tests/test_cross_judge.py still observe
# 3-of-4 ACCEPT / 2-of-4 DROPPED semantics without modification.
MIN_AGREE_FOR_CONSENSUS = 3  # >= 3 non-abstain votes that agree -> ACCEPTED (4-model default)
N_PANEL = 4  # 4-model legacy panel default. The D42 2-model live driver
            # passes 2 backends + min_agree=2 explicitly via cfg.consensus_min.
DEFAULT_MAX_TOKENS = 256  # raised from 64: newer models (claude-haiku-4-5, gemini-2.5-flash) need
                           # more tokens to finish chain-of-thought before emitting INDEX: <n>.

# Primary: last non-empty line matches "INDEX: <integer>".
_INDEX_LINE_RE = re.compile(r"^\s*INDEX\s*:\s*(-?\d+)\s*$", re.IGNORECASE)
# Fallback: last non-empty line is a bare integer (e.g. mistral-7b omits the prefix).
_BARE_INT_RE = re.compile(r"^\s*(-?\d+)\s*$")

# Status strings.
STATUS_ACCEPTED = "ACCEPTED"
STATUS_DROPPED = "DROPPED_LOW_AGREEMENT"

# Task type tags used in CSV / kappa reports.
TASK_TYPE_ENTITY = "entity"
TASK_TYPE_RELATION = "relation"
TASK_TYPE_PATH = "path"

# Allowed relation types (11) - documented in paper/MASTER_PLAN.md (D33-D36).
# Module is domain-agnostic; this list is exposed only for callers that need
# to validate candidate sets before constructing a RelationJudgeTask.
ALLOWED_RELATIONS = (
    "USES",
    "ENABLES",
    "PART_OF",
    "DEPENDS_ON",
    "CAUSES",
    "REDUCES",
    "INCREASES",
    "MEASURED_BY",
    "COMPARED_TO",
    "INCLUDES",
    "NONE",
)


# ---------------------------------------------------------------------------
# Task spec dataclasses (frozen).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityJudgeTask:
    task_id: str
    domain: str  # "semiconductor"|"construction"|"steel"|"textile" (carried as tag).
    paper_excerpt: str
    entity_mention: str
    candidate_node_ids: Tuple[str, ...]
    candidate_node_descriptions: Tuple[str, ...]


@dataclass(frozen=True)
class RelationJudgeTask:
    task_id: str
    domain: str
    paper_excerpt: str
    head_node: Tuple[str, str]   # (node_id, name)
    tail_node: Tuple[str, str]
    candidate_relations: Tuple[str, ...]  # subset of ALLOWED_RELATIONS or ("NONE",)


@dataclass(frozen=True)
class PathJudgeTask:
    task_id: str
    domain: str
    paper_excerpt: str
    question: str
    start_node: Tuple[str, str]
    end_node: Tuple[str, str]
    candidate_paths: Tuple[Tuple[str, ...], ...]  # each path: <=4 node ids.


# ---------------------------------------------------------------------------
# Result dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeVote:
    backend_name: str
    chosen_index: int  # -1 = ABSTAIN
    raw_response: str


@dataclass(frozen=True)
class ConsensusResult:
    task_id: str
    task_type: str  # entity / relation / path
    domain: str
    votes: Tuple[JudgeVote, ...]  # 4 votes
    consensus_index: Optional[int]  # None if no >=3 agreement among non-abstain
    n_agree: int
    status: str  # STATUS_ACCEPTED | STATUS_DROPPED


@dataclass(frozen=True)
class KappaReport:
    domain: str
    task_type: str
    n_tasks: int
    fleiss_kappa: float
    krippendorff_alpha: float


# ---------------------------------------------------------------------------
# Prompt templates (ASCII only; deterministic).
# ---------------------------------------------------------------------------


ENTITY_SYSTEM_PROMPT = (
    "You are a careful annotator linking a surface mention in a research excerpt "
    "to a knowledge-graph node. Read the excerpt and the mention, then choose the "
    "single best matching node from the numbered candidates. If none fit, choose "
    "NONE. Reply with the index of the best candidate or NONE. Use exactly the "
    "format INDEX: <n> on the final line. Do not output any other commentary."
)


RELATION_SYSTEM_PROMPT = (
    "You are a careful annotator classifying the relation between two entities in "
    "a research excerpt. Read the excerpt and the head / tail entity, then choose "
    "the single relation type from the numbered candidates that best fits. If no "
    "candidate fits, choose NONE. Reply with the index of the best candidate or "
    "NONE. Use exactly the format INDEX: <n> on the final line. Do not output any "
    "other commentary."
)


PATH_SYSTEM_PROMPT = (
    "You are a careful annotator selecting the reasoning path that best answers a "
    "question over a knowledge graph. Read the excerpt, the question, and the "
    "numbered candidate paths (sequences of node ids from start to end), then "
    "choose the single best path. If no candidate fits, choose NONE. Reply with "
    "the index of the best candidate or NONE. Use exactly the format INDEX: <n> "
    "on the final line. Do not output any other commentary."
)


def build_entity_user_prompt(task: EntityJudgeTask) -> str:
    """Deterministic ASCII-safe user prompt for an EntityJudgeTask."""
    lines = []
    lines.append("Domain: {0}".format(task.domain))
    lines.append("Paper excerpt:")
    lines.append(task.paper_excerpt.strip())
    lines.append("")
    lines.append("Mention: {0}".format(task.entity_mention))
    lines.append("")
    lines.append("Candidates:")
    for i, (nid, desc) in enumerate(
        zip(task.candidate_node_ids, task.candidate_node_descriptions)
    ):
        lines.append("{0}: {1} - {2}".format(i, nid, desc))
    lines.append("Output the chosen index (or NONE for no match).")
    lines.append("INDEX:")
    return "\n".join(lines)


def build_relation_user_prompt(task: RelationJudgeTask) -> str:
    head_id, head_name = task.head_node
    tail_id, tail_name = task.tail_node
    lines = []
    lines.append("Domain: {0}".format(task.domain))
    lines.append("Paper excerpt:")
    lines.append(task.paper_excerpt.strip())
    lines.append("")
    lines.append("Head: {0} ({1})".format(head_name, head_id))
    lines.append("Tail: {0} ({1})".format(tail_name, tail_id))
    lines.append("")
    lines.append("Candidate relations:")
    for i, rel in enumerate(task.candidate_relations):
        lines.append("{0}: {1}".format(i, rel))
    lines.append("Output the chosen index (or NONE for no match).")
    lines.append("INDEX:")
    return "\n".join(lines)


def build_path_user_prompt(task: PathJudgeTask) -> str:
    start_id, start_name = task.start_node
    end_id, end_name = task.end_node
    lines = []
    lines.append("Domain: {0}".format(task.domain))
    lines.append("Paper excerpt:")
    lines.append(task.paper_excerpt.strip())
    lines.append("")
    lines.append("Question: {0}".format(task.question))
    lines.append("Start node: {0} ({1})".format(start_name, start_id))
    lines.append("End node: {0} ({1})".format(end_name, end_id))
    lines.append("")
    lines.append("Candidate paths (sequence of node ids):")
    for i, path in enumerate(task.candidate_paths):
        lines.append("{0}: {1}".format(i, " -> ".join(path)))
    lines.append("Output the chosen index (or NONE for no match).")
    lines.append("INDEX:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def parse_index_or_abstain(raw: str) -> int:
    """Parse the final 'INDEX: <n>' line (or a bare integer fallback).

    Returns the integer index, or ABSTAIN_INDEX (-1) if the response is empty,
    is "NONE", or fails to match on its final non-empty line.

    Fallback: if the strict INDEX: <n> format is absent but the last line is
    a bare integer (e.g. mistral-7b emits just the digit), that integer is used.
    """
    if raw is None:
        return ABSTAIN_INDEX
    text = raw.strip()
    if not text:
        return ABSTAIN_INDEX
    # Take the last non-empty line.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ABSTAIN_INDEX
    last = lines[-1].strip()
    # Explicit NONE -> abstain.
    if re.search(r"\bNONE\b", last, re.IGNORECASE):
        return ABSTAIN_INDEX
    m = _INDEX_LINE_RE.match(last)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return ABSTAIN_INDEX
    # Fallback: bare integer on the last line (e.g. mistral-7b).
    m2 = _BARE_INT_RE.match(last)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            return ABSTAIN_INDEX
    return ABSTAIN_INDEX


# ---------------------------------------------------------------------------
# Consensus rule.
# ---------------------------------------------------------------------------


def _consensus_from_votes(
    votes: Tuple[JudgeVote, ...],
    *,
    min_agree: int = MIN_AGREE_FOR_CONSENSUS,
) -> Tuple[Optional[int], int, str]:
    """Return (consensus_index, n_agree, status).

    Rule: take the mode of the non-abstain votes. If that mode has
    >=`min_agree` votes, ACCEPTED with consensus_index = mode. Else
    DROPPED_LOW_AGREEMENT with consensus_index = None.

    `min_agree` defaults to the module constant MIN_AGREE_FOR_CONSENSUS
    (=3, 4-model legacy). The D42 2-model driver passes min_agree=2 so
    unanimous-of-2 maps to ACCEPTED.
    """
    non_abstain = [v.chosen_index for v in votes if v.chosen_index != ABSTAIN_INDEX]
    if not non_abstain:
        return None, 0, STATUS_DROPPED
    # Find mode (smallest index on tie for determinism).
    counts: dict = {}
    for idx in non_abstain:
        counts[idx] = counts.get(idx, 0) + 1
    best_idx = None
    best_count = 0
    for idx in sorted(counts.keys()):
        c = counts[idx]
        if c > best_count:
            best_count = c
            best_idx = idx
    if best_count >= min_agree:
        return best_idx, best_count, STATUS_ACCEPTED
    return None, best_count, STATUS_DROPPED


# ---------------------------------------------------------------------------
# Driver: vote -> consensus per task type.
# ---------------------------------------------------------------------------


def _judge_one(
    backends: List,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> Tuple[JudgeVote, ...]:
    votes: List[JudgeVote] = []
    for be in backends:
        try:
            raw = be.chat(system_prompt, user_prompt, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 - record as abstain, do not crash run
            raw = "BACKEND_ERROR: {0}".format(type(e).__name__)
            votes.append(JudgeVote(backend_name=be.name, chosen_index=ABSTAIN_INDEX, raw_response=raw))
            continue
        idx = parse_index_or_abstain(raw)
        votes.append(JudgeVote(backend_name=be.name, chosen_index=idx, raw_response=raw))
    return tuple(votes)


def judge_entity_tasks(
    tasks: List[EntityJudgeTask],
    backends: List,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_agree: int = MIN_AGREE_FOR_CONSENSUS,
    system_prompt: str = ENTITY_SYSTEM_PROMPT,
) -> List[ConsensusResult]:
    """Run cross-judge consensus on a list of entity-linking tasks.

    `min_agree` controls the consensus threshold (>= this many non-abstain
    votes must agree on the same index for STATUS_ACCEPTED). Defaults to
    the 4-model panel's 3; the D42 2-model driver passes 2.

    `system_prompt` defaults to the generic entity-linking instructions;
    callers that reuse the EntityJudgeTask YES/NO shape for a different
    judgment (e.g. the S4 bridge check) pass their own framing here.
    """
    results: List[ConsensusResult] = []
    for task in tasks:
        user_prompt = build_entity_user_prompt(task)
        votes = _judge_one(backends, system_prompt, user_prompt, max_tokens)
        cidx, n_agree, status = _consensus_from_votes(votes, min_agree=min_agree)
        results.append(
            ConsensusResult(
                task_id=task.task_id,
                task_type=TASK_TYPE_ENTITY,
                domain=task.domain,
                votes=votes,
                consensus_index=cidx,
                n_agree=n_agree,
                status=status,
            )
        )
    return results


def judge_relation_tasks(
    tasks: List[RelationJudgeTask],
    backends: List,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_agree: int = MIN_AGREE_FOR_CONSENSUS,
) -> List[ConsensusResult]:
    """Run cross-judge consensus on a list of relation-classification tasks.

    See judge_entity_tasks for the `min_agree` contract.
    """
    results: List[ConsensusResult] = []
    for task in tasks:
        user_prompt = build_relation_user_prompt(task)
        votes = _judge_one(backends, RELATION_SYSTEM_PROMPT, user_prompt, max_tokens)
        cidx, n_agree, status = _consensus_from_votes(votes, min_agree=min_agree)
        results.append(
            ConsensusResult(
                task_id=task.task_id,
                task_type=TASK_TYPE_RELATION,
                domain=task.domain,
                votes=votes,
                consensus_index=cidx,
                n_agree=n_agree,
                status=status,
            )
        )
    return results


def judge_path_tasks(
    tasks: List[PathJudgeTask],
    backends: List,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_agree: int = MIN_AGREE_FOR_CONSENSUS,
) -> List[ConsensusResult]:
    """Run cross-judge consensus on a list of reasoning-path-selection tasks.

    See judge_entity_tasks for the `min_agree` contract.
    """
    results: List[ConsensusResult] = []
    for task in tasks:
        user_prompt = build_path_user_prompt(task)
        votes = _judge_one(backends, PATH_SYSTEM_PROMPT, user_prompt, max_tokens)
        cidx, n_agree, status = _consensus_from_votes(votes, min_agree=min_agree)
        results.append(
            ConsensusResult(
                task_id=task.task_id,
                task_type=TASK_TYPE_PATH,
                domain=task.domain,
                votes=votes,
                consensus_index=cidx,
                n_agree=n_agree,
                status=status,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Fleiss kappa (Fleiss 1971, nominal scale).
# ---------------------------------------------------------------------------


def compute_fleiss_kappa(
    per_task_votes: List[Tuple[int, ...]], n_categories: int
) -> float:
    """Compute Fleiss kappa for nominal ratings.

    Args:
        per_task_votes: N tasks; each entry is a tuple of category indices
            (one per rater). Negative indices (ABSTAIN) are counted as their
            own implicit category-free vote, so we drop them BEFORE the matrix.
            Callers that want to count abstains as agreement should pre-map
            -1 to a real category. The runtime path here filters them out.
        n_categories: number of nominal categories (must be >= 2). When all
            non-abstain votes fall in a single category and every task has
            unanimous ratings, kappa is defined as 1.0 (no variability).

    Returns:
        Fleiss kappa as float. Returns 1.0 if all tasks are unanimous and
        only one category appears across the whole matrix (degenerate).
    """
    if n_categories < 1:
        raise ValueError("n_categories must be >= 1")

    # Build the N x k count matrix using only non-abstain votes.
    cleaned: List[List[int]] = []
    for votes in per_task_votes:
        votes_no_abs = [v for v in votes if v != ABSTAIN_INDEX]
        if not votes_no_abs:
            continue  # all raters abstained on this task -> drop
        cleaned.append(votes_no_abs)
    if not cleaned:
        return 0.0

    # All tasks must have the same number of raters for the classic formula.
    # If raters differ across tasks (some abstained), use the per-row n with
    # the generalization in Fleiss 1971 eq. (1)-(4). We implement that.
    N = len(cleaned)
    k = n_categories

    # Build N x k integer matrix.
    M = np.zeros((N, k), dtype=np.float64)
    for i, row in enumerate(cleaned):
        for v in row:
            if not (0 <= v < k):
                # Out-of-range vote: silently skip (callers should validate).
                continue
            M[i, v] += 1.0

    n_per_row = M.sum(axis=1)  # raters per task (after abstain removal)
    if np.any(n_per_row < 2):
        # Need at least 2 raters per task to define P_i. If any row has <2,
        # we drop those rows.
        mask = n_per_row >= 2
        M = M[mask]
        n_per_row = n_per_row[mask]
        N = M.shape[0]
        if N == 0:
            return 0.0

    # p_j = column proportion across all rater-judgments.
    total = float(M.sum())
    if total <= 0.0:
        return 0.0
    p_j = M.sum(axis=0) / total  # shape (k,)

    # Per-task agreement (Fleiss 1971):
    #   P_i = (sum_j n_ij^2 - n_i) / (n_i * (n_i - 1))
    sq = (M * M).sum(axis=1)
    P_i = (sq - n_per_row) / (n_per_row * (n_per_row - 1.0))

    P_bar = float(P_i.mean())
    P_e = float((p_j * p_j).sum())

    # Degenerate cases:
    if P_e >= 1.0:
        # Only one category populated: define kappa = 1 if also P_bar == 1.
        return 1.0 if P_bar >= 1.0 - 1e-12 else 0.0
    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(kappa)


# ---------------------------------------------------------------------------
# Cohen's kappa (2-rater nominal). D42 amendment 2026-05-23.
# ---------------------------------------------------------------------------


def compute_cohens_kappa(
    per_task_votes: List[Tuple[int, ...]], n_categories: int
) -> float:
    """Compute Cohen's kappa for exactly 2 raters on nominal data.

    Fleiss kappa is undefined or degenerate at N=2 raters (n_i*(n_i-1)=2 so
    P_i collapses to 1/2 * sum n_ij^2 - 1 -> exactly 1 when unanimous and 0
    otherwise, losing graded agreement information). Cohen's kappa is the
    standard 2-rater nominal-scale IAA metric (Cohen 1960) and is what the
    D42 2-model panel uses.

    Abstain votes (-1) on either rater drop the task from the comparison
    (treated as missing, consistent with Fleiss / Krippendorff conventions
    in this module).

    Args:
        per_task_votes: N tasks; each entry is a tuple of length 2 (one
            category per rater). Tuples shorter than 2 are skipped.
        n_categories: number of nominal categories (>=1). Returns 0.0 on
            empty matrix; returns 1.0 if all surviving pairs are unanimous.
    """
    if n_categories < 1:
        raise ValueError("n_categories must be >= 1")

    # Build only the 2-rater rows with both raters non-abstain.
    pairs: List[Tuple[int, int]] = []
    for votes in per_task_votes:
        if len(votes) < 2:
            continue
        a, b = votes[0], votes[1]
        if a == ABSTAIN_INDEX or b == ABSTAIN_INDEX:
            continue
        if not (0 <= a < n_categories and 0 <= b < n_categories):
            continue
        pairs.append((a, b))

    n = len(pairs)
    if n == 0:
        return 0.0

    # Observed agreement.
    n_agree = sum(1 for a, b in pairs if a == b)
    p_o = n_agree / n

    # Expected agreement under independence per category.
    counts_a = np.zeros(n_categories, dtype=np.float64)
    counts_b = np.zeros(n_categories, dtype=np.float64)
    for a, b in pairs:
        counts_a[a] += 1.0
        counts_b[b] += 1.0
    p_e = float(np.dot(counts_a, counts_b)) / (n * n)

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 - 1e-12 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


# ---------------------------------------------------------------------------
# Krippendorff alpha (nominal level).
# ---------------------------------------------------------------------------


def compute_krippendorff_alpha(
    per_task_votes: List[Tuple[int, ...]], n_categories: int
) -> float:
    """Nominal-level Krippendorff alpha.

    Implements Krippendorff (2004) eq. for nominal data using coincidence
    matrices. Abstain votes (-1) are treated as missing observations and
    excluded from the coincidence count, as is standard.
    """
    if n_categories < 1:
        raise ValueError("n_categories must be >= 1")
    k = n_categories

    # Build coincidence matrix o_{cc'} from rater pairs within each unit.
    coincidence = np.zeros((k, k), dtype=np.float64)
    n_total = 0.0  # total observed values across units (excluding missing)

    for votes in per_task_votes:
        observed = [v for v in votes if v != ABSTAIN_INDEX and 0 <= v < k]
        m = len(observed)
        if m < 2:
            continue
        # Each ordered pair (i, j) with i != j contributes 1/(m-1).
        # Equivalent: count(c) * count(c') over pairs minus self-pairs.
        cnt = np.zeros(k, dtype=np.float64)
        for v in observed:
            cnt[v] += 1.0
        for c in range(k):
            for cp in range(k):
                if c == cp:
                    coincidence[c, cp] += cnt[c] * (cnt[c] - 1.0) / (m - 1.0)
                else:
                    coincidence[c, cp] += cnt[c] * cnt[cp] / (m - 1.0)
        n_total += m

    if n_total <= 0.0:
        return 0.0

    n_c = coincidence.sum(axis=1)  # marginal totals
    # Observed disagreement D_o = sum_{c != c'} o_{cc'} / n_total
    # Expected disagreement D_e = sum_{c != c'} n_c * n_{c'} / (n_total * (n_total - 1))
    off_diag_o = float(coincidence.sum() - np.trace(coincidence))
    D_o = off_diag_o / n_total

    if n_total <= 1.0:
        return 1.0 if D_o <= 1e-12 else 0.0

    expected_off = 0.0
    for c in range(k):
        for cp in range(k):
            if c == cp:
                continue
            expected_off += n_c[c] * n_c[cp]
    D_e = expected_off / (n_total * (n_total - 1.0))

    if D_e <= 0.0:
        return 1.0 if D_o <= 1e-12 else 0.0
    alpha = 1.0 - (D_o / D_e)
    return float(alpha)


# ---------------------------------------------------------------------------
# Kappa per (domain, task_type) wrapper.
# ---------------------------------------------------------------------------


def _votes_matrix_from_results(
    results: List[ConsensusResult],
) -> Tuple[List[Tuple[int, ...]], int]:
    """Convert a list of ConsensusResult into (per_task_votes, n_categories).

    The category space is the union of all chosen_index values (excluding
    abstain). We re-encode votes to dense indices 0..k-1 to keep the matrix
    compact and well-defined when candidate indices are sparse.
    """
    seen: set = set()
    for r in results:
        for v in r.votes:
            if v.chosen_index != ABSTAIN_INDEX:
                seen.add(v.chosen_index)
    if not seen:
        return [], 0
    sorted_cats = sorted(seen)
    remap = {c: i for i, c in enumerate(sorted_cats)}
    per_task: List[Tuple[int, ...]] = []
    for r in results:
        row = []
        for v in r.votes:
            if v.chosen_index == ABSTAIN_INDEX:
                row.append(ABSTAIN_INDEX)
            else:
                row.append(remap[v.chosen_index])
        per_task.append(tuple(row))
    return per_task, len(sorted_cats)


def kappa_per_domain_task(
    results: List[ConsensusResult],
    task_type: str,
    domains: List[str],
) -> List[KappaReport]:
    """Compute one KappaReport per (domain, task_type) slice.

    D42 amendment (2026-05-23): when every task in the slice has exactly
    two raters, the `fleiss_kappa` field is populated with Cohen's kappa
    instead. Fleiss kappa degenerates with N=2 raters (P_i collapses to a
    0/1 indicator, losing graded agreement information), so Cohen's kappa
    is the correct nominal IAA metric for the 2-model panel. The field
    name is kept as `fleiss_kappa` for backwards compatibility of the JSON
    payload schema; downstream readers should treat the value as a
    "primary nominal kappa" rather than always Fleiss.
    """
    reports: List[KappaReport] = []
    for dom in domains:
        slice_ = [r for r in results if r.domain == dom and r.task_type == task_type]
        per_task, k = _votes_matrix_from_results(slice_)
        if k == 0 or not per_task:
            reports.append(
                KappaReport(
                    domain=dom,
                    task_type=task_type,
                    n_tasks=len(slice_),
                    fleiss_kappa=0.0,
                    krippendorff_alpha=0.0,
                )
            )
            continue
        # Detect 2-rater regime: every row has exactly 2 votes. Fall back
        # to Cohen's kappa in that case.
        if all(len(row) == 2 for row in per_task):
            primary = compute_cohens_kappa(per_task, k)
        else:
            primary = compute_fleiss_kappa(per_task, k)
        ka = compute_krippendorff_alpha(per_task, k)
        reports.append(
            KappaReport(
                domain=dom,
                task_type=task_type,
                n_tasks=len(slice_),
                fleiss_kappa=primary,
                krippendorff_alpha=ka,
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Writers.
# ---------------------------------------------------------------------------


CSV_HEADER = [
    "task_id",
    "task_type",
    "domain",
    "backend_1",
    "backend_2",
    "backend_3",
    "backend_4",
    "consensus_index",
    "n_agree",
    "status",
]


def write_agreement_csv(all_results: List[ConsensusResult], out_path: Path) -> None:
    """Write one row per ConsensusResult.

    Columns: task_id, task_type, domain, backend_1..backend_4 (raw vote indices,
    -1 = ABSTAIN), consensus_index ('' if None), n_agree, status.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" required for csv on Windows to avoid blank lines.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in all_results:
            votes_padded = list(r.votes) + [None] * (N_PANEL - len(r.votes))
            backend_cols = []
            for v in votes_padded[:N_PANEL]:
                if v is None:
                    backend_cols.append("")
                else:
                    backend_cols.append(str(v.chosen_index))
            consensus_str = "" if r.consensus_index is None else str(r.consensus_index)
            writer.writerow(
                [
                    r.task_id,
                    r.task_type,
                    r.domain,
                    backend_cols[0],
                    backend_cols[1],
                    backend_cols[2],
                    backend_cols[3],
                    consensus_str,
                    str(r.n_agree),
                    r.status,
                ]
            )


def write_kappa_json(reports: List[KappaReport], out_path: Path) -> None:
    """Write a JSON list of KappaReport dicts to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "domain": r.domain,
            "task_type": r.task_type,
            "n_tasks": r.n_tasks,
            "fleiss_kappa": r.fleiss_kappa,
            "krippendorff_alpha": r.krippendorff_alpha,
        }
        for r in reports
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
