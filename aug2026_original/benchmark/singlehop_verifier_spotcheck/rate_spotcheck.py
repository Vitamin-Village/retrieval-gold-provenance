# -*- coding: utf-8 -*-
"""S3 kappa 2nd-rater: a BLIND, cross-family Claude Haiku auto-labeler.

The S3 verifier is Gemini Flash (Google). Cohen's kappa needs a 2nd rater that
judges the SAME construct on the SAME items but is a DIFFERENT model family and
is BLIND to the verifier's vote. This module is that rater: a Claude Haiku model
whose system prompt is the published rater rubric
(paper/gold_v2/verification_rater_rubric.md) and whose user prompt is the
EXACT verification task the Gemini verifier saw.

Construct identity (why kappa means "reliability", not "rule difference"):
  - The rater reconstructs each spotcheck item with the SAME helper the verifier
    used (verify_answer_papers.build_verification_task +
    cross_judge.build_entity_user_prompt). No ad-hoc rebuild -> byte-identical
    options/excerpts/order to what Gemini saw.
  - The rater's response is parsed with the SAME parser the verifier used
    (cross_judge.parse_index_or_abstain: 'INDEX: <n>' / 'NONE' -> ABSTAIN).

BLINDNESS (statically asserted in _assert_blind):
  - The rater input dict carries ONLY question + numbered option excerpts (the
    reconstructed user prompt). It NEVER contains openai_index (Gemini vote),
    true_index, verifier_index, accepted, or any answer-key field. A guard
    asserts none of those keys/substrings are present before the call.

Output: pilot_kappa1/human_labels_claude.csv in the format
run_kappa_pilot --human-labels (and verify_answer_papers.load_human_labels_csv)
consume: columns task_id, human_index. human_index is the option index the
rater chose, or the literal NONE (mapped by the loader to the last option index,
i.e. the same NONE slot the verifier used).

SAFETY: prints a call-count + cost estimate first; refuses any live Anthropic
call unless --yes is passed. Unit tests inject a stub backend (zero real calls).

ACYCLICITY (CI guard, test_no_kg_imports): NO import of prototype.baselines,
proposed_kg, or qa_from_kg. Only the S1 passage pool, the reconstructed S3 task,
and a cross-family Claude backend are used.

DO NOT add hyperparameter tuning; structural reuse only.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Reuse the SAME task builder + prompt builder + parser the verifier used, so the
# rater judges a byte-identical construct (fair comparison -> kappa is meaningful).
from paper.eval.cross_judge import (
    ABSTAIN_INDEX,
    DEFAULT_MAX_TOKENS,
    build_entity_user_prompt,
    parse_index_or_abstain,
)
from paper.gold_v2 import verify_answer_papers as ver

# ---------------------------------------------------------------------------
# Intentionally NO import of:
#   prototype.baselines, proposed_kg, qa_from_kg
# (test_no_kg_imports scans this file's AST.)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The DEFAULT_* paths below reference the development-repo layout. The
# released spot-check artifacts live in this directory (qa_candidates.jsonl,
# qa_labeled.jsonl, agreement.csv, blind_rater_labels.csv); the offline
# reproduction path is recompute_kappa.py, which needs no live backend.
DEFAULT_PILOT_DIR = PROJECT_ROOT / "paper" / "gold_v2" / "pilot_kappa1"
DEFAULT_PASSAGES = PROJECT_ROOT / "paper" / "gold_v2" / "passages.jsonl"
DEFAULT_RUBRIC = PROJECT_ROOT / "paper" / "gold_v2" / "verification_rater_rubric.md"
DEFAULT_OUT_NAME = "human_labels_claude.csv"

# Operational output directive appended to the rubric when used as the rater
# system prompt. The discursive rubric alone let Claude Haiku emit chain-of-
# thought reasoning that overran max_tokens before the `INDEX:` line, so the
# parser saw no index and recorded ABSTAIN (all-NONE -> degenerate kappa). This
# forces the SAME terse, parseable single-line output the Gemini verifier
# already produces, so kappa compares JUDGMENTS, not output-format compliance.
RATER_OUTPUT_DIRECTIVE = (
    "\n\n=== OUTPUT FORMAT (STRICT, OVERRIDES ALL ABOVE) ===\n"
    "Do NOT explain. Do NOT reason in prose. Do NOT restate the candidates.\n"
    "Your ENTIRE reply MUST be exactly one line:\n"
    "INDEX: <n>\n"
    "where <n> is the chosen option number (an integer), or, if no candidate "
    "states the answer:\n"
    "INDEX: NONE\n"
    "Output only that single line and nothing else."
)

# Output CSV columns consumed by run_kappa_pilot --human-labels /
# verify_answer_papers.load_human_labels_csv (it reads task_id + human_index;
# 'options' is carried so the loader can map a NONE literal to the last index,
# matching what the verifier saw).
OUTPUT_CSV_HEADER = ["task_id", "domain", "question", "options", "human_index"]

# Forbidden keys/substrings that would leak the verifier vote or answer key into
# the rater input. Checked statically in _assert_blind before any call.
_BLIND_FORBIDDEN_KEYS = (
    "openai_index",
    "verifier_index",
    "true_index",
    "accepted",
    "answer_paper_ids",
    "answer",
    "s3_true_index",
    "s3_verifier_index",
    "s3_accepted",
)

# Rough Anthropic Haiku per-call cost (planning banner only; not used in any
# computation that affects output). ~3k input excerpt tokens + tiny output.
EST_USD_PER_HAIKU_CALL = 0.0013


# ---------------------------------------------------------------------------
# Reconstruct the verifier tasks (byte-identical) from pilot artifacts.
# ---------------------------------------------------------------------------


def reconstruct_verifier_tasks(
    candidates: Sequence[dict],
    passages: Sequence[dict],
    *,
    n_trap: int = ver.N_TRAP,
    n_trap_lexical: int = ver.N_TRAP_LEXICAL,
    trap_seed: int = ver.TRAP_SEED,
) -> Dict[str, dict]:
    """Rebuild every verification task EXACTLY as verify_candidates built it.

    Mirrors verify_answer_papers.verify_candidates' loop (single shared RNG
    consumed in candidate order, cand_index -> task_id 'qa{i:05d}') so the trap
    sampling and md5 option shuffle reproduce the verifier's options bit-for-bit.

    Returns {task_id: {task_id, domain, question, option_paper_ids,
    user_prompt}}. NOTE: this dict deliberately carries NO verifier vote / answer
    key (true_index is dropped here so it can never leak into the rater input).
    """
    rng = random.Random(trap_seed)
    passages_by_paper = ver.index_passages_by_paper(passages)
    domain_of_paper = {p["paper_id"]: p["domain"] for p in passages}

    out: Dict[str, dict] = {}
    for i, cand in enumerate(candidates):
        traps = ver.select_trap_papers(
            cand,
            passages_by_paper,
            domain_of_paper,
            rng,
            n_trap=n_trap,
            n_trap_lexical=n_trap_lexical,
        )
        task, option_paper_ids, _true_index = ver.build_verification_task(
            cand, i, passages_by_paper, domain_of_paper, traps
        )
        # Reconstruction depends on `candidates` being in the SAME order as the
        # original verify_candidates run (the single RNG + the cand_index->
        # task_id mapping both assume it). Assert the derived task_id matches the
        # enumerate index so a silently-reordered candidate list fails loudly
        # instead of producing a non-identical (and thus invalid) reconstruction.
        assert task.task_id == "qa{0:05d}".format(i), (
            "candidate order mismatch at {0}: got {1}".format(i, task.task_id)
        )
        # _true_index intentionally discarded -- it must never reach the rater
        # input (blindness). Kept as `_true_index` only to unpack the tuple.
        # build_entity_user_prompt is the SAME function judge_entity_tasks calls,
        # so this string is byte-identical to the verifier's user prompt.
        user_prompt = build_entity_user_prompt(task)
        out[task.task_id] = {
            "task_id": task.task_id,
            "domain": task.domain,
            "question": cand.get("question", ""),
            "option_paper_ids": option_paper_ids,
            "user_prompt": user_prompt,
        }
    return out


# ---------------------------------------------------------------------------
# Blindness guard.
# ---------------------------------------------------------------------------


def _assert_blind(rater_input: dict) -> None:
    """Static guarantee: the rater input leaks no verifier vote / answer key.

    Raises AssertionError if any forbidden key is present, OR if the user_prompt
    text contains a forbidden field token. This runs BEFORE every backend call.
    """
    keys = set(rater_input.keys())
    leaked = keys & set(_BLIND_FORBIDDEN_KEYS)
    assert not leaked, "BLINDNESS VIOLATION: rater input carries {0}".format(
        sorted(leaked)
    )
    # The prompt must not embed a vote/answer token either.
    prompt = rater_input.get("user_prompt", "")
    for tok in ("openai_index", "verifier_index", "true_index", "s3_accepted"):
        assert tok not in prompt, (
            "BLINDNESS VIOLATION: token '{0}' leaked into rater prompt".format(tok)
        )


# ---------------------------------------------------------------------------
# Rate one item with the (injected) Claude Haiku backend.
# ---------------------------------------------------------------------------


def rate_one(
    rater_input: dict,
    backend,
    rubric_system_prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> int:
    """Run the rater on a single reconstructed task -> chosen option index.

    Uses the SAME parser as the verifier (parse_index_or_abstain), so 'INDEX: n'
    maps to n and 'NONE' maps to ABSTAIN_INDEX -- a fair, identical decode path.
    _assert_blind runs first so a leak aborts before any spend.
    """
    _assert_blind(rater_input)
    raw = backend.chat(
        rubric_system_prompt, rater_input["user_prompt"], max_tokens=max_tokens
    )
    return parse_index_or_abstain(raw)


def rate_spotcheck(
    spotcheck_task_ids: Sequence[str],
    reconstructed: Dict[str, dict],
    backend,
    rubric_system_prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[dict]:
    """Rate every spotcheck task_id -> list of result rows (ordered).

    Each row: {task_id, domain, question, option_paper_ids, rater_index}.
    A spotcheck id absent from `reconstructed` is skipped (kept robust: a stale
    spotcheck CSV must not crash the run).
    """
    rows: List[dict] = []
    skipped: List[str] = []
    for tid in spotcheck_task_ids:
        item = reconstructed.get(tid)
        if item is None:
            skipped.append(tid)
            continue
        rater_input = {
            "task_id": item["task_id"],
            "user_prompt": item["user_prompt"],
        }
        idx = rate_one(rater_input, backend, rubric_system_prompt, max_tokens=max_tokens)
        rows.append(
            {
                "task_id": item["task_id"],
                "domain": item["domain"],
                "question": item["question"],
                "option_paper_ids": item["option_paper_ids"],
                "rater_index": idx,
            }
        )
    if skipped:
        # Visibility: a stale spotcheck CSV silently shrinks the kappa sample.
        # Warn so the operator can see the effective denominator (kappa is only
        # computed over the rows actually rated).
        import sys as _sys

        _sys.stderr.write(
            "WARNING: {0} spotcheck id(s) absent from reconstruction, "
            "skipped (kappa sample reduced): {1}\n".format(
                len(skipped), ", ".join(skipped[:10])
            )
        )
    return rows


# ---------------------------------------------------------------------------
# CSV I/O.
# ---------------------------------------------------------------------------


def load_spotcheck_task_ids(spotcheck_path: Path) -> List[str]:
    """Read the spotcheck CSV -> ordered list of its task_id values."""
    spotcheck_path = Path(spotcheck_path)
    ids: List[str] = []
    with open(spotcheck_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = (row.get("task_id") or "").strip()
            if tid:
                ids.append(tid)
    return ids


def _human_index_cell(rater_index: int, n_options: int) -> str:
    """Map the rater's index to the human_index cell value.

    The verifier's NONE option is the LAST option; the loader maps the literal
    'NONE' back to that last index. So abstain / out-of-range -> 'NONE'; an index
    pointing at the last (NONE) slot also -> 'NONE'; any content index -> str(i).
    This keeps the rater's NONE convention identical to the verifier's.
    """
    none_slot = max(0, n_options - 1)
    if rater_index == ABSTAIN_INDEX:
        return "NONE"
    if rater_index < 0 or rater_index >= n_options or rater_index == none_slot:
        return "NONE"
    return str(rater_index)


def write_human_labels_csv(rows: Sequence[dict], out_path: Path) -> int:
    """Write the rater labels in run_kappa_pilot --human-labels format.

    Columns: task_id, domain, question, options (paper_id pipe-list, NONE last),
    human_index (rater's chosen index or the literal NONE). Returns row count.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_CSV_HEADER)
        for r in rows:
            opt_ids = r["option_paper_ids"]
            opts = "|".join((pid if pid else "NONE") for pid in opt_ids)
            human_index = _human_index_cell(r["rater_index"], len(opt_ids))
            writer.writerow(
                [r["task_id"], r["domain"], r["question"], opts, human_index]
            )
            n += 1
    return n


# ---------------------------------------------------------------------------
# Estimate banner.
# ---------------------------------------------------------------------------


def estimate_cost(n_tasks: int) -> dict:
    """Per-task Haiku call count + USD band (planning banner only)."""
    usd = n_tasks * EST_USD_PER_HAIKU_CALL
    return {
        "haiku_calls": n_tasks,
        "usd_low": round(usd, 4),
        "usd_high": round(usd * 2.0, 4),
    }


def print_estimate_banner(est: dict) -> None:
    """ASCII-only pre-flight estimate."""
    print("=" * 60)
    print("KAPPA 2nd-RATER (Claude Haiku, BLIND) -- PRE-FLIGHT ESTIMATE")
    print("=" * 60)
    print("Haiku calls    : ~{0}".format(est["haiku_calls"]))
    print("estimated cost : ~${0} - ${1} USD".format(est["usd_low"], est["usd_high"]))
    print("model family   : Anthropic (cross-family vs Gemini verifier)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _build_default_backend():
    """Construct the Claude Haiku backend (live use only; needs ANTHROPIC_API_KEY)."""
    from paper.eval.cross_judge_backends import ClaudeHaikuBackend

    return ClaudeHaikuBackend()


def run(
    pilot_dir: Path,
    passages_path: Path,
    rubric_path: Path,
    out_path: Path,
    backend,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Tuple[int, str]:
    """Reconstruct -> rate -> write. Returns (n_rows, out_path_str)."""
    pilot_dir = Path(pilot_dir)
    candidates = ver.load_jsonl(pilot_dir / "qa_candidates.jsonl")
    passages = ver.load_jsonl(Path(passages_path))
    rubric_system_prompt = (
        Path(rubric_path).read_text(encoding="utf-8") + RATER_OUTPUT_DIRECTIVE
    )
    spotcheck_ids = load_spotcheck_task_ids(pilot_dir / "human_spotcheck.csv")

    reconstructed = reconstruct_verifier_tasks(candidates, passages)
    rows = rate_spotcheck(
        spotcheck_ids, reconstructed, backend, rubric_system_prompt, max_tokens=max_tokens
    )
    n = write_human_labels_csv(rows, out_path)
    return n, str(out_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="S3 kappa 2nd-rater: blind cross-family Claude Haiku labeler"
    )
    parser.add_argument("--pilot-dir", default=str(DEFAULT_PILOT_DIR))
    parser.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    parser.add_argument("--rubric", default=str(DEFAULT_RUBRIC))
    parser.add_argument(
        "--spotcheck",
        default=None,
        help="spotcheck CSV (default: <pilot-dir>/human_spotcheck.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output labels CSV (default: <pilot-dir>/human_labels_claude.csv)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm live Anthropic spend; without it only the estimate prints",
    )
    args = parser.parse_args(argv)

    pilot_dir = Path(args.pilot_dir)
    spotcheck_path = (
        Path(args.spotcheck) if args.spotcheck else pilot_dir / "human_spotcheck.csv"
    )
    out_path = Path(args.out) if args.out else pilot_dir / DEFAULT_OUT_NAME

    spotcheck_ids = load_spotcheck_task_ids(spotcheck_path)
    est = estimate_cost(len(spotcheck_ids))
    print_estimate_banner(est)

    if not args.yes:
        print(
            "STOPPED: no --yes flag. Re-run with --yes to make the live Claude "
            "Haiku calls estimated above. (No API call was made.)"
        )
        return 0

    backend = _build_default_backend()  # needs ANTHROPIC_API_KEY
    candidates = ver.load_jsonl(pilot_dir / "qa_candidates.jsonl")
    passages = ver.load_jsonl(Path(args.passages))
    # Append the strict output directive (same as run()): without it Claude
    # Haiku emits chain-of-thought that overruns max_tokens before the INDEX
    # line, parsing as ABSTAIN -> all-NONE -> degenerate kappa.
    rubric_system_prompt = (
        Path(args.rubric).read_text(encoding="utf-8") + RATER_OUTPUT_DIRECTIVE
    )
    reconstructed = reconstruct_verifier_tasks(candidates, passages)
    rows = rate_spotcheck(spotcheck_ids, reconstructed, backend, rubric_system_prompt)
    n = write_human_labels_csv(rows, out_path)

    print(
        "2nd-rater done: {0} labels -> {1} (feed run_kappa_pilot "
        "--human-labels {1})".format(n, out_path)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
