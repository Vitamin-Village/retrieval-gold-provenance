# -*- coding: utf-8 -*-
"""Two-hop slice driver: text-only seeds -> generation -> dual verification ->
leakage gate -> content-blind freeze.

Builds the frozen two-hop slice of the acyclic (C=0) benchmark, end to end,
WITHOUT ever reading the KG:

  discover : paper/gold_v2/discover_bridge_pairs_textonly.py (M0', KG-free)
  gen      : generate_from_seed_pairs with the released 2-hop bridge prompt;
             generator = Claude Sonnet (design D-A, same family panel as the
             frozen single-hop set; ANTHROPIC_API_KEY from .env)
  verify   : S4 structural pre-gate + TWO cross-family verifiers -- Gemini
             Flash (GEMINI_API_KEY, the S3/S4 panel verifier) AND a local
             Qwen2.5-14B via Ollama (third family). A candidate survives only
             if BOTH verifiers keep it (double-YES). Cohen kappa between the
             two verifiers is recorded; never fabricated (null if degenerate).
  leak     : the existing S5 leakage gate, unmodified.
  freeze   : up to FREEZE_TOTAL questions, per-domain cap FREEZE_PER_DOMAIN,
             selected CONTENT-BLIND by stable lexicographic sort on
             (domain, question) -- a deterministic rule fixed before looking at
             any candidate, so no quality cherry-picking is possible.

Stages run individually (--stage) or all in order (--stage all). The frozen
single-hop benchmark (qa_independent_v1.jsonl, dataset_card.json) is NEVER
touched. ASCII-only console output (Windows cp949 safe).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.gold_v2.discover_bridge_pairs_textonly import (  # noqa: E402
    DEFAULT_OUTPUT as TEXTONLY_PAIRS,
    DEFAULT_PASSAGES,
    discover,
    load_passages,
)
from paper.gold_v2.generate_qa_candidates import (  # noqa: E402
    generate_from_seed_pairs,
    load_bridge_pairs,
    write_candidates_jsonl,
)
from paper.gold_v2.verify_multihop_bridge import (  # noqa: E402
    VERDICT_KEPT,
    _candidate_key,
    _kept_multihop_row,
    select_multihop_candidates,
    verify_bridges,
)

SLICE_DIR = PROJECT_ROOT / "paper" / "gold_v2" / "twohop_slice"
FROZEN_OUT = PROJECT_ROOT / "paper" / "gold_v2" / "qa_independent_v2_twohop.jsonl"
CARD_OUT = PROJECT_ROOT / "paper" / "gold_v2" / "dataset_card_twohop.json"

GENERATOR_MODEL = "claude-sonnet"      # design D-A panel, as in the v1 card
VERIFIER_1 = "gemini-2.5-flash"        # cross-family vs generator (S3/S4 panel)
VERIFIER_2 = "qwen2.5:14b"             # third family, local Ollama

# Freeze rule, fixed BEFORE any candidate is inspected (integrity requirement):
# 50-question target, even per-domain cap. 50 = the slice size announced in the
# improvement plan; 13 = ceil(50/4), so no domain dominates.
FREEZE_TOTAL = 50
FREEZE_PER_DOMAIN = 13

# Generation cap per domain: 3x the per-domain freeze quota, absorbing the
# attrition through S4 + leakage without an open-ended API spend.
GEN_MAX_PER_DOMAIN = 39
# HARD attempt budget: generate_from_seed_pairs counts only
# ACCEPTED candidates toward max_per_domain, so failed generations would spend
# API calls without bound over the 51k seed rows. We therefore pre-slice the
# seeds to at most ATTEMPT_FACTOR x GEN_MAX_PER_DOMAIN rows per domain, one
# pair per bridge surface (deterministic first pair in stable sort order), so
# the total API-call ceiling is fixed and the bridges are diverse.
ATTEMPT_FACTOR = 4


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True))
            f.write("\n")


# ---------------------------------------------------------------------------
# Stages.
# ---------------------------------------------------------------------------


def stage_discover() -> None:
    passages = load_passages(DEFAULT_PASSAGES)
    rows = discover(passages)
    _write_jsonl(rows, TEXTONLY_PAIRS)
    print("discover: {0} pair rows -> {1}".format(len(rows), TEXTONLY_PAIRS))


def _build_generator(backend_name: str):
    if backend_name == "sonnet":
        from paper.eval.cross_judge_backends import ClaudeSonnetBackend

        return ClaudeSonnetBackend()
    if backend_name == "mistral":
        from paper.eval.cross_judge_backends import OllamaBackend

        return OllamaBackend("mistral-small:24b")
    raise SystemExit("unknown generator backend: %s" % backend_name)


def _preslice_seeds(seeds: Sequence[dict], max_per_domain: int) -> List[dict]:
    """Deterministic attempt-budget pre-slice.

    Keeps at most ATTEMPT_FACTOR * max_per_domain seed rows per domain, at most
    ONE pair per bridge surface within a domain (the first in the stable sort
    order of the discovery output), so the API-call ceiling is hard-capped and
    the attempted bridges are diverse. Content-blind: selection is by the
    pre-existing stable ordering only.
    """
    budget = ATTEMPT_FACTOR * max_per_domain
    taken: Dict[str, int] = defaultdict(int)
    seen_bridge: set = set()
    out: List[dict] = []
    for s in seeds:
        dom = str(s.get("domain") or "unknown")
        key = (dom, str(s.get("bridge_surface") or ""))
        if taken[dom] >= budget or key in seen_bridge:
            continue
        seen_bridge.add(key)
        taken[dom] += 1
        out.append(s)
    return out


def stage_gen(backend_name: str, max_per_domain: int) -> None:
    passages = load_passages(DEFAULT_PASSAGES)
    seeds = _preslice_seeds(load_bridge_pairs(TEXTONLY_PAIRS), max_per_domain)
    print("gen: attempt budget = {0} seeds after pre-slice".format(len(seeds)))
    backend = _build_generator(backend_name)
    rows = generate_from_seed_pairs(
        seeds, passages, backend, max_per_domain=max_per_domain
    )
    for r in rows:
        r["discovery"] = "textonly"
    write_candidates_jsonl(rows, SLICE_DIR / "qa_candidates.jsonl")
    by_dom: Dict[str, int] = defaultdict(int)
    for r in rows:
        by_dom[r.get("domain", "unknown")] += 1
    print(
        "gen: {0} candidates from {1} seeds (backend={2})".format(
            len(rows), len(seeds), backend_name
        )
    )
    for dom in sorted(by_dom):
        print("  {0}: {1}".format(dom, by_dom[dom]))


def _binary_cohen_kappa(pairs: Sequence[Tuple[int, int]]) -> Optional[float]:
    """Cohen kappa for two binary raters; None when unmeasured/degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / float(n)
    pa1 = sum(1 for a, _b in pairs if a == 1) / float(n)
    pb1 = sum(1 for _a, b in pairs if b == 1) / float(n)
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def stage_verify() -> None:
    from paper.eval.cross_judge_backends import GeminiFlashBackend, OllamaBackend

    passages = load_passages(DEFAULT_PASSAGES)
    rows = _read_jsonl(SLICE_DIR / "qa_candidates.jsonl")
    multihop = select_multihop_candidates(rows)

    rec_g = verify_bridges(multihop, passages, GeminiFlashBackend())
    rec_q = verify_bridges(multihop, passages, OllamaBackend(VERIFIER_2))
    # An empty verifier response over a non-empty candidate set
    # is a hard failure, never a silent empty benchmark.
    if multihop and (not rec_g or not rec_q):
        raise SystemExit(
            "verify: empty verifier output (gemini=%d, qwen=%d records for %d "
            "candidates)" % (len(rec_g), len(rec_q), len(multihop))
        )
    by_key_q = {_candidate_key(r["candidate"]): r for r in rec_q}

    kept: List[dict] = []
    audit: List[dict] = []
    kappa_pairs: List[Tuple[int, int]] = []
    n_missing = 0
    for rg in rec_g:
        key = _candidate_key(rg["candidate"])
        rq = by_key_q.get(key)
        if rq is None:
            # A missing second-verifier record is recorded
            # explicitly, never silently dropped.
            n_missing += 1
            audit.append(
                {
                    "question": rg["candidate"].get("question", ""),
                    "domain": rg["domain"],
                    "verdict_gemini": rg["verdict"],
                    "verdict_qwen": "MISSING",
                    "reason_gemini": rg["reason"],
                    "reason_qwen": "no record from second verifier",
                    "kept": False,
                }
            )
            continue
        vg, vq = rg["verdict"], rq["verdict"]
        # kappa over candidates on which BOTH verifiers returned a model
        # judgment (verifier_index 0=YES / 1=NO; -1 = pre-gate or abstain).
        if rg["verifier_index"] in (0, 1) and rq["verifier_index"] in (0, 1):
            kappa_pairs.append(
                (1 - rg["verifier_index"], 1 - rq["verifier_index"])
            )
        both_kept = vg == VERDICT_KEPT and vq == VERDICT_KEPT
        audit.append(
            {
                "question": rg["candidate"].get("question", ""),
                "domain": rg["domain"],
                "verdict_gemini": vg,
                "verdict_qwen": vq,
                "reason_gemini": rg["reason"],
                "reason_qwen": rq["reason"],
                "kept": both_kept,
            }
        )
        if both_kept:
            # Defensive copy in case the helper aliases the
            # candidate dict.
            row = dict(_kept_multihop_row(rg["candidate"]))
            row["s4_verifiers"] = [VERIFIER_1, VERIFIER_2]
            kept.append(row)

    kappa = _binary_cohen_kappa(kappa_pairs)
    _write_jsonl(kept, SLICE_DIR / "qa_labeled.jsonl")
    _write_jsonl(audit, SLICE_DIR / "multihop_verified_dual.jsonl")
    stats = {
        "n_candidates": len(multihop),
        "n_kept_both": len(kept),
        "n_missing_second_verifier": n_missing,
        "n_kappa_pairs": len(kappa_pairs),
        "verifier_cohen_kappa": None if kappa is None else round(kappa, 4),
        "verifiers": [VERIFIER_1, VERIFIER_2],
    }
    with open(SLICE_DIR / "verify_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(
        "verify: {0} candidates -> kept(both)={1}, kappa={2} over {3} pairs".format(
            len(multihop), len(kept), stats["verifier_cohen_kappa"],
            len(kappa_pairs),
        )
    )


def stage_leak() -> None:
    from paper.gold_v2.leakage_gate import main as leak_main

    rc = leak_main(
        [
            "--pilot-dir", str(SLICE_DIR),
            "--report-out", str(SLICE_DIR / "leakage_report_twohop.json"),
            "--passing-out", str(SLICE_DIR / "qa_leak_passed.jsonl"),
        ]
    )
    if rc != 0:
        raise SystemExit("leakage gate failed rc=%d" % rc)
    survivors = _read_jsonl(SLICE_DIR / "qa_leak_passed.jsonl")
    print("leak: {0} survivors".format(len(survivors)))


def stage_freeze(force: bool = False) -> None:
    # A frozen benchmark must never be silently overwritten.
    if FROZEN_OUT.exists() and not force:
        raise SystemExit(
            "freeze: %s already exists; pass --force to re-freeze" % FROZEN_OUT
        )
    rows = _read_jsonl(SLICE_DIR / "qa_leak_passed.jsonl")
    two_hop = [r for r in rows if r.get("hop_count") == 2]
    # CONTENT-BLIND deterministic selection: stable sort on (domain, question,
    # answer papers, answer) -- unique tie-breakers --
    # take up to FREEZE_PER_DOMAIN each, FREEZE_TOTAL overall. The rule was
    # fixed before any candidate was inspected.
    two_hop.sort(
        key=lambda r: (
            r.get("domain", ""),
            r.get("question", ""),
            "|".join(sorted(r.get("answer_paper_ids", []))),
            r.get("answer", ""),
        )
    )
    taken: Dict[str, int] = defaultdict(int)
    frozen: List[dict] = []
    for r in two_hop:
        dom = r.get("domain", "unknown")
        if taken[dom] >= FREEZE_PER_DOMAIN or len(frozen) >= FREEZE_TOTAL:
            continue
        taken[dom] += 1
        frozen.append(r)
    _write_jsonl(frozen, FROZEN_OUT)

    stats_path = SLICE_DIR / "verify_stats.json"
    vstats = {}
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            vstats = json.load(f)
    card = {
        "dataset": "qa_independent_v2_twohop",
        "frozen": True,
        "n_total": len(frozen),
        "per_domain_n": dict(sorted(taken.items())),
        "hop_count": 2,
        "qtype": "multihop_bridge",
        "models": {
            "generator": GENERATOR_MODEL,
            "verifiers": [VERIFIER_1, VERIFIER_2],
            "verifier_rule": "kept only when BOTH cross-family verifiers keep",
        },
        "verifier_cohen_kappa": vstats.get("verifier_cohen_kappa"),
        "selection_rule": (
            "content-blind deterministic freeze: stable sort on (domain, "
            "question), per-domain cap %d, total cap %d; rule fixed before "
            "inspecting any candidate" % (FREEZE_PER_DOMAIN, FREEZE_TOTAL)
        ),
        "discovery": (
            "bridge pairs are discovered from corpus passage text only "
            "(discover_bridge_pairs_textonly.py, inside the gold_v2 "
            "no-KG-imports CI boundary); the KG is never read at any stage"
        ),
        "non_circularity_guarantee": (
            "The gold labels are decided solely from raw corpus passage text, "
            "a text-only bridge discovery, and a dual cross-family "
            "non-extraction verifier panel. The proposed KG (its nodes, "
            "edges, and source_docs) is never imported or referenced anywhere "
            "in the gold-build pipeline (statically enforced by "
            "test_no_kg_imports), including bridge-pair discovery."
        ),
    }
    with open(CARD_OUT, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(
        "freeze: {0} questions -> {1}".format(len(frozen), FROZEN_OUT)
    )
    for dom in sorted(taken):
        print("  {0}: {1}".format(dom, taken[dom]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="two-hop slice driver")
    parser.add_argument(
        "--stage",
        choices=["discover", "gen", "verify", "leak", "freeze", "all"],
        required=True,
    )
    parser.add_argument(
        "--gen-backend", choices=["sonnet", "mistral"], default="sonnet"
    )
    parser.add_argument(
        "--gen-max-per-domain", type=int, default=GEN_MAX_PER_DOMAIN
    )
    parser.add_argument(
        "--force", action="store_true",
        help="allow re-freezing over an existing frozen two-hop slice",
    )
    args = parser.parse_args(argv)

    stages = (
        ["discover", "gen", "verify", "leak", "freeze"]
        if args.stage == "all"
        else [args.stage]
    )
    for st in stages:
        if st == "discover":
            stage_discover()
        elif st == "gen":
            stage_gen(args.gen_backend, args.gen_max_per_domain)
        elif st == "verify":
            stage_verify()
        elif st == "leak":
            stage_leak()
        elif st == "freeze":
            stage_freeze(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
