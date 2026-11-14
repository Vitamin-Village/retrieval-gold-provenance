# -*- coding: utf-8 -*-
"""S6: statistical-power gate + dataset freeze for the independent benchmark.

Inputs : the QA that survived S4 (multihop bridge verify) AND S5 (leakage gate)
         -- the final candidate set. Per design S6 this is the post-gate QA
         (e.g. <pilot-dir>/qa_leak_passed.jsonl).
Process: aggregate per-domain n + qtype distribution, then use
         paper/eval/statistics.power_analysis to estimate the power of a paired
         two-sided test at a medium effect (Cohen's d ~ 0.3, design section 4).
         The power is computed over the TOTAL QA count (all domains pooled, the
         paired-comparison sample size) and per-domain for reporting.
Gate    : freeze ONLY if the pooled paired-test power >= 0.8 at d ~ 0.3. If the
         power is below 0.8 the dataset is NOT frozen and the driver exits
         non-zero with a "corpus expansion needed" signal. The gate is NEVER
         lowered to force a freeze (design S6 + LOCKED DECISIONS).

Outputs (on PASS): <out> (default paper/gold_v2/qa_independent_v1.jsonl) -- the
         frozen QA, byte-stable (sorted, deterministic). Plus dataset_card.json
         next to it: per-domain n, qtype distribution, kappa (if present in the
         pilot's kappa.json), the power estimate, the generation / verification /
         rater model IDs, a non-circularity guarantee statement, and a
         cross-model IAA limitation statement.

This module makes NO API call and is fully deterministic (no --yes needed; the
only "decision" is the power gate). DO NOT add hyperparameter tuning: the
effect size (0.3 = medium), alpha (0.05), and power floor (0.8) are the
design-locked statistical convention, not tunable knobs.

ACYCLICITY (CI guard, test_no_kg_imports): no prototype.baselines / proposed_kg
/ qa_from_kg imports. Model IDs are recorded as STRINGS in the card; the KG and
its extraction model are never imported or referenced as gold inputs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_PACKAGE_ROOT_FOR_IMPORT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT_FOR_IMPORT)

# Reuse the statistics module's power analysis verbatim (design S6).
from paper.eval.statistics import power_analysis

# ---------------------------------------------------------------------------
# Intentionally NO import of prototype.baselines / proposed_kg / qa_from_kg.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "paper" / "gold_v2" / "qa_independent_v1.jsonl"

# Design-locked statistical convention (NOT tunable knobs):
#   d ~ 0.3 = Cohen's medium-ish effect for a paired retrieval comparison,
#   alpha = 0.05 two-sided, power floor = 0.80. These are the standard values
#   fixed in design section 4 / S6; changing them would weaken the gate.
EFFECT_D = 0.3
ALPHA = 0.05
POWER_FLOOR = 0.8

# Provenance strings recorded in the dataset card (design D-A). These are the
# model IDs used to BUILD the gold; they are recorded as text, never imported.
GENERATOR_MODEL = "claude-sonnet (design D-A; pilot runs may use local llama/mistral)"
VERIFIER_MODEL = "gemini-2.5-flash (cross-family vs generator; S3 + S4 panel)"
RATER_MODEL = "human spot-check (Cohen kappa 2nd rater; design D-A/D-E)"

NON_CIRCULARITY_STATEMENT = (
    "The gold labels are decided solely from raw corpus passage text and a "
    "cross-family non-extraction verifier panel. The proposed KG (its nodes, "
    "edges, and source_docs) is never imported or referenced anywhere in the "
    "gold-build pipeline (statically enforced by test_no_kg_imports), so the "
    "evaluation system cannot generate its own answers -- there is no circular "
    "code-dependency path from the KG to the gold."
)

IAA_LIMITATION_STATEMENT = (
    "Inter-annotator agreement is reported as cross-model / model-vs-human "
    "Cohen's kappa on a per-domain spot-check subset only, not full double "
    "annotation of every item. Kappa is null ('unmeasured') when no human "
    "labels were provided or when all surviving pairs fell in a single category; "
    "a kappa value is never fabricated."
)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_distribution(qa_rows: Sequence[dict]) -> Dict[str, Any]:
    """Per-domain n + qtype distribution + per-domain qtype breakdown."""
    by_domain: Dict[str, int] = defaultdict(int)
    qtype_total: Dict[str, int] = defaultdict(int)
    qtype_by_domain: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for qa in qa_rows:
        dom = qa.get("domain", "unknown")
        qt = qa.get("qtype", "unknown")
        by_domain[dom] += 1
        qtype_total[qt] += 1
        qtype_by_domain[dom][qt] += 1
    return {
        "n_total": len(qa_rows),
        "per_domain_n": dict(sorted(by_domain.items())),
        "qtype_distribution": dict(sorted(qtype_total.items())),
        "qtype_by_domain": {
            d: dict(sorted(v.items())) for d, v in sorted(qtype_by_domain.items())
        },
    }


def estimate_power(
    n_total: int,
    per_domain_n: Dict[str, int],
    *,
    effect: float = EFFECT_D,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """Pooled + per-domain paired-test power at the locked effect / alpha.

    power_analysis requires n > 1; domains (or a pooled set) with n <= 1 report
    power 0.0 (cannot run a paired test), which correctly fails the gate.
    """

    def _safe_power(n: int) -> float:
        if n <= 1:
            return 0.0
        return power_analysis(n, effect, alpha=alpha)

    return {
        "effect_d": effect,
        "alpha": alpha,
        "power_floor": POWER_FLOOR,
        "pooled_n": n_total,
        "pooled_power": _safe_power(n_total),
        "per_domain_power": {
            d: _safe_power(n) for d, n in sorted(per_domain_n.items())
        },
    }


def _load_kappa(pilot_dir: Path) -> Optional[dict]:
    """Best-effort load of the pilot's kappa.json (None if absent)."""
    kp = Path(pilot_dir) / "kappa.json"
    if not kp.exists():
        return None
    try:
        return json.loads(kp.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def build_dataset_card(
    distribution: Dict[str, Any],
    power: Dict[str, Any],
    kappa: Optional[dict],
    *,
    frozen: bool,
) -> Dict[str, Any]:
    """Assemble the dataset card (provenance + gates + guarantees)."""
    return {
        "dataset": "qa_independent_v1",
        "frozen": frozen,
        "n_total": distribution["n_total"],
        "per_domain_n": distribution["per_domain_n"],
        "qtype_distribution": distribution["qtype_distribution"],
        "qtype_by_domain": distribution["qtype_by_domain"],
        "power": power,
        "power_gate_passed": frozen,
        "kappa": kappa,  # null if no pilot kappa.json present
        "models": {
            "generator": GENERATOR_MODEL,
            "verifier": VERIFIER_MODEL,
            "rater": RATER_MODEL,
        },
        "non_circularity_guarantee": NON_CIRCULARITY_STATEMENT,
        "iaa_limitation": IAA_LIMITATION_STATEMENT,
    }


def freeze(
    qa_rows: Sequence[dict],
    pilot_dir: Path,
    *,
    effect: float = EFFECT_D,
    alpha: float = ALPHA,
    power_floor: float = POWER_FLOOR,
) -> Dict[str, Any]:
    """Compute distribution + power, decide the gate, build the card.

    Returns {distribution, power, card, frozen, passing_qa}. `frozen` is True
    iff pooled_power >= power_floor. Caller writes the frozen jsonl + card and
    sets the exit code from `frozen`.
    """
    distribution = aggregate_distribution(qa_rows)
    power = estimate_power(
        distribution["n_total"],
        distribution["per_domain_n"],
        effect=effect,
        alpha=alpha,
    )
    frozen = power["pooled_power"] >= power_floor
    kappa = _load_kappa(Path(pilot_dir))
    card = build_dataset_card(distribution, power, kappa, frozen=frozen)
    return {
        "distribution": distribution,
        "power": power,
        "card": card,
        "frozen": frozen,
        "passing_qa": list(qa_rows),
    }


def write_frozen_dataset(qa_rows: Sequence[dict], out_path: Path) -> int:
    """Write the frozen QA deterministically (sorted by question for stability)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        qa_rows,
        key=lambda r: (
            r.get("question", ""),
            "|".join(sorted(r.get("answer_paper_ids", []))),
        ),
    )
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for qa in ordered:
            f.write(json.dumps(qa, ensure_ascii=True, sort_keys=True))
            f.write("\n")
    return len(ordered)


def write_dataset_card(card: Dict[str, Any], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=True, indent=2, sort_keys=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S6 power gate + dataset freeze")
    parser.add_argument(
        "--pilot-dir",
        required=True,
        help="pilot directory holding the post-S5 QA (qa_leak_passed.jsonl)",
    )
    parser.add_argument(
        "--qa",
        default=None,
        help="QA jsonl to freeze (default <pilot-dir>/qa_leak_passed.jsonl)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--card-out",
        default=None,
        help="dataset_card.json path (default next to --out)",
    )
    args = parser.parse_args(argv)

    pilot_dir = Path(args.pilot_dir)
    qa_path = Path(args.qa) if args.qa else pilot_dir / "qa_leak_passed.jsonl"
    qa_rows = load_jsonl(qa_path)

    result = freeze(qa_rows, pilot_dir)
    card = result["card"]
    power = result["power"]

    card_out = (
        Path(args.card_out)
        if args.card_out
        else Path(args.out).with_name("dataset_card.json")
    )

    # Always write the dataset card (records the gate outcome either way).
    write_dataset_card(card, card_out)

    if not result["frozen"]:
        # Gate FAILED: do NOT freeze. Signal corpus expansion needed (exit 4).
        print(
            "ABORT (exit 4): pooled paired-test power {0:.3f} < floor {1} "
            "at d={2}, alpha={3}, n={4}. CORPUS EXPANSION NEEDED -- do NOT lower "
            "the power gate. dataset_card written to {5} (frozen=false).".format(
                power["pooled_power"],
                POWER_FLOOR,
                EFFECT_D,
                ALPHA,
                power["pooled_n"],
                card_out,
            )
        )
        return 4

    n = write_frozen_dataset(qa_rows, Path(args.out))
    print(
        "S6 done: FROZEN {0} QA (pooled power {1:.3f} >= {2}) -> {3}".format(
            n, power["pooled_power"], POWER_FLOOR, args.out
        )
    )
    print("dataset_card -> {0}".format(card_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
