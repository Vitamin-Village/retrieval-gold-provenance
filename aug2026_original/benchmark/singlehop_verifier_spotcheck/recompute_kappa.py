# -*- coding: utf-8 -*-
"""Recompute the single-hop verifier spot-check Cohen kappa from raw CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ABSTAIN_INDEX = -1


@dataclass(frozen=True)
class KappaResult:
    n_spotcheck: int
    n_pairs_used: int
    n_exact: int
    raw_agreement_all_spotcheck: float
    observed_agreement_nonabstain: float
    expected_agreement: float
    cohens_kappa: float


def load_verifier_indices(path: Path) -> Dict[str, int]:
    """Load task_id -> primary-verifier option index with duplicate checks."""
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task_id = row["task_id"].strip()
            if task_id in out:
                raise ValueError(f"duplicate task_id in agreement CSV: {task_id}")
            out[task_id] = int(row["verifier_index"])
    if not out:
        raise ValueError("agreement CSV contains no decisions")
    return out


def _parse_blind_index(raw: str, options: Sequence[str]) -> int:
    value = raw.strip()
    if not value:
        return ABSTAIN_INDEX
    if value.upper() == "NONE":
        return len(options) - 1
    return int(value)


def load_pairs(path: Path, verifier: Dict[str, int]) -> Tuple[int, List[Tuple[int, int]]]:
    """Return total sampled rows and non-abstain matched index pairs."""
    pairs: List[Tuple[int, int]] = []
    n_spotcheck = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            n_spotcheck += 1
            task_id = row["task_id"].strip()
            if task_id in seen:
                raise ValueError(f"duplicate task_id in blind labels: {task_id}")
            seen.add(task_id)
            if task_id not in verifier:
                raise ValueError(f"blind label has no verifier decision: {task_id}")
            options = row["options"].split("|")
            blind = _parse_blind_index(row["human_index"], options)
            primary = verifier[task_id]
            if primary != ABSTAIN_INDEX and blind != ABSTAIN_INDEX:
                pairs.append((primary, blind))
    if n_spotcheck == 0:
        raise ValueError("blind-label CSV contains no rows")
    if not pairs:
        raise ValueError("no non-abstain matched pairs")
    return n_spotcheck, pairs


def compute(n_spotcheck: int, pairs: Iterable[Tuple[int, int]]) -> KappaResult:
    materialized = list(pairs)
    n_used = len(materialized)
    n_exact = sum(left == right for left, right in materialized)
    categories = sorted({value for pair in materialized for value in pair})
    left_counts = {c: sum(left == c for left, _ in materialized) for c in categories}
    right_counts = {c: sum(right == c for _, right in materialized) for c in categories}
    p_o = n_exact / n_used
    p_e = sum(left_counts[c] * right_counts[c] for c in categories) / (n_used * n_used)
    if p_e >= 1.0:
        raise ValueError("Cohen kappa undefined: expected agreement is 1")
    return KappaResult(
        n_spotcheck=n_spotcheck,
        n_pairs_used=n_used,
        n_exact=n_exact,
        raw_agreement_all_spotcheck=n_exact / n_spotcheck,
        observed_agreement_nonabstain=p_o,
        expected_agreement=p_e,
        cohens_kappa=(p_o - p_e) / (1.0 - p_e),
    )


def main(argv: Sequence[str] | None = None) -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", type=Path, default=base / "agreement.csv")
    parser.add_argument("--blind-labels", type=Path, default=base / "blind_rater_labels.csv")
    parser.add_argument("--verdict", type=Path, default=base / "kappa_verdict.json")
    args = parser.parse_args(argv)

    verifier = load_verifier_indices(args.agreement)
    n_spotcheck, pairs = load_pairs(args.blind_labels, verifier)
    result = compute(n_spotcheck, pairs)

    with args.verdict.open("r", encoding="utf-8") as handle:
        verdict = json.load(handle)
    if result.n_spotcheck != int(verdict["n_spotcheck"]):
        raise AssertionError("spot-check size differs from verdict")
    if round(result.cohens_kappa, 4) != float(verdict["cohens_kappa"]):
        raise AssertionError("recomputed kappa differs from verdict")
    if abs(result.raw_agreement_all_spotcheck - float(verdict["raw_agreement"])) > 1e-12:
        raise AssertionError("raw agreement differs from verdict")

    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
