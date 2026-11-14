# -*- coding: utf-8 -*-
"""
Score the completed EXTERNAL (non-author) rating sheet against the sealed
census reference bins.

Reads rating_sheet_external.csv and writes external_rater_result.json.
Every row must carry an integer 0-4; empty rows abort the run.

The result records rated_by as REQUIRED INPUT: pass the rater identity on the
command line, e.g.

    python score_external_rater.py --rated-by "independent rater, not an author (initials: X.Y.)"

so the provenance of the rating is written into the artifact at scoring time.
ASCII-only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINS = [0, 1, 2, 3, 4]


def quadratic_weighted_kappa(a, b, k=5):
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1.0
    ra = [sum(obs[i][j] for j in range(k)) for i in range(k)]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    exp = [[ra[i] * rb[j] / n for j in range(k)] for i in range(k)]
    w = [[((i - j) ** 2) / float((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * exp[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return None
    return 1.0 - num / den


def krippendorff_alpha_ordinal(a, b):
    # two raters, ordinal metric, no missing values
    pairs = list(zip(a, b))
    n = len(pairs)
    vals = sorted(set(a) | set(b))
    if len(vals) < 2:
        return None
    # ordinal distance via cumulative margins
    counts = {v: 0 for v in vals}
    for x, y in pairs:
        counts[x] += 1
        counts[y] += 1
    total = 2.0 * n

    def delta(c, kk):
        lo, hi = min(c, kk), max(c, kk)
        if lo == hi:
            return 0.0
        s = sum(counts[g] for g in vals if lo <= g <= hi)
        return (s - (counts[lo] + counts[hi]) / 2.0) ** 2

    do = sum(delta(x, y) for x, y in pairs) / n
    de = 0.0
    for i, vi in enumerate(vals):
        for vj in vals:
            if vi == vj:
                continue
            de += counts[vi] * counts[vj] * delta(vi, vj)
    de = de / (total * (total - 1.0))
    if de == 0:
        return None
    return 1.0 - do / de

SHEET = HERE / "rating_sheet_external.csv"
OUT = HERE / "external_rater_result.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rated-by", required=True,
                    help="who rated the sheet; must identify a non-author "
                         "person, recorded verbatim in the result artifact")
    args = ap.parse_args()

    with SHEET.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 29:
        sys.exit("expected 29 rows in %s, found %d" % (SHEET.name, len(rows)))
    empty = [r["item_id"] for r in rows if not str(r["your_bin_0_to_4"]).strip()]
    if empty:
        sys.exit("ABORT: %d unrated items remain: %s"
                 % (len(empty), ", ".join(empty)))
    reference_path = HERE / "reference_bins.json"
    with reference_path.open(encoding="utf-8") as handle:
        ref = json.load(handle)["reference"]
    human, refs, detail = [], [], []
    for r in rows:
        h = int(str(r["your_bin_0_to_4"]).strip())
        if h not in BINS:
            sys.exit("ABORT: invalid bin %r for %s" % (h, r["item_id"]))
        g = ref[r["item_id"]]
        human.append(h)
        refs.append(g)
        detail.append({"item_id": r["item_id"], "external": h,
                       "reference": g, "abs_diff": abs(h - g)})
    exact = sum(1 for d in detail if d["abs_diff"] == 0)
    off1 = sum(1 for d in detail if d["abs_diff"] == 1)
    ge2 = sum(1 for d in detail if d["abs_diff"] >= 2)
    kappa = quadratic_weighted_kappa(human, refs)
    alpha = krippendorff_alpha_ordinal(human, refs)
    result = {
        "rated_by": args.rated_by,
        "sheet": SHEET.name,
        "n": 29, "exact": exact, "off_by_one": off1, "ge2": ge2,
        "quadratic_weighted_cohen_kappa":
            None if kappa is None else round(kappa, 4),
        "krippendorff_alpha_ordinal":
            None if alpha is None else round(alpha, 4),
        "detail": detail,
    }
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
    print("rated_by: %s" % args.rated_by)
    print("n=29  exact=%d  off-by-one=%d  >=2=%d" % (exact, off1, ge2))
    print("quadratic-weighted Cohen kappa = %s"
          % result["quadratic_weighted_cohen_kappa"])
    print("ordinal Krippendorff alpha     = %s"
          % result["krippendorff_alpha_ordinal"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
