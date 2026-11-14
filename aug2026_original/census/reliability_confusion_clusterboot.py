# -*- coding: utf-8 -*-
"""W6 + E2: 5x5 rater confusion matrix for the appendix, and a benchmark-level
clustered bootstrap 95% CI for the quadratic-weighted Cohen kappa (resampling
the 13 benchmarks rather than the 29 axis judgments). Deterministic seed.
ASCII-only output."""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reliability_kappa_ci import ROWS, weighted_kappa  # noqa: E402

random.seed(20260611)
N_RES = 10000


def main():
    pairs = [(a, b) for _, a, b in ROWS]
    print("n judgments:", len(pairs))

    # --- 5x5 confusion matrix (rater A rows, rater B cols) ---
    M = [[0] * 5 for _ in range(5)]
    for a, b in pairs:
        M[a][b] += 1
    print("confusion matrix (rows=rater A, cols=rater B, bins 0-4):")
    for i in range(5):
        print("  bin %d: %s" % (i, M[i]))
    exact = sum(M[i][i] for i in range(5))
    off1 = sum(M[i][j] for i in range(5) for j in range(5) if abs(i - j) == 1)
    print("exact=%d off_by_one=%d ge2=%d" % (exact, off1, len(pairs) - exact - off1))

    # --- point estimate ---
    k_full = weighted_kappa(pairs, "quadratic")
    print("quadratic-weighted kappa (29 judgments): %.4f" % k_full)

    # --- benchmark-level clusters ---
    clusters = {}
    for label, a, b in ROWS:
        bench = label.split("/")[0]
        clusters.setdefault(bench, []).append((a, b))
    names = sorted(clusters)
    print("clusters (%d):" % len(names), names)

    boots = []
    n = len(names)
    for _ in range(N_RES):
        sample = []
        for _i in range(n):
            sample.extend(clusters[names[random.randrange(n)]])
        k = weighted_kappa(sample, "quadratic")
        if k == k:  # not NaN
            boots.append(k)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]
    print("clustered bootstrap (%d benchmarks, %d resamples): "
          "95%% CI [%.3f, %.3f] over %d valid resamples"
          % (n, N_RES, lo, hi, len(boots)))


if __name__ == "__main__":
    main()
