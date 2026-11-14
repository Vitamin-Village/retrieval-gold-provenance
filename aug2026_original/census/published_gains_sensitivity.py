"""Sensitivity of the published-gains correlation to which pairs are recorded (Appendix B.11).

The paper's Table 17 records 30 published pairs. The tables it cites contain further pairs
that fit the same description. This script reads published_gains_sensitivity.csv, which
lists the 30 recorded pairs (tier T17) and 18 further pairs of the cited tables (tiers a
and b), and prints the per-benchmark medians and the Spearman correlations for three sets:

  T17      the 30 pairs of Table 17 (28 with a sparse or text-only baseline, 2 dense)
  T17+a    plus QA-GNN against BM25 on STaRK-Amazon and STaRK-Prime
  T17+a+b  plus every further pair of the cited tables under the metric that Table 17
           already takes from each table, one variant per system and benchmark
           (48 pairs, 42 with a sparse or text-only baseline)

As in the paper, pairs with a dense baseline (GraphRAG-Bench) are left out of the
correlations, and the benchmark-level correlation is taken over the medians of the
eleven remaining benchmarks, STaRK counted as its three sub-benchmarks.

Run from this folder: python published_gains_sensitivity.py
Expected: rho over benchmark medians 0.86 (T17), 0.66 (T17+a), 0.35 (T17+a+b).
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# KG-derived census component per benchmark (Table 1; STaRK's value for its three parts)
C_KG = {"MuSiQue": 0.0, "HotpotQA": 0.0, "MultiHop-RAG": 0.0, "NQ": 0.0, "TriviaQA": 0.0,
        "STaRK-Amazon": 0.75, "STaRK-MAG": 0.75, "STaRK-Prime": 0.75, "GraphRAG-Bench": 0.75,
        "2Wiki": 1.0, "MetaQA": 1.0, "WebQSP": 1.0}
SETS = [("T17", {"T17"}), ("T17+a", {"T17", "a"}), ("T17+a+b", {"T17", "a", "b"})]


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return rk


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def main():
    with open(os.path.join(HERE, "published_gains_sensitivity.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gain"] = float(r["gain_pp"])
        r["dense"] = r["dense_baseline"] == "True"
    for name, tiers in SETS:
        sel = [r for r in rows if r["tier"] in tiers]
        sparse = [r for r in sel if not r["dense"]]
        by = {}
        for r in sparse:
            by.setdefault(r["benchmark"], []).append(r["gain"])
        benches = [b for b in C_KG if b in by]
        med = {b: median(by[b]) for b in benches}
        rho_b = spearman([C_KG[b] for b in benches], [med[b] for b in benches])
        rho_p = spearman([C_KG[r["benchmark"]] for r in sparse], [r["gain"] for r in sparse])
        print("%-8s pairs %2d (sparse or text-only %2d, dense %d)  rho over %d benchmark medians %.4f  rho over pairs %.4f"
              % (name, len(sel), len(sparse), len(sel) - len(sparse), len(benches), rho_b, rho_p))
        for b in benches:
            print("    %-14s C_KG %.2f  pairs %d  median %6.2f  min %6.1f  max %6.1f"
                  % (b, C_KG[b], len(by[b]), med[b], min(by[b]), max(by[b])))


if __name__ == "__main__":
    main()
