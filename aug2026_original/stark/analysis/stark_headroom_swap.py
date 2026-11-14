# -*- coding: utf-8 -*-
"""Headroom-controlled STaRK swap (post-hoc stratification of the existing
eval, no new build or eval). The full STaRK swap is near-null because the
near-lexical entity retrieval leaves the graph retrievers little room over a
strong BM25. We test whether the swap appears on the subset where there IS
headroom -- the questions BM25 misses (hit@5 = 0). BM25 is bit-identical across
the silver and OpenIE conditions, so this stratum is condition-independent: it
selects on a fixed baseline property, NOT on the swap outcome, exactly as the
2WikiMultiHop coverage stratification does. If the silver-minus-OpenIE swap is
positive and significant on the BM25-miss stratum, the C_K=0.75 point yields a
detectable inflation once the headroom confound is removed, separating
"small C -> small inflation" from "no headroom".

Deterministic; bootstrap seed 20260615; exact two-sided sign test. ASCII only.
"""
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOT_SEED = 20260615
B = 10000
SYSTEMS = [("Audited", "proposed_v3", ""),
           ("PPR", "ppr", "ppr_"),
           ("Graph-centric", "proposed_v5_graphcentric",
            "proposed_v5_graphcentric_")]
SPLITS = ["sub200", "holdout200"]


def per_q(payload, system):
    return {str(r["qa_id"]): float(r["metrics"]["hit_at_k"])
            for r in payload["results"][system]["records"]}


def boot_ci(diffs):
    n = len(diffs)
    random.seed(BOOT_SEED)
    means = sorted(sum(diffs[random.randrange(n)] for _ in range(n)) / n
                   for _ in range(B))
    return means[int(0.025 * B)], means[int(0.975 * B)]


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def analyze(label, sys_key, prefix, split):
    d = ROOT / "stark" / "data" / split  # released per-question vectors
    ps = json.load(open(d / ("_kgswap_%skg_silver.json" % prefix),
                        encoding="utf-8"))
    po = json.load(open(d / ("_kgswap_%skg_openie.json" % prefix),
                        encoding="utf-8"))
    silver = per_q(ps, sys_key)
    openie = per_q(po, sys_key)
    bm25 = per_q(ps, "bm25")          # bit-identical across conditions
    bm25o = per_q(po, "bm25")
    for q in bm25:
        assert bm25[q] == bm25o.get(q), "bm25 differs across conditions"
    qs = sorted(q for q in silver if q in openie and q in bm25)

    def swap_on(subset):
        diffs = [silver[q] - openie[q] for q in subset]
        if not diffs:
            return None
        mean = sum(diffs) / len(diffs)
        lo, hi = boot_ci(diffs)
        pos, neg, p = sign_test(diffs)
        return {"n": len(diffs), "mean": round(mean, 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "pos": pos, "neg": neg, "p": p}

    full = swap_on(qs)
    miss = [q for q in qs if bm25[q] == 0.0]            # headroom stratum
    hitq = [q for q in qs if bm25[q] > 0.0]
    return {"label": label, "split": split,
            "bm25_miss_n": len(miss), "bm25_hit_n": len(hitq),
            "full": full, "headroom": swap_on(miss),
            "bm25_already_hit": swap_on(hitq)}


def main():
    rows = []
    print("HEADROOM-CONTROLLED STaRK SWAP (BM25-miss stratum)\n")
    for label, sys_key, prefix in SYSTEMS:
        for split in SPLITS:
            r = analyze(label, sys_key, prefix, split)
            rows.append(r)
            h = r["headroom"]
            f = r["full"]
            print("%-13s %-10s | BM25-miss n=%d | FULL swap=%+.3f p=%.2e | "
                  "HEADROOM swap=%+.3f CI[%+.3f,%+.3f] +%d/-%d p=%.2e"
                  % (label, split, r["bm25_miss_n"], f["mean"], f["p"],
                     h["mean"], h["ci"][0], h["ci"][1], h["pos"], h["neg"],
                     h["p"]))
    out = ROOT / "stark" / "results" / "stark_headroom_swap.json"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, indent=2)
    print("\nwrote %s" % out)
    # gate summary
    print("\nHeadroom-stratum gate (positive AND p<0.05):")
    for r in rows:
        h = r["headroom"]
        sig = h["p"] < 0.05 and h["mean"] > 0
        print("  %-13s %-10s: %s (swap=%+.3f p=%.2e n=%d)"
              % (r["label"], r["split"], "MET" if sig else "not met",
                 h["mean"], h["p"], h["n"]))


if __name__ == "__main__":
    main()
