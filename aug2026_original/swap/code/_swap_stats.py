# -*- coding: utf-8 -*-
"""
Paired statistics for the 2WikiMultiHop KG-swap. The lexical baseline is
identical across the two conditions, so the swap effect is the paired
difference in the graph retriever's per-question hit@5 between the gold-aligned
silver KG and the independent OpenIE KG. We report the mean paired difference, a
bootstrap 95% CI, and a two-sided sign test on discordant questions.
ASCII-only.
"""
import json
import random
from pathlib import Path

random.seed(20260608)
ROOT = Path(__file__).resolve().parents[2]


def graph_hits(eval_json):
    p = json.load(open(eval_json, encoding="utf-8"))
    recs = p["results"]["proposed_v3"]["records"]
    out = {}
    for r in recs:
        out[r["qa_id"]] = float(r["metrics"]["hit_at_k"])
    return out


def boot_ci(diffs, B=10000):
    n = len(diffs)
    means = []
    for _ in range(B):
        s = sum(diffs[random.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B)]


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    # two-sided exact binomial p at q=0.5
    from math import comb
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def run(name, d):
    d = Path(ROOT / d)
    hs = graph_hits(d / "_kgswap_kg_silver.json")
    ho = graph_hits(d / "_kgswap_kg_openie.json")
    qs = [q for q in hs if q in ho]
    diffs = [hs[q] - ho[q] for q in qs]
    mean = sum(diffs) / len(diffs)
    lo, hi = boot_ci(diffs)
    pos, neg, p = sign_test(diffs)
    print("%-18s n=%d | mean paired graph-hit diff (silver-openie) = %+.3f  "
          "95%% CI [%+.3f, %+.3f] | discordant +%d/-%d  sign-test p=%.2e"
          % (name, len(qs), mean, lo, hi, pos, neg, p), flush=True)
    return {"name": name, "n": len(qs), "mean_paired_diff": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "discordant_pos": pos,
            "discordant_neg": neg, "sign_test_p": p}


if __name__ == "__main__":
    rows = [run("2Wiki-sub200", "swap/data/sub200"),
            run("2Wiki-holdout200", "swap/data/holdout200")]
    output_path = ROOT / "swap" / "results" / "swap_stats.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, indent=2)
    print("wrote swap_stats.json", flush=True)
