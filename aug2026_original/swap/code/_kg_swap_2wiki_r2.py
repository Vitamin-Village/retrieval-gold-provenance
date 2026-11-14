# -*- coding: utf-8 -*-
"""Second-retriever replication of the 2WikiMultiHop KG swap (E1, 2026-06-11).

Same protocol as _kg_swap_2wiki.py -- same questions, same native gold, same
BM25 baseline; the ONLY change across conditions is the KG (gold-aligned
silver vs independent OpenIE) -- but the graph system is a SECOND retriever,
selected by --system (default proposed_v5_graphcentric, whose ranking is
carried by a personalized-PageRank channel rather than the schema-centric
gated fusion of proposed_v3).

Outputs per split: _kgswap_<system>_<kg>.json, plus paired stats
(mean paired hit@5 diff silver-openie, bootstrap 95% CI, exact two-sided
sign test over discordant questions) written to
paper/results/kg_swap_2wiki_<system>.json.

ASCII-only. Deterministic (RAG_V3_DENSE_BACKEND=deterministic, seeded boot).

NOTE (release audit 2026-07-19): this is a LIVE driver -- it re-runs
retrieval via paper.eval.eval_independent and needs the source passage
text at the development-repo path paper/external/... , which the released
archives withhold (see DATA_LICENSES.md and swap/data/
PASSAGE_TEXT_REMOVED.md). Its frozen outputs ship as per-question
vectors (_kgswap_*.json) under the released data directories, and every
downstream statistic recomputes from those without running this driver.
"""
import os
os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")
import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402

random.seed(20260611)
B = 10000


def run_kg(d, kg_sub, system):
    d = Path(ROOT / d)
    out = d / ("_kgswap_%s_%s.json" % (system, kg_sub))
    return ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl",
                  manifest_path=d / "manifest.json", kg_dir=d / kg_sub,
                  systems=("bm25", system), out_path=out, top_k=5)


def per_q_hits(payload, system):
    # NOTE: hit_at_k lives under record["metrics"] (the flat read in the
    # original _kg_swap_2wiki.per_q_hits silently returned 0.0 everywhere).
    out = {}
    for r in payload["results"][system]["records"]:
        out[str(r["qa_id"])] = float(r["metrics"]["hit_at_k"])
    return out


def overall_hit(payload, system):
    return float(payload["results"][system]["summary"]["overall"]["hit_at_k"])


def boot_ci(diffs):
    n = len(diffs)
    means = sorted(
        sum(diffs[random.randrange(n)] for _ in range(n)) / n for _ in range(B)
    )
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="proposed_v5_graphcentric")
    args = ap.parse_args(argv)
    system = args.system

    rows = []
    for name, d in [("2Wiki-sub200", "paper/external/2wiki/sub200"),
                    ("2Wiki-holdout200", "paper/external/2wiki/holdout200")]:
        ps = run_kg(d, "kg_silver", system)
        po = run_kg(d, "kg_openie", system)
        g_s, l_s = overall_hit(ps, system), overall_hit(ps, "bm25")
        g_o, l_o = overall_hit(po, system), overall_hit(po, "bm25")
        hs, ho = per_q_hits(ps, system), per_q_hits(po, system)
        qs = [q for q in hs if q in ho]
        if not qs:
            print("WARNING: no paired questions for %s, skipping" % name,
                  flush=True)
            continue
        diffs = [hs[q] - ho[q] for q in qs]
        mean = sum(diffs) / len(diffs)
        # Re-seed immediately before the bootstrap so the CI is reproducible
        # regardless of any random consumption inside ev.run (audit F7).
        random.seed(20260611)
        lo, hi = boot_ci(diffs)
        pos, neg, p = sign_test(diffs)
        print("%-18s %s | SILVER graph=%.3f bm25=%.3f D=%+.3f | OPENIE graph=%.3f "
              "bm25=%.3f D=%+.3f | SWAP=%+.3f | paired mean=%+.3f CI[%+.3f,%+.3f] "
              "discordant +%d/-%d p=%.2e"
              % (name, system, g_s, l_s, g_s - l_s, g_o, l_o, g_o - l_o,
                 (g_s - l_s) - (g_o - l_o), mean, lo, hi, pos, neg, p),
              flush=True)
        rows.append({
            "name": name, "system": system, "n": len(qs),
            "silver": {"graph": round(g_s, 4), "bm25": round(l_s, 4),
                       "delta": round(g_s - l_s, 4)},
            "openie": {"graph": round(g_o, 4), "bm25": round(l_o, 4),
                       "delta": round(g_o - l_o, 4)},
            "swap_delta": round((g_s - l_s) - (g_o - l_o), 4),
            "mean_paired_diff": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "discordant_pos": pos, "discordant_neg": neg,
            "sign_test_p": p,
        })
    outp = ROOT / "paper" / "results" / ("kg_swap_2wiki_%s.json" % system)
    json.dump(rows, open(outp, "w", encoding="utf-8"), indent=2)
    print("wrote", outp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
