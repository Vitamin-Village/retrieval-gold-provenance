# -*- coding: utf-8 -*-
"""
Win-pattern decomposition of the 2WikiMultiHop KG-swap. For each question we
cross the graph retriever's per-question hit@5 under the gold-aligned silver KG
and under the independent OpenIE KG with the fixed BM25 baseline's hit@5, and
count the joint patterns. The decomposition answers whether the swap effect is
carried by questions the lexical baseline also misses, i.e. whether provenance
alignment surfaces gold that lexical retrieval cannot reach, or merely
re-ranks questions BM25 already solves. Post-processing of released eval
vectors only; no retriever is re-run. ASCII-only.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def hits(eval_json, system):
    p = json.load(open(eval_json, encoding="utf-8"))
    recs = p["results"][system]["records"]
    return {r["qa_id"]: float(r["metrics"]["hit_at_k"]) for r in recs}


def run(name, d):
    d = Path(ROOT / d)
    hs = hits(d / "_kgswap_kg_silver.json", "proposed_v3")
    ho = hits(d / "_kgswap_kg_openie.json", "proposed_v3")
    hb = hits(d / "_kgswap_kg_silver.json", "bm25")
    hb2 = hits(d / "_kgswap_kg_openie.json", "bm25")
    qs = [q for q in hs if q in ho and q in hb]
    # sanity: bm25 identical across conditions
    mismatch = sum(1 for q in qs if hb[q] != hb2.get(q, hb[q]))
    gain = [q for q in qs if hs[q] > ho[q]]          # silver-only wins
    loss = [q for q in qs if hs[q] < ho[q]]
    gain_bm25_miss = sum(1 for q in gain if hb[q] == 0.0)
    gain_bm25_hit = sum(1 for q in gain if hb[q] > 0.0)
    loss_bm25_miss = sum(1 for q in loss if hb[q] == 0.0)
    loss_bm25_hit = sum(1 for q in loss if hb[q] > 0.0)
    out = {
        "name": name, "n": len(qs), "bm25_mismatch_across_conditions": mismatch,
        "silver_gain_questions": len(gain),
        "silver_gain_bm25_miss": gain_bm25_miss,
        "silver_gain_bm25_hit": gain_bm25_hit,
        "silver_loss_questions": len(loss),
        "silver_loss_bm25_miss": loss_bm25_miss,
        "silver_loss_bm25_hit": loss_bm25_hit,
    }
    print("%-18s n=%d | silver-gain %d (bm25-miss %d / bm25-hit %d) | "
          "silver-loss %d (bm25-miss %d / bm25-hit %d) | bm25 mismatch %d"
          % (name, len(qs), len(gain), gain_bm25_miss, gain_bm25_hit,
             len(loss), loss_bm25_miss, loss_bm25_hit, mismatch), flush=True)
    return out


if __name__ == "__main__":
    rows = [run("2Wiki-sub200", "swap/data/sub200"),
            run("2Wiki-holdout200", "swap/data/holdout200")]
    output_path = ROOT / "swap" / "results" / "swap_winpattern.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, indent=2)
    print("wrote swap_winpattern.json", flush=True)
