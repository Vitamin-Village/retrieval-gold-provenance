# -*- coding: utf-8 -*-
"""Run the three graph retrievers on an extra KG variant directory for a benchmark and
report paired hit@5 against the frozen OpenIE (independent) and silver (aligned) runs.

    python paper/eval/_kg_swap_variant.py --bench 2wiki --variant kg_wikidata
Outputs: external/<bench>/<split>/_kgswap_<system>_<variant>.json,
         results/kg_swap_variant_<bench>_<variant>.json, analysis/variant_<bench>_<variant>_summary.md
"""
import os
os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402
from paper.eval._kg_swap_metaqa import per_q_hits, overall_hit, boot_ci, sign_test  # noqa: E402

SEED = 20260920
SYSTEMS = ("proposed_v3", "ppr", "proposed_v5_graphcentric")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--splits", default="sub200,holdout200")
    a = ap.parse_args()
    rows = []
    for split in a.splits.split(","):
        d = ROOT / "paper" / "external" / a.bench / split
        qa = [json.loads(l) for l in open(d / "qa.jsonl", encoding="utf-8")]
        gold = set()
        for q in qa:
            g = q["gold_paper_ids"]
            gold.update(g if isinstance(g, list) else json.loads(g.replace("'", '"')))
        nodes = json.load(open(d / a.variant / "proposed_kg_nodes.json", encoding="utf-8"))
        linked = set()
        for n in nodes:
            linked.update(n.get("source_docs") or [])
        st = {"nodes": len(nodes), "linked": len(linked), "gold_frac": round(len(linked & gold) / max(1, len(linked)), 3), "gold_covered": round(len(linked & gold) / len(gold), 3)}
        for system in SYSTEMS:
            pre = "" if system == "proposed_v3" else system + "_"
            ref_o = json.load(open(d / f"_kgswap_{pre}kg_openie.json", encoding="utf-8"))
            ref_s = json.load(open(d / f"_kgswap_{pre}kg_silver.json", encoding="utf-8"))
            ho = per_q_hits(ref_o, system)
            payload = ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl", manifest_path=d / "manifest.json",
                             kg_dir=d / a.variant, systems=("bm25", system), out_path=d / f"_kgswap_{system}_{a.variant}.json", top_k=5)
            hv = per_q_hits(payload, system)
            qs = sorted(q for q in hv if q in ho)
            dv = [hv[q] - ho[q] for q in qs]
            random.seed(SEED)
            lo, hi = boot_ci(dv)
            pos, neg, p = sign_test(dv)
            row = {"bench": a.bench, "split": split, "system": system, "variant": a.variant, **st, "hit5": overall_hit(payload, system),
                   "bm25": overall_hit(payload, "bm25"), "hit5_silver": overall_hit(ref_s, system), "hit5_openie": overall_hit(ref_o, system),
                   "diff_vs_openie": sum(dv) / len(dv), "ci95": [lo, hi], "pos": pos, "neg": neg, "p": p}
            rows.append(row)
            print(f"{split} {system:24s} {a.variant}: hit5={row['hit5']:.3f} silver={row['hit5_silver']:.3f} openie={row['hit5_openie']:.3f} bm25={row['bm25']:.3f} "
                  f"gold_frac={st['gold_frac']:.2f} diff_vs_openie={row['diff_vs_openie']:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.1e}", flush=True)
    json.dump(rows, open(ROOT / "paper" / "results" / f"kg_swap_variant_{a.bench}_{a.variant}.json", "w", encoding="utf-8"), indent=1)
    L = [f"# KG variant `{a.variant}` on {a.bench}", "", "| Split | System | Nodes | Linked | Gold frac | hit@5 variant | Silver | OpenIE | BM25 | Δ vs OpenIE [95% CI] | +/- | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['split']} | {r['system']} | {r['nodes']} | {r['linked']} | {r['gold_frac']:.2f} | {r['hit5']:.3f} | {r['hit5_silver']:.3f} | {r['hit5_openie']:.3f} | {r['bm25']:.3f} | {r['diff_vs_openie']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | +{r['pos']}/-{r['neg']} | {r['p']:.1e} |")
    (ROOT / "paper" / "analysis" / f"variant_{a.bench}_{a.variant}_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
