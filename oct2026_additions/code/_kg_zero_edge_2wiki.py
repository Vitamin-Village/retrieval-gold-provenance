# -*- coding: utf-8 -*-
"""Zero-edge control on 2WikiMultiHop for all three graph retrievers (CPU only, no LLM).

The edge-subsample dose (eval/_swap_dose_response.py) was run for the audited retriever on
the first split only. This control deletes EVERY edge of the aligned (silver) KG while keeping
its nodes and their gold-derived node-to-passage links, and runs proposed_v3 (audited), ppr,
and proposed_v5_graphcentric on both splits under the frozen protocol (deterministic dense
surrogate, top-5), paired against the frozen OpenIE condition and the full silver KG.

If the aligned KG's advantage survives with no edges, the carrier is the index (which nodes a
question names and which passages they link to), not graph connectivity or multi-hop traversal.

Outputs:
  external/2wiki/<split>/_zero_edge/kg_silver_noedges/{proposed_kg_nodes,proposed_kg_edges}.json
  external/2wiki/<split>/_zero_edge/_kgswap_<system>_silver_noedges.json
  results/kg_zero_edge_2wiki.json, analysis/zero_edge_2wiki_summary.md
"""
import os
os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")
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
    rows = []
    for split in ("sub200", "holdout200"):
        d = ROOT / "paper" / "external" / "2wiki" / split
        kdir = d / "_zero_edge" / "kg_silver_noedges"
        kdir.mkdir(parents=True, exist_ok=True)
        nodes = json.load(open(d / "kg_silver" / "proposed_kg_nodes.json", encoding="utf-8"))
        edges_obj = json.load(open(d / "kg_silver" / "proposed_kg_edges.json", encoding="utf-8"))
        empty = [] if isinstance(edges_obj, list) else {k: ([] if isinstance(v, list) else v) for k, v in edges_obj.items()}
        json.dump(nodes, open(kdir / "proposed_kg_nodes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(empty, open(kdir / "proposed_kg_edges.json", "w", encoding="utf-8"))
        n_edges = len(edges_obj) if isinstance(edges_obj, list) else sum(len(v) for v in edges_obj.values() if isinstance(v, list))
        for system in SYSTEMS:
            pre = "" if system == "proposed_v3" else system + "_"
            ref_o = json.load(open(d / f"_kgswap_{pre}kg_openie.json", encoding="utf-8"))
            ref_s = json.load(open(d / f"_kgswap_{pre}kg_silver.json", encoding="utf-8"))
            ho, hs = per_q_hits(ref_o, system), per_q_hits(ref_s, system)
            out = d / "_zero_edge" / f"_kgswap_{system}_silver_noedges.json"
            payload = ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl", manifest_path=d / "manifest.json",
                             kg_dir=kdir, systems=("bm25", system), out_path=out, top_k=5)
            hv = per_q_hits(payload, system)
            qs = sorted(q for q in hv if q in ho)
            dv = [hv[q] - ho[q] for q in qs]
            random.seed(SEED)
            lo, hi = boot_ci(dv)
            pos, neg, p = sign_test(dv)
            ds = [hs[q] - hv[q] for q in qs]
            row = {"split": split, "system": system, "n_edges_removed": n_edges, "n_nodes": len(nodes),
                   "hit5_noedges": overall_hit(payload, system), "bm25": overall_hit(payload, "bm25"),
                   "silver_hit5": overall_hit(ref_s, system), "openie_hit5": overall_hit(ref_o, system),
                   "diff_vs_openie": sum(dv) / len(dv), "ci95": [lo, hi], "pos": pos, "neg": neg, "p": p,
                   "silver_minus_noedges": sum(ds) / len(ds)}
            rows.append(row)
            print(f"{split} {system:24s} no-edges hit5={row['hit5_noedges']:.3f} (silver {row['silver_hit5']:.3f}, openie {row['openie_hit5']:.3f}, "
                  f"bm25 {row['bm25']:.3f}) diff_vs_openie={row['diff_vs_openie']:+.3f} [{lo:+.3f},{hi:+.3f}] +{pos}/-{neg} p={p:.1e}", flush=True)
    json.dump({"rows": rows}, open(ROOT / "paper" / "results" / "kg_zero_edge_2wiki.json", "w", encoding="utf-8"), indent=1)
    L = ["# Zero-edge control on 2WikiMultiHop", "",
         "Aligned (silver) KG with every edge deleted; nodes and their gold-derived passage links kept. Paired hit@5 difference versus the frozen OpenIE condition, 10,000-resample bootstrap CI, exact sign test.", "",
         "| Split | System | hit@5 no edges | Silver (full) | OpenIE | BM25 | Δ vs OpenIE [95% CI] | +/- | p | silver − no-edges |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['split']} | {r['system']} | {r['hit5_noedges']:.3f} | {r['silver_hit5']:.3f} | {r['openie_hit5']:.3f} | {r['bm25']:.3f} | "
                 f"{r['diff_vs_openie']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | +{r['pos']}/-{r['neg']} | {r['p']:.1e} | {r['silver_minus_noedges']:+.3f} |")
    (ROOT / "paper" / "analysis" / "zero_edge_2wiki_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
