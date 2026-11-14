# -*- coding: utf-8 -*-
"""Gold-concentration dose experiment on 2WikiMultiHop (CPU only, no LLM).

Tests the mechanism stated in Sec. 6: inflation is carried by how concentrated the
retrieval structure's node-to-passage index is on the evaluation gold. Two-sided:

  dilute the aligned (silver) KG   -- add links from silver nodes to random NON-gold
                                       passages until the gold fraction of linked passages
                                       is 0.75, 0.50, 0.31 (the OpenIE KG's level)
  concentrate the independent KG   -- keep the OpenIE KG's nodes/edges but restrict every
                                       node's source_docs to the split's gold passages

Prediction: the swap effect decays with dilution and appears under concentration.
Runs proposed_v3, ppr, proposed_v5_graphcentric on both splits under the deterministic
dense surrogate (as in the frozen runs). Outputs:
  external/2wiki/<split>/_concentration/<variant>/{proposed_kg_nodes,proposed_kg_edges}.json
  external/2wiki/<split>/_concentration/_kgswap_<system>_<variant>.json
  results/kg_concentration_2wiki.json, analysis/concentration_2wiki_summary.md
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
TARGET_GOLD_FRACS = (0.75, 0.50, 0.31)


def load_kg(d):
    return (json.load(open(d / "proposed_kg_nodes.json", encoding="utf-8")),
            json.load(open(d / "proposed_kg_edges.json", encoding="utf-8")))


def write_kg(d, nodes, edges, manifest):
    d.mkdir(parents=True, exist_ok=True)
    json.dump(nodes, open(d / "proposed_kg_nodes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(edges, open(d / "proposed_kg_edges.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(manifest, open(d / "variant_manifest.json", "w", encoding="utf-8"), indent=1)


def gold_stats(nodes, gold):
    linked = set()
    for n in nodes:
        linked.update(n.get("source_docs") or [])
    return {"linked": len(linked), "linked_gold": len(linked & gold), "gold_frac": round(len(linked & gold) / max(1, len(linked)), 3)}


def build_variants(split_dir, gold, all_pids):
    out = {}
    s_nodes, s_edges = load_kg(split_dir / "kg_silver")
    o_nodes, o_edges = load_kg(split_dir / "kg_openie")
    base = gold_stats(s_nodes, gold)
    nongold = sorted(set(all_pids) - gold)
    rng = random.Random(SEED)
    for g in TARGET_GOLD_FRACS:
        n_add = min(len(nongold), round(base["linked_gold"] * (1 - g) / g))
        picks = rng.sample(nongold, n_add)
        nodes = json.loads(json.dumps(s_nodes))
        idx = [i for i, n in enumerate(nodes) if n.get("source_docs")]
        for p in picks:
            nodes[rng.choice(idx)]["source_docs"].append(p)
        name = f"silver_dilute_g{int(g * 100):02d}"
        d = split_dir / "_concentration" / name
        st = gold_stats(nodes, gold)
        write_kg(d, nodes, s_edges, {"variant": name, "target_gold_frac": g, "n_nongold_added": n_add, "seed": SEED, **st})
        out[name] = st
    nodes = json.loads(json.dumps(o_nodes))
    for n in nodes:
        n["source_docs"] = [p for p in (n.get("source_docs") or []) if p in gold]
    name = "openie_goldonly"
    st = gold_stats(nodes, gold)
    write_kg(split_dir / "_concentration" / name, nodes, o_edges, {"variant": name, "seed": SEED, **st})
    out[name] = st
    return out, base, gold_stats(o_nodes, gold)


def main():
    rows, stats_all = [], {}
    for split in ("sub200", "holdout200"):
        d = ROOT / "paper" / "external" / "2wiki" / split
        qa = [json.loads(l) for l in open(d / "qa.jsonl", encoding="utf-8")]
        gold = set(g for q in qa for g in q["gold_paper_ids"])
        all_pids = [json.loads(l)["passage_id"] for l in open(d / "passages.jsonl", encoding="utf-8")]
        variants, silver_st, openie_st = build_variants(d, gold, all_pids)
        stats_all[split] = {"kg_silver": silver_st, "kg_openie": openie_st, **variants}
        print(split, json.dumps(stats_all[split]), flush=True)
        for system in SYSTEMS:
            pre = "" if system == "proposed_v3" else system + "_"
            ref_o = json.load(open(d / f"_kgswap_{pre}kg_openie.json", encoding="utf-8"))
            ref_s = json.load(open(d / f"_kgswap_{pre}kg_silver.json", encoding="utf-8"))
            ho, hs = per_q_hits(ref_o, system), per_q_hits(ref_s, system)
            for name in list(variants):
                out = d / "_concentration" / f"_kgswap_{system}_{name}.json"
                payload = ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl", manifest_path=d / "manifest.json",
                                 kg_dir=d / "_concentration" / name, systems=("bm25", system), out_path=out, top_k=5)
                hv = per_q_hits(payload, system)
                qs = sorted(q for q in hv if q in ho)
                dv = [hv[q] - ho[q] for q in qs]
                random.seed(SEED)
                lo, hi = boot_ci(dv)
                pos, neg, p = sign_test(dv)
                row = {"split": split, "system": system, "variant": name, **variants[name],
                       "hit5": overall_hit(payload, system), "bm25": overall_hit(payload, "bm25"),
                       "silver_hit5": overall_hit(ref_s, system), "openie_hit5": overall_hit(ref_o, system),
                       "diff_vs_openie": sum(dv) / len(dv), "ci95": [lo, hi], "pos": pos, "neg": neg, "p": p}
                rows.append(row)
                print(f"{split} {system:24s} {name:18s} gold_frac={variants[name]['gold_frac']:.2f} hit5={row['hit5']:.3f} "
                      f"(silver {row['silver_hit5']:.3f}, openie {row['openie_hit5']:.3f}) diff_vs_openie={row['diff_vs_openie']:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.1e}", flush=True)
    json.dump({"rows": rows, "stats": stats_all}, open(ROOT / "paper" / "results" / "kg_concentration_2wiki.json", "w", encoding="utf-8"), indent=1)
    L = ["# Gold-concentration dose experiment on 2WikiMultiHop", "",
         "Aligned (silver) KG diluted with links to random non-gold passages; independent (OpenIE) KG concentrated by restricting its passage links to gold. Paired hit@5 difference versus the frozen OpenIE condition, 10,000-resample bootstrap CI, exact sign test.", "",
         "| Split | System | Variant | Linked passages | Gold fraction | hit@5 | Silver | OpenIE | Δ vs OpenIE [95% CI] | +/- | p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['split']} | {r['system']} | {r['variant']} | {r['linked']} | {r['gold_frac']:.2f} | {r['hit5']:.3f} | {r['silver_hit5']:.3f} | {r['openie_hit5']:.3f} | "
                 f"{r['diff_vs_openie']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | +{r['pos']}/-{r['neg']} | {r['p']:.1e} |")
    (ROOT / "paper" / "analysis" / "concentration_2wiki_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
