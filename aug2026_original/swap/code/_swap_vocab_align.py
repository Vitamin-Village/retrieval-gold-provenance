# -*- coding: utf-8 -*-
"""
Entity-vocabulary alignment statistic for the 2WikiMultiHop KG-swap. For each
question we measure the fraction of its gold passage titles that appear in the
KG's node inventory, under the gold-aligned silver KG and under the independent
OpenIE KG. A gold title counts as covered when it exact-matches a node id, a
normalized node name, or a normalized alias. The statistic turns "the aligned
entity vocabulary carries the gain" into a measured quantity: the silver KG's
node inventory is the gold's own entity normalization, so its gold-title
coverage should exceed the OpenIE KG's despite the OpenIE KG being larger.
Post-processing of released artifacts only. ASCII-only.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def norm(s):
    return " ".join(s.lower().split())


def vocab(kg_dir):
    nodes = json.load(open(kg_dir / "proposed_kg_nodes.json", encoding="utf-8"))
    v = set()
    for n in nodes:
        nid = n.get("node_id", "")
        if nid.startswith("n::"):
            v.add(norm(nid[3:]))
        if n.get("name"):
            v.add(norm(n["name"]))
        for a in n.get("aliases", []) or []:
            v.add(norm(a))
    return v, len(nodes)


def gold_sets(eval_json):
    p = json.load(open(eval_json, encoding="utf-8"))
    recs = p["results"]["proposed_v3"]["records"]
    return {r["qa_id"]: [norm(g) for g in r["gold_paper_ids"]] for r in recs}


def run(name, d):
    d = Path(ROOT / d)
    golds = gold_sets(d / "_kgswap_kg_silver.json")
    out = {"name": name, "n": len(golds)}
    for cond in ("kg_silver", "kg_openie"):
        v, n_nodes = vocab(d / cond)
        per_q = []
        for q, gs in golds.items():
            if not gs:
                continue
            per_q.append(sum(1 for g in gs if g in v) / len(gs))
        cov = sum(per_q) / len(per_q)
        full = sum(1 for c in per_q if c == 1.0) / len(per_q)
        out[cond] = {"n_nodes": n_nodes,
                     "mean_gold_title_coverage": round(cov, 4),
                     "frac_questions_fully_covered": round(full, 4)}
        print("%-18s %-9s nodes=%5d | mean gold-title coverage=%.3f | "
              "fully covered questions=%.3f"
              % (name, cond, n_nodes, cov, full), flush=True)
    return out


if __name__ == "__main__":
    rows = [run("2Wiki-sub200", "swap/data/sub200"),
            run("2Wiki-holdout200", "swap/data/holdout200")]
    output_path = ROOT / "swap" / "results" / "swap_vocab_align.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, indent=2)
    print("wrote swap_vocab_align.json", flush=True)
