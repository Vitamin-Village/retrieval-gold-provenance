# -*- coding: utf-8 -*-
"""MetaQA realistic-structure condition (CPU only, no LLM).

The frozen MetaQA swap uses a circular arm induced from the evaluated questions' own
2-hop paths (kg_silver). A practitioner would instead load the FULL MetaQA KB. This
script builds `kg_fullkb`: every KB triple whose movie is a corpus passage, with the
same node/edge schema as kg_silver (movie node -> own passage; entity node -> the
corpus movies it connects to), i.e. the structure the gold was derived from, but not
induced around the evaluated questions. Its passage index therefore covers every
corpus movie (gold fraction ~0.31, like the OpenIE KG) rather than only the gold.

Prediction under the gold-concentration mechanism: little or no inflation relative to
OpenIE. Prediction under a "same-source structure inflates" account: inflation persists.

Outputs: external/metaqa/<split>/kg_fullkb/, external/metaqa/<split>/_kgswap_<system>_kg_fullkb.json,
         results/kg_fullkb_metaqa.json, analysis/fullkb_metaqa_summary.md
"""
import os
os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")
import json
import random
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402
from paper.eval._kg_swap_metaqa import per_q_hits, overall_hit, boot_ci, sign_test  # noqa: E402

SEED = 20260920
SYSTEMS = ("proposed_v3", "ppr", "proposed_v5_graphcentric")
_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub(" ", s.strip()).lower()


def build_fullkb(split_dir):
    passages = [json.loads(l) for l in open(split_dir / "passages.jsonl", encoding="utf-8")]
    pid_by_title = {norm(p["title"]): p["passage_id"] for p in passages}
    nodes, edges = OrderedDict(), OrderedDict()
    ent_links = defaultdict(set)

    def touch(surface, pid=None):
        nid = "n::" + norm(surface)
        if nid not in nodes:
            nodes[nid] = {"node_id": nid, "name": surface, "type": "entity", "aliases": [surface], "source_docs": []}
        if pid and pid not in nodes[nid]["source_docs"]:
            nodes[nid]["source_docs"].append(pid)
        return nid

    def add_edge(s, r, o):
        eid = f"{s}|{r}|{o}"
        if eid not in edges:
            edges[eid] = {"edge_id": eid, "source": s, "target": o, "source_id": s, "target_id": o, "relation": r,
                          "relation_type": r, "source_doc": "", "reliability_score": 1.0}

    n_triples = 0
    for line in open(ROOT / "paper" / "external" / "metaqa" / "raw" / "kb.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) != 3:
            continue
        movie, rel, obj = (x.strip() for x in parts)
        pid = pid_by_title.get(norm(movie))
        if pid is None:
            continue
        m = touch(movie, pid)
        o = touch(obj)
        add_edge(m, rel, o)
        add_edge(o, rel + "_inv", m)
        ent_links[o].add(pid)
        n_triples += 1
    for nid, pids in ent_links.items():
        if not nodes[nid]["source_docs"]:
            nodes[nid]["source_docs"] = sorted(pids)
    d = split_dir / "kg_fullkb"
    d.mkdir(exist_ok=True)
    json.dump(list(nodes.values()), open(d / "proposed_kg_nodes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(list(edges.values()), open(d / "proposed_kg_edges.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return len(nodes), len(edges), n_triples


def main():
    rows = []
    for split in ("sub200", "holdout200"):
        d = ROOT / "paper" / "external" / "metaqa" / split
        qa = [json.loads(l) for l in open(d / "qa.jsonl", encoding="utf-8")]
        gold = set()
        for q in qa:
            g = q["gold_paper_ids"]
            gold.update(g if isinstance(g, list) else json.loads(g.replace("'", '"')))
        n_nodes, n_edges, n_tr = build_fullkb(d)
        nodes = json.load(open(d / "kg_fullkb" / "proposed_kg_nodes.json", encoding="utf-8"))
        linked = set()
        for n in nodes:
            linked.update(n["source_docs"])
        st = {"nodes": n_nodes, "edges": n_edges, "kb_triples_in_corpus": n_tr, "linked": len(linked), "gold_frac": round(len(linked & gold) / max(1, len(linked)), 3)}
        print(split, st, flush=True)
        for system in SYSTEMS:
            pre = "" if system == "proposed_v3" else system + "_"
            ref_o = json.load(open(d / f"_kgswap_{pre}kg_openie.json", encoding="utf-8"))
            ref_s = json.load(open(d / f"_kgswap_{pre}kg_silver.json", encoding="utf-8"))
            ho = per_q_hits(ref_o, system)
            payload = ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl", manifest_path=d / "manifest.json",
                             kg_dir=d / "kg_fullkb", systems=("bm25", system), out_path=d / f"_kgswap_{system}_kg_fullkb.json", top_k=5)
            hv = per_q_hits(payload, system)
            qs = sorted(q for q in hv if q in ho)
            dv = [hv[q] - ho[q] for q in qs]
            random.seed(SEED)
            lo, hi = boot_ci(dv)
            pos, neg, p = sign_test(dv)
            row = {"split": split, "system": system, **st, "hit5_fullkb": overall_hit(payload, system), "bm25": overall_hit(payload, "bm25"),
                   "hit5_silver": overall_hit(ref_s, system), "hit5_openie": overall_hit(ref_o, system),
                   "diff_vs_openie": sum(dv) / len(dv), "ci95": [lo, hi], "pos": pos, "neg": neg, "p": p}
            rows.append(row)
            print(f"{split} {system:24s} fullkb={row['hit5_fullkb']:.3f} silver={row['hit5_silver']:.3f} openie={row['hit5_openie']:.3f} bm25={row['bm25']:.3f} "
                  f"diff_vs_openie={row['diff_vs_openie']:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.1e}", flush=True)
    json.dump(rows, open(ROOT / "paper" / "results" / "kg_fullkb_metaqa.json", "w", encoding="utf-8"), indent=1)
    L = ["# MetaQA full-KB (realistic structure) condition", "",
         "kg_fullkb = every MetaQA KB triple whose movie is a corpus passage, node->passage index over all corpus movies (not induced from the evaluated questions). Paired hit@5 vs the frozen OpenIE condition.", "",
         "| Split | System | Linked passages | Gold fraction | hit@5 full-KB | hit@5 induced (silver) | hit@5 OpenIE | BM25 | Δ full-KB vs OpenIE [95% CI] | +/- | p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['split']} | {r['system']} | {r['linked']} | {r['gold_frac']:.2f} | {r['hit5_fullkb']:.3f} | {r['hit5_silver']:.3f} | {r['hit5_openie']:.3f} | {r['bm25']:.3f} | "
                 f"{r['diff_vs_openie']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | +{r['pos']}/-{r['neg']} | {r['p']:.1e} |")
    (ROOT / "paper" / "analysis" / "fullkb_metaqa_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
