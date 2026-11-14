# -*- coding: utf-8 -*-
"""
Dose-response for the 2WikiMultiHop KG swap: vary the FRACTION f of the
gold-aligned silver-KG edges available to the graph retriever (f=0 -> no
structure, approaches BM25; f=1 -> full gold-aligned KG) and measure graph
advantage over BM25. A monotone rise in Delta with f shows the effect scales
with the degree of provenance alignment, not a binary artifact of the maximal
case. Edges are subsampled deterministically; nodes are kept.
ASCII-only. Run from repo root with project env + RAG_V3_DENSE_BACKEND=deterministic.

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
import json
import random
import shutil
import sys
from pathlib import Path

random.seed(20260608)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402

BENCH = Path(ROOT / "paper/external/2wiki/sub200")
SILVER = BENCH / "kg_silver"
WORK = BENCH / "_dose_work"


def load_edges(p):
    obj = json.load(open(p, encoding="utf-8"))
    if isinstance(obj, list):
        return obj, ("list", None)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                return v, ("dict", k)
    raise ValueError("unrecognized edges json")


def write_edges(template_obj, kind, edges, out):
    if kind[0] == "list":
        json.dump(edges, open(out, "w", encoding="utf-8"))
    else:
        template_obj[kind[1]] = edges
        json.dump(template_obj, open(out, "w", encoding="utf-8"))


def hit(payload, system):
    return float(payload["results"][system]["summary"].get("overall", {}).get("hit_at_k", 0.0))


def main():
    edges_path = SILVER / "proposed_kg_edges.json"
    full_obj = json.load(open(edges_path, encoding="utf-8"))
    edges, kind = load_edges(edges_path)
    order = list(range(len(edges)))
    random.shuffle(order)
    rows = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        kdir = WORK / ("f%02d" % int(f * 100))
        if kdir.exists():
            shutil.rmtree(kdir)
        kdir.mkdir(parents=True)
        shutil.copy(SILVER / "proposed_kg_nodes.json", kdir / "proposed_kg_nodes.json")
        keep = [edges[i] for i in order[:int(round(f * len(edges)))]]
        write_edges(json.loads(json.dumps(full_obj)), kind, keep, kdir / "proposed_kg_edges.json")
        payload = ev.run(qa_path=BENCH / "qa.jsonl", passages_path=BENCH / "passages.jsonl",
                         manifest_path=BENCH / "manifest.json", kg_dir=kdir,
                         systems=("bm25", "proposed_v3"), out_path=WORK / ("eval_f%02d.json" % int(f * 100)),
                         top_k=5)
        g, l = hit(payload, "proposed_v3"), hit(payload, "bm25")
        rows.append({"f": f, "n_edges": len(keep), "graph": round(g, 4), "bm25": round(l, 4),
                     "delta": round(g - l, 4)})
        print("f=%.2f edges=%6d | graph=%.3f bm25=%.3f Delta=%+.3f" % (f, len(keep), g, l, g - l), flush=True)
    json.dump(rows, open(ROOT / "paper" / "results" / "swap_dose_response.json", "w", encoding="utf-8"), indent=2)
    print("wrote swap_dose_response.json", flush=True)


if __name__ == "__main__":
    main()
