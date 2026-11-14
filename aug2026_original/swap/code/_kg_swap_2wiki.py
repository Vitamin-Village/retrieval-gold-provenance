# -*- coding: utf-8 -*-
"""
Open-benchmark provenance effect, within-2WikiMultiHop KG swap.
Same questions, same native gold; the ONLY change is the KG the graph retriever
ranks over:
  - kg_silver : derived from the Wikidata structure the 2Wiki gold itself comes
                from  -> the CIRCULAR condition (graph uses the gold's source KG).
  - kg_openie : an OpenIE KG built independently from passage text
                -> the INDEPENDENT condition.
Delta(KG) = hit@5(graph|KG) - hit@5(bm25). The provenance effect is
swap = Delta(silver) - Delta(openie): if positive, ranking over the KG the gold
was derived from inflates the measured graph advantage on a public benchmark.
We also report a matched-reachable subset (questions where BOTH KGs place at
least one gold passage in the graph top-5) to address the coverage difference.

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402


def _records(payload, system):
    return payload["results"][system]["records"]


def _overall_hit(payload, system):
    return float(payload["results"][system]["summary"].get("overall", {}).get("hit_at_k", 0.0))


def run_kg(d, kg_sub):
    d = Path(ROOT / d)
    return ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl",
                  manifest_path=d / "manifest.json", kg_dir=d / kg_sub,
                  systems=("bm25", "proposed_v3"), out_path=d / ("_kgswap_" + kg_sub + ".json"),
                  top_k=5)


def per_q_hits(payload, system):
    out = {}
    for r in _records(payload, system):
        qid = str(r.get("qa_id") or r.get("id") or r.get("question"))
        # hit_at_k lives under record["metrics"]; a top-level r.get("hit_at_k")
        # always read 0.0, which emptied the matched-reachable subset in the
        # originally shipped kg_swap_2wiki.json (n=0). Fixed 2026-07-19; the
        # released results JSON was regenerated from the frozen per-question
        # vectors with this corrected read (headline values unchanged).
        out[qid] = float((r.get("metrics") or {}).get("hit_at_k", r.get("hit_at_k", 0.0)))
    return out


def main():
    rows = []
    for name, d in [("2Wiki-sub200", "paper/external/2wiki/sub200"),
                    ("2Wiki-holdout200", "paper/external/2wiki/holdout200")]:
        ps = run_kg(d, "kg_silver")
        po = run_kg(d, "kg_openie")
        g_s, l_s = _overall_hit(ps, "proposed_v3"), _overall_hit(ps, "bm25")
        g_o, l_o = _overall_hit(po, "proposed_v3"), _overall_hit(po, "bm25")
        d_s, d_o = g_s - l_s, g_o - l_o
        # matched-reachable subset: graph hits >0 under BOTH KGs
        hs, ho = per_q_hits(ps, "proposed_v3"), per_q_hits(po, "proposed_v3")
        common = [q for q in hs if q in ho and (hs[q] > 0 or ho[q] > 0)]
        ms = sum(hs[q] for q in common) / len(common) if common else 0.0
        mo = sum(ho[q] for q in common) / len(common) if common else 0.0
        print("%-18s n=%d | SILVER(circular) graph=%.3f bm25=%.3f Delta=%+.3f | "
              "OPENIE(indep) graph=%.3f bm25=%.3f Delta=%+.3f | SWAP=%+.3f | "
              "matched-reachable(n=%d) silver=%.3f openie=%.3f"
              % (name, ps["n_qa"], g_s, l_s, d_s, g_o, l_o, d_o, d_s - d_o,
                 len(common), ms, mo), flush=True)
        rows.append({"name": name, "n": ps["n_qa"],
                     "silver": {"graph": round(g_s, 4), "bm25": round(l_s, 4), "delta": round(d_s, 4)},
                     "openie": {"graph": round(g_o, 4), "bm25": round(l_o, 4), "delta": round(d_o, 4)},
                     "swap_delta": round(d_s - d_o, 4),
                     "matched_reachable": {"n": len(common), "silver_graph": round(ms, 4),
                                           "openie_graph": round(mo, 4)}})
    outp = ROOT / "paper" / "results" / "kg_swap_2wiki.json"
    json.dump(rows, open(outp, "w", encoding="utf-8"), indent=2)
    print("wrote", outp, flush=True)


if __name__ == "__main__":
    main()
