# -*- coding: utf-8 -*-
"""KG swap on the 2WikiMultiHop compositional split (comp200), retrieval stage.

Protocol: results/twowiki_comp200/PROTOCOL.md (fixed before any run on this split).

The existing drivers eval/_kg_swap_2wiki.py (hybrid = proposed_v3) and
eval/_kg_swap_2wiki_r2.py (--system ppr / proposed_v5_graphcentric) hard-code the split list
(sub200, holdout200) and overwrite results/kg_swap_2wiki*.json, so they are not run. This driver
imports their run_kg functions UNCHANGED and calls them for paper/external/2wiki/comp200 only, so
the per-question vectors are produced by exactly the same call:

    _kgswap_kg_{kg}.json                         _kg_swap_2wiki.run_kg(d, kg)            (bm25, proposed_v3)
    _kgswap_ppr_{kg}.json                        _kg_swap_2wiki_r2.run_kg(d, kg, "ppr")
    _kgswap_proposed_v5_graphcentric_{kg}.json   _kg_swap_2wiki_r2.run_kg(d, kg, "proposed_v5_graphcentric")

Statistics are computed by analysis/twowiki_comp200_summary.py from these vectors.

--env-check regenerates the sub200 aligned-KG vectors of the three retrievers with the same
eval_independent.run call into a separate output directory (never into external/2wiki/sub200) and
compares the per-question hit@5 and top-5 ids with the frozen files.

Run from the repo root (parent of paper/) with env new_env; RAG_V3_DENSE_BACKEND=deterministic is
set by the imported drivers (setdefault) and asserted here.
    python paper/eval/_kg_swap_2wiki_comp200.py --kgs kg_silver
    python paper/eval/_kg_swap_2wiki_comp200.py --kgs kg_openie
    python paper/eval/_kg_swap_2wiki_comp200.py --env-check <output_dir>
"""
import os

os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import _kg_swap_2wiki as v1  # noqa: E402  (hybrid driver)
from paper.eval import _kg_swap_2wiki_r2 as r2  # noqa: E402  (PPR / hybrid-PPR driver)
from paper.eval import eval_independent as ev  # noqa: E402

SPLIT_DIR = "paper/external/2wiki/comp200"
SYSTEMS = ("proposed_v3", "ppr", "proposed_v5_graphcentric")


def run_split(kg: str) -> None:
    d = ROOT / SPLIT_DIR
    assert (d / kg / "proposed_kg_nodes.json").exists(), f"missing {d / kg}"
    for system in SYSTEMS:
        t0 = time.time()
        if system == "proposed_v3":
            payload = v1.run_kg(SPLIT_DIR, kg)
            fname = "_kgswap_%s.json" % kg
        else:
            payload = r2.run_kg(SPLIT_DIR, kg, system)
            fname = "_kgswap_%s_%s.json" % (system, kg)
        g = r2.overall_hit(payload, system)
        b = r2.overall_hit(payload, "bm25")
        print("comp200 %-26s %-9s n=%d graph=%.3f bm25=%.3f delta=%+.3f  (%s, %.0fs)"
              % (system, kg, payload["n_qa"], g, b, g - b, fname, time.time() - t0), flush=True)


def env_check(scratch: Path) -> int:
    """Regenerate sub200 aligned vectors into a separate directory; compare with the frozen per-question vectors."""
    d = ROOT / "paper/external/2wiki/sub200"
    scratch.mkdir(parents=True, exist_ok=True)
    frozen = {"proposed_v3": d / "_kgswap_kg_silver.json",
              "ppr": d / "_kgswap_ppr_kg_silver.json",
              "proposed_v5_graphcentric": d / "_kgswap_proposed_v5_graphcentric_kg_silver.json"}
    bad = 0
    for system, fpath in frozen.items():
        out = scratch / ("envcheck_sub200_%s_kg_silver.json" % system)
        assert out.resolve().parent == scratch.resolve()
        t0 = time.time()
        new = ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl", manifest_path=d / "manifest.json",
                     kg_dir=d / "kg_silver", systems=("bm25", system), out_path=out, top_k=5)
        old = json.load(open(fpath, encoding="utf-8"))
        for s in ("bm25", system):
            a = {r["qa_id"]: (r["metrics"]["hit_at_k"], r["retrieved_paper_ids"]) for r in new["results"][s]["records"]}
            b = {r["qa_id"]: (r["metrics"]["hit_at_k"], r["retrieved_paper_ids"]) for r in old["results"][s]["records"]}
            same_hit = all(a[q][0] == b[q][0] for q in b) and set(a) == set(b)
            same_ids = all(a[q][1] == b[q][1] for q in b)
            print("env-check sub200 kg_silver %-26s %-26s hit@5 new=%.3f frozen=%.3f per-question hit identical=%s top-5 ids identical=%s (%.0fs)"
                  % (system, s, sum(v[0] for v in a.values()) / len(a), sum(v[0] for v in b.values()) / len(b),
                     same_hit, same_ids, time.time() - t0), flush=True)
            bad += not (same_hit and same_ids)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kgs", nargs="+", default=["kg_silver", "kg_openie"], choices=["kg_silver", "kg_openie"])
    ap.add_argument("--env-check", type=Path, default=None, help="output dir; run the sub200 reproduction check only")
    args = ap.parse_args(argv)
    assert os.environ["RAG_V3_DENSE_BACKEND"] == "deterministic", os.environ["RAG_V3_DENSE_BACKEND"]
    if args.env_check is not None:
        bad = env_check(args.env_check)
        print("ENV-CHECK", "PASS" if bad == 0 else "FAIL (%d mismatching vectors)" % bad, flush=True)
        return 1 if bad else 0
    for kg in args.kgs:
        run_split(kg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
