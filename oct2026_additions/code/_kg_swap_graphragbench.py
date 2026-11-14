# -*- coding: utf-8 -*-
"""GraphRAG-Bench (medical) third-swap evaluation + statistics. Mirrors
_kg_swap_metaqa.py; see _graphragbench_swap_design.md.

Conditions per split (same questions, gold chunks, corpus, BM25, retriever):
  - kg_silver : OpenIE over the benchmark's evidence_relations, keyed by gold chunk
                (circular: structure-derived inventory + gold-derived mapping)
  - kg_openie : OpenIE over the 1,584-chunk corpus, no question/evidence access

Outputs:
  paper/results/kg_swap_graphragbench[_<system>].json
  paper/results/swap_coverage_graphragbench.json
  paper/analysis/graphragbench_swap_summary.md
"""
import os
os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402
from paper.eval._kg_swap_metaqa import (  # noqa: E402
    out_name, per_q_hits, per_q_ids, overall_hit, boot_ci, sign_test, SYSTEMS)

BOOT_SEED = 20260918
BASE = "paper/external/graphragbench/"


def run_kg(d, kg_sub, system):
    d = Path(ROOT / d)
    return ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl",
                  manifest_path=d / "manifest.json", kg_dir=d / kg_sub,
                  systems=("bm25", system), out_path=d / out_name(system, kg_sub), top_k=5)


def coverage(name, d):
    """Gold-chunk mapping coverage: fraction of a question's gold chunks that at least
    one KG node maps to via source_docs."""
    d = Path(ROOT / d)
    qas = [json.loads(l) for l in open(d / "qa.jsonl", encoding="utf-8")]
    out = {"name": name, "n": len(qas)}
    for cond in ("kg_silver", "kg_openie"):
        nodes = json.load(open(d / cond / "proposed_kg_nodes.json", encoding="utf-8"))
        mapped = set()
        for n in nodes:
            mapped.update(n.get("source_docs", []) or [])
        per_q = [sum(1 for g in q["gold_paper_ids"] if g in mapped) / len(q["gold_paper_ids"]) for q in qas]
        cov = sum(per_q) / len(per_q)
        full = sum(1 for c in per_q if c == 1.0) / len(per_q)
        out[cond] = {"n_nodes": len(nodes), "n_mapped_passages": len(mapped),
                     "mean_gold_mapping_coverage": round(cov, 4), "frac_questions_fully_covered": round(full, 4)}
        print("%-22s %-9s nodes=%5d mapped_passages=%4d | mean gold-mapping coverage=%.3f | fully covered=%.3f"
              % (name, cond, len(nodes), len(mapped), cov, full), flush=True)
    return out


def swap_for_system(system, splits):
    rows = []
    for name, d in splits:
        ps = run_kg(d, "kg_silver", system)
        po = run_kg(d, "kg_openie", system)
        ids_s, ids_o = per_q_ids(ps, "bm25"), per_q_ids(po, "bm25")
        assert set(ids_s) == set(ids_o), "bm25 question sets differ (%s)" % name
        for q in ids_s:
            assert ids_s[q] == ids_o[q], "bm25 retrieval differs across KG conditions: %s %s" % (name, q)
        g_s, l_s = overall_hit(ps, system), overall_hit(ps, "bm25")
        g_o, l_o = overall_hit(po, system), overall_hit(po, "bm25")
        hs, ho = per_q_hits(ps, system), per_q_hits(po, system)
        qs = sorted(q for q in hs if q in ho)
        diffs = [hs[q] - ho[q] for q in qs]
        mean = sum(diffs) / len(diffs)
        random.seed(BOOT_SEED)
        lo, hi = boot_ci(diffs)
        pos, neg, p = sign_test(diffs)
        print("%-22s %s | SILVER graph=%.3f bm25=%.3f D=%+.3f | OPENIE graph=%.3f bm25=%.3f D=%+.3f | SWAP=%+.3f | "
              "paired mean=%+.3f CI[%+.3f,%+.3f] discordant +%d/-%d p=%.2e"
              % (name, system, g_s, l_s, g_s - l_s, g_o, l_o, g_o - l_o, (g_s - l_s) - (g_o - l_o), mean, lo, hi, pos, neg, p), flush=True)
        rows.append({"name": name, "system": system, "n": len(qs),
                     "silver": {"graph": round(g_s, 4), "bm25": round(l_s, 4), "delta": round(g_s - l_s, 4)},
                     "openie": {"graph": round(g_o, 4), "bm25": round(l_o, 4), "delta": round(g_o - l_o, 4)},
                     "swap_delta": round((g_s - l_s) - (g_o - l_o), 4), "mean_paired_diff": round(mean, 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "discordant_pos": pos, "discordant_neg": neg, "sign_test_p": p,
                     "bm25_identical_across_conditions": True})
    suffix = "" if system == "proposed_v3" else "_" + system
    outp = ROOT / "paper" / "results" / ("kg_swap_graphragbench%s.json" % suffix)
    json.dump(rows, open(outp, "w", encoding="utf-8"), indent=2)
    print("wrote %s" % outp, flush=True)
    return rows


def write_summary(all_rows, cov_rows):
    L = ["# GraphRAG-Bench (medical) KG-Swap Summary (fourth swap, non-maximal point)", "",
         "Generated %s. Protocol: paper/eval/_graphragbench_swap_design.md. Census estimate for the subgraph axis: "
         "Chat_KG = 0.75. kg_silver = OpenIE over the released evidence_relations keyed by gold chunk; kg_openie = OpenIE "
         "over the 1,584-chunk corpus without question access. Retriever configurations frozen as released; top_k=5; "
         "RAG_V3_DENSE_BACKEND=deterministic; bootstrap B=10,000, seed %d." % (datetime.now(timezone.utc).strftime("%Y-%m-%d"), BOOT_SEED), "",
         "## Swap results (hit@5; every gold chunk in top-5)", "",
         "| Split | System | Silver graph | Silver BM25 | Delta(silver) | OpenIE graph | OpenIE BM25 | Delta(openie) | Swap | Paired mean [95% CI] | Discordant +/- | Sign-test p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rows in all_rows:
        for r in rows:
            L.append("| %s | %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f [%+.3f, %+.3f] | +%d / -%d | %.2e |"
                     % (r["name"], r["system"], r["silver"]["graph"], r["silver"]["bm25"], r["silver"]["delta"], r["openie"]["graph"],
                        r["openie"]["bm25"], r["openie"]["delta"], r["swap_delta"], r["mean_paired_diff"], r["ci95"][0], r["ci95"][1],
                        r["discordant_pos"], r["discordant_neg"], r["sign_test_p"]))
    L += ["", "BM25 is bit-identical across KG conditions in every run (asserted per question).", "",
          "## Gold-chunk mapping coverage", "", "| Split | KG | Nodes | Mapped passages | Mean coverage | Fully covered |", "|---|---|---|---|---|---|"]
    for c in cov_rows:
        for cond in ("kg_silver", "kg_openie"):
            L.append("| %s | %s | %d | %d | %.3f | %.3f |" % (c["name"], cond, c[cond]["n_nodes"], c[cond]["n_mapped_passages"],
                                                            c[cond]["mean_gold_mapping_coverage"], c[cond]["frac_questions_fully_covered"]))
    L += ["", "## Pre-registered gate", ""]
    for rows in all_rows:
        for r in rows:
            sig, positive = r["sign_test_p"] < 0.05, r["mean_paired_diff"] > 0
            v = ("GATE MET: positive and significant (p=%.2e)." % r["sign_test_p"]) if positive and sig else \
                ("GATE NOT MET: paired mean not positive (%+.3f)." % r["mean_paired_diff"]) if not positive else \
                ("GATE NOT MET: positive but not significant (p=%.2e)." % r["sign_test_p"])
            L.append("- %s / %s: %s" % (r["name"], r["system"], v))
    outp = ROOT / "paper" / "analysis" / "graphragbench_swap_summary.md"
    outp.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote %s" % outp, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="sub200,holdout200")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    args = ap.parse_args(argv)
    splits = []
    for s in [x.strip() for x in args.splits.split(",") if x.strip()]:
        d = BASE + s
        for cond in ("kg_silver", "kg_openie"):
            if not (ROOT / d / cond / "proposed_kg_nodes.json").exists():
                print("SKIP %s: %s not built yet" % (s, cond), flush=True)
                d = None
                break
        if d:
            splits.append(("GraphRAG-Bench-" + s, d))
    if not splits:
        print("no splits ready; aborting", flush=True)
        return 1
    cov_rows = [coverage(n, d) for n, d in splits]
    json.dump(cov_rows, open(ROOT / "paper" / "results" / "swap_coverage_graphragbench.json", "w", encoding="utf-8"), indent=2)
    all_rows = [swap_for_system(sys_, splits) for sys_ in [x.strip() for x in args.systems.split(",") if x.strip()]]
    write_summary(all_rows, cov_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
