# -*- coding: utf-8 -*-
"""MetaQA second-swap evaluation + statistics (S4/S5 of the second causal
experiment; mirrors _kg_swap_2wiki.py / _kg_swap_2wiki_r2.py exactly).

Same questions, same native gold (the 2 bridge-movie docs); the ONLY change
across conditions is the KG the graph retriever ranks over:
  - kg_silver : KB subgraph induced by the questions' own 2-hop paths
                (the CIRCULAR condition; MetaQA is C_K = 1.0).
  - kg_openie : OpenIE KG built independently from the passage corpus
                (the INDEPENDENT condition; no KB / gold access).

Per system (proposed_v3, ppr, proposed_v5_graphcentric):
  paired per-question hit@5 diff silver-minus-openie, 10,000-resample
  bootstrap 95% CI (fixed seed), exact two-sided sign test on discordants.
BM25 must be bit-identical across conditions (asserted, not assumed).
Gold-title coverage of each KG via the _swap_vocab_align logic.

Outputs:
  paper/results/kg_swap_metaqa.json                      (proposed_v3)
  paper/results/kg_swap_metaqa_ppr.json
  paper/results/kg_swap_metaqa_proposed_v5_graphcentric.json
  paper/results/swap_vocab_align_metaqa.json
  paper/analysis/metaqa_swap_summary.md

ASCII-only. Deterministic (RAG_V3_DENSE_BACKEND=deterministic, seeded boot).

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
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402

BOOT_SEED = 20260613
B = 10000
SYSTEMS = ("proposed_v3", "ppr", "proposed_v5_graphcentric")
PROGRESS = ROOT / "paper" / "external" / "metaqa" / "_PROGRESS.log"


def log_progress(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write("%s %s\n" % (ts, msg))
    print("[progress] " + msg, flush=True)


def out_name(system, kg_sub):
    # Mirror the 2Wiki naming convention: proposed_v3 runs keep the original
    # unprefixed names (read by _swap_vocab_align), other systems are prefixed.
    if system == "proposed_v3":
        return "_kgswap_%s.json" % kg_sub
    return "_kgswap_%s_%s.json" % (system, kg_sub)


def run_kg(d, kg_sub, system):
    d = Path(ROOT / d)
    return ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl",
                  manifest_path=d / "manifest.json", kg_dir=d / kg_sub,
                  systems=("bm25", system), out_path=d / out_name(system, kg_sub),
                  top_k=5)


def per_q_hits(payload, system):
    return {str(r["qa_id"]): float(r["metrics"]["hit_at_k"])
            for r in payload["results"][system]["records"]}


def per_q_ids(payload, system):
    return {str(r["qa_id"]): list(r["retrieved_paper_ids"])
            for r in payload["results"][system]["records"]}


def overall_hit(payload, system):
    return float(payload["results"][system]["summary"]["overall"]["hit_at_k"])


def boot_ci(diffs):
    n = len(diffs)
    means = sorted(
        sum(diffs[random.randrange(n)] for _ in range(n)) / n for _ in range(B)
    )
    return means[int(0.025 * B)], means[int(0.975 * B)]


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Gold-title coverage (reuse of _swap_vocab_align logic over metaqa dirs)
# ---------------------------------------------------------------------------
def norm(s):
    return " ".join(s.lower().split())


def kg_vocab(kg_dir):
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


def coverage(name, d):
    d = Path(ROOT / d)
    payload = json.load(open(d / "_kgswap_kg_silver.json", encoding="utf-8"))
    recs = payload["results"]["proposed_v3"]["records"]
    golds = {r["qa_id"]: [norm(g) for g in r["gold_paper_ids"]] for r in recs}
    out = {"name": name, "n": len(golds)}
    for cond in ("kg_silver", "kg_openie"):
        v, n_nodes = kg_vocab(d / cond)
        per_q = [sum(1 for g in gs if g in v) / len(gs)
                 for gs in golds.values() if gs]
        cov = sum(per_q) / len(per_q)
        full = sum(1 for c in per_q if c == 1.0) / len(per_q)
        out[cond] = {"n_nodes": n_nodes,
                     "mean_gold_title_coverage": round(cov, 4),
                     "frac_questions_fully_covered": round(full, 4)}
        print("%-18s %-9s nodes=%5d | mean gold-title coverage=%.3f | "
              "fully covered questions=%.3f"
              % (name, cond, n_nodes, cov, full), flush=True)
    return out


# ---------------------------------------------------------------------------
# Per-system swap
# ---------------------------------------------------------------------------
def swap_for_system(system, splits):
    rows = []
    for name, d in splits:
        ps = run_kg(d, "kg_silver", system)
        po = run_kg(d, "kg_openie", system)

        # BM25 must be bit-identical across conditions (same corpus, no KG).
        ids_s, ids_o = per_q_ids(ps, "bm25"), per_q_ids(po, "bm25")
        assert set(ids_s) == set(ids_o), "bm25 question sets differ (%s)" % name
        for q in ids_s:
            assert ids_s[q] == ids_o[q], \
                "bm25 retrieval differs across KG conditions: %s %s" % (name, q)

        g_s, l_s = overall_hit(ps, system), overall_hit(ps, "bm25")
        g_o, l_o = overall_hit(po, system), overall_hit(po, "bm25")
        hs, ho = per_q_hits(ps, system), per_q_hits(po, system)
        qs = sorted(q for q in hs if q in ho)
        diffs = [hs[q] - ho[q] for q in qs]
        mean = sum(diffs) / len(diffs)
        random.seed(BOOT_SEED)  # re-seed right before the bootstrap
        lo, hi = boot_ci(diffs)
        pos, neg, p = sign_test(diffs)
        common = [q for q in qs if hs[q] > 0 or ho[q] > 0]
        ms = sum(hs[q] for q in common) / len(common) if common else 0.0
        mo = sum(ho[q] for q in common) / len(common) if common else 0.0
        print("%-18s %s | SILVER graph=%.3f bm25=%.3f D=%+.3f | OPENIE graph=%.3f "
              "bm25=%.3f D=%+.3f | SWAP=%+.3f | paired mean=%+.3f CI[%+.3f,%+.3f] "
              "discordant +%d/-%d p=%.2e"
              % (name, system, g_s, l_s, g_s - l_s, g_o, l_o, g_o - l_o,
                 (g_s - l_s) - (g_o - l_o), mean, lo, hi, pos, neg, p),
              flush=True)
        rows.append({
            "name": name, "system": system, "n": len(qs),
            "silver": {"graph": round(g_s, 4), "bm25": round(l_s, 4),
                       "delta": round(g_s - l_s, 4)},
            "openie": {"graph": round(g_o, 4), "bm25": round(l_o, 4),
                       "delta": round(g_o - l_o, 4)},
            "swap_delta": round((g_s - l_s) - (g_o - l_o), 4),
            "mean_paired_diff": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "discordant_pos": pos, "discordant_neg": neg,
            "sign_test_p": p,
            "matched_reachable": {"n": len(common),
                                  "silver_graph": round(ms, 4),
                                  "openie_graph": round(mo, 4)},
            "bm25_identical_across_conditions": True,
        })
    suffix = "" if system == "proposed_v3" else "_" + system
    outp = ROOT / "paper" / "results" / ("kg_swap_metaqa%s.json" % suffix)
    json.dump(rows, open(outp, "w", encoding="utf-8"), indent=2)
    print("wrote %s" % outp, flush=True)
    return rows


# ---------------------------------------------------------------------------
# Summary markdown (English table + pre-registered gate verdict)
# ---------------------------------------------------------------------------
def write_summary(all_rows, cov_rows):
    lines = []
    lines.append("# MetaQA KG-Swap Summary (Second Causal Experiment)")
    lines.append("")
    lines.append("Generated %s. Mirrors the 2WikiMultiHop swap protocol "
                 "(paper/eval/_metaqa_swap_design.md). MetaQA census C_K = 1.0: "
                 "every question is generated from the movie KB, so kg_silver "
                 "(the KB subgraph the gold derives from) is the circular "
                 "condition and kg_openie (llama3.1:8b OpenIE over the passage "
                 "corpus, no KB/gold access) is the independent condition. "
                 "Retriever configurations frozen as released; top_k=5; "
                 "RAG_V3_DENSE_BACKEND=deterministic; bootstrap B=10,000, "
                 "seed %d." % (datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                               BOOT_SEED))
    lines.append("")
    lines.append("## Swap results (hit@5; both gold bridge-movie docs in top-5)")
    lines.append("")
    lines.append("| Split | System | Silver graph | Silver BM25 | Delta(silver)"
                 " | OpenIE graph | OpenIE BM25 | Delta(openie) | Swap | Paired"
                 " mean [95% CI] | Discordant +/- | Sign-test p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rows in all_rows:
        for r in rows:
            lines.append(
                "| %s | %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f | %+.3f"
                " | %+.3f [%+.3f, %+.3f] | +%d / -%d | %.2e |"
                % (r["name"], r["system"], r["silver"]["graph"],
                   r["silver"]["bm25"], r["silver"]["delta"],
                   r["openie"]["graph"], r["openie"]["bm25"],
                   r["openie"]["delta"], r["swap_delta"],
                   r["mean_paired_diff"], r["ci95"][0], r["ci95"][1],
                   r["discordant_pos"], r["discordant_neg"], r["sign_test_p"]))
    lines.append("")
    lines.append("BM25 is bit-identical across KG conditions in every run "
                 "(asserted per question).")
    lines.append("")
    lines.append("## Gold-title coverage of each KG (vocab-align statistic)")
    lines.append("")
    lines.append("| Split | KG | Nodes | Mean gold-title coverage | Fully covered questions |")
    lines.append("|---|---|---|---|---|")
    for c in cov_rows:
        for cond in ("kg_silver", "kg_openie"):
            lines.append("| %s | %s | %d | %.3f | %.3f |"
                         % (c["name"], cond, c[cond]["n_nodes"],
                            c[cond]["mean_gold_title_coverage"],
                            c[cond]["frac_questions_fully_covered"]))
    lines.append("")
    lines.append("## Pre-registered interpretation gates (decided before results)")
    lines.append("")
    lines.append("Gate: silver-minus-openie inflation positive AND significant "
                 "on the paired sign test for the graph retrievers => second "
                 "causal point established. Null or negative => report as-is, "
                 "no tuning; inclusion decision passes to the author.")
    lines.append("")
    for rows in all_rows:
        for r in rows:
            sig = r["sign_test_p"] < 0.05
            positive = r["mean_paired_diff"] > 0
            if positive and sig:
                verdict = ("GATE MET: positive and significant "
                           "(p=%.2e < 0.05)." % r["sign_test_p"])
            elif not positive:
                verdict = ("GATE NOT MET: paired mean is not positive "
                           "(%+.3f); reported as-is." % r["mean_paired_diff"])
            else:
                verdict = ("GATE NOT MET: positive but not significant "
                           "(p=%.2e); reported as-is." % r["sign_test_p"])
            lines.append("- %s / %s: %s" % (r["name"], r["system"], verdict))
    lines.append("")
    outp = ROOT / "paper" / "analysis" / "metaqa_swap_summary.md"
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s" % outp, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="sub200,holdout200",
                    help="comma list of split dirs under paper/external/metaqa")
    args = ap.parse_args(argv)
    splits = []
    for s in [x.strip() for x in args.splits.split(",") if x.strip()]:
        d = "paper/external/metaqa/" + s
        if not (ROOT / d / "kg_openie" / "proposed_kg_nodes.json").exists():
            print("SKIP %s: kg_openie not built yet" % s, flush=True)
            continue
        splits.append(("MetaQA-" + s, d))
    if not splits:
        print("no splits ready; aborting", flush=True)
        return 1

    all_rows = []
    for system in SYSTEMS:
        all_rows.append(swap_for_system(system, splits))
    log_progress("S4 evaluation done: systems=%s splits=%s"
                 % (",".join(SYSTEMS), ",".join(n for n, _ in splits)))

    cov_rows = [coverage(name, d) for name, d in splits]
    json.dump(cov_rows, open(ROOT / "paper" / "results" /
                             "swap_vocab_align_metaqa.json", "w",
                             encoding="utf-8"), indent=2)
    write_summary(all_rows, cov_rows)
    log_progress("S5 statistics + summary done: results=kg_swap_metaqa*.json "
                 "coverage=swap_vocab_align_metaqa.json "
                 "summary=paper/analysis/metaqa_swap_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
