# -*- coding: utf-8 -*-
"""SPEC -- vocabulary-transplant runner for the 2WikiMultiHop KG swap (E2).

Question answered
-----------------
Does the silver KG's gold-aligned entity VOCABULARY alone (grafted into the
noisy independent OpenIE graph by paper/eval/build_transplant_kgs.py as
kg_openie_svocab) transfer the swap gain? The decisive comparison is
TRANSPLANT vs OPENIE (same OpenIE edges, only the vocabulary differs); the
original silver and openie numbers are reported as anchors so the transplant
sits between them in one table.

Protocol (mirrors _kg_swap_2wiki_r2.py exactly)
-----------------------------------------------
* RAG_V3_DENSE_BACKEND=deterministic; random.seed(20260611); B=10000.
* For each split D in {paper/external/2wiki/sub200, .../holdout200}:
    - ev.run(qa=D/qa.jsonl, passages=D/passages.jsonl,
             manifest=D/manifest.json, kg_dir=D/kg_openie_svocab,
             systems=("bm25", SYSTEM), top_k=5,
             out=D/_kgswap_<SYSTEM>_kg_openie_svocab.json)   [NEW file]
    - The OPENIE and SILVER anchors are READ from the existing result files
      (never re-run, never overwritten):
        D/_kgswap_<SYSTEM>_<kg>.json, falling back for SYSTEM=proposed_v3 to
        the original unsuffixed D/_kgswap_<kg>.json written by
        _kg_swap_2wiki.py. Re-running would reproduce them bit-identically
        (deterministic backend, no RAG_V3_SEED), so reading is equivalent.
    - Per-question paired diffs of the graph system:
        diff(q) = hit@5_transplant(q) - hit@5_openie(q)
      over the qa_ids present in both record lists (insertion order of the
      transplant records; same qa.jsonl => identical id sets).
    - random.seed(20260611) is re-issued IMMEDIATELY before each bootstrap
      (audit F7 convention); 10000 resamples, percentile 95% CI; exact
      two-sided sign test over discordant questions.
    - If D/kg_silver_devocab exists (T2; the builder currently SKIPs it),
      a T2 row is appended with diffs = devocab - silver. Absent dir =>
      no row, no error.

Output
------
paper/results/kg_transplant_2wiki.json for the default system, else
paper/results/kg_transplant_2wiki_<system>.json (cross-retriever
replication must not clobber the primary file). One row per split per
condition:
  {name, system, condition ("T1_openie_svocab" | "T2_silver_devocab"), n,
   transplant {graph, bm25, delta}, openie {graph, bm25, delta},
   silver {graph, bm25, delta}, delta_graph_vs_openie, delta_graph_vs_silver,
   swap_gap_closed_fraction = (g_transplant-g_openie)/(g_silver-g_openie)
   (None when the anchor gap is 0), mean_paired_diff, ci95,
   discordant_pos, discordant_neg, sign_test_p, bm25_max_abs_dev (sanity:
   |bm25_transplant - bm25_anchor|, expected 0.0 -- bm25 ignores the KG)}.

CLI
---
    python paper/eval/_kg_transplant_2wiki.py [--system proposed_v3]
--system defaults to proposed_v3, the audited graph retriever of the
original swap (_kg_swap_2wiki.py ran systems=("bm25", "proposed_v3")).

Hard rules honored: no hyperparameter tuning; deterministic seeds/sorts;
no existing file under paper/eval or paper/external is modified (all
ev.run out_paths are new *_kg_openie_svocab / *_kg_silver_devocab names).
ASCII-only output.
"""
import os
os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")
import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402

random.seed(20260611)
B = 10000
DEFAULT_SYSTEM = "proposed_v3"
SPLITS = [("2Wiki-sub200", "paper/external/2wiki/sub200"),
          ("2Wiki-holdout200", "paper/external/2wiki/holdout200")]
T1_KG = "kg_openie_svocab"
T1B_KG = "kg_openie_svocab_noprov"   # T1 with source_docs emptied on silver-sourced nodes
OC_KG = "kg_openie_resorted"         # pure order control (proposed_v3 only)
T2_KG = "kg_silver_devocab"
# q-curve partial-provenance conditions (build_transplant_kgs.py, seed
# 20260612): fraction q of the silver-sourced nodes keeps its gold
# source_docs; q=0 is T1b, q=1 is T1. Evaluated only under --q-curve so the
# default invocation reproduces the original T1/T1b/OC output unchanged.
Q_KGS = [(0.25, "kg_openie_svocab_q25"),
         (0.5, "kg_openie_svocab_q50"),
         (0.75, "kg_openie_svocab_q75")]


def run_kg(d, kg_sub, system):
    d = Path(ROOT / d)
    out = d / ("_kgswap_%s_%s.json" % (system, kg_sub))
    return ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl",
                  manifest_path=d / "manifest.json", kg_dir=d / kg_sub,
                  systems=("bm25", system), out_path=out, top_k=5)


def load_anchor(d, kg_sub, system):
    """Read the EXISTING swap result payload for (system, kg). Never re-runs."""
    d = Path(ROOT / d)
    candidates = [d / ("_kgswap_%s_%s.json" % (system, kg_sub))]
    if system == "proposed_v3":
        # Original _kg_swap_2wiki.py wrote unsuffixed files for proposed_v3.
        candidates.append(d / ("_kgswap_%s.json" % kg_sub))
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        "no existing swap result for system=%s kg=%s under %s (tried %s); "
        "run the original swap driver first" % (
            system, kg_sub, d, ", ".join(p.name for p in candidates)))


def per_q_hits(payload, system):
    # hit_at_k lives under record["metrics"] (r2 convention; the flat read
    # in the original _kg_swap_2wiki.per_q_hits silently returned 0.0).
    out = {}
    for r in payload["results"][system]["records"]:
        out[str(r["qa_id"])] = float(r["metrics"]["hit_at_k"])
    return out


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


def paired_row(name, system, condition, payload_new, payload_base,
               anchors, baseline_key):
    """Build one result row: payload_new vs payload_base paired on the graph
    system, with both anchors attached. ``baseline_key`` names which anchor
    the paired diff is taken against ("openie" for T1, "silver" for T2)."""
    g_new, l_new = overall_hit(payload_new, system), overall_hit(payload_new, "bm25")
    hn, hb = per_q_hits(payload_new, system), per_q_hits(payload_base, system)
    qs = [q for q in hn if q in hb]
    if not qs:
        return None
    if len(qs) != len(hn) or len(qs) != len(hb):
        # Soft check only: anchors may legitimately carry retriever-empty
        # records; a hard set-identity assert would reject valid runs. The
        # real risk is a SILENTLY shrunk paired sample, hence the warning.
        print("WARNING: paired qa_id sets differ on %s/%s: n_paired=%d "
              "n_new=%d n_base=%d" % (name, condition, len(qs), len(hn),
                                      len(hb)), flush=True)
    diffs = [hn[q] - hb[q] for q in qs]
    mean = sum(diffs) / len(diffs)
    # Re-seed immediately before the bootstrap so the CI is reproducible
    # regardless of any random consumption inside ev.run (audit F7).
    random.seed(20260611)
    lo, hi = boot_ci(diffs)
    pos, neg, p = sign_test(diffs)

    g_s, l_s = anchors["silver"]
    g_o, l_o = anchors["openie"]
    gap = g_s - g_o
    bm25_dev = max(abs(l_new - l_o), abs(l_new - l_s))
    row = {
        "name": name, "system": system, "condition": condition, "n": len(qs),
        "paired_against": baseline_key,
        "transplant": {"graph": round(g_new, 4), "bm25": round(l_new, 4),
                       "delta": round(g_new - l_new, 4)},
        "openie": {"graph": round(g_o, 4), "bm25": round(l_o, 4),
                   "delta": round(g_o - l_o, 4)},
        "silver": {"graph": round(g_s, 4), "bm25": round(l_s, 4),
                   "delta": round(g_s - l_s, 4)},
        "delta_graph_vs_openie": round(g_new - g_o, 4),
        "delta_graph_vs_silver": round(g_new - g_s, 4),
        "swap_gap_closed_fraction": (
            round((g_new - g_o) / gap, 4) if abs(gap) > 1e-12 else None
        ),
        "mean_paired_diff": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "discordant_pos": pos, "discordant_neg": neg,
        "sign_test_p": p,
        "bm25_max_abs_dev": round(bm25_dev, 4),
    }
    print("%-18s %s %s | OPENIE graph=%.3f | TRANSPLANT graph=%.3f bm25=%.3f "
          "| SILVER graph=%.3f | d_vs_openie=%+.3f d_vs_silver=%+.3f "
          "gap_closed=%s | paired(vs %s) mean=%+.3f CI[%+.3f,%+.3f] "
          "discordant +%d/-%d p=%.2e"
          % (name, system, condition, g_o, g_new, l_new, g_s,
             g_new - g_o, g_new - g_s,
             ("%.3f" % row["swap_gap_closed_fraction"])
             if row["swap_gap_closed_fraction"] is not None else "n/a",
             baseline_key, mean, lo, hi, pos, neg, p),
          flush=True)
    if bm25_dev > 1e-9:
        print("WARNING: bm25 deviates from anchors by %.6f on %s "
              "(expected 0.0; bm25 ignores the KG)" % (bm25_dev, name),
              flush=True)
    return row


def run_q_curve(system):
    """q-curve mode: evaluate the partial-provenance conditions (Q_KGS)
    against the OPENIE anchor with the exact paired protocol of the main
    mode, and write paper/results/kg_transplant_q_2wiki[_<system>].json.
    Same record schema as the main output plus a 'q' field per row.
    The T1/T1b endpoints are NOT re-run here; the q-curve summary joins
    them from the existing kg_transplant_2wiki*.json files (q=1/q=0)."""
    rows = []
    for name, d in SPLITS:
        missing = [kg for _, kg in Q_KGS
                   if not (ROOT / d / kg / "proposed_kg_nodes.json").exists()]
        if missing:
            print("ERROR: %s missing under %s -- run "
                  "paper/eval/build_transplant_kgs.py first"
                  % (",".join(missing), d), flush=True)
            return 1
        po = load_anchor(d, "kg_openie", system)
        ps = load_anchor(d, "kg_silver", system)
        anchors = {"openie": (overall_hit(po, system), overall_hit(po, "bm25")),
                   "silver": (overall_hit(ps, system), overall_hit(ps, "bm25"))}
        for q, kg in Q_KGS:
            pq = run_kg(d, kg, system)
            row = paired_row(name, system, "Tq_%s" % kg, pq, po,
                             anchors, "openie")
            if row is None:
                print("WARNING: no paired questions for %s/%s, skipping"
                      % (name, kg), flush=True)
                continue
            row["q"] = q
            rows.append(row)
    out_name = ("kg_transplant_q_2wiki.json" if system == DEFAULT_SYSTEM
                else "kg_transplant_q_2wiki_%s.json" % system)
    outp = ROOT / "paper" / "results" / out_name
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=True)
    print("wrote %s" % outp, flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default=DEFAULT_SYSTEM,
                    help="graph system to evaluate (default: %s, the audited "
                         "retriever of the original swap)" % DEFAULT_SYSTEM)
    ap.add_argument("--q-curve", action="store_true",
                    help="evaluate the partial-provenance q conditions "
                         "(kg_openie_svocab_q25/q50/q75) instead of "
                         "T1/T1b/OC/T2; writes kg_transplant_q_2wiki*.json")
    args = ap.parse_args(argv)
    system = args.system
    if args.q_curve:
        return run_q_curve(system)

    rows = []
    for name, d in SPLITS:
        if not (ROOT / d / T1_KG / "proposed_kg_nodes.json").exists():
            print("ERROR: %s missing under %s -- run "
                  "paper/eval/build_transplant_kgs.py first" % (T1_KG, d),
                  flush=True)
            return 1
        po = load_anchor(d, "kg_openie", system)
        ps = load_anchor(d, "kg_silver", system)
        anchors = {"openie": (overall_hit(po, system), overall_hit(po, "bm25")),
                   "silver": (overall_hit(ps, system), overall_hit(ps, "bm25"))}

        # T1: transplant vs OPENIE (the main comparison).
        pt = run_kg(d, T1_KG, system)
        row = paired_row(name, system, "T1_openie_svocab", pt, po,
                         anchors, "openie")
        if row is None:
            print("WARNING: no paired questions for %s, skipping" % name,
                  flush=True)
            continue
        rows.append(row)

        # T1b: provenance-neutralized transplant vs OPENIE. Decomposes the
        # lexical-seeding sub-channel from the passage-mapping sub-channel
        # (adjudication 2026-06-11).
        if (ROOT / d / T1B_KG / "proposed_kg_nodes.json").exists():
            pt1b = run_kg(d, T1B_KG, system)
            row1b = paired_row(name, system, "T1b_openie_svocab_noprov",
                               pt1b, po, anchors, "openie")
            if row1b is not None:
                rows.append(row1b)
        else:
            print("%-18s T1b %s not built -- run build_transplant_kgs.py "
                  "(no T1b row)" % (name, T1B_KG), flush=True)

        # Order control (proposed_v3 only): OpenIE nodes+edges re-sorted,
        # zero vocabulary change. Rules out a pure edge-file-order artifact
        # in the legacy BFS pairing. PPR-family systems are order-
        # insensitive and skip it.
        if system == "proposed_v3":
            if (ROOT / d / OC_KG / "proposed_kg_nodes.json").exists():
                poc = run_kg(d, OC_KG, system)
                rowoc = paired_row(name, system, "OC_openie_resorted",
                                   poc, po, anchors, "openie")
                if rowoc is not None:
                    rows.append(rowoc)
            else:
                print("%-18s OC %s not built -- rerun builder with "
                      "--with-order-control (no OC row)" % (name, OC_KG),
                      flush=True)

        # T2 (optional): devocab vs SILVER. Only if the builder produced it.
        if (ROOT / d / T2_KG / "proposed_kg_nodes.json").exists():
            pd2 = run_kg(d, T2_KG, system)
            row2 = paired_row(name, system, "T2_silver_devocab", pd2, ps,
                              anchors, "silver")
            if row2 is not None:
                rows.append(row2)
        else:
            print("%-18s T2 %s not built (builder SKIP) -- no T2 row"
                  % (name, T2_KG), flush=True)

    out_name = ("kg_transplant_2wiki.json" if system == DEFAULT_SYSTEM
                else "kg_transplant_2wiki_%s.json" % system)
    outp = ROOT / "paper" / "results" / out_name
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=True)
    print("wrote %s" % outp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
