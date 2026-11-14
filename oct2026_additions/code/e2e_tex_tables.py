"""Emit the main-text LaTeX tables for the end-to-end QA results.

  tables/tab_e2e_main.tex    per benchmark x retriever: hit@5 swap and QA swap for both readers,
                             both splits (first / held-out); GraphRAG-Bench uses judge accuracy
  tables/tab_decirc_main.tex de-circularized evaluation on the first split, qwen reader:
                             hit@5 and QA under aligned vs independent KG, with BM25 reference
  tables/e2e_numbers.json    every number used, for the prose

Run after e2e_qa_swap.py --score-only (all benchmarks) and e2e_qa_judge.py.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e2e_qa"
TAB = ROOT / "tables"
READERS = [("llama3_1_8b", "Llama-3.1-8B"), ("qwen2_5_14b", "Qwen2.5-14B")]
BENCH = [("2wiki", "2WikiMultiHop"), ("metaqa", "MetaQA"), ("stark", "STaRK-Prime"), ("graphragbench", "GraphRAG-Bench")]
SYS = [("audited", "Audited"), ("ppr", "PPR"), ("graph_centric", "Graph-centric")]
SPLITS = ["sub200", "holdout200"]
JUDGE = "judge_mistral_small_24b.json"


def load(tag):
    s = json.load(open(RES / tag / "summary.json", encoding="utf-8"))
    paired = {(e["bench"], e["split"], e["system"]): e for e in s["paired"]}
    cells = {(c["bench"], c["split"], c["system"], c["condition"]): c for c in s["cells"]}
    jp = RES / tag / JUDGE
    judge = json.load(open(jp, encoding="utf-8")) if jp.exists() else {"paired": [], "cells": []}
    jpaired = {(e["bench"], e["split"], e["system"]): e for e in judge["paired"]}
    jcells = {(c["bench"], c["split"], c["system"], c["condition"]): c for c in judge["cells"]}
    return paired, cells, jpaired, jcells


def star(p):
    return "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def qa_swap(tag_data, b, sp, sy):
    paired, cells, jpaired, jcells = tag_data
    if b == "graphragbench":
        e = jpaired.get((b, sp, sy))
        if not e:
            return None
        return e["mean_paired_diff"], e["ci95"], e["sign_test_p"], e["silver"], e["openie"]
    e = paired.get((b, sp, sy))
    if not e:
        return None
    m = e["em"]
    return m["mean_paired_diff"], m["ci95"], m["sign_test_p"], m["silver"], m["openie"]


def main():
    TAB.mkdir(exist_ok=True)
    data = {tag: load(tag) for tag, _ in READERS}
    numbers = {}
    # ---------------- main swap table
    L = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         " & & \\multicolumn{2}{c}{hit@5 swap} & \\multicolumn{2}{c}{QA swap, " + READERS[0][1] + "} & \\multicolumn{2}{c}{QA swap, " + READERS[1][1] + "}\\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "Benchmark & Retriever & first & held-out & first & held-out & first & held-out\\\\", "\\midrule"]
    for b, bl in BENCH:
        for sy, sl in SYS:
            row = [bl if sy == "audited" else "", sl]
            ref = data[READERS[0][0]][0]
            hits = []
            for sp in SPLITS:
                e = ref.get((b, sp, sy))
                hits.append(f"{e['mediation']['retrieval_hit_diff']:+.3f}" if e else "-")
            row += hits
            for tag, _ in READERS:
                for sp in SPLITS:
                    q = qa_swap(data[tag], b, sp, sy)
                    if q:
                        d, ci, p, s_, o_ = q
                        row.append(f"{d:+.3f}{star(p)}")
                        numbers[f"{b}|{sp}|{sy}|{tag}"] = {"qa_diff": d, "ci": ci, "p": p, "aligned": s_, "independent": o_,
                                                          "hit_diff": ref[(b, sp, sy)]["mediation"]["retrieval_hit_diff"] if (b, sp, sy) in ref else None}
                    else:
                        row.append("-")
            L.append(" & ".join(row) + "\\\\")
        if b != BENCH[-1][0]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(TAB / "tab_e2e_main.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---------------- de-circularized table (first split, second reader)
    tag = READERS[1][0]
    paired, cells, jpaired, jcells = data[tag]
    L = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{Retrieval hit@5} & \\multicolumn{3}{c}{QA accuracy}\\\\", "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "Benchmark & Retriever & aligned & indep. & BM25 & aligned & indep. & BM25\\\\", "\\midrule"]
    for b, bl in BENCH:
        sp = "sub200"
        if b == "graphragbench":
            qc, key = jcells, "judge_acc"
        else:
            qc, key = cells, "em"
        bm = cells.get((b, sp, "bm25", "none"))
        bmq = qc.get((b, sp, "bm25", "none"))
        for sy, sl in SYS:
            s_ = cells.get((b, sp, sy, "kg_silver"))
            o_ = cells.get((b, sp, sy, "kg_openie"))
            sq = qc.get((b, sp, sy, "kg_silver"))
            oq = qc.get((b, sp, sy, "kg_openie"))
            if not (s_ and o_ and sq and oq and bm and bmq):
                continue
            L.append(f"{bl if sy == 'audited' else ''} & {sl} & {s_['hit5']:.3f} & {o_['hit5']:.3f} & {bm['hit5']:.3f} & {sq[key]:.3f} & {oq[key]:.3f} & {bmq[key]:.3f}\\\\")
            numbers[f"decirc|{b}|{sy}"] = {"hit_aligned": s_["hit5"], "hit_indep": o_["hit5"], "hit_bm25": bm["hit5"],
                                            "qa_aligned": sq[key], "qa_indep": oq[key], "qa_bm25": bmq[key]}
        cb = qc.get((b, sp, "none", "closedbook"))
        orc = qc.get((b, sp, "none", "oracle"))
        if cb and orc:
            numbers[f"bounds|{b}"] = {"closedbook": cb[key], "oracle": orc[key]}
        if b != BENCH[-1][0]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(TAB / "tab_decirc_main.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    # closed-book for both readers
    for tag, _ in READERS:
        _, cells, _, jcells = data[tag]
        for b, _ in BENCH:
            qc, key = (jcells, "judge_acc") if b == "graphragbench" else (cells, "em")
            for sp in SPLITS:
                cb = qc.get((b, sp, "none", "closedbook"))
                if cb:
                    numbers[f"closedbook|{b}|{sp}|{tag}"] = cb[key]
    # ---------------- merged full-width table: de-circularized columns + QA swaps + reference rows
    tag2 = READERS[1][0]
    paired2, cells2, jpaired2, jcells2 = data[tag2]
    L = ["\\begin{tabular}{llrrrrrrrrrrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{hit@5, first split} & hit@5 swap & \\multicolumn{3}{c}{QA, " + READERS[1][1] + ", first} & \\multicolumn{2}{c}{QA swap, " + READERS[0][1] + "} & \\multicolumn{2}{c}{QA swap, " + READERS[1][1] + "}\\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}",
         "Benchmark & Retriever & aligned & indep. & BM25 & held-out & aligned & indep. & BM25 & first & held-out & first & held-out\\\\", "\\midrule"]
    for b, bl in BENCH:
        qc, key = (jcells2, "judge_acc") if b == "graphragbench" else (cells2, "em")
        bm = cells2.get((b, "sub200", "bm25", "none"))
        bmq = qc.get((b, "sub200", "bm25", "none"))
        cb = qc.get((b, "sub200", "none", "closedbook"))
        orc = qc.get((b, "sub200", "none", "oracle"))
        for sy, sl in SYS:
            s_, o_ = cells2.get((b, "sub200", sy, "kg_silver")), cells2.get((b, "sub200", sy, "kg_openie"))
            sq, oq = qc.get((b, "sub200", sy, "kg_silver")), qc.get((b, "sub200", sy, "kg_openie"))
            ho = data[READERS[0][0]][0].get((b, "holdout200", sy))
            row = [bl if sy == "audited" else "", sl,
                   f"{s_['hit5']:.3f}" if s_ else "-", f"{o_['hit5']:.3f}" if o_ else "-", f"{bm['hit5']:.3f}" if bm else "-",
                   f"{ho['mediation']['retrieval_hit_diff']:+.3f}" if ho else "-",
                   f"{sq[key]:.3f}" if sq else "-", f"{oq[key]:.3f}" if oq else "-", f"{bmq[key]:.3f}" if bmq else "-"]
            for tag, _ in READERS:
                for sp in SPLITS:
                    q = qa_swap(data[tag], b, sp, sy)
                    row.append(f"{q[0]:+.3f}{star(q[2])}" if q else "-")
            L.append(" & ".join(row) + "\\\\")
        if cb and orc:
            L.append(f" & \\textit{{closed-book / gold-only}} & & & & & \\multicolumn{{3}}{{l}}{{\\textit{{{cb[key]:.3f} / {orc[key]:.3f}}}}} & & & & \\\\")
        if b != BENCH[-1][0]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(TAB / "tab_e2e_merged.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(numbers, open(TAB / "e2e_numbers.json", "w", encoding="utf-8"), indent=1)
    print("wrote tab_e2e_main.tex, tab_decirc_main.tex, tab_e2e_merged.tex, e2e_numbers.json")


if __name__ == "__main__":
    main()
