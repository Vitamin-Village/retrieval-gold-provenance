"""Generate appendix tables for the v3 manuscript from the result files (no transcription):
  tables/tab_readerref.tex   closed-book / gold-only / BM25 answer accuracy per reader and split
  tables/tab_mediation.tex   2WikiMultiHop EM swap on retrieval-flip vs no-flip questions
  tables/tab_linkcounts.tex  passages linked by the aligned and independent KGs and their gold fraction
Run from paper/: python eval/_tab_appendix_v3.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e2e_qa"
TAB = ROOT / "tables"
READERS = [("llama3_1_8b", "Llama"), ("qwen2_5_14b", "Qwen")]
BENCH = [("2wiki", "2WikiMultiHop"), ("metaqa", "MetaQA"), ("stark", "STaRK-Prime"), ("graphragbench", "GraphRAG-Bench$^{j}$")]
SPLIT = [("sub200", "first"), ("holdout200", "held-out")]
SYS = [("audited", "Hybrid"), ("graph_centric", "Hybrid-PPR"), ("ppr", "PPR")]


def cells(tag, judge=False):
    if judge:
        d = json.load(open(RES / tag / "judge_mistral_small_24b.json", encoding="utf-8"))
        return {(c["bench"], c["split"], c["system"], c["condition"]): c["judge_acc"] for c in d["cells"]}
    d = json.load(open(RES / tag / "summary.json", encoding="utf-8"))
    return {(c["bench"], c["split"], c["system"], c["condition"]): c["em"] for c in d["cells"]}


C = {tag: (cells(tag), cells(tag, judge=True)) for tag, _ in READERS}
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
     r" & & \multicolumn{3}{c}{Llama-3.1-8B} & \multicolumn{3}{c}{Qwen2.5-14B}\\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"Benchmark & Split & closed & gold & BM25 & closed & gold & BM25\\", r"\midrule"]
for b, bl in BENCH:
    for sp, sl in SPLIT:
        row = [bl if sp == "sub200" else "", sl]
        for tag, _ in READERS:
            c = C[tag][1] if b == "graphragbench" else C[tag][0]
            row += [f"{c[(b, sp, 'none', 'closedbook')]:.3f}", f"{c[(b, sp, 'none', 'oracle')]:.3f}", f"{c[(b, sp, 'bm25', 'none')]:.3f}"]
        L.append(" & ".join(row) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_readerref.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

L = [r"\begin{tabular}{lllrrrr}", r"\toprule",
     r"Reader & Retriever & Split & $n$ flip & EM $\Delta$ flip & $n$ none & EM $\Delta$ none\\", r"\midrule"]
for tag, rl in READERS:
    s = json.load(open(RES / tag / "summary.json", encoding="utf-8"))
    P = {(p["bench"], p["split"], p["system"]): p["mediation"] for p in s["paired"]}
    for sy, syl in SYS:
        for sp, sl in SPLIT:
            m = P[("2wiki", sp, sy)]
            L.append(" & ".join([rl, syl, sl, str(m["n_flips_pos"]), f"{m['em_diff_on_retrieval_flips_pos']:+.3f}",
                                 str(m["n_no_flip"]), f"{m['em_diff_on_no_flip']:+.3f}"]) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_mediation.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

lc = json.load(open(ROOT / "results" / "kg_passage_link_counts.json", encoding="utf-8"))
L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
     r" & & & \multicolumn{2}{c}{Aligned KG} & \multicolumn{2}{c}{Independent KG}\\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"Benchmark & Split & Passages & linked & gold frac. & linked & gold frac.\\", r"\midrule"]
name = dict(BENCH)
name["graphragbench"] = "GraphRAG-Bench"
for r in lc:
    L.append(" & ".join([name[r["bench"]], dict(SPLIT)[r["split"]], f"{r['n_passages']:,}".replace(",", "{,}"),
                         f"{r['kg_silver']['linked']:,}".replace(",", "{,}"), f"{r['kg_silver']['gold_frac_of_linked']:.2f}",
                         f"{r['kg_openie']['linked']:,}".replace(",", "{,}"), f"{r['kg_openie']['gold_frac_of_linked']:.2f}"]) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_linkcounts.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote tab_readerref.tex, tab_mediation.tex, tab_linkcounts.tex")
