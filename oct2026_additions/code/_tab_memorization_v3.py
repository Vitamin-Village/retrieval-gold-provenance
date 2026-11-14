"""Build tables/tab_memorization.tex (appendix) and tables/tab_memorization_main.tex (main text)
from results/e2e_qa/<reader>/memorization_strata.json (eval/e2e_memorization_probe.py).
Pooled over the two splits. Run from paper/: python eval/_tab_memorization_v3.py"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e2e_qa"
TAB = ROOT / "tables"
READERS = [("llama3_1_8b", "Llama-3.1-8B"), ("qwen2_5_14b", "Qwen2.5-14B")]
SYS = [("audited", "Hybrid"), ("graph_centric", "Hybrid-PPR"), ("ppr", "PPR")]
STRATA_2W = [("all", "all questions"), ("closed-book wrong", "closed-book wrong"),
             ("neither (cb wrong, evidence not recalled)", "c.b. wrong, not all triples recalled"),
             ("strict (cb wrong, no evidence triple recalled)", "c.b. wrong, no triple recalled")]
STRATA_GRB = [("all", "all questions"), ("closed-book wrong", "closed-book wrong"), ("closed-book right", "closed-book right")]


def half_up(x):
    """Signed value to three decimals; an exact half rounds away from zero (0.0525 -> +0.053)."""
    d = Decimal(repr(round(x, 9))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{d:+.3f}"


def star(p):
    return "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


D = {}
recall = {}
for tag, _ in READERS:
    d = json.load(open(RES / tag / "memorization_strata.json", encoding="utf-8"))
    recall[tag] = d["triple_recall"]
    for s in d["strata"]:
        if s["split"] == "pooled":
            D[(tag, s["bench"], s["system"], s["stratum"])] = s


def cell(tag, b, sy, st):
    s = D.get((tag, b, sy, st))
    if not s or not s.get("n"):
        return "-", "-"
    return str(s["n"]), f"${half_up(s['diff'])}{star(s['p'])}$"


L = [r"\begin{tabular}{lllrrrr}", r"\toprule",
     r" & & & \multicolumn{2}{c}{Llama-3.1-8B} & \multicolumn{2}{c}{Qwen2.5-14B}\\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"Benchmark & Retriever & Questions & $n$ & swap & $n$ & swap\\", r"\midrule"]
for b, bl, strata in (("2wiki", "2WikiMultiHop", STRATA_2W), ("graphragbench", "GraphRAG-Bench", STRATA_GRB)):
    for sy, syl in SYS:
        for k, (st, stl) in enumerate(strata):
            row = [bl if (sy == "audited" and k == 0) else "", syl if k == 0 else "", stl]
            for tag, _ in READERS:
                row += list(cell(tag, b, sy, st))
            L.append(" & ".join(row) + r"\\")
    if b == "2wiki":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_memorization.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

# compact main-text table: 2Wiki, all vs strict stratum
L = [r"\begin{tabular}{lrrrr}", r"\toprule",
     r" & \multicolumn{2}{c}{Llama-3.1-8B} & \multicolumn{2}{c}{Qwen2.5-14B}\\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"Retriever & all & unknown & all & unknown\\", r"\midrule"]
for sy, syl in SYS:
    row = [syl]
    for tag, _ in READERS:
        row.append(cell(tag, "2wiki", sy, "all")[1])
        row.append(cell(tag, "2wiki", sy, "strict (cb wrong, no evidence triple recalled)")[1])
    L.append(" & ".join(row) + r"\\")
ns = []
for tag, _ in READERS:
    ns.append(cell(tag, "2wiki", "audited", "all")[0])
    ns.append(cell(tag, "2wiki", "audited", "strict (cb wrong, no evidence triple recalled)")[0])
L.append(r"\midrule")
L.append(r"$n$ questions & " + " & ".join(ns) + r"\\")
L.append(r"triple recall & \multicolumn{2}{c}{" + f"{recall['llama3_1_8b']:.3f}" + r"} & \multicolumn{2}{c}{" + f"{recall['qwen2_5_14b']:.3f}" + r"}\\")
L += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_memorization_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print("recall", recall)
print(open(TAB / "tab_memorization_main.tex", encoding="utf-8").read())
