"""Post-process tables/tab_e2e_merged.tex for the v3 manuscript: rename the retrievers
(Audited -> Hybrid, Graph-centric -> Hybrid-PPR) and mark with a dagger the answer-level
swap cells that survive the Holm correction listed in tables/e2e_multiple_comparison.json.
Writes tables/tab_e2e_merged_v3.tex. Run after eval/e2e_tex_tables.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
BENCH = {"2WikiMultiHop": "2wiki", "MetaQA": "metaqa", "STaRK-Prime": "stark", "GraphRAG-Bench": "graphragbench"}
SYS = {"Audited": "audited", "PPR": "ppr", "Graph-centric": "graph_centric"}
LABEL = {"Audited": "Hybrid", "PPR": "PPR", "Graph-centric": "Hybrid-PPR"}
QA_COLS = [(9, "llama3_1_8b", "sub200"), (10, "llama3_1_8b", "holdout200"), (11, "qwen2_5_14b", "sub200"), (12, "qwen2_5_14b", "holdout200")]

holm = set(json.load(open(TAB / "e2e_multiple_comparison.json", encoding="utf-8"))["holm_sig"])
out, bench = [], None
for line in open(TAB / "tab_e2e_merged.tex", encoding="utf-8").read().split("\n"):
    cells = line.split(" & ")
    if len(cells) == 13 and cells[1] in SYS:
        if cells[0] in BENCH:
            bench = BENCH[cells[0]]
        sy = SYS[cells[1]]
        for idx, tag, sp in QA_COLS:
            end = "\\\\" if idx == 12 else ""
            val = cells[idx][: -len(end)] if end and cells[idx].endswith(end) else cells[idx]
            if f"{bench}|{sp}|{sy}|{tag}" in holm:
                val = val + "$^{\\dagger}$"
            cells[idx] = val + end
        cells[1] = LABEL[cells[1]]
        line = " & ".join(cells)
    out.append(line)
n_dag = sum(l.count("dagger") for l in out)
(TAB / "tab_e2e_merged_v3.tex").write_text("\n".join(out), encoding="utf-8")
print("wrote tab_e2e_merged_v3.tex with", n_dag, "Holm daggers (expected", len(holm), ")")
