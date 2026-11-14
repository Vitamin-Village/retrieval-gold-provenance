"""Larger-reader table (Appendix tab:thirdreader): Mistral-Small-24B and Claude Sonnet 5.5 on 2WikiMultiHop.

Reads results/e2e_qa/{mistral_small_24b,claude_sonnet_5_5}/summary.json (written by eval/e2e_qa_swap.py --score-only
for the Claude reader, see eval/_e2e_claude_reader.py) and writes tables/tab_third_reader.tex plus the reference
conditions (closed-book, gold-only, BM25 passages) to stdout for the caption.
Usage (from paper/): python analysis/larger_readers_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READERS = [("mistral_small_24b", "Mistral-Small-24B"), ("claude_sonnet_5_5", "Claude Sonnet~5.5")]
SYSTEMS = [("audited", "Hybrid"), ("graph_centric", "Hybrid-PPR"), ("ppr", "PPR"), ("hipporag2", "HippoRAG~2")]
SPLITS = [("sub200", "first"), ("holdout200", "held-out")]


def stars(p: float) -> str:
    return "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def sgn(x: float) -> str:
    return f"{x:+.3f}"


def num(x: float) -> str:
    return f"$-${-x:.3f}" if x < 0 else f"{x:.3f}"


def pval(p: float) -> str:
    """p < 0.01 as $a\\times10^{-b}$; otherwise two significant digits (e.g. $0.015$, $0.64$)."""
    if p < 0.01:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${mant}\times10^{{{int(exp)}}}$"
    return f"${p:#.2g}$"


def main() -> None:
    S = {tag: json.load(open(ROOT / "results" / "e2e_qa" / tag / "summary.json", encoding="utf-8")) for tag, _ in READERS}
    P = {tag: {(r["split"], r["system"]): r for r in s["paired"] if r["bench"] == "2wiki"} for tag, s in S.items()}
    C = {tag: {(r["split"], r["system"], r["condition"]): r for r in s["cells"] if r["bench"] == "2wiki"} for tag, s in S.items()}
    for tag, _ in READERS:
        for sp, _ in SPLITS:
            for sy, _ in SYSTEMS:
                assert P[tag][(sp, sy)]["n"] == 200, (tag, sp, sy)
    L = [r"\begin{tabular}{llc" + "cccrc" * len(READERS) + "}", r"\toprule",
         " & & & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{name}}}" for _, name in READERS) + r"\\",
         "".join(rf"\cmidrule(lr){{{4 + 5 * i}-{8 + 5 * i}}}" for i in range(len(READERS))),
         "Retriever & Split & hit@5 swap & " + " & ".join([r"EM al. & EM ind. & EM swap [95\% CI] & $p$ & cover swap"] * len(READERS)) + r"\\",
         r"\midrule"]
    n_sig = {tag: 0 for tag, _ in READERS}
    for sy, sname in SYSTEMS:
        for sp, spname in SPLITS:
            hit = P[READERS[0][0]][(sp, sy)]["mediation"]["retrieval_hit_diff"]
            for tag, _ in READERS[1:]:
                assert abs(P[tag][(sp, sy)]["mediation"]["retrieval_hit_diff"] - hit) < 1e-12, "retrieval differs across readers"
            row = [sname, spname, f"${sgn(hit)}$"]
            for tag, _ in READERS:
                r = P[tag][(sp, sy)]
                em, cv = r["em"], r["cover"]
                n_sig[tag] += em["sign_test_p"] < 0.05
                row += [f"{em['silver']:.3f}", f"{em['openie']:.3f}",
                        f"${sgn(em['mean_paired_diff'])}{stars(em['sign_test_p'])}$ [{num(em['ci95'][0])}, {num(em['ci95'][1])}]",
                        pval(em["sign_test_p"]), f"${sgn(cv['mean_paired_diff'])}{stars(cv['sign_test_p'])}$"]
            L.append(" & ".join(row) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "tables" / "tab_third_reader.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote tables/tab_third_reader.tex")
    for tag, name in READERS:
        ref = {c: [C[tag][(sp, sy, c)]["em"] for sp, _ in SPLITS] for sy, c in (("none", "closedbook"), ("none", "oracle"))}
        ref["bm25"] = [C[tag][(sp, "bm25", "none")]["em"] for sp, _ in SPLITS]
        cover = {c: [C[tag][(sp, sy, c)]["cover"] for sp, _ in SPLITS] for sy, c in (("none", "closedbook"), ("none", "oracle"))}
        cover["bm25"] = [C[tag][(sp, "bm25", "none")]["cover"] for sp, _ in SPLITS]
        print(f"{name}: EM significant swaps {n_sig[tag]}/8; EM closed-book {ref['closedbook']}, gold-only {ref['oracle']}, "
              f"BM25 {ref['bm25']}; cover closed-book {cover['closedbook']}, gold-only {cover['oracle']}, BM25 {cover['bm25']}; "
              f"rows {S[tag]['n_rows']}")


if __name__ == "__main__":
    main()
