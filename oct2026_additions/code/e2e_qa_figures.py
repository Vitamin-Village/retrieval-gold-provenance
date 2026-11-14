"""Figures and tables for the end-to-end QA swap (E1/E2/E6).

Reads results/e2e_qa/<model>/summary.json and per_question.jsonl for one or more
readers and writes, under figures/ and tables/:

  fig_e2e_swap.pdf      per benchmark x retriever: paired swap effect on hit@5 and on QA EM/F1
  fig_e2e_dose.pdf      2Wiki sub200: edge fraction f and mapping fraction q vs hit@5 and EM
  tab_e2e_swap.md/.tex  the numbers behind fig_e2e_swap
  tab_decirc.md/.tex    de-circularized evaluation: original (aligned) vs independent, graph vs lexical, retrieval vs QA

Usage: python eval/e2e_qa_figures.py --models llama3_1_8b,qwen2_5_14b
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e2e_qa"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
BENCH_LABEL = {"2wiki": "2WikiMultiHop", "metaqa": "MetaQA", "stark": "STaRK-Prime", "graphragbench": "GraphRAG-Bench"}
SYS_LABEL = {"audited": "Hybrid", "ppr": "PPR", "graph_centric": "Hybrid-PPR"}
SPLIT_LABEL = {"sub200": "first", "holdout200": "held-out"}


READER_LABEL = {"llama3_1_8b": "Llama-3.1-8B", "qwen2_5_14b": "Qwen2.5-14B"}
BENCH_ORDER = ["2wiki", "metaqa", "stark", "graphragbench"]


def load(model_tag):
    """Summary with GraphRAG-Bench QA numbers replaced by LM-judge accuracy (its gold
    answers are sentences, so EM is uninformative)."""
    d = RES / model_tag
    summ = json.load(open(d / "summary.json", encoding="utf-8"))
    jp = d / "judge_mistral_small_24b.json"
    if jp.exists():
        judge = json.load(open(jp, encoding="utf-8"))
        jpaired = {(e["bench"], e["split"], e["system"]): e for e in judge["paired"]}
        for e in summ["paired"]:
            j = jpaired.get((e["bench"], e["split"], e["system"]))
            if j:
                e["em"] = {"silver": j["silver"], "openie": j["openie"], "mean_paired_diff": j["mean_paired_diff"],
                           "ci95": j["ci95"], "sign_test_p": j["sign_test_p"]}
        jcells = {(c["bench"], c["split"], c["system"], c["condition"]): c["judge_acc"] for c in judge["cells"]}
        for c in summ["cells"]:
            k = (c["bench"], c["split"], c["system"], c["condition"])
            if k in jcells:
                c["em"] = jcells[k]
    rows = [json.loads(l) for l in open(d / "per_question.jsonl", encoding="utf-8")]
    return summ, rows


def paired_table(summaries, metric="em"):
    """rows: (bench, split, system) -> per model: hit diff, metric diff, ci, p"""
    out = collections.OrderedDict()
    for tag, s in summaries.items():
        for e in s["paired"]:
            key = (e["bench"], e["split"], e["system"])
            out.setdefault(key, {})[tag] = {
                "hit_diff": e["mediation"]["retrieval_hit_diff"],
                "silver": e[metric]["silver"], "openie": e[metric]["openie"],
                "diff": e[metric]["mean_paired_diff"], "ci": e[metric]["ci95"], "p": e[metric]["sign_test_p"],
                "n": e["n"]}
    return out


def write_swap_table(tab, metric, tags):
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    md = ["| Benchmark | Split | Retriever | hit@5 Δ | " + " | ".join(f"{t}: {metric.upper()} aligned / indep. / Δ [CI] / p" for t in tags) + " |",
          "|---|---|---|---:|" + "---|" * len(tags)]
    tex = ["\\begin{tabular}{lll" + "r" + "rrl" * len(tags) + "}", "\\toprule",
           "Benchmark & Split & Retriever & hit@5 $\\Delta$ & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{t}: {metric.upper()} aligned / indep. / $\\Delta$ [95\\% CI], $p$}}" for t in tags) + "\\\\", "\\midrule"]
    for (b, sp, sy), per in sorted(tab.items()):
        hit = next(iter(per.values()))["hit_diff"]
        cells_md, cells_tex = [], []
        for t in tags:
            v = per.get(t)
            if not v:
                cells_md.append("-")
                cells_tex.append("- & - & -")
                continue
            cells_md.append(f"{v['silver']:.3f} / {v['openie']:.3f} / {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] / {v['p']:.1e}")
            cells_tex.append(f"{v['silver']:.3f} & {v['openie']:.3f} & {v['diff']:+.3f} [{v['ci'][0]:+.3f},\\,{v['ci'][1]:+.3f}], ${v['p']:.1e}$".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
        md.append(f"| {BENCH_LABEL.get(b, b)} | {SPLIT_LABEL.get(sp, sp)} | {SYS_LABEL.get(sy, sy)} | {hit:+.3f} | " + " | ".join(cells_md) + " |")
        tex.append(f"{BENCH_LABEL.get(b, b)} & {SPLIT_LABEL.get(sp, sp)} & {SYS_LABEL.get(sy, sy)} & {hit:+.3f} & " + " & ".join(cells_tex) + "\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(TAB / f"tab_e2e_swap_{metric}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(TAB / f"tab_e2e_swap_{metric}.tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")


def decirc_table(summaries):
    """Original (aligned KG) vs provenance-independent (OpenIE KG): graph and lexical, hit@5 and EM/F1."""
    lines = ["| Benchmark | Split | Reader | Retriever | Metric | Original (aligned) | Independent | Δ |", "|---|---|---|---|---|---:|---:|---:|"]
    for tag, s in summaries.items():
        cells = {(c["bench"], c["split"], c["system"], c["condition"]): c for c in s["cells"]}
        for (b, sp, sy, cond), c in sorted(cells.items()):
            if cond != "kg_silver":
                continue
            o = cells.get((b, sp, sy, "kg_openie"))
            if not o:
                continue
            for m in ("hit5", "em", "f1"):
                if c[m] is None:
                    continue
                lines.append(f"| {BENCH_LABEL.get(b, b)} | {SPLIT_LABEL.get(sp, sp)} | {tag} | {SYS_LABEL.get(sy, sy)} | {m} | {c[m]:.3f} | {o[m]:.3f} | {c[m] - o[m]:+.3f} |")
        for (b, sp, sy, cond), c in sorted(cells.items()):
            if sy == "bm25" and cond == "none":
                for m in ("hit5", "em", "f1"):
                    lines.append(f"| {BENCH_LABEL.get(b, b)} | {SPLIT_LABEL.get(sp, sp)} | {tag} | BM25 | {m} | {c[m]:.3f} | {c[m]:.3f} | +0.000 |")
            if sy == "none" and cond in ("closedbook", "oracle"):
                for m in ("em", "f1"):
                    lines.append(f"| {BENCH_LABEL.get(b, b)} | {SPLIT_LABEL.get(sp, sp)} | {tag} | {cond} | {m} | {c[m]:.3f} | {c[m]:.3f} | - |")
    open(TAB / "tab_decirc.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


def dose_curves(summaries):
    """2Wiki sub200 audited: dose_fXX (edge fraction) and q-curve (mapping fraction) in hit@5 and EM."""
    out = {}
    for tag, s in summaries.items():
        cells = {(c["bench"], c["split"], c["system"], c["condition"]): c for c in s["cells"]}
        f_curve = []
        for f in ("f00", "f25", "f50", "f75", "f100"):
            c = cells.get(("2wiki", "sub200", "audited", f"dose_{f}"))
            if c:
                f_curve.append((int(f[1:]) / 100, c["hit5"], c["em"], c["f1"]))
        q_curves = {}
        for sy in ("audited", "ppr", "graph_centric"):
            for sp in ("sub200", "holdout200"):
                pts = []
                for q, cond in ((0.0, "kg_openie_svocab_noprov"), (0.25, "kg_openie_svocab_q25"), (0.5, "kg_openie_svocab_q50"),
                                (0.75, "kg_openie_svocab_q75"), (1.0, "kg_openie_svocab")):
                    c = cells.get(("2wiki", sp, sy, cond))
                    if c:
                        pts.append((q, c["hit5"], c["em"], c["f1"]))
                if pts:
                    q_curves[(sy, sp)] = pts
        out[tag] = {"f": f_curve, "q": q_curves}
    return out


# Retrievers of the main swap figure, in plotting order (frozen surrogate runs; HippoRAG 2 and the
# BGE-M3 "@bge" reruns, also present in the summaries, are reported elsewhere and not drawn here).
SWAP_FIG_SYSTEMS = ["audited", "graph_centric", "ppr"]
# Final widths in the manuscript: fig_e2e_swap at \textwidth, fig_e2e_dose at \columnwidth (acl.sty),
# so the font sizes below are the printed sizes (>= 7 pt).
TEXTWIDTH_IN, COLUMNWIDTH_IN = 455.24 / 72, 219.09 / 72
FIG_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
          "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.0, "axes.linewidth": 0.6,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
          "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0}


def plot(summaries, tags):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib unavailable; tables only")
        return
    plt.rcParams.update(FIG_RC)
    tab = paired_table(summaries, "em")
    # main figure: frozen surrogate runs of the three retrievers only
    tab = {k: v for k, v in tab.items() if "@" not in k[2] and k[2] in SWAP_FIG_SYSTEMS}
    keys = sorted(tab, key=lambda k: (BENCH_ORDER.index(k[0]) if k[0] in BENCH_ORDER else 9, k[1] != "sub200", SWAP_FIG_SYSTEMS.index(k[2])))
    benches = [b for b in BENCH_ORDER if any(k[0] == b for k in keys)]
    fig, axes = plt.subplots(1, len(benches), figsize=(TEXTWIDTH_IN, 2.25), sharey=True)
    fig.subplots_adjust(left=0.062, right=0.997, top=0.83, bottom=0.215, wspace=0.07)
    if len(benches) == 1:
        axes = [axes]
    for ax, b in zip(axes, benches):
        ks = [k for k in keys if k[0] == b]
        # groups of three bars; a small gap separates the first and the held-out split
        x = np.array([i + (0.45 if k[1] != "sub200" else 0.0) for i, k in enumerate(ks)])
        w = 0.8 / (1 + len(tags))
        ax.bar(x, [next(iter(tab[k].values()))["hit_diff"] for k in ks], w, label="hit@5", color="#444")
        for i, t in enumerate(tags, 1):
            vals = [tab[k].get(t, {}).get("diff", np.nan) for k in ks]
            err = [[tab[k][t]["diff"] - tab[k][t]["ci"][0] if t in tab[k] else 0 for k in ks],
                   [tab[k][t]["ci"][1] - tab[k][t]["diff"] if t in tab[k] else 0 for k in ks]]
            ax.bar(x + i * w, vals, w, yerr=err, capsize=1.3, error_kw={"elinewidth": 0.6, "capthick": 0.6},
                   label=f"QA, {READER_LABEL.get(t, t)}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("GraphRAG-Bench\n(QA = judge accuracy)" if b == "graphragbench" else BENCH_LABEL.get(b, b), fontsize=8, pad=3,
                     linespacing=1.05)
        centers = x + w * len(tags) / 2
        ax.set_xticks(centers)
        ax.set_xticklabels([SYS_LABEL[k[2]] for k in ks], rotation=40, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=2)
        for sp, lab in (("sub200", "first split"), ("holdout200", "held-out")):
            cs = [c for c, k in zip(centers, ks) if k[1] == sp]
            if cs:
                ax.text(float(np.mean(cs)), 0.585, lab, ha="center", va="top", fontsize=7, fontstyle="italic", color="#555555")
        mid = [c for c, k in zip(centers, ks) if k[1] != "sub200"]
        if mid:
            ax.axvline(mid[0] - 0.72, color="#cccccc", lw=0.5, ls=":", zorder=0)
        ax.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
        ax.set_ylim(-0.12, 0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("aligned minus independent", fontsize=7.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.53, 1.0), handlelength=1.2,
               columnspacing=1.6, borderaxespad=0.2)
    fig.savefig(FIG / "fig_e2e_swap.pdf", metadata={"Creator": "eval/e2e_qa_figures.py"})
    fig.savefig(FIG / "fig_e2e_swap.png", dpi=300)
    plt.close(fig)

    dc = dose_curves(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(COLUMNWIDTH_IN, 2.2))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.905, bottom=0.355, wspace=0.32)
    colors = {"llama3_1_8b": "C0", "qwen2_5_14b": "C1"}
    mk = {"ms": 3.0, "lw": 0.9}
    for t in tags:
        fc = dc[t]["f"]
        if fc:
            axes[0].plot([p[0] for p in fc], [p[1] for p in fc], "k-o", **mk)
            axes[0].plot([p[0] for p in fc], [p[2] for p in fc], "-s", color=colors.get(t), **mk)
        for sy, ls in (("audited", "-"), ("ppr", "--")):
            qc = dc[t]["q"].get((sy, "sub200"))
            if qc:
                axes[1].plot([p[0] for p in qc], [p[1] for p in qc], "k" + ls + "o", **mk)
                axes[1].plot([p[0] for p in qc], [p[2] for p in qc], ls + "s", color=colors.get(t), **mk)
    axes[0].set_xlabel("edge fraction $f$\nof the aligned KG", fontsize=7, labelpad=1.5)
    axes[1].set_xlabel("gold mapping fraction $q$\non grafted inventory", fontsize=7, labelpad=1.5)
    for ax in axes:
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_title("hybrid retriever", fontsize=7, pad=2)
    axes[1].set_title("hybrid and PPR", fontsize=7, pad=2)
    fig.text(0.5, 0.995, "2WikiMultiHop, first split", ha="center", va="top", fontsize=7.5)
    metric = [Line2D([], [], color="k", marker="o", label="hit@5", **mk)]
    metric += [Line2D([], [], color=colors.get(t), marker="s", label=f"EM, {READER_LABEL.get(t, t)}", **mk) for t in tags]
    style = [Line2D([], [], color="#555555", ls="-", lw=0.9, label="hybrid"),
             Line2D([], [], color="#555555", ls="--", lw=0.9, label="PPR")]
    # legend rows: metrics (hit@5, EM per reader) | retriever line styles; matplotlib fills columns first
    handles = [h for pair in zip(metric, style + [None] * len(metric)) for h in pair if h is not None]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.8,
               columnspacing=1.0, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)
    fig.savefig(FIG / "fig_e2e_dose.pdf", metadata={"Creator": "eval/e2e_qa_figures.py"})
    fig.savefig(FIG / "fig_e2e_dose.png", dpi=300)
    plt.close(fig)
    json.dump({t: {"f": v["f"], "q": {f"{k[0]}|{k[1]}": pts for k, pts in v["q"].items()}} for t, v in dc.items()},
              open(TAB / "dose_curves.json", "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3_1_8b,qwen2_5_14b")
    args = ap.parse_args()
    tags = [t for t in args.models.split(",") if (RES / t / "summary.json").exists()]
    summaries = {t: load(t)[0] for t in tags}
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    for m in ("em", "f1", "cover"):
        write_swap_table(paired_table(summaries, m), m, tags)
    decirc_table(summaries)
    plot(summaries, tags)
    print("models:", tags, "-> tables/tab_e2e_swap_*.md, tables/tab_decirc.md, figures/fig_e2e_*.pdf")


if __name__ == "__main__":
    main()
