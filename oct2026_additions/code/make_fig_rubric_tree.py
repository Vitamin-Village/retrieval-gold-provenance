"""Figure 3 (fig:rubric): the five-bin provenance-circularity decision tree, applied per gold-unit axis.

Draws the figure as a vector figure at its final size, \\columnwidth = 219.09 pt (3.03 in), so every
text item is printed at its nominal size (7 pt; no sub- or superscripts). The node texts follow Step 2 of the rubric box
(manuscript, Appendix "Full Rubric Text"): bin 2 is gold derived from K or A by auto-generation and
then human-validated; the bins are labeled with the ordinal estimate \\Chat; the lowest bin reads
"independent of K and A".

Outputs: figures/fig2_rubric_tree_new.pdf (vector, TrueType fonts) and figures/fig2_rubric_tree_new.png
(600 dpi, same drawing). Usage (from paper/): python figures/make_fig_rubric_tree.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig2_rubric_tree_new"

W_PT = 219.09                      # \columnwidth of acl.sty (7.7 cm)
FS = 7.0                           # every text item
LINE = 8.2                         # line pitch for 7 pt text
KG, ANS, CH = r"$\mathcal{K}$", r"$A$", r"$\widehat{\mathrm{C}}$"

C_EDGE, C_ARROW, C_TEXT, C_Q = "#4D4D4D", "#4D4D4D", "#1A1A1A", "#EFEFEF"
# Bin fills: vermillion (bin 4) -> amber -> pale yellow -> light green -> bluish-green tint (bin 0),
# the circular / independent hues of Figures 1, 2, and 4.
BIN_FILL = {4: "#D55E00", 3: "#F3B27A", 2: "#FBE7A6", 1: "#CBE8C2", 0: "#A6E3CF"}
BIN_EDGE = {4: "#A34700", 3: "#C9711F", 2: "#C9A227", 1: "#5E9F55", 0: "#00714F"}
BIN_TEXT = {4: "#FFFFFF", 3: C_TEXT, 2: C_TEXT, 1: C_TEXT, 0: C_TEXT}
BIN_VALUE = {4: "1.0", 3: "0.75", 2: "0.50", 1: "0.25", 0: "0.0"}

# Questions, top to bottom; "Yes" leads to the bin on the right (rubric Step 2).
QUESTIONS = [
    (4, "Copied directly from the evaluation\nKG " + KG + ", or selected solely because\nit contains the answer string?"),
    (3, "Auto-generated from KG triples or\ntemplates, or evidence retained by\nanswer-string distant supervision?"),
    (2, "Gold derived from " + KG + " or " + ANS + " by\nauto-generation and then human-validated?"),
    (1, "Human or text gold with KG-guided\nor answer-guided selection?"),
]
BIN0_EXTRA = ("independent", "of " + KG + " and " + ANS)
CORRECTIONS = ("Corrections: KG augmentation, human refinement, and\n"
               "anti-leakage filtering each move the bin one step lower\n"
               "(at most twice, never below bin 0).")

QX0, QX1 = 3.0, 134.0              # question column
BX0, BX1 = 146.0, 216.0            # bin column
GAP_V = 11.0                       # vertical gap between question boxes (holds the "No" label)
PAD_V = 3.2


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.7, ls="-", r=2.2, z=3):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=1.9,head_width=1.0",
                                 color=C_ARROW, lw=0.8, shrinkA=0, shrinkB=0, zorder=2, mutation_scale=2.2))


def text(ax, x, y, s, ha="center", va="center", color=C_TEXT, weight="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=FS, color=color, fontweight=weight, linespacing=1.12, zorder=5)


def bin_height(n_lines: int) -> float:
    return n_lines * LINE + 2 * PAD_V - 1.0


def bin_box(ax, b, yc, extra: tuple[str, ...] = ()):
    n = 2 + len(extra)
    h = bin_height(n)
    rbox(ax, BX0, yc - h / 2, BX1, yc + h / 2, BIN_FILL[b], BIN_EDGE[b], lw=0.8)
    cx = (BX0 + BX1) / 2
    lines = [(f"Bin {b}", "bold"), (CH + f" = {BIN_VALUE[b]}", "normal")] + [(e, "normal") for e in extra]
    top = yc + (n - 1) * LINE / 2
    for i, (s, w) in enumerate(lines):
        text(ax, cx, top - i * LINE, s, color=BIN_TEXT[b], weight=w)


def layout():
    """Drawing height and the vertical extent of every node, as distances from the top."""
    y = 3.0
    start = (y, y + 13.0)
    y = start[1] + 9.0
    qs = []
    for b, q in QUESTIONS:
        h = (q.count("\n") + 1) * LINE + 2 * PAD_V
        qs.append((b, q, y, y + h))
        y += h + GAP_V
    h0 = bin_height(2 + len(BIN0_EXTRA))
    bin0 = qs[-1][3] + 6.0 + h0 / 2              # centre of the bin-0 box, below the last question
    y = bin0 + h0 / 2 + 6.0
    corr = (y, y + (CORRECTIONS.count("\n") + 1) * LINE + 2 * PAD_V)
    return corr[1] + 3.0, start, qs, bin0, corr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="output stem (default: figures/fig2_rubric_tree_new)")
    out = Path(ap.parse_args().out)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    H, start, qs, bin0, corr = layout()
    Y = lambda d: H - d  # noqa: E731  (distance from the top -> y coordinate)
    fig = plt.figure(figsize=(W_PT / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(0, H)
    ax.axis("off")
    qcx = (QX0 + QX1) / 2

    rbox(ax, qcx - 38, Y(start[1]), qcx + 38, Y(start[0]), "#FFFFFF", C_EDGE, lw=0.8)
    text(ax, qcx, Y((start[0] + start[1]) / 2), "Each gold-unit axis", weight="bold")
    arrow(ax, (qcx, Y(start[1])), (qcx, Y(qs[0][2])))

    for i, (b, q, top, bot) in enumerate(qs):
        yc = Y((top + bot) / 2)
        rbox(ax, QX0, Y(bot), QX1, Y(top), C_Q, C_EDGE, lw=0.7, r=3.0)
        text(ax, qcx, yc, q)
        arrow(ax, (QX1, yc), (BX0, yc))
        text(ax, (QX1 + BX0) / 2, yc + 1.4, "Yes", va="bottom")
        bin_box(ax, b, yc)
        if i + 1 < len(qs):
            arrow(ax, (qcx, Y(bot)), (qcx, Y(qs[i + 1][2])))
            text(ax, qcx + 3.0, Y(bot + GAP_V / 2), "No", ha="left")

    # last "No" -> bin 0
    y0, y_last = Y(bin0), Y(qs[-1][3])
    ax.plot([qcx, qcx], [y_last, y0], color=C_ARROW, lw=0.8, solid_capstyle="projecting", zorder=2)
    arrow(ax, (qcx, y0), (BX0, y0))
    text(ax, qcx + 3.0, (y_last + y0) / 2, "No", ha="left")
    bin_box(ax, 0, y0, extra=BIN0_EXTRA)

    # rubric Step 3
    rbox(ax, QX0, Y(corr[1]), BX1, Y(corr[0]), "#FFFFFF", "#8C8C8C", lw=0.7, ls=(0, (3, 2)), r=2.0)
    head, rest = CORRECTIONS.split(":", 1)
    text(ax, QX0 + 4.0, Y((corr[0] + corr[1]) / 2), r"$\mathbf{" + head + "}$:" + rest, ha="left")

    meta = {"Creator": "figures/make_fig_rubric_tree.py"}
    fig.savefig(out.with_suffix(".pdf"), metadata=meta)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')} and .png ({W_PT:.2f} x {H:.1f} pt)")


if __name__ == "__main__":
    main()
