"""Figure 1 (fig:circularity): retrieval-gold provenance, circular vs independent setting.

Draws the figure as a vector figure at its final size, \\columnwidth = 219.09 pt (3.03 in) in the
ACL two-column layout, so every text item is printed at its nominal size (>= 7 pt).

  (a) circular setting:     the gold is built from the same KG K the retriever ranks over.
  (b) independent setting:  the gold is built from source text or human annotation.

Outputs: figures/fig1_provenance.pdf (vector, TrueType fonts) and figures/fig1_provenance.png
(600 dpi, same drawing). Usage (from paper/): python figures/make_fig1_provenance.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig1_provenance"

# Final size in points: \columnwidth of acl.sty (7.7 cm) and a height below 122 pt.
W_PT, H_PT = 219.09, 112.0
FS_BOX, FS_LABEL, FS_TITLE = 7.5, 7.0, 7.5

# Okabe-Ito vermillion / bluish green: the circular / independent colors of Figures 2 and 4.
C_CIRC, C_INDEP = "#D55E00", "#009E73"
C_CIRC_DARK, C_INDEP_DARK = "#A34700", "#00714F"
BG_CIRC, BG_INDEP = "#FBE9E7", "#E8F8F0"
C_BOX_FACE, C_BOX_EDGE, C_ARROW, C_TEXT = "#FFFFFF", "#4D4D4D", "#4D4D4D", "#1A1A1A"
KG = r"$\mathcal{K}$"

PANEL_W, GAP = 107.5, 4.09
BOX_H = 13.0
ROW_Y = (80.5, 55.0, 29.5)          # box centres: corpus, KG, retriever (left column)
LEFT_CX, LEFT_W = 19.5, 32.0         # left column centre and box width, panel coordinates
RIGHT_CX, RIGHT_W = 92.5, 22.0       # right column centre and box width


def box(ax, cx, cy, w, text, edge=C_BOX_EDGE, lw=0.7, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - BOX_H / 2), w, BOX_H, boxstyle="round,pad=0,rounding_size=2.2",
                                fc=C_BOX_FACE, ec=edge, lw=lw, zorder=3))
    ax.text(cx, cy - 0.3, text, ha="center", va="center", fontsize=FS_BOX, color=C_TEXT, fontweight=weight, zorder=4)


def arrow(ax, p0, p1, color=C_ARROW, lw=0.8, conn="arc3", head=4.2):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=conn, arrowstyle=f"-|>,head_length={head / 2.2:.2f},head_width={head / 4.2:.2f}",
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=2, mutation_scale=2.2))


def label(ax, x, y, text, ha="left", va="center", color=C_TEXT, style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=FS_LABEL, color=color, fontstyle=style, zorder=5, linespacing=1.0)


def panel(ax, x0, circular: bool):
    bg = BG_CIRC if circular else BG_INDEP
    accent = C_CIRC if circular else C_INDEP
    dark = C_CIRC_DARK if circular else C_INDEP_DARK
    ax.add_patch(FancyBboxPatch((x0, 0.6), PANEL_W, H_PT - 1.2, boxstyle="round,pad=0,rounding_size=3",
                                fc=bg, ec="none", zorder=0))
    ax.text(x0 + 4.0, H_PT - 7.0, "(a) Circular setting" if circular else "(b) Independent setting",
            ha="left", va="center", fontsize=FS_TITLE, fontweight="bold", color=dark)

    lx, rx = x0 + LEFT_CX, x0 + RIGHT_CX
    y_corpus, y_kg, y_ret = ROW_Y
    y_gold, y_eval = y_kg, y_ret
    half = BOX_H / 2

    # Shared pipeline: corpus -> KG -> retriever -> eval
    box(ax, lx, y_corpus, LEFT_W, "Corpus")
    box(ax, lx, y_kg, LEFT_W, "KG " + KG, edge=accent if circular else C_BOX_EDGE, lw=1.1 if circular else 0.7)
    box(ax, lx, y_ret, LEFT_W, "Retriever")
    box(ax, rx, y_gold, RIGHT_W, "Gold", edge=accent, lw=1.1)
    box(ax, rx, y_eval, RIGHT_W, "Eval")
    arrow(ax, (lx, y_corpus - half), (lx, y_kg + half))
    label(ax, lx + 3.0, (y_corpus + y_kg) / 2, "build")
    arrow(ax, (lx, y_kg - half), (lx, y_ret + half), color=accent if circular else C_ARROW, lw=1.1 if circular else 0.8)
    label(ax, lx + 3.0, (y_kg + y_ret) / 2, "ranks over " + KG)
    arrow(ax, (lx + LEFT_W / 2, y_ret), (rx - RIGHT_W / 2, y_eval))
    label(ax, (lx + LEFT_W / 2 + rx - RIGHT_W / 2) / 2, y_ret - 2.0, "top-$k$", ha="center", va="top")
    arrow(ax, (rx, y_gold - half), (rx, y_eval + half))

    if circular:
        # gold built from the same K
        arrow(ax, (lx + LEFT_W / 2, y_kg), (rx - RIGHT_W / 2, y_gold), color=accent, lw=1.3)
        label(ax, (lx + LEFT_W / 2 + rx - RIGHT_W / 2) / 2, y_kg + 2.0, "gold built\nfrom " + KG,
              ha="center", va="bottom", color=dark)
        note = "retriever and gold share " + KG
    else:
        # gold built from source text or human annotation, without K
        arrow(ax, (lx + LEFT_W / 2, y_corpus), (rx, y_gold + half), color=accent, lw=1.3, conn="angle,angleA=0,angleB=90,rad=3")
        label(ax, rx - 2.5, y_corpus + 1.8, "source text or\nhuman annotation", ha="right", va="bottom", color=dark)
        note = "gold built without " + KG
    ax.text(x0 + PANEL_W / 2, 9.5, note, ha="center", va="center", fontsize=FS_LABEL, color=dark, fontstyle="italic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="output stem (default: figures/fig1_provenance)")
    out = Path(ap.parse_args().out)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(W_PT / 72, H_PT / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(0, H_PT)
    ax.axis("off")
    panel(ax, 0.0, circular=True)
    panel(ax, PANEL_W + GAP, circular=False)
    meta = {"Creator": "figures/make_fig1_provenance.py"}
    fig.savefig(out.with_suffix(".pdf"), metadata=meta)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')} and .png ({W_PT:.2f} x {H_PT:.1f} pt)")


if __name__ == "__main__":
    main()
