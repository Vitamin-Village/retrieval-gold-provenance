"""Generate Figure 4 (fig:census), the provenance-circularity census, at its final size.

The census values (headline and per-axis estimates) are those of the census tables of the
manuscript (Table 1 and the full per-axis census). The figure is included at
\\columnwidth (219.09 pt) in the appendix; the canvas is drawn at exactly that width, so every
text item is printed at its nominal size (>= 7 pt).

Labels and layout:
  * no in-figure title or subtitle, no footnote text (the caption carries them);
  * terms follow the manuscript: the bar is the headline \\Chat (maximum over gold-unit axes) and is
    colored by the mechanism of its highest axis, KG-derived (\\Chat_KG = \\Chat), derived from the
    answer surface A only (\\Chat_KG = 0, TriviaQA and Natural Questions), or independent (\\Chat = 0);
  * the anchors are labeled "Ours, upper anchor" / "Ours, lower anchor", as in Table 1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

Mechanism = Literal["kg", "answer", "independent"]


@dataclass(frozen=True)
class CensusRow:
    """One locked row of the provenance-circularity census."""

    label: str
    headline: float              # \Chat, maximum over all gold-unit axes
    axes: tuple[float, ...]      # per-axis estimates used for the hatched segment (unchanged)
    kg_headline: float           # \Chat_KG, maximum over the KG-derived axes (Table 1)
    is_anchor: bool = False

    @property
    def mechanism(self) -> Mechanism:
        if self.headline == 0:
            return "independent"
        return "kg" if abs(self.kg_headline - self.headline) < 1e-9 else "answer"


OUTPUT_STEM = Path(__file__).resolve().parent / "fig3_census"

WIDTH_PT = 219.09          # \columnwidth of acl.sty (7.7 cm); included at \columnwidth
HEIGHT_PT = 186.0
PNG_DPI = 600
FS = 7.0                   # every text item

COLOR: dict[Mechanism, str] = {"kg": "#D55E00", "answer": "#E69F00", "independent": "#009E73"}
LIGHT: dict[Mechanism, str] = {"kg": "#F0B79A", "answer": "#F4D79A", "independent": "#A6E3CF"}
LEGEND_LABEL: dict[Mechanism, str] = {
    "kg": "KG-derived",
    "answer": "answer-surface-derived",
    "independent": r"independent ($\widehat{\mathrm{C}}=0$)",
}

# Values identical to the 2026-09-29 figure; kg_headline from Table 1 (tab:census).
ROWS: tuple[CensusRow, ...] = (
    CensusRow("WebQSP", 1.00, (1.00,), 1.00),
    CensusRow("2WikiMultiHop", 1.00, (0.75, 1.00), 1.00),
    CensusRow("MetaQA", 1.00, (1.00,), 1.00),
    CensusRow("Ours, upper anchor", 1.00, (0.75, 1.00), 1.00, True),
    CensusRow("GraphRAG-Bench", 0.75, (0.75,), 0.75),
    CensusRow("STaRK", 0.75, (0.25, 0.75), 0.75),
    CensusRow("GRADE", 0.75, (0.25, 0.75), 0.75),
    CensusRow("TriviaQA", 0.75, (0.00, 0.75), 0.00),
    CensusRow("Natural Questions", 0.25, (0.00, 0.25), 0.00),
    CensusRow("MuSiQue", 0.00, (0.00,), 0.00),
    CensusRow("HotpotQA", 0.00, (0.00,), 0.00),
    CensusRow("MultiHop-RAG", 0.00, (0.00,), 0.00),
    CensusRow("Ours, lower anchor", 0.00, (0.00,), 0.00, True),
)
ANCHOR_EDGE = "#1A1A1A"
BAR_H = 0.62


def configure_typography() -> None:
    """Use embedded, Times-compatible serif fonts in raster and vector output."""

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FS,
        }
    )


def draw_rows(ax: Axes, y_positions: list[int]) -> None:
    """Draw the headline bars, the hatched span above a second, lower axis, and anchor outlines."""

    for row, y in zip(ROWS, y_positions):
        color, light = COLOR[row.mechanism], LIGHT[row.mechanism]
        lw = 1.1 if row.is_anchor else 0.6
        edge = ANCHOR_EDGE if row.is_anchor else "#333333"
        lower = [v for v in row.axes if abs(v - row.headline) > 1e-9]
        other = lower[0] if lower else None
        if row.headline > 0:
            ax.barh(y, row.headline, height=BAR_H, color=color, edgecolor=edge, linewidth=0.6, zorder=2)
            if other is not None and other < row.headline:
                ax.barh(y, row.headline - other, left=other, height=BAR_H, color=light, edgecolor=color,
                        linewidth=0.6, hatch="////", zorder=3)
            if row.is_anchor:
                ax.barh(y, row.headline, height=BAR_H, color="none", edgecolor=ANCHOR_EDGE, linewidth=lw,
                        linestyle=(0, (3, 1.5)), zorder=4)
        else:
            ax.plot([0, 0], [y - BAR_H / 2, y + BAR_H / 2], color=color, linewidth=2.6, solid_capstyle="butt", zorder=2)
            if row.is_anchor:
                ax.plot([0, 0], [y - BAR_H / 2, y + BAR_H / 2], color=ANCHOR_EDGE, linewidth=lw,
                        linestyle=(0, (3, 1.5)), zorder=3)
        ax.text(max(row.headline, 0.0) + 0.03, y, f"{row.headline:.2f}", va="center", ha="left", fontsize=FS,
                color="#222222", zorder=5)


def make_legend(fig: Figure) -> None:
    handles = [Patch(facecolor=COLOR[m], edgecolor="#333333", linewidth=0.6, label=LEGEND_LABEL[m])
               for m in ("kg", "answer", "independent")]
    handles += [
        Patch(facecolor=LIGHT["kg"], edgecolor=COLOR["kg"], hatch="////", linewidth=0.6,
              label="span above a lower second axis"),
        Patch(facecolor="white", edgecolor=ANCHOR_EDGE, linestyle=(0, (3, 1.5)), linewidth=1.1,
              label="our calibration anchors"),
    ]
    leg = fig.legend(handles=handles, loc="lower left", ncol=2, fontsize=FS,
                     bbox_to_anchor=(0.02, 0.0), frameon=False, handlelength=1.4, handleheight=0.9,
                     handletextpad=0.4, columnspacing=0.8, labelspacing=0.3, borderaxespad=0.15)
    return leg


def build_figure() -> Figure:
    """Build the complete figure without changing its locked data."""

    fig, ax = plt.subplots(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72))
    fig.subplots_adjust(left=0.305, right=0.985, top=0.985, bottom=0.285)

    y_positions = list(range(len(ROWS) - 1, -1, -1))
    draw_rows(ax, y_positions)
    # thin separators between the three headline levels (>= 0.75, 0.25, 0)
    for k in range(1, len(ROWS)):
        a, b = ROWS[k - 1].headline, ROWS[k].headline
        if (a >= 0.75) != (b >= 0.75) or (a > 0) != (b > 0):
            ax.axhline(y_positions[k] + 0.5, color="#BBBBBB", linewidth=0.5, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row.label for row in ROWS], fontsize=FS)
    for tick, row in zip(ax.get_yticklabels(), ROWS):
        if row.is_anchor:
            tick.set_fontstyle("italic")
    ax.set_xlim(-0.02, 1.17)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=FS)
    ax.set_xlabel(r"$\widehat{\mathrm{C}}$, maximum over gold-unit axes", fontsize=FS,
                  labelpad=2)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=0, pad=2)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E2E2E2", linewidth=0.5, zorder=0)
    make_legend(fig)
    return fig


def main() -> None:
    """Render synchronized vector and raster deliverables."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT_STEM), help="output stem (default: figures/fig3_census)")
    stem = Path(ap.parse_args().out)
    configure_typography()
    figure = build_figure()
    figure.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "figures/generate_fig3_census.py"})
    figure.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(figure)
    print(f"wrote: {stem.with_suffix('.pdf')}")
    print(f"wrote: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
