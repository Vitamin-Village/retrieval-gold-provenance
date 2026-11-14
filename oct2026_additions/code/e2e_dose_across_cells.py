"""Does the answer-level swap follow the retrieval-level swap across every cell?

Collects, for every (benchmark, split, retriever, reader) cell of the end-to-end experiment
(frozen runs, BGE-M3 reruns, and HippoRAG 2 where available), the paired hit@5 swap and the
paired answer-accuracy swap (EM; judge accuracy on GraphRAG-Bench), and the reader's headroom
(gold-only minus BM25 answer accuracy). Fits answer_swap = b * hit_swap (no intercept) overall
and per benchmark, and reports Spearman/Pearson correlations. Output:
analysis/e2e_dose_across_cells.{json,md}, figures/fig_e2e_cells.pdf
Usage (from paper/): python analysis/e2e_dose_across_cells.py
  --res results/e2e_qa_cls --suffix _cls  reads another results root (e.g. the [CLS]-pooled BGE-M3
  rerun) and writes analysis/e2e_dose_across_cells_cls.{json,md}, figures/fig_e2e_cells_cls.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--res", default=str(ROOT / "results" / "e2e_qa"))
_ap.add_argument("--suffix", default="")
_args = _ap.parse_args()
RES = Path(_args.res)
SFX = _args.suffix
READERS = {"llama3_1_8b": "Llama-3.1-8B", "qwen2_5_14b": "Qwen2.5-14B", "mistral_small_24b": "Mistral-Small-24B"}
BENCH = {"2wiki": "2WikiMultiHop", "metaqa": "MetaQA", "stark": "STaRK-Prime", "graphragbench": "GraphRAG-Bench"}

cells = []
for tag in READERS:
    s = json.load(open(RES / tag / "summary.json", encoding="utf-8"))
    ref = {(c["bench"], c["split"], c["system"], c["condition"]): c for c in s["cells"]}
    jpath = RES / tag / "judge_mistral_small_24b.json"
    judge = json.load(open(jpath, encoding="utf-8")) if jpath.exists() else {"paired": [], "cells": []}
    jp = {(e["bench"], e["split"], e["system"]): e for e in judge["paired"]}
    jc = {(c["bench"], c["split"], c["system"], c["condition"]): c["judge_acc"] for c in judge["cells"]}
    for p in s["paired"]:
        b, sp, sy = p["bench"], p["split"], p["system"]
        if b not in BENCH or sy.startswith("dense"):
            continue
        hit = p["mediation"]["retrieval_hit_diff"]
        if b == "graphragbench":
            e = jp.get((b, sp, sy))
            if not e:
                continue
            ans, pval = e["mean_paired_diff"], e["sign_test_p"]
            gold, bm = jc.get((b, sp, "none", "oracle")), jc.get((b, sp, "bm25", "none"))
        else:
            ans, pval = p["em"]["mean_paired_diff"], p["em"]["sign_test_p"]
            gold = ref.get((b, sp, "none", "oracle"), {}).get("em")
            bm = ref.get((b, sp, "bm25", "none"), {}).get("em")
        cells.append({"reader": tag, "bench": b, "split": sp, "system": sy, "hit_swap": hit, "ans_swap": ans, "p": pval,
                      "headroom": (gold - bm) if gold is not None and bm is not None else None})


def fit(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    b = float((xs * ys).sum() / (xs * xs).sum())
    return b, float(spearmanr(xs, ys).correlation), float(pearsonr(xs, ys)[0])


out = {"n_cells": len(cells), "cells": cells}
b, rs, rp = fit([c["hit_swap"] for c in cells], [c["ans_swap"] for c in cells])
out["all"] = {"slope": b, "spearman": rs, "pearson": rp}
per = {}
for bn in BENCH:
    sub = [c for c in cells if c["bench"] == bn]
    if len(sub) >= 4:
        bb, ss, pp = fit([c["hit_swap"] for c in sub], [c["ans_swap"] for c in sub])
        per[bn] = {"n": len(sub), "slope": bb, "spearman": ss, "pearson": pp,
                   "mean_hit_swap": float(np.mean([c["hit_swap"] for c in sub])), "mean_ans_swap": float(np.mean([c["ans_swap"] for c in sub])),
                   "mean_headroom": float(np.mean([c["headroom"] for c in sub if c["headroom"] is not None]))}
out["per_bench"] = per
# does a single slope predict the null benchmarks? predicted answer swap for MetaQA/STaRK cells from the 2Wiki+GRB slope
fitset = [c for c in cells if c["bench"] in ("2wiki", "graphragbench")]
b2, _, _ = fit([c["hit_swap"] for c in fitset], [c["ans_swap"] for c in fitset])
for bn in ("metaqa", "stark"):
    sub = [c for c in cells if c["bench"] == bn]
    pred = [b2 * c["hit_swap"] for c in sub]
    obs = [c["ans_swap"] for c in sub]
    out.setdefault("transfer", {})[bn] = {"slope_from_2wiki_grb": b2, "mean_predicted": float(np.mean(pred)), "mean_observed": float(np.mean(obs)),
                                          "max_abs_predicted": float(np.max(np.abs(pred)))}
# headroom-aware model: answer_swap = g * hit_swap * max(headroom, 0), no intercept
hc = [c for c in cells if c["headroom"] is not None]
xh = np.array([c["hit_swap"] * max(c["headroom"], 0) for c in hc])
yh = np.array([c["ans_swap"] for c in hc])
xo = np.array([c["hit_swap"] for c in hc])
g = float((xh * yh).sum() / (xh * xh).sum())
bo = float((xo * yh).sum() / (xo * xo).sum())
r2 = lambda pred: float(1 - ((yh - pred) ** 2).sum() / ((yh - yh.mean()) ** 2).sum())
out["headroom_model"] = {"n": len(hc), "gain": g, "spearman": float(spearmanr(xh, yh).correlation), "pearson": float(pearsonr(xh, yh)[0]),
                         "r2": r2(g * xh), "r2_hit_only": r2(bo * xo),
                         "per_bench": {bn: {"predicted": float(np.mean([g * c["hit_swap"] * max(c["headroom"], 0) for c in hc if c["bench"] == bn])),
                                            "observed": float(np.mean([c["ans_swap"] for c in hc if c["bench"] == bn]))} for bn in BENCH}}
json.dump(out, open(ROOT / f"analysis/e2e_dose_across_cells{SFX}.json", "w", encoding="utf-8"), indent=1)
md = [f"# Answer swap vs retrieval swap across {len(cells)} cells", "",
      f"All cells: slope {out['all']['slope']:.2f}, Spearman {out['all']['spearman']:.2f}, Pearson {out['all']['pearson']:.2f}", "",
      "| benchmark | n | mean hit@5 swap | mean answer swap | slope | Spearman | mean headroom (gold - BM25) |", "|---|---|---|---|---|---|---|"]
for bn, e in per.items():
    md.append(f"| {BENCH[bn]} | {e['n']} | {e['mean_hit_swap']:+.3f} | {e['mean_ans_swap']:+.3f} | {e['slope']:.2f} | {e['spearman']:.2f} | {e['mean_headroom']:+.3f} |")
md += ["", "Transfer (slope fitted on 2WikiMultiHop + GraphRAG-Bench):"]
for bn, e in out.get("transfer", {}).items():
    md.append(f"- {BENCH[bn]}: slope {e['slope_from_2wiki_grb']:.2f}, predicted mean answer swap {e['mean_predicted']:+.3f}, observed {e['mean_observed']:+.3f}")
hm = out["headroom_model"]
md += ["", f"Headroom model (answer swap = g x hit swap x headroom): g {hm['gain']:.2f}, Spearman {hm['spearman']:.2f}, R2 {hm['r2']:.2f} (hit swap alone: R2 {hm['r2_hit_only']:.2f})"]
md += [f"- {BENCH[bn]}: predicted {e['predicted']:+.3f}, observed {e['observed']:+.3f}" for bn, e in hm["per_bench"].items()]
(ROOT / f"analysis/e2e_dose_across_cells{SFX}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
col = {"2wiki": "#1f77b4", "metaqa": "#2ca02c", "stark": "#9467bd", "graphragbench": "#d62728"}
# Drawn at its final width in the manuscript (0.82\textwidth = 373.3 pt), so font sizes are printed sizes (>= 7 pt).
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
fig, axes = plt.subplots(1, 2, figsize=(0.82 * 455.24 / 72, 2.15), sharey=True)
for k, ax in enumerate(axes):
    for bn in BENCH:
        sub = [c for c in hc if c["bench"] == bn]
        xv = [c["hit_swap"] if k == 0 else c["hit_swap"] * max(c["headroom"], 0) for c in sub]
        ax.scatter(xv, [c["ans_swap"] for c in sub], s=12, c=col[bn], label=BENCH[bn], alpha=0.8)
    xs = np.linspace(-0.05, 0.52 if k == 0 else 0.26, 10)
    ax.plot(xs, (bo if k == 0 else g) * xs, "k--", lw=0.8)
    ax.axhline(0, c="grey", lw=0.5)
    ax.axvline(0, c="grey", lw=0.5)
    ax.set_xlabel("hit@5 swap" if k == 0 else "hit@5 swap × reader headroom", fontsize=7.5)
    ax.set_title(f"R² = {(hm['r2_hit_only'] if k == 0 else hm['r2']):.2f}", fontsize=7.5, pad=3)
    ax.tick_params(labelsize=7)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axes[0].set_ylabel("answer-accuracy swap", fontsize=7.5)
axes[0].legend(fontsize=7, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(ROOT / f"figures/fig_e2e_cells{SFX}.pdf", metadata={"Creator": "analysis/e2e_dose_across_cells.py"})
print(f"wrote figures/fig_e2e_cells{SFX}.pdf")
