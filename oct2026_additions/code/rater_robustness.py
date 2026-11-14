"""Do the census conclusions depend on who applied the rubric?

LM raters could add a third layer of error. This script recomputes
the census-level conclusions under every rater that scored the axes, one rater at a time:

* 29-axis raters: the adjudicated census, the two primary LM raters (A, B), the non-author human
  re-scoring, the three human raters of the v4 study and their median (human_validation/v4), and four logged LM
  raters (Claude Opus 5.5, Claude Sonnet 5.5, Gemini 3.8 Flash, median of three runs each; Claude
  Haiku 4.5, one run) that applied the full step procedure (eval/llm_census_logged.py);
* headline-axis raters: the five cross-model LMs, which scored each benchmark's headline
  axis only (for TriviaQA and Natural Questions the headline axis is answer-surface, so
  their KG-derived value is 0 by construction, as in Table 1).

For each rater: number of the eleven external benchmarks with KG-derived C_K >= 0.75
(bin >= 3) and >= 0.5 (bin >= 2), the flagged set, whether the benchmarks used in the
causal swaps are flagged, and the descriptive Spearman rho between C_K and the published
graph-over-lexical gain medians (Appendix Table "scatter", GraphRAG-Bench excluded as there).
The causal swap results do not use any rater and are unaffected by construction.

Usage: python analysis/rater_robustness.py   (from paper/)
Output: analysis/rater_robustness_summary.{md,json}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
HV = ROOT / "human_validation"

KG_AXES = {  # benchmark -> KG-derived axes (mechanism column of Appendix Table fullcensus)
    "GRADE": ["GRADE_EDGE"], "STaRK": ["STARK_ANSWERENTITY"], "GraphRAG-Bench": ["GRB_SUBGRAPH"],
    "2WikiMultiHop": ["TWOWIKI_SENTENCE", "TWOWIKI_EDGE"], "MetaQA": ["METAQA_ANSWER", "METAQA_PATH"],
    "WebQSP": ["WEBQSP_ANSWERENTITY", "WEBQSP_PATH"],
    "MuSiQue": [], "HotpotQA": [], "MultiHop-RAG": [], "Natural Questions": [], "TriviaQA": [],
}
EXTERNAL = list(KG_AXES)
CAUSAL = ["2WikiMultiHop", "MetaQA", "GraphRAG-Bench", "STaRK"]
# published same-paper gain medians (pp), Appendix Table scatter; STaRK has three sub-benchmarks
GAINS = [("MuSiQue", 19.6), ("HotpotQA", 5.5), ("MultiHop-RAG", 1.9), ("Natural Questions", 4.8),
         ("TriviaQA", 1.0), ("STaRK", 7.3), ("STaRK", 9.7), ("STaRK", 23.7),
         ("2WikiMultiHop", 25.1), ("MetaQA", 28.2), ("WebQSP", 41.5)]

LM_A_B = {  # analysis/_census_kappa_ci.py ROWS (primary LM raters A, B)
    "MUSIQUE_PASSAGE": (0, 0), "MUSIQUE_ANSWER": (0, 0), "HOTPOT_SENTENCE": (0, 0), "HOTPOT_ANSWER": (0, 0),
    "TWOWIKI_EDGE": (4, 4), "TWOWIKI_SENTENCE": (3, 3), "TWOWIKI_ANSWER": (4, 3),
    "GRB_SUBGRAPH": (3, 3), "GRB_ANSWER": (3, 3), "GRADE_EDGE": (3, 3), "GRADE_PASSAGE": (1, 1),
    "OURSUP_EDGE": (4, 4), "OURSUP_ENTITY": (4, 4), "OURSUP_PATH": (4, 4), "OURSUP_ANSWER": (4, 3),
    "OURSLOW_PASSAGE": (0, 0), "OURSLOW_ANSWER": (0, 0), "MHRAG_EVIDENCE": (0, 0), "MHRAG_ANSWER": (0, 0),
    "STARK_ANSWERENTITY": (3, 3), "STARK_TEXTUAL": (1, 1), "METAQA_ANSWER": (4, 4), "METAQA_PATH": (4, 4),
    "NQ_PASSAGE": (1, 1), "NQ_ANSWER": (0, 1), "TRIVIA_EVIDENCE": (4, 3), "TRIVIA_ANSWER": (0, 0),
    "WEBQSP_ANSWERENTITY": (4, 4), "WEBQSP_PATH": (4, 4),
}
HEADLINE_NAME = {"Natural Questions": "Natural Questions", "TriviaQA": "TriviaQA"}


def load_axis_raters():
    raters = {}
    raters["census (adjudicated LM)"] = json.load(open(HV / "reference_bins.json", encoding="utf-8"))["reference"]
    raters["primary LM rater A"] = {k: v[0] for k, v in LM_A_B.items()}
    raters["primary LM rater B"] = {k: v[1] for k, v in LM_A_B.items()}
    # Human re-scoring: the non-author rater.
    d = json.load(open(HV / "external_rater_result.json", encoding="utf-8"))
    raters["human: non-author"] = {x["item_id"]: x["external"] for x in d["detail"]}
    # Logged LM raters that applied the full step procedure to all 29 axes (eval/llm_census_logged.py):
    # median of three runs per model, workbook condition; claude-haiku-4-5 enters as its single complete
    # workbook run, as the protocol's deviation note specifies.
    med = json.load(open(HV / "llm_census_logged/result.json", encoding="utf-8"))["per_model_median"]
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "gemini-3.8-flash", "claude-haiku-4-5"):
        raters[f"LM logged: {model}"] = med[f"{model}_A"]["bins"]
    # Human raters of the v4 study (human_validation/v4/PROTOCOL_v4.md), each rater separately, and their item-wise median.
    humans = load_v4_humans(set(raters["census (adjudicated LM)"]))
    for c, bins in humans.items():
        raters[f"human v4: {c}"] = bins
    if len(humans) >= 3:
        items = sorted(next(iter(humans.values())))
        raters["human v4: median"] = {i: sorted(h[i] for h in humans.values())[(len(humans) - 1) // 2] for i in items}
    return raters


def load_v4_humans(items: set[str]) -> dict[str, dict[str, int]]:
    """Final bins of the eligible v4 human raters on the census items, from the anonymized export
    (human_validation/v4/public/, written by export_public_v4.py from score_v4.py's result); {} if absent."""
    pub = HV / "v4" / "public"
    if not (pub / "ratings_v4.csv").exists():
        return {}
    eligible = [r["rater"] for r in csv.DictReader(open(pub / "raters_v4.csv", encoding="utf-8")) if r["eligible"] == "True"]
    humans: dict[str, dict[str, int]] = {c: {} for c in eligible}
    for r in csv.DictReader(open(pub / "ratings_v4.csv", encoding="utf-8")):
        if r["rater"] in humans and r["item_id"] in items:
            humans[r["rater"]][r["item_id"]] = int(r["final_bin_0_4"])
    assert all(set(h) == items for h in humans.values()), "incomplete v4 ratings"
    return humans


def kg_bins_axis(r):
    return {b: (max(r[a] for a in axes) if axes else 0) for b, axes in KG_AXES.items()}


def kg_bins_headline(model_bins):
    out = {}
    for b in EXTERNAL:
        if not KG_AXES[b]:
            out[b] = 0
        else:
            out[b] = model_bins[b]
    return out


def summarize(name, kg, kind):
    ge3 = [b for b in EXTERNAL if kg[b] >= 3]
    ge2 = [b for b in EXTERNAL if kg[b] >= 2]
    x = [kg[b] / 4 for b, _ in GAINS]
    y = [g for _, g in GAINS]
    rho = float(spearmanr(x, y).correlation) if len(set(x)) > 1 else float("nan")
    return {"rater": name, "kind": kind, "n_ge_0.75": len(ge3), "n_ge_0.5": len(ge2), "flagged_ge_0.75": ge3,
            "causal_flagged_ge_0.5": [b for b in CAUSAL if kg[b] >= 2],
            "text_grounded_flagged": [b for b in ("MuSiQue", "HotpotQA", "MultiHop-RAG", "Natural Questions", "TriviaQA") if kg[b] >= 2],
            "spearman_gain_medians": rho, "kg_bins": kg}


def main():
    rows = []
    for name, r in load_axis_raters().items():
        rows.append(summarize(name, kg_bins_axis(r), "29 axes"))
    mm = json.load(open(ROOT / "results/multimodel_census_raw.json", encoding="utf-8"))["raters"]
    for model, bins in mm.items():
        rows.append(summarize(f"LM: {model.split(':', 1)[-1] if model.startswith('claude') else model}", kg_bins_headline(bins), "headline axis"))
    ge3 = [r["n_ge_0.75"] for r in rows]
    always = [b for b in EXTERNAL if all(b in r["flagged_ge_0.75"] for r in rows)]
    always2 = [b for b in EXTERNAL if all(r["kg_bins"][b] >= 2 for r in rows)]
    never = [b for b in EXTERNAL if all(r["kg_bins"][b] < 2 for r in rows)]
    rhos = [r["spearman_gain_medians"] for r in rows]
    agg = {"n_raters": len(rows), "n_ge_0.75_range": [min(ge3), max(ge3)],
           "flagged_ge_0.75_by_every_rater": always, "flagged_ge_0.5_by_every_rater": always2,
           "never_flagged_ge_0.5": never, "spearman_range": [float(np.nanmin(rhos)), float(np.nanmax(rhos))]}
    json.dump({"aggregate": agg, "raters": rows}, open(ROOT / "analysis/rater_robustness_summary.json", "w", encoding="utf-8"), indent=1)
    md = ["# Census conclusions under each rater", "",
          f"Raters: {len(rows)}. External benchmarks with C_K >= 0.75: {min(ge3)}-{max(ge3)}. "
          f"Flagged >= 0.75 by every rater: {', '.join(always)}. Flagged >= 0.5 by every rater: {', '.join(always2)}. "
          f"Never flagged >= 0.5: {', '.join(never)}. Spearman rho (C_K vs published gain medians): "
          f"{agg['spearman_range'][0]:.2f}-{agg['spearman_range'][1]:.2f}.", "",
          "| rater | scope | # C_K>=0.75 | # C_K>=0.5 | causal benchmarks >=0.5 | text-grounded flagged | rho |",
          "|---|---|---:|---:|---|---|---:|"]
    for r in rows:
        md.append(f"| {r['rater']} | {r['kind']} | {r['n_ge_0.75']} | {r['n_ge_0.5']} | {', '.join(r['causal_flagged_ge_0.5'])} | "
                  f"{', '.join(r['text_grounded_flagged']) or '-'} | {r['spearman_gain_medians']:.2f} |")
    md += ["", "Per-rater KG-derived bins:", "", "| rater | " + " | ".join(EXTERNAL) + " |", "|---|" + "---:|" * len(EXTERNAL)]
    for r in rows:
        md.append(f"| {r['rater']} | " + " | ".join(str(r["kg_bins"][b]) for b in EXTERNAL) + " |")
    (ROOT / "analysis/rater_robustness_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
