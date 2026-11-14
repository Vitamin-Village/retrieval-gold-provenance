"""Agreement of every census rater with the census and with the human raters (2026-10-01).

Human raters:
- the three human raters of the v4 study (human_validation/v4/PROTOCOL_v4.md), who rated all 29 axis
  judgments from verbatim construction excerpts alone; their eligible workbooks as scored by
  human_validation/v4/score_v4.py, read from the anonymized export human_validation/v4/public/, and their
  item-wise median;
- an earlier non-author rater who re-scored all 29 judgments from the short prompt and condensed
  descriptions, without seeing the census bins (human_validation/external_rater_result.json).

Raters compared: the census (adjudicated from primary LM raters A and B, so not independent of them), the
primary LM raters A and B (bins as released in rater_robustness.py), and the logged LM raters of
human_validation/llm_census_logged (model medians; workbook and excerpt-only conditions; Haiku 4.5 one
workbook run). Agreement is descriptive; the alternative annotator test is in result_v4.md.

Run from paper/:  python analysis/rater_validation.py
Writes analysis/rater_validation_summary.{md,json}.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

PAPER = Path(__file__).resolve().parents[1]
HV = PAPER / "human_validation"
OUT = PAPER / "analysis"
LOGGED = ["claude-opus-5-5", "claude-sonnet-5-5", "gemini-3.8-flash", "claude-haiku-4-5"]


def load():
    sys.path.insert(0, str(PAPER / "analysis"))
    from rater_robustness import LM_A_B, load_v4_humans

    census = {k: int(v) for k, v in json.loads((HV / "reference_bins.json").read_text(encoding="utf-8"))["reference"].items()}
    d = json.loads((HV / "external_rater_result.json").read_text(encoding="utf-8"))
    earlier = {x["item_id"]: int(x["external"]) for x in d["detail"]}
    v4 = load_v4_humans(set(census))  # eligible raters, from the anonymized export human_validation/v4/public/
    assert len(v4) >= 3, "v4 ratings missing: run human_validation/v4/score_v4.py and export_public_v4.py"
    raters = {"primary LM A": {k: int(v[0]) for k, v in LM_A_B.items()},
              "primary LM B": {k: int(v[1]) for k, v in LM_A_B.items()}}
    med = json.loads((HV / "llm_census_logged" / "result.json").read_text(encoding="utf-8"))["per_model_median"]
    for m in LOGGED:
        for cond, tag in (("A", "workbook"), ("B", "excerpt-only")):
            if f"{m}_{cond}" in med:
                raters[f"{m} ({tag})"] = {k: int(v) for k, v in med[f"{m}_{cond}"]["bins"].items() if v is not None}
    return census, earlier, v4, raters


def qwk(x, y):
    return float(cohen_kappa_score(x, y, weights="quadratic", labels=[0, 1, 2, 3, 4]))


def boot_ci(fn, items, B=10000, seed=20260930):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        s = list(rng.choice(items, size=len(items), replace=True))
        try:
            v = fn(s)
        except Exception:
            continue
        if v == v:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def vs(r, ref, items):
    its = [i for i in items if i in r]
    return {"n": len(its), "qwk": qwk([r[i] for i in its], [ref[i] for i in its]),
            "exact": int(sum(r[i] == ref[i] for i in its)),
            "within1": int(sum(abs(r[i] - ref[i]) <= 1 for i in its)),
            "derived_agree": int(sum((r[i] >= 2) == (ref[i] >= 2) for i in its))}


def main():
    census, earlier, v4, raters = load()
    items = sorted(census)
    assert len(items) == 29
    ext = [i for i in items if not i.startswith("OURS")]
    hs = sorted(v4)
    median = {i: sorted(v4[h][i] for h in hs)[(len(hs) - 1) // 2] for i in items}
    out = {"v4_humans": hs, "items": len(items)}

    # humans against the census and against one another
    out["humans_vs_census"] = {**{f"v4 {h}": {"all29": vs(v4[h], census, items), "external23": vs(v4[h], census, ext)} for h in hs},
                               "v4 median": {"all29": vs(median, census, items), "external23": vs(median, census, ext)},
                               "earlier non-author": {"all29": vs(earlier, census, items), "external23": vs(earlier, census, ext)}}
    out["human_human"] = {f"{a}-{b}": vs(v4[a], v4[b], items) for a, b in itertools.combinations(hs, 2)}
    out["earlier_vs_v4"] = {**{h: vs(earlier, v4[h], items) for h in hs}, "median": vs(earlier, median, items)}

    # every LM rater (and the census) against the census, the v4 median, each v4 human, and the earlier human
    rows = []
    for nm, r in [("census", census)] + list(raters.items()):
        row = {"rater": nm, "vs_census": vs(r, census, items), "vs_v4_median": vs(r, median, items),
               "vs_v4_median_ci": boot_ci(lambda s: qwk([r[i] for i in s], [median[i] for i in s]), items),
               "vs_v4_each_qwk": {h: vs(r, v4[h], items)["qwk"] for h in hs},
               "vs_earlier_human": vs(r, earlier, items)}
        rows.append(row)
    out["raters"] = rows
    (OUT / "rater_validation_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    L = ["# Agreement of the census raters with the census and with the human raters (analysis/rater_validation.py)", "",
         f"Human raters: the v4 study ({', '.join(hs)}; excerpt-only; human_validation/v4/result_v4.md) and their item-wise median; "
         "an earlier non-author rater (short prompt and condensed descriptions; external_rater_result.json).", "",
         "## Humans against the census", "", "| rater | scope | QWK | exact | within 1 | derived-vs-independent agree |", "|---|---|---:|---:|---:|---:|"]
    for nm, v in out["humans_vs_census"].items():
        for sc in ("all29", "external23"):
            s = v[sc]
            L.append(f"| {nm} | {sc} | {s['qwk']:.3f} | {s['exact']}/{s['n']} | {s['within1']}/{s['n']} | {s['derived_agree']}/{s['n']} |")
    L += ["", "## Human against human (29 items)", "", "| pair | QWK | exact | within 1 |", "|---|---:|---:|---:|"]
    for p, s in out["human_human"].items():
        L.append(f"| v4 {p} | {s['qwk']:.3f} | {s['exact']} | {s['within1']} |")
    for h, s in out["earlier_vs_v4"].items():
        L.append(f"| earlier non-author - v4 {h} | {s['qwk']:.3f} | {s['exact']} | {s['within1']} |")
    L += ["", "## Each rater against the census and the human raters (29 items)", "",
          "| rater | QWK vs census | exact | within 1 | derived agree | QWK vs v4 median [95% CI] | exact vs v4 median | QWK vs v4 " + "/".join(hs) + " | QWK vs earlier human |",
          "|---|---:|---:|---:|---:|---|---:|---|---:|"]
    for x in rows:
        c, m = x["vs_census"], x["vs_v4_median"]
        L.append(f"| {x['rater']} | {c['qwk']:.3f} | {c['exact']} | {c['within1']} | {c['derived_agree']} | "
                 f"{m['qwk']:.3f} [{x['vs_v4_median_ci'][0]:.3f}, {x['vs_v4_median_ci'][1]:.3f}] | {m['exact']} | "
                 + "/".join(f"{x['vs_v4_each_qwk'][h]:.3f}" for h in hs) + f" | {x['vs_earlier_human']['qwk']:.3f} |")
    L += ["", "The census is adjudicated from primary LM raters A and B and is not independent of them. Agreement is descriptive; "
          "the alternative annotator test against the three v4 humans is in human_validation/v4/result_v4.md."]
    (OUT / "rater_validation_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
