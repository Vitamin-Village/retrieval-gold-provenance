# -*- coding: utf-8 -*-
"""Score the judge spot-check v3 workbooks rater_{1,2,3}_spotcheck.xlsx against the sealed judge verdicts.
Reads them from returned/ next to this script, or from the folder in SPOT_RETURNED; writes result_v3.json and
result_v3.md next to this script, or into the folder in SPOT_OUT. Usage: python score_spotcheck_v3.py"""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
RET = Path(os.environ.get("SPOT_RETURNED", HERE / "returned"))
OUT = Path(os.environ.get("SPOT_OUT", HERE))  # ../llm_spotcheck_v3/ sets this for the AI spot-check
KEY = json.load(open(HERE / "sealed_key_v3.json", encoding="utf-8"))["items"]
P_POOL = json.load(open(HERE / "pool_stats.json"))["judge_correct_rate"]
LAB = ("CORRECT", "INCORRECT", "UNSURE")


def read(r):
    ws = load_workbook(RET / f"rater_{r}_spotcheck.xlsx", data_only=True)["SpotCheck"]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    out = {}
    for row in rows[1:]:
        d = dict(zip(head, row))
        v = str(d["your_verdict"] or "").strip().upper()
        if v not in LAB:
            sys.exit(f"ABORT: rater {r} item {d['item_id']}: verdict {v!r}")
        out[d["item_id"]] = v
    if set(out) != set(KEY):
        sys.exit(f"ABORT: rater {r} sheet does not cover the 100 items")
    return out


def cohen(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def fleiss(table):  # table: list of per-item label lists (no UNSURE)
    cats = ("CORRECT", "INCORRECT")
    rows = [t for t in table if len(t) >= 2]
    P, tot = [], Counter()
    for t in rows:
        c = Counter(t)
        m = len(t)
        P.append((sum(v * v for v in c.values()) - m) / (m * (m - 1)))
        tot.update(t)
    pbar = sum(P) / len(P)
    n = sum(tot.values())
    pe = sum((tot[c] / n) ** 2 for c in cats)
    return (pbar - pe) / (1 - pe)


def vs_judge(labels, ids):
    keep = [i for i in ids if labels[i] != "UNSURE"]
    h = [labels[i] for i in keep]
    j = [KEY[i]["judge"] for i in keep]
    k, po = cohen(h, j)
    tp = sum(x == "CORRECT" and y == "CORRECT" for x, y in zip(h, j))
    fp = sum(x == "INCORRECT" and y == "CORRECT" for x, y in zip(h, j))
    fn = sum(x == "CORRECT" and y == "INCORRECT" for x, y in zip(h, j))
    nJc = sum(y == "CORRECT" for y in j)
    nJi = len(j) - nJc
    a = tp / nJc if nJc else float("nan")  # P(human C | judge C) = precision
    b = fn / nJi if nJi else float("nan")  # P(human C | judge I)
    rec_w = P_POOL * a / (P_POOL * a + (1 - P_POOL) * b) if (a + b) > 0 else float("nan")
    return {"n": len(keep), "n_unsure": len(ids) - len(keep), "agreement": po, "kappa": k,
            "precision": a, "recall_sample": tp / (tp + fn) if tp + fn else float("nan"), "recall_pool_weighted": rec_w}


def main():
    raters = [r for r in (1, 2, 3) if (RET / f"rater_{r}_spotcheck.xlsx").exists()]
    if len(raters) < 3:
        sys.exit(f"ABORT: need three returned sheets in {RET}")
    L = {r: read(r) for r in raters}
    ids = sorted(KEY)
    res = {"per_rater": {r: vs_judge(L[r], ids) for r in raters},
           "v1_human_rater": vs_judge({i: KEY[i]["v1_human"] or "UNSURE" for i in ids}, ids)}
    res["pairwise_raters"] = {}
    for a, b in itertools.combinations(raters, 2):
        keep = [i for i in ids if L[a][i] != "UNSURE" and L[b][i] != "UNSURE"]
        k, po = cohen([L[a][i] for i in keep], [L[b][i] for i in keep])
        res["pairwise_raters"][f"R{a}-R{b}"] = {"n": len(keep), "agreement": po, "kappa": k}
    res["fleiss_kappa_3raters"] = fleiss([[L[r][i] for r in raters if L[r][i] != "UNSURE"] for i in ids])
    maj = {}
    for i in ids:
        c = Counter(L[r][i] for r in raters if L[r][i] != "UNSURE")
        top = c.most_common()
        maj[i] = "UNSURE" if not top or (len(top) > 1 and top[0][1] == top[1][1]) else top[0][0]
    res["majority_vs_judge"] = vs_judge(maj, ids)
    for reader in ("qwen2_5_14b", "llama3_1_8b"):
        sub = [i for i in ids if KEY[i]["reader"] == reader]
        if sub:
            res[f"majority_vs_judge_{reader}"] = vs_judge(maj, sub)
    json.dump(res, open(OUT / "result_v3.json", "w", encoding="utf-8"), indent=1)
    md = ["# Judge spot-check v3", "", f"pool judge-CORRECT rate used for weighting: {P_POOL:.3f}", "",
          "| rater | n | unsure | agreement | kappa | precision | recall (sample) | recall (pool-weighted) |", "|---|---|---|---|---|---|---|---|"]
    rows = [(f"R{r}", res["per_rater"][r]) for r in raters] + [("majority", res["majority_vs_judge"]), ("v1 human rater", res["v1_human_rater"])]
    rows += [(k.replace("majority_vs_judge_", "majority, "), v) for k, v in res.items() if k.startswith("majority_vs_judge_")]
    for name, e in rows:
        md.append(f"| {name} | {e['n']} | {e['n_unsure']} | {e['agreement']:.3f} | {e['kappa']:.3f} | {e['precision']:.3f} | {e['recall_sample']:.3f} | {e['recall_pool_weighted']:.3f} |")
    md += ["", f"Fleiss kappa (3 raters): {res['fleiss_kappa_3raters']:.3f}", ""]
    md += [f"- {k}: n {v['n']}, agreement {v['agreement']:.3f}, kappa {v['kappa']:.3f}" for k, v in res["pairwise_raters"].items()]
    (OUT / "result_v3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
