# -*- coding: utf-8 -*-
"""Score the returned v4 workbooks (plan: PROTOCOL_v4.md, Sections 4-6).

Put each rater's original attachment, unrenamed (rater_k_v4.xlsx), in human_validation/v4/returned/.
Usage (from human_validation/v4/):
    python score_v4.py                              # reads returned/, writes result_v4.md and result_v4.json here
    python score_v4.py --returned DIR --out DIR     # other folders (synthetic tests go to a scratch folder)

Per workbook: (1) integrity against SEALED_build.json (document properties, sheet and row order, item-text
hashes, locked cells, protection); (2) Rater_info completeness and the three attestations; (3) quiz (>= 4/6
final bins) and attention items (fail if >= 2 bins off on both); then (4) statistics over the eligible raters:
each human vs the census, human-human, each human vs each LM rater, and the alternative annotator test
(Calderon, Reichart and Dror, ACL 2025) for each LM rater and constant-bin controls; (5) per-benchmark
KG-derived headline bins. The statistics helpers are copied from analysis/rater_validation.py (2026-09-29);
LM_A_B and KG_AXES are imported from analysis/rater_robustness.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.stats import ttest_1samp
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
HV = HERE.parent
PAPER = HV.parent
sys.dont_write_bytecode = True  # importing the build and analysis modules must not write caches into the repo
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PAPER / "analysis"))
from build_package_v4 import cell_hash, docprops, item_hash  # noqa: E402  (same hashing as the build)
from rater_robustness import KG_AXES, LM_A_B  # noqa: E402

warnings.filterwarnings("ignore")
SEALED = json.load(open(HERE / "SEALED_build.json", encoding="utf-8"))
CENSUS = {k: int(v) for k, v in json.load(open(HV / "reference_bins.json", encoding="utf-8"))["reference"].items()}
ITEMS = sorted(CENSUS)
ANCHOR_BENCH = {"Anchor benchmark L", "Anchor benchmark U"}
EXTERNAL = [i for i in ITEMS if SEALED["item_meta"][i]["benchmark"] not in ANCHOR_BENCH]
EPS_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
QUOTED_EPS = 0.15
HEADLINE_MIN = {"GRADE": 3, "2WikiMultiHop": 3, "MetaQA": 3, "WebQSP": 3, "GraphRAG-Bench": 2, "STaRK": 2}
REQUIRED_INFO = ["background", "years_nlp_ir", "relation", "s1_start", "s1_end", "total_minutes", "att_a", "att_b", "att_c", "signature"]
ATTESTATIONS = {"att_a": "rated alone, no discussion", "att_b": "no AI assistant or language model", "att_c": "only package materials"}
COLS = {"source": "G", "base": "H", "corrections": "I", "final": "J", "phrase": "K", "confidence": "L", "minutes": "M"}


# ----------------------------------------------------------------------------- statistics (from analysis/rater_validation.py)
def qwk(x, y):
    try:
        return float(cohen_kappa_score(x, y, weights="quadratic", labels=[0, 1, 2, 3, 4]))
    except Exception:
        return float("nan")


def kripp_alpha_ordinal(rows):
    vals = sorted({v for r in rows for v in r if v is not None})
    idx = {v: i for i, v in enumerate(vals)}
    K = len(vals)
    o = np.zeros((K, K))
    for r in rows:
        r = [v for v in r if v is not None]
        m = len(r)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            o[idx[r[a]], idx[r[b]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    delta = np.zeros((K, K))
    for c in range(K):
        for k in range(K):
            lo, hi = min(c, k), max(c, k)
            delta[c, k] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[k]) / 2.0) ** 2
    Do = (o * delta).sum() / n
    De = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1 - Do / De) if De > 0 else float("nan")


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
    if not vals:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def by_reject(pvals, q=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    c = np.sum(1.0 / np.arange(1, m + 1))
    k = -1
    for i in range(m):
        if p[order[i]] <= (i + 1) / m * q / c:
            k = i
    return set(order[:k + 1].tolist()) if k >= 0 else set()


def alt_test(llm, humans, items, eps):
    """Leave one human out; score -RMSE against the remaining humans (-|diff| with one); one-sided t-test of
    H0: mean(W_h - W_f) >= eps; BY over the left-out humans. omega = share of humans the rater beats."""
    names = list(humans)
    pvals, rhos, counts = [], [], []
    for j in names:
        rest = [h for h in names if h != j]
        Wf, Wh = [], []
        for i in items:
            sf = -np.sqrt(np.mean([(llm[i] - humans[h][i]) ** 2 for h in rest]))
            sh = -np.sqrt(np.mean([(humans[j][i] - humans[h][i]) ** 2 for h in rest]))
            Wf.append(1 if sf >= sh else 0)
            Wh.append(1 if sh >= sf else 0)
        d = np.array(Wh) - np.array(Wf)
        p = ttest_1samp(d, eps, alternative="less").pvalue if d.std() > 0 else (0.0 if d.mean() < eps else 1.0)
        pvals.append(float(p))
        rhos.append(float(np.mean(Wf)))
        counts.append({"human_better": int((d > 0).sum()), "lm_better": int((d < 0).sum()), "tied": int((d == 0).sum())})
    rej = by_reject(pvals)
    return {"omega": len(rej) / len(names), "rho": float(np.mean(rhos)), "p": pvals, "counts": dict(zip(names, counts))}


def pair(a: dict, b: dict, items) -> dict:
    x, y = [a[i] for i in items], [b[i] for i in items]
    return {"n": len(items), "exact": sum(p == q for p, q in zip(x, y)) / len(items),
            "within_1": sum(abs(p - q) <= 1 for p, q in zip(x, y)) / len(items), "qwk": qwk(x, y),
            "derived_vs_independent": sum((p >= 2) == (q >= 2) for p, q in zip(x, y)) / len(items)}


def lower_median(vals):
    v = sorted(vals)
    return v[(len(v) - 1) // 2]


# ----------------------------------------------------------------------------- reading a workbook
def as_int(v, lo, hi):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(str(v).strip())
    except ValueError:
        return None
    return int(f) if f == int(f) and lo <= f <= hi else None


def as_text(v):
    return None if v is None or not str(v).strip() else str(v).strip()


def parse_time(v):
    if isinstance(v, datetime):
        return v
    s = as_text(v)
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y.%m.%d %H:%M", "%Y-%m-%dT%H:%M", "%Y. %m. %d. %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def integrity(k: int, data: bytes, wb) -> dict:
    S = SEALED["raters"][str(k)]
    hard, warn = [], []
    res = {"sha256": hashlib.sha256(data).hexdigest()}
    res["identical_to_sent_blank"] = res["sha256"] == S["files_sha256"][f"to_send/rater_{k}/rater_{k}_v4.xlsx"]
    if wb.sheetnames != SEALED["sheet_order"]:
        hard.append(f"sheets changed: {wb.sheetnames} (sent {SEALED['sheet_order']})")
    # document properties
    try:
        dp = docprops(data)
    except Exception as e:  # missing docProps parts
        dp = {"core": {}, "app": {}}
        hard.append(f"document properties unreadable: {e}")
    sent = S["docprops_sent"]
    diff = {}
    for part in ("core", "app"):
        for key in sorted(set(sent[part]) | set(dp[part])):
            if sent[part].get(key) != dp[part].get(key):
                diff[f"{part}.{key}"] = {"sent": sent[part].get(key), "returned": dp[part].get(key)}
    res["docprops_returned"] = dp
    res["docprops_diff"] = diff
    core = dp["core"]
    if core.get("creator") != sent["core"]["creator"]:
        hard.append(f"creator changed ({sent['core']['creator']!r} -> {core.get('creator')!r}): the file was re-created")
    if core.get("created") != sent["core"]["created"]:
        hard.append(f"created changed ({sent['core']['created']} -> {core.get('created')}): the file was re-created")
    if not core.get("lastModifiedBy"):
        warn.append("lastModifiedBy empty: not saved by Excel under a user account")
    if core.get("modified") and core.get("modified") < sent["core"]["created"]:
        warn.append("modified earlier than created")
    app = dp["app"].get("Application") or ""
    if not any(a in app for a in ("Microsoft Excel", "LibreOffice")):
        warn.append(f"application {app!r} is neither Excel nor LibreOffice")
    # protection
    for ws in wb.worksheets:
        if not ws.protection.sheet:
            hard.append(f"sheet protection removed on {ws.title}")
        elif ws.protection.password != SEALED["password_hash_legacy"]:
            warn.append(f"protection hash on {ws.title} differs from the sent one (re-saved by a tool that re-hashes it)")
    if not (wb.security and wb.security.lockStructure):
        hard.append("workbook structure lock removed")
    elif wb.security.workbookPassword != SEALED["password_hash_legacy"]:
        warn.append("workbook structure-lock hash differs from the sent one")
    # locked cells unchanged, nothing written outside the input cells
    changed, outside, filled = [], [], 0
    for name, locked in S["locked_cells"].items():
        if name not in wb.sheetnames:
            continue
        ws = wb[name]
        inputs = set(S["input_cells"][name])
        for coord, h in locked.items():
            v = ws[coord].value
            if v is None or cell_hash(v) != h:
                changed.append(f"{name}!{coord}")
        for c in list(ws._cells.values()):
            if c.value is None or (isinstance(c.value, str) and not c.value.strip()):
                continue
            if c.coordinate in inputs:
                filled += 1
            elif c.coordinate not in locked:
                outside.append(f"{name}!{c.coordinate}")
    if changed:
        hard.append(f"{len(changed)} locked cell(s) changed: {changed[:8]}")
    if outside:
        hard.append(f"{len(outside)} value(s) outside the input cells: {outside[:8]}")
    res["filled_input_cells"] = filled
    if filled and core.get("modified") == sent["core"]["modified"] and dp["app"] == sent["app"]:
        hard.append("filled, but modified time and app.xml are exactly as sent: filled by a script, not saved by a spreadsheet application")
    # item text hashes and row order
    for name, key in (("Quiz", "quiz_codes"), ("Census", "census_codes")):
        if name not in wb.sheetnames:
            continue
        ws = wb[name]
        codes = [ws[f"B{r}"].value for r in range(2, len(S[key]) + 2)]
        if codes != list(S[key]):
            hard.append(f"{name}: row codes or row order differ from the sent sheet")
        bad = []
        for r, code in enumerate(codes, 2):
            item = S[key].get(code)
            text = [ws[f"{c}{r}"].value or "" for c in "CDEF"]
            if item is None or item_hash(*text) != SEALED["item_text_sha256"][item]:
                bad.append(code)
        if bad:
            hard.append(f"{name}: item text differs from the sent text for rows {bad}")
    res["hard"], res["warn"] = hard, warn
    return res


def read_items(wb, k: int, name: str, key: str) -> dict:
    S = SEALED["raters"][str(k)]
    ws = wb[name]
    out = {}
    for r in range(2, len(S[key]) + 2):
        item = S[key].get(ws[f"B{r}"].value)
        if item is None:
            continue
        g = {f: ws[f"{c}{r}"].value for f, c in COLS.items()}
        src = as_text(g["source"])
        out[item] = {"source": src.upper() if src and src.upper() in ("K", "A", "T") else None,
                     "base": as_int(g["base"], 0, 4), "final": as_int(g["final"], 0, 4),
                     "corrections": as_text(g["corrections"]), "phrase": as_text(g["phrase"]),
                     "confidence": as_int(g["confidence"], 1, 3),
                     "minutes": (float(g["minutes"]) if isinstance(g["minutes"], (int, float)) else None)}
    return out


def read_info(wb) -> dict:
    ws = wb["Rater_info"]
    return {r["key"]: ws[f"B{r['row']}"].value for r in SEALED["rater_info_rows"]}


def check_info(k: int, info: dict) -> dict:
    missing = [key for key in REQUIRED_INFO if as_text(info.get(key)) is None]
    att = {key: (str(info.get(key)).strip().lower() if info.get(key) is not None else None) for key in ATTESTATIONS}
    t = {key: parse_time(info.get(key)) for key in ("s1_start", "s1_end", "s2_start", "s2_end")}
    span = 0.0
    for a, b in (("s1_start", "s1_end"), ("s2_start", "s2_end")):
        if t[a] and t[b]:
            span += (t[b] - t[a]).total_seconds() / 60
    problems = []
    if missing:
        problems.append(f"missing: {missing}")
    if info.get("rater_code") != f"R{k}":
        problems.append(f"rater code {info.get('rater_code')!r} is not R{k}")
    if as_text(info.get("relation")) != SEALED["relation_required"]:
        problems.append(f"relation is {info.get('relation')!r}")
    for key, v in att.items():
        if v != "yes":
            problems.append(f"attestation ({key[-1]}) {ATTESTATIONS[key]}: {v!r}")
    if t["s1_start"] is None or t["s1_end"] is None:
        problems.append("session 1 times not readable as date and time")
    return {"values": {k2: (v.isoformat() if isinstance(v, datetime) else v) for k2, v in info.items()},
            "attestations": att, "session_minutes_from_times": round(span, 1), "problems": problems, "ok": not problems}


# ----------------------------------------------------------------------------- statistics over a set of humans
def lm_raters() -> dict:
    lms = {"census": CENSUS, "primary LM A": {k: int(v[0]) for k, v in LM_A_B.items()},
           "primary LM B": {k: int(v[1]) for k, v in LM_A_B.items()}}
    med = json.load(open(HV / "llm_census_logged" / "result.json", encoding="utf-8"))["per_model_median"]
    for key, v in med.items():
        model, cond = key.rsplit("_", 1)
        tag = {"A": "workbook", "B": "excerpt-only"}[cond]
        runs = len(v.get("runs_used", []))
        lms[f"{model} ({tag}{'' if runs != 1 else ', one run'})"] = {i: int(b) for i, b in v["bins"].items() if b is not None}
    for nm, r in lms.items():
        assert set(ITEMS) <= set(r), nm
    return lms


def headline(bins: dict) -> dict:
    kg = {b: (max(bins[a] for a in axes) if axes else 0) for b, axes in KG_AXES.items()}
    checks = {b: kg[b] >= m for b, m in HEADLINE_MIN.items()}
    return {"kg_bins": kg, "checks": checks, "all_checks_hold": all(checks.values()),
            "n_external_ge3": sum(v >= 3 for v in kg.values()), "n_external_ge2": sum(v >= 2 for v in kg.values())}


def statistics(humans: dict, ratings: dict) -> dict:
    names = list(humans)
    out = {"raters": names, "n_humans": len(names)}
    out["vs_census"] = {h: {"all29": pair(humans[h], CENSUS, ITEMS), "external23": pair(humans[h], CENSUS, EXTERNAL)} for h in names}
    out["secondary"] = {h: {"source_counts": {s: sum(ratings[h][i]["source"] == s for i in ITEMS) for s in "KAT"},
                            "mean_confidence": float(np.mean([ratings[h][i]["confidence"] for i in ITEMS])),
                            "base_equals_final": sum(ratings[h][i]["base"] == ratings[h][i]["final"] for i in ITEMS),
                            "minutes_sum_rows": float(np.nansum([ratings[h][i]["minutes"] or np.nan for i in ITEMS]))}
                        for h in names}
    lms = lm_raters()
    out["vs_lm"] = {h: {nm: pair(humans[h], r, ITEMS) for nm, r in lms.items()} for h in names}
    out["headline"] = {h: headline(humans[h]) for h in names}
    out["headline"]["census (reference)"] = headline(CENSUS)
    if len(names) >= 3:
        med = {i: lower_median([humans[h][i] for h in names]) for i in ITEMS}
        out["median_humans"] = {"bins": med, "vs_census": {"all29": pair(med, CENSUS, ITEMS), "external23": pair(med, CENSUS, EXTERNAL)}}
        out["headline"]["median of humans"] = headline(med)
    if len(names) < 2:
        return out
    hh = {}
    for a, b in itertools.combinations(names, 2):
        A, B = humans[a], humans[b]
        e = {}
        for scope, its in (("all29", ITEMS), ("external23", EXTERNAL)):
            s = pair(A, B, its)
            s["qwk_ci"] = boot_ci(lambda x: qwk([A[i] for i in x], [B[i] for i in x]), its)
            s["alpha"] = kripp_alpha_ordinal([[A[i], B[i]] for i in its])
            s["alpha_ci"] = boot_ci(lambda x: kripp_alpha_ordinal([[A[i], B[i]] for i in x]), its)
            e[scope] = s
        e["source_agree"] = sum(ratings[a][i]["source"] == ratings[b][i]["source"] for i in ITEMS) / len(ITEMS)
        e["base_exact"] = sum(ratings[a][i]["base"] == ratings[b][i]["base"] for i in ITEMS) / len(ITEMS)
        e["ge2_apart"] = [(i, A[i], B[i]) for i in ITEMS if abs(A[i] - B[i]) >= 2]
        hh[f"{a}-{b}"] = e
    out["human_human"] = hh
    out["alpha_all_humans"] = {scope: {"alpha": kripp_alpha_ordinal([[humans[h][i] for h in names] for i in its]),
                                       "ci": boot_ci(lambda x: kripp_alpha_ordinal([[humans[h][i] for h in names] for i in x]), its)}
                               for scope, its in (("all29", ITEMS), ("external23", EXTERNAL))}
    mode = ("standard test (three or more humans)" if len(names) >= 3 else
            "DESCRIPTIVE ONLY: two humans; the alternative annotator test needs three or more annotators")
    alt = {}
    for nm, r in lms.items():
        alt[nm] = {f"{e:.2f}": alt_test(r, humans, ITEMS, e) for e in EPS_GRID}
    for c in range(5):
        alt[f"constant {c} (control)"] = {f"{e:.2f}": alt_test({i: c for i in ITEMS}, humans, ITEMS, e) for e in EPS_GRID}
    out["alt_test"] = {"mode": mode, "n_humans": len(names), "n_items": len(ITEMS), "eps_grid": EPS_GRID,
                       "quoted_eps": QUOTED_EPS, "results": alt}
    return out


# ----------------------------------------------------------------------------- report
def f3(x):
    return "nan" if x is None or x != x else f"{x:.3f}"


def report_stats(st: dict, title: str) -> list[str]:
    L = [f"## {title}", "", f"Raters: {', '.join(st['raters'])} (n = {st['n_humans']}).", ""]
    L += ["### Each human vs the census", "", "| rater | scope | QWK | exact | within-1 | derived vs independent |", "|---|---|---:|---:|---:|---:|"]
    rows = [(h, v) for h, v in st["vs_census"].items()] + ([("median of humans", st["median_humans"]["vs_census"])] if "median_humans" in st else [])
    for h, v in rows:
        for scope in ("all29", "external23"):
            s = v[scope]
            L.append(f"| {h} | {scope} | {f3(s['qwk'])} | {f3(s['exact'])} | {f3(s['within_1'])} | {f3(s['derived_vs_independent'])} |")
    L += ["", "Secondary: " + "; ".join(f"{h}: sources {v['source_counts']}, mean confidence {v['mean_confidence']:.2f}, "
                                          f"base = final on {v['base_equals_final']}/29" for h, v in st["secondary"].items()), ""]
    if "human_human" in st:
        L += ["### Human-human", "", "| pair | scope | QWK [95% CI] | exact | within-1 | ordinal alpha [95% CI] |", "|---|---|---|---:|---:|---|"]
        for p, e in st["human_human"].items():
            for scope in ("all29", "external23"):
                s = e[scope]
                L.append(f"| {p} | {scope} | {f3(s['qwk'])} [{f3(s['qwk_ci'][0])}, {f3(s['qwk_ci'][1])}] | {f3(s['exact'])} | "
                         f"{f3(s['within_1'])} | {f3(s['alpha'])} [{f3(s['alpha_ci'][0])}, {f3(s['alpha_ci'][1])}] |")
        for scope, a in st["alpha_all_humans"].items():
            L.append(f"\nOrdinal alpha over all {st['n_humans']} humans, {scope}: {f3(a['alpha'])} [{f3(a['ci'][0])}, {f3(a['ci'][1])}]")
        for p, e in st["human_human"].items():
            L.append(f"- {p}: source agreement {f3(e['source_agree'])}, base-bin exact {f3(e['base_exact'])}, items >= 2 bins apart: {e['ge2_apart']}")
        L.append("")
    hs = st["raters"]
    L += ["### Each human vs each LM rater (29 items; QWK / exact)", "", "| LM rater | " + " | ".join(hs) + " |", "|---|" + "---|" * len(hs)]
    for nm in st["vs_lm"][hs[0]]:
        L.append(f"| {nm} | " + " | ".join(f"{f3(st['vs_lm'][h][nm]['qwk'])} / {f3(st['vs_lm'][h][nm]['exact'])}" for h in hs) + " |")
    if "human_human" in st:
        L.append("\nFor comparison, human-human QWK / exact: " + "; ".join(f"{p}: {f3(e['all29']['qwk'])} / {f3(e['all29']['exact'])}"
                                                                      for p, e in st["human_human"].items()))
    L.append("")
    if "alt_test" in st:
        at = st["alt_test"]
        L += [f"### Alternative annotator test ({at['mode']})", "",
              f"Leave one human out; score -|bin difference| (one remaining human) or -RMSE (two or more); one-sided t-test of "
              f"H0: mean(W_h - W_f) >= eps; Benjamini-Yekutieli over the {at['n_humans']} left-out humans, q = 0.05; "
              f"{at['n_items']} items. omega = share of humans the rater beats (pass = omega >= 0.5); rho = advantage probability at eps {at['quoted_eps']:.2f}.", "",
              "| rater | " + " | ".join(f"eps {e:.2f}" for e in at["eps_grid"]) + " | rho |", "|---|" + "---:|" * (len(at["eps_grid"]) + 1)]
        q = f"{at['quoted_eps']:.2f}"
        for nm, a in at["results"].items():
            L.append(f"| {nm} | " + " | ".join(f"{a[f'{e:.2f}']['omega']:.2f}" for e in at["eps_grid"]) + f" | {a[q]['rho']:.2f} |")
        L += ["", f"Item counts at eps {q} (human better / LM better / tied), per left-out human:"]
        for nm, a in at["results"].items():
            L.append(f"- {nm}: " + "; ".join(f"{h}: {c['human_better']}/{c['lm_better']}/{c['tied']}" for h, c in a[q]["counts"].items()))
        L += ["", "The census is adjudicated from primary LM raters A and B, so these three raters are not independent of each other.", ""]
    L += ["### KG-derived headline bins per benchmark", "", "| rater | " + " | ".join(KG_AXES) + " | GRADE/2Wiki/MetaQA/WebQSP >= 3, GRB/STaRK >= 2 |",
          "|---|" + "---:|" * len(KG_AXES) + "---|"]
    for h, v in st["headline"].items():
        L.append(f"| {h} | " + " | ".join(str(v["kg_bins"][b]) for b in KG_AXES) + f" | {'all hold' if v['all_checks_hold'] else 'fails: ' + ', '.join(b for b, ok in v['checks'].items() if not ok)} |")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned", default=str(HERE / "returned"))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    ret, out = Path(args.returned), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found = {}
    ignored = []
    for p in sorted(ret.iterdir()) if ret.exists() else []:
        m = re.fullmatch(r"rater_(\d+)_v4\.xlsx", p.name)
        if m and m.group(1) in SEALED["raters"]:
            found[int(m.group(1))] = p
        elif p.is_file():
            ignored.append(p.name)
    try:  # report the folder relative to this script so that no local path enters the outputs
        ret_shown = ret.resolve().relative_to(HERE).as_posix()
    except ValueError:
        ret_shown = ret.name
    res = {"generated": datetime.now().isoformat(timespec="seconds"), "returned_dir": ret_shown, "ignored_files": ignored,
           "not_returned": [f"R{k}" for k in map(int, SEALED["raters"]) if k not in found], "workbooks": {}}
    ratings, humans = {}, {}
    for k, p in found.items():
        code = f"R{k}"
        data = p.read_bytes()
        w = {"file": p.name}
        try:
            wb = load_workbook(io.BytesIO(data), data_only=True)
        except Exception as e:
            w.update({"integrity": {"hard": [f"cannot open: {e}"], "warn": []}, "eligible": False, "reasons": ["cannot open"]})
            res["workbooks"][code] = w
            continue
        w["integrity"] = integrity(k, data, wb)
        w["rater_info"] = check_info(k, read_info(wb)) if "Rater_info" in wb.sheetnames else {"ok": False, "problems": ["no Rater_info sheet"]}
        quiz = read_items(wb, k, "Quiz", "quiz_codes") if "Quiz" in wb.sheetnames else {}
        cen = read_items(wb, k, "Census", "census_codes") if "Census" in wb.sheetnames else {}
        key = SEALED["answer_keys"]["quiz"]
        qres = {q: {"final": quiz.get(q, {}).get("final"), "key": v["final"], "source": quiz.get(q, {}).get("source"),
                    "key_source": v["source"], "ok": quiz.get(q, {}).get("final") == v["final"]} for q, v in key.items()}
        n_ok = sum(x["ok"] for x in qres.values())
        w["quiz"] = {"correct_final": n_ok, "of": len(key), "pass": n_ok >= 4, "items": qres}
        att = {}
        for a, e in SEALED["answer_keys"]["attention_expected_bins"].items():
            g = cen.get(a, {}).get("final")
            att[a] = {"given": g, "expected": e, "off_ge2": g is None or abs(g - e) >= 2}
        w["attention"] = {"items": att, "pass": not all(x["off_ge2"] for x in att.values())}
        incomplete = [i for i in ITEMS + list(SEALED["answer_keys"]["attention_expected_bins"])
                      if any(cen.get(i, {}).get(f) is None for f in ("source", "base", "final", "confidence"))]
        no_phrase = [i for i in ITEMS if not cen.get(i, {}).get("phrase")]
        w["census_completeness"] = {"incomplete_rows": incomplete, "rows_without_deciding_phrase": no_phrase,
                                    "base_minus_final_negative": [i for i in ITEMS if cen.get(i, {}).get("base") is not None
                                                                  and cen.get(i, {}).get("final") is not None and cen[i]["final"] > cen[i]["base"]]}
        reasons = []
        if w["integrity"]["hard"]:
            reasons.append("hard integrity flag")
        if not w["rater_info"]["ok"]:
            reasons.append("Rater_info/attestations")
        if not w["quiz"]["pass"]:
            reasons.append(f"quiz {n_ok}/6")
        if not w["attention"]["pass"]:
            reasons.append("attention items")
        if incomplete:
            reasons.append(f"{len(incomplete)} incomplete Census rows")
        w["eligible"], w["reasons"] = not reasons, reasons
        w["census"] = {i: cen.get(i) for i in ITEMS}
        res["workbooks"][code] = w
        if not incomplete:
            ratings[code] = cen
            humans[code] = {i: cen[i]["final"] for i in ITEMS}
    # cross-workbook checks
    cross = []
    codes = list(res["workbooks"])
    for a, b in itertools.combinations(codes, 2):
        A, B = res["workbooks"][a], res["workbooks"][b]
        ia, ib = A.get("integrity", {}), B.get("integrity", {})
        if ia.get("sha256") and ia.get("sha256") == ib.get("sha256"):
            cross.append(f"HARD {a} and {b} returned byte-identical files")
            for x in (A, B):
                x["eligible"] = False
                x["reasons"].append("identical file to another rater")
        la = ia.get("docprops_returned", {}).get("core", {}).get("lastModifiedBy")
        lb = ib.get("docprops_returned", {}).get("core", {}).get("lastModifiedBy")
        if la and la == lb:
            cross.append(f"WARN {a} and {b} were last saved under the same user account")
        if a in ratings and b in ratings:
            both = [i for i in ITEMS if ratings[a][i]["phrase"] and ratings[b][i]["phrase"]]
            same = [i for i in both if ratings[a][i]["phrase"].lower() == ratings[b][i]["phrase"].lower()]
            if both and len(same) / len(both) >= 0.5:
                cross.append(f"WARN {a} and {b}: identical deciding phrases on {len(same)}/{len(both)} items")
            if all(humans[a][i] == humans[b][i] for i in ITEMS) and all(ratings[a][i]["source"] == ratings[b][i]["source"] for i in ITEMS):
                cross.append(f"WARN {a} and {b}: identical final bins and sources on all 29 items")
        sa = A.get("rater_info", {}).get("values", {}).get("signature")
        sb = B.get("rater_info", {}).get("values", {}).get("signature")
        if sa and sa == sb:
            cross.append(f"WARN {a} and {b}: identical signature")
    res["cross_checks"] = cross
    eligible = [c for c in codes if res["workbooks"][c]["eligible"] and c in humans]
    res["eligible"] = eligible
    if eligible:
        res["primary"] = statistics({c: humans[c] for c in eligible}, {c: ratings[c] for c in eligible})
    complete = [c for c in codes if c in humans]
    if len(complete) > len(eligible) and len(complete) >= 1:
        res["sensitivity_all_complete"] = statistics({c: humans[c] for c in complete}, {c: ratings[c] for c in complete})
    json.dump(res, open(out / "result_v4.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)

    # markdown
    L = ["# Human validation v4: results (generated by score_v4.py)", "",
         f"Generated {res['generated']} from `{ret_shown}`. Workbooks: {', '.join(codes) or 'none'}; not returned: {', '.join(res['not_returned']) or 'none'}; "
         f"ignored files: {ignored or 'none'}.", "",
         "## 1. Integrity", "", "| rater | SHA-256 | blank as sent | filled input cells | hard flags | warnings |", "|---|---|---|---:|---|---|"]
    for c, w in res["workbooks"].items():
        i = w["integrity"]
        L.append(f"| {c} | {i.get('sha256', '')[:16]} | {i.get('identical_to_sent_blank')} | {i.get('filled_input_cells')} | "
                 f"{'; '.join(i['hard']) or 'none'} | {'; '.join(i['warn']) or 'none'} |")
    L.append("")
    for c, w in res["workbooks"].items():
        d = w["integrity"].get("docprops_diff", {})
        shown = {k: ({"sent": v["sent"], "returned": ("(set)" if v["returned"] else v["returned"])} if k == "core.lastModifiedBy" else v)
                 for k, v in d.items() if k != "app.Company"}
        L.append(f"- {c} document properties changed from sent: {shown or 'none'}")
    L += ["", "Cross-workbook checks: " + ("; ".join(cross) if cross else "none"), "",
          "## 2. Rater_info and attestations", "", "| rater | background | years | (a) alone | (b) no AI | (c) package only | total min | min from session times | problems |",
          "|---|---|---:|---|---|---|---:|---:|---|"]
    for c, w in res["workbooks"].items():
        ri = w.get("rater_info", {})
        v, a = ri.get("values", {}), ri.get("attestations", {})
        L.append(f"| {c} | {v.get('background')} | {v.get('years_nlp_ir')} | {a.get('att_a')} | {a.get('att_b')} | {a.get('att_c')} | "
                 f"{v.get('total_minutes')} | {ri.get('session_minutes_from_times')} | {'; '.join(ri.get('problems', [])) or 'none'} |")
    L += ["", "## 3. Quiz, attention, completeness, eligibility", "", "| rater | quiz (final bin) | attention (given/expected) | incomplete rows | eligible | reasons |", "|---|---|---|---|---|---|"]
    for c, w in res["workbooks"].items():
        q, at = w.get("quiz", {}), w.get("attention", {})
        att_s = ", ".join(f"{a}: {x['given']}/{x['expected']}" for a, x in at.get("items", {}).items())
        L.append(f"| {c} | {q.get('correct_final')}/6 {'pass' if q.get('pass') else 'FAIL'} | {att_s} {'pass' if at.get('pass') else 'FAIL'} | "
                 f"{len(w.get('census_completeness', {}).get('incomplete_rows', []))} | {w['eligible']} | {'; '.join(w['reasons']) or '-'} |")
    L.append("")
    if "primary" in res:
        L += report_stats(res["primary"], "4-5. Statistics, eligible raters (primary)")
    else:
        L += ["## 4-5. Statistics", "", "No eligible workbook; no statistics are reported as primary.", ""]
    if "sensitivity_all_complete" in res:
        L += report_stats(res["sensitivity_all_complete"], "Sensitivity: all returned workbooks with a complete Census sheet (NOT primary)")
    if humans:
        hs = [c for c in codes if c in humans]
        L += ["## Per item (final bins)", "", "| item | census | " + " | ".join(hs) + " |", "|---|---:|" + "---:|" * len(hs)]
        for i in ITEMS:
            L.append(f"| {i} | {CENSUS[i]} | " + " | ".join(str(humans[c][i]) for c in hs) + " |")
    (out / "result_v4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
