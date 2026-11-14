# -*- coding: utf-8 -*-
"""Score the returned judge spot-check v4 workbooks (plan: PROTOCOL_judge_v4.md, Sections 4-6).

Put each rater's original attachment, unrenamed (rater_k_judge_v4.xlsx), in judge_spotcheck_v4/returned/.
Usage (from judge_spotcheck_v4/):
    python score_judge_v4.py                              # reads returned/, writes result_judge_v4.md and .json here
    python score_judge_v4.py --returned DIR --out DIR     # other folders
    python score_judge_v4.py --returned DIR --out DIR --allow-script-fill
        TEST ONLY, for synthetic workbooks filled with openpyxl outside this folder: the script-fill hard flag
        becomes a warning, the report is marked as a test, and nothing may be written into this folder.

Per workbook: (1) integrity against SEALED_judge_build.json, with the checks and definitions of
../v4/score_v4.py (document properties, sheet order, row codes and order, item-text hashes, locked cells,
values outside the input cells, protection, and the script-fill check), plus cross-workbook checks;
(2) Rater_info completeness and the three attestations; (3) attention items (excluded if wrong on both) and
completeness (every item labelled; UNSURE counts as labelled); then (4) the outcomes of the protocol over the
eligible raters: each rater vs the judge, Fleiss' and pairwise Cohen's kappa with item-bootstrap CIs, the
majority vs the judge with precision, recall, and pool-weighted recall,
per reader, and comparisons with the v1 human and the Claude labels of ../llm_spotcheck_v3/.
cohen, fleiss, and vs_judge are copied from ../judge_spotcheck_v3/score_spotcheck_v3.py; boot_ci, as_int,
as_text, parse_time, and the integrity and Rater_info checks from ../v4/score_v4.py.
No user name, signature, or file-system path from a returned file is written to the markdown; only whether
such values are equal across workbooks. The JSON stores keyed hashes (HMAC with the sealed password) in their place.
Deviation 1 of the protocol (2026-10-01, before sending) adds a second script-fill condition: app.xml naming
"Microsoft Excel" with a non-Excel AppVersion and no DocSecurity field.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import itertools
import json
import re
import sys
import warnings
import zipfile
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
HV = HERE.parent
sys.dont_write_bytecode = True  # importing the build module must not write a cache into the package folder
sys.path.insert(0, str(HERE))
from build_package_judge_v4 import cell_hash, docprops, item_hash  # noqa: E402  (same hashing as the build)

warnings.filterwarnings("ignore")
SEALED = json.load(open(HERE / "SEALED_judge_build.json", encoding="utf-8"))
LAB = ("CORRECT", "INCORRECT", "UNSURE")
CI_LABS = ("CORRECT", "INCORRECT")
READERS = {"qwen2_5_14b": "Qwen 2.5 14B", "llama3_1_8b": "Llama 3.1 8B"}
CLAUDE = {"claude-opus-5": "1", "claude-sonnet-5": "2", "claude-haiku-4-5": "3"}  # rater numbers in ../llm_spotcheck_v3/ (README)
B_BOOT, BOOT_SEED = 10000, 20261001
REQUIRED_INFO = ["background", "years_nlp_ir", "relation", "s1_start", "s1_end", "total_minutes", "att_a", "att_b", "att_c", "signature"]
ATTESTATIONS = {"att_a": "rated alone, no discussion", "att_b": "no AI assistant or language model", "att_c": "only package materials"}
TEXT_COLS = SEALED["item_columns"]["text_cols"]
COL_VERDICT, COL_NOTE = SEALED["item_columns"]["input_cols"]
SAFE_PROPS = {"core.created", "core.modified", "core.title", "app.Application", "app.AppVersion", "app.DocSecurity",
              "app.ScaleCrop", "app.LinksUpToDate", "app.SharedDoc", "app.HyperlinksChanged", "app.TotalTime"}
EXCEL_APPVERSION = re.compile(r"^\d{2}\.\d{4}$")  # what Excel writes, e.g. 16.0300; openpyxl writes its own version, e.g. 3.0


def sha(s) -> str:
    """Keyed hash (HMAC-SHA-256 with the sealed password, 16 hex digits): equality can be checked, a list of
    names cannot be hashed and matched against it without SEALED_judge_build.json."""
    return hmac.new(SEALED["password"].encode("utf-8"), str(s).encode("utf-8"), hashlib.sha256).hexdigest()[:16]


# ----------------------------------------------------------------------------- reference labels (checked against the build)
def check_inputs() -> dict:
    changed = [rel for rel, h in SEALED["inputs_sha256"].items() if hashlib.sha256((HV / rel).read_bytes()).hexdigest() != h]
    if changed:
        sys.exit(f"ABORT: input files changed since the build (SHA-256 differs from SEALED_judge_build.json): {changed}")
    return {rel: True for rel in SEALED["inputs_sha256"]}


INPUTS_OK = check_inputs()
KEY = json.load(open(HV / "judge_spotcheck_v3" / "sealed_key_v3.json", encoding="utf-8"))["items"]
ITEMS = list(SEALED["items"])
assert sorted(KEY) == sorted(ITEMS)
JUDGE = {i: KEY[i]["judge"] for i in ITEMS}
READER = {i: KEY[i]["reader"] for i in ITEMS}
V1_HUMAN = {i: (KEY[i]["v1_human"] or "UNSURE") for i in ITEMS}
P_POOL = json.load(open(HV / "judge_spotcheck_v3" / "pool_stats.json", encoding="utf-8"))["judge_correct_rate"]


def claude_labels() -> dict:
    out = {m: {} for m in CLAUDE}
    for line in open(HV / "llm_spotcheck_v3" / "raw.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["model_requested"] in out and r.get("parsed"):
            v = str(r["verdict"]).strip().upper()
            out[r["model_requested"]][r["item_id"]] = v if v in LAB else "UNSURE"
    for m, d in out.items():
        assert set(d) == set(ITEMS), m
    return out


CLAUDE_LABELS = claude_labels()


# ----------------------------------------------------------------------------- statistics (from score_spotcheck_v3.py and score_v4.py)
def cohen(a, b):
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def fleiss(table):  # table: list of per-item label lists (no UNSURE)
    cats = ("CORRECT", "INCORRECT")
    rows = [t for t in table if len(t) >= 2]
    if not rows:
        return float("nan")
    P, tot = [], Counter()
    for t in rows:
        c = Counter(t)
        m = len(t)
        P.append((sum(v * v for v in c.values()) - m) / (m * (m - 1)))
        tot.update(t)
    pbar = sum(P) / len(P)
    n = sum(tot.values())
    pe = sum((tot[c] / n) ** 2 for c in cats)
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_ci(fn, items, B=B_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        s = [str(x) for x in rng.choice(items, size=len(items), replace=True)]
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


def vs_judge(labels, ids):
    keep = [i for i in ids if labels.get(i) in CI_LABS]
    h = [labels[i] for i in keep]
    j = [JUDGE[i] for i in keep]
    k, po = cohen(h, j)
    tp = sum(x == "CORRECT" and y == "CORRECT" for x, y in zip(h, j))
    fp = sum(x == "INCORRECT" and y == "CORRECT" for x, y in zip(h, j))
    fn = sum(x == "CORRECT" and y == "INCORRECT" for x, y in zip(h, j))
    tn = len(keep) - tp - fp - fn
    nJc = sum(y == "CORRECT" for y in j)
    nJi = len(j) - nJc
    a = tp / nJc if nJc else float("nan")  # P(human C | judge C) = precision
    b = fn / nJi if nJi else float("nan")  # P(human C | judge I)
    rec_w = P_POOL * a / (P_POOL * a + (1 - P_POOL) * b) if (a == a and b == b and (a + b) > 0) else float("nan")
    return {"n": len(keep), "n_unsure": len(ids) - len(keep), "agreement": po, "kappa": k, "precision": a,
            "recall_sample": tp / (tp + fn) if tp + fn else float("nan"), "recall_pool_weighted": rec_w,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def with_ci(labels, ref, ids) -> dict:
    """Bootstrap CIs of kappa and agreement between labels and ref over items where both are CORRECT/INCORRECT."""
    def stat(s, idx):
        keep = [i for i in s if labels.get(i) in CI_LABS and ref.get(i) in CI_LABS]
        return cohen([labels[i] for i in keep], [ref[i] for i in keep])[idx]
    return {"kappa_ci": boot_ci(lambda s: stat(s, 0), ids), "agreement_ci": boot_ci(lambda s: stat(s, 1), ids)}


def agree(a, b, ids) -> dict:
    keep = [i for i in ids if a.get(i) in CI_LABS and b.get(i) in CI_LABS]
    k, po = cohen([a[i] for i in keep], [b[i] for i in keep])
    return {"n": len(keep), "agreement": po, "kappa": k}


def majority(L: dict, ids) -> dict:
    """Three or more raters: majority over non-UNSURE labels, tie or no label = UNSURE (score_spotcheck_v3.py).
    Two raters: the label on which both agree; anything else = UNSURE (PROTOCOL_judge_v4.md, Section 6)."""
    names = list(L)
    out = {}
    for i in ids:
        if len(names) == 2:
            x, y = L[names[0]].get(i), L[names[1]].get(i)
            out[i] = x if x == y and x in CI_LABS else "UNSURE"
        else:
            c = Counter(L[n][i] for n in names if L[n].get(i) in CI_LABS)
            top = c.most_common()
            out[i] = "UNSURE" if not top or (len(top) > 1 and top[0][1] == top[1][1]) else top[0][0]
    return out


CLAUDE_MAJ = majority(CLAUDE_LABELS, ITEMS)


def statistics(L: dict) -> dict:
    names = list(L)
    st = {"raters": names, "n_raters": len(names)}
    st["label_counts"] = {h: {lab: sum(L[h][i] == lab for i in ITEMS) for lab in LAB} for h in names}
    st["vs_judge"] = {h: {**vs_judge(L[h], ITEMS), **with_ci(L[h], JUDGE, ITEMS)} for h in names}
    maj = None
    if len(names) >= 2:
        st["pairwise"] = {f"{a}-{b}": {**agree(L[a], L[b], ITEMS), **with_ci(L[a], L[b], ITEMS)}
                          for a, b in itertools.combinations(names, 2)}
        table = lambda ids: [[L[h][i] for h in names if L[h][i] in CI_LABS] for i in ids]
        st["fleiss"] = {"kappa": fleiss(table(ITEMS)), "ci": boot_ci(lambda s: fleiss(table(s)), ITEMS),
                        "n_items_used": sum(len(t) >= 2 for t in table(ITEMS)),
                        "note": "three raters" if len(names) >= 3 else "two raters (equals Scott's pi); labelled, not the planned three"}
        maj = majority(L, ITEMS)
        st["majority"] = {"rule": ("majority of three over non-UNSURE labels; tie or no label = UNSURE" if len(names) >= 3 else
                                   "TWO RATERS ONLY: the label both give; anything else = UNSURE"),
                          "labels": maj, "counts": {lab: sum(maj[i] == lab for i in ITEMS) for lab in LAB},
                          "vs_judge": {**vs_judge(maj, ITEMS), **with_ci(maj, JUDGE, ITEMS)}}
    st["per_reader"] = {}
    for reader in READERS:
        sub = [i for i in ITEMS if READER[i] == reader]
        e = {h: {**vs_judge(L[h], sub), **with_ci(L[h], JUDGE, sub)} for h in names}
        if maj:
            e["majority"] = {**vs_judge(maj, sub), **with_ci(maj, JUDGE, sub)}
        st["per_reader"][reader] = {"n_items": len(sub), "results": e}
    refs = {"first human rater": V1_HUMAN, **{m: CLAUDE_LABELS[m] for m in CLAUDE}, "Claude majority": CLAUDE_MAJ}
    rows = {h: L[h] for h in names}
    if maj:
        rows["majority"] = maj
    st["comparisons"] = {h: {r: agree(lab, ref, ITEMS) for r, ref in refs.items()} for h, lab in rows.items()}
    return st


def reference_rows() -> dict:
    """The v1 human and the Claude labels against the judge, and whether they reproduce the earlier results."""
    v1 = json.load(open(HV / "judge_spotcheck" / "spotcheck_result.json", encoding="utf-8"))
    lm = json.load(open(HV / "llm_spotcheck_v3" / "result_v3.json", encoding="utf-8"))
    rows = {"first human rater": vs_judge(V1_HUMAN, ITEMS)}
    for m in CLAUDE:
        rows[f"{m} (LM labels)"] = vs_judge(CLAUDE_LABELS[m], ITEMS)
    rows["Claude majority (LM labels)"] = vs_judge(CLAUDE_MAJ, ITEMS)
    close = lambda x, y: abs(x - y) < 1e-9
    repro = {"v1 human vs spotcheck_result.json": close(rows["first human rater"]["agreement"], v1["agreement"])
             and close(rows["first human rater"]["kappa"], v1["kappa"])}
    for m, r in CLAUDE.items():
        repro[f"{m} vs result_v3.json rater {r}"] = (close(rows[f"{m} (LM labels)"]["agreement"], lm["per_rater"][r]["agreement"])
                                                     and close(rows[f"{m} (LM labels)"]["kappa"], lm["per_rater"][r]["kappa"]))
    repro["Claude majority vs result_v3.json majority"] = (close(rows["Claude majority (LM labels)"]["agreement"], lm["majority_vs_judge"]["agreement"])
                                                          and close(rows["Claude majority (LM labels)"]["kappa"], lm["majority_vs_judge"]["kappa"]))
    return {"rows": rows, "reproduces_earlier_results": repro, "pool_judge_correct_rate": P_POOL}


# ----------------------------------------------------------------------------- reading a workbook (from score_v4.py)
def as_text(v):
    return None if v is None or not str(v).strip() else str(v).strip()


def parse_time(v):
    if isinstance(v, datetime):
        return v
    if isinstance(v, time):  # a time without a date (Excel converts e.g. 14:05)
        return datetime.combine(date(2000, 1, 1), v)
    s = as_text(v)
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y.%m.%d %H:%M", "%Y-%m-%dT%H:%M", "%Y. %m. %d. %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def masked(key: str, v):
    if key in SAFE_PROPS or v in (None, ""):
        return v
    return f"(set; sha256 {sha(v)})"


def fingerprint(data: bytes) -> dict:
    """Save fingerprints in xl/workbook.xml. Only hashes are kept: equality is reported, never the value."""
    try:
        wbx = zipfile.ZipFile(io.BytesIO(data)).read("xl/workbook.xml").decode("utf-8", "replace")
    except Exception:
        return {"save_path_recorded": False, "profile_sha": None, "folder_sha": None, "excel_build": None}
    ap = re.search(r'absPath[^>]*?url="([^"]*)"', wbx)
    url = ap.group(1) if ap else None
    prof = re.match(r"(?:[A-Za-z]:\\Users\\|/Users/|/home/)([^\\/]+)", url) if url else None
    build = re.search(r'rupBuild="(\d+)"', wbx)
    return {"save_path_recorded": url is not None, "profile_sha": sha(prof.group(1).lower()) if prof else None,
            "folder_sha": sha(url.lower()) if url else None, "excel_build": build.group(1) if build else None}


def integrity(k: int, data: bytes, wb, allow_script_fill: bool) -> dict:
    S = SEALED["raters"][str(k)]
    hard, warn = [], []
    res = {"sha256": hashlib.sha256(data).hexdigest()}
    res["identical_to_sent_blank"] = res["sha256"] == S["files_sha256"][f"to_send/rater_{k}/rater_{k}_judge_v4.xlsx"]
    if wb.sheetnames != SEALED["sheet_order"]:
        missing = [s for s in SEALED["sheet_order"] if s not in wb.sheetnames]
        added = len([s for s in wb.sheetnames if s not in SEALED["sheet_order"]])
        hard.append(f"sheets changed (sent {SEALED['sheet_order']}; missing {missing}; {added} added; order "
                    f"{'same' if [s for s in wb.sheetnames if s in SEALED['sheet_order']] == [s for s in SEALED['sheet_order'] if s in wb.sheetnames] else 'changed'})")
    # document properties
    try:
        dp = docprops(data)
    except Exception as e:  # missing docProps parts
        dp = {"core": {}, "app": {}}
        hard.append(f"document properties unreadable: {type(e).__name__}")
    sent = S["docprops_sent"]
    diff = {}
    for part in ("core", "app"):
        for key in sorted(set(sent[part]) | set(dp[part])):
            if sent[part].get(key) != dp[part].get(key):
                diff[f"{part}.{key}"] = {"sent": sent[part].get(key), "returned": masked(f"{part}.{key}", dp[part].get(key))}
    res["docprops_returned"] = {part: {key: masked(f"{part}.{key}", v) for key, v in d.items()} for part, d in dp.items()}
    res["docprops_diff"] = diff
    res["_last_modified_by"] = dp["core"].get("lastModifiedBy") or None  # kept in memory for the cross check; removed before writing
    core = dp["core"]
    if core.get("creator") != sent["core"]["creator"]:
        hard.append(f"creator changed (no longer {sent['core']['creator']!r}): the file was re-created")
    if core.get("created") != sent["core"]["created"]:
        hard.append(f"created changed ({sent['core']['created']} -> {core.get('created')}): the file was re-created")
    if not core.get("lastModifiedBy"):
        warn.append("lastModifiedBy empty: not saved by Excel under a user account")
    if core.get("modified") and core.get("modified") < sent["core"]["created"]:
        warn.append("modified earlier than created")
    app = dp["app"].get("Application") or ""
    if not any(a in app for a in ("Microsoft Excel", "LibreOffice")):
        warn.append(f"application {app[:40]!r} is neither Excel nor LibreOffice")
    # protection
    for ws in wb.worksheets:
        if not ws.protection.sheet:
            hard.append(f"sheet protection removed on {ws.title if ws.title in SEALED['sheet_order'] else 'an added sheet'}")
        elif ws.protection.password != SEALED["password_hash_legacy"]:
            warn.append(f"protection hash on {ws.title if ws.title in SEALED['sheet_order'] else 'an added sheet'} differs from the sent one (re-saved by a tool that re-hashes it)")
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
    script = []
    if filled and core.get("modified") == sent["core"]["modified"] and dp["app"] == sent["app"]:
        script.append("filled, but modified time and app.xml are exactly as sent: filled by a script, not saved by a spreadsheet application")
    av = dp["app"].get("AppVersion") or ""
    if filled and app == "Microsoft Excel" and not EXCEL_APPVERSION.match(av) and "DocSecurity" not in dp["app"]:
        # Deviation 1 of PROTOCOL_judge_v4.md: a fill with another openpyxl version writes its own AppVersion
        script.append(f"filled, and app.xml carries a library's signature (Application 'Microsoft Excel', AppVersion {av[:12]!r}, "
                      "no DocSecurity): written by a script, not saved by a spreadsheet application")
    for msg in script:
        if allow_script_fill:
            warn.append("[TEST MODE --allow-script-fill, hard flag downgraded] " + msg)
        else:
            hard.append(msg)
    # item text hashes and row order
    if "Items" in wb.sheetnames:
        ws = wb["Items"]
        codes = [ws[f"B{r}"].value for r in range(2, len(S["codes"]) + 2)]
        if codes != list(S["codes"]):
            hard.append("Items: row codes or row order differ from the sent sheet")
        bad = []
        for r, code in enumerate(codes, 2):
            item = S["codes"].get(code)
            text = [ws[f"{c}{r}"].value or "" for c in TEXT_COLS]
            if item is None or item_hash(*[str(t) for t in text]) != SEALED["item_text_sha256"][item]:
                bad.append(f"row {r}")
        if bad:
            hard.append(f"Items: item text differs from the sent text for {len(bad)} row(s): {bad[:8]}")
    res["hard"], res["warn"] = hard, warn
    return res


def read_items(wb, k: int) -> tuple[dict, dict]:
    S = SEALED["raters"][str(k)]
    ws = wb["Items"]
    out, invalid = {}, {}
    for r in range(2, len(S["codes"]) + 2):
        item = S["codes"].get(ws[f"B{r}"].value)
        if item is None:
            continue
        raw = ws[f"{COL_VERDICT}{r}"].value
        v = as_text(raw)
        v = v.upper() if v else None
        if v is not None and v not in LAB:
            invalid[item] = str(raw)[:40]
            v = None
        out[item] = {"verdict": v, "note": as_text(ws[f"{COL_NOTE}{r}"].value)}
    return out, invalid


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
    problems, notes = [], []  # problems exclude (PROTOCOL Section 5.2); notes are only reported
    if missing:
        problems.append(f"missing: {missing}")
    if info.get("rater_code") != f"R{k}":
        problems.append("rater code is not the one sent")
    if as_text(info.get("relation")) != SEALED["relation_required"]:
        rel = as_text(info.get("relation"))
        problems.append(f"relation is {rel if rel in (None, 'author of the paper', 'involved in the project in another way') else '(other text)'!r}")
    for key, v in att.items():
        if v != "yes":
            problems.append(f"attestation ({key[-1]}) {ATTESTATIONS[key]}: {v if v in (None, 'no') else '(other text)'!r}")
    if (as_text(info.get("s1_start")) and as_text(info.get("s1_end"))) and (t["s1_start"] is None or t["s1_end"] is None):
        notes.append("session 1 times filled but not readable as date and time")
    if span < 0:
        notes.append("session end earlier than start")
    values = {k2: (v.isoformat() if isinstance(v, (datetime, time)) else v) for k2, v in info.items()}
    values["signature"] = "(given)" if as_text(info.get("signature")) else None  # never stored or printed
    return {"values": values, "attestations": att, "session_minutes_from_times": round(span, 1), "problems": problems,
            "notes": notes, "ok": not problems, "_signature": as_text(info.get("signature"))}


# ----------------------------------------------------------------------------- report
def f3(x):
    return "nan" if x is None or x != x else f"{x:.3f}"


def ci(c):
    return f"[{f3(c[0])}, {f3(c[1])}]"


def judge_row(name, e, with_cis=True):
    a = f"{f3(e['agreement'])} {ci(e['agreement_ci'])}" if with_cis and "agreement_ci" in e else f3(e["agreement"])
    k = f"{f3(e['kappa'])} {ci(e['kappa_ci'])}" if with_cis and "kappa_ci" in e else f3(e["kappa"])
    return (f"| {name} | {e['n']} | {e['n_unsure']} | {a} | {k} | {f3(e['precision'])} | {f3(e['recall_sample'])} | "
            f"{f3(e['recall_pool_weighted'])} | {e['tp']}/{e['fp']}/{e['fn']}/{e['tn']} |")


JUDGE_HEAD = ["| rater | n | UNSURE (excluded) | agreement [95% CI] | Cohen's kappa [95% CI] | judge precision | judge recall (sample) | "
              "judge recall (pool-weighted) | TP/FP/FN/TN |", "|---|---:|---:|---|---|---:|---:|---:|---|"]


def report_stats(st: dict, title: str) -> list[str]:
    L = [f"## {title}", "", f"Raters: {', '.join(st['raters'])} (n = {st['n_raters']}). 100 items; attention items excluded. "
         f"CIs: item bootstrap, {B_BOOT} resamples, seed {BOOT_SEED}, percentile. Pool judge-CORRECT rate p = {P_POOL:.3f}. "
         "TP = rater CORRECT and judge CORRECT; FP = rater INCORRECT, judge CORRECT; FN = rater CORRECT, judge INCORRECT.", ""]
    L += ["### 1. Each rater vs the judge", ""] + JUDGE_HEAD
    L += [judge_row(h, st["vs_judge"][h]) for h in st["raters"]]
    L += ["", "Label counts: " + "; ".join(f"{h}: " + ", ".join(f"{lab} {n}" for lab, n in c.items()) for h, c in st["label_counts"].items()), ""]
    if "pairwise" in st:
        fl = st["fleiss"]
        L += ["### 2. Among the raters", "", f"Fleiss' kappa ({fl['note']}): {f3(fl['kappa'])} {ci(fl['ci'])}, over {fl['n_items_used']} items "
              "with at least two non-UNSURE labels.", "",
              "| pair | n (both CORRECT/INCORRECT) | agreement [95% CI] | Cohen's kappa [95% CI] |", "|---|---:|---|---|"]
        for p, e in st["pairwise"].items():
            L.append(f"| {p} | {e['n']} | {f3(e['agreement'])} {ci(e['agreement_ci'])} | {f3(e['kappa'])} {ci(e['kappa_ci'])} |")
        m = st["majority"]
        L += ["", f"### 3. Majority vs the judge ({m['rule']})", "",
              "Majority labels: " + ", ".join(f"{lab} {n}" for lab, n in m["counts"].items()) + ".", ""] + JUDGE_HEAD
        L += [judge_row("majority", m["vs_judge"]), ""]
    L += ["### 4. Per reader (the model that wrote the candidate)", "",
          f"CIs resample the items of that reader. Pool-weighted recall uses the overall pool rate p = {P_POOL:.3f} "
          "(no per-reader rate is defined).", ""] + JUDGE_HEAD
    for reader, e in st["per_reader"].items():
        for h, v in e["results"].items():
            L.append(judge_row(f"{h}, {READERS[reader]} ({e['n_items']} items)", v))
    L += ["", "### 5. Comparisons (not primary): agreement / Cohen's kappa (n), over items where both labels are CORRECT or INCORRECT", ""]
    refs = list(next(iter(st["comparisons"].values())))
    L += ["| rater | " + " | ".join(refs) + " |", "|---|" + "---|" * len(refs)]
    for h, e in st["comparisons"].items():
        L.append(f"| {h} | " + " | ".join(f"{f3(e[r]['agreement'])} / {f3(e[r]['kappa'])} ({e[r]['n']})" for r in refs) + " |")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned", default=str(HERE / "returned"))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--allow-script-fill", action="store_true",
                    help="TEST ONLY: for synthetic workbooks filled with openpyxl; downgrades the script-fill hard flag to a "
                         "warning, marks the report as a test, and refuses to write into the package folder")
    ap.add_argument("--exclude", action="append", default=[], metavar="CODE=REASON",
                    help="exclude a returned workbook from the primary statistics by a recorded protocol deviation "
                         "(e.g. R3=deviation 2); it stays in the sensitivity analysis over all complete workbooks")
    args = ap.parse_args()
    excluded = dict(x.split("=", 1) for x in args.exclude)
    ret, out = Path(args.returned).resolve(), Path(args.out).resolve()
    test = args.allow_script_fill
    if test and (out == HERE or HERE in out.parents):
        sys.exit("--allow-script-fill is for synthetic tests only and never writes into the package folder; give --out <scratch folder>")
    out.mkdir(parents=True, exist_ok=True)
    found, ignored = {}, []
    for p in sorted(ret.iterdir()) if ret.exists() else []:
        m = re.fullmatch(r"rater_(\d+)_judge_v4\.xlsx", p.name)
        if m and m.group(1) in SEALED["raters"]:
            found[int(m.group(1))] = p
        elif p.is_file():
            ignored.append(p.name)
    res = {"generated": datetime.now().isoformat(timespec="seconds"), "test_mode_allow_script_fill": test,
           "returned_folder": "the default folder returned/" if ret == (HERE / "returned").resolve() else "a folder given with --returned (path not stored)",
           "inputs_unchanged_since_build": INPUTS_OK,
           "ignored_files": {"count": len(ignored), "name_hashes": [sha(n) for n in ignored]},  # names never stored
           "not_returned": [f"R{k}" for k in map(int, SEALED["raters"]) if k not in found], "workbooks": {}}
    labels, private = {}, {}
    for k, p in sorted(found.items()):
        code = f"R{k}"
        data = p.read_bytes()
        w = {"file": p.name}
        try:
            wb = load_workbook(io.BytesIO(data), data_only=True)
        except Exception as e:
            w.update({"integrity": {"sha256": hashlib.sha256(data).hexdigest(), "hard": [f"cannot open: {type(e).__name__}"], "warn": []},
                      "status": "cannot open", "eligible": False, "reasons": ["cannot open"]})
            res["workbooks"][code] = w
            continue
        w["integrity"] = integrity(k, data, wb, test)
        private[code] = {"lmb": w["integrity"].pop("_last_modified_by")}
        w["fingerprint"] = fingerprint(data)
        filled = w["integrity"]["filled_input_cells"]
        w["status"] = ("not filled (byte-identical to the sent blank)" if w["integrity"]["identical_to_sent_blank"] else
                       "not filled (no input cell has a value)" if filled == 0 else "filled")
        if "Rater_info" in wb.sheetnames:
            w["rater_info"] = check_info(k, read_info(wb))
            private[code]["signature"] = w["rater_info"].pop("_signature")
        else:
            w["rater_info"] = {"ok": False, "problems": ["no Rater_info sheet"]}
        items, invalid = read_items(wb, k) if "Items" in wb.sheetnames else ({}, {})
        att = {}
        for a, e in SEALED["answer_keys"]["attention_expected_verdicts"].items():
            g = items.get(a, {}).get("verdict")
            att[a] = {"given": g, "expected": e, "wrong": g != e}
        w["attention"] = {"items": att, "pass": not all(x["wrong"] for x in att.values())}
        lab = {i: items.get(i, {}).get("verdict") for i in ITEMS}
        unlabelled = [i for i in ITEMS if lab[i] is None]
        w["completeness"] = {"unlabelled_items": unlabelled, "invalid_values": invalid,
                             "label_counts": {x: sum(lab[i] == x for i in ITEMS) for x in LAB},
                             "n_notes": sum(bool(items.get(i, {}).get("note")) for i in ITEMS)}
        reasons = []
        if filled == 0:
            reasons.append("not filled")
        if w["integrity"]["hard"]:
            reasons.append("hard integrity flag")
        if not w["rater_info"]["ok"]:
            reasons.append("Rater_info/attestations")
        if not w["attention"]["pass"]:
            reasons.append("attention items wrong on both")
        if unlabelled:
            reasons.append(f"{len(unlabelled)} unlabelled items")
        w["eligible"], w["reasons"] = not reasons, reasons
        w["labels"] = lab
        w["notes"] = {i: items[i]["note"] for i in ITEMS if items.get(i, {}).get("note")}
        res["workbooks"][code] = w
        if not unlabelled:
            labels[code] = lab
    # cross-workbook checks
    cross, info = [], []
    codes = list(res["workbooks"])
    for a, b in itertools.combinations(codes, 2):
        A, B = res["workbooks"][a], res["workbooks"][b]
        if A["integrity"].get("sha256") == B["integrity"].get("sha256"):
            cross.append(f"HARD {a} and {b} returned byte-identical files")
            for x in (A, B):
                x["eligible"] = False
                x["reasons"].append("identical file to another rater")
        pa, pb = private.get(a, {}), private.get(b, {})
        if pa.get("lmb") and pa.get("lmb") == pb.get("lmb"):
            cross.append(f"WARN {a} and {b} were last saved under the same user account (lastModifiedBy equal)")
        fa, fb = A.get("fingerprint", {}), B.get("fingerprint", {})
        if fa.get("profile_sha") and fa.get("profile_sha") == fb.get("profile_sha"):
            cross.append(f"WARN {a} and {b}: the save path Excel recorded is in the same user-profile folder")
        la, lb = A.get("labels", {}), B.get("labels", {})
        if la and lb and all(la.get(i) is not None and la.get(i) == lb.get(i) for i in ITEMS):
            cross.append(f"WARN {a} and {b}: identical verdicts on all {len(ITEMS)} items")
        na, nb = A.get("notes", {}), B.get("notes", {})
        both = [i for i in ITEMS if i in na and i in nb]
        same = [i for i in both if na[i].lower() == nb[i].lower()]
        if len(both) >= 3 and len(same) / len(both) >= 0.5:
            cross.append(f"WARN {a} and {b}: identical notes on {len(same)}/{len(both)} items where both wrote a note")
        if pa.get("signature") and pa.get("signature") == pb.get("signature"):
            cross.append(f"WARN {a} and {b}: identical signature")
    opened = [c for c in codes if "fingerprint" in res["workbooks"][c]]
    if len(opened) >= 2:
        def eq(f):  # equality among the workbooks where the value is set; the value itself is never printed
            vals = [f(c) for c in opened if f(c)]
            return f"set in {len(vals)}/{len(opened)}" + (f"; all equal: {len(set(vals)) == 1}" if len(vals) >= 2 else "")
        info = [f"Office user name (lastModifiedBy): {eq(lambda c: private[c].get('lmb'))}",
                f"user-profile folder of the save path Excel records: {eq(lambda c: res['workbooks'][c]['fingerprint']['profile_sha'])}",
                "Excel build: " + ", ".join(f"{c} {res['workbooks'][c]['fingerprint']['excel_build']}" for c in opened)]
    res["cross_checks"], res["cross_info"] = cross, info
    for c, why in excluded.items():  # exclusions by a recorded protocol deviation (PROTOCOL_judge_v4.md, Deviations)
        if c in res["workbooks"]:
            res["workbooks"][c]["eligible"] = False
            res["workbooks"][c]["reasons"].append(f"excluded by protocol {why}")
    res["excluded_by_deviation"] = excluded
    eligible = [c for c in codes if res["workbooks"][c]["eligible"] and c in labels]
    res["eligible"] = eligible
    res["reference_labels"] = reference_rows()
    if eligible:
        res["primary"] = statistics({c: labels[c] for c in eligible})
    complete = [c for c in codes if c in labels]
    if len(complete) > len(eligible) and complete:
        res["sensitivity_all_complete"] = statistics({c: labels[c] for c in complete})
    json.dump(res, open(out / "result_judge_v4.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)

    # markdown
    L = []
    if test:
        L += ["> **TEST MODE (--allow-script-fill).** Synthetic test run: the script-fill hard flag was downgraded to a warning. "
              "This is not a result of the study.", ""]
    L += ["# Judge spot-check v4: results (generated by score_judge_v4.py)", "",
          f"Generated {res['generated']} from {res['returned_folder']}. Workbooks: {', '.join(codes) or 'none'}; "
          f"not returned: {', '.join(res['not_returned']) or 'none'}; other files in the folder, not read: {len(ignored)} "
          "(names not printed; a file counts only under its sent name rater_k_judge_v4.xlsx). "
          f"All {len(INPUTS_OK)} input files (including ../judge_spotcheck_v3/sealed_key_v3.json) match their SHA-256 in SEALED_judge_build.json.", "",
          "## 1. Integrity", "", "| rater | SHA-256 | status | filled input cells | hard flags | warnings |", "|---|---|---|---:|---|---|"]
    for c, w in res["workbooks"].items():
        i = w["integrity"]
        L.append(f"| {c} | {i.get('sha256', '')[:16]} | {w['status']} | {i.get('filled_input_cells')} | "
                 f"{'; '.join(i['hard']) or 'none'} | {'; '.join(i['warn']) or 'none'} |")
    L.append("")
    for c, w in res["workbooks"].items():
        d = w["integrity"].get("docprops_diff", {})
        shown = {k: {"sent": v["sent"], "returned": ("(set)" if str(v["returned"]).startswith("(set") else v["returned"])} for k, v in d.items()}
        L.append(f"- {c} document properties changed from sent: {shown or 'none'}")
    L += ["", "Cross-workbook checks: " + ("; ".join(cross) if cross else "none") + "."]
    L += [f"- {x}" for x in info]
    L += ["", "## 2. Rater_info and attestations", "",
          "| rater | background | years | (a) alone | (b) no AI | (c) package only | total min | min from session times | problems (exclude) | notes (reported only) |",
          "|---|---|---:|---|---|---|---:|---:|---|---|"]
    for c, w in res["workbooks"].items():
        ri = w.get("rater_info", {})
        v, a = ri.get("values", {}), ri.get("attestations", {})
        L.append(f"| {c} | {v.get('background')} | {v.get('years_nlp_ir')} | {a.get('att_a')} | {a.get('att_b')} | {a.get('att_c')} | "
                 f"{v.get('total_minutes')} | {ri.get('session_minutes_from_times')} | {'; '.join(ri.get('problems', [])) or 'none'} | "
                 f"{'; '.join(ri.get('notes', [])) or 'none'} |")
    L += ["", "## 3. Attention items, completeness, eligibility", "",
          "| rater | CHECK_CORRECT (given) | CHECK_INCORRECT (given) | attention | unlabelled items | invalid values | CORRECT / INCORRECT / UNSURE | notes | eligible | reasons |",
          "|---|---|---|---|---:|---:|---|---:|---|---|"]
    for c, w in res["workbooks"].items():
        at, cp = w.get("attention", {}), w.get("completeness", {})
        it = at.get("items", {})
        lc = cp.get("label_counts", {})
        L.append(f"| {c} | {it.get('CHECK_CORRECT', {}).get('given')} | {it.get('CHECK_INCORRECT', {}).get('given')} | "
                 f"{'pass' if at.get('pass') else 'FAIL'} | {len(cp.get('unlabelled_items', []))} | {len(cp.get('invalid_values', {}))} | "
                 f"{lc.get('CORRECT')} / {lc.get('INCORRECT')} / {lc.get('UNSURE')} | {cp.get('n_notes')} | {w['eligible']} | {'; '.join(w['reasons']) or '-'} |")
    L.append("")
    if "primary" in res:
        L += report_stats(res["primary"], "4. Statistics, eligible raters (primary)")
    else:
        L += ["## 4. Statistics", "", "No eligible workbook; no statistics are reported as primary.", ""]
    if "sensitivity_all_complete" in res:
        L += report_stats(res["sensitivity_all_complete"], "Sensitivity: all returned workbooks with all 100 items labelled (NOT primary)")
    rf = res["reference_labels"]
    L += ["## Reference: earlier labels against the judge (not part of the v4 statistics)", "",
          "The first human rater is reported separately; the Claude models give LM labels, not human labels.", ""] + JUDGE_HEAD
    L += [judge_row(n, e) for n, e in rf["rows"].items()]
    L += ["", "Reproduces the earlier results: " + "; ".join(f"{k}: {v}" for k, v in rf["reproduces_earlier_results"].items()) + ".", ""]
    hs = [c for c in codes if c in labels]
    if hs:
        maj = res.get("primary", {}).get("majority", {}).get("labels")
        ab = {"CORRECT": "C", "INCORRECT": "I", "UNSURE": "U", None: "-"}
        cols = hs + (["majority (primary)"] if maj else []) + ["v1 human", "opus-5", "sonnet-5", "haiku-4-5"]
        L += ["## Per item (C = CORRECT, I = INCORRECT, U = UNSURE)", "", "| item | reader | judge | " + " | ".join(cols) + " |",
              "|---|---|---|" + "---|" * len(cols)]
        for i in ITEMS:
            row = [ab[labels[c][i]] for c in hs] + ([ab[maj[i]]] if maj else [])
            row += [ab[V1_HUMAN[i]]] + [ab[CLAUDE_LABELS[m][i]] for m in CLAUDE]
            L.append(f"| {i} | {READERS[READER[i]]} | {ab[JUDGE[i]]} | " + " | ".join(row) + " |")
    (out / "result_judge_v4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
