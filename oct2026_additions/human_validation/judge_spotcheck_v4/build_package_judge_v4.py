# -*- coding: utf-8 -*-
"""Build the judge spot-check v4 human-rater package (plan: PROTOCOL_judge_v4.md, written 2026-10-01T12:26Z
before this build).

One workbook per rater (sheets Read_me, Rater_info, Items). Items holds the 100 GraphRAG-Bench (question,
reference answer, candidate answer) triples S001-S100 of ../judge_spotcheck_v3/sealed_key_v3.json, with the
texts of ../judge_spotcheck/judge_spotcheck_sheet_blank.csv (row i = S{i:03d}), plus two invented attention
items, shuffled per rater (seed 2026100100 + k). A row shows a running number, a per-rater random row code,
and the three texts; the input columns are verdict (CORRECT / INCORRECT / UNSURE) and note. Every sheet is
protected; only input cells are unlocked. The password, seeds, orders, row codes, file hashes, locked-cell
hashes, item-text hashes, and the attention items' expected verdicts go to SEALED_judge_build.json, which is
never sent. The judge verdicts are not read: of sealed_key_v3.json only the v1 key of each id is used (to
check the id-to-text mapping), and the file's SHA-256 is recorded.

Copied from ../v4/build_package_v4.py (2026-09-30) and kept unchanged: the Rater_info rows (same keys and
field texts; the notes of 'years', 'total minutes', and the session-time examples are reworded), relation options, the protection,
drop-down, document-property, zip-normalisation, snapshot, and hashing helpers.

The build is deterministic: fixed document times, fixed zip entry times, seeds, and the password kept in
SEALED_judge_build.json (created once, reused on every rebuild). A rebuild reproduces every file byte for byte.

Outputs (human_validation/judge_spotcheck_v4/):
  to_send/rater_{1,2,3}/{rater_k_judge_v4.xlsx, GUIDE_v4.md}
  to_send/rater_{1,2,3}_judge_v4.zip
  SEALED_judge_build.json
Usage:
  python build_package_judge_v4.py           # build (reuses the password in SEALED_judge_build.json if present)
  python build_package_judge_v4.py --check   # rebuild in memory and compare with the files on disk; writes nothing
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import secrets
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.utils.protection import hash_password
from openpyxl.workbook.protection import WorkbookProtection
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.protection import SheetProtection

HERE = Path(__file__).resolve().parent
HV = HERE.parent
V1 = HV / "judge_spotcheck"
V3 = HV / "judge_spotcheck_v3"
LLM3 = HV / "llm_spotcheck_v3"
TO_SEND = HERE / "to_send"
SEALED = HERE / "SEALED_judge_build.json"
GUIDE = HERE / "GUIDE_v4.md"

BUILD_TIME = datetime(2026, 10, 1, 12, 30, 0)  # UTC, written to docProps created/modified
ZIP_TIME = (2026, 10, 1, 12, 30, 0)
SEED_BASE = 2026100100
RATERS = (1, 2, 3)
SHEETS = ["Read_me", "Rater_info", "Items"]
VERDICTS = ["CORRECT", "INCORRECT", "UNSURE"]
CODE_ALPHABET = "ACDEFGHJKLMNPQRTUVWXY34679"
N_ITEMS = 100

# ----------------------------------------------------------------------------- attention items (invented)
ATTENTION = [  # excluded from all statistics
    {"item_id": "CHECK_CORRECT",
     "question": "Which organ is affected first in Halvren disease?",
     "reference": "Halvren disease affects the liver first; the spleen can be involved later in the course.",
     "candidate": "Halvren disease affects the liver first.",
     "expected": "CORRECT"},
    {"item_id": "CHECK_INCORRECT",
     "question": "Which enzyme is deficient in patients with Ostrik syndrome?",
     "reference": "Patients with Ostrik syndrome have a deficiency of the enzyme glucokinase.",
     "candidate": "Ostrik syndrome is caused by a deficiency of the enzyme lactase.",
     "expected": "INCORRECT"},
]

# ----------------------------------------------------------------------------- rater-facing text
KOREAN_SUMMARY = ("**한국어 요약.** 한 행마다 질문, 참조 답(reference), 모델이 낸 답(candidate)이 있습니다. 모델 답이 참조 답의 "
                  "핵심 사실(질문에 실제로 답하는 부분)을 말하면 CORRECT, 아니면 INCORRECT입니다. 두 글만 비교해서는 판단이 안 될 "
                  "때만 UNSURE를 고릅니다. 의학 지식은 필요 없고 쓰지도 마세요. 혼자 평정하고, ChatGPT·Claude·Gemini 같은 AI 도구나 "
                  "번역기, 인터넷 검색은 쓰지 마세요. 다른 평정자와 문항을 두고 이야기하지 마세요. 시작·종료 시각과 확인란 3개를 "
                  "`Rater_info`에 채운 뒤, 파일 이름을 바꾸지 말고 그대로 회신해 주세요.")
READ_ME = [
    "Answer-grading spot-check v4: rater workbook",
    "Rater code: {code}",
    "",
    "1. Before you start, read GUIDE_v4.md (in the same folder as this workbook).",
    "2. Fill Rater_info first, then the Items sheet (102 rows). Only the yellow cells can be edited.",
    "3. In each Items row, compare the candidate answer with the reference answer and choose CORRECT, INCORRECT, or UNSURE in the verdict column, following the rules in GUIDE_v4.md. The note column is optional; add a short note when you choose UNSURE.",
    "4. Rate alone. Do not discuss the items with anyone, including other raters, and do not look at another rater's workbook.",
    "5. Do not use AI assistants or language models in any way (e.g., ChatGPT, Claude, Gemini, Copilot, AI features in Excel or browsers, or translators built on them).",
    "6. Do not search the web and do not use outside medical knowledge. Decide each row only from the texts on that row.",
    "7. Use only GUIDE_v4.md and this workbook. Do not look at the results of the paper this study belongs to.",
    "8. Use desktop Excel or LibreOffice Calc. Do not upload the workbook to an online service, copy it into a new file, or rename it.",
    "9. When you are done, send this workbook as an email or messenger attachment directly to the coordinator.",
    "",
    "Expected time: 40-60 minutes.",
]
# Rater_info: the rows of ../v4/build_package_v4.py (same keys, fields, and kinds)
RELATION_OPTIONS = ["not an author, not involved in the project", "author of the paper", "involved in the project in another way"]
INFO_ROWS = [  # (key, field shown, note, kind)
    ("rater_code", "rater code", "given by the coordinator", "given"),
    ("background", "field / background", "e.g. NLP PhD student, IR engineer", "text"),
    ("years_nlp_ir", "years working in NLP or IR", "a number (0 if none)", "number"),
    ("relation", "relation to the authors", "choose from the list; must be: not an author, not involved in the project", "relation"),
    ("s1_start", "session 1 start (date and time)", "e.g. 2026-10-02 14:05", "text"),
    ("s1_end", "session 1 end (date and time)", "e.g. 2026-10-02 14:55", "text"),
    ("s2_start", "session 2 start (date and time, optional)", "only if you used a second session", "optional"),
    ("s2_end", "session 2 end (date and time, optional)", "only if you used a second session", "optional"),
    ("total_minutes", "total minutes spent rating", "a number of minutes, e.g. 50", "number"),
    ("att_a", "(a) I rated alone, without discussing items with anyone", "yes or no", "yesno"),
    ("att_b", "(b) I used no AI assistant or language model (e.g., ChatGPT, Claude, Gemini, Copilot) at any point", "yes or no", "yesno"),
    ("att_c", "(c) I used only the materials in this package and did not look at the paper's results or any other rater's sheet", "yes or no", "yesno"),
    ("signature", "signature / initials", "", "text"),
    ("comments", "comments (optional)", "anything you want to tell us", "optional"),
]
ROW_HEAD = ["no.", "row_code", "question", "reference_answer", "candidate_answer (model output, verbatim)", "verdict", "note (optional)"]
ROW_WIDTHS = [5, 8, 40, 50, 46, 13, 30]
TEXT_COLS = ["C", "D", "E"]  # displayed item text, hashed per item
INPUT_COLS = ["F", "G"]
LINE_CHARS = {"C": 38, "D": 48, "E": 44}  # conservative characters per wrapped line at Calibri 10
# words that must not appear in a rater-facing file ('_judge_v4' in the file name is removed before checking)
FORBIDDEN = ["mistral", "sealed", "judge", "pilot", "50/50", "kappa", "precision", "recall", "qwen", "llama",
             "graphrag", "llm_spotcheck", "spotcheck_v3", "v1_key", "pool"]

HEAD_FILL = PatternFill("solid", fgColor="DDEBF7")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
UNLOCKED = Protection(locked=False)
FONT = Font(name="Calibri", size=10)
BOLD = Font(name="Calibri", size=10, bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ----------------------------------------------------------------------------- hashing (as in v4)
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cell_hash(v) -> str:
    return sha_text(f"{type(v).__name__}:{v}")[:16]


def item_hash(question: str, reference: str, candidate: str) -> str:
    return sha_text("\x1f".join([question, reference, candidate]))


# ----------------------------------------------------------------------------- items
def blank_v3_sheet(k: int) -> Path:
    """Blank v3 sheet of rater k, under either of its two file names (same bytes)."""
    for name in (f"rater_{k}_spotcheck.xlsx", f"rater_{k}_spotcheck_og.xlsx"):
        if (V3 / "sheets" / name).exists():
            return V3 / "sheets" / name
    raise FileNotFoundError(f"no blank v3 sheet for rater {k} in {V3 / 'sheets'}")


def load_items() -> list[dict]:
    rows = list(csv.DictReader(open(V1 / "judge_spotcheck_sheet_blank.csv", encoding="utf-8-sig")))
    assert [int(r["position"]) for r in rows] == list(range(1, N_ITEMS + 1))
    key = json.load(open(V3 / "sealed_key_v3.json", encoding="utf-8"))["items"]
    v1_keys = {i: v["v1_key"] for i, v in key.items()}  # only the v1 key is used; no verdict is read
    del key
    assert sorted(v1_keys) == [f"S{i:03d}" for i in range(1, N_ITEMS + 1)]
    items = []
    for i, r in enumerate(rows, 1):
        iid = f"S{i:03d}"
        assert v1_keys[iid] == r["key"], iid
        items.append({"item_id": iid, "v1_key": r["key"], "question": r["question"], "reference": r["gold_answer"],
                      "candidate": r["candidate_answer"]})
    # parity: the same texts as the v3 workbooks the Claude models labelled
    by_id = {x["item_id"]: (x["question"], x["reference"], x["candidate"]) for x in items}
    for p in [blank_v3_sheet(k) for k in RATERS] + \
             [LLM3 / "returned" / f"rater_{k}_spotcheck.xlsx" for k in RATERS]:
        wb = load_workbook(p, read_only=True)
        vals = list(wb["SpotCheck"].iter_rows(values_only=True))
        wb.close()
        got = {v[1]: tuple("" if x is None else str(x) for x in v[2:5]) for v in vals[1:] if v[1]}
        assert got == by_id, f"text differs from {p.name}"
    for x in items + ATTENTION:
        for f in ("question", "reference", "candidate"):
            t = x[f]
            assert t and not ILLEGAL.search(t) and not t.startswith("=") and t == t.strip() and "\r" not in t, (x["item_id"], f)
        x["sha256"] = item_hash(x["question"], x["reference"], x["candidate"])
    return items


# ----------------------------------------------------------------------------- workbook (helpers as in v4)
def est_lines(text: str, chars: int) -> int:
    return sum(max(1, math.ceil(len(p) / chars)) for p in str(text).split("\n"))


def protect(ws, password: str):
    ws.protection = SheetProtection(sheet=True, objects=True, scenarios=True, formatCells=True,
                                    formatColumns=False, formatRows=False, insertColumns=True, insertRows=True,
                                    insertHyperlinks=True, deleteColumns=True, deleteRows=True,
                                    selectLockedCells=False, selectUnlockedCells=False, sort=True,
                                    autoFilter=True, pivotTables=True)
    ws.protection.set_password(password)


def add_list(ws, cells: str, options: str, prompt: str):
    dv = DataValidation(type="list", formula1=options, allow_blank=True, showDropDown=False,
                        showErrorMessage=True, showInputMessage=True, errorStyle="stop")
    dv.prompt = prompt
    dv.error = "Choose a value from the list."
    ws.add_data_validation(dv)
    dv.add(cells)


def read_me_sheet(wb, k: int, password: str):
    ws = wb.create_sheet("Read_me")
    for line in READ_ME:
        ws.append([line.format(code=f"R{k}") if line else None])
    ws.column_dimensions["A"].width = 130
    for row in ws.iter_rows():
        for c in row:
            c.font = FONT
            c.alignment = WRAP
    ws["A1"].font = Font(name="Calibri", size=12, bold=True)
    ws["A2"].font = BOLD
    for i, line in enumerate(READ_ME, 1):
        ws.row_dimensions[i].height = 13.5 * est_lines(line, 125) + 3
    protect(ws, password)


def rater_info_sheet(wb, k: int, password: str):
    ws = wb.create_sheet("Rater_info")
    ws.append(["field", "your answer", "format / note"])
    for key, field, note, kind in INFO_ROWS:
        ws.append([field, f"R{k}" if kind == "given" else None, note or None])
    for i, opt in enumerate(RELATION_OPTIONS, 2):  # list source for the relation drop-down (hidden column)
        ws[f"E{i}"] = opt
    ws.column_dimensions["E"].hidden = True
    for col, w in zip("ABC", (70, 44, 48)):
        ws.column_dimensions[col].width = w
    for c in ws[1]:
        c.font = BOLD
        c.fill = HEAD_FILL
        c.alignment = WRAP
    for i, (key, field, note, kind) in enumerate(INFO_ROWS, 2):
        for col in "ABC":
            ws[f"{col}{i}"].font = FONT
            ws[f"{col}{i}"].alignment = WRAP
        ws.row_dimensions[i].height = 28
        if kind != "given":
            ws[f"B{i}"].fill = INPUT_FILL
            ws[f"B{i}"].protection = UNLOCKED
        if kind == "yesno":
            add_list(ws, f"B{i}", '"yes,no"', "yes or no")
        elif kind == "relation":
            add_list(ws, f"B{i}", f"$E$2:$E${1 + len(RELATION_OPTIONS)}", "choose one")
        elif kind == "number":
            dv = DataValidation(type="decimal", operator="between", formula1="0", formula2="100000", allow_blank=True,
                                showErrorMessage=True, errorStyle="stop")
            dv.error = "Enter a number."
            ws.add_data_validation(dv)
            dv.add(f"B{i}")
    protect(ws, password)


def items_sheet(wb, rows: list[dict], codes: list[str], password: str):
    ws = wb.create_sheet("Items")
    ws.append(ROW_HEAD)
    for i, (r, c) in enumerate(zip(rows, codes), 1):
        ws.append([i, c, r["question"], r["reference"], r["candidate"], None, None])
    for j, w in enumerate(ROW_WIDTHS):
        ws.column_dimensions[chr(65 + j)].width = w
    for c in ws[1]:
        c.font = BOLD
        c.fill = HEAD_FILL
        c.alignment = WRAP
    ws.row_dimensions[1].height = 28
    n = len(rows)
    for i in range(2, n + 2):
        for col in "ABCDE":
            ws[f"{col}{i}"].font = FONT
            ws[f"{col}{i}"].alignment = WRAP
        for col in TEXT_COLS:
            ws[f"{col}{i}"].number_format = "@"  # keep literal model output such as None as text
        for col in INPUT_COLS:
            c = ws[f"{col}{i}"]
            c.fill = INPUT_FILL
            c.protection = UNLOCKED
            c.font = FONT
            c.alignment = WRAP
        lines = max(max(est_lines(ws[f"{col}{i}"].value, ch) for col, ch in LINE_CHARS.items()), 2)
        ws.row_dimensions[i].height = min(409, 13.5 * lines + 6)
    add_list(ws, f"F2:F{n + 1}", '"' + ",".join(VERDICTS) + '"', "CORRECT, INCORRECT, or UNSURE (GUIDE_v4.md)")
    ws.freeze_panes = "A2"
    protect(ws, password)


def build_workbook(k: int, rows: list[dict], codes: list[str], password: str) -> Workbook:
    wb = Workbook()
    wb.remove(wb.active)
    read_me_sheet(wb, k, password)
    rater_info_sheet(wb, k, password)
    items_sheet(wb, rows, codes, password)
    assert wb.sheetnames == SHEETS
    wb.active = 0
    wb.security = WorkbookProtection(lockStructure=True)
    wb.security.set_workbook_password(password)
    p = wb.properties
    p.creator = "coordinator"
    p.lastModifiedBy = None
    p.created = BUILD_TIME
    p.modified = BUILD_TIME
    p.title = "Answer-grading spot-check v4"
    return wb


def normalize_zip(data: bytes) -> bytes:
    """Rewrite a zip with fixed entry times and attributes so identical content gives identical bytes."""
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 0
            zi.external_attr = 0
            z.writestr(zi, src.read(info.filename))
    return out.getvalue()


def docprops(data: bytes) -> dict:
    z = zipfile.ZipFile(io.BytesIO(data))
    out = {}
    for part in ("docProps/core.xml", "docProps/app.xml"):
        root = ET.fromstring(z.read(part))
        out[part.split("/")[1].split(".")[0]] = {el.tag.split("}")[-1]: (el.text or "") for el in root if len(el) == 0}
    return out


def snapshot(wb) -> tuple[dict, dict]:
    locked, inputs = {}, {}
    for ws in wb.worksheets:
        locked[ws.title], inputs[ws.title] = {}, []
        for (r, col), c in sorted(ws._cells.items()):
            if c.protection.locked:
                if c.value is not None:
                    locked[ws.title][c.coordinate] = cell_hash(c.value)
            else:
                inputs[ws.title].append(c.coordinate)
    return locked, inputs


def forbidden_hits(text: str) -> list[str]:
    t = text.lower().replace("_judge_v4", "")
    return [w for w in FORBIDDEN if re.search(rf"\b{re.escape(w)}", t)]  # word start, so 'Copilot' is not 'pilot'


def leak_check(data: bytes, items: list[dict], password: str):
    z = zipfile.ZipFile(io.BytesIO(data))
    blob = "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist())
    bad = [x["item_id"] for x in items if re.search(rf"\b{x['item_id']}\b", blob)]
    bad += [x["v1_key"] for x in items if x["v1_key"] in blob]
    bad += [x["item_id"] for x in ATTENTION if x["item_id"] in blob]
    bad += (["password"] if password in blob else []) + forbidden_hits(blob)
    assert not bad, f"leak in workbook: {bad}"


def check_guide(guide: str, items: list[dict], password: str):
    v3 = (V3 / "GUIDE_v3.md").read_text(encoding="utf-8")
    rules_and_examples = v3[v3.index("## Rules"):v3.index("\nReturn the workbook")].rstrip("\n")
    assert rules_and_examples in guide, "GUIDE_v4.md: rules 1-6 and the example table must be verbatim from GUIDE_v3.md"
    assert KOREAN_SUMMARY in guide, "GUIDE_v4.md: the Korean summary must be the fixed text"
    assert "40-60 minutes" in guide
    bad = [x["item_id"] for x in items if x["item_id"] in guide] + [x["v1_key"] for x in items if x["v1_key"] in guide]
    bad += (["password"] if password in guide else []) + forbidden_hits(guide)
    assert not bad, f"leak in GUIDE_v4.md: {bad}"


# ----------------------------------------------------------------------------- build
def build(password: str) -> tuple[dict[str, bytes], dict]:
    """Return {relative path: bytes} of every sent file and the sealed record (nothing is written)."""
    items = load_items()
    all_rows = items + ATTENTION
    guide = GUIDE.read_bytes()
    check_guide(guide.decode("utf-8"), items, password)
    out: dict[str, bytes] = {}
    sealed = {
        "_comment": "SEALED. Never send this file or show it to a rater or the coordinator. It holds the sheet password, "
                    "seeds, row orders and codes, the attention items' expected verdicts, and the hashes score_judge_v4.py "
                    "checks returned workbooks against. It holds no judge verdict: those stay in ../judge_spotcheck_v3/sealed_key_v3.json.",
        "protocol": "PROTOCOL_judge_v4.md (written 2026-10-01T12:26Z, before this build)",
        "build_time_utc": BUILD_TIME.isoformat() + "Z",
        "openpyxl_version": openpyxl.__version__,
        "password": password,
        "password_hash_legacy": hash_password(password),
        "sheet_order": SHEETS,
        "item_columns": {"header": ROW_HEAD, "text_cols": TEXT_COLS, "input_cols": INPUT_COLS, "verdicts": VERDICTS},
        "rater_info_rows": [{"row": i, "key": key, "field": field, "kind": kind} for i, (key, field, _, kind) in enumerate(INFO_ROWS, 2)],
        "relation_required": RELATION_OPTIONS[0],
        "items": [x["item_id"] for x in items],
        "attention_items": [x["item_id"] for x in ATTENTION],
        "item_text_sha256": {x["item_id"]: x["sha256"] for x in all_rows},
        "answer_keys": {"attention_expected_verdicts": {x["item_id"]: x["expected"] for x in ATTENTION}},
        "sealed_key_v3_sha256": sha256_bytes((V3 / "sealed_key_v3.json").read_bytes()),
        "inputs_sha256": {p: sha256_bytes((HV / p).read_bytes()) for p in
                          ["judge_spotcheck_v3/sealed_key_v3.json", "judge_spotcheck_v3/pool_stats.json",
                           "judge_spotcheck_v3/GUIDE_v3.md", "judge_spotcheck/judge_spotcheck_sheet_blank.csv",
                           "judge_spotcheck/spotcheck_result.json", "llm_spotcheck_v3/raw.jsonl",
                           "llm_spotcheck_v3/result_v3.json", "judge_spotcheck_v4/GUIDE_v4.md"]},
        "raters": {},
    }
    for k in RATERS:
        seed = SEED_BASE + k
        rng = random.Random(seed)
        rows = list(all_rows)
        rng.shuffle(rows)
        codes = []
        while len(codes) < len(rows):
            code = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
            if code not in codes:
                codes.append(code)
        wb = build_workbook(k, rows, codes, password)
        locked, inputs = snapshot(wb)
        buf = io.BytesIO()
        wb.save(buf)
        data = normalize_zip(buf.getvalue())
        leak_check(data, items, password)
        files = {f"rater_{k}_judge_v4.xlsx": data, "GUIDE_v4.md": guide}
        for name, b in files.items():
            out[f"to_send/rater_{k}/{name}"] = b
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):
                zi = zipfile.ZipInfo(f"rater_{k}/{name}", date_time=ZIP_TIME)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.create_system = 0
                zi.external_attr = 0
                z.writestr(zi, files[name])
        out[f"to_send/rater_{k}_judge_v4.zip"] = zbuf.getvalue()
        sealed["raters"][str(k)] = {
            "rater_code": f"R{k}", "seed": seed,
            "order": [r["item_id"] for r in rows], "codes": dict(zip(codes, [r["item_id"] for r in rows])),
            "files_sha256": {p: sha256_bytes(b) for p, b in out.items() if p.startswith((f"to_send/rater_{k}/", f"to_send/rater_{k}_"))},
            "docprops_sent": docprops(data),
            "locked_cells": locked, "input_cells": inputs,
        }
    return out, sealed


def sealed_bytes(sealed: dict) -> bytes:
    return (json.dumps(sealed, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the files on disk; write nothing")
    args = ap.parse_args()
    prev = json.load(open(SEALED, encoding="utf-8")) if SEALED.exists() else {}
    if args.check:
        if not prev:
            sys.exit("no SEALED_judge_build.json to check against")
        if openpyxl.__version__ != prev.get("openpyxl_version"):
            print(f"NOTE: the package was built with openpyxl {prev.get('openpyxl_version')}, this is {openpyxl.__version__}; "
                  "workbook bytes can differ for that reason alone")
        out, sealed = build(prev["password"])
        bad = []
        on_disk = sorted(str(p.relative_to(HERE)).replace("\\", "/") for p in TO_SEND.rglob("*") if p.is_file())
        if on_disk != sorted(out):
            bad.append(f"file list differs: on disk {on_disk}, rebuilt {sorted(out)}")
        for p, b in out.items():
            f = HERE / p
            if not f.exists() or f.read_bytes() != b:
                bad.append(f"differs: {p}")
        if SEALED.read_bytes() != sealed_bytes(sealed):
            bad.append("differs: SEALED_judge_build.json")
        for p, b in out.items():
            print(f"  {sha256_bytes(b)}  {p}")
        if bad:
            print("REBUILD CHECK FAILED:\n  " + "\n  ".join(bad))
            sys.exit(1)
        print(f"rebuild check passed: {len(out)} sent files and SEALED_judge_build.json reproduced byte for byte")
        return
    password = prev.get("password") or secrets.token_urlsafe(9)
    out, sealed = build(password)
    if TO_SEND.exists():
        shutil.rmtree(TO_SEND)
    for p, b in out.items():
        (HERE / p).parent.mkdir(parents=True, exist_ok=True)
        (HERE / p).write_bytes(b)
    SEALED.write_bytes(sealed_bytes(sealed))
    print(f"built judge spot-check v4 package for raters {list(RATERS)}: {N_ITEMS} items + {len(ATTENTION)} attention items each")
    for p, b in out.items():
        print(f"  {sha256_bytes(b)}  {p}")


if __name__ == "__main__":
    main()
