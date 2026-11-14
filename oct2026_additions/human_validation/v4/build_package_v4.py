# -*- coding: utf-8 -*-
"""Build the v4 human-rater package (plan: PROTOCOL_v4.md, written 2026-09-30T10:27Z before this build).

One workbook per rater (sheets Read_me, Rater_info, Quiz, Census). Census rows show only the benchmark
name, the axis, a one-line neutral axis definition, and the verified verbatim excerpts that the logged
LM excerpt-only condition saw (../llm_census_logged/excerpts.json), with source kind, version, and
page or code line. Rows are shuffled per rater (seed 2026093000 + k). Every sheet is protected; only
input cells are unlocked. The password, seeds, orders, row codes, file hashes, locked-cell hashes,
item-text hashes, and answer keys go to SEALED_build.json, which is never sent.

Reused from v3 unchanged: the six quiz items and their key, the two attention items, the six worked
examples (constants copied from ../v3/build_package_v3.py and checked against ../v3/answer_key_v3.json).

The build is deterministic: fixed document times, fixed zip entry times, seeds, and the password kept in
SEALED_build.json (created once, reused on every rebuild). A rebuild reproduces every file byte for byte.

Outputs (human_validation/v4/):
  WORKED_EXAMPLES_v4.md
  to_send/rater_{1,2,3}/{rater_k_v4.xlsx, INSTRUCTIONS_v4.md, WORKED_EXAMPLES_v4.md}
  to_send/rater_{1,2,3}_v4.zip
  SEALED_build.json
Usage: python build_package_v4.py
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import secrets
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.utils.protection import hash_password
from openpyxl.workbook.protection import WorkbookProtection
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.protection import SheetProtection

HERE = Path(__file__).resolve().parent
HV = HERE.parent
V3 = HV / "v3"
LOGGED = HV / "llm_census_logged"
TO_SEND = HERE / "to_send"
SEALED = HERE / "SEALED_build.json"

BUILD_TIME = datetime(2026, 9, 30, 10, 30, 0)  # UTC, written to docProps created/modified
ZIP_TIME = (2026, 9, 30, 10, 30, 0)
SEED_BASE = 2026093000
RATERS = (1, 2, 3)
SHEETS = ["Read_me", "Rater_info", "Quiz", "Census"]
VERIFIED_OK = {"EXACT", "LOOSE", "LOOSE-INTERRUPTED"}
CODE_SOURCES = {"upper", "lower", "lower_test"}
CODE_ALPHABET = "ACDEFGHJKLMNPQRTUVWXY34679"

# ----------------------------------------------------------------------------- reused from v3 (verbatim)
ANCHORS = [  # attention items, excluded from all statistics
    {"item_id": "CHECK_TEXT", "benchmark": "DocQA-N", "axis": "supporting sentence",
     "description": "Annotators read a news article, write a question about it, and highlight the sentence that answers it. No knowledge graph is used, and nothing is selected by matching the answer string.",
     "quote": "", "expected_bin": 0},
    {"item_id": "CHECK_KG", "benchmark": "KGFact-M", "axis": "supporting edge",
     "description": "Each gold supporting fact is a triple exported unchanged from the evaluation knowledge graph by a script; no human or model edits or filters the triples.",
     "quote": "", "expected_bin": 4},
]
QUIZ = [
    {"qid": "Q1", "benchmark": "Quiz-A", "axis": "answer",
     "description": "Annotators write gold logical forms for each question; the gold answers are the entities returned by executing those logical forms against the evaluation knowledge base.",
     "key": {"source": "K", "base": 4, "corr": "none", "final": 4},
     "why": "The answer is read off K by executing a query. Human-written parses do not make it independent; the humans wrote the query, K produced the answer."},
    {"qid": "Q2", "benchmark": "Quiz-B", "axis": "supporting sentence",
     "description": "A sentence is marked as a supporting fact if it contains the object value of the question's knowledge-graph triple.",
     "key": {"source": "K", "base": 3, "corr": "none", "final": 3},
     "why": "A KG value is projected onto text by a matching heuristic: generated from K, not copied (the sentence is text), so bin 3."},
    {"qid": "Q3", "benchmark": "Quiz-C", "axis": "evidence document",
     "description": "For each question, web documents are retrieved with a search engine using the question text; a retrieved document is kept as gold evidence if it contains the answer string.",
     "key": {"source": "A", "base": 3, "corr": "none", "final": 3},
     "why": "Distant supervision: search proposes candidates, answer-string matching keeps them. Not 'answer string alone' (that would be bin 4)."},
    {"qid": "Q4", "benchmark": "Quiz-D", "axis": "reasoning path",
     "description": "Gold reasoning paths are sampled from the knowledge graph after links discovered by embedding-based clustering (not present in the original graph) have been added to it.",
     "key": {"source": "K", "base": 4, "corr": "B3-1", "final": 3},
     "why": "Paths are copied from the graph (base 4), but the graph was augmented beyond K (B3-1 lowers 4 to 3)."},
    {"qid": "Q5", "benchmark": "Quiz-E", "axis": "answer",
     "description": "A language model writes questions and answers from news paragraphs; each gold answer is a span of its source paragraph. No knowledge graph is involved, and a sample of the items is manually reviewed.",
     "key": {"source": "T", "base": 0, "corr": "none", "final": 0},
     "why": "Automatic generation from source text is provenance-independent. A review of an independent item cannot lower bin 0."},
    {"qid": "Q6", "benchmark": "Quiz-F", "axis": "gold passage",
     "description": "Annotators choose the gold passage from a shortlist of five passages that a knowledge-graph entity linker proposed for the question; the final choice and the answer span are made by the annotator from the text.",
     "key": {"source": "T", "base": 1, "corr": "none", "final": 1},
     "why": "Human text gold whose candidates were guided by the KG: bin 1."},
]
WORKED = [  # appear in WORKED_EXAMPLES_v4.md with full answers (different from the quiz)
    ("Crowd workers write questions about a Wikipedia paragraph and mark the answer span in it.", "T", 0, "none", 0,
     "Human annotation of source text; no K, no answer-string selection."),
    ("Every corpus passage that contains the gold answer string is labeled a gold passage.", "A", 4, "none", 4,
     "The answer string alone decides the gold."),
    ("In a product-catalog KG, a script picks a (product, made_by, company) triple, writes 'Who makes <product>?', and stores the company node as the gold answer.", "K", 4, "none", 4,
     "The gold answer is a KG node copied as-is; generating the question text does not change that."),
    ("Supporting passages are written by a language model from KG triples, then crowd workers check each passage and delete those that misstate the triple.", "K", 3, "B3-2", 2,
     "Generated from K (base 3); human validation that removes gold lowers one step (B3-2)."),
    ("In a recipe KG, gold ingredient lists are copied from the KG for each dish, and annotators delete ingredients they judge wrong before release.", "K", 4, "B3-2", 3,
     "Copied from K (base 4); human refinement that removes or edits gold lowers one step (B3-2)."),
    ("Human evaluators rate the fluency of system answers; the gold answers were copied from the KG.", "K", 4, "B3-3 (no change)", 4,
     "Evaluating system outputs does not touch the gold: B3-3, no change."),
]

# ----------------------------------------------------------------------------- v4 additions
AXIS_DEF = {  # what the unit is, never how it was produced
    "supporting passage": "The passage(s) marked as the gold support that a system should retrieve or use to answer the question.",
    "answer": "The gold answer that a system's answer to the question is scored against.",
    "supporting sentence": "The sentence(s) marked as the gold supporting facts for answering the question.",
    "supporting evidence": "The pieces of text marked as the gold evidence for answering the question.",
    "short answer": "The short gold answer (for example an entity, a short span, or yes/no) that a system's answer is scored against.",
    "long-answer passage": "The passage (for example a paragraph) marked as the gold long answer for the question.",
    "retrieval chunk": "The text chunk(s) marked as the gold retrieval target for the question.",
    "supporting-fact edge set": "The set of relation facts between entities (edges) marked as the gold supporting facts for the question.",
    "textual property": "The free-text attribute(s) of the answer entity that the question mentions (the textual part of what the question asks for).",
    "answer entity": "The entity or entities marked as the gold answer to the question.",
    "evidence document": "The document(s) marked as the gold evidence for answering the question.",
    "evidence subgraph": "The set of connected entity-relation facts marked as the gold evidence for the question.",
    "evidence triple": "The (subject, relation, object) facts marked as the gold evidence for the question.",
    "reasoning path": "The chain of entities and relations marked as the gold route from the question's entity to the answer.",
    "inferential chain": "The chain of relations marked as the gold link between the question's entity and its answer.",
    "gold edge": "The relation fact(s) between two entities (edges) marked as gold for the question.",
    "gold entity": "The entities marked as the gold entities of the question.",
    "gold path": "The sequence of entities marked as the gold path for the question.",
    "supporting edge": "The relation facts between entities (edges) marked as the gold supporting facts for the question.",
    "gold passage": "The passage marked as the gold passage for the question.",
}
SRC = {  # source key of excerpts.json -> (kind, version shown to raters)
    "musique": ("benchmark paper", "arXiv:2108.00573v3"),
    "hotpotqa": ("benchmark paper", "arXiv:1809.09600v1"),
    "multihoprag": ("benchmark paper", "arXiv:2401.15391v1"),
    "nq": ("benchmark paper", "ACL Anthology Q19-1026"),
    "triviaqa": ("benchmark paper", "arXiv:1705.03551v2"),
    "grade_v1": ("benchmark paper", "arXiv:2508.16994v1"),
    "stark_v1": ("benchmark paper", "arXiv:2404.13207v1"),
    "graphragbench": ("benchmark paper", "arXiv:2506.05690v3"),
    "twowiki": ("benchmark paper", "arXiv:2011.01060v2"),
    "metaqa": ("benchmark paper", "arXiv:1709.04071v5"),
    "metaqa_readme": ("benchmark README", "github.com/yuyuz/MetaQA README.md at commit b1e69733 (2021-10-10)"),
    "webqsp": ("benchmark paper", "ACL Anthology P16-2033"),
    "upper": ("benchmark generator source code", "file MD5 5047c312"),
    "lower": ("benchmark generator source code", "file MD5 b384f0e9"),
    "lower_test": ("benchmark generator source code", "file MD5 231ac78d"),
}
READ_ME = [
    "Provenance rating study v4: rater workbook",
    "Rater code: {code}",
    "",
    "1. Before you start, read INSTRUCTIONS_v4.md and WORKED_EXAMPLES_v4.md (in the same folder as this workbook).",
    "2. Fill the sheets in order: Rater_info, Quiz (6 rows), Census (31 rows). Only the yellow cells can be edited.",
    "3. Rate alone. Do not discuss items with anyone, including other raters, and do not look at another rater's workbook.",
    "4. Do not use AI assistants or language models in any way (e.g., ChatGPT, Claude, Gemini, Copilot, or AI features in Excel, browsers, or translation tools).",
    "5. Do not search the web for this task. Judge each row only from what the workbook shows.",
    "6. Use only the materials in this package. Do not look at the results of the paper this study belongs to.",
    "7. Use desktop Excel or LibreOffice Calc. Do not upload the workbook to an online service, copy it into a new file, or rename it.",
    "8. When you are done, send this workbook as an email or messenger attachment directly to the coordinator.",
    "",
    "Expected time: 60-100 minutes, in one or two sessions.",
]
RELATION_OPTIONS = ["not an author, not involved in the project", "author of the paper", "involved in the project in another way"]
INFO_ROWS = [  # (key, field shown, note, kind)
    ("rater_code", "rater code", "given by the coordinator", "given"),
    ("background", "field / background", "e.g. NLP PhD student, IR engineer", "text"),
    ("years_nlp_ir", "years working in NLP or IR", "a number", "number"),
    ("relation", "relation to the authors", "choose from the list; must be: not an author, not involved in the project", "relation"),
    ("s1_start", "session 1 start (date and time)", "e.g. 2026-10-01 14:05", "text"),
    ("s1_end", "session 1 end (date and time)", "e.g. 2026-10-01 15:10", "text"),
    ("s2_start", "session 2 start (date and time, optional)", "only if you used a second session", "optional"),
    ("s2_end", "session 2 end (date and time, optional)", "only if you used a second session", "optional"),
    ("total_minutes", "total minutes spent rating", "a number", "number"),
    ("att_a", "(a) I rated alone, without discussing items with anyone", "yes or no", "yesno"),
    ("att_b", "(b) I used no AI assistant or language model (e.g., ChatGPT, Claude, Gemini, Copilot) at any point", "yes or no", "yesno"),
    ("att_c", "(c) I used only the materials in this package and did not look at the paper's results or any other rater's sheet", "yes or no", "yesno"),
    ("signature", "signature / initials", "", "text"),
    ("comments", "comments (optional)", "anything you want to tell us", "optional"),
]
ROW_HEAD = ["no.", "row_code", "benchmark", "axis", "axis_definition", "verbatim_excerpts_with_source",
            "source_K_A_T", "base_bin_0_4", "corrections", "final_bin_0_4", "deciding_phrase", "confidence_1_3", "minutes (optional)"]
ROW_WIDTHS = [5, 8, 16, 15, 30, 95, 9, 9, 14, 9, 34, 10, 9]
TEXT_COLS = ["C", "D", "E", "F"]  # displayed item text, hashed per item
INPUT_COLS = ["G", "H", "I", "J", "K", "L", "M"]

HEAD_FILL = PatternFill("solid", fgColor="DDEBF7")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
UNLOCKED = Protection(locked=False)
FONT = Font(name="Calibri", size=10)
BOLD = Font(name="Calibri", size=10, bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cell_hash(v) -> str:
    return sha_text(f"{type(v).__name__}:{v}")[:16]


def item_hash(benchmark: str, axis: str, definition: str, excerpts: str) -> str:
    return sha_text("\x1f".join([benchmark, axis, definition, excerpts]))


# ----------------------------------------------------------------------------- item rows
def source_label(e: dict) -> str:
    kind, ver = SRC[e["source"]]
    pl = e["pages_or_line"]
    where = pl["pages"] if isinstance(pl, dict) else pl
    if e["source"] in CODE_SOURCES:
        return f"{kind}, {ver}, from line {where[0]}"
    if e["source"] == "metaqa_readme":
        return f"{kind}, {ver}"
    return f"{kind}, {ver}, PDF p. {where[0]}" if len(where) == 1 else f"{kind}, {ver}, PDF pp. {where[0]}-{where[-1]}"


def render(exs: list[tuple[str, str, bool]]) -> str:
    parts = []
    for i, (label, text, code) in enumerate(exs, 1):
        parts.append(f"[{i}] ({label}):\n```\n{text}\n```" if code else f"[{i}] ({label}) \"{text}\"")
    return "\n\n".join(parts)


def load_rows():
    ex = json.load(open(LOGGED / "excerpts.json", encoding="utf-8"))
    for s, (kind, ver) in SRC.items():  # shown versions must match the recorded source versions
        rec = ex["sources"][s]
        if s in CODE_SOURCES:
            assert rec["md5"].startswith(ver.split()[-1]), s
        elif s == "metaqa_readme":
            assert "b1e69733" in rec["version"] and "2021-10-10" in rec["version"], s
        else:
            assert ver in rec["version"], s
    condb = {}
    for line in open(LOGGED / "prompts" / "partA_conditionB.jsonl", encoding="utf-8"):
        r = json.loads(line)
        condb[r["item_id"]] = r["text_with_position_0"]
    items = json.load(open(HV / "items_29.json", encoding="utf-8"))["items"]
    assert len(items) == 29
    census = []
    for it in items:
        exs = ex["items"][it["item_id"]]["excerpts"]
        assert exs and all(e["verified"] in VERIFIED_OK for e in exs), it["item_id"]
        t = condb[it["item_id"]]  # parity with the LM excerpt-only condition
        assert f"\nbenchmark: {it['benchmark']}\n" in t and f"\naxis: {it['axis']}\n" in t, it["item_id"]
        assert all(e["text"] in t for e in exs) and t.count("\n[") == len(exs), it["item_id"]
        census.append({"item_id": it["item_id"], "benchmark": it["benchmark"], "axis": it["axis"],
                       "excerpts": render([(source_label(e), e["text"], e["source"] in CODE_SOURCES) for e in exs]),
                       "n_excerpt_words": sum(len(e["text"].split()) for e in exs)})
    for a in ANCHORS:
        t = condb[a["item_id"]]
        assert a["description"] in t and f"\naxis: {a['axis']}\n" in t
        census.append({"item_id": a["item_id"], "benchmark": a["benchmark"], "axis": a["axis"],
                       "excerpts": render([("benchmark documentation", a["description"], False)]),
                       "n_excerpt_words": len(a["description"].split())})
    quiz = [{"item_id": q["qid"], "benchmark": q["benchmark"], "axis": q["axis"],
             "excerpts": render([("benchmark documentation", q["description"], False)]),
             "n_excerpt_words": len(q["description"].split())} for q in QUIZ]
    for r in census + quiz:
        r["definition"] = AXIS_DEF[r["axis"]]
        r["sha256"] = item_hash(r["benchmark"], r["axis"], r["definition"], r["excerpts"])
    # v3 keys unchanged
    k3 = json.load(open(V3 / "answer_key_v3.json", encoding="utf-8"))
    assert k3["quiz"] == {q["qid"]: q["key"] for q in QUIZ}
    assert k3["anchors"] == {a["item_id"]: a["expected_bin"] for a in ANCHORS}
    return census, quiz


# ----------------------------------------------------------------------------- workbook
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


def item_sheet(wb, name: str, rows: list[dict], codes: list[str], password: str):
    ws = wb.create_sheet(name)
    ws.append(ROW_HEAD)
    for i, (r, c) in enumerate(zip(rows, codes), 1):
        ws.append([i, c, r["benchmark"], r["axis"], r["definition"], r["excerpts"], None, None, None, None, None, None, None])
    for j, w in enumerate(ROW_WIDTHS):
        ws.column_dimensions[chr(65 + j)].width = w
    for c in ws[1]:
        c.font = BOLD
        c.fill = HEAD_FILL
        c.alignment = WRAP
    ws.row_dimensions[1].height = 28
    n = len(rows)
    for i in range(2, n + 2):
        for col in "ABCDEF":
            ws[f"{col}{i}"].font = FONT
            ws[f"{col}{i}"].alignment = WRAP
        for col in INPUT_COLS:
            c = ws[f"{col}{i}"]
            c.fill = INPUT_FILL
            c.protection = UNLOCKED
            c.font = FONT
            c.alignment = WRAP
        lines = max(est_lines(ws[f"F{i}"].value, 92), est_lines(ws[f"E{i}"].value, 30), 3)
        ws.row_dimensions[i].height = min(409, 13.5 * lines + 6)
    add_list(ws, f"G2:G{n + 1}", '"K,A,T"', "K, A, or T (Step 2)")
    add_list(ws, f"H2:H{n + 1}", '"0,1,2,3,4"', "base bin before corrections (Step 3)")
    add_list(ws, f"J2:J{n + 1}", '"0,1,2,3,4"', "final bin (Step 5)")
    add_list(ws, f"L2:L{n + 1}", '"1,2,3"', "3 = sure, 2 = fairly sure, 1 = torn")
    dv = DataValidation(type="decimal", operator="between", formula1="0", formula2="600", allow_blank=True,
                        showErrorMessage=True, errorStyle="stop")
    dv.error = "Enter minutes as a number (optional)."
    ws.add_data_validation(dv)
    dv.add(f"M2:M{n + 1}")
    ws.freeze_panes = "C2"
    protect(ws, password)


def build_workbook(k: int, quiz_rows, quiz_codes, census_rows, census_codes, password: str) -> Workbook:
    wb = Workbook()
    wb.remove(wb.active)
    # Read_me
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
    protect(ws, password)
    # Rater_info
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
    item_sheet(wb, "Quiz", quiz_rows, quiz_codes, password)
    item_sheet(wb, "Census", census_rows, census_codes, password)
    wb.active = 0
    wb.security = WorkbookProtection(lockStructure=True)
    wb.security.set_workbook_password(password)
    p = wb.properties
    p.creator = "coordinator"
    p.lastModifiedBy = None
    p.created = BUILD_TIME
    p.modified = BUILD_TIME
    p.title = "Provenance rating study v4"
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


def leak_check(data: bytes, ids: list[str], password: str):
    z = zipfile.ZipFile(io.BytesIO(data))
    blob = "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist())
    bad = [i for i in ids if i in blob] + ([password] if password in blob else [])
    for word in ("reference_bins", "census bin", "answer_key", "SEALED"):
        if word in blob:
            bad.append(word)
    assert not bad, f"leak in workbook: {bad}"


def worked_md() -> str:
    L = ["# Worked examples (invented benchmarks)", "",
         "These six examples are invented benchmarks, not part of the quiz or the census. Each gives a construction "
         "statement and the answer that the procedure of INSTRUCTIONS_v4.md (Section 5) gives for it, with the reason.", ""]
    for i, (d, s, b, c, f, why) in enumerate(WORKED, 1):
        L += [f"**E{i}.** {d}", "",
              f"- source: {s}; base bin: {b}; corrections: {c}; final bin: **{f}**",
              f"- why: {why}", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main():
    census, quiz = load_rows()
    by_id = {r["item_id"]: r for r in census + quiz}
    prev = json.load(open(SEALED, encoding="utf-8")) if SEALED.exists() else {}
    password = prev.get("password") or secrets.token_urlsafe(9)

    worked = worked_md()
    (HERE / "WORKED_EXAMPLES_v4.md").write_bytes(worked.encode("utf-8"))
    instr = (HERE / "INSTRUCTIONS_v4.md").read_bytes()
    for text in (instr.decode("utf-8"), worked):
        assert not [i for i in by_id if i in text and not i.startswith("Q")] and password not in text
    if TO_SEND.exists():
        shutil.rmtree(TO_SEND)
    TO_SEND.mkdir(parents=True)

    sealed = {
        "_comment": "SEALED. Never send this file or show it to a rater or the coordinator. It holds the sheet password, "
                    "seeds, row orders and codes, answer keys, and the hashes score_v4.py checks returned workbooks against.",
        "protocol": "PROTOCOL_v4.md (written 2026-09-30T10:27Z, before this build)",
        "build_time_utc": BUILD_TIME.isoformat() + "Z",
        "password": password,
        "password_hash_legacy": hash_password(password),
        "sheet_order": SHEETS,
        "item_columns": {"header": ROW_HEAD, "text_cols": TEXT_COLS, "input_cols": INPUT_COLS},
        "rater_info_rows": [{"row": i, "key": key, "field": field, "kind": kind} for i, (key, field, _, kind) in enumerate(INFO_ROWS, 2)],
        "relation_required": RELATION_OPTIONS[0],
        "item_text_sha256": {i: r["sha256"] for i, r in by_id.items()},
        "item_meta": {i: {"benchmark": r["benchmark"], "axis": r["axis"], "n_excerpt_words": r["n_excerpt_words"]} for i, r in by_id.items()},
        "answer_keys": {"quiz": {q["qid"]: q["key"] for q in QUIZ},
                        "attention_expected_bins": {a["item_id"]: a["expected_bin"] for a in ANCHORS}},
        "inputs_sha256": {p: sha256_bytes((HV / p).read_bytes()) for p in
                          ["items_29.json", "reference_bins.json", "llm_census_logged/excerpts.json",
                           "llm_census_logged/prompts/partA_conditionB.jsonl", "v3/answer_key_v3.json", "v4/INSTRUCTIONS_v4.md"]},
        "raters": {},
    }
    for k in RATERS:
        seed = SEED_BASE + k
        rng = random.Random(seed)
        q_rows = list(quiz)
        rng.shuffle(q_rows)
        c_rows = list(census)
        rng.shuffle(c_rows)
        c_codes = []
        while len(c_codes) < len(c_rows):
            code = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
            if code not in c_codes:
                c_codes.append(code)
        q_codes = [r["item_id"] for r in q_rows]
        wb = build_workbook(k, q_rows, q_codes, c_rows, c_codes, password)
        locked, inputs = snapshot(wb)
        buf = io.BytesIO()
        wb.save(buf)
        data = normalize_zip(buf.getvalue())
        leak_check(data, [r["item_id"] for r in census], password)
        folder = TO_SEND / f"rater_{k}"
        folder.mkdir()
        files = {f"rater_{k}_v4.xlsx": data, "INSTRUCTIONS_v4.md": instr, "WORKED_EXAMPLES_v4.md": worked.encode("utf-8")}
        for name, b in files.items():
            (folder / name).write_bytes(b)
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):
                zi = zipfile.ZipInfo(f"rater_{k}/{name}", date_time=ZIP_TIME)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.create_system = 0
                zi.external_attr = 0
                z.writestr(zi, files[name])
        zdata = zbuf.getvalue()
        (TO_SEND / f"rater_{k}_v4.zip").write_bytes(zdata)
        sealed["raters"][str(k)] = {
            "rater_code": f"R{k}", "seed": seed,
            "quiz_order": [r["item_id"] for r in q_rows], "quiz_codes": dict(zip(q_codes, [r["item_id"] for r in q_rows])),
            "census_order": [r["item_id"] for r in c_rows], "census_codes": dict(zip(c_codes, [r["item_id"] for r in c_rows])),
            "files_sha256": {**{f"to_send/rater_{k}/{n}": sha256_bytes(b) for n, b in files.items()},
                             f"to_send/rater_{k}_v4.zip": sha256_bytes(zdata)},
            "docprops_sent": docprops(data),
            "locked_cells": locked, "input_cells": inputs,
        }
    json.dump(sealed, open(SEALED, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    words_census = sum(r["n_excerpt_words"] for r in census)
    words_quiz = sum(r["n_excerpt_words"] for r in quiz)
    words_def = sum(len(AXIS_DEF[r["axis"]].split()) for r in census + quiz)
    print(f"built v4 package for raters {list(RATERS)}")
    print(f"excerpt words shown: census 29 items {sum(r['n_excerpt_words'] for r in census[:29])}, attention 2 items "
          f"{sum(r['n_excerpt_words'] for r in census[29:])}, quiz {words_quiz}; axis definitions {words_def}; "
          f"total excerpt words {words_census + words_quiz}")
    for k, v in sealed["raters"].items():
        for f, h in v["files_sha256"].items():
            print(f"  {h[:16]}  {f}")


if __name__ == "__main__":
    main()
