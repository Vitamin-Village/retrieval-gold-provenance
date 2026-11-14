# -*- coding: utf-8 -*-
"""Judge spot-check v3 (2026-09-24).

v1 (../judge_spotcheck/) had one non-author rater label 100 GraphRAG-Bench (question, reference,
candidate) triples, 50 that the primary judge (mistral-small:24b) marked CORRECT and 50 INCORRECT.
AUDIT_v1.md verifies the v1 sample, the sealed verdicts, and the texts item by item. One rater gives no
human-human agreement, the v1 decision rules left cases open (hedged lists, 'None' outputs, extra
contradicting detail), and precision/recall on a 50/50 sample describe the sample, not the judge's full output.

v3 keeps the same 100 items, writes the v1 sampling code as a function and checks whether it
reproduces the sealed sample (pool_stats.json), and builds three workbooks (independent random orders)
with explicit decision rules.

Outputs (human_validation/judge_spotcheck_v3/):
  GUIDE_v3.md, AUDIT_v1.md (written by hand)
  sheets/rater_{1,2,3}_spotcheck.xlsx
  sealed_key_v3.json     item_id -> judge verdict, v1 hash key, reader, condition, v1 human label
  pool_stats.json        judge CORRECT rate over the sampling pool (for population-weighted recall)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
V1 = HERE.parent / "judge_spotcheck"
JUDGE = "mistral-small:24b"
(HERE / "sheets").mkdir(parents=True, exist_ok=True)


def v1_pool(per_question_suffix="per_question_20260924.jsonl"):
    """The v1 sampling pool, exactly as in the 2026-09-20 build code, from the reader outputs as of 2026-09-24
    (a copy of each per_question.jsonl of that day, not included; pool_stats.json records the pool): unique (question, gold, prediction)
    triples of both readers on GraphRAG-Bench with a non-empty prediction and a cached judge verdict."""
    qtext = {}
    for sp in ("sub200", "holdout200"):
        for l in open(PAPER / f"external/graphragbench/{sp}/qa.jsonl", encoding="utf-8"):
            q = json.loads(l)
            qtext[q["qa_id"]] = q["question"]
    cache = {}
    for l in open(PAPER / "results/e2e_qa/judge_cache_mistral_small_24b.jsonl", encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        cache[r["key"]] = r["verdict"]
    rows, seen = [], set()
    for reader in ("llama3_1_8b", "qwen2_5_14b"):
        for l in open(PAPER / f"results/e2e_qa/{reader}/{per_question_suffix}", encoding="utf-8"):
            r = json.loads(l)
            if r["bench"] != "graphragbench" or not r["pred"].strip():
                continue
            gold = r["golds"][0]
            key = hashlib.sha1("\x00".join([JUDGE, r["bench"], r["qa_id"], gold, r["pred"]]).encode()).hexdigest()
            if key in seen or key not in cache:
                continue
            seen.add(key)
            rows.append({"key": key, "reader": reader, "condition": r["condition"], "system": r["system"], "split": r["split"],
                         "qa_id": r["qa_id"], "question": qtext.get(r["qa_id"], ""), "gold_answer": gold,
                         "candidate_answer": r["pred"], "judge": cache[key]})
    return rows


def v1_sample(rows):
    random.seed(20260920)
    rows = list(rows)
    random.shuffle(rows)
    cor = [x for x in rows if x["judge"] == "CORRECT"][:50]
    inc = [x for x in rows if x["judge"] == "INCORRECT"][:50]
    sample = cor + inc
    random.shuffle(sample)
    return sample


def main():
    sealed_v1 = json.load(open(V1 / "sealed_judge_verdicts.json"))
    v1_rows = list(csv.DictReader(open(V1 / "judge_spotcheck_sheet_blank.csv", encoding="utf-8-sig")))
    v1_labels = {r["key"]: r["your_verdict_CORRECT_or_INCORRECT"].strip().upper()
                 for r in csv.DictReader(open(V1 / "judge_spotcheck_sheet.csv", encoding="utf-8-sig"))}
    pool = v1_pool()
    repro = v1_sample(pool)
    same_set = {x["key"] for x in repro} == set(sealed_v1)
    same_order = [x["key"] for x in repro] == [r["key"] for r in v1_rows]
    meta = {x["key"]: x for x in pool}
    pool_stats = {"pool_unique_predictions": len(pool), "judge_correct_rate": sum(x["judge"] == "CORRECT" for x in pool) / len(pool),
                  "reproduces_v1_sample_set": same_set, "reproduces_v1_order": same_order,
                  "v1_sample_by_reader": dict(Counter(meta[k]["reader"] for k in sealed_v1 if k in meta)),
                  "pool_by_reader": dict(Counter(x["reader"] for x in pool))}
    json.dump(pool_stats, open(HERE / "pool_stats.json", "w"), indent=1)
    print(json.dumps(pool_stats, indent=1))

    items = []
    for i, r in enumerate(v1_rows, 1):
        m = meta.get(r["key"], {})
        items.append({"item_id": f"S{i:03d}", "v1_key": r["key"], "question": r["question"], "reference": r["gold_answer"],
                      "candidate": r["candidate_answer"], "judge": sealed_v1[r["key"]], "v1_human": v1_labels.get(r["key"]),
                      "reader": m.get("reader"), "condition": m.get("condition"), "system": m.get("system"), "split": m.get("split")})
    key = {"_comment": "Answer key; used only by the scorer.",
           "items": {x["item_id"]: {k: x[k] for k in ("v1_key", "judge", "v1_human", "reader", "condition", "system", "split")} for x in items}}
    json.dump(key, open(HERE / "sealed_key_v3.json", "w", encoding="utf-8"), indent=1)

    for r in (1, 2, 3):
        rng = random.Random(20260924 + r)
        order = list(items)
        rng.shuffle(order)
        wb = Workbook()
        ws = wb.active
        ws.title = "Rater_info"
        ws.append(["field", "your answer"])
        for f in ("rater code", "field and degree", "are you an author of the paper? (must be no)",
                  "have you seen the judge's verdicts or the answer key? (must be no)",
                  "will you rate without any AI assistant such as ChatGPT or Claude? (must be yes)",
                  "consent to use your labels anonymously in a publication (yes/no)",
                  "start time (HH:MM)", "end time (HH:MM)", "breaks longer than 10 min (minutes)", "comments"):
            ws.append([f, ""])
        ws.column_dimensions["A"].width = 60
        ws.column_dimensions["B"].width = 40
        ws2 = wb.create_sheet("SpotCheck")
        ws2.append(["position", "item_id", "question", "reference_answer", "candidate_answer (model output, verbatim)",
                    "your_verdict", "note (optional)"])
        for i, x in enumerate(order, 1):
            ws2.append([i, x["item_id"], x["question"], x["reference"], x["candidate"], "", ""])
        for col, w in zip("ABCDEFG", (8, 9, 45, 55, 55, 14, 30)):
            ws2.column_dimensions[col].width = w
        for c in ws2[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="DDEBF7")
        for row in ws2.iter_rows(min_row=1):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        for row in ws2.iter_rows(min_row=2, min_col=5, max_col=5):  # keep literal model output such as None as text
            for c in row:
                c.number_format = "@"
                c.value = "" if c.value is None else str(c.value)
        ws2.freeze_panes = "A2"
        dv = DataValidation(type="list", formula1='"CORRECT,INCORRECT,UNSURE"', allow_blank=True)
        ws2.add_data_validation(dv)
        dv.add(f"F2:F{len(order) + 1}")
        wb.save(HERE / "sheets" / f"rater_{r}_spotcheck.xlsx")
    print("built", sorted(p.name for p in (HERE / "sheets").iterdir()))


if __name__ == "__main__":
    main()
