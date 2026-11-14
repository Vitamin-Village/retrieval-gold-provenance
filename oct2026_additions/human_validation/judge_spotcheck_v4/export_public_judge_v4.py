# -*- coding: utf-8 -*-
"""Anonymized export of the returned judge spot-check v4 workbooks (run after score_judge_v4.py, from this folder).

The returned workbooks and result_judge_v4.json are not released: the workbooks' document properties and save
paths name the raters' institution, and Rater_info holds the raters' initials. This script writes what the
analysis uses, and the SHA-256 of each returned file:

  public/labels_judge_v4.csv        rater, item_id, verdict, note (the 100 items and the two attention items)
  public/raters_judge_v4.csv        rater, background, years in NLP/IR, relation, minutes, the three attestations,
                                    attention result, eligibility
  public/returned_sha256.txt        SHA-256 of each returned workbook
  public/README.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "public"


def main():
    res = json.load(open(HERE / "result_judge_v4.json", encoding="utf-8"))
    W = res["workbooks"]
    OUT.mkdir(exist_ok=True)
    with open(OUT / "labels_judge_v4.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "item_id", "verdict", "note"])
        for c in sorted(W):
            labels = dict(W[c].get("labels", {}))
            for a, x in W[c]["attention"]["items"].items():
                labels[a] = x["given"]
            notes = W[c].get("notes", {})
            for i in sorted(labels):
                w.writerow([c, i, labels[i], notes.get(i, "")])
    with open(OUT / "raters_judge_v4.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "background", "years_nlp_ir", "relation", "minutes_from_session_times", "total_minutes_as_entered",
                    "att_a_alone_no_discussion", "att_b_no_AI", "att_c_package_only", "attention_pass", "hard_integrity_flags", "eligible"])
        for c in sorted(W):
            ri = W[c]["rater_info"]
            v = ri.get("values", {})
            att = ri.get("attestations", {})
            w.writerow([c, v.get("background"), v.get("years_nlp_ir"), v.get("relation"), ri.get("session_minutes_from_times"),
                        v.get("total_minutes"), att.get("att_a"), att.get("att_b"), att.get("att_c"),
                        W[c]["attention"]["pass"], len(W[c]["integrity"]["hard"]), W[c]["eligible"]])
    (OUT / "returned_sha256.txt").write_text(
        "".join(f"{W[c]['integrity']['sha256']}  returned/{W[c]['file']}\n" for c in sorted(W)), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Judge spot-check v4 human labels, anonymized export\n\n"
        "Written by `../export_public_judge_v4.py` from `../result_judge_v4.json` (the output of `../score_judge_v4.py`). "
        "The returned workbooks are not released, because their document properties and save paths name the raters' "
        "institution and `Rater_info` holds the raters' initials; `returned_sha256.txt` lists their hashes. Integrity "
        "results, eligibility, and every statistic are in `../result_judge_v4.md`; deviations are recorded at the end of "
        "`../PROTOCOL_judge_v4.md`. The sent workbooks are in `../to_send/` and match the hashes in "
        "`../SEALED_judge_build.json`.\n", encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
