# -*- coding: utf-8 -*-
"""Anonymized export of the returned v4 workbooks for release (run after score_v4.py, from human_validation/v4/).

The returned workbooks are not released: their document properties and save paths name the raters'
institution, and Rater_info holds the raters' initials. This script writes what the analysis uses, and the
SHA-256 of each returned file so that the files can be matched if they are released later:

  public/ratings_v4.csv       rater, item_id, source, base bin, corrections, final bin, deciding phrase,
                              confidence, minutes (29 census items and the two attention items)
  public/quiz_v4.csv          rater, quiz item, final bin, source, key final bin, key source
  public/raters_v4.csv        rater, background, years in NLP/IR, relation, session minutes from the recorded
                              times, total time as entered, the three attestations, eligibility
  public/returned_sha256.txt  SHA-256 of each returned workbook
  public/README.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "public"


def main():
    res = json.load(open(HERE / "result_v4.json", encoding="utf-8"))
    sealed = json.load(open(HERE / "SEALED_build.json", encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    W = res["workbooks"]
    with open(OUT / "ratings_v4.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "item_id", "source_K_A_T", "base_bin_0_4", "corrections", "final_bin_0_4", "deciding_phrase", "confidence_1_3", "minutes"])
        for c in sorted(W):
            cen = dict(W[c]["census"])
            for a, x in W[c]["attention"]["items"].items():
                cen[a] = {"final": x["given"]}
            for i in sorted(cen):
                x = cen[i] or {}
                w.writerow([c, i, x.get("source"), x.get("base"), x.get("corrections"), x.get("final"), x.get("phrase"), x.get("confidence"), x.get("minutes")])
    with open(OUT / "quiz_v4.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "quiz_item", "final_bin", "source", "key_final_bin", "key_source"])
        for c in sorted(W):
            for q, x in sorted(W[c]["quiz"]["items"].items()):
                w.writerow([c, q, x["final"], x["source"], x["key"], x["key_source"]])
    with open(OUT / "raters_v4.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "background", "years_nlp_ir", "relation", "minutes_from_session_times", "total_time_as_entered",
                    "att_a_alone_no_discussion", "att_b_no_AI", "att_c_package_only", "quiz_correct_of_6", "attention_pass",
                    "hard_integrity_flags", "eligible"])
        for c in sorted(W):
            ri, v = W[c]["rater_info"], W[c]["rater_info"]["values"]
            w.writerow([c, v.get("background"), v.get("years_nlp_ir"), v.get("relation"), ri["session_minutes_from_times"],
                        v.get("total_minutes"), ri["attestations"]["att_a"], ri["attestations"]["att_b"], ri["attestations"]["att_c"],
                        W[c]["quiz"]["correct_final"], W[c]["attention"]["pass"], len(W[c]["integrity"]["hard"]), W[c]["eligible"]])
    (OUT / "returned_sha256.txt").write_text(
        "".join(f"{W[c]['integrity']['sha256']}  returned/{W[c]['file']}\n" for c in sorted(W)), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# v4 human ratings, anonymized export\n\n"
        "Written by `../export_public_v4.py` from `../result_v4.json` (the output of `../score_v4.py`). The returned workbooks are "
        "not released, because their document properties and save paths name the raters' institution and `Rater_info` holds the "
        "raters' initials; `returned_sha256.txt` lists their hashes. The integrity results, eligibility, and every statistic are in "
        "`../result_v4.md`; the follow-up on its warnings is in `../independence_checks_v4.md`. The sent workbooks are in "
        "`../to_send/` and match the hashes in `../SEALED_build.json`.\n\n"
        "The `total_time_as_entered` column is the rater's own entry in the total-minutes field; all three raters entered hours "
        "there (1.5, 2, 2), so the paper reports `minutes_from_session_times`.\n", encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
