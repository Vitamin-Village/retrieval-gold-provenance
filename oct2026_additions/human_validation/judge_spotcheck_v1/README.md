# LM-judge spot-check (GraphRAG-Bench answer grading)

`judge_spotcheck_sheet.csv`: 100 (question, reference answer, candidate answer) triples sampled from the
end-to-end QA outputs, 50 that the primary judge (mistral-small:24b) marked CORRECT and 50 INCORRECT,
shuffled. The rater fills `your_verdict_CORRECT_or_INCORRECT` (CORRECT / INCORRECT / UNSURE) without seeing the judge's verdict
(`sealed_judge_verdicts.json` stays closed). Rule: CORRECT if the candidate states the same key fact,
entity, value, or conclusion as the reference, even if shorter; INCORRECT if empty, contradicting,
a different entity/value, or off-question. No medical knowledge is needed: compare the candidate with the reference answer that is given on the same row; abbreviations the reference itself introduces (e.g., "basal cell carcinoma (BCC)") count as the same entity. Use UNSURE only when equivalence cannot be decided from the two texts; UNSURE rows are excluded from agreement and their count is reported.

Score, from inside this folder (the scorer reads and writes its files by relative name): `python score_spotcheck.py` -> human-vs-judge agreement and Cohen's kappa, written to `spotcheck_result.json`.
The sample, the sealed verdicts and the texts are verified item by item in `../llm_spotcheck_v3/package/AUDIT_v1.md`.

A follow-up labels the same 100 items with three Claude models; these are LM labels, not human labels (`human_validation/llm_spotcheck_v3/`).
