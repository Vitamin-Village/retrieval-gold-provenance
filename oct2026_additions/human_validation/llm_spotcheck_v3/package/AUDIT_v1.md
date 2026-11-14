# Audit of the v1 judge spot-check (2026-09-24)

Scope: `../../judge_spotcheck_v1/` (sheet, blank sheet, sealed verdicts, scorer, result).

| check | result |
|---|---|
| Sample size and stratification | 100 items; 50 judged CORRECT and 50 INCORRECT by mistral-small:24b, as documented |
| Duplicates | none (no repeated key, no repeated question/reference/candidate triple) |
| Sealed verdicts vs the judge's cache | 100 of 100 identical (`results/e2e_qa/judge_cache_mistral_small_24b.jsonl`) |
| Sheet text vs reader outputs | 100 of 100 question/reference/candidate texts identical to the reader predictions in `per_question.jsonl` |
| Returned sheet vs blank sheet | identical except row 76 (position 76), whose candidate is the model's literal output "None"; the returned sheet shows it as empty. Human and judge both say INCORRECT, so no statistic changes |
| Scorer | agreement, Cohen's kappa, precision and recall computed correctly; UNSURE handled; recomputation gives n = 100, agreement 0.86, kappa 0.72, precision 0.98, recall 0.79 |
| Reader balance | the sample holds 80 Qwen2.5-14B and 20 Llama-3.1-8B predictions, whereas the final pool is 47% Qwen (2,163 of 4,652 unique predictions) |
| Conditions covered | aligned KG, independent KG, BM25, closed-book, gold-only, and BGE-M3 conditions, i.e. every condition the judge scores |
| Rater | one non-author rater; no human-human agreement is available |
| Precision/recall | computed on a 50/50 sample, so they describe the sample; the judge marks 51.1% of the final pool CORRECT, which `score_spotcheck_v3.py` uses to weight the recall estimate (`pool_stats.json`) |
