Released copy: passages outside the scope of this paper are omitted and marked [omitted]; SHA-256 of the full protocol as written: 909ba35e41ff6478ebd2d5c1783aa5fbd1527f2cb6f47132240384c9859e6891.

# Logged LM re-application of the census procedure: protocol

Written 2026-09-29T11:34Z [omitted], before any API request of this study was sent. Sections 1–9 are fixed from this point. Pre-run checks and any deviation are appended below the line at the end, each with its own timestamp.

## 1. Purpose

[omitted] [omitted] the condensed item descriptions were written by the authors and echo rubric wording ("No knowledge graph is used", "copied directly from the evaluation knowledge graph by code"), so agreement with the census may partly be agreement with the authors' own reading. Condition B removes the condensed descriptions and gives only verbatim text from each benchmark's own paper or documentation.

Everything below is run and scored by `eval/llm_census_logged.py`. Outputs are in this folder.

## 2. Instrument

The v3 instrument, unchanged in content:
- rater instructions `../v3/INSTRUCTIONS_v3.md` (full step procedure, six worked examples);
- quiz key `../v3/QUIZ_KEY_v3.md`;
- rows exactly as they appear in the blank workbooks `../v3/sheets/rater_1_stage1_quiz.xlsx` (sheet Quiz, 6 rows) and `../v3/sheets/rater_1_stage2_main.xlsx` (sheet PartA, 29 items and 2 attention items; sheet PartB, 130 items). The three blank workbooks hold the same rows in different orders.

System prompts (exact texts and SHA-256 in `prompts/`):
- The instructions are used verbatim except for two sentences that address the human `Rater_info` sheet, which held the rater's background and the no-AI attestation: "Please fill the `Rater_info` sheet in each workbook." (Section 1) and "Record start and end times in `Rater_info`. Breaks are fine; note breaks longer than 10 minutes." (Section 3). [omitted] Nothing else is removed. The worked examples are part of Section 4 of the instructions and are therefore included.
- Stage 1 (quiz): instructions + a response-format appendix.
- Stage 2 (Part A and Part B): instructions + the quiz key + the same response-format appendix. Human raters were told to read the quiz key before stage 2 (Section 2 of the instructions), so a stage-2 LM rater receives it as well.
- The response-format appendix says that each message holds one workbook row, that each message is independent, and that the reply must end with one machine-readable line: `RATING: {json}` with the six workbook fields (`source_K_A_T`, `base_bin_0_4`, `corrections`, `final_bin_0_4`, `key_phrase`, `confidence_1_3`) for quiz and Part A rows, or `PARTB: {json}` with `your_Q1`, `your_Q2`, `flag_data_problem` for Part B rows. The format line uses placeholders, not example values. Free reasoning before the final line is allowed.
- The system prompt is identical in conditions A and B; the conditions differ only in the Part A row text.

## 3. Conditions

- **Condition A, workbook.** Each Part A row shows exactly the workbook columns: position, item_id, benchmark, axis, construction_description (the author-condensed description), and verbatim_excerpt_from_benchmark_paper (the workbook excerpt, which is empty for four rows).
- **Condition B, excerpt-only.** Each Part A row shows position, benchmark name, axis, and one or more verbatim excerpts from the benchmark's own paper, README, or (for the two calibration anchors) generator source code, each labelled with its source kind and PDF page. No condensed description and no item_id is shown. The axis label is kept because it defines which gold unit is judged. The row text states that the sheet gives no condensed description.
- Quiz rows (invented benchmarks) are identical in both conditions. The two attention items (invented benchmarks, excluded from all statistics) have no source paper; in condition B their workbook description is shown as their only excerpt, labelled "benchmark documentation".
- Part B (130 per-unit rows, exactly the workbook columns) is run in condition A only.

### Condition-B excerpts (`excerpts.json`)

The rule was fixed before any condition-B request: for each item, (census) every census-record, workbook, or manuscript Table-quotes quote for the item, extended to the full sentence(s) in which it occurs; (parity) for each factual statement of the condensed description, the source sentence(s) that state it, where one exists; (production) if neither covers how the gold unit of that axis was produced, the construction sentence(s) that do. Description statements with no source sentence are listed per item (for example "No knowledge graph is used" for MuSiQue and MultiHop-RAG). Every excerpt was checked against the downloaded source text: 69 of 111 are exact substrings after whitespace normalisation; 38 match after keeping letters and digits only (PDF line-break hyphenation, small caps, sub/superscripts); 4 match in order around a footnote, table, or figure page, and the skipped text is stored in `excerpts.json` for inspection. Anchor excerpts are exact substrings of the repository files, whose MD5 is recorded. Source versions: current arXiv/Anthology versions, except GRADE (arXiv v1) and STaRK (arXiv v1), where the census-record quotes are verbatim only in v1.

Items lacking a verbatim workbook excerpt: **HOTPOT_ANSWER** and **MUSIQUE_ANSWER** (empty cells). Their condition-B excerpts were taken from the HotpotQA (arXiv 1809.09600v1, pp. 2 and 11) and MuSiQue (arXiv 2108.00573v3, pp. 4–6) papers. Workbook excerpts that are not verbatim in the checked source versions (condition A shows them unchanged, since condition A reproduces the workbook): NQ_PASSAGE ("all information" for "all of the information"); GRB_SUBGRAPH ("transform raw text" for "transforms raw text"; "the question is discarded" for "the corresponding question is discarded"); GRADE_PASSAGE ("supporting evidence traces back to original source documents", found in neither GRADE version); METAQA_ANSWER (triplet notation rendered differently); OURSUP_EDGE ("{src}--{rel}-->{tgt}" for "{src_id}--{rel}-->{tgt_id}"); OURSUP_ANSWER (the text after "code:" is a paraphrase, not code); STARK_TEXTUAL (carries the annotation "(from the gold node's free text)"); STARK_ANSWERENTITY (verbatim in arXiv v1 only).

## 4. Models and request parameters

Requested model ids, no fallback of any kind (no server-side fallback, no alternative model on error):
- `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-4-5` through the Anthropic Messages API (Python SDK `anthropic`, key from `../.env`), non-streaming, `max_tokens=16000`, `system` as one text block with `cache_control: {"type": "ephemeral"}`, one user message.
- `temperature=1.0` (the API default) is sent; if a model rejects `temperature`, it is omitted for that model and the model's defaults are used. The outcome per model is recorded in Section 10.
- `thinking` and `output_config.effort` are not sent, so each model runs at its API default (documented at run time: Opus 5.5 adaptive thinking at effort medium; Sonnet 5.5 adaptive thinking at effort high; Haiku 4.5 no extended thinking). Haiku 4.5's minimum cacheable prefix (4,096 tokens) is longer than the system prompt, so its requests are expected not to be cached.
- One Gemini model if `GEMINI_API_KEY` in `../.env` works: the models endpoint is listed and the strongest generally available (non-preview, fixed-id, non-alias) model is used, with its exact id and returned `modelVersion` recorded. Request: REST `v1beta/models/{id}:generateContent`, `systemInstruction` = the same system prompt, one user turn, `generationConfig = {temperature: 1.0, maxOutputTokens: 16384}`, thinking at its default. If the key or model fails, this is documented here and the study continues with the three Claude models.
- Every record in `raw.jsonl` stores the full request parameters (system prompt by SHA-256), the returned model id, stop reason, usage (input, output, cache read, cache write), request id, full reply text, parse result, and UTC timestamp.

## 5. Runs

- Three independent runs per model per condition. A run = stage 1 (6 quiz rows) + stage 2 Part A (31 rows) + stage 2 Part B (130 rows, condition A only).
- One request per row, each in a fresh context (single user turn, no conversation history). No request sees another request's output.
- Each run has its own seed: `seed = 20260929000 + 100*model_index + 10*condition_index + run` (model_index: opus 1, sonnet 2, haiku 3, gemini 4; condition A 1, B 2; run 1–3). One `random.Random(seed)` shuffles the quiz, then Part A, then Part B; requests are dispatched in that order and the position shown in a row is its position in the shuffled sheet.
- Concurrency at most 6 requests. Resumable: a (model, condition, run, stage, item) key with a stored reply is never re-requested. An API error after the SDK's retries (5) is logged as an error line and the key is retried on the next invocation; replies that are refusals, truncated, or unparsable are kept as they are and never re-requested.
- Before the full run, parsing is verified on 3 rows (one quiz, one Part A, one Part B) per model; those pilot requests are stored separately and not used in any statistic.
- Budget: prompt caching on; about 2,450 requests in total. The full run stops and this protocol is amended if the running cost estimate exceeds USD 150.

## 6. Parsing

The last `RATING:` or `PARTB:` line of the reply is parsed as JSON. A Part A or quiz reply is valid if `final_bin_0_4` and `base_bin_0_4` are integers 0–4, `source_K_A_T` is K, A, or T, and `confidence_1_3` is 1–3; `corrections` is recorded as written and flagged if it is not in the workbook's drop-down list. A Part B reply is valid if `your_Q1` is 0–2 and `your_Q2` is 0–1. Invalid, refused (`stop_reason` refusal), or truncated replies count as missing for that item; they are reported per run and never imputed.

## 7. Scoring (`python eval/llm_census_logged.py --score`, recomputed from `raw.jsonl` alone)

References: census bins (`../reference_bins.json`, equal to the v3 answer key); [omitted]; the non-author re-scoring (`../external_rater_result.json`). Anchors = the six items of the two calibration anchors in `../items_29.json` (benchmark "Anchor benchmark L" or "Anchor benchmark U"); the other 23 are external.

Per run, and per model-median (item-wise median of the three runs; with a missing run, the lower median of those present), in each condition:
1. quiz score (final bin) and attention-item check, with the human exclusion rules [omitted] reported as flags (a run failing them is reported and excluded from model medians; results with it included are reported as well);
2. quadratically weighted κ, exact agreement, within-one agreement against the census, on all 29 items and on the 23 external items;
3. [omitted];
4. agreement on the binary reading (final bin ≥ 2, derived, versus ≤ 1, independent) against the census [omitted];
5. per-benchmark headline Ĉ_KG (maximum bin over the KG-derived axes, as in `analysis/rater_robustness.py`) and whether GRADE, 2WikiMultiHop, MetaQA, WebQSP, STaRK, and GraphRAG-Bench are at bin ≥ 3 (Ĉ_KG ≥ 0.75).

Across runs: ordinal Krippendorff α among the three runs of each model, across the model medians, and across all runs of a condition (missing values allowed).

[omitted]

Condition A minus condition B: for each model median and for the mean over models, the differences in κ, exact, within-one, and binary agreement with the census (29 and 23 items), each with a 95% paired bootstrap CI over items (10,000 resamples, seed 20260929).

[omitted]

Part B (condition A): per model and part, three-run ordinal α on Q1, the majority distributions, and majority agreement with the lexical rules, [omitted].

[omitted]

## 8. Reporting

All numbers above are reported whatever their values, together with quiz and attention results, missing replies, refusals, returned model ids, token usage, and cost. The logged runs are LM raters and are never reported as human raters.

## 9. Files

`raw.jsonl`; `runs/<model>_<condition>_<run>.json` (parsed ratings); `result.json`, `result.md`; `prompts/` (exact system prompts, removed sentences, every user message per condition, SHA-256 manifest); `excerpts.json`; [omitted]; `pilot_raw.jsonl` (the parse check of Section 5, not used in statistics).

---

## 10. Pre-run checks and deviations (appended)

### 10.1 Pre-run checks, 2026-09-29T11:36Z–11:40Z (before the full run)

- **Gemini model choice.** `GEMINI_API_KEY` works. The models endpoint (listed 2026-09-29T11:2xZ) offers Pro-tier 3.x models only as previews (`gemini-3.1-pro-preview`), aliases without a fixed id (`gemini-pro-latest`, `gemini-flash-latest`), the older `gemini-2.5-pro` (stable release of June 2025), and fixed-id non-preview 3.x Flash models up to `gemini-3.8-flash`. We use `gemini-3.8-flash`, the newest generally available fixed-id model; the endpoint reports its `version` as "3.0". This is a judgement: no public benchmark comparison between `gemini-3.8-flash` and `gemini-2.5-pro` was checked. Every reply records the returned `modelVersion`.
- **Temperature.** All three Claude models accepted `temperature=1.0` (the API default value) in the pilot, so it is sent for every Claude request (`model_params.json`). Gemini receives `temperature=1.0` in `generationConfig`.
- **Parse check (3 rows per model, `pilot_raw.jsonl`, not used in statistics).** All 12 pilot replies ended normally (`end_turn` / `STOP`) and parsed. Returned ids: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-4-5-20251001` (the dated snapshot behind the `claude-haiku-4-5` alias), `gemini-3.8-flash`.
- **Caching.** Opus 5.5 and Sonnet 5.5 wrote the system prompt to the cache (3,168 tokens stage 1; 3,654 tokens stage 2) and read it on the next stage-2 request. Haiku 4.5 did not cache, as expected (prompt below its 4,096-token minimum). Gemini reported no cached tokens on the three pilot requests; implicit caching, if any, is recorded per request.
- No change to Sections 1–9.

### 10.2 Deviation: Anthropic credit exhausted (recorded 2026-09-29T12:20Z)

- The full run started at 11:40Z. At 12:01:09Z every Anthropic request began to fail with HTTP 400 "Your credit balance is too low to access the Anthropic API", after an estimated USD 11.2 of Claude spend (far below the USD 150 guard of Section 5). Opus 5.5 and Sonnet 5.5 had finished all 12 of their runs. Haiku 4.5 had finished workbook run 1 (167/167 rows) and 18 of 167 rows of workbook run 2; its workbook run 3 and all three excerpt-only runs received no reply. The 427 failed requests are logged in `raw.jsonl` as error lines. A one-token probe at 12:12Z failed the same way, so they were not re-sent. Gemini, which does not use the Anthropic account, finished all 6 runs.
- Handling (added to the scorer after the failure): a run with fewer stored replies than rows because of API errors is reported per run, but is not used in model medians, within-model α, the all-runs α, Part B, or the robustness table. Haiku 4.5 therefore enters as one complete workbook run (its "median" is that run) and does not enter the condition A minus B comparison. Replies that were received but did not parse remain missing values as in Section 6 (one case: Gemini, excerpt-only run 2, MUSIQUE_ANSWER, malformed JSON).
- To complete Haiku once credit is added: `python eval/llm_census_logged.py --run --models claude-haiku-4-5`, then `--score`. The run is resumable and re-sends only the 427 keys without a reply, with the same seeds and prompts.
- Interim scoring was run on partial data at about 11:43Z and 12:00Z to test the scorer. No design element was changed after it.
- Gemini cost is not estimated by the script (no price table); its token counts are reported in `result.md`.

### 10.3 Completion of the Haiku 4.5 runs (recorded 2026-10-02)

- Credit was added to the Anthropic account on 2026-10-01. A one-token probe succeeded at 13:08Z, and `python eval/llm_census_logged.py --run --models claude-haiku-4-5` then sent the 427 keys without a stored reply (13:08Z–13:15Z), with the seeds, prompts, and parameters of Sections 2–5; all 427 replies ended normally and parsed. No other model was re-run and no stored reply was changed. The error lines of 10.2 remain in `raw.jsonl`.
- `--score` was re-run at 13:16Z. Haiku 4.5 now enters with three complete runs in each condition, like the other models, so the handling rule of 10.2 no longer excludes any run. Everything else in Sections 6–9 is unchanged.
