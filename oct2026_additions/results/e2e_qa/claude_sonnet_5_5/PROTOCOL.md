# Proprietary reader (Claude Sonnet 5.5) on the core 2WikiMultiHop conditions: protocol

Written 2026-10-02, before any request of this run was sent. The text above "Deviations" is not edited after the first request; later notes go under "Deviations" with a timestamp.

## Purpose

The paper measures the answer-level swap with open-weight readers of at most 24B parameters (Llama-3.1-8B, Qwen2.5-14B, and Mistral-Small-24B on 2WikiMultiHop). This run adds one proprietary frontier reader under the same conditions, prompt, scoring, and statistics as the third reader (`eval/_e2e_third_reader.py`), so that the reader-size limitation can be checked with a much stronger reader.

## Design (fixed)

- Reader: `claude-sonnet-5-5` through the Anthropic Message Batches API, one request per unique prompt, no system prompt, the user message equal to the prompt of `eval/e2e_qa_swap.py` (`PROMPT` with the top-five passages, `PROMPT_CLOSED` for closed-book) byte for byte.
- Parameters: `max_tokens` 256; `thinking: {"type": "between_tools"}` (no reasoning before the answer, the closest available setting to the other readers' plain decoding); effort left at the API default (`high`); no `temperature` (the model accepts only the default); no tools; no fallbacks (every answer must come from this model).
- Conditions: exactly the third reader's: 2WikiMultiHop, both 200-question splits; the aligned and independent KG vectors of the hybrid, hybrid-PPR, PPR, and HippoRAG 2 retrievers; BM25; closed-book; gold-only. Unique prompts are keyed as in `eval/e2e_qa_swap.py` (`sha1(model + "\x00" + prompt)`).
- Scoring and statistics: `eval/e2e_qa_swap.py --score-only` unchanged (first line of the reply, `clean_prediction`, alias-aware EM, token F1, cover-EM; paired per-question swap aligned minus independent with 10,000-resample bootstrap CI and exact two-sided sign test).
- Reporting: every cell is reported whatever its value, beside the three open-weight readers. A refused, truncated, or errored request is retried once if the error is a server error; otherwise it is scored as an empty answer and counted in the report.
- Logging: every request's parameters, the batch id, every result object (model id, stop reason, usage, text), and the cost estimate are stored in this folder.

## Deviations

(none)

## Run record (appended 2026-10-02T14:50Z, after scoring)

- Parameter pilot, 14:14Z: three single requests with the fixed parameters (`pilot_raw.jsonl`); their replies are not in the cache and not scored.
- Batch `msgbatch_011MMfGKoHhxr8TuyAAvV3TJ`: 4,252 requests (every unique prompt) submitted 14:32:38Z, results collected 14:40:01Z; 4,252 succeeded, none errored, expired, or refused, so no retry was needed. Every reply reports model `claude-sonnet-5-5`. Usage 5,327,503 input and 195,729 output tokens, about USD 6.31 at the batch price (`batch_state.json`).
- Truncation rule as fixed above: 135 replies stopped at `max_tokens` (92 closed-book, 41 with five retrieved passages, 2 gold-only) and are scored as empty answers; the cache keeps their text in `response_raw`. Scoring them as written instead leaves every EM value unchanged and moves no cover-EM swap by more than 0.005 (reported in the paper's appendix).
- Scored with `eval/e2e_qa_swap.py --score-only` through `--score`; 4,400 scored rows (`summary.md/json`, `per_question.jsonl`). Table 27 of the paper is written by `analysis/larger_readers_table.py`.
