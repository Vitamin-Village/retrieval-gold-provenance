# Census Rater Model Identities and Run Dates

## Cross-model census (five raters, four families)

The five models that independently re-applied the rubric to the 13 headline
assignments, with the exact identifiers under which each was called (the same
identifiers appear as rater keys in `cross_model_census_raw.json` and in
`cross_model_census.py`):

| model (as named in the paper) | identifier as called | access |
|---|---|---|
| Claude Opus 4.8 | `claude-opus-4-8` | Anthropic API |
| Claude Sonnet 4.5 | `claude-sonnet-4-5` | Anthropic API |
| Qwen2.5-14B | `qwen2.5:14b` | local, Ollama |
| Mistral-Small-24B | `mistral-small:24b` | local, Ollama |
| Llama-3.1-8B | `llama3.1:8b` | local, Ollama |

Run date: 2026-06-08. This is the file modification date of the run log and of
the released raw five-model matrix (`cross_model_census_raw.json`), which is
the output of that run; per-call API timestamps were not separately logged.

## Primary two-rater census

The two primary blind raters produced the 29 axis-level judgments in two
recorded sittings: 23 judgments dated 2026-06-01 and six judgments (Natural
Questions, TriviaQA, WebQSP) dated 2026-06-05.
Rater A's complete per-axis records, including the verbatim quote behind each
assignment, are `rater_A_assignments.md` and
`rater_A_assignments_expansion.md`; both raters' bins for all 29 judgments are
inlined as `ROWS` in `reliability_kappa_ci.py` and tabulated in
`per_axis_assignments.json`, and the six judgments of 2026-06-05 with both
raters' bins and quotes are in `quotes_remaining_benchmarks.md`.

## Primary-rater identity preservation status

Exact identities and decoding settings for the two primary census raters were not recorded. The five models above belong only to the later sensitivity run. See `PRIMARY_RATER_PROVENANCE.md`; no primary model identity should be inferred from this table.
