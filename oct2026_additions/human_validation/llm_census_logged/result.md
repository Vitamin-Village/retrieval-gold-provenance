# Logged LM re-application of the census procedure: results

Generated 2026-10-10T10:42:13.007106+00:00 by `eval/llm_census_logged.py --score` from `raw.jsonl`. LM raters, not human raters. Design: PROTOCOL.md. κ = quadratically weighted Cohen's kappa; exact / within-1 = share of items; binary = agreement on derived (bin ≥ 2) versus independent.

## Requests

| model | returned ids | stored replies | API errors logged (retried) | unparsed or refused | input tok | output tok | cache read | cache write | est. cost USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-opus-5-5 | claude-opus-5-5 | 612 | 0 | 0 | 262203 | 290214 | 2218752 | 0 | 7.30 |
| claude-sonnet-5-5 | claude-sonnet-5-5 | 612 | 0 | 0 | 262203 | 189695 | 2205108 | 13644 | 2.90 |
| claude-haiku-4-5 | claude-haiku-4-5-20251001 | 612 | 427 | 0 | 1893102 | 325181 | 0 | 0 | 3.52 |
| gemini-3.8-flash | gemini-3.8-flash | 612 | 0 | 1 | 1797372 | 30022 | 0 | 0 | 0.00 |

## Quiz, attention items, parse failures

| run | complete | quiz (final bin) | attention CHECK_TEXT / CHECK_KG | excluded by human rules | failed parses | stop reasons |
|---|---|---:|---|---|---|---|
| claude-opus-5-5_A_1 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-opus-5-5_A_2 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-opus-5-5_A_3 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-opus-5-5_B_1 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-opus-5-5_B_2 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-opus-5-5_B_3 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-sonnet-5-5_A_1 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-sonnet-5-5_A_2 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-sonnet-5-5_A_3 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-sonnet-5-5_B_1 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-sonnet-5-5_B_2 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-sonnet-5-5_B_3 | yes | 6/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-haiku-4-5_A_1 | yes | 4/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-haiku-4-5_A_2 | yes | 4/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-haiku-4-5_A_3 | yes | 4/6 | 0 / 4 | no | 0 [] | {'end_turn': 167} |
| claude-haiku-4-5_B_1 | yes | 4/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-haiku-4-5_B_2 | yes | 4/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| claude-haiku-4-5_B_3 | yes | 5/6 | 0 / 4 | no | 0 [] | {'end_turn': 37} |
| gemini-3.8-flash_A_1 | yes | 6/6 | 0 / 4 | no | 0 [] | {'STOP': 167} |
| gemini-3.8-flash_A_2 | yes | 6/6 | 0 / 4 | no | 0 [] | {'STOP': 167} |
| gemini-3.8-flash_A_3 | yes | 6/6 | 0 / 4 | no | 0 [] | {'STOP': 167} |
| gemini-3.8-flash_B_1 | yes | 6/6 | 0 / 4 | no | 0 [] | {'STOP': 37} |
| gemini-3.8-flash_B_2 | yes | 6/6 | 0 / 4 | no | 1 ['partA:MUSIQUE_ANSWER'] | {'STOP': 37} |
| gemini-3.8-flash_B_3 | yes | 6/6 | 0 / 4 | no | 0 [] | {'STOP': 37} |

## Agreement with the census (29 items; 23 external in brackets)

| rater | n | κ census | exact | within-1 | binary | κ census [23 ext] |
|---|---:|---:|---:|---:|---:|---:|
| **condition A** | | | | | | |
| claude-opus-5-5_A_1 | 29 | 0.885 | 0.793 | 0.897 | 0.897 | 0.848 |
| claude-opus-5-5_A_2 | 29 | 0.885 | 0.793 | 0.897 | 0.897 | 0.848 |
| claude-opus-5-5_A_3 | 29 | 0.885 | 0.793 | 0.897 | 0.897 | 0.848 |
| **claude-opus-5-5 A median of 3 run(s)** | 29 | 0.885 | 0.793 | 0.897 | 0.897 | 0.848 |
| claude-sonnet-5-5_A_1 | 29 | 0.959 | 0.862 | 0.966 | 0.966 | 0.946 |
| claude-sonnet-5-5_A_2 | 29 | 0.936 | 0.828 | 0.931 | 0.931 | 0.915 |
| claude-sonnet-5-5_A_3 | 29 | 0.959 | 0.862 | 0.966 | 0.966 | 0.946 |
| **claude-sonnet-5-5 A median of 3 run(s)** | 29 | 0.959 | 0.862 | 0.966 | 0.966 | 0.946 |
| claude-haiku-4-5_A_1 | 29 | 0.902 | 0.793 | 0.931 | 0.931 | 0.869 |
| claude-haiku-4-5_A_2 | 29 | 0.893 | 0.690 | 0.897 | 0.897 | 0.868 |
| claude-haiku-4-5_A_3 | 29 | 0.875 | 0.724 | 0.897 | 0.897 | 0.835 |
| **claude-haiku-4-5 A median of 3 run(s)** | 29 | 0.875 | 0.724 | 0.897 | 0.897 | 0.835 |
| gemini-3.8-flash_A_1 | 29 | 0.965 | 0.897 | 0.966 | 0.966 | 0.954 |
| gemini-3.8-flash_A_2 | 29 | 0.965 | 0.897 | 0.966 | 0.966 | 0.954 |
| gemini-3.8-flash_A_3 | 29 | 0.942 | 0.862 | 0.931 | 0.931 | 0.923 |
| **gemini-3.8-flash A median of 3 run(s)** | 29 | 0.965 | 0.897 | 0.966 | 0.966 | 0.954 |
| **condition B** | | | | | | |
| claude-opus-5-5_B_1 | 29 | 0.789 | 0.759 | 0.828 | 0.828 | 0.719 |
| claude-opus-5-5_B_2 | 29 | 0.791 | 0.793 | 0.828 | 0.828 | 0.720 |
| claude-opus-5-5_B_3 | 29 | 0.789 | 0.759 | 0.828 | 0.828 | 0.719 |
| **claude-opus-5-5 B median of 3 run(s)** | 29 | 0.789 | 0.759 | 0.828 | 0.828 | 0.719 |
| claude-sonnet-5-5_B_1 | 29 | 0.891 | 0.828 | 0.897 | 0.897 | 0.857 |
| claude-sonnet-5-5_B_2 | 29 | 0.936 | 0.828 | 0.931 | 0.931 | 0.915 |
| claude-sonnet-5-5_B_3 | 29 | 0.891 | 0.828 | 0.897 | 0.897 | 0.857 |
| **claude-sonnet-5-5 B median of 3 run(s)** | 29 | 0.891 | 0.828 | 0.897 | 0.897 | 0.857 |
| claude-haiku-4-5_B_1 | 29 | 0.832 | 0.724 | 0.862 | 0.862 | 0.777 |
| claude-haiku-4-5_B_2 | 29 | 0.834 | 0.793 | 0.862 | 0.828 | 0.775 |
| claude-haiku-4-5_B_3 | 29 | 0.798 | 0.690 | 0.862 | 0.862 | 0.747 |
| **claude-haiku-4-5 B median of 3 run(s)** | 29 | 0.835 | 0.759 | 0.862 | 0.862 | 0.779 |
| gemini-3.8-flash_B_1 | 29 | 0.936 | 0.828 | 0.931 | 0.931 | 0.915 |
| gemini-3.8-flash_B_2 | 28 | 0.932 | 0.821 | 0.929 | 0.929 | 0.909 |
| gemini-3.8-flash_B_3 | 29 | 0.936 | 0.828 | 0.931 | 0.931 | 0.915 |
| **gemini-3.8-flash B median of 3 run(s)** | 29 | 0.936 | 0.828 | 0.931 | 0.931 | 0.915 |

## Krippendorff's α (ordinal)

| scope | 29 items | 23 external | with census as extra rater |
|---|---:|---:|---:|
| within_claude-opus-5-5_A | 1.000 | 1.000 | - |
| within_claude-sonnet-5-5_A | 0.995 | 0.993 | - |
| within_claude-haiku-4-5_A | 0.955 | 0.958 | - |
| within_gemini-3.8-flash_A | 0.996 | 0.994 | - |
| across_model_medians_A | 0.963 | 0.947 | 0.952 |
| all_runs_A | 0.967 | 0.958 | 0.962 |
| within_claude-opus-5-5_B | 0.990 | 0.989 | - |
| within_claude-sonnet-5-5_B | 0.993 | 0.991 | - |
| within_claude-haiku-4-5_B | 0.893 | 0.875 | - |
| within_gemini-3.8-flash_B | 1.000 | 1.000 | - |
| across_model_medians_B | 0.936 | 0.911 | 0.921 |
| all_runs_B | 0.928 | 0.906 | 0.923 |

## Condition A (workbook) minus condition B (excerpt-only), agreement with the census

Differences of model medians; 95% paired bootstrap CI over items (10,000 resamples).

| model | scope | Δκ | Δexact | Δwithin-1 | Δbinary |
|---|---|---|---|---|---|
| claude-opus-5-5 | all29 | 0.096 [0.000, 0.250] | 0.034 [0.000, 0.103] | 0.069 [0.000, 0.172] | 0.069 [0.000, 0.172] |
| claude-opus-5-5 | external23 | 0.129 [0.000, 0.329] | 0.043 [0.000, 0.130] | 0.087 [0.000, 0.217] | 0.087 [0.000, 0.217] |
| claude-opus-5-5 | all29, mean over runs | 0.095 [-0.002, 0.250] | 0.023 [-0.034, 0.103] | | |
| claude-sonnet-5-5 | all29 | 0.068 [0.000, 0.193] | 0.034 [0.000, 0.103] | 0.069 [0.000, 0.172] | 0.069 [0.000, 0.172] |
| claude-sonnet-5-5 | external23 | 0.089 [0.000, 0.251] | 0.043 [0.000, 0.130] | 0.087 [0.000, 0.217] | 0.087 [0.000, 0.217] |
| claude-sonnet-5-5 | all29, mean over runs | 0.045 [0.000, 0.129] | 0.023 [0.000, 0.069] | | |
| claude-haiku-4-5 | all29 | 0.040 [-0.097, 0.203] | -0.034 [-0.138, 0.069] | 0.034 [-0.069, 0.138] | 0.034 [-0.069, 0.138] |
| claude-haiku-4-5 | external23 | 0.056 [-0.123, 0.267] | -0.043 [-0.174, 0.087] | 0.043 [-0.087, 0.174] | 0.043 [-0.087, 0.174] |
| claude-haiku-4-5 | all29, mean over runs | 0.069 [-0.030, 0.198] | 0.000 [-0.057, 0.057] | | |
| gemini-3.8-flash | all29 | 0.030 [0.000, 0.096] | 0.069 [0.000, 0.172] | 0.034 [0.000, 0.103] | 0.034 [0.000, 0.103] |
| gemini-3.8-flash | external23 | 0.039 [0.000, 0.127] | 0.087 [0.000, 0.217] | 0.043 [0.000, 0.130] | 0.043 [0.000, 0.130] |
| gemini-3.8-flash | all29, mean over runs | 0.023 [0.000, 0.072] | 0.060 [0.000, 0.155] | | |
| mean_over_models | all29 | 0.058 [-0.011, 0.159] | 0.026 [-0.026, 0.086] | 0.052 [0.000, 0.121] | 0.052 [0.000, 0.121] |
| mean_over_models | external23 | 0.078 [-0.014, 0.210] | 0.033 [-0.033, 0.109] | 0.065 [0.000, 0.152] | 0.065 [0.000, 0.152] |

## Headline KG-derived bins (Ĉ_KG = bin/4) of the six KG-constructed benchmarks

| rater | GRADE | STaRK | GraphRAG-Bench | 2WikiMultiHop | MetaQA | WebQSP | # external at ≥3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude-opus-5-5_A_1 | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-opus-5-5_A_2 | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-opus-5-5_A_3 | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-opus-5-5_B_1 | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-opus-5-5_B_2 | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-opus-5-5_B_3 | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-sonnet-5-5_A_1 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-sonnet-5-5_A_2 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-sonnet-5-5_A_3 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-sonnet-5-5_B_1 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-sonnet-5-5_B_2 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-sonnet-5-5_B_3 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-haiku-4-5_A_1 | 3 | 2 | 0 | 4 | 4 | 4 | 4 |
| claude-haiku-4-5_A_2 | 3 | 3 | 1 | 4 | 4 | 4 | 5 |
| claude-haiku-4-5_A_3 | 3 | 2 | 0 | 4 | 4 | 4 | 4 |
| claude-haiku-4-5_B_1 | 3 | 4 | 1 | 4 | 4 | 4 | 5 |
| claude-haiku-4-5_B_2 | 3 | 3 | 1 | 4 | 4 | 4 | 5 |
| claude-haiku-4-5_B_3 | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| gemini-3.8-flash_A_1 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| gemini-3.8-flash_A_2 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| gemini-3.8-flash_A_3 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| gemini-3.8-flash_B_1 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| gemini-3.8-flash_B_2 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| gemini-3.8-flash_B_3 | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-opus-5-5_A_median | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-opus-5-5_B_median | 3 | 3 | 0 | 4 | 4 | 4 | 5 |
| claude-sonnet-5-5_A_median | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-sonnet-5-5_B_median | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| claude-haiku-4-5_A_median | 3 | 2 | 0 | 4 | 4 | 4 | 4 |
| claude-haiku-4-5_B_median | 3 | 3 | 1 | 4 | 4 | 4 | 5 |
| gemini-3.8-flash_A_median | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| gemini-3.8-flash_B_median | 3 | 3 | 3 | 4 | 4 | 4 | 6 |
| census | 3 | 3 | 3 | 4 | 4 | 4 |  |

## Per-item bins (final bin; runs 1-3 and median)

| item | census | claude-opus-5-5 A | claude-opus-5-5 B | claude-sonnet-5-5 A | claude-sonnet-5-5 B | claude-haiku-4-5 A | claude-haiku-4-5 B | gemini-3.8-flash A | gemini-3.8-flash B |
|---|---:|---|---|---|---|---|---|---|---|
| MUSIQUE_PASSAGE | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/3 (0) | 0/0/0 (0) | 0/0/0 (0) |
| MUSIQUE_ANSWER | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/-/0 (0) |
| HOTPOT_SENTENCE | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| HOTPOT_ANSWER | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| MHRAG_EVIDENCE | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| MHRAG_ANSWER | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| OURSLOW_PASSAGE (anchor) | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| OURSLOW_ANSWER (anchor) | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| NQ_ANSWER | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| NQ_PASSAGE | 1 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| GRADE_PASSAGE | 1 | 3/3/3 (3) | 3/3/3 (3) | 1/3/1 (1) | 3/3/3 (3) | 1/3/3 (3) | 0/2/0 (0) | 1/1/3 (1) | 3/3/3 (3) |
| GRADE_EDGE | 3 | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) |
| STARK_TEXTUAL | 1 | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) |
| STARK_ANSWERENTITY | 3 | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 2/3/2 (2) | 4/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) |
| TRIVIA_ANSWER | 0 | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) | 0/0/0 (0) |
| TRIVIA_EVIDENCE | 3 | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/4/4 (4) | 3/3/4 (3) | 3/3/3 (3) | 3/3/3 (3) |
| GRB_SUBGRAPH | 3 | 0/0/0 (0) | 0/0/0 (0) | 3/3/3 (3) | 3/3/3 (3) | 0/1/0 (0) | 1/1/0 (1) | 3/3/3 (3) | 3/3/3 (3) |
| GRB_ANSWER | 3 | 2/2/2 (2) | 0/0/0 (0) | 2/2/2 (2) | 0/2/0 (0) | 2/2/2 (2) | 0/0/0 (0) | 3/3/3 (3) | 2/2/2 (2) |
| TWOWIKI_SENTENCE | 3 | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) |
| TWOWIKI_ANSWER | 3 | 4/4/4 (4) | 4/3/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/3/4 (4) | 4/4/4 (4) | 4/4/4 (4) |
| TWOWIKI_EDGE | 4 | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) |
| METAQA_ANSWER | 4 | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) |
| METAQA_PATH | 4 | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) |
| WEBQSP_ANSWERENTITY | 4 | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) |
| WEBQSP_PATH | 4 | 4/4/4 (4) | 1/1/1 (1) | 4/4/4 (4) | 4/4/4 (4) | 4/3/4 (4) | 1/1/4 (1) | 4/4/4 (4) | 4/4/4 (4) |
| OURSUP_ANSWER (anchor) | 3 | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) | 3/3/3 (3) |
| OURSUP_ENTITY (anchor) | 4 | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) |
| OURSUP_EDGE (anchor) | 4 | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/3/4 (4) | 4/4/3 (4) | 4/4/4 (4) | 4/4/4 (4) |
| OURSUP_PATH (anchor) | 4 | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) | 4/4/4 (4) |

## Part B (condition A)

| model | part | n | Q1 α (3 runs) | Q1 majority | Q2 majority rate | Q2 all agree | rule rate | majority vs rule |
|---|---|---:|---:|---|---:|---:|---:|---|
| claude-opus-5-5 | B1 | 40 | 0.864 | {2: 35, 1: 3, 0: 2} | 1.000 | 1.000 | 0.850 | 32/40 |
| claude-opus-5-5 | B2 | 40 | 1.000 | {1: 22, 2: 12, 0: 6} | 0.525 | 0.975 | 0.600 | 28/40 |
| claude-opus-5-5 | B3 | 20 | 0.523 | {2: 18, 1: 2} | 0.500 | 1.000 | 0.550 | 19/20 |
| claude-opus-5-5 | B4 | 20 | 1.000 | {2: 14, 1: 6} | 0.300 | 1.000 | 0.300 | 20/20 |
| claude-opus-5-5 | B5 | 10 | - | {2: 10} | 0.000 | 1.000 | - | - |
| claude-sonnet-5-5 | B1 | 40 | 1.000 | {2: 35, 1: 3, 0: 2} | 1.000 | 1.000 | 0.850 | 32/40 |
| claude-sonnet-5-5 | B2 | 40 | 0.905 | {1: 19, 2: 15, 0: 6} | 0.425 | 0.950 | 0.600 | 28/40 |
| claude-sonnet-5-5 | B3 | 20 | - | {2: 20} | 0.500 | 1.000 | 0.550 | 19/20 |
| claude-sonnet-5-5 | B4 | 20 | 0.909 | {2: 17, 0: 3} | 0.300 | 1.000 | 0.300 | 20/20 |
| claude-sonnet-5-5 | B5 | 10 | - | {2: 10} | 0.000 | 1.000 | - | - |
| claude-haiku-4-5 | B1 | 40 | 0.864 | {2: 38, 0: 2} | 1.000 | 1.000 | 0.850 | 32/40 |
| claude-haiku-4-5 | B2 | 40 | 0.923 | {1: 24, 2: 10, 0: 6} | 0.225 | 0.850 | 0.600 | 28/40 |
| claude-haiku-4-5 | B3 | 20 | 0.393 | {2: 16, 1: 4} | 0.500 | 1.000 | 0.550 | 19/20 |
| claude-haiku-4-5 | B4 | 20 | 0.533 | {1: 9, 2: 9, 0: 2} | 0.300 | 1.000 | 0.300 | 20/20 |
| claude-haiku-4-5 | B5 | 10 | - | {2: 10} | 0.000 | 1.000 | - | - |
| gemini-3.8-flash | B1 | 40 | 0.903 | {2: 37, 0: 2, 1: 1} | 1.000 | 1.000 | 0.850 | 32/40 |
| gemini-3.8-flash | B2 | 40 | 0.971 | {1: 20, 2: 12, 0: 8} | 0.475 | 0.950 | 0.600 | 28/40 |
| gemini-3.8-flash | B3 | 20 | 0.000 | {2: 20} | 0.500 | 1.000 | 0.550 | 19/20 |
| gemini-3.8-flash | B4 | 20 | 0.927 | {2: 15, 1: 3, 0: 2} | 0.300 | 1.000 | 0.300 | 20/20 |
| gemini-3.8-flash | B5 | 10 | - | {2: 10} | 0.000 | 1.000 | - | - |
