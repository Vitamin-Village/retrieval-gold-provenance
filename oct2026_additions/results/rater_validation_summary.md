# Agreement of the census raters with the census and with the human raters (analysis/rater_validation.py)

Human raters: the v4 study (R1, R2, R3; excerpt-only; human_validation/v4/result_v4.md) and their item-wise median; an earlier non-author rater (short prompt and condensed descriptions; external_rater_result.json).

## Humans against the census

| rater | scope | QWK | exact | within 1 | derived-vs-independent agree |
|---|---|---:|---:|---:|---:|
| v4 R1 | all29 | 0.875 | 22/29 | 27/29 | 27/29 |
| v4 R1 | external23 | 0.835 | 16/23 | 21/23 | 21/23 |
| v4 R2 | all29 | 0.836 | 22/29 | 25/29 | 25/29 |
| v4 R2 | external23 | 0.784 | 16/23 | 19/23 | 19/23 |
| v4 R3 | all29 | 0.816 | 22/29 | 25/29 | 25/29 |
| v4 R3 | external23 | 0.758 | 16/23 | 19/23 | 19/23 |
| v4 median | all29 | 0.856 | 22/29 | 26/29 | 26/29 |
| v4 median | external23 | 0.810 | 16/23 | 20/23 | 20/23 |
| earlier non-author | all29 | 0.908 | 20/29 | 28/29 | 27/29 |
| earlier non-author | external23 | 0.885 | 15/23 | 22/23 | 21/23 |

## Human against human (29 items)

| pair | QWK | exact | within 1 |
|---|---:|---:|---:|
| v4 R1-R2 | 0.907 | 27 | 27 |
| v4 R1-R3 | 0.934 | 27 | 27 |
| v4 R2-R3 | 0.933 | 27 | 27 |
| earlier non-author - v4 R1 | 0.734 | 19 | 24 |
| earlier non-author - v4 R2 | 0.725 | 19 | 24 |
| earlier non-author - v4 R3 | 0.662 | 19 | 23 |
| earlier non-author - v4 median | 0.715 | 19 | 23 |

## Each rater against the census and the human raters (29 items)

| rater | QWK vs census | exact | within 1 | derived agree | QWK vs v4 median [95% CI] | exact vs v4 median | QWK vs v4 R1/R2/R3 | QWK vs earlier human |
|---|---:|---:|---:|---:|---|---:|---|---:|
| census | 1.000 | 29 | 29 | 29 | 0.856 [0.680, 0.972] | 22 | 0.875/0.836/0.816 | 0.908 |
| primary LM A | 0.983 | 26 | 29 | 29 | 0.827 [0.605, 0.974] | 22 | 0.846/0.807/0.790 | 0.886 |
| primary LM B | 0.994 | 28 | 29 | 29 | 0.848 [0.672, 0.966] | 21 | 0.867/0.828/0.808 | 0.901 |
| claude-opus-5-5 (workbook) | 0.885 | 23 | 26 | 26 | 0.819 [0.591, 0.975] | 24 | 0.776/0.865/0.844 | 0.819 |
| claude-opus-5-5 (excerpt-only) | 0.789 | 22 | 24 | 24 | 0.792 [0.539, 0.971] | 24 | 0.747/0.838/0.817 | 0.733 |
| claude-sonnet-5-5 (workbook) | 0.959 | 25 | 28 | 28 | 0.876 [0.669, 0.990] | 24 | 0.831/0.857/0.837 | 0.895 |
| claude-sonnet-5-5 (excerpt-only) | 0.891 | 24 | 26 | 26 | 0.858 [0.640, 0.995] | 25 | 0.814/0.903/0.820 | 0.855 |
| gemini-3.8-flash (workbook) | 0.965 | 26 | 28 | 28 | 0.850 [0.633, 0.990] | 24 | 0.806/0.831/0.812 | 0.900 |
| gemini-3.8-flash (excerpt-only) | 0.936 | 24 | 27 | 27 | 0.834 [0.609, 0.988] | 24 | 0.790/0.879/0.796 | 0.894 |
| claude-haiku-4-5 (workbook) | 0.875 | 21 | 26 | 26 | 0.821 [0.596, 0.976] | 24 | 0.778/0.866/0.846 | 0.769 |
| claude-haiku-4-5 (excerpt-only) | 0.835 | 22 | 25 | 25 | 0.856 [0.628, 0.995] | 25 | 0.809/0.806/0.858 | 0.754 |

The census is adjudicated from primary LM raters A and B and is not independent of them. Agreement is descriptive; the alternative annotator test against the three v4 humans is in human_validation/v4/result_v4.md.
