# Judge spot-check v3, LM labels (R1-R3 are three Claude models; see README.md)

pool judge-CORRECT rate used for weighting: 0.511

| rater | n | unsure | agreement | kappa | precision | recall (sample) | recall (pool-weighted) |
|---|---|---|---|---|---|---|---|
| R1 | 100 | 0 | 0.870 | 0.740 | 0.840 | 0.894 | 0.898 |
| R2 | 100 | 0 | 0.880 | 0.760 | 0.820 | 0.932 | 0.935 |
| R3 | 100 | 0 | 0.810 | 0.620 | 0.620 | 1.000 | 1.000 |
| majority | 100 | 0 | 0.870 | 0.740 | 0.800 | 0.930 | 0.933 |
| v1 human rater | 100 | 0 | 0.860 | 0.720 | 0.980 | 0.790 | 0.798 |
| majority, qwen2_5_14b | 80 | 0 | 0.900 | 0.797 | 0.811 | 0.968 | 0.973 |
| majority, llama3_1_8b | 20 | 0 | 0.750 | 0.468 | 0.769 | 0.833 | 0.738 |

Fleiss kappa (3 raters): 0.738

- R1-R2: n 100, agreement 0.930, kappa 0.859
- R1-R3: n 100, agreement 0.820, kappa 0.632
- R2-R3: n 100, agreement 0.870, kappa 0.728
