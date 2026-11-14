# Training-exposure strata: reader `qwen2.5:14b`

Evidence-triple closed-book recall on 2WikiMultiHop: 0.154 over 800 triple instances.

| bench | split | system | stratum | n | EM swap | 95% CI | +/- | p |
|---|---|---|---|---:|---:|---|---|---:|
| 2wiki | sub200 | audited | all | 200 | +0.045 | [-0.010, +0.100] | 21/12 | 0.16 |
| 2wiki | sub200 | audited | closed-book wrong | 192 | +0.052 | [-0.005, +0.109] | 21/11 | 0.11 |
| 2wiki | sub200 | audited | closed-book right | 8 | -0.125 | [-0.375, +0.000] | 0/1 | 1 |
| 2wiki | sub200 | audited | evidence not recalled | 191 | +0.042 | [-0.016, +0.099] | 20/12 | 0.22 |
| 2wiki | sub200 | audited | evidence recalled | 9 | +0.111 | [+0.000, +0.333] | 1/0 | 1 |
| 2wiki | sub200 | audited | neither (cb wrong, evidence not recalled) | 185 | +0.049 | [-0.011, +0.108] | 20/11 | 0.15 |
| 2wiki | sub200 | audited | strict (cb wrong, no evidence triple recalled) | 151 | +0.046 | [-0.020, +0.113] | 16/9 | 0.23 |
| 2wiki | holdout200 | audited | all | 200 | +0.065 | [+0.010, +0.120] | 23/10 | 0.035 |
| 2wiki | holdout200 | audited | closed-book wrong | 189 | +0.063 | [+0.005, +0.122] | 22/10 | 0.05 |
| 2wiki | holdout200 | audited | closed-book right | 11 | +0.091 | [+0.000, +0.273] | 1/0 | 1 |
| 2wiki | holdout200 | audited | evidence not recalled | 187 | +0.053 | [-0.005, +0.107] | 20/10 | 0.099 |
| 2wiki | holdout200 | audited | evidence recalled | 13 | +0.231 | [+0.000, +0.462] | 3/0 | 0.25 |
| 2wiki | holdout200 | audited | neither (cb wrong, evidence not recalled) | 177 | +0.051 | [-0.006, +0.107] | 19/10 | 0.14 |
| 2wiki | holdout200 | audited | strict (cb wrong, no evidence triple recalled) | 144 | +0.042 | [-0.028, +0.111] | 15/9 | 0.31 |
| 2wiki | pooled | audited | all | 400 | +0.055 | [+0.017, +0.095] | 44/22 | 0.0092 |
| 2wiki | pooled | audited | closed-book wrong | 381 | +0.058 | [+0.018, +0.100] | 43/21 | 0.0081 |
| 2wiki | pooled | audited | closed-book right | 19 | +0.000 | [-0.158, +0.158] | 1/1 | 1 |
| 2wiki | pooled | audited | evidence not recalled | 378 | +0.048 | [+0.008, +0.087] | 40/22 | 0.03 |
| 2wiki | pooled | audited | evidence recalled | 22 | +0.182 | [+0.045, +0.364] | 4/0 | 0.12 |
| 2wiki | pooled | audited | neither (cb wrong, evidence not recalled) | 362 | +0.050 | [+0.008, +0.091] | 39/21 | 0.027 |
| 2wiki | pooled | audited | strict (cb wrong, no evidence triple recalled) | 295 | +0.044 | [+0.000, +0.092] | 31/18 | 0.085 |
| 2wiki | sub200 | graph_centric | all | 200 | +0.165 | [+0.095, +0.230] | 43/10 | 5.6e-06 |
| 2wiki | sub200 | graph_centric | closed-book wrong | 192 | +0.172 | [+0.104, +0.240] | 41/8 | 2e-06 |
| 2wiki | sub200 | graph_centric | closed-book right | 8 | +0.000 | [-0.500, +0.500] | 2/2 | 1 |
| 2wiki | sub200 | graph_centric | evidence not recalled | 191 | +0.162 | [+0.094, +0.230] | 40/9 | 9.3e-06 |
| 2wiki | sub200 | graph_centric | evidence recalled | 9 | +0.222 | [-0.222, +0.667] | 3/1 | 0.62 |
| 2wiki | sub200 | graph_centric | neither (cb wrong, evidence not recalled) | 185 | +0.168 | [+0.097, +0.238] | 39/8 | 5.5e-06 |
| 2wiki | sub200 | graph_centric | strict (cb wrong, no evidence triple recalled) | 151 | +0.172 | [+0.093, +0.245] | 32/6 | 2.4e-05 |
| 2wiki | holdout200 | graph_centric | all | 200 | +0.170 | [+0.105, +0.240] | 43/9 | 2e-06 |
| 2wiki | holdout200 | graph_centric | closed-book wrong | 189 | +0.159 | [+0.095, +0.228] | 38/8 | 9.2e-06 |
| 2wiki | holdout200 | graph_centric | closed-book right | 11 | +0.364 | [+0.000, +0.727] | 5/1 | 0.22 |
| 2wiki | holdout200 | graph_centric | evidence not recalled | 187 | +0.171 | [+0.102, +0.241] | 41/9 | 5.6e-06 |
| 2wiki | holdout200 | graph_centric | evidence recalled | 13 | +0.154 | [+0.000, +0.385] | 2/0 | 0.5 |
| 2wiki | holdout200 | graph_centric | neither (cb wrong, evidence not recalled) | 177 | +0.164 | [+0.096, +0.237] | 37/8 | 1.5e-05 |
| 2wiki | holdout200 | graph_centric | strict (cb wrong, no evidence triple recalled) | 144 | +0.181 | [+0.097, +0.264] | 33/7 | 4.2e-05 |
| 2wiki | pooled | graph_centric | all | 400 | +0.168 | [+0.120, +0.215] | 86/19 | 2.3e-11 |
| 2wiki | pooled | graph_centric | closed-book wrong | 381 | +0.165 | [+0.118, +0.213] | 79/16 | 3.5e-11 |
| 2wiki | pooled | graph_centric | closed-book right | 19 | +0.211 | [-0.105, +0.526] | 7/3 | 0.34 |
| 2wiki | pooled | graph_centric | evidence not recalled | 378 | +0.167 | [+0.119, +0.214] | 81/18 | 1e-10 |
| 2wiki | pooled | graph_centric | evidence recalled | 22 | +0.182 | [+0.000, +0.364] | 5/1 | 0.22 |
| 2wiki | pooled | graph_centric | neither (cb wrong, evidence not recalled) | 362 | +0.166 | [+0.119, +0.215] | 76/16 | 1.6e-10 |
| 2wiki | pooled | graph_centric | strict (cb wrong, no evidence triple recalled) | 295 | +0.176 | [+0.122, +0.231] | 65/13 | 1.8e-09 |
| 2wiki | sub200 | ppr | all | 200 | +0.345 | [+0.260, +0.425] | 83/14 | 4.2e-13 |
| 2wiki | sub200 | ppr | closed-book wrong | 192 | +0.344 | [+0.260, +0.432] | 79/13 | 1.1e-12 |
| 2wiki | sub200 | ppr | closed-book right | 8 | +0.375 | [-0.125, +0.875] | 4/1 | 0.38 |
| 2wiki | sub200 | ppr | evidence not recalled | 191 | +0.346 | [+0.262, +0.429] | 79/13 | 1.1e-12 |
| 2wiki | sub200 | ppr | evidence recalled | 9 | +0.333 | [-0.111, +0.778] | 4/1 | 0.38 |
| 2wiki | sub200 | ppr | neither (cb wrong, evidence not recalled) | 185 | +0.341 | [+0.254, +0.427] | 76/13 | 5.4e-12 |
| 2wiki | sub200 | ppr | strict (cb wrong, no evidence triple recalled) | 151 | +0.331 | [+0.238, +0.424] | 60/10 | 8e-10 |
| 2wiki | holdout200 | ppr | all | 200 | +0.300 | [+0.220, +0.380] | 72/12 | 1.4e-11 |
| 2wiki | holdout200 | ppr | closed-book wrong | 189 | +0.317 | [+0.238, +0.397] | 70/10 | 3.2e-12 |
| 2wiki | holdout200 | ppr | closed-book right | 11 | +0.000 | [-0.364, +0.364] | 2/2 | 1 |
| 2wiki | holdout200 | ppr | evidence not recalled | 187 | +0.299 | [+0.214, +0.380] | 68/12 | 1.2e-10 |
| 2wiki | holdout200 | ppr | evidence recalled | 13 | +0.308 | [+0.077, +0.538] | 4/0 | 0.12 |
| 2wiki | holdout200 | ppr | neither (cb wrong, evidence not recalled) | 177 | +0.316 | [+0.232, +0.401] | 66/10 | 3e-11 |
| 2wiki | holdout200 | ppr | strict (cb wrong, no evidence triple recalled) | 144 | +0.319 | [+0.222, +0.417] | 55/9 | 3.5e-09 |
| 2wiki | pooled | ppr | all | 400 | +0.323 | [+0.263, +0.380] | 155/26 | 1.5e-23 |
| 2wiki | pooled | ppr | closed-book wrong | 381 | +0.331 | [+0.270, +0.388] | 149/23 | 8.5e-24 |
| 2wiki | pooled | ppr | closed-book right | 19 | +0.158 | [-0.158, +0.474] | 6/3 | 0.51 |
| 2wiki | pooled | ppr | evidence not recalled | 378 | +0.323 | [+0.262, +0.381] | 147/25 | 3.2e-22 |
| 2wiki | pooled | ppr | evidence recalled | 22 | +0.318 | [+0.091, +0.545] | 8/1 | 0.039 |
| 2wiki | pooled | ppr | neither (cb wrong, evidence not recalled) | 362 | +0.329 | [+0.268, +0.390] | 142/23 | 4e-22 |
| 2wiki | pooled | ppr | strict (cb wrong, no evidence triple recalled) | 295 | +0.325 | [+0.258, +0.393] | 115/19 | 6.1e-18 |
| metaqa | sub200 | audited | all | 200 | -0.015 | [-0.050, +0.025] | 6/9 | 0.61 |
| metaqa | sub200 | audited | closed-book wrong | 171 | -0.012 | [-0.053, +0.029] | 6/8 | 0.79 |
| metaqa | sub200 | audited | closed-book right | 29 | -0.034 | [-0.103, +0.000] | 0/1 | 1 |
| metaqa | holdout200 | audited | all | 200 | -0.010 | [-0.050, +0.030] | 7/9 | 0.8 |
| metaqa | holdout200 | audited | closed-book wrong | 165 | -0.024 | [-0.067, +0.012] | 4/8 | 0.39 |
| metaqa | holdout200 | audited | closed-book right | 35 | +0.057 | [-0.057, +0.171] | 3/1 | 0.62 |
| metaqa | pooled | audited | all | 400 | -0.013 | [-0.040, +0.015] | 13/18 | 0.47 |
| metaqa | pooled | audited | closed-book wrong | 336 | -0.018 | [-0.048, +0.012] | 10/16 | 0.33 |
| metaqa | pooled | audited | closed-book right | 64 | +0.016 | [-0.047, +0.078] | 3/2 | 1 |
| metaqa | sub200 | graph_centric | all | 200 | -0.010 | [-0.060, +0.045] | 14/16 | 0.86 |
| metaqa | sub200 | graph_centric | closed-book wrong | 171 | -0.018 | [-0.076, +0.041] | 12/15 | 0.7 |
| metaqa | sub200 | graph_centric | closed-book right | 29 | +0.034 | [-0.069, +0.172] | 2/1 | 1 |
| metaqa | holdout200 | graph_centric | all | 200 | +0.015 | [-0.035, +0.065] | 15/12 | 0.7 |
| metaqa | holdout200 | graph_centric | closed-book wrong | 165 | +0.006 | [-0.048, +0.061] | 12/11 | 1 |
| metaqa | holdout200 | graph_centric | closed-book right | 35 | +0.057 | [-0.057, +0.171] | 3/1 | 0.62 |
| metaqa | pooled | graph_centric | all | 400 | +0.003 | [-0.035, +0.040] | 29/28 | 1 |
| metaqa | pooled | graph_centric | closed-book wrong | 336 | -0.006 | [-0.048, +0.036] | 24/26 | 0.89 |
| metaqa | pooled | graph_centric | closed-book right | 64 | +0.047 | [-0.031, +0.125] | 5/2 | 0.45 |
| metaqa | sub200 | ppr | all | 200 | +0.010 | [-0.030, +0.050] | 9/7 | 0.8 |
| metaqa | sub200 | ppr | closed-book wrong | 171 | +0.018 | [-0.023, +0.058] | 9/6 | 0.61 |
| metaqa | sub200 | ppr | closed-book right | 29 | -0.034 | [-0.103, +0.000] | 0/1 | 1 |
| metaqa | holdout200 | ppr | all | 200 | -0.015 | [-0.065, +0.035] | 11/14 | 0.69 |
| metaqa | holdout200 | ppr | closed-book wrong | 165 | -0.006 | [-0.061, +0.042] | 9/10 | 1 |
| metaqa | holdout200 | ppr | closed-book right | 35 | -0.057 | [-0.200, +0.086] | 2/4 | 0.69 |
| metaqa | pooled | ppr | all | 400 | -0.003 | [-0.035, +0.030] | 20/21 | 1 |
| metaqa | pooled | ppr | closed-book wrong | 336 | +0.006 | [-0.027, +0.039] | 18/16 | 0.86 |
| metaqa | pooled | ppr | closed-book right | 64 | -0.047 | [-0.125, +0.031] | 2/5 | 0.45 |
| stark | sub200 | audited | all | 200 | +0.000 | [-0.040, +0.040] | 8/8 | 1 |
| stark | sub200 | audited | closed-book wrong | 197 | +0.000 | [-0.041, +0.041] | 8/8 | 1 |
| stark | sub200 | audited | closed-book right | 3 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | holdout200 | audited | all | 200 | -0.010 | [-0.050, +0.030] | 7/9 | 0.8 |
| stark | holdout200 | audited | closed-book wrong | 193 | -0.010 | [-0.052, +0.031] | 7/9 | 0.8 |
| stark | holdout200 | audited | closed-book right | 7 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | pooled | audited | all | 400 | -0.005 | [-0.033, +0.022] | 15/17 | 0.86 |
| stark | pooled | audited | closed-book wrong | 390 | -0.005 | [-0.033, +0.023] | 15/17 | 0.86 |
| stark | pooled | audited | closed-book right | 10 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | sub200 | graph_centric | all | 200 | +0.030 | [-0.010, +0.070] | 11/5 | 0.21 |
| stark | sub200 | graph_centric | closed-book wrong | 197 | +0.030 | [-0.010, +0.071] | 11/5 | 0.21 |
| stark | sub200 | graph_centric | closed-book right | 3 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | holdout200 | graph_centric | all | 200 | +0.030 | [-0.015, +0.075] | 14/8 | 0.29 |
| stark | holdout200 | graph_centric | closed-book wrong | 193 | +0.026 | [-0.021, +0.073] | 13/8 | 0.38 |
| stark | holdout200 | graph_centric | closed-book right | 7 | +0.143 | [+0.000, +0.429] | 1/0 | 1 |
| stark | pooled | graph_centric | all | 400 | +0.030 | [+0.000, +0.060] | 25/13 | 0.073 |
| stark | pooled | graph_centric | closed-book wrong | 390 | +0.028 | [-0.003, +0.059] | 24/13 | 0.099 |
| stark | pooled | graph_centric | closed-book right | 10 | +0.100 | [+0.000, +0.300] | 1/0 | 1 |
| stark | sub200 | ppr | all | 200 | +0.000 | [-0.030, +0.030] | 5/5 | 1 |
| stark | sub200 | ppr | closed-book wrong | 197 | +0.000 | [-0.030, +0.030] | 5/5 | 1 |
| stark | sub200 | ppr | closed-book right | 3 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | holdout200 | ppr | all | 200 | +0.020 | [-0.030, +0.070] | 14/10 | 0.54 |
| stark | holdout200 | ppr | closed-book wrong | 193 | +0.000 | [-0.047, +0.047] | 10/10 | 1 |
| stark | holdout200 | ppr | closed-book right | 7 | +0.571 | [+0.143, +0.857] | 4/0 | 0.12 |
| stark | pooled | ppr | all | 400 | +0.010 | [-0.018, +0.040] | 19/15 | 0.61 |
| stark | pooled | ppr | closed-book wrong | 390 | +0.000 | [-0.028, +0.028] | 15/15 | 1 |
| stark | pooled | ppr | closed-book right | 10 | +0.400 | [+0.100, +0.700] | 4/0 | 0.12 |
| graphragbench | sub200 | audited | all | 200 | +0.015 | [-0.025, +0.050] | 9/6 | 0.61 |
| graphragbench | sub200 | audited | closed-book wrong | 85 | +0.000 | [-0.059, +0.059] | 3/3 | 1 |
| graphragbench | sub200 | audited | closed-book right | 115 | +0.026 | [-0.026, +0.078] | 6/3 | 0.51 |
| graphragbench | holdout200 | audited | all | 200 | +0.010 | [-0.035, +0.055] | 12/10 | 0.83 |
| graphragbench | holdout200 | audited | closed-book wrong | 90 | +0.022 | [-0.044, +0.089] | 6/4 | 0.75 |
| graphragbench | holdout200 | audited | closed-book right | 110 | +0.000 | [-0.064, +0.064] | 6/6 | 1 |
| graphragbench | pooled | audited | all | 400 | +0.013 | [-0.018, +0.043] | 21/16 | 0.51 |
| graphragbench | pooled | audited | closed-book wrong | 175 | +0.011 | [-0.034, +0.057] | 9/7 | 0.8 |
| graphragbench | pooled | audited | closed-book right | 225 | +0.013 | [-0.027, +0.053] | 12/9 | 0.66 |
| graphragbench | sub200 | graph_centric | all | 200 | +0.040 | [-0.025, +0.105] | 26/18 | 0.29 |
| graphragbench | sub200 | graph_centric | closed-book wrong | 85 | +0.106 | [+0.000, +0.212] | 16/7 | 0.093 |
| graphragbench | sub200 | graph_centric | closed-book right | 115 | -0.009 | [-0.087, +0.070] | 10/11 | 1 |
| graphragbench | holdout200 | graph_centric | all | 200 | +0.065 | [+0.005, +0.125] | 26/13 | 0.053 |
| graphragbench | holdout200 | graph_centric | closed-book wrong | 90 | +0.056 | [-0.033, +0.144] | 11/6 | 0.33 |
| graphragbench | holdout200 | graph_centric | closed-book right | 110 | +0.073 | [-0.009, +0.155] | 15/7 | 0.13 |
| graphragbench | pooled | graph_centric | all | 400 | +0.052 | [+0.010, +0.098] | 52/31 | 0.028 |
| graphragbench | pooled | graph_centric | closed-book wrong | 175 | +0.080 | [+0.011, +0.149] | 27/13 | 0.038 |
| graphragbench | pooled | graph_centric | closed-book right | 225 | +0.031 | [-0.022, +0.089] | 25/18 | 0.36 |
| graphragbench | sub200 | ppr | all | 200 | +0.220 | [+0.145, +0.295] | 57/13 | 1e-07 |
| graphragbench | sub200 | ppr | closed-book wrong | 85 | +0.282 | [+0.153, +0.400] | 30/6 | 7e-05 |
| graphragbench | sub200 | ppr | closed-book right | 115 | +0.174 | [+0.078, +0.270] | 27/7 | 0.00082 |
| graphragbench | holdout200 | ppr | all | 200 | +0.100 | [+0.015, +0.180] | 46/26 | 0.024 |
| graphragbench | holdout200 | ppr | closed-book wrong | 90 | +0.133 | [+0.000, +0.267] | 24/12 | 0.065 |
| graphragbench | holdout200 | ppr | closed-book right | 110 | +0.073 | [-0.036, +0.182] | 22/14 | 0.24 |
| graphragbench | pooled | ppr | all | 400 | +0.160 | [+0.105, +0.215] | 103/39 | 7.6e-08 |
| graphragbench | pooled | ppr | closed-book wrong | 175 | +0.206 | [+0.114, +0.297] | 54/18 | 2.6e-05 |
| graphragbench | pooled | ppr | closed-book right | 225 | +0.124 | [+0.053, +0.196] | 49/21 | 0.0011 |
