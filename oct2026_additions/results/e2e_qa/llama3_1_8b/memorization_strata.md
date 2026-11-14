# Training-exposure strata: reader `llama3.1:8b`

Evidence-triple closed-book recall on 2WikiMultiHop: 0.161 over 800 triple instances.

| bench | split | system | stratum | n | EM swap | 95% CI | +/- | p |
|---|---|---|---|---:|---:|---|---|---:|
| 2wiki | sub200 | audited | all | 200 | +0.050 | [+0.000, +0.100] | 18/8 | 0.076 |
| 2wiki | sub200 | audited | closed-book wrong | 193 | +0.057 | [+0.010, +0.109] | 18/7 | 0.043 |
| 2wiki | sub200 | audited | closed-book right | 7 | -0.143 | [-0.429, +0.000] | 0/1 | 1 |
| 2wiki | sub200 | audited | evidence not recalled | 192 | +0.052 | [+0.000, +0.104] | 18/8 | 0.076 |
| 2wiki | sub200 | audited | evidence recalled | 8 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| 2wiki | sub200 | audited | neither (cb wrong, evidence not recalled) | 185 | +0.059 | [+0.005, +0.114] | 18/7 | 0.043 |
| 2wiki | sub200 | audited | strict (cb wrong, no evidence triple recalled) | 141 | +0.071 | [+0.014, +0.128] | 14/4 | 0.031 |
| 2wiki | holdout200 | audited | all | 200 | +0.060 | [+0.015, +0.105] | 18/6 | 0.023 |
| 2wiki | holdout200 | audited | closed-book wrong | 193 | +0.067 | [+0.021, +0.114] | 18/5 | 0.011 |
| 2wiki | holdout200 | audited | closed-book right | 7 | -0.143 | [-0.429, +0.000] | 0/1 | 1 |
| 2wiki | holdout200 | audited | evidence not recalled | 191 | +0.063 | [+0.016, +0.115] | 18/6 | 0.023 |
| 2wiki | holdout200 | audited | evidence recalled | 9 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| 2wiki | holdout200 | audited | neither (cb wrong, evidence not recalled) | 184 | +0.071 | [+0.022, +0.120] | 18/5 | 0.011 |
| 2wiki | holdout200 | audited | strict (cb wrong, no evidence triple recalled) | 136 | +0.066 | [+0.007, +0.132] | 14/5 | 0.064 |
| 2wiki | pooled | audited | all | 400 | +0.055 | [+0.022, +0.090] | 36/14 | 0.0026 |
| 2wiki | pooled | audited | closed-book wrong | 386 | +0.062 | [+0.028, +0.096] | 36/12 | 0.00072 |
| 2wiki | pooled | audited | closed-book right | 14 | -0.143 | [-0.357, +0.000] | 0/2 | 0.5 |
| 2wiki | pooled | audited | evidence not recalled | 383 | +0.057 | [+0.023, +0.094] | 36/14 | 0.0026 |
| 2wiki | pooled | audited | evidence recalled | 17 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| 2wiki | pooled | audited | neither (cb wrong, evidence not recalled) | 369 | +0.065 | [+0.030, +0.103] | 36/12 | 0.00072 |
| 2wiki | pooled | audited | strict (cb wrong, no evidence triple recalled) | 277 | +0.069 | [+0.025, +0.112] | 28/9 | 0.0026 |
| 2wiki | sub200 | graph_centric | all | 200 | +0.140 | [+0.080, +0.200] | 34/6 | 8.4e-06 |
| 2wiki | sub200 | graph_centric | closed-book wrong | 193 | +0.150 | [+0.093, +0.212] | 34/5 | 2.4e-06 |
| 2wiki | sub200 | graph_centric | closed-book right | 7 | -0.143 | [-0.429, +0.000] | 0/1 | 1 |
| 2wiki | sub200 | graph_centric | evidence not recalled | 192 | +0.146 | [+0.083, +0.208] | 34/6 | 8.4e-06 |
| 2wiki | sub200 | graph_centric | evidence recalled | 8 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| 2wiki | sub200 | graph_centric | neither (cb wrong, evidence not recalled) | 185 | +0.157 | [+0.097, +0.222] | 34/5 | 2.4e-06 |
| 2wiki | sub200 | graph_centric | strict (cb wrong, no evidence triple recalled) | 141 | +0.128 | [+0.064, +0.199] | 22/4 | 0.00053 |
| 2wiki | holdout200 | graph_centric | all | 200 | +0.100 | [+0.040, +0.160] | 30/10 | 0.0022 |
| 2wiki | holdout200 | graph_centric | closed-book wrong | 193 | +0.109 | [+0.047, +0.171] | 30/9 | 0.0011 |
| 2wiki | holdout200 | graph_centric | closed-book right | 7 | -0.143 | [-0.429, +0.000] | 0/1 | 1 |
| 2wiki | holdout200 | graph_centric | evidence not recalled | 191 | +0.105 | [+0.042, +0.168] | 30/10 | 0.0022 |
| 2wiki | holdout200 | graph_centric | evidence recalled | 9 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| 2wiki | holdout200 | graph_centric | neither (cb wrong, evidence not recalled) | 184 | +0.114 | [+0.054, +0.179] | 30/9 | 0.0011 |
| 2wiki | holdout200 | graph_centric | strict (cb wrong, no evidence triple recalled) | 136 | +0.118 | [+0.044, +0.191] | 21/5 | 0.0025 |
| 2wiki | pooled | graph_centric | all | 400 | +0.120 | [+0.077, +0.163] | 64/16 | 5.9e-08 |
| 2wiki | pooled | graph_centric | closed-book wrong | 386 | +0.130 | [+0.085, +0.174] | 64/14 | 8.6e-09 |
| 2wiki | pooled | graph_centric | closed-book right | 14 | -0.143 | [-0.357, +0.000] | 0/2 | 0.5 |
| 2wiki | pooled | graph_centric | evidence not recalled | 383 | +0.125 | [+0.081, +0.170] | 64/16 | 5.9e-08 |
| 2wiki | pooled | graph_centric | evidence recalled | 17 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| 2wiki | pooled | graph_centric | neither (cb wrong, evidence not recalled) | 369 | +0.136 | [+0.092, +0.182] | 64/14 | 8.6e-09 |
| 2wiki | pooled | graph_centric | strict (cb wrong, no evidence triple recalled) | 277 | +0.123 | [+0.076, +0.173] | 43/9 | 2e-06 |
| 2wiki | sub200 | ppr | all | 200 | +0.330 | [+0.260, +0.400] | 70/4 | 1.3e-16 |
| 2wiki | sub200 | ppr | closed-book wrong | 193 | +0.316 | [+0.244, +0.389] | 65/4 | 3.1e-15 |
| 2wiki | sub200 | ppr | closed-book right | 7 | +0.714 | [+0.429, +1.000] | 5/0 | 0.062 |
| 2wiki | sub200 | ppr | evidence not recalled | 192 | +0.339 | [+0.266, +0.411] | 69/4 | 2.4e-16 |
| 2wiki | sub200 | ppr | evidence recalled | 8 | +0.125 | [+0.000, +0.375] | 1/0 | 1 |
| 2wiki | sub200 | ppr | neither (cb wrong, evidence not recalled) | 185 | +0.324 | [+0.249, +0.395] | 64/4 | 5.9e-15 |
| 2wiki | sub200 | ppr | strict (cb wrong, no evidence triple recalled) | 141 | +0.298 | [+0.213, +0.383] | 46/4 | 4.5e-10 |
| 2wiki | holdout200 | ppr | all | 200 | +0.260 | [+0.180, +0.340] | 67/15 | 5.3e-09 |
| 2wiki | holdout200 | ppr | closed-book wrong | 193 | +0.259 | [+0.176, +0.342] | 65/15 | 1.4e-08 |
| 2wiki | holdout200 | ppr | closed-book right | 7 | +0.286 | [+0.000, +0.571] | 2/0 | 0.5 |
| 2wiki | holdout200 | ppr | evidence not recalled | 191 | +0.262 | [+0.178, +0.346] | 65/15 | 1.4e-08 |
| 2wiki | holdout200 | ppr | evidence recalled | 9 | +0.222 | [+0.000, +0.556] | 2/0 | 0.5 |
| 2wiki | holdout200 | ppr | neither (cb wrong, evidence not recalled) | 184 | +0.261 | [+0.174, +0.348] | 63/15 | 3.7e-08 |
| 2wiki | holdout200 | ppr | strict (cb wrong, no evidence triple recalled) | 136 | +0.279 | [+0.176, +0.375] | 49/11 | 7.6e-07 |
| 2wiki | pooled | ppr | all | 400 | +0.295 | [+0.240, +0.350] | 137/19 | 3.1e-23 |
| 2wiki | pooled | ppr | closed-book wrong | 386 | +0.288 | [+0.231, +0.342] | 130/19 | 1.6e-21 |
| 2wiki | pooled | ppr | closed-book right | 14 | +0.500 | [+0.214, +0.786] | 7/0 | 0.016 |
| 2wiki | pooled | ppr | evidence not recalled | 383 | +0.300 | [+0.245, +0.355] | 134/19 | 1.7e-22 |
| 2wiki | pooled | ppr | evidence recalled | 17 | +0.176 | [+0.000, +0.353] | 3/0 | 0.25 |
| 2wiki | pooled | ppr | neither (cb wrong, evidence not recalled) | 369 | +0.293 | [+0.236, +0.350] | 127/19 | 8.4e-21 |
| 2wiki | pooled | ppr | strict (cb wrong, no evidence triple recalled) | 277 | +0.289 | [+0.224, +0.354] | 95/15 | 2.1e-15 |
| metaqa | sub200 | audited | all | 200 | -0.030 | [-0.065, +0.000] | 3/9 | 0.15 |
| metaqa | sub200 | audited | closed-book wrong | 170 | -0.024 | [-0.059, +0.012] | 3/7 | 0.34 |
| metaqa | sub200 | audited | closed-book right | 30 | -0.067 | [-0.167, +0.000] | 0/2 | 0.5 |
| metaqa | holdout200 | audited | all | 200 | -0.015 | [-0.050, +0.020] | 5/8 | 0.58 |
| metaqa | holdout200 | audited | closed-book wrong | 166 | -0.018 | [-0.048, +0.012] | 2/5 | 0.45 |
| metaqa | holdout200 | audited | closed-book right | 34 | +0.000 | [-0.147, +0.147] | 3/3 | 1 |
| metaqa | pooled | audited | all | 400 | -0.022 | [-0.048, +0.003] | 8/17 | 0.11 |
| metaqa | pooled | audited | closed-book wrong | 336 | -0.021 | [-0.045, +0.003] | 5/12 | 0.14 |
| metaqa | pooled | audited | closed-book right | 64 | -0.031 | [-0.125, +0.047] | 3/5 | 0.73 |
| metaqa | sub200 | graph_centric | all | 200 | +0.020 | [-0.035, +0.075] | 17/13 | 0.58 |
| metaqa | sub200 | graph_centric | closed-book wrong | 170 | +0.018 | [-0.041, +0.076] | 14/11 | 0.69 |
| metaqa | sub200 | graph_centric | closed-book right | 30 | +0.033 | [-0.100, +0.167] | 3/2 | 1 |
| metaqa | holdout200 | graph_centric | all | 200 | +0.025 | [-0.020, +0.075] | 14/9 | 0.4 |
| metaqa | holdout200 | graph_centric | closed-book wrong | 166 | +0.030 | [-0.018, +0.084] | 12/7 | 0.36 |
| metaqa | holdout200 | graph_centric | closed-book right | 34 | +0.000 | [-0.118, +0.118] | 2/2 | 1 |
| metaqa | pooled | graph_centric | all | 400 | +0.022 | [-0.015, +0.058] | 31/22 | 0.27 |
| metaqa | pooled | graph_centric | closed-book wrong | 336 | +0.024 | [-0.015, +0.062] | 26/18 | 0.29 |
| metaqa | pooled | graph_centric | closed-book right | 64 | +0.016 | [-0.078, +0.109] | 5/4 | 1 |
| metaqa | sub200 | ppr | all | 200 | +0.060 | [+0.010, +0.110] | 20/8 | 0.036 |
| metaqa | sub200 | ppr | closed-book wrong | 170 | +0.065 | [+0.006, +0.124] | 19/8 | 0.052 |
| metaqa | sub200 | ppr | closed-book right | 30 | +0.033 | [+0.000, +0.100] | 1/0 | 1 |
| metaqa | holdout200 | ppr | all | 200 | +0.010 | [-0.040, +0.055] | 13/11 | 0.84 |
| metaqa | holdout200 | ppr | closed-book wrong | 166 | +0.024 | [-0.030, +0.078] | 12/8 | 0.5 |
| metaqa | holdout200 | ppr | closed-book right | 34 | -0.059 | [-0.176, +0.059] | 1/3 | 0.62 |
| metaqa | pooled | ppr | all | 400 | +0.035 | [+0.000, +0.070] | 33/19 | 0.07 |
| metaqa | pooled | ppr | closed-book wrong | 336 | +0.045 | [+0.006, +0.083] | 31/16 | 0.04 |
| metaqa | pooled | ppr | closed-book right | 64 | -0.016 | [-0.078, +0.047] | 2/3 | 1 |
| stark | sub200 | audited | all | 200 | -0.030 | [-0.060, +0.000] | 2/8 | 0.11 |
| stark | sub200 | audited | closed-book wrong | 195 | -0.031 | [-0.062, +0.000] | 2/8 | 0.11 |
| stark | sub200 | audited | closed-book right | 5 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | holdout200 | audited | all | 200 | -0.015 | [-0.045, +0.015] | 3/6 | 0.51 |
| stark | holdout200 | audited | closed-book wrong | 195 | -0.015 | [-0.046, +0.015] | 3/6 | 0.51 |
| stark | holdout200 | audited | closed-book right | 5 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | pooled | audited | all | 400 | -0.022 | [-0.045, -0.003] | 5/14 | 0.064 |
| stark | pooled | audited | closed-book wrong | 390 | -0.023 | [-0.046, -0.003] | 5/14 | 0.064 |
| stark | pooled | audited | closed-book right | 10 | +0.000 | [+0.000, +0.000] | 0/0 | 1 |
| stark | sub200 | graph_centric | all | 200 | -0.010 | [-0.040, +0.015] | 3/5 | 0.73 |
| stark | sub200 | graph_centric | closed-book wrong | 195 | -0.005 | [-0.031, +0.021] | 3/4 | 1 |
| stark | sub200 | graph_centric | closed-book right | 5 | -0.200 | [-0.600, +0.000] | 0/1 | 1 |
| stark | holdout200 | graph_centric | all | 200 | +0.010 | [-0.025, +0.045] | 7/5 | 0.77 |
| stark | holdout200 | graph_centric | closed-book wrong | 195 | +0.015 | [-0.015, +0.051] | 7/4 | 0.55 |
| stark | holdout200 | graph_centric | closed-book right | 5 | -0.200 | [-0.600, +0.000] | 0/1 | 1 |
| stark | pooled | graph_centric | all | 400 | +0.000 | [-0.022, +0.022] | 10/10 | 1 |
| stark | pooled | graph_centric | closed-book wrong | 390 | +0.005 | [-0.015, +0.026] | 10/8 | 0.81 |
| stark | pooled | graph_centric | closed-book right | 10 | -0.200 | [-0.500, +0.000] | 0/2 | 0.5 |
| stark | sub200 | ppr | all | 200 | +0.000 | [-0.035, +0.040] | 7/7 | 1 |
| stark | sub200 | ppr | closed-book wrong | 195 | +0.005 | [-0.026, +0.036] | 6/5 | 1 |
| stark | sub200 | ppr | closed-book right | 5 | -0.200 | [-0.800, +0.400] | 1/2 | 1 |
| stark | holdout200 | ppr | all | 200 | +0.020 | [-0.030, +0.065] | 14/10 | 0.54 |
| stark | holdout200 | ppr | closed-book wrong | 195 | +0.000 | [-0.046, +0.046] | 10/10 | 1 |
| stark | holdout200 | ppr | closed-book right | 5 | +0.800 | [+0.400, +1.000] | 4/0 | 0.12 |
| stark | pooled | ppr | all | 400 | +0.010 | [-0.020, +0.040] | 21/17 | 0.63 |
| stark | pooled | ppr | closed-book wrong | 390 | +0.003 | [-0.026, +0.031] | 16/15 | 1 |
| stark | pooled | ppr | closed-book right | 10 | +0.300 | [-0.200, +0.800] | 5/2 | 0.45 |
| graphragbench | sub200 | audited | all | 200 | +0.050 | [+0.005, +0.100] | 18/8 | 0.076 |
| graphragbench | sub200 | audited | closed-book wrong | 81 | +0.062 | [-0.012, +0.136] | 7/2 | 0.18 |
| graphragbench | sub200 | audited | closed-book right | 119 | +0.042 | [-0.025, +0.109] | 11/6 | 0.33 |
| graphragbench | holdout200 | audited | all | 200 | +0.005 | [-0.040, +0.050] | 11/10 | 1 |
| graphragbench | holdout200 | audited | closed-book wrong | 86 | +0.012 | [-0.058, +0.093] | 6/5 | 1 |
| graphragbench | holdout200 | audited | closed-book right | 114 | +0.000 | [-0.053, +0.053] | 5/5 | 1 |
| graphragbench | pooled | audited | all | 400 | +0.028 | [-0.005, +0.060] | 29/18 | 0.14 |
| graphragbench | pooled | audited | closed-book wrong | 167 | +0.036 | [-0.018, +0.090] | 13/7 | 0.26 |
| graphragbench | pooled | audited | closed-book right | 233 | +0.021 | [-0.021, +0.064] | 16/11 | 0.44 |
| graphragbench | sub200 | graph_centric | all | 200 | +0.090 | [+0.025, +0.160] | 34/16 | 0.015 |
| graphragbench | sub200 | graph_centric | closed-book wrong | 81 | +0.136 | [+0.025, +0.259] | 18/7 | 0.043 |
| graphragbench | sub200 | graph_centric | closed-book right | 119 | +0.059 | [-0.025, +0.143] | 16/9 | 0.23 |
| graphragbench | holdout200 | graph_centric | all | 200 | +0.070 | [+0.000, +0.140] | 33/19 | 0.07 |
| graphragbench | holdout200 | graph_centric | closed-book wrong | 86 | +0.023 | [-0.081, +0.128] | 12/10 | 0.83 |
| graphragbench | holdout200 | graph_centric | closed-book right | 114 | +0.105 | [+0.018, +0.202] | 21/9 | 0.043 |
| graphragbench | pooled | graph_centric | all | 400 | +0.080 | [+0.033, +0.130] | 67/35 | 0.002 |
| graphragbench | pooled | graph_centric | closed-book wrong | 167 | +0.078 | [+0.000, +0.156] | 30/17 | 0.079 |
| graphragbench | pooled | graph_centric | closed-book right | 233 | +0.082 | [+0.021, +0.142] | 37/18 | 0.014 |
| graphragbench | sub200 | ppr | all | 200 | +0.175 | [+0.085, +0.265] | 62/27 | 0.00027 |
| graphragbench | sub200 | ppr | closed-book wrong | 81 | +0.259 | [+0.111, +0.407] | 31/10 | 0.0015 |
| graphragbench | sub200 | ppr | closed-book right | 119 | +0.118 | [+0.008, +0.227] | 31/17 | 0.059 |
| graphragbench | holdout200 | ppr | all | 200 | +0.045 | [-0.040, +0.130] | 43/34 | 0.36 |
| graphragbench | holdout200 | ppr | closed-book wrong | 86 | +0.070 | [-0.047, +0.198] | 18/12 | 0.36 |
| graphragbench | holdout200 | ppr | closed-book right | 114 | +0.026 | [-0.096, +0.149] | 25/22 | 0.77 |
| graphragbench | pooled | ppr | all | 400 | +0.110 | [+0.048, +0.172] | 105/61 | 0.00079 |
| graphragbench | pooled | ppr | closed-book wrong | 167 | +0.162 | [+0.066, +0.257] | 49/22 | 0.0018 |
| graphragbench | pooled | ppr | closed-book right | 233 | +0.073 | [-0.009, +0.155] | 56/39 | 0.1 |
