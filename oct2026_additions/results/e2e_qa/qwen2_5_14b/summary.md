# End-to-end QA over frozen swap vectors: reader `qwen2.5:14b`

## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader

| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| 2wiki | holdout200 | audited | 200 | +0.115 | 0.285 | 0.220 | +0.065 [+0.010, +0.120] | 3.51e-02 | +0.046 [+0.005, +0.087] | +0.065 |
| 2wiki | holdout200 | audited@bge | 200 | +0.080 | 0.210 | 0.170 | +0.040 [-0.010, +0.090] | 1.69e-01 | +0.017 [-0.019, +0.053] | +0.050 |
| 2wiki | holdout200 | graph_centric | 200 | +0.335 | 0.400 | 0.230 | +0.170 [+0.105, +0.240] | 2.04e-06 | +0.100 [+0.051, +0.152] | +0.160 |
| 2wiki | holdout200 | graph_centric@bge | 200 | +0.240 | 0.330 | 0.205 | +0.125 [+0.070, +0.185] | 4.13e-05 | +0.083 [+0.034, +0.133] | +0.130 |
| 2wiki | holdout200 | hipporag2 | 200 | +0.275 | 0.370 | 0.310 | +0.060 [-0.010, +0.130] | 1.19e-01 | +0.032 [-0.021, +0.085] | +0.065 |
| 2wiki | holdout200 | ppr | 200 | +0.310 | 0.555 | 0.255 | +0.300 [+0.220, +0.380] | 1.39e-11 | +0.242 [+0.183, +0.303] | +0.345 |
| 2wiki | sub200 | audited | 200 | +0.160 | 0.265 | 0.220 | +0.045 [-0.010, +0.100] | 1.63e-01 | +0.051 [+0.010, +0.094] | +0.060 |
| 2wiki | sub200 | audited@bge | 200 | +0.105 | 0.270 | 0.195 | +0.075 [+0.025, +0.125] | 5.92e-03 | +0.048 [+0.007, +0.090] | +0.070 |
| 2wiki | sub200 | graph_centric | 200 | +0.355 | 0.365 | 0.200 | +0.165 [+0.095, +0.230] | 5.55e-06 | +0.111 [+0.056, +0.165] | +0.170 |
| 2wiki | sub200 | graph_centric@bge | 200 | +0.295 | 0.395 | 0.230 | +0.165 [+0.100, +0.230] | 1.96e-06 | +0.109 [+0.059, +0.161] | +0.180 |
| 2wiki | sub200 | hipporag2 | 200 | +0.240 | 0.375 | 0.320 | +0.055 [-0.020, +0.130] | 2.00e-01 | +0.016 [-0.041, +0.075] | +0.035 |
| 2wiki | sub200 | ppr | 200 | +0.315 | 0.560 | 0.215 | +0.345 [+0.260, +0.425] | 4.22e-13 | +0.258 [+0.197, +0.319] | +0.340 |
| graphragbench | holdout200 | audited | 200 | +0.060 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.003 [-0.013, +0.019] | +0.000 |
| graphragbench | holdout200 | audited@bge | 200 | +0.100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.009 [-0.012, +0.030] | +0.000 |
| graphragbench | holdout200 | graph_centric | 200 | +0.280 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.036 [+0.013, +0.061] | +0.000 |
| graphragbench | holdout200 | graph_centric@bge | 200 | +0.215 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.027 [+0.005, +0.050] | +0.000 |
| graphragbench | holdout200 | hipporag2 | 200 | +0.225 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.021 [-0.001, +0.042] | +0.000 |
| graphragbench | holdout200 | ppr | 200 | +0.380 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.044 [+0.011, +0.076] | +0.000 |
| graphragbench | sub200 | audited | 200 | +0.065 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.010 [-0.006, +0.026] | +0.000 |
| graphragbench | sub200 | audited@bge | 200 | +0.125 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.022 [+0.003, +0.042] | +0.000 |
| graphragbench | sub200 | graph_centric | 200 | +0.290 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.006 [-0.017, +0.030] | +0.000 |
| graphragbench | sub200 | graph_centric@bge | 200 | +0.245 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.014 [-0.008, +0.036] | +0.000 |
| graphragbench | sub200 | hipporag2 | 200 | +0.265 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.012 [-0.007, +0.032] | +0.000 |
| graphragbench | sub200 | ppr | 200 | +0.490 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.049 [+0.018, +0.080] | +0.000 |
| metaqa | holdout200 | audited | 200 | +0.045 | 0.290 | 0.300 | -0.010 [-0.050, +0.030] | 8.04e-01 | +0.007 [-0.019, +0.033] | +0.010 |
| metaqa | holdout200 | audited@bge | 200 | +0.135 | 0.320 | 0.415 | -0.095 [-0.150, -0.040] | 1.32e-03 | -0.041 [-0.078, -0.005] | -0.020 |
| metaqa | holdout200 | graph_centric | 200 | +0.085 | 0.300 | 0.285 | +0.015 [-0.035, +0.065] | 7.01e-01 | +0.028 [-0.007, +0.066] | +0.025 |
| metaqa | holdout200 | graph_centric@bge | 200 | +0.280 | 0.290 | 0.365 | -0.075 [-0.135, -0.015] | 2.01e-02 | +0.024 [-0.019, +0.066] | +0.090 |
| metaqa | holdout200 | hipporag2 | 200 | +0.170 | 0.270 | 0.260 | +0.010 [-0.050, +0.070] | 8.68e-01 | +0.058 [+0.009, +0.109] | +0.110 |
| metaqa | holdout200 | ppr | 200 | +0.070 | 0.280 | 0.295 | -0.015 [-0.065, +0.035] | 6.90e-01 | +0.004 [-0.027, +0.036] | +0.025 |
| metaqa | sub200 | audited | 200 | +0.035 | 0.335 | 0.350 | -0.015 [-0.050, +0.025] | 6.07e-01 | -0.016 [-0.041, +0.010] | -0.015 |
| metaqa | sub200 | audited@bge | 200 | +0.145 | 0.385 | 0.410 | -0.025 [-0.075, +0.025] | 4.42e-01 | -0.018 [-0.063, +0.025] | -0.015 |
| metaqa | sub200 | graph_centric | 200 | +0.085 | 0.350 | 0.360 | -0.010 [-0.060, +0.045] | 8.56e-01 | +0.019 [-0.019, +0.059] | +0.035 |
| metaqa | sub200 | graph_centric@bge | 200 | +0.260 | 0.320 | 0.315 | +0.005 [-0.050, +0.060] | 1.00e+00 | +0.086 [+0.039, +0.135] | +0.140 |
| metaqa | sub200 | hipporag2 | 200 | +0.180 | 0.330 | 0.280 | +0.050 [+0.000, +0.105] | 8.72e-02 | +0.082 [+0.037, +0.128] | +0.105 |
| metaqa | sub200 | ppr | 200 | +0.065 | 0.350 | 0.340 | +0.010 [-0.030, +0.050] | 8.04e-01 | +0.022 [-0.004, +0.050] | +0.025 |
| stark | holdout200 | audited | 200 | +0.025 | 0.230 | 0.240 | -0.010 [-0.050, +0.030] | 8.04e-01 | -0.003 [-0.038, +0.034] | +0.005 |
| stark | holdout200 | audited@bge | 200 | +0.015 | 0.195 | 0.190 | +0.005 [-0.035, +0.050] | 1.00e+00 | -0.007 [-0.046, +0.034] | +0.005 |
| stark | holdout200 | graph_centric | 200 | +0.065 | 0.225 | 0.195 | +0.030 [-0.015, +0.075] | 2.86e-01 | +0.027 [-0.009, +0.063] | +0.025 |
| stark | holdout200 | graph_centric@bge | 200 | +0.050 | 0.185 | 0.180 | +0.005 [-0.035, +0.045] | 1.00e+00 | +0.018 [-0.024, +0.061] | +0.005 |
| stark | holdout200 | ppr | 200 | +0.020 | 0.130 | 0.110 | +0.020 [-0.030, +0.070] | 5.41e-01 | +0.020 [-0.027, +0.067] | +0.020 |
| stark | sub200 | audited | 200 | -0.015 | 0.135 | 0.135 | +0.000 [-0.040, +0.040] | 1.00e+00 | +0.008 [-0.028, +0.043] | +0.010 |
| stark | sub200 | audited@bge | 200 | -0.010 | 0.155 | 0.185 | -0.030 [-0.070, +0.005] | 2.10e-01 | -0.032 [-0.070, +0.004] | -0.025 |
| stark | sub200 | graph_centric | 200 | +0.025 | 0.135 | 0.105 | +0.030 [-0.010, +0.070] | 2.10e-01 | +0.040 [-0.003, +0.084] | +0.030 |
| stark | sub200 | graph_centric@bge | 200 | +0.025 | 0.175 | 0.150 | +0.025 [-0.025, +0.075] | 4.24e-01 | +0.035 [-0.007, +0.077] | +0.030 |
| stark | sub200 | ppr | 200 | +0.025 | 0.060 | 0.060 | +0.000 [-0.030, +0.030] | 1.00e+00 | +0.017 [-0.020, +0.055] | +0.030 |

## Mediation: EM Δ conditional on retrieval flip status

| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |
|---|---|---|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | 33 | +0.424 | 157 | +0.025 |
| 2wiki | holdout200 | audited@bge | 24 | +0.375 | 168 | +0.018 |
| 2wiki | holdout200 | graph_centric | 68 | +0.382 | 131 | +0.061 |
| 2wiki | holdout200 | graph_centric@bge | 49 | +0.408 | 150 | +0.033 |
| 2wiki | holdout200 | hipporag2 | 60 | +0.350 | 135 | -0.059 |
| 2wiki | holdout200 | ppr | 73 | +0.534 | 116 | +0.172 |
| 2wiki | sub200 | audited | 37 | +0.351 | 158 | -0.019 |
| 2wiki | sub200 | audited@bge | 25 | +0.400 | 171 | +0.035 |
| 2wiki | sub200 | graph_centric | 75 | +0.427 | 121 | +0.008 |
| 2wiki | sub200 | graph_centric@bge | 63 | +0.492 | 133 | +0.015 |
| 2wiki | sub200 | hipporag2 | 58 | +0.414 | 132 | -0.068 |
| 2wiki | sub200 | ppr | 75 | +0.600 | 113 | +0.239 |
| graphragbench | holdout200 | audited | 13 | +0.000 | 186 | +0.000 |
| graphragbench | holdout200 | audited@bge | 20 | +0.000 | 180 | +0.000 |
| graphragbench | holdout200 | graph_centric | 58 | +0.000 | 140 | +0.000 |
| graphragbench | holdout200 | graph_centric@bge | 45 | +0.000 | 153 | +0.000 |
| graphragbench | holdout200 | hipporag2 | 48 | +0.000 | 149 | +0.000 |
| graphragbench | holdout200 | ppr | 87 | +0.000 | 102 | +0.000 |
| graphragbench | sub200 | audited | 13 | +0.000 | 187 | +0.000 |
| graphragbench | sub200 | audited@bge | 26 | +0.000 | 173 | +0.000 |
| graphragbench | sub200 | graph_centric | 58 | +0.000 | 142 | +0.000 |
| graphragbench | sub200 | graph_centric@bge | 50 | +0.000 | 149 | +0.000 |
| graphragbench | sub200 | hipporag2 | 53 | +0.000 | 147 | +0.000 |
| graphragbench | sub200 | ppr | 105 | +0.000 | 88 | +0.000 |
| metaqa | holdout200 | audited | 9 | -0.444 | 191 | +0.010 |
| metaqa | holdout200 | audited@bge | 33 | -0.515 | 161 | -0.019 |
| metaqa | holdout200 | graph_centric | 17 | -0.294 | 183 | +0.044 |
| metaqa | holdout200 | graph_centric@bge | 57 | -0.263 | 142 | +0.000 |
| metaqa | holdout200 | hipporag2 | 43 | +0.116 | 148 | -0.027 |
| metaqa | holdout200 | ppr | 14 | +0.143 | 186 | -0.027 |
| metaqa | sub200 | audited | 9 | -0.333 | 189 | +0.000 |
| metaqa | sub200 | audited@bge | 36 | -0.194 | 157 | -0.006 |
| metaqa | sub200 | graph_centric | 17 | -0.176 | 183 | +0.005 |
| metaqa | sub200 | graph_centric@bge | 52 | +0.038 | 148 | -0.007 |
| metaqa | sub200 | hipporag2 | 40 | +0.100 | 156 | +0.045 |
| metaqa | sub200 | ppr | 13 | -0.154 | 187 | +0.021 |
| stark | holdout200 | audited | 8 | +0.375 | 189 | -0.021 |
| stark | holdout200 | audited@bge | 8 | +0.500 | 187 | +0.005 |
| stark | holdout200 | graph_centric | 16 | +0.562 | 181 | -0.006 |
| stark | holdout200 | graph_centric@bge | 15 | +0.533 | 180 | -0.017 |
| stark | holdout200 | ppr | 12 | +0.333 | 180 | +0.028 |
| stark | sub200 | audited | 3 | -0.333 | 191 | +0.021 |
| stark | sub200 | audited@bge | 5 | +0.000 | 188 | -0.011 |
| stark | sub200 | graph_centric | 10 | +0.200 | 185 | +0.038 |
| stark | sub200 | graph_centric@bge | 9 | +0.333 | 187 | +0.021 |
| stark | sub200 | ppr | 7 | +0.143 | 191 | +0.005 |

## All cells

| bench | split | system | condition | n | hit@5 | EM | F1 | cover |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | bge_kg_openie | 200 | 0.255 | 0.170 | 0.441 | 0.180 |
| 2wiki | holdout200 | audited | bge_kg_silver | 200 | 0.335 | 0.210 | 0.457 | 0.230 |
| 2wiki | holdout200 | audited | kg_openie | 200 | 0.365 | 0.220 | 0.489 | 0.235 |
| 2wiki | holdout200 | audited | kg_openie_resorted | 200 | 0.365 | 0.220 | 0.489 | 0.235 |
| 2wiki | holdout200 | audited | kg_openie_svocab | 200 | 0.390 | 0.235 | 0.508 | 0.250 |
| 2wiki | holdout200 | audited | kg_openie_svocab_noprov | 200 | 0.345 | 0.230 | 0.500 | 0.245 |
| 2wiki | holdout200 | audited | kg_openie_svocab_q25 | 200 | 0.350 | 0.230 | 0.504 | 0.245 |
| 2wiki | holdout200 | audited | kg_openie_svocab_q50 | 200 | 0.375 | 0.230 | 0.507 | 0.245 |
| 2wiki | holdout200 | audited | kg_openie_svocab_q75 | 200 | 0.390 | 0.235 | 0.509 | 0.250 |
| 2wiki | holdout200 | audited | kg_silver | 200 | 0.480 | 0.285 | 0.535 | 0.300 |
| 2wiki | holdout200 | audited | kg_wikidata | 200 | 0.380 | 0.210 | 0.476 | 0.225 |
| 2wiki | holdout200 | bm25 | none | 200 | 0.355 | 0.185 | 0.438 | 0.195 |
| 2wiki | holdout200 | dense | bge_kg_silver | 200 | 0.125 | 0.115 | 0.367 | 0.125 |
| 2wiki | holdout200 | graph_centric | bge_kg_openie | 200 | 0.345 | 0.205 | 0.478 | 0.225 |
| 2wiki | holdout200 | graph_centric | bge_kg_silver | 200 | 0.585 | 0.330 | 0.561 | 0.355 |
| 2wiki | holdout200 | graph_centric | kg_openie | 200 | 0.370 | 0.230 | 0.496 | 0.250 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab | 200 | 0.395 | 0.215 | 0.492 | 0.240 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_noprov | 200 | 0.325 | 0.190 | 0.477 | 0.215 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_q25 | 200 | 0.330 | 0.185 | 0.467 | 0.210 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_q50 | 200 | 0.360 | 0.185 | 0.469 | 0.215 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_q75 | 200 | 0.375 | 0.215 | 0.493 | 0.240 |
| 2wiki | holdout200 | graph_centric | kg_silver | 200 | 0.705 | 0.400 | 0.596 | 0.410 |
| 2wiki | holdout200 | graph_centric | kg_wikidata | 200 | 0.390 | 0.225 | 0.470 | 0.240 |
| 2wiki | holdout200 | hipporag2 | kg_openie | 200 | 0.615 | 0.310 | 0.571 | 0.350 |
| 2wiki | holdout200 | hipporag2 | kg_silver | 200 | 0.890 | 0.370 | 0.603 | 0.415 |
| 2wiki | holdout200 | hipporag2 | kg_wikidata | 200 | 0.580 | 0.280 | 0.538 | 0.305 |
| 2wiki | holdout200 | none | closedbook | 200 | - | 0.055 | 0.292 | 0.075 |
| 2wiki | holdout200 | none | oracle | 200 | 1.000 | 0.685 | 0.834 | 0.750 |
| 2wiki | holdout200 | ppr | kg_openie | 200 | 0.430 | 0.255 | 0.502 | 0.275 |
| 2wiki | holdout200 | ppr | kg_openie_svocab | 200 | 0.510 | 0.300 | 0.541 | 0.315 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_noprov | 200 | 0.415 | 0.280 | 0.525 | 0.295 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_q25 | 200 | 0.445 | 0.285 | 0.528 | 0.300 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_q50 | 200 | 0.470 | 0.295 | 0.537 | 0.315 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_q75 | 200 | 0.475 | 0.300 | 0.535 | 0.320 |
| 2wiki | holdout200 | ppr | kg_silver | 200 | 0.740 | 0.555 | 0.744 | 0.620 |
| 2wiki | holdout200 | ppr | kg_wikidata | 200 | 0.535 | 0.335 | 0.545 | 0.355 |
| 2wiki | sub200 | audited | bge_kg_openie | 200 | 0.305 | 0.195 | 0.468 | 0.220 |
| 2wiki | sub200 | audited | bge_kg_silver | 200 | 0.410 | 0.270 | 0.516 | 0.290 |
| 2wiki | sub200 | audited | dose_f00 | 200 | 0.505 | 0.240 | 0.528 | 0.290 |
| 2wiki | sub200 | audited | dose_f100 | 200 | 0.535 | 0.265 | 0.537 | 0.315 |
| 2wiki | sub200 | audited | dose_f25 | 200 | 0.510 | 0.250 | 0.530 | 0.295 |
| 2wiki | sub200 | audited | dose_f50 | 200 | 0.525 | 0.255 | 0.536 | 0.300 |
| 2wiki | sub200 | audited | dose_f75 | 200 | 0.530 | 0.265 | 0.539 | 0.310 |
| 2wiki | sub200 | audited | kg_openie | 200 | 0.375 | 0.220 | 0.486 | 0.255 |
| 2wiki | sub200 | audited | kg_openie_resorted | 200 | 0.375 | 0.220 | 0.486 | 0.255 |
| 2wiki | sub200 | audited | kg_openie_svocab | 200 | 0.435 | 0.255 | 0.528 | 0.305 |
| 2wiki | sub200 | audited | kg_openie_svocab_noprov | 200 | 0.395 | 0.260 | 0.528 | 0.310 |
| 2wiki | sub200 | audited | kg_openie_svocab_q25 | 200 | 0.405 | 0.260 | 0.522 | 0.305 |
| 2wiki | sub200 | audited | kg_openie_svocab_q50 | 200 | 0.420 | 0.265 | 0.529 | 0.310 |
| 2wiki | sub200 | audited | kg_openie_svocab_q75 | 200 | 0.435 | 0.265 | 0.533 | 0.310 |
| 2wiki | sub200 | audited | kg_silver | 200 | 0.535 | 0.265 | 0.537 | 0.315 |
| 2wiki | sub200 | audited | kg_wikidata | 200 | 0.330 | 0.195 | 0.486 | 0.225 |
| 2wiki | sub200 | bm25 | none | 200 | 0.365 | 0.200 | 0.460 | 0.225 |
| 2wiki | sub200 | dense | bge_kg_silver | 200 | 0.130 | 0.120 | 0.405 | 0.130 |
| 2wiki | sub200 | graph_centric | bge_kg_openie | 200 | 0.345 | 0.230 | 0.498 | 0.250 |
| 2wiki | sub200 | graph_centric | bge_kg_silver | 200 | 0.640 | 0.395 | 0.606 | 0.430 |
| 2wiki | sub200 | graph_centric | kg_openie | 200 | 0.395 | 0.200 | 0.484 | 0.235 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab | 200 | 0.420 | 0.205 | 0.478 | 0.230 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_noprov | 200 | 0.360 | 0.170 | 0.452 | 0.205 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_q25 | 200 | 0.390 | 0.200 | 0.473 | 0.235 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_q50 | 200 | 0.390 | 0.200 | 0.478 | 0.235 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_q75 | 200 | 0.420 | 0.190 | 0.474 | 0.220 |
| 2wiki | sub200 | graph_centric | kg_silver | 200 | 0.750 | 0.365 | 0.595 | 0.405 |
| 2wiki | sub200 | graph_centric | kg_wikidata | 200 | 0.380 | 0.205 | 0.493 | 0.235 |
| 2wiki | sub200 | hipporag2 | kg_openie | 200 | 0.645 | 0.320 | 0.598 | 0.375 |
| 2wiki | sub200 | hipporag2 | kg_silver | 200 | 0.885 | 0.375 | 0.614 | 0.410 |
| 2wiki | sub200 | hipporag2 | kg_wikidata | 200 | 0.560 | 0.285 | 0.551 | 0.335 |
| 2wiki | sub200 | none | closedbook | 200 | - | 0.040 | 0.280 | 0.045 |
| 2wiki | sub200 | none | oracle | 200 | 1.000 | 0.725 | 0.864 | 0.770 |
| 2wiki | sub200 | ppr | kg_openie | 200 | 0.425 | 0.215 | 0.506 | 0.260 |
| 2wiki | sub200 | ppr | kg_openie_svocab | 200 | 0.510 | 0.265 | 0.543 | 0.305 |
| 2wiki | sub200 | ppr | kg_openie_svocab_noprov | 200 | 0.400 | 0.205 | 0.477 | 0.245 |
| 2wiki | sub200 | ppr | kg_openie_svocab_q25 | 200 | 0.440 | 0.245 | 0.504 | 0.285 |
| 2wiki | sub200 | ppr | kg_openie_svocab_q50 | 200 | 0.460 | 0.245 | 0.514 | 0.280 |
| 2wiki | sub200 | ppr | kg_openie_svocab_q75 | 200 | 0.495 | 0.255 | 0.532 | 0.290 |
| 2wiki | sub200 | ppr | kg_silver | 200 | 0.740 | 0.560 | 0.764 | 0.600 |
| 2wiki | sub200 | ppr | kg_wikidata | 200 | 0.500 | 0.275 | 0.543 | 0.310 |
| graphragbench | holdout200 | audited | bge_kg_openie | 200 | 0.530 | 0.000 | 0.302 | 0.000 |
| graphragbench | holdout200 | audited | bge_kg_silver | 200 | 0.630 | 0.000 | 0.311 | 0.000 |
| graphragbench | holdout200 | audited | kg_openie | 200 | 0.520 | 0.000 | 0.289 | 0.000 |
| graphragbench | holdout200 | audited | kg_silver | 200 | 0.580 | 0.000 | 0.292 | 0.000 |
| graphragbench | holdout200 | bm25 | none | 200 | 0.505 | 0.000 | 0.279 | 0.000 |
| graphragbench | holdout200 | dense | bge_kg_silver | 200 | 0.460 | 0.000 | 0.271 | 0.000 |
| graphragbench | holdout200 | graph_centric | bge_kg_openie | 200 | 0.500 | 0.000 | 0.280 | 0.000 |
| graphragbench | holdout200 | graph_centric | bge_kg_silver | 200 | 0.715 | 0.000 | 0.307 | 0.000 |
| graphragbench | holdout200 | graph_centric | kg_openie | 200 | 0.400 | 0.000 | 0.264 | 0.000 |
| graphragbench | holdout200 | graph_centric | kg_silver | 200 | 0.680 | 0.000 | 0.301 | 0.000 |
| graphragbench | holdout200 | hipporag2 | kg_openie | 200 | 0.575 | 0.000 | 0.301 | 0.000 |
| graphragbench | holdout200 | hipporag2 | kg_silver | 200 | 0.800 | 0.000 | 0.322 | 0.000 |
| graphragbench | holdout200 | none | closedbook | 200 | - | 0.000 | 0.150 | 0.000 |
| graphragbench | holdout200 | none | oracle | 200 | 1.000 | 0.000 | 0.294 | 0.000 |
| graphragbench | holdout200 | ppr | kg_openie | 200 | 0.245 | 0.000 | 0.207 | 0.000 |
| graphragbench | holdout200 | ppr | kg_silver | 200 | 0.625 | 0.000 | 0.251 | 0.000 |
| graphragbench | sub200 | audited | bge_kg_openie | 200 | 0.480 | 0.000 | 0.256 | 0.000 |
| graphragbench | sub200 | audited | bge_kg_silver | 200 | 0.605 | 0.000 | 0.279 | 0.000 |
| graphragbench | sub200 | audited | kg_openie | 200 | 0.500 | 0.000 | 0.261 | 0.000 |
| graphragbench | sub200 | audited | kg_silver | 200 | 0.565 | 0.000 | 0.271 | 0.000 |
| graphragbench | sub200 | bm25 | none | 200 | 0.520 | 0.000 | 0.258 | 0.000 |
| graphragbench | sub200 | dense | bge_kg_silver | 200 | 0.395 | 0.000 | 0.263 | 0.000 |
| graphragbench | sub200 | graph_centric | bge_kg_openie | 200 | 0.445 | 0.000 | 0.260 | 0.000 |
| graphragbench | sub200 | graph_centric | bge_kg_silver | 200 | 0.690 | 0.000 | 0.273 | 0.000 |
| graphragbench | sub200 | graph_centric | kg_openie | 200 | 0.405 | 0.000 | 0.257 | 0.000 |
| graphragbench | sub200 | graph_centric | kg_silver | 200 | 0.695 | 0.000 | 0.263 | 0.000 |
| graphragbench | sub200 | hipporag2 | kg_openie | 200 | 0.540 | 0.000 | 0.270 | 0.000 |
| graphragbench | sub200 | hipporag2 | kg_silver | 200 | 0.805 | 0.000 | 0.282 | 0.000 |
| graphragbench | sub200 | none | closedbook | 200 | - | 0.000 | 0.140 | 0.000 |
| graphragbench | sub200 | none | oracle | 200 | 1.000 | 0.000 | 0.275 | 0.000 |
| graphragbench | sub200 | ppr | kg_openie | 200 | 0.200 | 0.000 | 0.195 | 0.000 |
| graphragbench | sub200 | ppr | kg_silver | 200 | 0.690 | 0.000 | 0.244 | 0.000 |
| metaqa | holdout200 | audited | bge_kg_openie | 200 | 0.540 | 0.415 | 0.653 | 0.850 |
| metaqa | holdout200 | audited | bge_kg_silver | 200 | 0.675 | 0.320 | 0.612 | 0.830 |
| metaqa | holdout200 | audited | kg_fullkb | 200 | 0.805 | 0.330 | 0.625 | 0.855 |
| metaqa | holdout200 | audited | kg_openie | 200 | 0.875 | 0.300 | 0.651 | 0.920 |
| metaqa | holdout200 | audited | kg_silver | 200 | 0.920 | 0.290 | 0.658 | 0.930 |
| metaqa | holdout200 | bm25 | none | 200 | 0.855 | 0.295 | 0.644 | 0.905 |
| metaqa | holdout200 | dense | bge_kg_silver | 200 | 0.035 | 0.150 | 0.205 | 0.255 |
| metaqa | holdout200 | graph_centric | bge_kg_openie | 200 | 0.645 | 0.365 | 0.634 | 0.840 |
| metaqa | holdout200 | graph_centric | bge_kg_silver | 200 | 0.925 | 0.290 | 0.658 | 0.930 |
| metaqa | holdout200 | graph_centric | kg_fullkb | 200 | 0.845 | 0.320 | 0.656 | 0.895 |
| metaqa | holdout200 | graph_centric | kg_openie | 200 | 0.860 | 0.285 | 0.623 | 0.900 |
| metaqa | holdout200 | graph_centric | kg_silver | 200 | 0.945 | 0.300 | 0.651 | 0.925 |
| metaqa | holdout200 | hipporag2 | kg_fullkb | 200 | 0.590 | 0.310 | 0.583 | 0.785 |
| metaqa | holdout200 | hipporag2 | kg_openie | 200 | 0.745 | 0.260 | 0.569 | 0.800 |
| metaqa | holdout200 | hipporag2 | kg_silver | 200 | 0.915 | 0.270 | 0.627 | 0.910 |
| metaqa | holdout200 | none | closedbook | 200 | - | 0.175 | 0.250 | 0.305 |
| metaqa | holdout200 | none | oracle | 200 | 1.000 | 0.295 | 0.667 | 0.945 |
| metaqa | holdout200 | ppr | kg_fullkb | 200 | 1.000 | 0.295 | 0.663 | 0.930 |
| metaqa | holdout200 | ppr | kg_openie | 200 | 0.930 | 0.295 | 0.659 | 0.925 |
| metaqa | holdout200 | ppr | kg_silver | 200 | 1.000 | 0.280 | 0.663 | 0.950 |
| metaqa | sub200 | audited | bge_kg_openie | 200 | 0.505 | 0.410 | 0.633 | 0.815 |
| metaqa | sub200 | audited | bge_kg_silver | 200 | 0.650 | 0.385 | 0.615 | 0.800 |
| metaqa | sub200 | audited | kg_fullkb | 200 | 0.820 | 0.365 | 0.657 | 0.890 |
| metaqa | sub200 | audited | kg_openie | 200 | 0.900 | 0.350 | 0.667 | 0.910 |
| metaqa | sub200 | audited | kg_silver | 200 | 0.935 | 0.335 | 0.651 | 0.895 |
| metaqa | sub200 | bm25 | none | 200 | 0.870 | 0.380 | 0.665 | 0.895 |
| metaqa | sub200 | dense | bge_kg_silver | 200 | 0.060 | 0.160 | 0.206 | 0.235 |
| metaqa | sub200 | graph_centric | bge_kg_openie | 200 | 0.675 | 0.315 | 0.572 | 0.780 |
| metaqa | sub200 | graph_centric | bge_kg_silver | 200 | 0.935 | 0.320 | 0.658 | 0.920 |
| metaqa | sub200 | graph_centric | kg_fullkb | 200 | 0.845 | 0.340 | 0.649 | 0.895 |
| metaqa | sub200 | graph_centric | kg_openie | 200 | 0.865 | 0.360 | 0.646 | 0.885 |
| metaqa | sub200 | graph_centric | kg_silver | 200 | 0.950 | 0.350 | 0.665 | 0.920 |
| metaqa | sub200 | hipporag2 | kg_fullkb | 200 | 0.695 | 0.330 | 0.602 | 0.805 |
| metaqa | sub200 | hipporag2 | kg_openie | 200 | 0.755 | 0.280 | 0.575 | 0.795 |
| metaqa | sub200 | hipporag2 | kg_silver | 200 | 0.935 | 0.330 | 0.657 | 0.900 |
| metaqa | sub200 | none | closedbook | 200 | - | 0.145 | 0.234 | 0.300 |
| metaqa | sub200 | none | oracle | 200 | 1.000 | 0.340 | 0.690 | 0.955 |
| metaqa | sub200 | ppr | kg_fullkb | 200 | 0.995 | 0.315 | 0.653 | 0.925 |
| metaqa | sub200 | ppr | kg_openie | 200 | 0.935 | 0.340 | 0.678 | 0.940 |
| metaqa | sub200 | ppr | kg_silver | 200 | 1.000 | 0.350 | 0.700 | 0.965 |
| stark | holdout200 | audited | bge_kg_openie | 200 | 0.240 | 0.190 | 0.269 | 0.235 |
| stark | holdout200 | audited | bge_kg_silver | 200 | 0.255 | 0.195 | 0.262 | 0.240 |
| stark | holdout200 | audited | kg_openie | 200 | 0.250 | 0.240 | 0.294 | 0.265 |
| stark | holdout200 | audited | kg_silver | 200 | 0.275 | 0.230 | 0.291 | 0.270 |
| stark | holdout200 | bm25 | none | 200 | 0.215 | 0.185 | 0.240 | 0.220 |
| stark | holdout200 | dense | bge_kg_silver | 200 | 0.185 | 0.180 | 0.253 | 0.230 |
| stark | holdout200 | graph_centric | bge_kg_openie | 200 | 0.200 | 0.180 | 0.240 | 0.220 |
| stark | holdout200 | graph_centric | bge_kg_silver | 200 | 0.250 | 0.185 | 0.258 | 0.225 |
| stark | holdout200 | graph_centric | kg_openie | 200 | 0.205 | 0.195 | 0.262 | 0.240 |
| stark | holdout200 | graph_centric | kg_silver | 200 | 0.270 | 0.225 | 0.289 | 0.265 |
| stark | holdout200 | none | closedbook | 200 | - | 0.035 | 0.084 | 0.050 |
| stark | holdout200 | none | oracle | 200 | 1.000 | 0.805 | 0.869 | 0.885 |
| stark | holdout200 | ppr | kg_openie | 200 | 0.100 | 0.110 | 0.159 | 0.135 |
| stark | holdout200 | ppr | kg_silver | 200 | 0.120 | 0.130 | 0.179 | 0.155 |
| stark | sub200 | audited | bge_kg_openie | 200 | 0.170 | 0.185 | 0.273 | 0.235 |
| stark | sub200 | audited | bge_kg_silver | 200 | 0.160 | 0.155 | 0.241 | 0.210 |
| stark | sub200 | audited | kg_openie | 200 | 0.145 | 0.135 | 0.211 | 0.160 |
| stark | sub200 | audited | kg_silver | 200 | 0.130 | 0.135 | 0.219 | 0.170 |
| stark | sub200 | bm25 | none | 200 | 0.135 | 0.115 | 0.193 | 0.140 |
| stark | sub200 | dense | bge_kg_silver | 200 | 0.125 | 0.165 | 0.260 | 0.225 |
| stark | sub200 | graph_centric | bge_kg_openie | 200 | 0.155 | 0.150 | 0.220 | 0.185 |
| stark | sub200 | graph_centric | bge_kg_silver | 200 | 0.180 | 0.175 | 0.256 | 0.215 |
| stark | sub200 | graph_centric | kg_openie | 200 | 0.100 | 0.105 | 0.176 | 0.140 |
| stark | sub200 | graph_centric | kg_silver | 200 | 0.125 | 0.135 | 0.215 | 0.170 |
| stark | sub200 | none | closedbook | 200 | - | 0.015 | 0.054 | 0.025 |
| stark | sub200 | none | oracle | 200 | 1.000 | 0.755 | 0.856 | 0.845 |
| stark | sub200 | ppr | kg_openie | 200 | 0.040 | 0.060 | 0.097 | 0.065 |
| stark | sub200 | ppr | kg_silver | 200 | 0.065 | 0.060 | 0.114 | 0.095 |
