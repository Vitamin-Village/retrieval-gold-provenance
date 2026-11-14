# End-to-end QA over frozen swap vectors: reader `llama3.1:8b`

## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader

| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| 2wiki | holdout200 | audited | 200 | +0.115 | 0.215 | 0.155 | +0.060 [+0.015, +0.105] | 2.27e-02 | +0.047 [+0.006, +0.089] | +0.050 |
| 2wiki | holdout200 | audited@bge | 200 | +0.150 | 0.210 | 0.185 | +0.025 [-0.035, +0.080] | 4.87e-01 | +0.033 [-0.012, +0.077] | +0.035 |
| 2wiki | holdout200 | graph_centric | 200 | +0.335 | 0.265 | 0.165 | +0.100 [+0.040, +0.160] | 2.22e-03 | +0.074 [+0.024, +0.126] | +0.110 |
| 2wiki | holdout200 | graph_centric@bge | 200 | +0.250 | 0.245 | 0.220 | +0.025 [-0.035, +0.085] | 5.11e-01 | -0.004 [-0.051, +0.045] | +0.025 |
| 2wiki | holdout200 | hipporag2 | 200 | +0.275 | 0.285 | 0.265 | +0.020 [-0.045, +0.085] | 6.65e-01 | +0.001 [-0.056, +0.056] | +0.020 |
| 2wiki | holdout200 | ppr | 200 | +0.310 | 0.490 | 0.230 | +0.260 [+0.180, +0.340] | 5.26e-09 | +0.224 [+0.155, +0.293] | +0.290 |
| 2wiki | sub200 | audited | 200 | +0.160 | 0.180 | 0.130 | +0.050 [+0.000, +0.100] | 7.55e-02 | +0.046 [+0.005, +0.085] | +0.060 |
| 2wiki | sub200 | audited@bge | 200 | +0.195 | 0.200 | 0.145 | +0.055 [+0.000, +0.110] | 8.01e-02 | +0.037 [-0.002, +0.076] | +0.060 |
| 2wiki | sub200 | graph_centric | 200 | +0.355 | 0.240 | 0.100 | +0.140 [+0.080, +0.200] | 8.36e-06 | +0.108 [+0.062, +0.155] | +0.140 |
| 2wiki | sub200 | graph_centric@bge | 200 | +0.270 | 0.230 | 0.165 | +0.065 [+0.010, +0.120] | 3.51e-02 | +0.071 [+0.024, +0.117] | +0.080 |
| 2wiki | sub200 | hipporag2 | 200 | +0.240 | 0.255 | 0.180 | +0.075 [+0.005, +0.145] | 4.89e-02 | +0.059 [+0.004, +0.113] | +0.075 |
| 2wiki | sub200 | ppr | 200 | +0.315 | 0.475 | 0.145 | +0.330 [+0.260, +0.400] | 1.29e-16 | +0.268 [+0.214, +0.323] | +0.365 |
| graphragbench | holdout200 | audited | 200 | +0.060 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.004 [-0.010, +0.020] | +0.000 |
| graphragbench | holdout200 | audited@bge | 200 | +0.105 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.010 [-0.009, +0.029] | +0.000 |
| graphragbench | holdout200 | graph_centric | 200 | +0.280 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.035 [+0.011, +0.061] | +0.000 |
| graphragbench | holdout200 | graph_centric@bge | 200 | +0.250 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.026 [-0.000, +0.054] | +0.000 |
| graphragbench | holdout200 | hipporag2 | 200 | +0.225 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.001 [-0.019, +0.021] | +0.000 |
| graphragbench | holdout200 | ppr | 200 | +0.380 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.056 [+0.023, +0.089] | +0.000 |
| graphragbench | sub200 | audited | 200 | +0.065 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | -0.005 [-0.024, +0.015] | +0.000 |
| graphragbench | sub200 | audited@bge | 200 | +0.090 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.009 [-0.011, +0.029] | +0.000 |
| graphragbench | sub200 | graph_centric | 200 | +0.290 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.006 [-0.018, +0.031] | +0.000 |
| graphragbench | sub200 | graph_centric@bge | 200 | +0.245 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.003 [-0.022, +0.028] | +0.000 |
| graphragbench | sub200 | hipporag2 | 200 | +0.265 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.013 [-0.011, +0.037] | +0.000 |
| graphragbench | sub200 | ppr | 200 | +0.490 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.00e+00 | +0.051 [+0.016, +0.085] | +0.000 |
| metaqa | holdout200 | audited | 200 | +0.045 | 0.435 | 0.450 | -0.015 [-0.050, +0.020] | 5.81e-01 | -0.013 [-0.037, +0.011] | +0.000 |
| metaqa | holdout200 | audited@bge | 200 | +0.080 | 0.410 | 0.400 | +0.010 [-0.025, +0.045] | 7.74e-01 | +0.025 [-0.005, +0.056] | +0.030 |
| metaqa | holdout200 | graph_centric | 200 | +0.085 | 0.450 | 0.425 | +0.025 [-0.020, +0.075] | 4.05e-01 | +0.010 [-0.029, +0.050] | +0.010 |
| metaqa | holdout200 | graph_centric@bge | 200 | +0.225 | 0.415 | 0.435 | -0.020 [-0.080, +0.040] | 6.18e-01 | +0.033 [-0.014, +0.081] | +0.060 |
| metaqa | holdout200 | hipporag2 | 200 | +0.170 | 0.445 | 0.385 | +0.060 [+0.005, +0.115] | 5.76e-02 | +0.097 [+0.046, +0.150] | +0.110 |
| metaqa | holdout200 | ppr | 200 | +0.070 | 0.495 | 0.485 | +0.010 [-0.040, +0.055] | 8.39e-01 | +0.047 [+0.010, +0.087] | +0.070 |
| metaqa | sub200 | audited | 200 | +0.035 | 0.470 | 0.500 | -0.030 [-0.065, +0.000] | 1.46e-01 | -0.016 [-0.041, +0.009] | -0.020 |
| metaqa | sub200 | audited@bge | 200 | +0.075 | 0.445 | 0.460 | -0.015 [-0.055, +0.025] | 6.29e-01 | -0.001 [-0.030, +0.029] | +0.005 |
| metaqa | sub200 | graph_centric | 200 | +0.085 | 0.465 | 0.445 | +0.020 [-0.035, +0.075] | 5.85e-01 | +0.015 [-0.027, +0.059] | +0.000 |
| metaqa | sub200 | graph_centric@bge | 200 | +0.180 | 0.450 | 0.430 | +0.020 [-0.035, +0.070] | 5.72e-01 | +0.041 [-0.001, +0.083] | +0.040 |
| metaqa | sub200 | hipporag2 | 200 | +0.180 | 0.450 | 0.395 | +0.055 [-0.005, +0.115] | 9.89e-02 | +0.080 [+0.030, +0.132] | +0.085 |
| metaqa | sub200 | ppr | 200 | +0.065 | 0.515 | 0.455 | +0.060 [+0.010, +0.110] | 3.57e-02 | +0.057 [+0.017, +0.097] | +0.025 |
| stark | holdout200 | audited | 200 | +0.025 | 0.175 | 0.190 | -0.015 [-0.045, +0.015] | 5.08e-01 | +0.014 [-0.016, +0.043] | -0.010 |
| stark | holdout200 | audited@bge | 200 | +0.035 | 0.180 | 0.190 | -0.010 [-0.045, +0.025] | 7.74e-01 | -0.001 [-0.035, +0.033] | -0.005 |
| stark | holdout200 | graph_centric | 200 | +0.065 | 0.170 | 0.160 | +0.010 [-0.025, +0.045] | 7.74e-01 | +0.023 [-0.010, +0.057] | +0.020 |
| stark | holdout200 | graph_centric@bge | 200 | +0.055 | 0.165 | 0.155 | +0.010 [-0.025, +0.045] | 7.74e-01 | +0.021 [-0.010, +0.053] | +0.020 |
| stark | holdout200 | ppr | 200 | +0.020 | 0.105 | 0.085 | +0.020 [-0.030, +0.065] | 5.41e-01 | +0.006 [-0.041, +0.051] | -0.005 |
| stark | sub200 | audited | 200 | -0.015 | 0.080 | 0.110 | -0.030 [-0.060, +0.000] | 1.09e-01 | -0.025 [-0.063, +0.012] | -0.040 |
| stark | sub200 | audited@bge | 200 | -0.030 | 0.080 | 0.120 | -0.040 [-0.075, -0.005] | 5.74e-02 | -0.017 [-0.056, +0.019] | -0.040 |
| stark | sub200 | graph_centric | 200 | +0.025 | 0.065 | 0.075 | -0.010 [-0.040, +0.015] | 7.27e-01 | +0.012 [-0.020, +0.046] | -0.010 |
| stark | sub200 | graph_centric@bge | 200 | +0.020 | 0.110 | 0.105 | +0.005 [-0.035, +0.045] | 1.00e+00 | +0.009 [-0.029, +0.047] | -0.005 |
| stark | sub200 | ppr | 200 | +0.025 | 0.050 | 0.050 | +0.000 [-0.035, +0.040] | 1.00e+00 | +0.016 [-0.024, +0.057] | +0.020 |

## Mediation: EM Δ conditional on retrieval flip status

| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |
|---|---|---|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | 33 | +0.303 | 157 | +0.038 |
| 2wiki | holdout200 | audited@bge | 40 | +0.200 | 150 | +0.013 |
| 2wiki | holdout200 | graph_centric | 68 | +0.265 | 131 | +0.008 |
| 2wiki | holdout200 | graph_centric@bge | 52 | +0.192 | 146 | -0.034 |
| 2wiki | holdout200 | hipporag2 | 60 | +0.267 | 135 | -0.081 |
| 2wiki | holdout200 | ppr | 73 | +0.479 | 116 | +0.121 |
| 2wiki | sub200 | audited | 37 | +0.243 | 158 | +0.019 |
| 2wiki | sub200 | audited@bge | 43 | +0.279 | 153 | +0.000 |
| 2wiki | sub200 | graph_centric | 75 | +0.293 | 121 | +0.033 |
| 2wiki | sub200 | graph_centric@bge | 57 | +0.316 | 140 | -0.029 |
| 2wiki | sub200 | hipporag2 | 58 | +0.328 | 132 | -0.015 |
| 2wiki | sub200 | ppr | 75 | +0.427 | 113 | +0.257 |
| graphragbench | holdout200 | audited | 13 | +0.000 | 186 | +0.000 |
| graphragbench | holdout200 | audited@bge | 21 | +0.000 | 179 | +0.000 |
| graphragbench | holdout200 | graph_centric | 58 | +0.000 | 140 | +0.000 |
| graphragbench | holdout200 | graph_centric@bge | 53 | +0.000 | 144 | +0.000 |
| graphragbench | holdout200 | hipporag2 | 48 | +0.000 | 149 | +0.000 |
| graphragbench | holdout200 | ppr | 87 | +0.000 | 102 | +0.000 |
| graphragbench | sub200 | audited | 13 | +0.000 | 187 | +0.000 |
| graphragbench | sub200 | audited@bge | 20 | +0.000 | 178 | +0.000 |
| graphragbench | sub200 | graph_centric | 58 | +0.000 | 142 | +0.000 |
| graphragbench | sub200 | graph_centric@bge | 51 | +0.000 | 147 | +0.000 |
| graphragbench | sub200 | hipporag2 | 53 | +0.000 | 147 | +0.000 |
| graphragbench | sub200 | ppr | 105 | +0.000 | 88 | +0.000 |
| metaqa | holdout200 | audited | 9 | -0.222 | 191 | -0.005 |
| metaqa | holdout200 | audited@bge | 17 | +0.000 | 182 | +0.011 |
| metaqa | holdout200 | graph_centric | 17 | -0.176 | 183 | +0.044 |
| metaqa | holdout200 | graph_centric@bge | 45 | -0.111 | 155 | +0.006 |
| metaqa | holdout200 | hipporag2 | 43 | +0.186 | 148 | +0.041 |
| metaqa | holdout200 | ppr | 14 | +0.214 | 186 | -0.005 |
| metaqa | sub200 | audited | 9 | -0.333 | 189 | -0.016 |
| metaqa | sub200 | audited@bge | 15 | -0.200 | 185 | +0.000 |
| metaqa | sub200 | graph_centric | 17 | -0.176 | 183 | +0.038 |
| metaqa | sub200 | graph_centric@bge | 36 | +0.056 | 164 | +0.012 |
| metaqa | sub200 | hipporag2 | 40 | +0.250 | 156 | +0.019 |
| metaqa | sub200 | ppr | 13 | +0.154 | 187 | +0.053 |
| stark | holdout200 | audited | 8 | +0.125 | 189 | -0.021 |
| stark | holdout200 | audited@bge | 8 | +0.125 | 191 | -0.010 |
| stark | holdout200 | graph_centric | 16 | +0.250 | 181 | +0.000 |
| stark | holdout200 | graph_centric@bge | 12 | +0.250 | 187 | +0.000 |
| stark | holdout200 | ppr | 12 | +0.583 | 180 | +0.017 |
| stark | sub200 | audited | 3 | +0.000 | 191 | -0.010 |
| stark | sub200 | audited@bge | 1 | +0.000 | 192 | -0.026 |
| stark | sub200 | graph_centric | 10 | -0.100 | 185 | +0.005 |
| stark | sub200 | graph_centric@bge | 6 | +0.167 | 192 | +0.005 |
| stark | sub200 | ppr | 7 | +0.429 | 191 | -0.005 |

## All cells

| bench | split | system | condition | n | hit@5 | EM | F1 | cover |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | bge_kg_openie | 200 | 0.400 | 0.185 | 0.416 | 0.190 |
| 2wiki | holdout200 | audited | bge_kg_silver | 200 | 0.550 | 0.210 | 0.449 | 0.225 |
| 2wiki | holdout200 | audited | kg_openie | 200 | 0.365 | 0.155 | 0.413 | 0.170 |
| 2wiki | holdout200 | audited | kg_openie_resorted | 200 | 0.365 | 0.155 | 0.413 | 0.170 |
| 2wiki | holdout200 | audited | kg_openie_svocab | 200 | 0.390 | 0.185 | 0.442 | 0.195 |
| 2wiki | holdout200 | audited | kg_openie_svocab_noprov | 200 | 0.345 | 0.175 | 0.433 | 0.185 |
| 2wiki | holdout200 | audited | kg_openie_svocab_q25 | 200 | 0.350 | 0.180 | 0.442 | 0.190 |
| 2wiki | holdout200 | audited | kg_openie_svocab_q50 | 200 | 0.375 | 0.185 | 0.442 | 0.195 |
| 2wiki | holdout200 | audited | kg_openie_svocab_q75 | 200 | 0.390 | 0.185 | 0.443 | 0.195 |
| 2wiki | holdout200 | audited | kg_silver | 200 | 0.480 | 0.215 | 0.460 | 0.220 |
| 2wiki | holdout200 | audited | kg_wikidata | 200 | 0.380 | 0.180 | 0.450 | 0.200 |
| 2wiki | holdout200 | bm25 | none | 200 | 0.355 | 0.145 | 0.387 | 0.150 |
| 2wiki | holdout200 | dense | bge_kg_silver | 200 | 0.450 | 0.165 | 0.434 | 0.175 |
| 2wiki | holdout200 | graph_centric | bge_kg_openie | 200 | 0.470 | 0.220 | 0.465 | 0.240 |
| 2wiki | holdout200 | graph_centric | bge_kg_silver | 200 | 0.720 | 0.245 | 0.462 | 0.265 |
| 2wiki | holdout200 | graph_centric | kg_openie | 200 | 0.370 | 0.165 | 0.422 | 0.175 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab | 200 | 0.395 | 0.175 | 0.418 | 0.190 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_noprov | 200 | 0.325 | 0.135 | 0.392 | 0.150 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_q25 | 200 | 0.330 | 0.150 | 0.405 | 0.165 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_q50 | 200 | 0.360 | 0.160 | 0.411 | 0.175 |
| 2wiki | holdout200 | graph_centric | kg_openie_svocab_q75 | 200 | 0.375 | 0.160 | 0.406 | 0.175 |
| 2wiki | holdout200 | graph_centric | kg_silver | 200 | 0.705 | 0.265 | 0.496 | 0.285 |
| 2wiki | holdout200 | graph_centric | kg_wikidata | 200 | 0.390 | 0.155 | 0.400 | 0.170 |
| 2wiki | holdout200 | hipporag2 | kg_openie | 200 | 0.615 | 0.265 | 0.513 | 0.285 |
| 2wiki | holdout200 | hipporag2 | kg_silver | 200 | 0.890 | 0.285 | 0.515 | 0.305 |
| 2wiki | holdout200 | hipporag2 | kg_wikidata | 200 | 0.580 | 0.205 | 0.450 | 0.205 |
| 2wiki | holdout200 | none | closedbook | 200 | - | 0.035 | 0.301 | 0.065 |
| 2wiki | holdout200 | none | oracle | 200 | 1.000 | 0.610 | 0.759 | 0.645 |
| 2wiki | holdout200 | ppr | kg_openie | 200 | 0.430 | 0.230 | 0.448 | 0.240 |
| 2wiki | holdout200 | ppr | kg_openie_svocab | 200 | 0.510 | 0.245 | 0.475 | 0.250 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_noprov | 200 | 0.415 | 0.205 | 0.433 | 0.210 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_q25 | 200 | 0.445 | 0.230 | 0.460 | 0.235 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_q50 | 200 | 0.470 | 0.255 | 0.475 | 0.260 |
| 2wiki | holdout200 | ppr | kg_openie_svocab_q75 | 200 | 0.475 | 0.245 | 0.471 | 0.250 |
| 2wiki | holdout200 | ppr | kg_silver | 200 | 0.740 | 0.490 | 0.672 | 0.530 |
| 2wiki | holdout200 | ppr | kg_wikidata | 200 | 0.535 | 0.255 | 0.488 | 0.270 |
| 2wiki | sub200 | audited | bge_kg_openie | 200 | 0.430 | 0.145 | 0.449 | 0.175 |
| 2wiki | sub200 | audited | bge_kg_silver | 200 | 0.625 | 0.200 | 0.485 | 0.235 |
| 2wiki | sub200 | audited | dose_f00 | 200 | 0.505 | 0.160 | 0.452 | 0.200 |
| 2wiki | sub200 | audited | dose_f100 | 200 | 0.535 | 0.180 | 0.463 | 0.220 |
| 2wiki | sub200 | audited | dose_f25 | 200 | 0.510 | 0.175 | 0.458 | 0.215 |
| 2wiki | sub200 | audited | dose_f50 | 200 | 0.525 | 0.180 | 0.465 | 0.220 |
| 2wiki | sub200 | audited | dose_f75 | 200 | 0.530 | 0.180 | 0.465 | 0.220 |
| 2wiki | sub200 | audited | kg_openie | 200 | 0.375 | 0.130 | 0.418 | 0.160 |
| 2wiki | sub200 | audited | kg_openie_resorted | 200 | 0.375 | 0.130 | 0.418 | 0.160 |
| 2wiki | sub200 | audited | kg_openie_svocab | 200 | 0.435 | 0.140 | 0.432 | 0.175 |
| 2wiki | sub200 | audited | kg_openie_svocab_noprov | 200 | 0.395 | 0.135 | 0.422 | 0.165 |
| 2wiki | sub200 | audited | kg_openie_svocab_q25 | 200 | 0.405 | 0.140 | 0.427 | 0.170 |
| 2wiki | sub200 | audited | kg_openie_svocab_q50 | 200 | 0.420 | 0.145 | 0.433 | 0.175 |
| 2wiki | sub200 | audited | kg_openie_svocab_q75 | 200 | 0.435 | 0.140 | 0.430 | 0.175 |
| 2wiki | sub200 | audited | kg_silver | 200 | 0.535 | 0.180 | 0.463 | 0.220 |
| 2wiki | sub200 | audited | kg_wikidata | 200 | 0.330 | 0.155 | 0.449 | 0.195 |
| 2wiki | sub200 | bm25 | none | 200 | 0.365 | 0.100 | 0.405 | 0.130 |
| 2wiki | sub200 | dense | bge_kg_silver | 200 | 0.470 | 0.160 | 0.462 | 0.190 |
| 2wiki | sub200 | graph_centric | bge_kg_openie | 200 | 0.460 | 0.165 | 0.440 | 0.190 |
| 2wiki | sub200 | graph_centric | bge_kg_silver | 200 | 0.730 | 0.230 | 0.511 | 0.270 |
| 2wiki | sub200 | graph_centric | kg_openie | 200 | 0.395 | 0.100 | 0.404 | 0.125 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab | 200 | 0.420 | 0.095 | 0.398 | 0.130 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_noprov | 200 | 0.360 | 0.085 | 0.385 | 0.115 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_q25 | 200 | 0.390 | 0.095 | 0.393 | 0.125 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_q50 | 200 | 0.390 | 0.095 | 0.396 | 0.125 |
| 2wiki | sub200 | graph_centric | kg_openie_svocab_q75 | 200 | 0.420 | 0.095 | 0.403 | 0.130 |
| 2wiki | sub200 | graph_centric | kg_silver | 200 | 0.750 | 0.240 | 0.512 | 0.265 |
| 2wiki | sub200 | graph_centric | kg_wikidata | 200 | 0.380 | 0.150 | 0.425 | 0.175 |
| 2wiki | sub200 | hipporag2 | kg_openie | 200 | 0.645 | 0.180 | 0.500 | 0.210 |
| 2wiki | sub200 | hipporag2 | kg_silver | 200 | 0.885 | 0.255 | 0.558 | 0.285 |
| 2wiki | sub200 | hipporag2 | kg_wikidata | 200 | 0.560 | 0.230 | 0.525 | 0.260 |
| 2wiki | sub200 | none | closedbook | 200 | - | 0.035 | 0.305 | 0.040 |
| 2wiki | sub200 | none | oracle | 200 | 1.000 | 0.595 | 0.779 | 0.655 |
| 2wiki | sub200 | ppr | kg_openie | 200 | 0.425 | 0.145 | 0.435 | 0.170 |
| 2wiki | sub200 | ppr | kg_openie_svocab | 200 | 0.510 | 0.175 | 0.471 | 0.210 |
| 2wiki | sub200 | ppr | kg_openie_svocab_noprov | 200 | 0.400 | 0.135 | 0.409 | 0.160 |
| 2wiki | sub200 | ppr | kg_openie_svocab_q25 | 200 | 0.440 | 0.145 | 0.419 | 0.170 |
| 2wiki | sub200 | ppr | kg_openie_svocab_q50 | 200 | 0.460 | 0.150 | 0.430 | 0.175 |
| 2wiki | sub200 | ppr | kg_openie_svocab_q75 | 200 | 0.495 | 0.145 | 0.440 | 0.180 |
| 2wiki | sub200 | ppr | kg_silver | 200 | 0.740 | 0.475 | 0.703 | 0.535 |
| 2wiki | sub200 | ppr | kg_wikidata | 200 | 0.500 | 0.180 | 0.465 | 0.200 |
| graphragbench | holdout200 | audited | bge_kg_openie | 200 | 0.545 | 0.000 | 0.290 | 0.000 |
| graphragbench | holdout200 | audited | bge_kg_silver | 200 | 0.650 | 0.000 | 0.300 | 0.000 |
| graphragbench | holdout200 | audited | kg_openie | 200 | 0.520 | 0.000 | 0.267 | 0.000 |
| graphragbench | holdout200 | audited | kg_silver | 200 | 0.580 | 0.000 | 0.271 | 0.000 |
| graphragbench | holdout200 | bm25 | none | 200 | 0.505 | 0.000 | 0.265 | 0.000 |
| graphragbench | holdout200 | dense | bge_kg_silver | 200 | 0.535 | 0.000 | 0.294 | 0.000 |
| graphragbench | holdout200 | graph_centric | bge_kg_openie | 200 | 0.485 | 0.000 | 0.279 | 0.000 |
| graphragbench | holdout200 | graph_centric | bge_kg_silver | 200 | 0.735 | 0.000 | 0.305 | 0.000 |
| graphragbench | holdout200 | graph_centric | kg_openie | 200 | 0.400 | 0.000 | 0.266 | 0.000 |
| graphragbench | holdout200 | graph_centric | kg_silver | 200 | 0.680 | 0.000 | 0.301 | 0.000 |
| graphragbench | holdout200 | hipporag2 | kg_openie | 200 | 0.575 | 0.000 | 0.307 | 0.000 |
| graphragbench | holdout200 | hipporag2 | kg_silver | 200 | 0.800 | 0.000 | 0.308 | 0.000 |
| graphragbench | holdout200 | none | closedbook | 200 | - | 0.000 | 0.188 | 0.000 |
| graphragbench | holdout200 | none | oracle | 200 | 1.000 | 0.000 | 0.297 | 0.000 |
| graphragbench | holdout200 | ppr | kg_openie | 200 | 0.245 | 0.000 | 0.203 | 0.000 |
| graphragbench | holdout200 | ppr | kg_silver | 200 | 0.625 | 0.000 | 0.259 | 0.000 |
| graphragbench | sub200 | audited | bge_kg_openie | 200 | 0.525 | 0.000 | 0.288 | 0.000 |
| graphragbench | sub200 | audited | bge_kg_silver | 200 | 0.615 | 0.000 | 0.297 | 0.000 |
| graphragbench | sub200 | audited | kg_openie | 200 | 0.500 | 0.000 | 0.282 | 0.000 |
| graphragbench | sub200 | audited | kg_silver | 200 | 0.565 | 0.000 | 0.277 | 0.000 |
| graphragbench | sub200 | bm25 | none | 200 | 0.520 | 0.000 | 0.271 | 0.000 |
| graphragbench | sub200 | dense | bge_kg_silver | 200 | 0.470 | 0.000 | 0.300 | 0.000 |
| graphragbench | sub200 | graph_centric | bge_kg_openie | 200 | 0.490 | 0.000 | 0.279 | 0.000 |
| graphragbench | sub200 | graph_centric | bge_kg_silver | 200 | 0.735 | 0.000 | 0.282 | 0.000 |
| graphragbench | sub200 | graph_centric | kg_openie | 200 | 0.405 | 0.000 | 0.264 | 0.000 |
| graphragbench | sub200 | graph_centric | kg_silver | 200 | 0.695 | 0.000 | 0.270 | 0.000 |
| graphragbench | sub200 | hipporag2 | kg_openie | 200 | 0.540 | 0.000 | 0.289 | 0.000 |
| graphragbench | sub200 | hipporag2 | kg_silver | 200 | 0.805 | 0.000 | 0.302 | 0.000 |
| graphragbench | sub200 | none | closedbook | 200 | - | 0.000 | 0.159 | 0.000 |
| graphragbench | sub200 | none | oracle | 200 | 1.000 | 0.000 | 0.288 | 0.000 |
| graphragbench | sub200 | ppr | kg_openie | 200 | 0.200 | 0.000 | 0.202 | 0.000 |
| graphragbench | sub200 | ppr | kg_silver | 200 | 0.690 | 0.000 | 0.254 | 0.000 |
| metaqa | holdout200 | audited | bge_kg_openie | 200 | 0.655 | 0.400 | 0.593 | 0.780 |
| metaqa | holdout200 | audited | bge_kg_silver | 200 | 0.735 | 0.410 | 0.618 | 0.810 |
| metaqa | holdout200 | audited | kg_fullkb | 200 | 0.805 | 0.480 | 0.684 | 0.855 |
| metaqa | holdout200 | audited | kg_openie | 200 | 0.875 | 0.450 | 0.668 | 0.870 |
| metaqa | holdout200 | audited | kg_silver | 200 | 0.920 | 0.435 | 0.656 | 0.870 |
| metaqa | holdout200 | bm25 | none | 200 | 0.855 | 0.435 | 0.649 | 0.860 |
| metaqa | holdout200 | dense | bge_kg_silver | 200 | 0.265 | 0.355 | 0.462 | 0.570 |
| metaqa | holdout200 | graph_centric | bge_kg_openie | 200 | 0.720 | 0.435 | 0.620 | 0.800 |
| metaqa | holdout200 | graph_centric | bge_kg_silver | 200 | 0.945 | 0.415 | 0.652 | 0.860 |
| metaqa | holdout200 | graph_centric | kg_fullkb | 200 | 0.845 | 0.455 | 0.671 | 0.865 |
| metaqa | holdout200 | graph_centric | kg_openie | 200 | 0.860 | 0.425 | 0.651 | 0.860 |
| metaqa | holdout200 | graph_centric | kg_silver | 200 | 0.945 | 0.450 | 0.661 | 0.870 |
| metaqa | holdout200 | hipporag2 | kg_fullkb | 200 | 0.590 | 0.440 | 0.607 | 0.770 |
| metaqa | holdout200 | hipporag2 | kg_openie | 200 | 0.745 | 0.385 | 0.576 | 0.770 |
| metaqa | holdout200 | hipporag2 | kg_silver | 200 | 0.915 | 0.445 | 0.673 | 0.880 |
| metaqa | holdout200 | none | closedbook | 200 | - | 0.170 | 0.267 | 0.385 |
| metaqa | holdout200 | none | oracle | 200 | 1.000 | 0.490 | 0.725 | 0.920 |
| metaqa | holdout200 | ppr | kg_fullkb | 200 | 1.000 | 0.455 | 0.681 | 0.905 |
| metaqa | holdout200 | ppr | kg_openie | 200 | 0.930 | 0.485 | 0.684 | 0.875 |
| metaqa | holdout200 | ppr | kg_silver | 200 | 1.000 | 0.495 | 0.731 | 0.945 |
| metaqa | sub200 | audited | bge_kg_openie | 200 | 0.675 | 0.460 | 0.645 | 0.825 |
| metaqa | sub200 | audited | bge_kg_silver | 200 | 0.750 | 0.445 | 0.644 | 0.830 |
| metaqa | sub200 | audited | kg_fullkb | 200 | 0.820 | 0.460 | 0.667 | 0.860 |
| metaqa | sub200 | audited | kg_openie | 200 | 0.900 | 0.500 | 0.701 | 0.895 |
| metaqa | sub200 | audited | kg_silver | 200 | 0.935 | 0.470 | 0.685 | 0.875 |
| metaqa | sub200 | bm25 | none | 200 | 0.870 | 0.430 | 0.644 | 0.845 |
| metaqa | sub200 | dense | bge_kg_silver | 200 | 0.290 | 0.325 | 0.462 | 0.570 |
| metaqa | sub200 | graph_centric | bge_kg_openie | 200 | 0.765 | 0.430 | 0.631 | 0.825 |
| metaqa | sub200 | graph_centric | bge_kg_silver | 200 | 0.945 | 0.450 | 0.672 | 0.865 |
| metaqa | sub200 | graph_centric | kg_fullkb | 200 | 0.845 | 0.470 | 0.693 | 0.885 |
| metaqa | sub200 | graph_centric | kg_openie | 200 | 0.865 | 0.445 | 0.652 | 0.845 |
| metaqa | sub200 | graph_centric | kg_silver | 200 | 0.950 | 0.465 | 0.667 | 0.845 |
| metaqa | sub200 | hipporag2 | kg_fullkb | 200 | 0.695 | 0.420 | 0.622 | 0.775 |
| metaqa | sub200 | hipporag2 | kg_openie | 200 | 0.755 | 0.395 | 0.591 | 0.765 |
| metaqa | sub200 | hipporag2 | kg_silver | 200 | 0.935 | 0.450 | 0.671 | 0.850 |
| metaqa | sub200 | none | closedbook | 200 | - | 0.150 | 0.251 | 0.365 |
| metaqa | sub200 | none | oracle | 200 | 1.000 | 0.500 | 0.736 | 0.925 |
| metaqa | sub200 | ppr | kg_fullkb | 200 | 0.995 | 0.470 | 0.696 | 0.900 |
| metaqa | sub200 | ppr | kg_openie | 200 | 0.935 | 0.455 | 0.673 | 0.895 |
| metaqa | sub200 | ppr | kg_silver | 200 | 1.000 | 0.515 | 0.730 | 0.920 |
| stark | holdout200 | audited | bge_kg_openie | 200 | 0.265 | 0.190 | 0.242 | 0.235 |
| stark | holdout200 | audited | bge_kg_silver | 200 | 0.300 | 0.180 | 0.242 | 0.230 |
| stark | holdout200 | audited | kg_openie | 200 | 0.250 | 0.190 | 0.230 | 0.225 |
| stark | holdout200 | audited | kg_silver | 200 | 0.275 | 0.175 | 0.244 | 0.215 |
| stark | holdout200 | bm25 | none | 200 | 0.215 | 0.165 | 0.220 | 0.210 |
| stark | holdout200 | dense | bge_kg_silver | 200 | 0.260 | 0.160 | 0.212 | 0.205 |
| stark | holdout200 | graph_centric | bge_kg_openie | 200 | 0.225 | 0.155 | 0.206 | 0.200 |
| stark | holdout200 | graph_centric | bge_kg_silver | 200 | 0.280 | 0.165 | 0.227 | 0.220 |
| stark | holdout200 | graph_centric | kg_openie | 200 | 0.205 | 0.160 | 0.203 | 0.195 |
| stark | holdout200 | graph_centric | kg_silver | 200 | 0.270 | 0.170 | 0.226 | 0.215 |
| stark | holdout200 | none | closedbook | 200 | - | 0.025 | 0.068 | 0.040 |
| stark | holdout200 | none | oracle | 200 | 1.000 | 0.675 | 0.721 | 0.710 |
| stark | holdout200 | ppr | kg_openie | 200 | 0.100 | 0.085 | 0.131 | 0.130 |
| stark | holdout200 | ppr | kg_silver | 200 | 0.120 | 0.105 | 0.137 | 0.125 |
| stark | sub200 | audited | bge_kg_openie | 200 | 0.165 | 0.120 | 0.176 | 0.160 |
| stark | sub200 | audited | bge_kg_silver | 200 | 0.135 | 0.080 | 0.159 | 0.120 |
| stark | sub200 | audited | kg_openie | 200 | 0.145 | 0.110 | 0.185 | 0.155 |
| stark | sub200 | audited | kg_silver | 200 | 0.130 | 0.080 | 0.160 | 0.115 |
| stark | sub200 | bm25 | none | 200 | 0.135 | 0.090 | 0.153 | 0.120 |
| stark | sub200 | dense | bge_kg_silver | 200 | 0.165 | 0.115 | 0.205 | 0.190 |
| stark | sub200 | graph_centric | bge_kg_openie | 200 | 0.145 | 0.105 | 0.173 | 0.145 |
| stark | sub200 | graph_centric | bge_kg_silver | 200 | 0.165 | 0.110 | 0.182 | 0.140 |
| stark | sub200 | graph_centric | kg_openie | 200 | 0.100 | 0.075 | 0.126 | 0.115 |
| stark | sub200 | graph_centric | kg_silver | 200 | 0.125 | 0.065 | 0.138 | 0.105 |
| stark | sub200 | none | closedbook | 200 | - | 0.025 | 0.059 | 0.045 |
| stark | sub200 | none | oracle | 200 | 1.000 | 0.585 | 0.647 | 0.645 |
| stark | sub200 | ppr | kg_openie | 200 | 0.040 | 0.050 | 0.090 | 0.070 |
| stark | sub200 | ppr | kg_silver | 200 | 0.065 | 0.050 | 0.106 | 0.090 |
