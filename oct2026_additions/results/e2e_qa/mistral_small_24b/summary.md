# End-to-end QA over frozen swap vectors: reader `mistral-small:24b`

## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader

| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| 2wiki | holdout200 | audited | 200 | +0.115 | 0.325 | 0.235 | +0.090 [+0.035, +0.145] | 3.93e-03 | +0.062 [+0.016, +0.109] | +0.095 |
| 2wiki | holdout200 | graph_centric | 200 | +0.335 | 0.380 | 0.275 | +0.105 [+0.050, +0.160] | 5.08e-04 | +0.111 [+0.063, +0.159] | +0.115 |
| 2wiki | holdout200 | hipporag2 | 200 | +0.275 | 0.495 | 0.390 | +0.105 [+0.045, +0.165] | 1.45e-03 | +0.091 [+0.037, +0.144] | +0.105 |
| 2wiki | holdout200 | ppr | 200 | +0.310 | 0.560 | 0.335 | +0.225 [+0.155, +0.295] | 1.18e-08 | +0.219 [+0.156, +0.281] | +0.240 |
| 2wiki | sub200 | audited | 200 | +0.160 | 0.235 | 0.170 | +0.065 [+0.020, +0.115] | 1.46e-02 | +0.088 [+0.045, +0.132] | +0.070 |
| 2wiki | sub200 | graph_centric | 200 | +0.355 | 0.330 | 0.190 | +0.140 [+0.080, +0.205] | 2.54e-05 | +0.149 [+0.092, +0.207] | +0.145 |
| 2wiki | sub200 | hipporag2 | 200 | +0.240 | 0.380 | 0.290 | +0.090 [+0.030, +0.150] | 6.43e-03 | +0.080 [+0.024, +0.135] | +0.080 |
| 2wiki | sub200 | ppr | 200 | +0.315 | 0.440 | 0.235 | +0.205 [+0.135, +0.275] | 5.26e-08 | +0.229 [+0.168, +0.292] | +0.185 |

## Mediation: EM Δ conditional on retrieval flip status

| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |
|---|---|---|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | 33 | +0.394 | 157 | +0.057 |
| 2wiki | holdout200 | graph_centric | 68 | +0.353 | 131 | -0.023 |
| 2wiki | holdout200 | hipporag2 | 60 | +0.400 | 135 | -0.015 |
| 2wiki | holdout200 | ppr | 73 | +0.452 | 116 | +0.121 |
| 2wiki | sub200 | audited | 37 | +0.378 | 158 | -0.006 |
| 2wiki | sub200 | graph_centric | 75 | +0.440 | 121 | -0.041 |
| 2wiki | sub200 | hipporag2 | 58 | +0.379 | 132 | -0.008 |
| 2wiki | sub200 | ppr | 75 | +0.360 | 113 | +0.142 |

## All cells

| bench | split | system | condition | n | hit@5 | EM | F1 | cover |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | kg_openie | 200 | 0.365 | 0.235 | 0.456 | 0.250 |
| 2wiki | holdout200 | audited | kg_silver | 200 | 0.480 | 0.325 | 0.519 | 0.345 |
| 2wiki | holdout200 | bm25 | none | 200 | 0.355 | 0.255 | 0.443 | 0.260 |
| 2wiki | holdout200 | graph_centric | kg_openie | 200 | 0.370 | 0.275 | 0.468 | 0.290 |
| 2wiki | holdout200 | graph_centric | kg_silver | 200 | 0.705 | 0.380 | 0.579 | 0.405 |
| 2wiki | holdout200 | hipporag2 | kg_openie | 200 | 0.615 | 0.390 | 0.583 | 0.410 |
| 2wiki | holdout200 | hipporag2 | kg_silver | 200 | 0.890 | 0.495 | 0.673 | 0.515 |
| 2wiki | holdout200 | none | closedbook | 200 | - | 0.055 | 0.242 | 0.055 |
| 2wiki | holdout200 | none | oracle | 200 | 1.000 | 0.620 | 0.815 | 0.660 |
| 2wiki | holdout200 | ppr | kg_openie | 200 | 0.430 | 0.335 | 0.521 | 0.345 |
| 2wiki | holdout200 | ppr | kg_silver | 200 | 0.740 | 0.560 | 0.740 | 0.585 |
| 2wiki | sub200 | audited | kg_openie | 200 | 0.375 | 0.170 | 0.403 | 0.190 |
| 2wiki | sub200 | audited | kg_silver | 200 | 0.535 | 0.235 | 0.491 | 0.260 |
| 2wiki | sub200 | bm25 | none | 200 | 0.365 | 0.135 | 0.386 | 0.165 |
| 2wiki | sub200 | graph_centric | kg_openie | 200 | 0.395 | 0.190 | 0.412 | 0.215 |
| 2wiki | sub200 | graph_centric | kg_silver | 200 | 0.750 | 0.330 | 0.561 | 0.360 |
| 2wiki | sub200 | hipporag2 | kg_openie | 200 | 0.645 | 0.290 | 0.542 | 0.320 |
| 2wiki | sub200 | hipporag2 | kg_silver | 200 | 0.885 | 0.380 | 0.622 | 0.400 |
| 2wiki | sub200 | none | closedbook | 200 | - | 0.050 | 0.232 | 0.050 |
| 2wiki | sub200 | none | oracle | 200 | 1.000 | 0.555 | 0.773 | 0.570 |
| 2wiki | sub200 | ppr | kg_openie | 200 | 0.425 | 0.235 | 0.432 | 0.270 |
| 2wiki | sub200 | ppr | kg_silver | 200 | 0.740 | 0.440 | 0.662 | 0.455 |
