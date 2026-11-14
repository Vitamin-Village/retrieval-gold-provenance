# End-to-end QA over frozen swap vectors: reader `qwen2.5:14b`

## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader

| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| 2wiki | comp200 | audited | 200 | +0.100 | 0.260 | 0.225 | +0.035 [-0.005, +0.080] | 1.67e-01 | +0.049 [+0.006, +0.094] | +0.050 |
| 2wiki | comp200 | graph_centric | 200 | +0.215 | 0.290 | 0.210 | +0.080 [+0.030, +0.130] | 3.72e-03 | +0.096 [+0.043, +0.147] | +0.095 |
| 2wiki | comp200 | hipporag2 | 200 | +0.200 | 0.525 | 0.465 | +0.060 [-0.010, +0.125] | 1.19e-01 | +0.051 [-0.016, +0.117] | +0.070 |
| 2wiki | comp200 | ppr | 200 | +0.615 | 0.570 | 0.225 | +0.345 [+0.265, +0.420] | 7.47e-15 | +0.398 [+0.321, +0.474] | +0.450 |

## Mediation: EM Δ conditional on retrieval flip status

| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |
|---|---|---|---:|---:|---:|---:|
| 2wiki | comp200 | audited | 22 | +0.273 | 176 | +0.011 |
| 2wiki | comp200 | graph_centric | 44 | +0.364 | 155 | +0.006 |
| 2wiki | comp200 | hipporag2 | 48 | +0.375 | 144 | -0.014 |
| 2wiki | comp200 | ppr | 132 | +0.523 | 59 | +0.034 |

## All cells

| bench | split | system | condition | n | hit@5 | EM | F1 | cover |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2wiki | comp200 | audited | kg_openie | 200 | 0.210 | 0.225 | 0.274 | 0.260 |
| 2wiki | comp200 | audited | kg_silver | 200 | 0.310 | 0.260 | 0.323 | 0.310 |
| 2wiki | comp200 | bm25 | none | 200 | 0.215 | 0.215 | 0.273 | 0.245 |
| 2wiki | comp200 | graph_centric | kg_openie | 200 | 0.220 | 0.210 | 0.271 | 0.255 |
| 2wiki | comp200 | graph_centric | kg_silver | 200 | 0.435 | 0.290 | 0.367 | 0.350 |
| 2wiki | comp200 | hipporag2 | kg_openie | 200 | 0.725 | 0.465 | 0.553 | 0.565 |
| 2wiki | comp200 | hipporag2 | kg_silver | 200 | 0.925 | 0.525 | 0.604 | 0.635 |
| 2wiki | comp200 | none | closedbook | 200 | - | 0.080 | 0.122 | 0.085 |
| 2wiki | comp200 | none | oracle | 200 | 1.000 | 0.610 | 0.718 | 0.760 |
| 2wiki | comp200 | ppr | kg_openie | 200 | 0.240 | 0.225 | 0.278 | 0.265 |
| 2wiki | comp200 | ppr | kg_silver | 200 | 0.855 | 0.570 | 0.676 | 0.715 |
