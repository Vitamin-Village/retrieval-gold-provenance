# End-to-end QA over frozen swap vectors: reader `llama3.1:8b`

## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader

| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| 2wiki | comp200 | audited | 200 | +0.100 | 0.245 | 0.210 | +0.035 [-0.005, +0.080] | 1.67e-01 | +0.042 [-0.002, +0.086] | +0.040 |
| 2wiki | comp200 | graph_centric | 200 | +0.215 | 0.315 | 0.205 | +0.110 [+0.055, +0.165] | 1.13e-04 | +0.114 [+0.061, +0.167] | +0.125 |
| 2wiki | comp200 | hipporag2 | 200 | +0.200 | 0.490 | 0.385 | +0.105 [+0.035, +0.175] | 4.60e-03 | +0.122 [+0.056, +0.188] | +0.125 |
| 2wiki | comp200 | ppr | 200 | +0.615 | 0.545 | 0.195 | +0.350 [+0.275, +0.430] | 4.15e-15 | +0.393 [+0.318, +0.468] | +0.440 |

## Mediation: EM Δ conditional on retrieval flip status

| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |
|---|---|---|---:|---:|---:|---:|
| 2wiki | comp200 | audited | 22 | +0.409 | 176 | -0.006 |
| 2wiki | comp200 | graph_centric | 44 | +0.500 | 155 | +0.006 |
| 2wiki | comp200 | hipporag2 | 48 | +0.500 | 144 | +0.000 |
| 2wiki | comp200 | ppr | 132 | +0.538 | 59 | +0.051 |

## All cells

| bench | split | system | condition | n | hit@5 | EM | F1 | cover |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2wiki | comp200 | audited | kg_openie | 200 | 0.210 | 0.210 | 0.242 | 0.250 |
| 2wiki | comp200 | audited | kg_silver | 200 | 0.310 | 0.245 | 0.284 | 0.290 |
| 2wiki | comp200 | bm25 | none | 200 | 0.215 | 0.180 | 0.221 | 0.215 |
| 2wiki | comp200 | graph_centric | kg_openie | 200 | 0.220 | 0.205 | 0.241 | 0.235 |
| 2wiki | comp200 | graph_centric | kg_silver | 200 | 0.435 | 0.315 | 0.354 | 0.360 |
| 2wiki | comp200 | hipporag2 | kg_openie | 200 | 0.725 | 0.385 | 0.448 | 0.470 |
| 2wiki | comp200 | hipporag2 | kg_silver | 200 | 0.925 | 0.490 | 0.570 | 0.595 |
| 2wiki | comp200 | none | closedbook | 200 | - | 0.110 | 0.144 | 0.130 |
| 2wiki | comp200 | none | oracle | 200 | 1.000 | 0.555 | 0.662 | 0.710 |
| 2wiki | comp200 | ppr | kg_openie | 200 | 0.240 | 0.195 | 0.236 | 0.245 |
| 2wiki | comp200 | ppr | kg_silver | 200 | 0.855 | 0.545 | 0.629 | 0.685 |
