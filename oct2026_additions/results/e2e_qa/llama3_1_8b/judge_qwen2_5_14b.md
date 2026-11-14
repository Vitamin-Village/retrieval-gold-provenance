# Judge accuracy (judge `qwen2.5:14b`) for reader `llama3.1:8b`

| bench | split | system | n | hit@5 Δ | acc aligned | acc indep. | Δ [95% CI] | p (sign) | Δ on flips+ (n) | Δ on no-flip (n) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| graphragbench | holdout200 | audited | 200 | +0.060 | 0.470 | 0.495 | -0.025 [-0.070, +0.020] | 4.05e-01 | -0.077 (13) | -0.016 (186) |
| graphragbench | holdout200 | graph_centric | 200 | +0.280 | 0.550 | 0.490 | +0.060 [-0.005, +0.130] | 1.11e-01 | +0.190 (58) | +0.021 (140) |
| graphragbench | holdout200 | ppr | 200 | +0.380 | 0.375 | 0.340 | +0.035 [-0.045, +0.115] | 4.64e-01 | +0.195 (87) | +0.000 (102) |
| graphragbench | sub200 | audited | 200 | +0.065 | 0.565 | 0.535 | +0.030 [-0.020, +0.085] | 3.62e-01 | +0.154 (13) | +0.021 (187) |
| graphragbench | sub200 | graph_centric | 200 | +0.290 | 0.570 | 0.520 | +0.050 [-0.015, +0.115] | 1.64e-01 | +0.190 (58) | -0.007 (142) |
| graphragbench | sub200 | ppr | 200 | +0.490 | 0.475 | 0.355 | +0.120 [+0.040, +0.200] | 6.31e-03 | +0.314 (105) | -0.057 (88) |

| bench | split | system | condition | n | judge acc |
|---|---|---|---|---:|---:|
| graphragbench | holdout200 | audited | bge_kg_openie | 200 | 0.505 |
| graphragbench | holdout200 | audited | bge_kg_silver | 200 | 0.540 |
| graphragbench | holdout200 | audited | kg_openie | 200 | 0.495 |
| graphragbench | holdout200 | audited | kg_silver | 200 | 0.470 |
| graphragbench | holdout200 | bm25 | none | 200 | 0.480 |
| graphragbench | holdout200 | dense | bge_kg_silver | 200 | 0.455 |
| graphragbench | holdout200 | graph_centric | bge_kg_openie | 200 | 0.500 |
| graphragbench | holdout200 | graph_centric | bge_kg_silver | 200 | 0.565 |
| graphragbench | holdout200 | graph_centric | kg_openie | 200 | 0.490 |
| graphragbench | holdout200 | graph_centric | kg_silver | 200 | 0.550 |
| graphragbench | holdout200 | none | closedbook | 200 | 0.425 |
| graphragbench | holdout200 | none | oracle | 200 | 0.555 |
| graphragbench | holdout200 | ppr | kg_openie | 200 | 0.340 |
| graphragbench | holdout200 | ppr | kg_silver | 200 | 0.375 |
| graphragbench | sub200 | audited | bge_kg_openie | 200 | 0.560 |
| graphragbench | sub200 | audited | bge_kg_silver | 200 | 0.595 |
| graphragbench | sub200 | audited | kg_openie | 200 | 0.535 |
| graphragbench | sub200 | audited | kg_silver | 200 | 0.565 |
| graphragbench | sub200 | bm25 | none | 200 | 0.535 |
| graphragbench | sub200 | dense | bge_kg_silver | 200 | 0.505 |
| graphragbench | sub200 | graph_centric | bge_kg_openie | 200 | 0.545 |
| graphragbench | sub200 | graph_centric | bge_kg_silver | 200 | 0.595 |
| graphragbench | sub200 | graph_centric | kg_openie | 200 | 0.520 |
| graphragbench | sub200 | graph_centric | kg_silver | 200 | 0.570 |
| graphragbench | sub200 | none | closedbook | 200 | 0.435 |
| graphragbench | sub200 | none | oracle | 200 | 0.570 |
| graphragbench | sub200 | ppr | kg_openie | 200 | 0.355 |
| graphragbench | sub200 | ppr | kg_silver | 200 | 0.475 |
