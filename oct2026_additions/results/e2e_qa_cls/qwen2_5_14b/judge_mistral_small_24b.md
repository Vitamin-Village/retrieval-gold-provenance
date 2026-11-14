# Judge accuracy (judge `mistral-small:24b`) for reader `qwen2.5:14b`

| bench | split | system | n | hit@5 Δ | acc aligned | acc indep. | Δ [95% CI] | p (sign) | Δ on flips+ (n) | Δ on no-flip (n) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| graphragbench | holdout200 | audited | 200 | +0.060 | 0.690 | 0.680 | +0.010 [-0.035, +0.055] | 8.32e-01 | +0.077 (13) | +0.011 (186) |
| graphragbench | holdout200 | audited@bge | 200 | +0.105 | 0.710 | 0.705 | +0.005 [-0.045, +0.055] | 1.00e+00 | +0.238 (21) | -0.022 (179) |
| graphragbench | holdout200 | graph_centric | 200 | +0.280 | 0.725 | 0.660 | +0.065 [+0.005, +0.125] | 5.33e-02 | +0.224 (58) | +0.007 (140) |
| graphragbench | holdout200 | graph_centric@bge | 200 | +0.250 | 0.750 | 0.660 | +0.090 [+0.030, +0.150] | 6.43e-03 | +0.245 (53) | +0.042 (144) |
| graphragbench | holdout200 | hipporag2 | 200 | +0.225 | 0.760 | 0.750 | +0.010 [-0.040, +0.060] | 8.51e-01 | +0.062 (48) | +0.007 (149) |
| graphragbench | holdout200 | ppr | 200 | +0.380 | 0.585 | 0.485 | +0.100 [+0.015, +0.180] | 2.45e-02 | +0.287 (87) | +0.049 (102) |
| graphragbench | sub200 | audited | 200 | +0.065 | 0.735 | 0.720 | +0.015 [-0.025, +0.050] | 6.07e-01 | +0.154 (13) | +0.005 (187) |
| graphragbench | sub200 | audited@bge | 200 | +0.090 | 0.735 | 0.730 | +0.005 [-0.035, +0.045] | 1.00e+00 | +0.050 (20) | +0.000 (178) |
| graphragbench | sub200 | graph_centric | 200 | +0.290 | 0.740 | 0.700 | +0.040 [-0.025, +0.105] | 2.91e-01 | +0.207 (58) | -0.028 (142) |
| graphragbench | sub200 | graph_centric@bge | 200 | +0.245 | 0.755 | 0.730 | +0.025 [-0.030, +0.080] | 4.73e-01 | +0.059 (51) | +0.014 (147) |
| graphragbench | sub200 | hipporag2 | 200 | +0.265 | 0.765 | 0.725 | +0.040 [-0.010, +0.095] | 2.00e-01 | +0.151 (53) | +0.000 (147) |
| graphragbench | sub200 | ppr | 200 | +0.490 | 0.675 | 0.455 | +0.220 [+0.145, +0.295] | 1.03e-07 | +0.400 (105) | +0.045 (88) |

| bench | split | system | condition | n | judge acc |
|---|---|---|---|---:|---:|
| graphragbench | holdout200 | audited | bge_kg_openie | 200 | 0.705 |
| graphragbench | holdout200 | audited | bge_kg_silver | 200 | 0.710 |
| graphragbench | holdout200 | audited | kg_openie | 200 | 0.680 |
| graphragbench | holdout200 | audited | kg_silver | 200 | 0.690 |
| graphragbench | holdout200 | bm25 | none | 200 | 0.660 |
| graphragbench | holdout200 | dense | bge_kg_silver | 200 | 0.720 |
| graphragbench | holdout200 | graph_centric | bge_kg_openie | 200 | 0.660 |
| graphragbench | holdout200 | graph_centric | bge_kg_silver | 200 | 0.750 |
| graphragbench | holdout200 | graph_centric | kg_openie | 200 | 0.660 |
| graphragbench | holdout200 | graph_centric | kg_silver | 200 | 0.725 |
| graphragbench | holdout200 | hipporag2 | kg_openie | 200 | 0.750 |
| graphragbench | holdout200 | hipporag2 | kg_silver | 200 | 0.760 |
| graphragbench | holdout200 | none | closedbook | 200 | 0.550 |
| graphragbench | holdout200 | none | oracle | 200 | 0.735 |
| graphragbench | holdout200 | ppr | kg_openie | 200 | 0.485 |
| graphragbench | holdout200 | ppr | kg_silver | 200 | 0.585 |
| graphragbench | sub200 | audited | bge_kg_openie | 200 | 0.730 |
| graphragbench | sub200 | audited | bge_kg_silver | 200 | 0.735 |
| graphragbench | sub200 | audited | kg_openie | 200 | 0.720 |
| graphragbench | sub200 | audited | kg_silver | 200 | 0.735 |
| graphragbench | sub200 | bm25 | none | 200 | 0.710 |
| graphragbench | sub200 | dense | bge_kg_silver | 200 | 0.705 |
| graphragbench | sub200 | graph_centric | bge_kg_openie | 200 | 0.730 |
| graphragbench | sub200 | graph_centric | bge_kg_silver | 200 | 0.755 |
| graphragbench | sub200 | graph_centric | kg_openie | 200 | 0.700 |
| graphragbench | sub200 | graph_centric | kg_silver | 200 | 0.740 |
| graphragbench | sub200 | hipporag2 | kg_openie | 200 | 0.725 |
| graphragbench | sub200 | hipporag2 | kg_silver | 200 | 0.765 |
| graphragbench | sub200 | none | closedbook | 200 | 0.575 |
| graphragbench | sub200 | none | oracle | 200 | 0.785 |
| graphragbench | sub200 | ppr | kg_openie | 200 | 0.455 |
| graphragbench | sub200 | ppr | kg_silver | 200 | 0.675 |
