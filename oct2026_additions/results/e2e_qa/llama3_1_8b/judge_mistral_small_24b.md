# Judge accuracy (judge `mistral-small:24b`) for reader `llama3.1:8b`

| bench | split | system | n | hit@5 Δ | acc aligned | acc indep. | Δ [95% CI] | p (sign) | Δ on flips+ (n) | Δ on no-flip (n) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| graphragbench | holdout200 | audited | 200 | +0.060 | 0.575 | 0.570 | +0.005 [-0.040, +0.050] | 1.00e+00 | +0.077 (13) | +0.005 (186) |
| graphragbench | holdout200 | audited@bge | 200 | +0.100 | 0.655 | 0.585 | +0.070 [+0.020, +0.125] | 1.61e-02 | +0.250 (20) | +0.050 (180) |
| graphragbench | holdout200 | graph_centric | 200 | +0.280 | 0.650 | 0.580 | +0.070 [+0.000, +0.140] | 7.04e-02 | +0.172 (58) | +0.036 (140) |
| graphragbench | holdout200 | graph_centric@bge | 200 | +0.215 | 0.650 | 0.590 | +0.060 [+0.000, +0.125] | 8.07e-02 | +0.133 (45) | +0.046 (153) |
| graphragbench | holdout200 | hipporag2 | 200 | +0.225 | 0.665 | 0.675 | -0.010 [-0.070, +0.050] | 8.68e-01 | +0.083 (48) | -0.034 (149) |
| graphragbench | holdout200 | ppr | 200 | +0.380 | 0.465 | 0.420 | +0.045 [-0.040, +0.130] | 3.62e-01 | +0.253 (87) | -0.029 (102) |
| graphragbench | sub200 | audited | 200 | +0.065 | 0.690 | 0.640 | +0.050 [+0.005, +0.100] | 7.55e-02 | +0.154 (13) | +0.043 (187) |
| graphragbench | sub200 | audited@bge | 200 | +0.125 | 0.675 | 0.680 | -0.005 [-0.055, +0.045] | 1.00e+00 | +0.077 (26) | -0.017 (173) |
| graphragbench | sub200 | graph_centric | 200 | +0.290 | 0.680 | 0.590 | +0.090 [+0.025, +0.160] | 1.53e-02 | +0.276 (58) | +0.014 (142) |
| graphragbench | sub200 | graph_centric@bge | 200 | +0.245 | 0.705 | 0.670 | +0.035 [-0.025, +0.095] | 3.24e-01 | +0.180 (50) | -0.013 (149) |
| graphragbench | sub200 | hipporag2 | 200 | +0.265 | 0.715 | 0.705 | +0.010 [-0.050, +0.070] | 8.64e-01 | +0.170 (53) | -0.048 (147) |
| graphragbench | sub200 | ppr | 200 | +0.490 | 0.585 | 0.410 | +0.175 [+0.085, +0.265] | 2.66e-04 | +0.410 (105) | -0.045 (88) |

| bench | split | system | condition | n | judge acc |
|---|---|---|---|---:|---:|
| graphragbench | holdout200 | audited | bge_kg_openie | 200 | 0.585 |
| graphragbench | holdout200 | audited | bge_kg_silver | 200 | 0.655 |
| graphragbench | holdout200 | audited | kg_openie | 200 | 0.570 |
| graphragbench | holdout200 | audited | kg_silver | 200 | 0.575 |
| graphragbench | holdout200 | bm25 | none | 200 | 0.560 |
| graphragbench | holdout200 | dense | bge_kg_silver | 200 | 0.575 |
| graphragbench | holdout200 | graph_centric | bge_kg_openie | 200 | 0.590 |
| graphragbench | holdout200 | graph_centric | bge_kg_silver | 200 | 0.650 |
| graphragbench | holdout200 | graph_centric | kg_openie | 200 | 0.580 |
| graphragbench | holdout200 | graph_centric | kg_silver | 200 | 0.650 |
| graphragbench | holdout200 | hipporag2 | kg_openie | 200 | 0.675 |
| graphragbench | holdout200 | hipporag2 | kg_silver | 200 | 0.665 |
| graphragbench | holdout200 | none | closedbook | 200 | 0.570 |
| graphragbench | holdout200 | none | oracle | 200 | 0.695 |
| graphragbench | holdout200 | ppr | kg_openie | 200 | 0.420 |
| graphragbench | holdout200 | ppr | kg_silver | 200 | 0.465 |
| graphragbench | sub200 | audited | bge_kg_openie | 200 | 0.680 |
| graphragbench | sub200 | audited | bge_kg_silver | 200 | 0.675 |
| graphragbench | sub200 | audited | kg_openie | 200 | 0.640 |
| graphragbench | sub200 | audited | kg_silver | 200 | 0.690 |
| graphragbench | sub200 | bm25 | none | 200 | 0.640 |
| graphragbench | sub200 | dense | bge_kg_silver | 200 | 0.580 |
| graphragbench | sub200 | graph_centric | bge_kg_openie | 200 | 0.670 |
| graphragbench | sub200 | graph_centric | bge_kg_silver | 200 | 0.705 |
| graphragbench | sub200 | graph_centric | kg_openie | 200 | 0.590 |
| graphragbench | sub200 | graph_centric | kg_silver | 200 | 0.680 |
| graphragbench | sub200 | hipporag2 | kg_openie | 200 | 0.705 |
| graphragbench | sub200 | hipporag2 | kg_silver | 200 | 0.715 |
| graphragbench | sub200 | none | closedbook | 200 | 0.595 |
| graphragbench | sub200 | none | oracle | 200 | 0.710 |
| graphragbench | sub200 | ppr | kg_openie | 200 | 0.410 |
| graphragbench | sub200 | ppr | kg_silver | 200 | 0.585 |
