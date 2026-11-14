# Judge accuracy (judge `llama3.1:8b`) for reader `qwen2.5:14b`

| bench | split | system | n | hit@5 Δ | acc aligned | acc indep. | Δ [95% CI] | p (sign) | Δ on flips+ (n) | Δ on no-flip (n) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| graphragbench | holdout200 | audited | 200 | +0.060 | 0.855 | 0.845 | +0.010 [-0.035, +0.055] | 8.32e-01 | +0.077 (13) | +0.011 (186) |
| graphragbench | holdout200 | graph_centric | 200 | +0.280 | 0.915 | 0.810 | +0.105 [+0.055, +0.155] | 1.04e-04 | +0.190 (58) | +0.079 (140) |
| graphragbench | holdout200 | ppr | 200 | +0.380 | 0.730 | 0.670 | +0.060 [-0.020, +0.140] | 1.88e-01 | +0.241 (87) | -0.010 (102) |
| graphragbench | sub200 | audited | 200 | +0.065 | 0.880 | 0.860 | +0.020 [-0.020, +0.060] | 4.81e-01 | +0.077 (13) | +0.016 (187) |
| graphragbench | sub200 | graph_centric | 200 | +0.290 | 0.870 | 0.800 | +0.070 [+0.005, +0.135] | 4.36e-02 | +0.276 (58) | -0.014 (142) |
| graphragbench | sub200 | ppr | 200 | +0.490 | 0.805 | 0.650 | +0.155 [+0.080, +0.230] | 1.94e-04 | +0.343 (105) | -0.034 (88) |

| bench | split | system | condition | n | judge acc |
|---|---|---|---|---:|---:|
| graphragbench | holdout200 | audited | bge_kg_openie | 200 | 0.880 |
| graphragbench | holdout200 | audited | bge_kg_silver | 200 | 0.905 |
| graphragbench | holdout200 | audited | kg_openie | 200 | 0.845 |
| graphragbench | holdout200 | audited | kg_silver | 200 | 0.855 |
| graphragbench | holdout200 | bm25 | none | 200 | 0.825 |
| graphragbench | holdout200 | dense | bge_kg_silver | 200 | 0.845 |
| graphragbench | holdout200 | graph_centric | bge_kg_openie | 200 | 0.865 |
| graphragbench | holdout200 | graph_centric | bge_kg_silver | 200 | 0.905 |
| graphragbench | holdout200 | graph_centric | kg_openie | 200 | 0.810 |
| graphragbench | holdout200 | graph_centric | kg_silver | 200 | 0.915 |
| graphragbench | holdout200 | none | closedbook | 200 | 0.790 |
| graphragbench | holdout200 | none | oracle | 200 | 0.900 |
| graphragbench | holdout200 | ppr | kg_openie | 200 | 0.670 |
| graphragbench | holdout200 | ppr | kg_silver | 200 | 0.730 |
| graphragbench | sub200 | audited | bge_kg_openie | 200 | 0.860 |
| graphragbench | sub200 | audited | bge_kg_silver | 200 | 0.905 |
| graphragbench | sub200 | audited | kg_openie | 200 | 0.860 |
| graphragbench | sub200 | audited | kg_silver | 200 | 0.880 |
| graphragbench | sub200 | bm25 | none | 200 | 0.855 |
| graphragbench | sub200 | dense | bge_kg_silver | 200 | 0.840 |
| graphragbench | sub200 | graph_centric | bge_kg_openie | 200 | 0.835 |
| graphragbench | sub200 | graph_centric | bge_kg_silver | 200 | 0.865 |
| graphragbench | sub200 | graph_centric | kg_openie | 200 | 0.800 |
| graphragbench | sub200 | graph_centric | kg_silver | 200 | 0.870 |
| graphragbench | sub200 | none | closedbook | 200 | 0.700 |
| graphragbench | sub200 | none | oracle | 200 | 0.870 |
| graphragbench | sub200 | ppr | kg_openie | 200 | 0.650 |
| graphragbench | sub200 | ppr | kg_silver | 200 | 0.805 |
