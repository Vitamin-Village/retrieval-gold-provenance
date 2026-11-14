# End-to-end QA over frozen swap vectors: reader `claude-sonnet-5-5`

## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader

| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| 2wiki | holdout200 | audited | 200 | +0.115 | 0.520 | 0.500 | +0.020 [-0.045, +0.080] | 6.44e-01 | +0.025 [-0.017, +0.067] | +0.035 |
| 2wiki | holdout200 | graph_centric | 200 | +0.335 | 0.640 | 0.470 | +0.170 [+0.105, +0.235] | 1.16e-06 | +0.128 [+0.079, +0.178] | +0.100 |
| 2wiki | holdout200 | hipporag2 | 200 | +0.275 | 0.830 | 0.590 | +0.240 [+0.175, +0.305] | 3.50e-11 | +0.172 [+0.123, +0.225] | +0.080 |
| 2wiki | holdout200 | ppr | 200 | +0.310 | 0.765 | 0.510 | +0.255 [+0.175, +0.335] | 1.69e-09 | +0.225 [+0.170, +0.282] | +0.160 |
| 2wiki | sub200 | audited | 200 | +0.160 | 0.585 | 0.455 | +0.130 [+0.070, +0.190] | 6.88e-05 | +0.120 [+0.071, +0.168] | +0.070 |
| 2wiki | sub200 | graph_centric | 200 | +0.355 | 0.685 | 0.480 | +0.205 [+0.135, +0.280] | 1.67e-07 | +0.150 [+0.097, +0.204] | +0.125 |
| 2wiki | sub200 | hipporag2 | 200 | +0.240 | 0.740 | 0.580 | +0.160 [+0.085, +0.235] | 5.78e-05 | +0.126 [+0.069, +0.184] | +0.080 |
| 2wiki | sub200 | ppr | 200 | +0.315 | 0.745 | 0.450 | +0.295 [+0.225, +0.365] | 4.12e-14 | +0.212 [+0.157, +0.268] | +0.165 |

## Mediation: EM Δ conditional on retrieval flip status

| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |
|---|---|---|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | 33 | +0.424 | 157 | -0.025 |
| 2wiki | holdout200 | graph_centric | 68 | +0.456 | 131 | +0.031 |
| 2wiki | holdout200 | hipporag2 | 60 | +0.650 | 135 | +0.074 |
| 2wiki | holdout200 | ppr | 73 | +0.603 | 116 | +0.095 |
| 2wiki | sub200 | audited | 37 | +0.568 | 158 | +0.044 |
| 2wiki | sub200 | graph_centric | 75 | +0.560 | 121 | +0.008 |
| 2wiki | sub200 | hipporag2 | 58 | +0.690 | 132 | -0.015 |
| 2wiki | sub200 | ppr | 75 | +0.653 | 113 | +0.115 |

## All cells

| bench | split | system | condition | n | hit@5 | EM | F1 | cover |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2wiki | holdout200 | audited | kg_openie | 200 | 0.365 | 0.500 | 0.661 | 0.750 |
| 2wiki | holdout200 | audited | kg_silver | 200 | 0.480 | 0.520 | 0.686 | 0.785 |
| 2wiki | holdout200 | bm25 | none | 200 | 0.355 | 0.485 | 0.656 | 0.750 |
| 2wiki | holdout200 | graph_centric | kg_openie | 200 | 0.370 | 0.470 | 0.632 | 0.725 |
| 2wiki | holdout200 | graph_centric | kg_silver | 200 | 0.705 | 0.640 | 0.759 | 0.825 |
| 2wiki | holdout200 | hipporag2 | kg_openie | 200 | 0.615 | 0.590 | 0.723 | 0.815 |
| 2wiki | holdout200 | hipporag2 | kg_silver | 200 | 0.890 | 0.830 | 0.895 | 0.895 |
| 2wiki | holdout200 | none | closedbook | 200 | - | 0.125 | 0.264 | 0.435 |
| 2wiki | holdout200 | none | oracle | 200 | 1.000 | 0.870 | 0.958 | 0.925 |
| 2wiki | holdout200 | ppr | kg_openie | 200 | 0.430 | 0.510 | 0.644 | 0.740 |
| 2wiki | holdout200 | ppr | kg_silver | 200 | 0.740 | 0.765 | 0.869 | 0.900 |
| 2wiki | sub200 | audited | kg_openie | 200 | 0.375 | 0.455 | 0.618 | 0.720 |
| 2wiki | sub200 | audited | kg_silver | 200 | 0.535 | 0.585 | 0.738 | 0.790 |
| 2wiki | sub200 | bm25 | none | 200 | 0.365 | 0.450 | 0.630 | 0.730 |
| 2wiki | sub200 | graph_centric | kg_openie | 200 | 0.395 | 0.480 | 0.653 | 0.725 |
| 2wiki | sub200 | graph_centric | kg_silver | 200 | 0.750 | 0.685 | 0.803 | 0.850 |
| 2wiki | sub200 | hipporag2 | kg_openie | 200 | 0.645 | 0.580 | 0.724 | 0.775 |
| 2wiki | sub200 | hipporag2 | kg_silver | 200 | 0.885 | 0.740 | 0.851 | 0.855 |
| 2wiki | sub200 | none | closedbook | 200 | - | 0.090 | 0.254 | 0.390 |
| 2wiki | sub200 | none | oracle | 200 | 1.000 | 0.830 | 0.917 | 0.890 |
| 2wiki | sub200 | ppr | kg_openie | 200 | 0.425 | 0.450 | 0.621 | 0.710 |
| 2wiki | sub200 | ppr | kg_silver | 200 | 0.740 | 0.745 | 0.833 | 0.875 |
