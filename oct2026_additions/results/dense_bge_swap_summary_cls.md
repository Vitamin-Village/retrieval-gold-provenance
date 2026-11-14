# KG swap with a real dense channel (BGE-M3, [CLS] pooling)

| Benchmark | Split | System | BM25 | Dense-only | Graph, aligned KG | Graph, independent KG | Swap [95% CI] | +/- | p |
|---|---|---|---|---|---|---|---|---|---|
| 2wiki | sub200 | proposed_v3 | 0.365 | 0.470 | 0.625 | 0.430 | +0.195 [+0.135, +0.255] | +43/-4 | 2.78e-09 |
| 2wiki | sub200 | proposed_v5_graphcentric | 0.365 | 0.470 | 0.730 | 0.460 | +0.270 [+0.205, +0.335] | +57/-3 | 6.25e-14 |
| 2wiki | holdout200 | proposed_v3 | 0.355 | 0.450 | 0.550 | 0.400 | +0.150 [+0.085, +0.215] | +40/-10 | 2.39e-05 |
| 2wiki | holdout200 | proposed_v5_graphcentric | 0.355 | 0.450 | 0.720 | 0.470 | +0.250 [+0.190, +0.315] | +52/-2 | 1.65e-13 |
| metaqa | sub200 | proposed_v3 | 0.870 | 0.290 | 0.750 | 0.675 | +0.075 [+0.040, +0.110] | +15/-0 | 6.10e-05 |
| metaqa | sub200 | proposed_v5_graphcentric | 0.870 | 0.290 | 0.945 | 0.765 | +0.180 [+0.130, +0.235] | +36/-0 | 2.91e-11 |
| metaqa | holdout200 | proposed_v3 | 0.855 | 0.265 | 0.735 | 0.655 | +0.080 [+0.040, +0.120] | +17/-1 | 1.45e-04 |
| metaqa | holdout200 | proposed_v5_graphcentric | 0.855 | 0.265 | 0.945 | 0.720 | +0.225 [+0.170, +0.285] | +45/-0 | 5.68e-14 |
| stark | sub200 | proposed_v3 | 0.135 | 0.165 | 0.135 | 0.165 | -0.030 [-0.060, -0.005] | +1/-7 | 7.03e-02 |
| stark | sub200 | proposed_v5_graphcentric | 0.135 | 0.165 | 0.165 | 0.145 | +0.020 [-0.005, +0.050] | +6/-2 | 2.89e-01 |
| stark | holdout200 | proposed_v3 | 0.215 | 0.260 | 0.300 | 0.265 | +0.035 [+0.010, +0.065] | +8/-1 | 3.91e-02 |
| stark | holdout200 | proposed_v5_graphcentric | 0.215 | 0.260 | 0.280 | 0.225 | +0.055 [+0.025, +0.090] | +12/-1 | 3.42e-03 |
| graphragbench | sub200 | proposed_v3 | 0.520 | 0.470 | 0.615 | 0.525 | +0.090 [+0.045, +0.135] | +20/-2 | 1.21e-04 |
| graphragbench | sub200 | proposed_v5_graphcentric | 0.520 | 0.470 | 0.735 | 0.490 | +0.245 [+0.180, +0.305] | +51/-2 | 3.18e-13 |
| graphragbench | holdout200 | proposed_v3 | 0.505 | 0.535 | 0.650 | 0.545 | +0.105 [+0.065, +0.150] | +21/-0 | 9.54e-07 |
| graphragbench | holdout200 | proposed_v5_graphcentric | 0.505 | 0.535 | 0.735 | 0.485 | +0.250 [+0.185, +0.315] | +53/-3 | 8.14e-13 |

## Mean pooling (2026-09-20) vs [CLS] pooling

Mean pooling = `results/kg_swap_dense_bge_m3.json` (attention-mask mean pooling); [CLS] pooling = this run. Same questions, pool, input text, max_length 512, seeds, bootstrap and sign test; only the pooling differs.

| Benchmark | Split | System | Dense-only mean / [CLS] | Aligned mean / [CLS] | Independent mean / [CLS] | Swap mean | Swap [CLS] [95% CI] | p mean / [CLS] |
|---|---|---|---|---|---|---|---|---|
| 2wiki | sub200 | proposed_v3 | 0.130 / 0.470 | 0.410 / 0.625 | 0.305 / 0.430 | +0.105 | +0.195 [+0.135, +0.255] | 1.04e-04 / 2.78e-09 |
| 2wiki | sub200 | proposed_v5_graphcentric | 0.130 / 0.470 | 0.640 / 0.730 | 0.345 / 0.460 | +0.295 | +0.270 [+0.205, +0.335] | 1.11e-14 / 6.25e-14 |
| 2wiki | holdout200 | proposed_v3 | 0.125 / 0.450 | 0.335 / 0.550 | 0.255 / 0.400 | +0.080 | +0.150 [+0.085, +0.215] | 7.00e-03 / 2.39e-05 |
| 2wiki | holdout200 | proposed_v5_graphcentric | 0.125 / 0.450 | 0.585 / 0.720 | 0.345 / 0.470 | +0.240 | +0.250 [+0.190, +0.315] | 9.06e-14 / 1.65e-13 |
| metaqa | sub200 | proposed_v3 | 0.060 / 0.290 | 0.650 / 0.750 | 0.505 / 0.675 | +0.145 | +0.075 [+0.040, +0.110] | 8.96e-06 / 6.10e-05 |
| metaqa | sub200 | proposed_v5_graphcentric | 0.060 / 0.290 | 0.935 / 0.945 | 0.675 / 0.765 | +0.260 | +0.180 [+0.130, +0.235] | 4.44e-16 / 2.91e-11 |
| metaqa | holdout200 | proposed_v3 | 0.035 / 0.265 | 0.675 / 0.735 | 0.540 / 0.655 | +0.135 | +0.080 [+0.040, +0.120] | 1.43e-05 / 1.45e-04 |
| metaqa | holdout200 | proposed_v5_graphcentric | 0.035 / 0.265 | 0.925 / 0.945 | 0.645 / 0.720 | +0.280 | +0.225 [+0.170, +0.285] | 4.09e-16 / 5.68e-14 |
| stark | sub200 | proposed_v3 | 0.125 / 0.165 | 0.160 / 0.135 | 0.170 / 0.165 | -0.010 | -0.030 [-0.060, -0.005] | 7.74e-01 / 7.03e-02 |
| stark | sub200 | proposed_v5_graphcentric | 0.125 / 0.165 | 0.180 / 0.165 | 0.155 / 0.145 | +0.025 | +0.020 [-0.005, +0.050] | 2.67e-01 / 2.89e-01 |
| stark | holdout200 | proposed_v3 | 0.185 / 0.260 | 0.255 / 0.300 | 0.240 / 0.265 | +0.015 | +0.035 [+0.010, +0.065] | 5.81e-01 / 3.91e-02 |
| stark | holdout200 | proposed_v5_graphcentric | 0.185 / 0.260 | 0.250 / 0.280 | 0.200 / 0.225 | +0.050 | +0.055 [+0.025, +0.090] | 4.14e-02 / 3.42e-03 |
| graphragbench | sub200 | proposed_v3 | 0.395 / 0.470 | 0.605 / 0.615 | 0.480 / 0.525 | +0.125 | +0.090 [+0.045, +0.135] | 4.17e-07 / 1.21e-04 |
| graphragbench | sub200 | proposed_v5_graphcentric | 0.395 / 0.470 | 0.690 / 0.735 | 0.445 / 0.490 | +0.245 | +0.245 [+0.180, +0.305] | 4.62e-14 / 3.18e-13 |
| graphragbench | holdout200 | proposed_v3 | 0.460 / 0.535 | 0.630 / 0.650 | 0.530 / 0.545 | +0.100 | +0.105 [+0.065, +0.150] | 1.91e-06 / 9.54e-07 |
| graphragbench | holdout200 | proposed_v5_graphcentric | 0.460 / 0.535 | 0.715 / 0.735 | 0.500 / 0.485 | +0.215 | +0.250 [+0.185, +0.315] | 1.60e-11 / 8.14e-13 |
