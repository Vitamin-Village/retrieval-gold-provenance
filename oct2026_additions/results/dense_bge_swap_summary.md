# KG swap with a real dense channel (BGE-M3)

| Benchmark | Split | System | BM25 | Dense-only | Graph, aligned KG | Graph, independent KG | Swap [95% CI] | +/- | p |
|---|---|---|---|---|---|---|---|---|---|
| 2wiki | sub200 | proposed_v3 | 0.365 | 0.130 | 0.410 | 0.305 | +0.105 [+0.055, +0.155] | +25/-4 | 1.04e-04 |
| 2wiki | sub200 | proposed_v5_graphcentric | 0.365 | 0.130 | 0.640 | 0.345 | +0.295 [+0.225, +0.365] | +63/-4 | 1.11e-14 |
| 2wiki | holdout200 | proposed_v3 | 0.355 | 0.125 | 0.335 | 0.255 | +0.080 [+0.025, +0.135] | +24/-8 | 7.00e-03 |
| 2wiki | holdout200 | proposed_v5_graphcentric | 0.355 | 0.125 | 0.585 | 0.345 | +0.240 [+0.180, +0.300] | +49/-1 | 9.06e-14 |
| metaqa | sub200 | proposed_v3 | 0.870 | 0.060 | 0.650 | 0.505 | +0.145 [+0.085, +0.205] | +36/-7 | 8.96e-06 |
| metaqa | sub200 | proposed_v5_graphcentric | 0.870 | 0.060 | 0.935 | 0.675 | +0.260 [+0.200, +0.320] | +52/-0 | 4.44e-16 |
| metaqa | holdout200 | proposed_v3 | 0.855 | 0.035 | 0.675 | 0.540 | +0.135 [+0.080, +0.195] | +33/-6 | 1.43e-05 |
| metaqa | holdout200 | proposed_v5_graphcentric | 0.855 | 0.035 | 0.925 | 0.645 | +0.280 [+0.215, +0.345] | +57/-1 | 4.09e-16 |
| stark | sub200 | proposed_v3 | 0.135 | 0.125 | 0.160 | 0.170 | -0.010 [-0.045, +0.025] | +5/-7 | 7.74e-01 |
| stark | sub200 | proposed_v5_graphcentric | 0.135 | 0.125 | 0.180 | 0.155 | +0.025 [-0.010, +0.060] | +9/-4 | 2.67e-01 |
| stark | holdout200 | proposed_v3 | 0.215 | 0.185 | 0.255 | 0.240 | +0.015 [-0.020, +0.050] | +8/-5 | 5.81e-01 |
| stark | holdout200 | proposed_v5_graphcentric | 0.215 | 0.185 | 0.250 | 0.200 | +0.050 [+0.010, +0.095] | +15/-5 | 4.14e-02 |
| graphragbench | sub200 | proposed_v3 | 0.520 | 0.395 | 0.605 | 0.480 | +0.125 [+0.080, +0.175] | +26/-1 | 4.17e-07 |
| graphragbench | sub200 | proposed_v5_graphcentric | 0.520 | 0.395 | 0.690 | 0.445 | +0.245 [+0.185, +0.305] | +50/-1 | 4.62e-14 |
| graphragbench | holdout200 | proposed_v3 | 0.505 | 0.460 | 0.630 | 0.530 | +0.100 [+0.060, +0.145] | +20/-0 | 1.91e-06 |
| graphragbench | holdout200 | proposed_v5_graphcentric | 0.505 | 0.460 | 0.715 | 0.500 | +0.215 [+0.155, +0.280] | +45/-2 | 1.60e-11 |
