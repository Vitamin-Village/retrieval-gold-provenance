# Gold-concentration dose experiment on 2WikiMultiHop

Aligned (silver) KG diluted with links to random non-gold passages; independent (OpenIE) KG concentrated by restricting its passage links to gold. Paired hit@5 difference versus the frozen OpenIE condition, 10,000-resample bootstrap CI, exact sign test.

| Split | System | Variant | Linked passages | Gold fraction | hit@5 | Silver | OpenIE | Δ vs OpenIE [95% CI] | +/- | p |
|---|---|---|---|---|---|---|---|---|---|---|
| sub200 | proposed_v3 | silver_dilute_g75 | 456 | 0.75 | 0.530 | 0.535 | 0.375 | +0.155 [+0.095, +0.215] | +36/-5 | 7.8e-07 |
| sub200 | proposed_v3 | silver_dilute_g50 | 684 | 0.50 | 0.530 | 0.535 | 0.375 | +0.155 [+0.100, +0.215] | +36/-5 | 7.8e-07 |
| sub200 | proposed_v3 | silver_dilute_g31 | 1103 | 0.31 | 0.505 | 0.535 | 0.375 | +0.130 [+0.075, +0.190] | +32/-6 | 2.4e-05 |
| sub200 | proposed_v3 | openie_goldonly | 384 | 1.00 | 0.485 | 0.535 | 0.375 | +0.110 [+0.070, +0.155] | +22/-0 | 4.8e-07 |
| sub200 | ppr | silver_dilute_g75 | 456 | 0.75 | 0.725 | 0.740 | 0.425 | +0.300 [+0.220, +0.380] | +74/-14 | 5.1e-11 |
| sub200 | ppr | silver_dilute_g50 | 684 | 0.50 | 0.670 | 0.740 | 0.425 | +0.245 [+0.160, +0.330] | +69/-20 | 1.8e-07 |
| sub200 | ppr | silver_dilute_g31 | 1103 | 0.31 | 0.395 | 0.740 | 0.425 | -0.030 [-0.120, +0.060] | +40/-46 | 5.9e-01 |
| sub200 | ppr | openie_goldonly | 384 | 1.00 | 0.640 | 0.740 | 0.425 | +0.215 [+0.160, +0.275] | +43/-0 | 2.3e-13 |
| sub200 | proposed_v5_graphcentric | silver_dilute_g75 | 456 | 0.75 | 0.750 | 0.750 | 0.395 | +0.355 [+0.285, +0.425] | +75/-4 | 5.2e-18 |
| sub200 | proposed_v5_graphcentric | silver_dilute_g50 | 684 | 0.50 | 0.735 | 0.750 | 0.395 | +0.340 [+0.270, +0.410] | +72/-4 | 3.6e-17 |
| sub200 | proposed_v5_graphcentric | silver_dilute_g31 | 1103 | 0.31 | 0.725 | 0.750 | 0.395 | +0.330 [+0.260, +0.400] | +71/-5 | 5.2e-16 |
| sub200 | proposed_v5_graphcentric | openie_goldonly | 384 | 1.00 | 0.695 | 0.750 | 0.395 | +0.300 [+0.240, +0.365] | +60/-0 | 1.7e-18 |
| holdout200 | proposed_v3 | silver_dilute_g75 | 456 | 0.75 | 0.480 | 0.480 | 0.365 | +0.115 [+0.055, +0.175] | +33/-10 | 6.1e-04 |
| holdout200 | proposed_v3 | silver_dilute_g50 | 684 | 0.50 | 0.490 | 0.480 | 0.365 | +0.125 [+0.060, +0.190] | +35/-10 | 2.5e-04 |
| holdout200 | proposed_v3 | silver_dilute_g31 | 1103 | 0.31 | 0.475 | 0.480 | 0.365 | +0.110 [+0.050, +0.170] | +31/-9 | 6.8e-04 |
| holdout200 | proposed_v3 | openie_goldonly | 382 | 1.00 | 0.485 | 0.480 | 0.365 | +0.120 [+0.080, +0.165] | +24/-0 | 1.2e-07 |
| holdout200 | ppr | silver_dilute_g75 | 456 | 0.75 | 0.730 | 0.740 | 0.430 | +0.300 [+0.220, +0.380] | +72/-12 | 1.4e-11 |
| holdout200 | ppr | silver_dilute_g50 | 684 | 0.50 | 0.660 | 0.740 | 0.430 | +0.230 [+0.150, +0.315] | +64/-18 | 3.3e-07 |
| holdout200 | ppr | silver_dilute_g31 | 1103 | 0.31 | 0.410 | 0.740 | 0.430 | -0.020 [-0.110, +0.070] | +41/-45 | 7.5e-01 |
| holdout200 | ppr | openie_goldonly | 382 | 1.00 | 0.665 | 0.740 | 0.430 | +0.235 [+0.180, +0.295] | +47/-0 | 1.4e-14 |
| holdout200 | proposed_v5_graphcentric | silver_dilute_g75 | 456 | 0.75 | 0.710 | 0.705 | 0.370 | +0.340 [+0.275, +0.410] | +69/-1 | 1.2e-19 |
| holdout200 | proposed_v5_graphcentric | silver_dilute_g50 | 684 | 0.50 | 0.700 | 0.705 | 0.370 | +0.330 [+0.260, +0.400] | +69/-3 | 2.6e-17 |
| holdout200 | proposed_v5_graphcentric | silver_dilute_g31 | 1103 | 0.31 | 0.680 | 0.705 | 0.370 | +0.310 [+0.245, +0.375] | +64/-2 | 6.0e-17 |
| holdout200 | proposed_v5_graphcentric | openie_goldonly | 382 | 1.00 | 0.650 | 0.705 | 0.370 | +0.280 [+0.220, +0.345] | +56/-0 | 2.8e-17 |
