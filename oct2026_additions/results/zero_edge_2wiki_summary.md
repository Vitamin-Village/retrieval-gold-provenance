# Zero-edge control on 2WikiMultiHop

Aligned (silver) KG with every edge deleted; nodes and their gold-derived passage links kept. Paired hit@5 difference versus the frozen OpenIE condition, 10,000-resample bootstrap CI, exact sign test.

| Split | System | hit@5 no edges | Silver (full) | OpenIE | BM25 | Δ vs OpenIE [95% CI] | +/- | p | silver − no-edges |
|---|---|---|---|---|---|---|---|---|---|
| sub200 | proposed_v3 | 0.505 | 0.535 | 0.375 | 0.365 | +0.130 [+0.075, +0.190] | +32/-6 | 2.4e-05 | +0.030 |
| sub200 | ppr | 0.025 | 0.740 | 0.425 | 0.365 | -0.400 [-0.475, -0.325] | +3/-83 | 2.7e-21 | +0.715 |
| sub200 | proposed_v5_graphcentric | 0.390 | 0.750 | 0.395 | 0.365 | -0.005 [-0.055, +0.045] | +13/-14 | 1.0e+00 | +0.360 |
| holdout200 | proposed_v3 | 0.440 | 0.480 | 0.365 | 0.355 | +0.075 [+0.015, +0.135] | +27/-12 | 2.4e-02 | +0.040 |
| holdout200 | ppr | 0.015 | 0.740 | 0.430 | 0.355 | -0.415 [-0.485, -0.345] | +2/-85 | 4.9e-23 | +0.725 |
| holdout200 | proposed_v5_graphcentric | 0.375 | 0.705 | 0.370 | 0.355 | +0.005 [-0.040, +0.045] | +10/-9 | 1.0e+00 | +0.330 |
