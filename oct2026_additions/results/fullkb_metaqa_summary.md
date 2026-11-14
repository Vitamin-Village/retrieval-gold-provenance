# MetaQA full-KB (realistic structure) condition

kg_fullkb = every MetaQA KB triple whose movie is a corpus passage, node->passage index over all corpus movies (not induced from the evaluated questions). Paired hit@5 vs the frozen OpenIE condition.

| Split | System | Linked passages | Gold fraction | hit@5 full-KB | hit@5 induced (silver) | hit@5 OpenIE | BM25 | Δ full-KB vs OpenIE [95% CI] | +/- | p |
|---|---|---|---|---|---|---|---|---|---|---|
| sub200 | proposed_v3 | 1250 | 0.31 | 0.820 | 0.935 | 0.900 | 0.870 | -0.080 [-0.125, -0.035] | +3/-19 | 8.6e-04 |
| sub200 | ppr | 1250 | 0.31 | 0.995 | 1.000 | 0.935 | 0.870 | +0.060 [+0.025, +0.100] | +13/-1 | 1.8e-03 |
| sub200 | proposed_v5_graphcentric | 1250 | 0.31 | 0.845 | 0.950 | 0.865 | 0.870 | -0.020 [-0.060, +0.020] | +6/-10 | 4.5e-01 |
| holdout200 | proposed_v3 | 1250 | 0.31 | 0.805 | 0.920 | 0.875 | 0.855 | -0.070 [-0.115, -0.030] | +3/-17 | 2.6e-03 |
| holdout200 | ppr | 1250 | 0.31 | 1.000 | 1.000 | 0.930 | 0.855 | +0.070 [+0.035, +0.105] | +14/-0 | 1.2e-04 |
| holdout200 | proposed_v5_graphcentric | 1250 | 0.31 | 0.845 | 0.945 | 0.860 | 0.855 | -0.015 [-0.055, +0.025] | +7/-10 | 6.3e-01 |
