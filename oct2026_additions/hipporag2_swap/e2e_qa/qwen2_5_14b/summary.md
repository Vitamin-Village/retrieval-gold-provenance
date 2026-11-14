# End-to-end QA over frozen swap vectors: reader `qwen2.5:14b`

## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader

| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| 2wiki | holdout200 | hipporag2 | 200 | +0.275 | 0.370 | 0.310 | +0.060 [-0.010, +0.130] | 1.19e-01 | +0.032 [-0.021, +0.085] | +0.065 |
| 2wiki | sub200 | hipporag2 | 200 | +0.240 | 0.375 | 0.320 | +0.055 [-0.020, +0.130] | 2.00e-01 | +0.016 [-0.041, +0.075] | +0.035 |

## Mediation: EM Δ conditional on retrieval flip status

| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |
|---|---|---|---:|---:|---:|---:|
| 2wiki | holdout200 | hipporag2 | 60 | +0.350 | 135 | -0.059 |
| 2wiki | sub200 | hipporag2 | 58 | +0.414 | 132 | -0.068 |

## All cells

| bench | split | system | condition | n | hit@5 | EM | F1 | cover |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2wiki | holdout200 | hipporag2 | kg_openie | 200 | 0.615 | 0.310 | 0.571 | 0.350 |
| 2wiki | holdout200 | hipporag2 | kg_silver | 200 | 0.890 | 0.370 | 0.603 | 0.415 |
| 2wiki | holdout200 | none | closedbook | 200 | - | 0.055 | 0.292 | 0.075 |
| 2wiki | holdout200 | none | oracle | 200 | 1.000 | 0.685 | 0.834 | 0.750 |
| 2wiki | sub200 | hipporag2 | kg_openie | 200 | 0.645 | 0.320 | 0.598 | 0.375 |
| 2wiki | sub200 | hipporag2 | kg_silver | 200 | 0.885 | 0.375 | 0.614 | 0.410 |
| 2wiki | sub200 | none | closedbook | 200 | - | 0.040 | 0.280 | 0.045 |
| 2wiki | sub200 | none | oracle | 200 | 1.000 | 0.725 | 0.864 | 0.770 |
