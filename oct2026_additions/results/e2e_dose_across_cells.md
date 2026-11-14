# Answer swap vs retrieval swap across 100 cells

All cells: slope 0.35, Spearman 0.69, Pearson 0.65

| benchmark | n | mean hit@5 swap | mean answer swap | slope | Spearman | mean headroom (gold - BM25) |
|---|---|---|---|---|---|---|
| 2WikiMultiHop | 32 | +0.242 | +0.121 | 0.51 | 0.76 | +0.470 |
| MetaQA | 24 | +0.130 | -0.001 | -0.01 | 0.14 | +0.021 |
| STaRK-Prime | 20 | +0.023 | +0.003 | 0.33 | 0.54 | +0.566 |
| GraphRAG-Bench | 24 | +0.228 | +0.053 | 0.25 | 0.56 | +0.089 |

Transfer (slope fitted on 2WikiMultiHop + GraphRAG-Bench):
- MetaQA: slope 0.40, predicted mean answer swap +0.052, observed -0.001
- STaRK-Prime: slope 0.40, predicted mean answer swap +0.009, observed +0.003

Headroom model (answer swap = g x hit swap x headroom): g 1.12, Spearman 0.78, R2 0.56 (hit swap alone: R2 0.40)
- 2WikiMultiHop: predicted +0.127, observed +0.121
- MetaQA: predicted +0.005, observed -0.001
- STaRK-Prime: predicted +0.014, observed +0.003
- GraphRAG-Bench: predicted +0.022, observed +0.053
