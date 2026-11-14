# Answer swap vs retrieval swap across 100 cells

All cells: slope 0.36, Spearman 0.75, Pearson 0.69

| benchmark | n | mean hit@5 swap | mean answer swap | slope | Spearman | mean headroom (gold - BM25) |
|---|---|---|---|---|---|---|
| 2WikiMultiHop | 32 | +0.251 | +0.123 | 0.50 | 0.64 | +0.470 |
| MetaQA | 24 | +0.108 | +0.004 | 0.05 | 0.24 | +0.021 |
| STaRK-Prime | 20 | +0.023 | +0.000 | 0.26 | 0.48 | +0.566 |
| GraphRAG-Bench | 24 | +0.229 | +0.049 | 0.24 | 0.69 | +0.089 |

Transfer (slope fitted on 2WikiMultiHop + GraphRAG-Bench):
- MetaQA: slope 0.39, predicted mean answer swap +0.042, observed +0.004
- STaRK-Prime: slope 0.39, predicted mean answer swap +0.009, observed +0.000

Headroom model (answer swap = g x hit swap x headroom): g 1.10, Spearman 0.78, R2 0.56 (hit swap alone: R2 0.45)
- 2WikiMultiHop: predicted +0.129, observed +0.123
- MetaQA: predicted +0.004, observed +0.004
- STaRK-Prime: predicted +0.014, observed +0.000
- GraphRAG-Bench: predicted +0.022, observed +0.049
