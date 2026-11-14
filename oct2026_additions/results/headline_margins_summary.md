# Headline margins: graph-over-BM25 advantage under the aligned vs the independent KG

Generated 2026-10-02T23:04:11Z by `analysis/headline_margins.py` from frozen per-question vectors; no retrieval or reader run. 804/804 cross-checks pass (frozen summaries, HippoRAG 2 summary, e2e summaries and judge files, Table 3 = `tables/tab_e2e_merged_v3.tex`, `tab:hipporag2more`, prose ranges).

Margin = graph hit@5 minus BM25 hit@5 on the same 200 questions. Verdict *beats* = margin > 0 and exact two-sided sign test on the per-question pairs p < 0.05 (uncorrected); *below* = margin < 0 at p < 0.05; *n.s.* otherwise. Holm column: Holm over the 30 cells of one arm. Cells: 4 benchmarks x 2 splits x {Hybrid, PPR, Hybrid-PPR} + HippoRAG 2 on 2WikiMultiHop, MetaQA, GraphRAG-Bench x 2 splits.

## (a) Retrieval verdicts

| | aligned KG | independent KG | flips beats -> not |
|---|---:|---:|---:|
| beats BM25 (p<0.05) | 24/30 | 5/30 | 19 |
| significantly below BM25 | 2 | 8 | beats -> below: 5 |
| beats BM25 (Holm) | 20/30 | 3/30 | 17 |

Reverse flips (not beats -> beats): 0 uncorrected, 0 Holm.

Flipped cells (uncorrected): 2WikiMultiHop / first / Hybrid; 2WikiMultiHop / first / PPR; 2WikiMultiHop / first / Hybrid-PPR; 2WikiMultiHop / held-out / Hybrid; 2WikiMultiHop / held-out / PPR; 2WikiMultiHop / held-out / Hybrid-PPR; MetaQA / first / Hybrid; MetaQA / first / Hybrid-PPR; MetaQA / first / HippoRAG 2; MetaQA / held-out / Hybrid; MetaQA / held-out / Hybrid-PPR; STaRK-Prime / held-out / Hybrid; STaRK-Prime / held-out / Hybrid-PPR; GraphRAG-Bench / first / PPR; GraphRAG-Bench / first / Hybrid-PPR; GraphRAG-Bench / first / HippoRAG 2; GraphRAG-Bench / held-out / Hybrid; GraphRAG-Bench / held-out / PPR; GraphRAG-Bench / held-out / Hybrid-PPR.

| cell | BM25 | aligned hit@5 | margin | p | verdict | indep. hit@5 | margin | p | verdict | flip |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| 2WikiMultiHop / first / Hybrid | 0.365 | 0.535 | +0.170 | 1.4e-07 | beats | 0.375 | +0.010 | 0.83 | n.s. | yes |
| 2WikiMultiHop / first / PPR | 0.365 | 0.740 | +0.375 | 1.6e-14 | beats | 0.425 | +0.060 | 0.2 | n.s. | yes |
| 2WikiMultiHop / first / Hybrid-PPR | 0.365 | 0.750 | +0.385 | 2.0e-20 | beats | 0.395 | +0.030 | 0.34 | n.s. | yes |
| 2WikiMultiHop / first / HippoRAG 2 | 0.365 | 0.885 | +0.520 | 3.4e-28 | beats | 0.645 | +0.280 | 2.3e-12 | beats |  |
| 2WikiMultiHop / held-out / Hybrid | 0.355 | 0.480 | +0.125 | 1.7e-04 | beats | 0.365 | +0.010 | 0.83 | n.s. | yes |
| 2WikiMultiHop / held-out / PPR | 0.355 | 0.740 | +0.385 | 4.3e-14 | beats | 0.430 | +0.075 | 0.11 | n.s. | yes |
| 2WikiMultiHop / held-out / Hybrid-PPR | 0.355 | 0.705 | +0.350 | 3.1e-20 | beats | 0.370 | +0.015 | 0.69 | n.s. | yes |
| 2WikiMultiHop / held-out / HippoRAG 2 | 0.355 | 0.890 | +0.535 | 3.5e-28 | beats | 0.615 | +0.260 | 5.3e-10 | beats |  |
| MetaQA / first / Hybrid | 0.870 | 0.935 | +0.065 | 9.8e-04 | beats | 0.900 | +0.030 | 0.07 | n.s. | yes |
| MetaQA / first / PPR | 0.870 | 1.000 | +0.130 | 3.0e-08 | beats | 0.935 | +0.065 | 0.0023 | beats |  |
| MetaQA / first / Hybrid-PPR | 0.870 | 0.950 | +0.080 | 3.1e-05 | beats | 0.865 | -0.005 | 1 | n.s. | yes |
| MetaQA / first / HippoRAG 2 | 0.870 | 0.935 | +0.065 | 0.029 | beats | 0.755 | -0.115 | 0.0014 | below | yes |
| MetaQA / held-out / Hybrid | 0.855 | 0.920 | +0.065 | 2.4e-04 | beats | 0.875 | +0.020 | 0.42 | n.s. | yes |
| MetaQA / held-out / PPR | 0.855 | 1.000 | +0.145 | 3.7e-09 | beats | 0.930 | +0.075 | 0.0041 | beats |  |
| MetaQA / held-out / Hybrid-PPR | 0.855 | 0.945 | +0.090 | 7.6e-06 | beats | 0.860 | +0.005 | 1 | n.s. | yes |
| MetaQA / held-out / HippoRAG 2 | 0.855 | 0.915 | +0.060 | 0.073 | n.s. | 0.745 | -0.110 | 0.0021 | below |  |
| STaRK-Prime / first / Hybrid | 0.135 | 0.130 | -0.005 | 1 | n.s. | 0.145 | +0.010 | 0.69 | n.s. |  |
| STaRK-Prime / first / PPR | 0.135 | 0.065 | -0.070 | 0.0094 | below | 0.040 | -0.095 | 3.8e-06 | below |  |
| STaRK-Prime / first / Hybrid-PPR | 0.135 | 0.125 | -0.010 | 0.77 | n.s. | 0.100 | -0.035 | 0.092 | n.s. |  |
| STaRK-Prime / held-out / Hybrid | 0.215 | 0.275 | +0.060 | 0.0075 | beats | 0.250 | +0.035 | 0.065 | n.s. | yes |
| STaRK-Prime / held-out / PPR | 0.215 | 0.120 | -0.095 | 0.0019 | below | 0.100 | -0.115 | 1.5e-06 | below |  |
| STaRK-Prime / held-out / Hybrid-PPR | 0.215 | 0.270 | +0.055 | 0.027 | beats | 0.205 | -0.010 | 0.79 | n.s. | yes |
| GraphRAG-Bench / first / Hybrid | 0.520 | 0.565 | +0.045 | 0.078 | n.s. | 0.500 | -0.020 | 0.5 | n.s. |  |
| GraphRAG-Bench / first / PPR | 0.520 | 0.690 | +0.170 | 2.7e-04 | beats | 0.200 | -0.320 | 1.6e-17 | below | yes |
| GraphRAG-Bench / first / Hybrid-PPR | 0.520 | 0.695 | +0.175 | 2.8e-09 | beats | 0.405 | -0.115 | 6.6e-05 | below | yes |
| GraphRAG-Bench / first / HippoRAG 2 | 0.520 | 0.805 | +0.285 | 1.3e-12 | beats | 0.540 | +0.020 | 0.64 | n.s. | yes |
| GraphRAG-Bench / held-out / Hybrid | 0.505 | 0.580 | +0.075 | 7.3e-04 | beats | 0.520 | +0.015 | 0.63 | n.s. | yes |
| GraphRAG-Bench / held-out / PPR | 0.505 | 0.625 | +0.120 | 0.013 | beats | 0.245 | -0.260 | 2.4e-11 | below | yes |
| GraphRAG-Bench / held-out / Hybrid-PPR | 0.505 | 0.680 | +0.175 | 2.8e-09 | beats | 0.400 | -0.105 | 5.1e-04 | below | yes |
| GraphRAG-Bench / held-out / HippoRAG 2 | 0.505 | 0.800 | +0.295 | 2.5e-15 | beats | 0.575 | +0.070 | 0.024 | beats |  |

## (a2) Realistic structures not induced from the evaluated questions (live Wikidata KG on 2WikiMultiHop, full MetaQA KB)

On these 16 cells graph retrieval beats BM25 in 15 with the aligned KG, 6 with the realistic structure (6 under Holm over these cells; 2 significantly below), and 4 with the independent KG.

| cell | structure | hit@5 | margin over BM25 | p | verdict | minus independent | p |
|---|---|---:|---:|---:|---|---:|---:|
| 2WikiMultiHop / first / Hybrid | live Wikidata KG | 0.330 | -0.035 | 0.35 | n.s. | -0.045 | 0.16 |
| 2WikiMultiHop / first / PPR | live Wikidata KG | 0.500 | +0.135 | 0.0055 | beats | +0.075 | 0.14 |
| 2WikiMultiHop / first / Hybrid-PPR | live Wikidata KG | 0.380 | +0.015 | 0.77 | n.s. | -0.015 | 0.75 |
| 2WikiMultiHop / first / HippoRAG 2 | live Wikidata KG | 0.560 | +0.195 | 8.1e-08 | beats | -0.085 | 0.019 |
| 2WikiMultiHop / held-out / Hybrid | live Wikidata KG | 0.380 | +0.025 | 0.51 | n.s. | +0.015 | 0.74 |
| 2WikiMultiHop / held-out / PPR | live Wikidata KG | 0.535 | +0.180 | 8.7e-05 | beats | +0.105 | 0.038 |
| 2WikiMultiHop / held-out / Hybrid-PPR | live Wikidata KG | 0.390 | +0.035 | 0.31 | n.s. | +0.020 | 0.6 |
| 2WikiMultiHop / held-out / HippoRAG 2 | live Wikidata KG | 0.580 | +0.225 | 6.1e-09 | beats | -0.035 | 0.4 |
| MetaQA / first / Hybrid | full MetaQA KB | 0.820 | -0.050 | 0.052 | n.s. | -0.080 | 8.6e-04 |
| MetaQA / first / PPR | full MetaQA KB | 0.995 | +0.125 | 4.2e-07 | beats | +0.060 | 0.0018 |
| MetaQA / first / Hybrid-PPR | full MetaQA KB | 0.845 | -0.025 | 0.33 | n.s. | -0.020 | 0.45 |
| MetaQA / first / HippoRAG 2 | full MetaQA KB | 0.695 | -0.175 | 6.9e-07 | below | -0.060 | 0.14 |
| MetaQA / held-out / Hybrid | full MetaQA KB | 0.805 | -0.050 | 0.064 | n.s. | -0.070 | 0.0026 |
| MetaQA / held-out / PPR | full MetaQA KB | 1.000 | +0.145 | 3.7e-09 | beats | +0.070 | 1.2e-04 |
| MetaQA / held-out / Hybrid-PPR | full MetaQA KB | 0.845 | -0.010 | 0.81 | n.s. | -0.015 | 0.63 |
| MetaQA / held-out / HippoRAG 2 | full MetaQA KB | 0.590 | -0.265 | 1.6e-10 | below | -0.155 | 8.8e-05 |

## (b) Kendall tau-b between retriever rankings, aligned vs independent KG (retrieval hit@5)

Items: BM25 and BGE-M3 dense-only ([CLS] pooling, `results/kg_swap_dense_bge_m3_cls.json`; KG-independent, identical in both arms) plus the graph retrievers run on that benchmark. tau-b handles ties. The second tau drops dense-only.

| benchmark | split | tau-b | tau-b w/o dense | ranking, aligned KG | ranking, independent KG |
|---|---|---:|---:|---|---|
| 2WikiMultiHop | first | +0.47 | +0.80 | HippoRAG 2 > Hybrid-PPR > PPR > Hybrid > Dense-only > BM25 | HippoRAG 2 > Dense-only > PPR > Hybrid-PPR > Hybrid > BM25 |
| 2WikiMultiHop | held-out | +0.60 | +1.00 | HippoRAG 2 > PPR > Hybrid-PPR > Hybrid > Dense-only > BM25 | HippoRAG 2 > Dense-only > PPR > Hybrid-PPR > Hybrid > BM25 |
| MetaQA | first | +0.55 | +0.32 | PPR > Hybrid-PPR > Hybrid = HippoRAG 2 > BM25 > Dense-only | PPR > Hybrid > BM25 > Hybrid-PPR > HippoRAG 2 > Dense-only |
| MetaQA | held-out | +0.73 | +0.60 | PPR > Hybrid-PPR > Hybrid > HippoRAG 2 > BM25 > Dense-only | PPR > Hybrid > Hybrid-PPR > BM25 > HippoRAG 2 > Dense-only |
| STaRK-Prime | first | +0.80 | +0.67 | Dense-only > BM25 > Hybrid > Hybrid-PPR > PPR | Dense-only > Hybrid > BM25 > Hybrid-PPR > PPR |
| STaRK-Prime | held-out | +0.40 | +0.67 | Hybrid > Hybrid-PPR > Dense-only > BM25 > PPR | Dense-only > Hybrid > BM25 > Hybrid-PPR > PPR |
| GraphRAG-Bench | first | +0.07 | +0.00 | HippoRAG 2 > Hybrid-PPR > PPR > Hybrid > BM25 > Dense-only | HippoRAG 2 > BM25 > Hybrid > Dense-only > Hybrid-PPR > PPR |
| GraphRAG-Bench | held-out | +0.07 | +0.20 | HippoRAG 2 > Hybrid-PPR > PPR > Hybrid > Dense-only > BM25 | HippoRAG 2 > Dense-only > Hybrid > BM25 > Hybrid-PPR > PPR |

## (c) Answer level (alias-aware EM; Mistral-Small-24B judge accuracy on GraphRAG-Bench)

### Qwen2.5-14B

beats BM25 (p<0.05): aligned 9/30, independent 4/30; flips beats -> not: 6 (reverse: 1); significantly below BM25: aligned 1, independent 5. Holm: aligned 6, independent 3, flips 4.

Flipped cells: 2WikiMultiHop / first / Hybrid; 2WikiMultiHop / first / PPR; 2WikiMultiHop / first / Hybrid-PPR; 2WikiMultiHop / held-out / Hybrid; 2WikiMultiHop / held-out / PPR; 2WikiMultiHop / held-out / Hybrid-PPR.

| cell | BM25 | aligned | margin | p | verdict | indep. | margin | p | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 2WikiMultiHop / first / Hybrid | 0.200 | 0.265 | +0.065 | 0.047 | beats | 0.220 | +0.020 | 0.58 | n.s. |
| 2WikiMultiHop / first / PPR | 0.200 | 0.560 | +0.360 | 4.0e-14 | beats | 0.215 | +0.015 | 0.78 | n.s. |
| 2WikiMultiHop / first / Hybrid-PPR | 0.200 | 0.365 | +0.165 | 1.9e-05 | beats | 0.200 | +0.000 | 1 | n.s. |
| 2WikiMultiHop / first / HippoRAG 2 | 0.200 | 0.375 | +0.175 | 1.6e-05 | beats | 0.320 | +0.120 | 9.4e-04 | beats |
| 2WikiMultiHop / held-out / Hybrid | 0.185 | 0.285 | +0.100 | 0.0029 | beats | 0.220 | +0.035 | 0.21 | n.s. |
| 2WikiMultiHop / held-out / PPR | 0.185 | 0.555 | +0.370 | 2.5e-15 | beats | 0.255 | +0.070 | 0.059 | n.s. |
| 2WikiMultiHop / held-out / Hybrid-PPR | 0.185 | 0.400 | +0.215 | 1.8e-08 | beats | 0.230 | +0.045 | 0.12 | n.s. |
| 2WikiMultiHop / held-out / HippoRAG 2 | 0.185 | 0.370 | +0.185 | 6.5e-06 | beats | 0.310 | +0.125 | 2.5e-04 | beats |
| MetaQA / first / Hybrid | 0.380 | 0.335 | -0.045 | 0.064 | n.s. | 0.350 | -0.030 | 0.18 | n.s. |
| MetaQA / first / PPR | 0.380 | 0.350 | -0.030 | 0.36 | n.s. | 0.340 | -0.040 | 0.17 | n.s. |
| MetaQA / first / Hybrid-PPR | 0.380 | 0.350 | -0.030 | 0.21 | n.s. | 0.360 | -0.020 | 0.54 | n.s. |
| MetaQA / first / HippoRAG 2 | 0.380 | 0.330 | -0.050 | 0.099 | n.s. | 0.280 | -0.100 | 0.0029 | below |
| MetaQA / held-out / Hybrid | 0.295 | 0.290 | -0.005 | 1 | n.s. | 0.300 | +0.005 | 1 | n.s. |
| MetaQA / held-out / PPR | 0.295 | 0.280 | -0.015 | 0.74 | n.s. | 0.295 | +0.000 | 1 | n.s. |
| MetaQA / held-out / Hybrid-PPR | 0.295 | 0.300 | +0.005 | 1 | n.s. | 0.285 | -0.010 | 0.85 | n.s. |
| MetaQA / held-out / HippoRAG 2 | 0.295 | 0.270 | -0.025 | 0.5 | n.s. | 0.260 | -0.035 | 0.35 | n.s. |
| STaRK-Prime / first / Hybrid | 0.115 | 0.135 | +0.020 | 0.48 | n.s. | 0.135 | +0.020 | 0.42 | n.s. |
| STaRK-Prime / first / PPR | 0.115 | 0.060 | -0.055 | 0.027 | below | 0.060 | -0.055 | 0.013 | below |
| STaRK-Prime / first / Hybrid-PPR | 0.115 | 0.135 | +0.020 | 0.5 | n.s. | 0.105 | -0.010 | 0.81 | n.s. |
| STaRK-Prime / held-out / Hybrid | 0.185 | 0.230 | +0.045 | 0.064 | n.s. | 0.240 | +0.055 | 9.8e-04 | beats |
| STaRK-Prime / held-out / PPR | 0.185 | 0.130 | -0.055 | 0.071 | n.s. | 0.110 | -0.075 | 0.011 | below |
| STaRK-Prime / held-out / Hybrid-PPR | 0.185 | 0.225 | +0.040 | 0.12 | n.s. | 0.195 | +0.010 | 0.8 | n.s. |
| GraphRAG-Bench / first / Hybrid | 0.710 | 0.735 | +0.025 | 0.27 | n.s. | 0.720 | +0.010 | 0.81 | n.s. |
| GraphRAG-Bench / first / PPR | 0.710 | 0.675 | -0.035 | 0.39 | n.s. | 0.455 | -0.255 | 1.0e-10 | below |
| GraphRAG-Bench / first / Hybrid-PPR | 0.710 | 0.740 | +0.030 | 0.36 | n.s. | 0.700 | -0.010 | 0.86 | n.s. |
| GraphRAG-Bench / first / HippoRAG 2 | 0.710 | 0.765 | +0.055 | 0.061 | n.s. | 0.725 | +0.015 | 0.74 | n.s. |
| GraphRAG-Bench / held-out / Hybrid | 0.660 | 0.690 | +0.030 | 0.41 | n.s. | 0.680 | +0.020 | 0.6 | n.s. |
| GraphRAG-Bench / held-out / PPR | 0.660 | 0.585 | -0.075 | 0.072 | n.s. | 0.485 | -0.175 | 5.1e-05 | below |
| GraphRAG-Bench / held-out / Hybrid-PPR | 0.660 | 0.725 | +0.065 | 0.053 | n.s. | 0.660 | +0.000 | 1 | n.s. |
| GraphRAG-Bench / held-out / HippoRAG 2 | 0.660 | 0.760 | +0.100 | 0.0091 | beats | 0.750 | +0.090 | 0.015 | beats |

| benchmark | split | tau-b | tau-b w/o dense | ranking, aligned KG | ranking, independent KG |
|---|---|---:|---:|---|---|
| 2WikiMultiHop | first | +0.14 | +0.32 | PPR > HippoRAG 2 > Hybrid-PPR > Hybrid > Dense-only > BM25 | HippoRAG 2 > Dense-only > Hybrid > PPR > BM25 = Hybrid-PPR |
| 2WikiMultiHop | held-out | +0.33 | +0.60 | PPR > Hybrid-PPR > HippoRAG 2 > Hybrid > Dense-only > BM25 | HippoRAG 2 > Dense-only > PPR > Hybrid-PPR > Hybrid > BM25 |
| MetaQA | first | +0.79 | +0.74 | BM25 > PPR = Hybrid-PPR > Hybrid > Dense-only = HippoRAG 2 | BM25 > Hybrid-PPR > Hybrid > PPR > Dense-only > HippoRAG 2 |
| MetaQA | held-out | +0.41 | +0.11 | Dense-only > Hybrid-PPR > BM25 > Hybrid > PPR > HippoRAG 2 | Dense-only > Hybrid > BM25 = PPR > Hybrid-PPR > HippoRAG 2 |
| STaRK-Prime | first | +0.63 | +0.55 | Dense-only = Hybrid = Hybrid-PPR > BM25 > PPR | Dense-only = Hybrid > BM25 > Hybrid-PPR > PPR |
| STaRK-Prime | held-out | +0.80 | +1.00 | Hybrid > Hybrid-PPR > Dense-only > BM25 > PPR | Hybrid > Dense-only > Hybrid-PPR > BM25 > PPR |
| GraphRAG-Bench | first | +0.60 | +0.60 | HippoRAG 2 > Hybrid-PPR > Hybrid > BM25 > Dense-only > PPR | HippoRAG 2 > Hybrid > BM25 > Dense-only > Hybrid-PPR > PPR |
| GraphRAG-Bench | held-out | +0.69 | +0.74 | HippoRAG 2 > Hybrid-PPR > Dense-only > Hybrid > BM25 > PPR | HippoRAG 2 > Dense-only > Hybrid > BM25 = Hybrid-PPR > PPR |

### Llama-3.1-8B

beats BM25 (p<0.05): aligned 12/30, independent 5/30; flips beats -> not: 8 (reverse: 1); significantly below BM25: aligned 2, independent 4. Holm: aligned 6, independent 1, flips 5.

Flipped cells: 2WikiMultiHop / first / Hybrid; 2WikiMultiHop / first / PPR; 2WikiMultiHop / first / Hybrid-PPR; 2WikiMultiHop / held-out / Hybrid; 2WikiMultiHop / held-out / Hybrid-PPR; MetaQA / first / PPR; GraphRAG-Bench / first / HippoRAG 2; GraphRAG-Bench / held-out / Hybrid-PPR.

| cell | BM25 | aligned | margin | p | verdict | indep. | margin | p | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 2WikiMultiHop / first / Hybrid | 0.100 | 0.180 | +0.080 | 0.0037 | beats | 0.130 | +0.030 | 0.26 | n.s. |
| 2WikiMultiHop / first / PPR | 0.100 | 0.475 | +0.375 | 5.8e-16 | beats | 0.145 | +0.045 | 0.21 | n.s. |
| 2WikiMultiHop / first / Hybrid-PPR | 0.100 | 0.240 | +0.140 | 2.5e-05 | beats | 0.100 | +0.000 | 1 | n.s. |
| 2WikiMultiHop / first / HippoRAG 2 | 0.100 | 0.255 | +0.155 | 4.7e-05 | beats | 0.180 | +0.080 | 0.014 | beats |
| 2WikiMultiHop / held-out / Hybrid | 0.145 | 0.215 | +0.070 | 0.016 | beats | 0.155 | +0.010 | 0.82 | n.s. |
| 2WikiMultiHop / held-out / PPR | 0.145 | 0.490 | +0.345 | 9.4e-16 | beats | 0.230 | +0.085 | 0.019 | beats |
| 2WikiMultiHop / held-out / Hybrid-PPR | 0.145 | 0.265 | +0.120 | 3.9e-04 | beats | 0.165 | +0.020 | 0.52 | n.s. |
| 2WikiMultiHop / held-out / HippoRAG 2 | 0.145 | 0.285 | +0.140 | 2.3e-04 | beats | 0.265 | +0.120 | 1.2e-04 | beats |
| MetaQA / first / Hybrid | 0.430 | 0.470 | +0.040 | 0.15 | n.s. | 0.500 | +0.070 | 0.0043 | beats |
| MetaQA / first / PPR | 0.430 | 0.515 | +0.085 | 0.0076 | beats | 0.455 | +0.025 | 0.44 | n.s. |
| MetaQA / first / Hybrid-PPR | 0.430 | 0.465 | +0.035 | 0.25 | n.s. | 0.445 | +0.015 | 0.73 | n.s. |
| MetaQA / first / HippoRAG 2 | 0.430 | 0.450 | +0.020 | 0.64 | n.s. | 0.395 | -0.035 | 0.36 | n.s. |
| MetaQA / held-out / Hybrid | 0.435 | 0.435 | +0.000 | 1 | n.s. | 0.450 | +0.015 | 0.71 | n.s. |
| MetaQA / held-out / PPR | 0.435 | 0.495 | +0.060 | 0.081 | n.s. | 0.485 | +0.050 | 0.15 | n.s. |
| MetaQA / held-out / Hybrid-PPR | 0.435 | 0.450 | +0.015 | 0.68 | n.s. | 0.425 | -0.010 | 0.86 | n.s. |
| MetaQA / held-out / HippoRAG 2 | 0.435 | 0.445 | +0.010 | 0.86 | n.s. | 0.385 | -0.050 | 0.16 | n.s. |
| STaRK-Prime / first / Hybrid | 0.090 | 0.080 | -0.010 | 0.75 | n.s. | 0.110 | +0.020 | 0.34 | n.s. |
| STaRK-Prime / first / PPR | 0.090 | 0.050 | -0.040 | 0.13 | n.s. | 0.050 | -0.040 | 0.021 | below |
| STaRK-Prime / first / Hybrid-PPR | 0.090 | 0.065 | -0.025 | 0.12 | n.s. | 0.075 | -0.015 | 0.51 | n.s. |
| STaRK-Prime / held-out / Hybrid | 0.165 | 0.175 | +0.010 | 0.82 | n.s. | 0.190 | +0.025 | 0.33 | n.s. |
| STaRK-Prime / held-out / PPR | 0.165 | 0.105 | -0.060 | 0.029 | below | 0.085 | -0.080 | 0.0015 | below |
| STaRK-Prime / held-out / Hybrid-PPR | 0.165 | 0.170 | +0.005 | 1 | n.s. | 0.160 | -0.005 | 1 | n.s. |
| GraphRAG-Bench / first / Hybrid | 0.640 | 0.690 | +0.050 | 0.099 | n.s. | 0.640 | +0.000 | 1 | n.s. |
| GraphRAG-Bench / first / PPR | 0.640 | 0.585 | -0.055 | 0.2 | n.s. | 0.410 | -0.230 | 6.2e-08 | below |
| GraphRAG-Bench / first / Hybrid-PPR | 0.640 | 0.680 | +0.040 | 0.24 | n.s. | 0.590 | -0.050 | 0.18 | n.s. |
| GraphRAG-Bench / first / HippoRAG 2 | 0.640 | 0.715 | +0.075 | 0.024 | beats | 0.705 | +0.065 | 0.066 | n.s. |
| GraphRAG-Bench / held-out / Hybrid | 0.560 | 0.575 | +0.015 | 0.72 | n.s. | 0.570 | +0.010 | 0.85 | n.s. |
| GraphRAG-Bench / held-out / PPR | 0.560 | 0.465 | -0.095 | 0.027 | below | 0.420 | -0.140 | 0.002 | below |
| GraphRAG-Bench / held-out / Hybrid-PPR | 0.560 | 0.650 | +0.090 | 0.011 | beats | 0.580 | +0.020 | 0.65 | n.s. |
| GraphRAG-Bench / held-out / HippoRAG 2 | 0.560 | 0.665 | +0.105 | 0.0065 | beats | 0.675 | +0.115 | 0.0032 | beats |

| benchmark | split | tau-b | tau-b w/o dense | ranking, aligned KG | ranking, independent KG |
|---|---|---:|---:|---|---|
| 2WikiMultiHop | first | +0.28 | +0.53 | PPR > HippoRAG 2 > Hybrid-PPR > Hybrid > Dense-only > BM25 | HippoRAG 2 > Dense-only > PPR > Hybrid > BM25 = Hybrid-PPR |
| 2WikiMultiHop | held-out | +0.69 | +0.80 | PPR > HippoRAG 2 > Hybrid-PPR > Hybrid > Dense-only > BM25 | HippoRAG 2 > PPR > Dense-only = Hybrid-PPR > Hybrid > BM25 |
| MetaQA | first | +0.73 | +0.60 | PPR > Hybrid > Hybrid-PPR > HippoRAG 2 > BM25 > Dense-only | Hybrid > PPR > Hybrid-PPR > BM25 > HippoRAG 2 > Dense-only |
| MetaQA | held-out | +0.41 | +0.11 | PPR > Hybrid-PPR > HippoRAG 2 > BM25 = Hybrid > Dense-only | PPR > Hybrid > BM25 > Hybrid-PPR > HippoRAG 2 > Dense-only |
| STaRK-Prime | first | +0.80 | +0.67 | Dense-only > BM25 > Hybrid > Hybrid-PPR > PPR | Dense-only > Hybrid > BM25 > Hybrid-PPR > PPR |
| STaRK-Prime | held-out | +0.74 | +0.67 | Hybrid > Hybrid-PPR > BM25 > Dense-only > PPR | Hybrid > BM25 > Dense-only = Hybrid-PPR > PPR |
| GraphRAG-Bench | first | +0.55 | +0.74 | HippoRAG 2 > Hybrid > Hybrid-PPR > Dense-only > BM25 > PPR | HippoRAG 2 > Dense-only > BM25 = Hybrid > Hybrid-PPR > PPR |
| GraphRAG-Bench | held-out | +1.00 | +1.00 | HippoRAG 2 > Dense-only > Hybrid-PPR > Hybrid > BM25 > PPR | HippoRAG 2 > Dense-only > Hybrid-PPR > Hybrid > BM25 > PPR |

## Caption draft

Graph-over-BM25 hit@5 margin of each graph retriever with the gold-aligned KG (circles) and with an independent OpenIE KG (squares), on the first split (large) and the held-out split (small); filled markers beat BM25 and crossed markers fall below it by an exact paired sign test ($p<0.05$). With the aligned KG 24 of 30 benchmark--split--retriever cells beat BM25; with the independent KG 5 do, and 19 verdicts flip (Holm over the 30 cells of an arm: 20, 3, and 17).

Figure: `figures/fig_headline_margins.pdf` (vector, TrueType fonts) and `.png` preview; 3.04 in (\columnwidth) wide.
