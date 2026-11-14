# MetaQA KG-Swap Summary (Second Causal Experiment)

Generated 2026-06-13. Mirrors the 2WikiMultiHop swap protocol (paper/eval/_metaqa_swap_design.md). MetaQA census C_K = 1.0: every question is generated from the movie KB, so kg_silver (the KB subgraph the gold derives from) is the circular condition and kg_openie (llama3.1:8b OpenIE over the passage corpus, no KB/gold access) is the independent condition. Retriever configurations frozen as released; top_k=5; RAG_V3_DENSE_BACKEND=deterministic; bootstrap B=10,000, seed 20260613.

## Swap results (hit@5; both gold bridge-movie docs in top-5)

| Split | System | Silver graph | Silver BM25 | Delta(silver) | OpenIE graph | OpenIE BM25 | Delta(openie) | Swap | Paired mean [95% CI] | Discordant +/- | Sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA-sub200 | proposed_v3 | 0.935 | 0.870 | +0.065 | 0.900 | 0.870 | +0.030 | +0.035 | +0.035 [+0.005, +0.070] | +9 / -2 | 6.54e-02 |
| MetaQA-holdout200 | proposed_v3 | 0.920 | 0.855 | +0.065 | 0.875 | 0.855 | +0.020 | +0.045 | +0.045 [+0.020, +0.075] | +9 / -0 | 3.91e-03 |
| MetaQA-sub200 | ppr | 1.000 | 0.870 | +0.130 | 0.935 | 0.870 | +0.065 | +0.065 | +0.065 [+0.035, +0.100] | +13 / -0 | 2.44e-04 |
| MetaQA-holdout200 | ppr | 1.000 | 0.855 | +0.145 | 0.930 | 0.855 | +0.075 | +0.070 | +0.070 [+0.035, +0.105] | +14 / -0 | 1.22e-04 |
| MetaQA-sub200 | proposed_v5_graphcentric | 0.950 | 0.870 | +0.080 | 0.865 | 0.870 | -0.005 | +0.085 | +0.085 [+0.050, +0.130] | +17 / -0 | 1.53e-05 |
| MetaQA-holdout200 | proposed_v5_graphcentric | 0.945 | 0.855 | +0.090 | 0.860 | 0.855 | +0.005 | +0.085 | +0.085 [+0.050, +0.125] | +17 / -0 | 1.53e-05 |

BM25 is bit-identical across KG conditions in every run (asserted per question).

## Gold-title coverage of each KG (vocab-align statistic)

| Split | KG | Nodes | Mean gold-title coverage | Fully covered questions |
|---|---|---|---|---|
| MetaQA-sub200 | kg_silver | 935 | 1.000 | 1.000 |
| MetaQA-sub200 | kg_openie | 10601 | 0.958 | 0.925 |
| MetaQA-holdout200 | kg_silver | 869 | 1.000 | 1.000 |
| MetaQA-holdout200 | kg_openie | 10954 | 0.973 | 0.950 |

## Pre-registered interpretation gates (decided before results)

Gate: silver-minus-openie inflation positive AND significant on the paired sign test for the graph retrievers => second causal point established. Null or negative => report as-is, no tuning; inclusion decision passes to the author.

- MetaQA-sub200 / proposed_v3: GATE NOT MET: positive but not significant (p=6.54e-02); reported as-is.
- MetaQA-holdout200 / proposed_v3: GATE MET: positive and significant (p=3.91e-03 < 0.05).
- MetaQA-sub200 / ppr: GATE MET: positive and significant (p=2.44e-04 < 0.05).
- MetaQA-holdout200 / ppr: GATE MET: positive and significant (p=1.22e-04 < 0.05).
- MetaQA-sub200 / proposed_v5_graphcentric: GATE MET: positive and significant (p=1.53e-05 < 0.05).
- MetaQA-holdout200 / proposed_v5_graphcentric: GATE MET: positive and significant (p=1.53e-05 < 0.05).

