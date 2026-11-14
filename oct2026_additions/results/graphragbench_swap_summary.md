# GraphRAG-Bench (medical) KG-Swap Summary (fourth swap, non-maximal point)

Generated 2026-09-18. Protocol: paper/eval/_graphragbench_swap_design.md. Census estimate for the subgraph axis: Chat_KG = 0.75. kg_silver = OpenIE over the released evidence_relations keyed by gold chunk; kg_openie = OpenIE over the 1,584-chunk corpus without question access. Retriever configurations frozen as released; top_k=5; RAG_V3_DENSE_BACKEND=deterministic; bootstrap B=10,000, seed 20260918.

## Swap results (hit@5; every gold chunk in top-5)

| Split | System | Silver graph | Silver BM25 | Delta(silver) | OpenIE graph | OpenIE BM25 | Delta(openie) | Swap | Paired mean [95% CI] | Discordant +/- | Sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphRAG-Bench-sub200 | proposed_v3 | 0.565 | 0.520 | +0.045 | 0.500 | 0.520 | -0.020 | +0.065 | +0.065 [+0.035, +0.100] | +13 / -0 | 2.44e-04 |
| GraphRAG-Bench-holdout200 | proposed_v3 | 0.580 | 0.505 | +0.075 | 0.520 | 0.505 | +0.015 | +0.060 | +0.060 [+0.025, +0.100] | +13 / -1 | 1.83e-03 |
| GraphRAG-Bench-sub200 | ppr | 0.690 | 0.520 | +0.170 | 0.200 | 0.520 | -0.320 | +0.490 | +0.490 [+0.415, +0.570] | +105 / -7 | 1.49e-23 |
| GraphRAG-Bench-holdout200 | ppr | 0.625 | 0.505 | +0.120 | 0.245 | 0.505 | -0.260 | +0.380 | +0.380 [+0.295, +0.460] | +87 / -11 | 8.07e-16 |
| GraphRAG-Bench-sub200 | proposed_v5_graphcentric | 0.695 | 0.520 | +0.175 | 0.405 | 0.520 | -0.115 | +0.290 | +0.290 [+0.230, +0.355] | +58 / -0 | 6.94e-18 |
| GraphRAG-Bench-holdout200 | proposed_v5_graphcentric | 0.680 | 0.505 | +0.175 | 0.400 | 0.505 | -0.105 | +0.280 | +0.280 [+0.215, +0.345] | +58 / -2 | 3.18e-15 |

BM25 is bit-identical across KG conditions in every run (asserted per question).

## Gold-chunk mapping coverage

| Split | KG | Nodes | Mapped passages | Mean coverage | Fully covered |
|---|---|---|---|---|---|
| GraphRAG-Bench-sub200 | kg_silver | 783 | 202 | 1.000 | 1.000 |
| GraphRAG-Bench-sub200 | kg_openie | 10345 | 1581 | 1.000 | 1.000 |
| GraphRAG-Bench-holdout200 | kg_silver | 846 | 208 | 1.000 | 1.000 |
| GraphRAG-Bench-holdout200 | kg_openie | 10345 | 1581 | 0.998 | 0.995 |

## Pre-registered gate

- GraphRAG-Bench-sub200 / proposed_v3: GATE MET: positive and significant (p=2.44e-04).
- GraphRAG-Bench-holdout200 / proposed_v3: GATE MET: positive and significant (p=1.83e-03).
- GraphRAG-Bench-sub200 / ppr: GATE MET: positive and significant (p=1.49e-23).
- GraphRAG-Bench-holdout200 / ppr: GATE MET: positive and significant (p=8.07e-16).
- GraphRAG-Bench-sub200 / proposed_v5_graphcentric: GATE MET: positive and significant (p=6.94e-18).
- GraphRAG-Bench-holdout200 / proposed_v5_graphcentric: GATE MET: positive and significant (p=3.18e-15).
