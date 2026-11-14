# STaRK-Prime KG-Swap Summary (Third Causal Experiment)

Generated 2026-06-15. Mirrors the 2WikiMultiHop / MetaQA swap protocol (paper/eval/_stark_swap_design.md). STaRK answer-entity axis is census C_K = 0.75 -- a NON-maximal circular point: gold answer entities are selected by relation traversal over the biomedical KB. kg_silver is the KB subgraph induced on the 1,250-entity corpus (circular); kg_openie is a llama3.1:8b OpenIE KG over the entity-description passages (independent, no KB/gold access), scale-comparable to silver. Retriever configs frozen; top_k=5; RAG_V3_DENSE_BACKEND=deterministic; bootstrap B=10,000, seed 20260615.

## Swap results (hit@5; all gold answer entities in top-5)

| Split | System | Silver graph | Silver BM25 | Delta(silver) | OpenIE graph | OpenIE BM25 | Delta(openie) | Swap | Paired mean [95% CI] | Discordant +/- | Sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| STaRK-sub200 | proposed_v3 | 0.130 | 0.135 | -0.005 | 0.145 | 0.135 | +0.010 | -0.015 | -0.015 [-0.045, +0.015] | +3 / -6 | 5.08e-01 |
| STaRK-holdout200 | proposed_v3 | 0.275 | 0.215 | +0.060 | 0.250 | 0.215 | +0.035 | +0.025 | +0.025 [-0.005, +0.060] | +8 / -3 | 2.27e-01 |
| STaRK-sub200 | ppr | 0.065 | 0.135 | -0.070 | 0.040 | 0.135 | -0.095 | +0.025 | +0.025 [-0.005, +0.055] | +7 / -2 | 1.80e-01 |
| STaRK-holdout200 | ppr | 0.120 | 0.215 | -0.095 | 0.100 | 0.215 | -0.115 | +0.020 | +0.020 [-0.025, +0.065] | +12 / -8 | 5.03e-01 |
| STaRK-sub200 | proposed_v5_graphcentric | 0.125 | 0.135 | -0.010 | 0.100 | 0.135 | -0.035 | +0.025 | +0.025 [-0.010, +0.065] | +10 / -5 | 3.02e-01 |
| STaRK-holdout200 | proposed_v5_graphcentric | 0.270 | 0.215 | +0.055 | 0.205 | 0.215 | -0.010 | +0.065 | +0.065 [+0.025, +0.110] | +16 / -3 | 4.43e-03 |

BM25 is bit-identical across KG conditions in every run (asserted per question).

## Gold coverage of each KG (entity-name)

| Split | KG | Nodes | Mean gold-name coverage | Fully covered |
|---|---|---|---|---|
| STaRK-sub200 | kg_silver | 1250 | 1.000 | 1.000 |
| STaRK-sub200 | kg_openie | 7130 | 0.805 | 0.725 |
| STaRK-holdout200 | kg_silver | 1250 | 1.000 | 1.000 |
| STaRK-holdout200 | kg_openie | 7824 | 0.790 | 0.740 |

## Pre-registered interpretation gates (decided before results)

Gate: silver-minus-openie inflation positive AND significant on the paired sign test => third causal point established, now at a NON-maximal census tick (C_K=0.75). Null or negative => report as-is, no tuning; inclusion decision passes to the author.

- STaRK-sub200 / proposed_v3: GATE NOT MET: mean not positive (-0.015).
- STaRK-holdout200 / proposed_v3: GATE NOT MET: positive but not significant (p=2.27e-01).
- STaRK-sub200 / ppr: GATE NOT MET: positive but not significant (p=1.80e-01).
- STaRK-holdout200 / ppr: GATE NOT MET: positive but not significant (p=5.03e-01).
- STaRK-sub200 / proposed_v5_graphcentric: GATE NOT MET: positive but not significant (p=3.02e-01).
- STaRK-holdout200 / proposed_v5_graphcentric: GATE MET: positive and significant (p=4.43e-03).

