# HippoRAG 2 KG swap on 2WikiMultiHop

Generated 2026-09-24T09:55:55+00:00 by `eval/_kg_swap_hipporag2_score.py`. Methods: `METHODS.md`. hit@5 = both gold passages in the top 5. Paired difference = aligned (kg_silver) minus independent (kg_openie), same questions; bootstrap B=10,000, seed 20260918; exact two-sided sign test on discordant questions.

| split | n | hit@5 aligned | hit@5 independent | hit@5 BM25 | hit@5 dense-only | paired diff [95% CI] | discordant +/- | sign-test p |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| sub200 (first split) | 200 | 0.885 | 0.645 | 0.365 | 0.495 | +0.240 [+0.165, +0.315] | +58/-10 | 2.36e-09 |
| holdout200 (held-out split) | 200 | 0.890 | 0.615 | 0.355 | 0.440 | +0.275 [+0.210, +0.345] | +60/-5 | 4.87e-13 |

## Each condition against HippoRAG's own dense-only channel (BGE-M3, KG-independent)

| split | condition | hit@5 | minus dense-only [95% CI] | discordant +/- | sign-test p |
|---|---|---:|---|---:|---:|
| sub200 | aligned | 0.885 | +0.390 [+0.320, +0.460] | +80/-2 | 1.41e-21 |
| sub200 | independent | 0.645 | +0.150 [+0.085, +0.215] | +40/-10 | 2.39e-05 |
| holdout200 | aligned | 0.890 | +0.450 [+0.380, +0.520] | +91/-1 | 3.76e-26 |
| holdout200 | independent | 0.615 | +0.175 [+0.115, +0.240] | +41/-6 | 1.77e-07 |

## Graph HippoRAG built from each injected KG

| split | condition | entity nodes | fact edges (unique triples) | synonymy edges | passages linked / pool | gold passages linked | gold fraction of linked | injected passage-triples |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sub200 | aligned | 574 | 380 | 112 | 342 / 1256 | 342 / 386 | 1.000 | 535 |
| sub200 | independent | 7374 | 9456 | 5945 | 1243 / 1256 | 384 / 386 | 0.309 | 26219 |
| holdout200 | aligned | 575 | 381 | 70 | 342 / 1229 | 342 / 388 | 1.000 | 536 |
| holdout200 | independent | 7166 | 9558 | 5821 | 1207 / 1229 | 382 / 388 | 0.316 | 25988 |

## Sensitivity (pre-declared): independent KG with edge-level provenance

OpenIE triples injected only at the passage they were extracted from (edge `source_docs`), which is what HippoRAG's own OpenIE step would produce; the 3.2k/3.4k provenance-free surface-synonym edges of the OpenIE KG are dropped (HippoRAG adds its own synonymy edges). Not the protocol condition; reported for robustness.

| split | hit@5 independent (edge prov.) | aligned minus it [95% CI] | discordant +/- | p | edge minus node prov. [95% CI] | linked passages | gold fraction of linked |
|---|---:|---|---:|---:|---|---:|---:|
| sub200 | 0.610 | +0.275 [+0.200, +0.350] | +63/-8 | 1.03e-11 | -0.035 [-0.090, +0.020] | 1205 / 1256 | 0.312 |
| holdout200 | 0.585 | +0.305 [+0.235, +0.375] | +67/-6 | 3.95e-14 | -0.030 [-0.075, +0.015] | 1176 / 1229 | 0.320 |

## Context: the paper's own retrievers under the same two KGs (frozen vectors, not re-run)

| split | retriever | hit@5 aligned | hit@5 independent | paired diff [95% CI] | p |
|---|---|---:|---:|---|---:|
| sub200 | ppr | 0.740 | 0.425 | +0.315 [+0.235, +0.395] | 2.71e-12 |
| sub200 | proposed_v3 | 0.535 | 0.375 | +0.160 [+0.100, +0.220] | 4.43e-07 |
| holdout200 | ppr | 0.740 | 0.430 | +0.310 [+0.230, +0.390] | 2.25e-12 |
| holdout200 | proposed_v3 | 0.480 | 0.365 | +0.115 [+0.055, +0.175] | 6.06e-04 |

## Retrieval path (recognition-memory filter output)

`ppr` = the LLM fact filter kept at least one fact and HippoRAG ran PPR; `dpr_fallback` = no fact kept, HippoRAG returned its dense-passage ranking (default behaviour).

| split | condition | ppr | dpr_fallback | filter exceptions | LLM calls cached |
|---|---|---:|---:|---:|---:|
| sub200 | aligned | 184 | 16 | 0 | 200 |
| sub200 | independent | 193 | 7 | 0 | 200 |
| holdout200 | aligned | 186 | 14 | 0 | 200 |
| holdout200 | independent | 189 | 11 | 0 | 200 |

## End-to-end QA (reader qwen2.5:14b; eval/e2e_qa_swap.py prompt and scoring; `e2e_qa/qwen2_5_14b/summary.md`)

| split | EM aligned | EM independent | EM diff [95% CI] | p (sign) | F1 diff [95% CI] | EM diff on retrieval flips + (n) | EM diff on no flip (n) | EM closed-book / oracle |
|---|---:|---:|---|---:|---|---:|---:|---:|
| sub200 | 0.375 | 0.320 | +0.055 [-0.020, +0.130] | 2.00e-01 | +0.016 [-0.041, +0.075] | +0.414 (58) | -0.068 (132) | 0.040 / 0.725 |
| holdout200 | 0.370 | 0.310 | +0.060 [-0.010, +0.130] | 1.19e-01 | +0.032 [-0.021, +0.085] | +0.350 (60) | -0.059 (135) | 0.055 / 0.685 |

## 2WikiMultiHop, realistic structure: live-Wikidata KG (kg_wikidata)

Same HippoRAG 2 configuration, BGE-M3 embeddings, llama3.1:8b filter and node-provenance injection as above. BM25 = the frozen BM25 vectors of this benchmark; dense-only = HippoRAG's own BGE-M3 passage channel (KG-independent); paper PPR = the paper's PPR retriever on the same KG (frozen vectors).

| split | n | hit@5 aligned | hit@5 independent | hit@5 wikidata (realistic) | BM25 | dense-only | paper PPR aligned | paper PPR independent | paper PPR wikidata (realistic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sub200 (first split) | 200 | 0.885 | 0.645 | 0.560 | 0.365 | 0.495 | 0.740 | 0.425 | 0.500 |
| holdout200 (held-out split) | 200 | 0.890 | 0.615 | 0.580 | 0.355 | 0.440 | 0.740 | 0.430 | 0.535 |

| split | paired difference | mean [95% CI] | discordant +/- | sign-test p |
|---|---|---|---:|---:|
| sub200 | wikidata (realistic) minus independent | -0.085 [-0.150, -0.020] | +15/-32 | 1.86e-02 |
| sub200 | wikidata (realistic) minus aligned | -0.325 [-0.395, -0.250] | +6/-71 | 3.42e-15 |
| holdout200 | wikidata (realistic) minus independent | -0.035 [-0.105, +0.035] | +22/-29 | 4.01e-01 |
| holdout200 | wikidata (realistic) minus aligned | -0.310 [-0.385, -0.235] | +9/-71 | 4.37e-13 |

| split | condition | entity nodes | fact edges | synonymy edges | passages linked / pool | gold passages linked | gold fraction of linked | injected passage-triples | graph edges | ppr / dpr_fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sub200 | aligned | 574 | 380 | 112 | 342 / 1256 | 342 / 386 | 1.000 | 535 | 1750 | 184 / 16 |
| sub200 | independent | 7374 | 9456 | 5945 | 1243 / 1256 | 384 / 386 | 0.309 | 26219 | 49688 | 193 / 7 |
| sub200 | wikidata (realistic) | 11291 | 44426 | 18969 | 1086 / 1256 | 338 / 386 | 0.311 | 4258720 | 1230783 | 194 / 6 |
| holdout200 | aligned | 575 | 381 | 70 | 342 / 1229 | 342 / 388 | 1.000 | 536 | 1713 | 186 / 14 |
| holdout200 | independent | 7166 | 9558 | 5821 | 1207 / 1229 | 382 / 388 | 0.316 | 25988 | 48550 | 189 / 11 |
| holdout200 | wikidata (realistic) | 11959 | 47100 | 37584 | 1086 / 1229 | 342 / 388 | 0.315 | 4245570 | 1250215 | 196 / 4 |

Checks: sub200: dense-only ranking identical across conditions = True; holdout200: dense-only ranking identical across conditions = True.

## MetaQA (2-hop; gold = the two bridge documents)

Same HippoRAG 2 configuration, BGE-M3 embeddings, llama3.1:8b filter and node-provenance injection as above. BM25 = the frozen BM25 vectors of this benchmark; dense-only = HippoRAG's own BGE-M3 passage channel (KG-independent); paper PPR = the paper's PPR retriever on the same KG (frozen vectors).

| split | n | hit@5 aligned (question-induced) | hit@5 independent | hit@5 full KB (realistic) | BM25 | dense-only | paper PPR aligned (question-induced) | paper PPR independent | paper PPR full KB (realistic) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sub200 (first split) | 200 | 0.935 | 0.755 | 0.695 | 0.870 | 0.240 | 1.000 | 0.935 | 0.995 |
| holdout200 (held-out split) | 200 | 0.915 | 0.745 | 0.590 | 0.855 | 0.215 | 1.000 | 0.930 | 1.000 |

| split | paired difference | mean [95% CI] | discordant +/- | sign-test p |
|---|---|---|---:|---:|
| sub200 | aligned (question-induced) minus independent | +0.180 [+0.120, +0.240] | +40/-4 | 1.71e-08 |
| sub200 | full KB (realistic) minus independent | -0.060 [-0.130, +0.010] | +22/-34 | 1.41e-01 |
| sub200 | full KB (realistic) minus aligned (question-induced) | -0.240 [-0.300, -0.180] | +1/-49 | 9.06e-14 |
| holdout200 | aligned (question-induced) minus independent | +0.170 [+0.100, +0.235] | +43/-9 | 2.04e-06 |
| holdout200 | full KB (realistic) minus independent | -0.155 [-0.230, -0.080] | +15/-46 | 8.84e-05 |
| holdout200 | full KB (realistic) minus aligned (question-induced) | -0.325 [-0.390, -0.260] | +0/-65 | 5.42e-20 |

| split | condition | entity nodes | fact edges | synonymy edges | passages linked / pool | gold passages linked | gold fraction of linked | injected passage-triples | graph edges | ppr / dpr_fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sub200 | aligned (question-induced) | 935 | 1776 | 26 | 387 / 1250 | 386 / 386 | 0.997 | 4640 | 4356 | 196 / 4 |
| sub200 | independent | 9218 | 10182 | 2305 | 1246 / 1250 | 384 / 386 | 0.308 | 29442 | 52718 | 185 / 15 |
| sub200 | full KB (realistic) | 5414 | 19910 | 721 | 1250 / 1250 | 386 / 386 | 0.309 | 596350 | 276442 | 198 / 2 |
| holdout200 | aligned (question-induced) | 869 | 1680 | 20 | 385 / 1250 | 385 / 385 | 1.000 | 5350 | 4590 | 196 / 4 |
| holdout200 | independent | 9490 | 10549 | 2409 | 1246 / 1250 | 384 / 385 | 0.308 | 30619 | 54561 | 190 / 10 |
| holdout200 | full KB (realistic) | 5490 | 20390 | 776 | 1250 / 1250 | 385 / 385 | 0.308 | 589790 | 278035 | 197 / 3 |

Checks: sub200: dense-only ranking identical across conditions = True; holdout200: dense-only ranking identical across conditions = True.

## GraphRAG-Bench (medical; gold = 1-2 chunks, hit@5 = all gold in top 5)

Same HippoRAG 2 configuration, BGE-M3 embeddings, llama3.1:8b filter and node-provenance injection as above. BM25 = the frozen BM25 vectors of this benchmark; dense-only = HippoRAG's own BGE-M3 passage channel (KG-independent); paper PPR = the paper's PPR retriever on the same KG (frozen vectors).

| split | n | hit@5 aligned | hit@5 independent | BM25 | dense-only | paper PPR aligned | paper PPR independent |
|---|---:|---:|---:|---:|---:|---:|---:|
| sub200 (first split) | 200 | 0.805 | 0.540 | 0.520 | 0.525 | 0.690 | 0.200 |
| holdout200 (held-out split) | 200 | 0.800 | 0.575 | 0.505 | 0.540 | 0.625 | 0.245 |

| split | paired difference | mean [95% CI] | discordant +/- | sign-test p |
|---|---|---|---:|---:|
| sub200 | aligned minus independent | +0.265 [+0.205, +0.325] | +53/-0 | 2.22e-16 |
| holdout200 | aligned minus independent | +0.225 [+0.165, +0.290] | +48/-3 | 1.97e-11 |

| split | condition | entity nodes | fact edges | synonymy edges | passages linked / pool | gold passages linked | gold fraction of linked | injected passage-triples | graph edges | ppr / dpr_fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sub200 | aligned | 674 | 984 | 355 | 199 / 1584 | 199 / 202 | 1.000 | 5038 | 6421 | 186 / 14 |
| sub200 | independent | 9481 | 21678 | 19026 | 1581 / 1584 | 202 / 202 | 0.128 | 1067971 | 814626 | 183 / 17 |
| holdout200 | aligned | 715 | 1028 | 381 | 202 / 1584 | 202 / 208 | 1.000 | 5421 | 6903 | 187 / 13 |
| holdout200 | independent | 9481 | 21678 | 19026 | 1581 / 1584 | 207 / 208 | 0.131 | 1067971 | 814626 | 183 / 17 |

Checks: sub200: dense-only ranking identical across conditions = True; holdout200: dense-only ranking identical across conditions = True.
