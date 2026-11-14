# KG swap on a compositional 2WikiMultiHop split (comp200)

Generated 2026-10-02T22:18:08Z by `analysis/twowiki_comp200_summary.py` from frozen vectors and reader outputs. Protocol (fixed before any run on the split): `results/twowiki_comp200/PROTOCOL.md`. Every result below is reported whatever its value.

Split: 200 `compositional` questions from 2WikiMultiHop dev (`select_subsample`, seed 20261002), QA-id-disjoint from the inference-type `sub200`/`holdout200`; corpus, gold, aligned KG (`kg_silver`, Wikidata evidence triples) and independent KG (`kg_openie`, Llama-3.1-8B OpenIE, no gold access) built by the unchanged builders; retrievers frozen, deterministic dense surrogate. hit@5 = both gold passages in the top five.

## Deviations

- No departure from PROTOCOL.md in selection, builders, KG settings, retriever configurations, metrics, statistics or readers. The points below are differences from how the existing splits were run, all non-substantive or pre-declared.
- Drivers: the existing swap drivers hard-code sub200/holdout200 and overwrite results/kg_swap_2wiki*.json, so their run_kg functions were called from a new driver (eval/_kg_swap_2wiki_comp200.py); HippoRAG 2 and the readers ran through thin wrappers (split choice and output directory only).
- Hybrid swap CI: eval/_swap_stats.py seeds random once at import and draws sub200 then holdout200 from one stream; for comp200 the same seed (20260608) is set immediately before resampling, as eval/_kg_swap_2wiki_r2.py does (pre-declared).
- OpenIE build: same command and settings. Ollama is 0.34.3; the server version used for the May 2026 builds is not recorded, the model digest (46e0c10c039e) is the same. 16 of 1,155 passages (1.4%) returned no parsable JSON after the retry (0.4% and 0.8% on sub200/holdout200); as before they contribute no nodes.
- Corpus size 1,155 passages (existing splits 1,229-1,256): the rule takes each question's ten context paragraphs and deduplicates by title; compositional questions share more paragraphs. 48 / 49 comp200 passages also occur in the sub200 / holdout200 pools and 6 / 3 comp200 gold passages are also gold there (the existing pipeline enforces QA-id disjointness only, as does this split).
- HippoRAG 2: identical BaseConfig (no key differs from the configuration of the sub200 runs, results/hipporag2_swap/METHODS.md), same clone/commit and package; BGE-M3 on CPU with the default thread count. One OpenIE entity becomes an empty string after HippoRAG's text processing (empty_entity_strings = 1; 0 in the runs of the other splits); the graph-equals-injection checks pass.
- Readers: run first on the three retrievers, then re-run after HippoRAG 2 was scored to add its two conditions, as PROTOCOL.md Section 6 provides; the 1,709 earlier generations were reused from the cache, 396 new ones were generated per reader.

## (a) Retrieval

| retriever | hit@5 aligned | hit@5 indep. | hit@5 BM25 | swap aligned − indep. [95% CI] | disc. +/− | sign p | margin aligned (p, verdict) | margin indep. (p, verdict) |
|---|---:|---:|---:|---|---:|---:|---|---|
| Hybrid | 0.310 | 0.210 | 0.215 | +0.100 [+0.055, +0.150] | +22/−2 | 3.6e-05 | +0.095 (8.8e-04, beats) | -0.005 (1, n.s.) |
| PPR | 0.855 | 0.240 | 0.215 | +0.615 [+0.535, +0.690] | +132/−9 | 3.6e-29 | +0.640 (6.1e-30, beats) | +0.025 (0.56, n.s.) |
| Hybrid-PPR | 0.435 | 0.220 | 0.215 | +0.215 [+0.160, +0.275] | +44/−1 | 2.6e-12 | +0.220 (1.3e-10, beats) | +0.005 (1, n.s.) |
| HippoRAG 2 | 0.925 | 0.725 | 0.215 | +0.200 [+0.135, +0.270] | +48/−8 | 4.7e-08 | +0.710 (2.4e-40, beats) | +0.510 (1.3e-27, beats) |

Graph beats BM25 (sign test p<0.05, uncorrected): aligned 4/4, independent 1/4; flips beats→not: 3. Holm over the 4 comp200 cells of an arm: aligned 4, independent 1.
CI: 10,000-resample percentile bootstrap, Python `random` re-seeded right before resampling with the original driver's seed (Hybrid 20260608, PPR and Hybrid-PPR 20260611, HippoRAG 2 20260918 over sorted ids). Swap = Delta(aligned) − Delta(independent) = mean paired difference.

### Context: the existing inference-type splits (frozen, `analysis/headline_margins_summary.json`)

| split | retriever | aligned | indep. | BM25 | swap | swap p | verdict aligned / indep. |
|---|---|---:|---:|---:|---:|---:|---|
| first | Hybrid | 0.535 | 0.375 | 0.365 | +0.160 | 4.4e-07 | beats / n.s. |
| first | PPR | 0.740 | 0.425 | 0.365 | +0.315 | 2.7e-12 | beats / n.s. |
| first | Hybrid-PPR | 0.750 | 0.395 | 0.365 | +0.355 | 5.2e-18 | beats / n.s. |
| first | HippoRAG 2 | 0.885 | 0.645 | 0.365 | +0.240 | 2.4e-09 | beats / beats |
| held-out | Hybrid | 0.480 | 0.365 | 0.355 | +0.115 | 6.1e-04 | beats / n.s. |
| held-out | PPR | 0.740 | 0.430 | 0.355 | +0.310 | 2.2e-12 | beats / n.s. |
| held-out | Hybrid-PPR | 0.705 | 0.370 | 0.355 | +0.335 | 2.4e-19 | beats / n.s. |
| held-out | HippoRAG 2 | 0.890 | 0.615 | 0.355 | +0.275 | 4.9e-13 | beats / beats |

## (b) End-to-end answers (alias-aware EM; `eval/e2e_qa_swap.py` unchanged)

### Llama-3.1-8B (`llama3.1:8b`)

closed-book EM 0.110, gold-only (oracle) EM 0.555, BM25 EM 0.180.

| retriever | EM aligned | EM indep. | EM BM25 | answer swap [95% CI] | disc. +/− | sign p | margin aligned (p, verdict) | margin indep. (p, verdict) |
|---|---:|---:|---:|---|---:|---:|---|---|
| Hybrid | 0.245 | 0.210 | 0.180 | +0.035 [-0.005, +0.080] | +13/−6 | 0.17 | +0.065 (0.011, beats) | +0.030 (0.15, n.s.) |
| PPR | 0.545 | 0.195 | 0.180 | +0.350 [+0.275, +0.430] | +79/−9 | 4.2e-15 | +0.365 (7.1e-16, beats) | +0.015 (0.71, n.s.) |
| Hybrid-PPR | 0.315 | 0.205 | 0.180 | +0.110 [+0.055, +0.165] | +27/−5 | 1.1e-04 | +0.135 (3.5e-06, beats) | +0.025 (0.33, n.s.) |
| HippoRAG 2 | 0.490 | 0.385 | 0.180 | +0.105 [+0.035, +0.175] | +36/−15 | 0.0046 | +0.310 (4.4e-13, beats) | +0.205 (1.3e-08, beats) |

Mediation (from `e2e_qa_swap.py`): Hybrid: EM diff +0.409 on 22 retrieval flips to hit, -0.006 on 176 non-flips; PPR: EM diff +0.538 on 132 retrieval flips to hit, +0.051 on 59 non-flips; Hybrid-PPR: EM diff +0.500 on 44 retrieval flips to hit, +0.006 on 155 non-flips; HippoRAG 2: EM diff +0.500 on 48 retrieval flips to hit, +0.000 on 144 non-flips.

### Qwen2.5-14B (`qwen2.5:14b`)

closed-book EM 0.080, gold-only (oracle) EM 0.610, BM25 EM 0.215.

| retriever | EM aligned | EM indep. | EM BM25 | answer swap [95% CI] | disc. +/− | sign p | margin aligned (p, verdict) | margin indep. (p, verdict) |
|---|---:|---:|---:|---|---:|---:|---|---|
| Hybrid | 0.260 | 0.225 | 0.215 | +0.035 [-0.005, +0.080] | +13/−6 | 0.17 | +0.045 (0.078, n.s.) | +0.010 (0.75, n.s.) |
| PPR | 0.570 | 0.225 | 0.215 | +0.345 [+0.265, +0.420] | +78/−9 | 7.5e-15 | +0.355 (1.4e-14, beats) | +0.010 (0.85, n.s.) |
| Hybrid-PPR | 0.290 | 0.210 | 0.215 | +0.080 [+0.030, +0.130] | +22/−6 | 0.0037 | +0.075 (0.011, beats) | -0.005 (1, n.s.) |
| HippoRAG 2 | 0.525 | 0.465 | 0.215 | +0.060 [-0.010, +0.125] | +31/−19 | 0.12 | +0.310 (1.0e-12, beats) | +0.250 (7.6e-11, beats) |

Mediation (from `e2e_qa_swap.py`): Hybrid: EM diff +0.273 on 22 retrieval flips to hit, +0.011 on 176 non-flips; PPR: EM diff +0.523 on 132 retrieval flips to hit, +0.034 on 59 non-flips; Hybrid-PPR: EM diff +0.364 on 44 retrieval flips to hit, +0.006 on 155 non-flips; HippoRAG 2: EM diff +0.375 on 48 retrieval flips to hit, -0.014 on 144 non-flips.

### Context: existing splits, answer swap (frozen)

| reader | split | retriever | EM aligned | EM indep. | EM BM25 | swap | p |
|---|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | first | Hybrid | 0.180 | 0.130 | 0.100 | +0.050 | 0.076 |
| Llama-3.1-8B | first | PPR | 0.475 | 0.145 | 0.100 | +0.330 | 1.3e-16 |
| Llama-3.1-8B | first | Hybrid-PPR | 0.240 | 0.100 | 0.100 | +0.140 | 8.4e-06 |
| Llama-3.1-8B | first | HippoRAG 2 | 0.255 | 0.180 | 0.100 | +0.075 | 0.049 |
| Llama-3.1-8B | held-out | Hybrid | 0.215 | 0.155 | 0.145 | +0.060 | 0.023 |
| Llama-3.1-8B | held-out | PPR | 0.490 | 0.230 | 0.145 | +0.260 | 5.3e-09 |
| Llama-3.1-8B | held-out | Hybrid-PPR | 0.265 | 0.165 | 0.145 | +0.100 | 0.0022 |
| Llama-3.1-8B | held-out | HippoRAG 2 | 0.285 | 0.265 | 0.145 | +0.020 | 0.67 |
| Qwen2.5-14B | first | Hybrid | 0.265 | 0.220 | 0.200 | +0.045 | 0.16 |
| Qwen2.5-14B | first | PPR | 0.560 | 0.215 | 0.200 | +0.345 | 4.2e-13 |
| Qwen2.5-14B | first | Hybrid-PPR | 0.365 | 0.200 | 0.200 | +0.165 | 5.6e-06 |
| Qwen2.5-14B | first | HippoRAG 2 | 0.375 | 0.320 | 0.200 | +0.055 | 0.2 |
| Qwen2.5-14B | held-out | Hybrid | 0.285 | 0.220 | 0.185 | +0.065 | 0.035 |
| Qwen2.5-14B | held-out | PPR | 0.555 | 0.255 | 0.185 | +0.300 | 1.4e-11 |
| Qwen2.5-14B | held-out | Hybrid-PPR | 0.400 | 0.230 | 0.185 | +0.170 | 2.0e-06 |
| Qwen2.5-14B | held-out | HippoRAG 2 | 0.370 | 0.310 | 0.185 | +0.060 | 0.12 |

## (c) HippoRAG 2

hit@5 aligned 0.925, independent 0.725, BM25 0.215, HippoRAG's own dense-only channel (BGE-M3) 0.415; swap +0.200 [+0.135, +0.270], +48/−8, p = 4.7e-08. Retrieval path (ppr / dense fallback): kg_silver 196/4, kg_openie 186/14. Dense-only ranking identical across conditions: True. Details: `results/twowiki_comp200/hipporag2/summary.md`.

## Sanity checks

- Question types: {'compositional': 200}; gold passages per question: {2: 200}; 398 unique gold passages.
- Corpus: 1155 passages (sub200 1,256; holdout200 1,229). QA-id overlap with sub200 / holdout200: 0 / 0. Passages also in the sub200 / holdout200 pool: 48 / 49; comp200 gold passages that are also gold there: 6 / 3.
- Aligned KG: 585 nodes (376 with source_docs), 399 edges (sub200 589/391, holdout200 593/392).
- Independent KG: 10205 nodes, 10336 edges (7580 extracted relations, 2756 surface-synonym), parse failures 16/1155 (0.0139), 2444 s (sub200 9,949/9,497, holdout200 9,629/9,586).

| split | passages | gold | aligned: linked | linked gold | gold frac. of linked | gold covered | indep.: linked | linked gold | gold frac. of linked | gold covered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp200 | 1155 | 398 | 373 | 373 | 1.0 | 0.937 | 1139 | 386 | 0.339 | 0.97 |
| sub200 | 1256 | 386 | 342 | 342 | 1.0 | 0.886 | 1251 | 384 | 0.307 | 0.995 |
| holdout200 | 1229 | 388 | 342 | 342 | 1.0 | 0.881 | 1219 | 382 | 0.313 | 0.985 |

- BM25 hit@5: comp200 0.215 (sub200 0.365, holdout200 0.355); BM25 vector identical across all 6 comp200 vector files: True.
