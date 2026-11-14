# Coverage-stratified delta under the independent OpenIE KG (2WikiMultiHop)

Per-question hit@5 delta (graph minus BM25) under the OpenIE KG, stratified by
gold-title coverage in the OpenIE node inventory. Matching rule is identical to
`swap/code/_swap_vocab_align.py` (exact match on normalized node id, name, or
alias); aggregate coverage reproduced exactly (sub200: mean 0.5600, fully
covered 0.3400; holdout200: 0.6125, 0.3950). Bootstrap: 10,000 resamples, seed
20260612, percentile 95% CI on mean delta(full) minus mean delta(partial+none).

All table values below are generated directly from
`swap/results/coverage_stratification.json` (regenerated 2026-07-19).

## sub200

| System | Stratum | n | graph hit@5 | BM25 hit@5 | paired delta | win-flips |
|---|---|---:|---:|---:|---:|---:|
| ppr | full | 68 | 0.5147 | 0.2941 | +0.2206 | 19 |
| ppr | partial | 88 | 0.4091 | 0.3182 | +0.0909 | 20 |
| ppr | none | 44 | 0.3182 | 0.5682 | -0.2500 | 4 |
| proposed_v3 | full | 68 | 0.3676 | 0.2941 | +0.0735 | 5 |
| proposed_v3 | partial | 88 | 0.3295 | 0.3182 | +0.0114 | 5 |
| proposed_v3 | none | 44 | 0.4773 | 0.5682 | -0.0909 | 2 |
| proposed_v5_graphcentric | full | 68 | 0.3824 | 0.2941 | +0.0882 | 9 |
| proposed_v5_graphcentric | partial | 88 | 0.3409 | 0.3182 | +0.0227 | 5 |
| proposed_v5_graphcentric | none | 44 | 0.5227 | 0.5682 | -0.0455 | 3 |

Full vs rest (partial+none): ppr +0.2433, CI95 [+0.0784, +0.4073];
proposed_v3 +0.0963, CI95 [+0.0138, +0.1858];
proposed_v5_graphcentric +0.0882, CI95 [-0.0232, +0.2054].

## holdout200

| System | Stratum | n | graph hit@5 | BM25 hit@5 | paired delta | win-flips |
|---|---|---:|---:|---:|---:|---:|
| ppr | full | 79 | 0.5949 | 0.3038 | +0.2911 | 33 |
| ppr | partial | 87 | 0.2989 | 0.3448 | -0.0460 | 9 |
| ppr | none | 34 | 0.3824 | 0.5000 | -0.1176 | 3 |
| proposed_v3 | full | 79 | 0.3418 | 0.3038 | +0.0380 | 4 |
| proposed_v3 | partial | 87 | 0.3563 | 0.3448 | +0.0115 | 7 |
| proposed_v3 | none | 34 | 0.4412 | 0.5000 | -0.0588 | 1 |
| proposed_v5_graphcentric | full | 79 | 0.3544 | 0.3038 | +0.0506 | 6 |
| proposed_v5_graphcentric | partial | 87 | 0.3563 | 0.3448 | +0.0115 | 7 |
| proposed_v5_graphcentric | none | 34 | 0.4412 | 0.5000 | -0.0588 | 1 |

Full vs rest (partial+none): ppr +0.3573, CI95 [+0.1806, +0.5296];
proposed_v3 +0.0462, CI95 [-0.0408, +0.1371];
proposed_v5_graphcentric +0.0589, CI95 [-0.0369, +0.1586].

## Interpretation

Across all six system-split cells the per-question delta is monotone in
gold-title coverage: positive in the fully-covered stratum, near zero in the
partial stratum, and negative in the zero-coverage stratum, where every graph
retriever loses to BM25. The concentration is strongest and statistically
clear for PPR (full-vs-rest difference +0.24 and +0.36, both bootstrap CIs
excluding zero) and directionally consistent for the audited and
graph-centric systems, whose smaller overall gains leave the CIs crossing
zero in three of four cells; win-flip questions likewise come predominantly
from the covered strata (sub200: 63 of 72 flips across the three systems fall
in the full or partial stratum, only 9 in none; holdout200: 66 of 71, only 5
in none). Because the OpenIE KG is built with no access to gold annotations,
this pattern shows that a self-constructed KG inherits the same
provenance-style confound in proportion to how much of the gold entity
vocabulary its node inventory happens to cover, rather than the confound
being an artifact of the gold-aligned silver KG alone.

Note on the BM25 levels: BM25 hit@5 is not flat across the strata -- it is
highest in the zero-coverage stratum (0.5682 on sub200, 0.5000 on holdout200,
versus 0.29-0.35 in the covered strata), so part of the negative delta in that
stratum reflects reduced headroom over a stronger lexical baseline. The
size-weighted mean of the per-stratum BM25 values equals the overall BM25
hit@5 of each split (0.3650 and 0.3550), which is the internal consistency
check that the levels in this file must satisfy.
