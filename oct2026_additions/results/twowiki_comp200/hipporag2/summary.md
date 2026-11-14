# HippoRAG 2 KG swap on the 2WikiMultiHop compositional split (comp200)

Generated 2026-10-02T15:31:59+00:00 by `eval/_kg_swap_hipporag2_comp200_score.py`; configuration as Appendix C.7 (`results/hipporag2_swap/METHODS.md`), protocol `results/twowiki_comp200/PROTOCOL.md`.

| n | hit@5 aligned | hit@5 independent | hit@5 BM25 | hit@5 dense-only (BGE-M3) | aligned minus independent [95% CI] | discordant +/- | sign-test p |
|---:|---:|---:|---:|---:|---|---:|---:|
| 200 | 0.925 | 0.725 | 0.215 | 0.415 | +0.200 [+0.135, +0.270] | +48/-8 | 4.69e-08 |

| condition | minus BM25 [95% CI] | +/- | p | minus dense-only [95% CI] | +/- | p |
|---|---|---:|---:|---|---:|---:|
| aligned | +0.710 [+0.640, +0.775] | +144/-2 | 2.41e-40 | +0.510 [+0.440, +0.580] | +104/-2 | 1.40e-28 |
| independent | +0.510 [+0.435, +0.585] | +105/-3 | 1.29e-27 | +0.310 [+0.240, +0.380] | +67/-5 | 6.39e-15 |

| condition | entity nodes | fact edges | synonymy edges | passages linked / pool | gold passages linked | gold fraction of linked | injected passage-triples | ppr / dpr_fallback | filter exceptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned | 577 | 393 | 28 | 373 / 1155 | 373 / 398 | 1.000 | 571 | 196 / 4 | 0 |
| independent | 8324 | 10331 | 4301 | 1134 / 1155 | 386 / 398 | 0.340 | 26142 | 186 / 14 | 0 |

Checks: dense-only ranking identical across the two conditions = True; graph = injection (facts, entities, 0 passage-neighbourhood mismatches) asserted by the driver for both conditions.
