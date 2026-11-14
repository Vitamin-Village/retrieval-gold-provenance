# Partial-Mapping Transplant Dose Curve (q-curve), 2WikiMultiHop

Code: `paper/eval/build_transplant_kgs.py` (q conditions, seed 20260612, nested
deterministic selection: silver-sourced node ids sorted, shuffled once, first
ceil(q*N) keep their gold `source_docs`; q=0 is identical to T1b
`kg_openie_svocab_noprov`, q=1 to T1 `kg_openie_svocab`, verified by
`verify_q_boundaries` and the test suite, 37 passed / 1 skipped).
Runner: `paper/eval/_kg_transplant_2wiki.py --q-curve [--system ppr|proposed_v5_graphcentric]`.
Results: `paper/results/kg_transplant_q_2wiki{,_ppr,_proposed_v5_graphcentric}.json`;
endpoints from `paper/results/kg_transplant_2wiki*.json`.
Protocol identical to the published transplant rows: paired hit@5 vs the OpenIE
condition, 10,000-resample bootstrap 95% CI, exact two-sided sign test.

## Mean paired hit@5 difference vs OpenIE (gold-mapping fraction q)

| Retriever | Split | q=0 (T1b) | q=0.25 | q=0.50 | q=0.75 | q=1 (T1) |
|---|---|---:|---:|---:|---:|---:|
| Audited (proposed_v3) | first | +0.020 | +0.030 | +0.045 | +0.060 | +0.060 |
| Audited (proposed_v3) | held-out | -0.020 | -0.015 | +0.010 | +0.025 | +0.025 |
| PPR | first | -0.025 | +0.015 | +0.035 | +0.070 | +0.085 |
| PPR | held-out | -0.015 | +0.015 | +0.040 | +0.045 | +0.080 |
| Graph-centric | first | -0.035 | -0.005 | -0.005 | +0.025 | +0.025 |
| Graph-centric | held-out | -0.045 | -0.040 | -0.010 | +0.005 | +0.025 |

Selected CIs (q=0.75): audited first [0.005, 0.110] p=0.043; PPR first
[0.025, 0.120] p=0.0066; PPR held-out [0.005, 0.090] p=0.064.

## Verdict

The transferred share is monotone non-decreasing in q on all six
retriever-split cells (one flat tie, graph-centric first split q=0.25/0.50).
Restoring the gold node-to-passage mapping on a larger deterministic fraction
of grafted nodes moves the transfer smoothly from the mapping-emptied floor to
the full-transplant value, which makes the provenance dose-response explicit:
inflation scales with the fraction of gold-derived passage links a system's
inventory carries, the system-side alignment quantity the manuscript's
Appendix "System-Side Alignment" defines.
