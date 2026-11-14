Pre-declaration (written 2026-09-29 BEFORE any index-precision number or fit was computed)

Unit of analysis (cell): (benchmark, split, retriever, KG condition), condition != the OpenIE arm.
Outcome: gain = hit@5(retriever | condition KG) - hit@5(retriever | OpenIE KG), same split, same questions,
computed from the frozen per-question vectors (_kgswap_*.json).

Index at question-matched nodes (per question q, per retriever's matcher, per KG):
  M_q = nodes the retriever's own matcher returns for q
        (ProposedV3Retriever._match_nodes for proposed_v3 and proposed_v5_graphcentric;
         PPRRetriever._seed_indices for ppr)
  L_q = union of those nodes' passage links restricted to the corpus (the same accessor the retriever uses)
  G_q = the question's gold passages
  P_q = |L_q & G_q| / |L_q|  (0 if L_q empty),  R_q = |L_q & G_q| / |G_q|,  F1_q = 2PR/(P+R) (0 if P+R=0)

PRIMARY predictor: dF1 = mean_q F1_q(condition KG) - mean_q F1_q(OpenIE KG), same matcher, same split.
PRIMARY test: Spearman rho(dF1, gain) pooled over all cells of all three retrievers;
  95% CI by two-stage cluster bootstrap (outer: resample the 4 benchmarks with replacement;
  inner: resample questions within each benchmark-split, jointly for all its cells), B=2000, seed fixed.
Secondary (reported, not used for the headline): per-retriever rho; dPrecision, dRecall, d(P*R);
  level (not delta) F1; question-agnostic gold fraction of all KG-linked passages; 1-hop-expanded index
  (matched nodes + KG neighbours); headroom-normalised gain; C_KG (census, benchmark-level) as predictor
  on the same cells; leave-one-benchmark-out rho; within-2Wiki rho; cross-benchmark aligned-vs-OpenIE cells only.
