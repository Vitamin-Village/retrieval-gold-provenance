# Verbatim Construction Quotes: Remaining Benchmarks

The paper's quote tables (Tables 6 and 7) give, for each of the 29 axis
judgments, the verbatim text that supports its bin; their typeset source is
`oct2026_additions/results/tables/tab_quotes_full.tex`. The rater A
records in this directory (`rater_A_assignments.md`,
`rater_A_assignments_expansion.md`) carry the per-axis quotes for MuSiQue,
HotpotQA, 2WikiMultiHop, GraphRAG-Bench, GRADE, MultiHop-RAG, STaRK, MetaQA,
and the two calibration anchors. This file carries the six remaining
axis-level judgments, for the three benchmarks added in the census broadening:
Natural Questions, TriviaQA, and WebQSP. Quotes are reproduced
character-for-character from the construction papers cited per row, given as
ACL Anthology identifiers.

| benchmark | axis | rater A | rater B | adjudicated | C | verbatim construction quote (source) |
|---|---|---|---|---|---|---|
| Natural Questions | passage (long answer) | bin1 | bin1 | **bin1** | 0.25 | "Annotators identify long answers by selecting the smallest HTML bounding box that contains all of the information required to answer the question" (Kwiatkowski et al. 2019, Q19-1026) |
| Natural Questions | answer (short answer) | bin0 | bin1 | **bin0** (one-bin disagreement, resolved to lower) | 0.0 | "the short answer is one or more entities that give a short answer to the question, or a boolean" (Q19-1026) |
| TriviaQA | evidence (distant-supervision docs) | bin4 | bin3 | **bin3** (one-bin disagreement, resolved to lower) | 0.75 | "the presence of the answer string in an evidence document implies that the document does answer the question" (Joshi et al. 2017, P17-1147) -- answer-surface (A) derivation |
| TriviaQA | answer | bin0 | bin0 | **bin0** | 0.0 | "95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents" (P17-1147) |
| WebQSP | answer-entity | bin4 | bin4 | **bin4** | 1.0 | "... provides the semantic parses in SPARQL with standard Freebase entity identifiers, which are directly executable on Freebase" (Yih et al. 2016, P16-2033) |
| WebQSP | entity/edge/path | bin4 | bin4 | **bin4** | 1.0 | "the core component is to represent the relation between the entity and the answer, referred as the inferential chain" (P16-2033) |
