# GraphRAG-Bench (medical) KG-swap design — fourth swap, a text-corpus benchmark

Written 2026-09-18, before any result was computed. Mirrors `_metaqa_swap_design.md`.

## Why this benchmark

The census assigns GraphRAG-Bench a methodology-level estimate Ĉ_KG = 0.75 on the
subgraph axis (evidence isolated as subgraphs of an ontology mined from the corpus,
then human-confirmed). It is the only audited benchmark besides 2WikiMultiHop and
MetaQA whose released fields expose both a text corpus and the structure-derived
evidence (`evidence_relations`) for every question, so the same swap protocol can be
run: hold questions, gold passages, corpus, BM25 and retriever configuration fixed and
change only the KG the graph channel ranks over. WebQSP was considered: it releases no
text corpus, so a lexical arm cannot be built.

## S1  Corpus and gold passages (deterministic, no model)

- Corpus: `external/graphragbench/raw/medical_corpus.json` (one document, ~1.05M chars).
  Split into sentences (`(?<=[.!?])\s+`) and packed into consecutive chunks of at most
  120 words (whole sentences). Chunk id `c%04d`; title `Medical guideline, segment N`.
  All chunks form the candidate pool for both splits (1,584 chunks); the pool is
  identical across splits so the independent KG is built once.
- Gold: each question's released `evidence` sentences are paraphrases of corpus text
  (only 1% are verbatim), so each is mapped to the chunk with the highest content-token
  recall (stop words removed, tokens > 2 chars). A question is eligible when every
  evidence sentence maps with recall >= 0.6 and the union of mapped chunks has size 1 or 2.
  `evidence_spans` records, per evidence sentence, the best-overlapping sentence inside the
  gold chunk. hit@5 requires every gold chunk in the top five, as on 2WikiMultiHop.
- Eligible question types: Fact Retrieval, Complex Reasoning, Contextual Summarize.
  Creative Generation is excluded (its gold is not scored by overlap; rubric Step 0).
- Sampling: seed 20260918, shuffle within type, round-robin across types to 400;
  sub200 = first 200, holdout200 = next 200 (disjoint by construction).
- Determinism: the build runs twice in memory and must serialize identically.

## S2  Circular condition (kg_silver)

GraphRAG-Bench does not release its ontology graph. The released structure-derived text
is `evidence_relations`, the relation strings from the mined subgraph that the gold
evidence was isolated from. The circular KG is induced from them: for each gold chunk of
the split, the evidence_relations of every selected question whose gold set contains that
chunk are concatenated into one pseudo-passage keyed by the chunk id, and the same
OpenIE extractor used for the independent arm (llama3.1:8b, temperature 0, seed 0,
surface synonyms) turns the relation text into nodes and edges. Consequently every node's
`source_docs` is a gold chunk id: the inventory comes from the benchmark's own structure
and the node-to-passage mapping is gold-derived, which is the carrier identified on
2WikiMultiHop. This is an approximation of the benchmark's mined subgraph, not the
subgraph itself.

## S3  Independent condition (kg_openie)

`baselines/build_openie_kg.py` over the 1,584-chunk corpus with llama3.1:8b, surface
synonyms, no access to questions, evidence, or relations. Built once, shared by both splits.

## S4  Evaluation

`_kg_swap_graphragbench.py`: systems proposed_v3 (audited), ppr, proposed_v5_graphcentric;
top_k = 5; BM25 asserted bit-identical across conditions; paired per-question hit@5
difference silver minus openie, 10,000-resample bootstrap CI (seed 20260918), exact
two-sided sign test on discordant pairs. Coverage statistic: fraction of a question's
gold chunks that some KG node maps to (`source_docs`), reported per condition.

## S5  Pre-registered gate

Positive paired mean and sign-test p < 0.05 in a retriever–split cell = effect detected at
this non-maximal point. Null or negative = reported as-is, no tuning. As with STaRK-Prime,
a headroom-controlled secondary analysis on the BM25-miss stratum is allowed but labelled
exploratory.

## S6  End-to-end QA

The frozen vectors feed `eval/e2e_qa_swap.py`. GraphRAG-Bench answers are sentences, so
EM is not meaningful; the design planned token-F1 and cover-EM. The paper reports LM-judge
accuracy (Appendix C.10, Judge validation); token-F1 and cover-EM are in
`results/e2e_qa/<reader>/summary.*`.
