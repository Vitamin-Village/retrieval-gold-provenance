# SC1 C-C Census EXPANSION (main-targeting) — Independent Rater A scoring (2026-06-01)

> Rater A, blind to other raters. Rubric: `SC1_census_rubric_v1_2026-06-01.md` (STEP 0-6), applied verbatim.
> NEW 3 benchmarks only: MultiHop-RAG (2401.15391), STaRK (2404.13207, NeurIPS'24 D&B), MetaQA (Zhang et al. AAAI'18, arXiv:1709.04071).
> Each construction § read from original paper (ar5iv full text + project README) before scoring; verbatim quotes attached (STEP4).
> Scoring input = ONLY (construction text + field schema). retriever/hit/Delta NEVER used (A6 guard).
> bin codes: 0->bin0, 0~0.3->bin1, 0.3~0.7->bin2, 0.7~1->bin3, 1->bin4.

---

## A. Per-benchmark x unit-axis scoring table

| benchmark | axis | STEP0 | derivation (K/A/T) | trigger signal | base bin | STEP3 correction | **final bin** | **C** | construction-§ quote (source) |
|---|---|---|---|---|---|---|---|---|---|
| **MultiHop-RAG** (2401.15391) | supporting-evidence (text chunk/sentence) | PASS | **T** | gold evidence = factual sentences extracted from news articles by LLM; NO KG | bin0 | B3-2 (human review of dataset) — floor, no move | **bin0** | **0** | "For each article, we extract factual or opinion sentences using a trained language model"; gold evidence = source text sentences paired with article metadata; no knowledge graph used (ar5iv:2401.15391) |
| **MultiHop-RAG** | answer | PASS | **T** | answer composed from text-derived claims; "we employ GPT-4 to paraphrase the evidence ... claims" then compose query; answer grounded in source claims, not KG triple | bin0 | B3-2 (manual review subset) — floor | **bin0** | **0** | "we feed the claim set to GPT-4 and prompt it ... to generate a query"; claims = paraphrased article sentences; "we manually review a subset sample" (ar5iv:2401.15391) |
| **MultiHop-RAG** | bridge-entity (generation provenance) | PASS | T | bridge = entity linking two textual claims, "not formal structured representations" | bin0 | none | **bin0** | **0** | "we prompt GPT-4 to identify the bridge-entity and bridge-topic for each claim" — textual bridge, no KG (ar5iv:2401.15391) |
| **STaRK** (2404.13207) | answer-entity (gold answer node set) | PASS | **K** | gold answer = KB node satisfying a sampled relation template; "we sample a relation template ... select one of the candidate nodes that satisfies the relational requirement, which is referred to as the gold answer" | bin4 | **B3-2** (3-model verification of gold answer set) bin4->bin3 | **bin3** | **0.7~1** | "we sample a relation template, such as '(a product) belongs to <brand>' ... select one of the candidate nodes that satisfies the relational requirement, which is referred to as the gold answer"; "we employ three Claude models. Only candidates that pass the verification across all models are included in the final ground truth answer set" (STaRK §query synthesis, 2404.13207) |
| **STaRK** | edge/relation (gold reasoning requirement) | PASS | **K** | relational requirement = instantiated KB relation template; the relation path is the KB structure itself | bin4 | B3-2 (3-model gold verification) bin4->bin3 | **bin3** | **0.7~1** | "Initially, we sample a relation template ... and instantiate it into a relational requirement" (2404.13207) — **[BORDERLINE, see §C]** |
| **STaRK** | textual-property / supporting-text | PASS | T (KB node text) | textual properties extracted from candidate node's text; this axis is KB node free-text, not the relational structure | bin1 | none | **bin1** | **0~0.3** | "We extract textual properties that align with the interests" from the gold node's text; queries fuse textual+relational via LLM (Claude-v2, GPT-4-Turbo) (2404.13207) — **[BORDERLINE axis-split, see §C]** |
| **MetaQA** (1709.04071, AAAI'18) | answer (multi-hop, tail entity) | PASS | **K** | answer = entity reached by KG traversal; "each edge ... is a triplet (a1, r, a2)"; templates over relation paths; answer = tail node of reasoning rule | bin4 | none (no augmentation; no human-refine of generated gold) | **bin4** | **1** | "each edge from E(G) is a triplet (a_i^1, r_i, a_i^2), representing a directed relation"; "we design 21 question types ... 10 question templates and uniformly randomly sample"; answer reached via "reasoning rule (y, acted_by, a_i)" path traversal (ar5iv:1709.04071) |
| **MetaQA** | edge/reasoning-path | PASS | **K** | gold reasoning path = sequence of KG triples (2-hop/3-hop relation chain over the same KB) | bin4 | none | **bin4** | **1** | "The 2-hop and 3-hop datasets are generated from the same knowledge base"; path traversal "author->authored paper->authored_by author to find answers" (ar5iv:1709.04071; README) |
| **MetaQA** | question (1-hop subset, provenance) | PASS | K (mixed: 1-hop from WikiMovies) | 1-hop questions reuse WikiMovies (themselves KG-template-generated); 2/3-hop fully template-from-KG | bin4 | none | **bin4** | **1** | "The 1-hop dataset is derived from the wiki_entities branch of ... MovieQA"; WikiMovies is itself KG-template generated; "schema is subject\|relation\|object" (README; ar5iv:1709.04071) — **[1-hop axis slightly less pure, see §C]** |

> No N/A rows in this expansion (all 3 have supporting-gold and are NOT win-rate). All 3 PASS STEP0.

---

## B. Per-benchmark composite C (axis-vector)

| benchmark | axis-vector (final bins) | composite reading |
|---|---|---|
| **MultiHop-RAG** | evidence=bin0, answer=bin0, bridge=bin0 | **C≈0** — LLM-over-news-text pipeline, NO KG; gold = source text sentences. Lower-end neighbour of MuSiQue/HotpotQA. |
| **STaRK** | answer-entity=bin3, edge/relation=bin3, textual-property=bin1 | **C high, axis-split** — gold answer entities + relation requirement sampled directly from KB (K, bin3 after 3-model verify); only the textual-property axis is KB free-text (bin1). |
| **MetaQA** | answer=bin4, edge/path=bin4, (question 1-hop=bin4) | **C≈1** — pure KG-template construction; answer = tail node of KG reasoning path. Co-anchor with 2WikiMultiHopQA-edge and our qa_v4_from_kg at the C=1 ceiling. |

---

## C. Uncertain / borderline scores flagged for adjudication

1. **STaRK answer-entity / relation axis: bin3 vs bin4 (most uncertain).**
   The gold answer is a KB node selected to satisfy a sampled KB relation template ("select one of the candidate nodes that satisfies the relational requirement, which is referred to as the gold answer") — pure K, which is base bin4. I applied **B3-2 (bin4->bin3)** because a 3-model Claude verification gates the final answer set ("Only candidates that pass the verification across all models are included in the final ground truth answer set"). The rubric's STEP3/STEP5 distinguishes "human/model **refining the gold**" (B3-2, -1 bin) from "model only **evaluating output**" (B3-3, no move). STaRK's 3-model pass is verification of which candidate KB nodes enter the gold answer set — it **shapes the gold** (filters candidates), so B3-2 applies -> bin3. But a reasonable Rater B could read this verification as quality-gating that does NOT change the K-derivation (the answers are still KB nodes, none injected from text) and score bin4. **±1 bin (bin3 vs bin4); |distance|=1, conservative-low rule keeps bin3.** This is my most uncertain score in the expansion.

2. **STaRK axis collision: answer-entity (bin3) vs textual-property/supporting-text (bin1).**
   STaRK queries fuse a relational requirement (K, from KB structure) with textual properties (T, from KB node free-text). The "supporting-gold" the rubric's STEP1-default (supporting-passage) points at is ambiguous: STaRK retrieval scores against the gold **answer node set** (K, bin3), but the query's *textual* grounding is node free-text (bin1). I scored both per STEP1 transparency. Default-axis assignment (answer-entity bin3 vs textual bin1) is a 2-bin gap where Rater B may diverge — structurally identical to the GRADE axis-collision flagged in the core census. Note STaRK is a **retrieval-on-KB** benchmark, so the answer-node axis (K) is the operative retrieval gold -> leans bin3, not bin1.

3. **MetaQA 1-hop subset purity (minor).**
   2-hop/3-hop = fully KG-template-from-triples (unambiguous bin4). The 1-hop subset is inherited from WikiMovies; WikiMovies is itself KG-template-generated, so I kept bin4, but a Rater B might argue 1-hop reuse introduces a thin non-template layer. The headline multi-hop axes (answer, edge/path) are unambiguously bin4, so this does not move the composite. Flagged only for completeness (|distance| likely 0-1).

---

## D. STEP6 inputs (for kappa, when Rater B done)

- Observations added: **MultiHop-RAG** (3 axes), **STaRK** (3 axes), **MetaQA** (3 axes) = 9 axis-rows; supporting-gold axes for the passage-default sub-vector:
  - MultiHop-RAG = bin0 (evidence), STaRK = bin3 (answer-entity, retrieval gold) | bin1 (textual) **[split]**, MetaQA = bin4 (answer/path).
- All 3 PASS; none excluded from kappa denominator.
- Cross-census placement: MetaQA joins the C=1 ceiling cluster (2Wiki-edge, qa_v4_from_kg); MultiHop-RAG joins the C=0 floor cluster (MuSiQue, HotpotQA, qa_independent_v1); STaRK is a new high-C KB-retrieval point (axis-split bin3/bin1), analogous structurally to GRADE.

---

## Authors' note on the quotes (added November 2026)

Everything above this line is rater A's record of 2026-06-01 and is left as written. Before the public release, the authors compared each string in quote marks that the record gives as text of a benchmark paper or README with that source. The sources were the PDF text of MultiHop-RAG (arXiv:2401.15391v1), STaRK (arXiv:2404.13207, v1 to v3), and MetaQA (arXiv:1709.04071v5), and the README of the MetaQA dataset (github.com/yuyuz/MetaQA, commit b1e6973). A string counts as word for word when its letters and digits occur in the source in the same order; line-break hyphenation, punctuation, capitalization, and the layout of subscripts and superscripts are ignored. Strings in quote marks that are rater A's own terms or that cite the rubric were not part of this check.

### Quote column

Apart from the two MetaQA quotes described in the next paragraph, every quote of this column is word for word in the source. The STaRK quotes are all in arXiv v1, and not all of them remain in v2 or v3.

The two MetaQA quotes have the words of the paper and write on one line what the paper typesets. "(a_i^1, r_i, a_i^2)" is the triplet that the paper prints with superscripts 1 and 2 and subscript i (Section 3). "author->authored paper->authored_by author to find answers" is the chain that the paper prints as "author → paper → author to find answers" with the labels "authored" and "authored by" above the two arrows (Section 1). `oct2026_additions/human_validation/llm_census_logged/excerpts.json` lists the first under its item METAQA_ANSWER as not found verbatim; the two differ only in the order in which the subscript and the superscript are written.

### Notes (trigger-signal column)

This column holds rater A's notes on what led to a bin. The strings below stand in quote marks there and are shortened or reworded.

| Where | In the record | In the source |
|---|---|---|
| MultiHop-RAG, bridge-entity | "not formal structured representations" | Rater A's own wording. The paper does not contain these words. |
| MetaQA, answer | "each edge ... is a triplet (a1, r, a2)" | Shortened form of the quote in the same row, with the indices left out. |
