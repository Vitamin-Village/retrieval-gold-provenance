# STaRK-Prime Third-Swap Design (Experiment 5, 2026-06-15)

Goal: replicate the 2WikiMultiHop / MetaQA KG-swap at a THIRD census point
(STaRK-Prime, biomedical entity retrieval) to convert the paper's causal
evidence from two benchmarks to three -- and, critically, to add a
NON-MAXIMAL circular point. Mirror the 2Wiki / MetaQA protocol exactly; reuse
the existing harness and OpenIE builder unchanged.

## Why STaRK-Prime, and why it matters more than a fourth maximal point

Census = answer-entity axis C_K = 0.75. STaRK synthesizes every query from a
relational template executed over its precomputed knowledge base (PrimeKG):
the gold ANSWER ENTITY is selected by KB relation traversal. The 0.75 (vs
1.0 on 2Wiki and MetaQA) reflects that not every query's full reasoning chain
is forced through the KB graph -- some answer entities are reachable by
surface/lexical match alone, and some templates leave slack. This is exactly
the value: two maximal (C_K = 1.0) benchmarks could have been chosen to
maximize circularity. A non-maximal point that STILL shows the
silver-minus-openie inflation rules that out, because the effect
appears even where the benchmark's own construction is only partially
KB-circular. A null here would also be informative (it would bound where the
mechanism stops mattering), so the point is diagnostic either way.

## Inputs (at paper/external/stark/raw/)
- prime_qa.csv: columns id, query, answer_ids (list of SKB node indices).
- processed/ (from skb/prime/processed.zip, 28 MB):
  - node_info.pkl: dict node_index -> {id, type, name, source, details}.
    `details` carries per-type prose (drug: description / indication /
    mechanism_of_action; disease: mondo_definition / umls_description;
    gene/protein: summary + alias list). Name-only types
    (pathway / molecular_function / biological_process / cellular_component /
    anatomy / effect-phenotype / exposure) have no prose.
  - edge_index.pt [2 x 8,100,498], edge_types.pt: directed (src, dst) pairs
    with an integer relation id.
  - node_type_dict.pkl (10 types), edge_type_dict.pkl (18 relations:
    ppi, target, enzyme, indication, contraindication, side effect,
    interacts with, expression present/absent, phenotype present/absent,
    associated with, parent-child, synergistic interaction, linked to,
    carrier, transporter, off-label use).

### SKB parsing method (no stark-qa package required)
The processed.zip artifacts are parsed directly. node_info.pkl is a plain
dict keyed by the same integer index that appears in prime_qa.csv's
answer_ids (verified: all 12,828 distinct answer ids resolve; 0 missing).
Edges are loaded from the two .pt tensors with torch (2.6, weights_only=False)
and joined to relation strings via edge_type_dict. STaRK treats PrimeKG as
undirected, so each (s, rel, d) is materialized in both directions during
silver-KG construction. Passage abstracts are assembled from a per-type
ordered field list (see _TEXT_FIELDS in build_stark_swap_data.py), always
prefixed by "<name> (<type>)." so name-only entity types still yield a
self-describing, retrievable passage. NaN floats in disease details are
filtered.

## Subset construction (deterministic, seed 20260615)
1. Eligibility: keep questions whose gold answer-entity set (answer_ids) has
   size 1..3, with all gold ids present in node_info. Pool = 9,025 questions
   (size 1: 7,016; size 2: 1,278; size 3: 731). Size-1 is preferred because
   it gives a clean binary "gold-in-top-5" hit@5; sizes 2-3 are admitted so
   the subset is not artificially single-answer (and to exercise the harness's
   all-gold-in-top-5 semantics, identical to the 2Wiki two-gold protocol).
2. Each gold entity = one STaRK SKB node = one corpus passage. No manual
   choice enters gold construction: the gold is STaRK's own answer_ids, the
   passage is the entity's own SKB record. This IS the C_K = 0.75
   construction, taken as-is.
3. Stratified round-robin over the dominant gold-entity TYPE (the type of the
   first sorted gold id) so the 200 span all biomedical types rather than
   collapsing onto genes/diseases. Seeded shuffle within stratum, round-robin
   across sorted strata. A disjoint holdout200 is frozen from the next 200.
4. Corpus: union of all gold entities + a deterministic distractor sample of
   other SKB entities up to 1,250 passages (2Wiki / MetaQA scale).
   passage_id = normalized entity name; on name collision the SKB index is
   appended (`name::n<idx>`) so ids stay readable yet unique. title = entity
   name, abstract = assembled entity text. qa.jsonl / passages.jsonl /
   manifest.json are written in the exact 2Wiki sub200 schema
   (gold_paper_ids = the gold entity passage ids, hop_count = 2 when |gold|>=2
   else 1, evidence_spans = the entity-name span inside its own passage).

## Conditions
- kg_silver (gold-aligned): the SKB subgraph INDUCED ON THE 1,250-ENTITY
  CORPUS -- nodes are exactly the corpus entities, edges are every real SKB
  relation (both directions; PrimeKG is undirected) whose BOTH endpoints lie
  in the corpus. This deliberately matches the OpenIE KG scale (~1,250 nodes)
  so the swap is NOT confounded by graph size ("silver wins because it is
  bigger"). Keying is by ENTITY ID: passage_id = "e<id>", node_id = "n::e<id>",
  source_docs = ["e<id>"], so every node maps to exactly one corpus passage and
  the graph retriever can rank passages (the earlier name-keyed build "n::ttr"
  collided across same-named entities and mapped only 457/1,250 nodes -- fixed).
  Gold-entity coverage = 1.0 (every gold is in the corpus, asserted).
  Built by paper/eval/rebuild_stark_silver_rr.py.
  Distractor selection is a deterministic PER-GOLD ROUND ROBIN over each gold's
  seeded-shuffled SKB 1-hop neighbours (every gold gets a corpus neighbour
  before any gold gets a second), then random fill -> 0 gold entities are
  isolated in the induced subgraph (a uniform random distractor draw left
  ~35% of golds with no corpus neighbour, unreachable by any graph walk). Mean
  silver degree per gold: ~60 (sub200) / ~92 (holdout200). Silver edges: 61,424
  (sub200) / 57,730 (holdout200).
- kg_openie (independent): paper/baselines/build_openie_kg.py over the
  ~1,250-passage corpus, kg_option B_openie_realistic, local LLM via Ollama
  (same model family as 2Wiki / MetaQA). No access to the SKB or gold. Built
  in S3 by the main session via new_env. Input is passages.jsonl in the exact
  format the builder already consumes (passage_id / paper_id / text / title).
- Output layout identical to 2Wiki / MetaQA: paper/external/stark/sub200/{
  qa.jsonl, passages.jsonl, manifest.json, kg_silver/, kg_openie/}.

## STaRK-specific points
- Entity retrieval, not document retrieval: each "passage" is a biomedical
  entity record, so BM25 leans on the entity name + alias surface forms. The
  swap stresses whether the graph condition still helps when the gold is a
  named entity already present verbatim in its own passage.
- Gold-size handling: hit@5 = 1.0 iff ALL gold entity passages are in top-5
  (the existing harness semantics). Single-gold -> "the answer entity in
  top-5"; size 2-3 -> all answer entities in top-5. is_multihop is keyed on
  |gold| >= 2 (the harness convention), independent of the literal hop_count.
- Metric choice rationale: hit@5 (all-gold-in-top-5) is the metric the 2Wiki /
  MetaQA swaps already report, so the third point is directly comparable; no
  new metric is introduced.
- C_K = 0.75 is non-maximal by design; see top section for why this is the
  scientific payload of this swap.

## Evaluation and statistics (identical to prior swaps; run in S4)
- ev.run (paper/eval/eval_independent.py) with systems bm25 + proposed_v3,
  then ppr and proposed_v5_graphcentric, top_k = 5, on both KG conditions.
  RAG_V3_DENSE_BACKEND = deterministic.
- Paired per-question hit@5 difference silver-minus-openie; 10,000-resample
  bootstrap 95% CI (fixed seed); exact two-sided sign test on discordants;
  BM25 must be bit-identical across conditions (assert).
- Secondary: gold-entity coverage of each KG (silver = 1.0 by construction;
  openie measured) so the silver/openie asymmetry is documented as on the
  prior benchmarks.

## Pre-registered interpretation gates (decided BEFORE seeing results)
- If silver-minus-openie inflation is positive AND significant on the paired
  sign test for the graph retrievers: a third, NON-MAXIMAL causal point is
  established -> candidate for manuscript section 6 one sentence + appendix
  table, strengthening the "not only maximal" argument materially.
- If null or negative: report honestly, as-is; do NOT tune. A null at a
  C_K = 0.75 point is itself interpretable -- it would bound the mechanism to
  near-maximal circularity, which is a publishable, honest finding. The
  inclusion decision passes to the author.
- No hyperparameter changes to any retriever; configurations are frozen as
  released. Structural data plumbing only.

## Risks
- Gold entities (genes, pathways) have large SKB fan-out, but the silver KG is
  the subgraph INDUCED ON THE CORPUS (1,250 nodes), not the full neighbourhood,
  so silver and OpenIE are scale-matched (~1,250 nodes each). This removes the
  "graph is bigger therefore wins" confound. The within-corpus induced edge
  count (sub200 61,424 / holdout200 57,730) is still denser than a typical
  OpenIE KG because PrimeKG edges are dense among biomedical entities; this is
  a real property of the aligned KB, not an artifact, and is reported as the
  silver/openie structural asymmetry (the analogue of 2Wiki/MetaQA).
- Entity passages are short and the answer name is present verbatim, so BM25
  is strong -> a smaller but, at C_K = 0.75, still potentially measurable swap
  effect. A null is a genuine possible outcome and is acceptable.
- Name-only entity types contribute thin passages ("<name> (<type>).");
  retrieval for those leans entirely on the name, which is the honest
  STaRK setting.
