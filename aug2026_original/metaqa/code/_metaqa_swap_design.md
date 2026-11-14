# MetaQA Second-Swap Design (Experiment 4, 2026-06-13)

Goal: replicate the 2WikiMultiHop KG-swap at a second census point (MetaQA,
census C_K = 1.0, "All questions in MetaQA are generated from the movie
knowledge base") to convert the paper's causal evidence from one benchmark to
two. Mirror the 2Wiki protocol exactly; reuse the existing harness unchanged.

## Inputs (at paper/external/metaqa/raw/)
- kb.txt: movie KB triples "subject|relation|object"
- 2-hop vanilla QA (test split preferred; train/dev as fallback pool)
- wiki.txt: per-movie text documents (MetaQA-text corpus)

## Subset construction (deterministic, seed 20260613)
1. Parse 2-hop questions: topic entity in [brackets]; answers tab-separated,
   '|'-delimited. Derive the KB path mechanically: topic --r1--> bridge movies
   --r2--> answers (relation pair from the question template). NO manual
   choice enters gold construction; the path is determined by MetaQA's own KB
   and question, which is precisely the benchmark's C_K=1.0 construction.
2. Keep questions with EXACTLY 2 bridge movies, both having wiki.txt docs,
   and the answer string verifiably present in at least one bridge doc.
   Rationale: gold = the 2 bridge-movie docs (the docs that state the answer);
   |gold|=2 preserves the harness's "BOTH gold contained" hit@5 semantics and
   matches the 2Wiki two-gold protocol.
3. Sample 200 questions stratified across relation-pair templates (seeded
   shuffle within template, round-robin) to avoid template monoculture.
   If a second disjoint 200 is available, freeze it as holdout200-metaqa.
4. Corpus: union of all gold docs + deterministic distractor sample of other
   wiki.txt docs up to ~1,250 passages (2Wiki scale). passage_id = normalized
   movie name (lowercase, whitespace-collapsed), title = movie name,
   abstract = doc text. Write qa.jsonl / passages.jsonl / manifest.json in the
   exact 2Wiki sub200 format (gold_paper_ids = the 2 bridge doc ids,
   hop_count=2, evidence_spans = answer-string spans in each gold doc).

## Conditions
- kg_silver (gold-aligned): the KB subgraph induced by the selected
  questions' 2-hop paths: nodes = topic/bridge/answer entities (+KB aliases
  if any), edges = the path triples (plus their reverse). source_docs:
  movie node -> its own doc; non-movie node -> the docs of movies it is
  KB-connected to within the subgraph. This is the gold's own source
  structure, mirroring 2Wiki's silver (small, question-scoped, 100% gold-title
  coverage by construction).
- kg_openie (independent): paper/baselines/build_openie_kg.py over the
  ~1,250-doc corpus, kg_option B_openie_realistic, model llama3.1:8b via
  Ollama (same as 2Wiki; ~1.3 s/passage, ~27 min). No access to KB or gold.
- Output layout identical to 2Wiki: paper/external/metaqa/sub200/{qa.jsonl,
  passages.jsonl, manifest.json, kg_silver/, kg_openie/}.

## Evaluation and statistics (identical to 2Wiki swap)
- ev.run (paper/eval/eval_independent.py) with systems bm25 + proposed_v3,
  then ppr and proposed_v5_graphcentric, top_k=5, on both KG conditions.
  RAG_V3_DENSE_BACKEND=deterministic.
- Paired per-question hit@5 difference silver-minus-openie; 10,000-resample
  bootstrap 95% CI (fixed seed); exact two-sided sign test on discordants;
  BM25 must be bit-identical across conditions (assert).
- Secondary: gold-title coverage of each KG (reuse _swap_vocab_align.py
  logic) so the silver/openie asymmetry is documented as on 2Wiki.
- Results: paper/results/kg_swap_metaqa{,_ppr,_proposed_v5_graphcentric}.json
  same schema as kg_swap_2wiki*.json.

## Interpretation gates (decided BEFORE seeing results; report as-is)
- If silver-minus-openie inflation is positive and significant on the paired
  sign test for the graph retrievers: second causal point established;
  candidate for manuscript section 6 one sentence + appendix table.
- If null or negative: report honestly; do NOT tune. A null at a C_K=1.0
  point would itself need explanation (e.g., template questions are fully
  lexical-solvable) and the inclusion decision passes to the author.
- No hyperparameter changes to any retriever; configurations are frozen as
  released. Structural adaptation only (data plumbing).

## Risks
- wiki.txt docs are short and formulaic; BM25 may be strong -> smaller but
  still measurable swap effect expected on bridge-doc retrieval.
- Person entities have no docs; gold is movie docs only (by design).
- Fan-out filter (=2) shrinks the eligible pool; if <200 questions remain,
  take the maximum and report n.
