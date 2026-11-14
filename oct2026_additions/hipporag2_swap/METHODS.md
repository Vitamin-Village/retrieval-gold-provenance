# HippoRAG 2 KG swap on 2WikiMultiHop, MetaQA and GraphRAG-Bench: methods

This note records exactly what was run for the HippoRAG 2 replication of the controlled KG swap
(2026-09-24). Numbers and graph statistics are in `summary.md` / `summary.json`.

## Question

The paper's KG swap holds the questions, gold, candidate pool and retriever fixed and changes only
the KG: an *aligned* KG (the Wikidata evidence triples the 2Wiki gold is derived from) versus an
*independent* KG (OpenIE over the passage text, no gold access). Here the retriever is a published
graph-RAG system, HippoRAG 2 (Gutiérrez et al., 2025), run with its default retrieval pipeline. Only
the KG it indexes changes.

## Data (unchanged from the paper's swap)

- 2WikiMultiHopQA `sub200` (first split) and `holdout200` (held-out split): 200 questions each
  (`external/2wiki/<split>/qa.jsonl`), two gold passages per question.
- Candidate pool: all passages of the split (1,256 / 1,229; `passages.jsonl`), each indexed as one
  HippoRAG document `"{title}\n{text}"` (HippoRAG's own 2Wiki corpus format), no chunking.
- Aligned KG: `kg_silver/` (sub200: 589 nodes, 391 edges; holdout200: 593 / 392). Nodes are Wikidata
  entities whose `source_docs` are the passages they map to (gold passages only).
- Independent KG: `kg_openie/` (sub200: 9,949 nodes, 9,497 edges; holdout200: 9,629 / 9,586), built by
  Llama-3.1-8B OpenIE over all passage texts. 6,327 / 6,178 edges are extracted relations with an
  extraction passage (`source_docs`); 3,170 / 3,408 are the builder's surface-synonym edges, which have
  no passage provenance.

## System and environment

- HippoRAG 2: pip package `hipporag` 2.0.0a4, built from OSU-NLP-Group/HippoRAG commit
  `d437bfb1805278b81e20c82357ed3f7d90f14901` (the only upstream tag, v1.0.0, is the older HippoRAG 1
  code). Conda env `rag-graphrag-py311`, Python 3.11.15, Windows 11.
- Install-only patches to the clone (made before this experiment):
  1. `setup.py`: read README as UTF-8; relax exact pins to lower bounds (`openai==1.91.1` is not on
     the index); move `gritlm` and `vllm` (not installable on native Windows) to extras.
  2. `src/hipporag/embedding_model/__init__.py`: import each embedding backend lazily inside
     `_get_embedding_model_class` instead of at module import.
  3. `src/hipporag/HippoRAG.py`: import the offline (vLLM / Transformers) OpenIE classes only in
     their `openie_mode` branches.
  None of these touch indexing or retrieval logic. **No further patches were made for this
  experiment**; the installed package is file-identical to the clone's `src/hipporag`.
- Runtime library versions (the upstream pins `torch==2.5.1`, `transformers==4.45.2` were relaxed):
  torch 2.12.0+cpu, transformers 5.9.0, sentence-transformers 5.5.1, python-igraph 0.11.8,
  openai 2.38.0, numpy 2.4.6, pandas 2.3.3.
- LLM: `llama3.1:8b` (Ollama 0.34.3, Q4_K_M, digest `46e0c10c039e`), the same model that built the
  OpenIE KG, served through Ollama's OpenAI-compatible endpoint `http://localhost:11434/v1`,
  temperature 0 (HippoRAG default). HippoRAG places `llm_name` in file names and `:` is illegal in
  Windows paths, so HippoRAG was given the label `hipporag-llama3.1-8b`. The driver sets the request
  model to `llama3.1:8b` (`llm_config.generate_params["model"]` and `DSPyFilter.model_name`). Ollama's
  default context was 32,768 tokens. Across the 1,200 filter calls of the six reported runs, prompts were 2,837–3,037
  tokens (no truncation). 1,192 responses ended with `stop`. Eight hit the filter's upstream
  512-token output cap (`length`); HippoRAG either parsed them or fell back to its dense ranking. The
  LLM is used only by HippoRAG's recognition-memory fact filter, once per question. It made no OpenIE
  calls (see Injection).
- Embedding model: BAAI/bge-m3 (Hugging Face snapshot `5617a9f61b028005a4858fdac845db406aefb181`)
  through HippoRAG's native `Transformers/` backend (`embedding_model_name="Transformers/BAAI/bge-m3"`,
  SentenceTransformer, CLS pooling + L2 normalisation, batch 64), run on CPU. The same model encodes
  passages, entity nodes, facts and queries. The default NV-Embed-v2 (7B) was not used because of its
  GPU memory needs. Contriever is natively supported but much weaker than current retrievers. BGE-M3 is the dense retriever the paper already uses for its dense-channel rerun.
  This backend ignores HippoRAG's query instructions, which BGE-M3 does not need.
- HippoRAG configuration: `BaseConfig` defaults except `save_dir`, `llm_name`, `llm_base_url`,
  `embedding_model_name`. The retrieval-relevant defaults are `linking_top_k=5` (facts kept for the
  filter and phrase seeding), `passage_node_weight=0.05`, `damping=0.5`, `synonymy_edge_topk=2047`,
  `synonymy_edge_sim_threshold=0.8`, `is_directed_graph=False`, `rerank_dspy_file_path=None` (the
  built-in DSPy filter prompt) and `openie_mode="online"`. They are identical for every condition,
  with no per-condition tuning. Retrieval uses `HippoRAG.retrieve(queries, num_to_retrieve=20)`, the
  default HippoRAG 2 path: query-to-fact scoring, then the LLM recognition-memory filter, then
  phrase-node weights plus dense passage-node weights, then personalized PageRank. When the filter
  keeps no fact, HippoRAG returns its dense passage ranking (upstream behaviour, left unchanged). The
  top 5 of the ranking are scored.

## KG injection

HippoRAG 2 `index(docs)` runs NER and triple extraction per document and caches the result in
`<save_dir>/openie_results_ner_<llm_name>.json` as
`{"docs": [{"idx", "passage", "extracted_entities", "extracted_triples"}], "avg_ent_chars", "avg_ent_words"}`.
On load, HippoRAG recomputes `idx = "chunk-" + md5(passage)`. Only documents missing from the cache
are sent to the LLM. We wrote this file before calling `index()`, with one entry for every pool
passage:

- For passage *p*, `extracted_triples` holds every KG edge (s, relation, o) with
  *p* ∈ source_docs(s) ∪ source_docs(o), using node names as surface forms. This is node-to-passage
  provenance, the same construction for both KGs and the same provenance the paper's PPR retriever
  uses. `extracted_entities` holds the names of those triples' endpoints (HippoRAG uses this field
  only for statistics). Passages with no linked edge get empty lists.
- HippoRAG then processes the injected output exactly as it processes its own OpenIE output. It
  applies `filter_invalid_triples` (deduplication) and `text_processing` (lowercasing,
  non-alphanumerics to spaces). Entity nodes are the processed triple endpoints. Each passage gets
  edges to the entities in its triples. Every unique triple becomes a fact. BGE-M3 kNN synonymy edges
  (cosine ≥ 0.8) link entity nodes. Finally HippoRAG writes `graph.pickle`.
- No LLM OpenIE was run. Every pool passage is in the cache, so the set of chunks to extract is
  empty. The driver also replaces `openie.batch_openie` with a function that raises, so any
  extraction attempt would abort the run. None did.
- Verification after each index load (asserted in the driver):
  - HippoRAG's fact store equals the set of processed injected triples.
  - Its entity store equals the set of their endpoints.
  - Every passage node's entity neighbourhood in `graph.pickle` equals its injected entity set
    (0 mismatches in all runs).
  - The vertex count equals passages plus entities.
  The resulting graph statistics are in `summary.md`.

## Runs

One HippoRAG save directory per (split, condition). Passage vectors do not
depend on the KG. The second condition of a split therefore had its passage store seeded with the
first condition's `vdb_chunk.parquet`, which HippoRAG computed; hash ids and texts were checked
equal before copying. The dense-only ranking (HippoRAG's `retrieve_dpr`, same BGE-M3 passage vectors)
is identical across the two conditions for all 200 questions on both splits. This confirms that the
passage channel is shared.

Every reported retrieval sent its filter requests to `llama3.1:8b`; an index was reused only after
its fact and entity counts were verified.

## Scoring and statistics

- `eval/_kg_swap_hipporag2_score.py` scores each question's top 5 with the paper's shared scorer
  (`eval/eval_independent.score_record` / `summarize_system`). It writes
  `external/2wiki/<split>/_kgswap_hipporag2_{kg_silver,kg_openie}.json`, with the same layout as
  `_kgswap_ppr_*.json` and system name `hipporag2`.
- hit@5 = 1 iff both gold passages are in the top 5.
- Paired statistics per split, aligned minus independent over the same 200 questions: mean
  difference; 10,000-resample bootstrap 95% CI (`random.seed(20260918)` right before resampling,
  sorted qa ids, `boot_ci` of `eval/_kg_swap_metaqa.py`); exact two-sided sign test over discordant
  questions.
- BM25 reference hit@5 comes from the frozen BM25 vectors shared by every swap condition
  (0.365 / 0.355). The dense-only reference is HippoRAG's own `retrieve_dpr` with BGE-M3.

## Sensitivity (declared before the full results were seen)

Node-to-passage provenance attaches an OpenIE triple to every passage that mentions either endpoint.
It also carries the builder's provenance-free synonym edges. A second independent condition,
`--provenance edge`, instead injects each extracted relation only at the passage it was extracted
from (edge `source_docs`), which is what HippoRAG's own OpenIE step would produce. It drops the
synonym edges; HippoRAG adds its own. Everything else is unchanged. The aligned KG has no edge-level
provenance, so it has no counterpart. These vectors are not written to `external/`; the release ships
them as `hipporag2_swap/2wiki/<split>/_kgswap_hipporag2_kg_openie_edgeprov.json`, scored with the same
`vectors` function of `eval/_kg_swap_hipporag2_score.py`.

## End-to-end reader

`eval/_e2e_qa_hipporag2.py` runs `eval/e2e_qa_swap.py` unchanged: same prompt, `qwen2.5:14b` reader
at temperature 0, same EM / F1 / cover scoring and paired statistics. Its condition list is
restricted to the hipporag2 vectors. It writes to `results/hipporag2_swap/e2e_qa/`, with its cache
seeded from a copy of the shared qwen cache. `e2e_qa_swap.py` knows the `hipporag2` alias and file
prefix, so a full run in `results/e2e_qa/` includes these vectors as the paired conditions
`kg_silver` / `kg_openie` of system `hipporag2`.

## Follow-up conditions and benchmarks (same config, embedding model, LLM and injection)

The HippoRAG configuration, BGE-M3 backend, `llama3.1:8b` filter, node-provenance injection,
verification and statistics are exactly as above. The only changes are the KG files and the
benchmark data. Their end-to-end answers come from the full reader runs (system `hipporag2` in
`results/e2e_qa/<reader>/`), not from `eval/_e2e_qa_hipporag2.py`.
- **2WikiMultiHop `kg_wikidata`** (realistic structure, `eval/build_wikidata_kg_2wiki.py`). Every
  corpus passage title is resolved to a Wikidata entity. All of its entity-valued claims are kept,
  and nothing is induced from the evaluated questions. A corpus entity maps to its own passage; an
  object entity maps to the passages whose subjects link to it. sub200 has 11,303 nodes and 44,430
  edges; holdout200 has 11,973 / 47,144. The edges carry no extraction provenance. Class and
  attribute objects therefore have very long `source_docs` lists (sub200: 'human' 1,071 passages,
  'male' 761, 'female' 309). Node provenance
  therefore attaches about 3,900 triples to each linked passage (4.26M / 4.25M passage-triple links,
  1,086 linked passages each). This is the same construction the paper's PPR retriever uses with
  this KG. The passage vectors are the 2Wiki ones from above (identical strings).
- **MetaQA** (`external/metaqa/<split>/`): 1,250-passage pools that differ by split (151 passages
  shared), 200 two-hop questions, gold = the two bridge documents. The KGs are `kg_silver`
  (question-induced aligned KG; source_docs outside the pool are ignored), `kg_openie` and
  `kg_fullkb` (the full MetaQA KB restricted to the corpus).
- **GraphRAG-Bench** (medical; `external/graphragbench/<split>/`): one 1,584-chunk corpus for both
  splits, with documents `"Medical guideline, segment N\n<text>"`. Gold is 1–2 chunks, and hit@5 = 1
  iff all gold chunks are in the top 5. `kg_silver` comes from the released evidence relations
  keyed by gold chunk. `kg_openie` is one OpenIE KG over the corpus, identical for both splits
  (identical injection fingerprint). It was built once, for holdout200. The sub200 run uses a
  verified copy of that index, and the graph verification is rerun against the sub200 injection.
- Passage vectors are reused across conditions or splits only when the set of passage strings is
  identical (hash ids and texts compared). The dense-only ranking check in `summary.md` confirms it.
- Encoding ran on CPU with `OMP_NUM_THREADS=10`.
- LLM gate: the driver waits before retrieval until Ollama has had no other model active for
  3 minutes (`wait_for_llm_slot`), so that the filter model has the GPU to itself. Indexes can be built
  separately with `--index-only`. A completed build (`build_graph_info.json`) is required before
  any index is reused. The gate affects only when the filter calls run, not what they return
  (temperature 0; cached responses are reused on reruns).
- "Synonymy edges" in `summary.md` are counted as entity–entity graph edges minus twice the number
  of entity pairs that carry a fact (HippoRAG stores fact links in both directions).
  HippoRAG's own `get_graph_info()["num_synonymy_triples"]` subtracts unique facts instead and does
  not count synonymy edges.

## Files

- Drivers: `eval/_kg_swap_hipporag2.py` (index + retrieve, env `rag-graphrag-py311`),
  `eval/_kg_swap_hipporag2_score.py` (scoring + summary, project env),
  `eval/_e2e_qa_hipporag2.py` (reader).
- Per-question vectors (top-5 ids, scored): `external/<bench>/<split>/_kgswap_hipporag2_<kg>.json`.
- Summary: `summary.md`, `summary.json`; reader: `e2e_qa/qwen2_5_14b/summary.md`.
