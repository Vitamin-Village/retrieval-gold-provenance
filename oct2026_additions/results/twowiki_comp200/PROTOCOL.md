Released copy: passages outside the scope of this paper are omitted and marked [omitted]; SHA-256 of the full protocol as written: e7ad7a9debc98d42407a8a237d8644addbc077527ce90f7472de5f4373334d83.

# Protocol: 2WikiMultiHop compositional split (`comp200`) for the controlled KG swap

Fixed 2026-10-02T14:20Z [omitted], before the split was built and before any retrieval,
OpenIE, HippoRAG or reader run on it. [omitted] Nothing below is changed after results are seen; any departure that becomes necessary is
recorded as a deviation in [omitted] `analysis/twowiki_comp200_summary.md`, with the reason.

**Every result of this protocol is reported whatever its value or direction**, including null,
reversed or non-significant swaps, failed sanity checks and runs that could not be completed.

## Purpose

The paper's 2WikiMultiHop swap uses two 200-question splits (`sub200`, `holdout200`) that both
contain only `inference`-type questions (`select_subsample` takes `PREFERRED_TYPES[0]` first and the
inference pool is large enough to fill 200). [omitted] This protocol adds one
200-question split of the `compositional` type, built and evaluated exactly like the existing splits.

## 1. Selection

- Source: `external/2wiki/raw/data_ids/dev.json` (12,576 rows; 5,236 of type `compositional`, all
  with two supporting titles, two evidence triples and ten context paragraphs).
- Pool: rows with `type == "compositional"` whose `_id` is not a `qa_id` of `external/2wiki/sub200/qa.jsonl`
  or `external/2wiki/holdout200/qa.jsonl` (both splits are inference-only, so this exclusion is a
  no-op by type; it is still applied and asserted).
- The existing pipeline (`eval/adapters/make_holdout200.py`) enforces QA-id disjointness only; it does
  not exclude questions whose passages overlap another split's pool. The same rule is used here.
  Passage overlap of the new pool with the `sub200` and `holdout200` pools is reported as a sanity
  check, not used for selection.
- Draw: `select_subsample(pool, limit=200, seed=20261002, preferred_types=("compositional",))`
  from `eval/adapters/twowiki_adapter.py`, unchanged (seeded `random.Random`, shuffle, take 200).
  Seed 20261002 follows the date convention of the existing seeds (20260530, 20260531). One draw;
  no redraw.

## 2. Corpus, gold and aligned KG (identical builders)

Driver: `eval/adapters/make_comp200.py` (new). It calls the adapter's functions unchanged, in the
same order as `make_holdout200.py`:
`build_passages_and_manifest` (corpus = deduplicated union of each question's ten context
paragraphs, `passage_id = paper_id = normalize_title(title)`, first-seen text), `build_qa`
(gold = unique supporting-fact titles, `evidence_spans`), and
`build_silver_kg(rows, id_aliases=load_id_aliases(external/2wiki/raw/data_ids/id_aliases.json))`
for the aligned KG `kg_silver/` (the questions' Wikidata evidence triples; node `source_docs` =
supporting passages whose title the entity surface matches). Output:
`external/2wiki/comp200/{qa.jsonl, passages.jsonl, manifest.json, comp200_manifest.json, kg_silver/}`.

[omitted]

## 3. Independent KG (same extractor, no gold access)

`baselines/build_openie_kg.py --passages external/2wiki/comp200/passages.jsonl --out-dir
external/2wiki/comp200/kg_openie --model llama3.1:8b --synonyms surface --verbose`, unchanged:
same `OPENIE_PROMPT`, one NER+OpenIE call per passage (text truncated at 3,500 characters),
temperature 0, `num_predict` 700, seed 0, one JSON-retry, surface token-containment synonym edges
(min 2 tokens), checkpoint every 50 passages. The builder reads only `passages.jsonl`; it never sees
`qa.jsonl`, evidences or supporting facts. Environment: conda `new_env` (Python 3.9.21, `ollama`
0.6.1), Ollama 0.34.3 at 127.0.0.1:11434, `llama3.1:8b` digest `46e0c10c039e` (Q4_K_M), the same
model digest as the existing OpenIE KGs and HippoRAG runs. [omitted]

## 4. Retrievers and frozen configurations

All from `paper.eval.eval_independent` with `top_k=5`, run in `new_env` with
`RAG_V3_DENSE_BACKEND=deterministic` (the deterministic dense surrogate of the existing main runs),
default `V3Weights`, no tuning, no change of any retriever code:

| paper name | system | per-question vector file (in `external/2wiki/comp200/`) | produced by |
|---|---|---|---|
| BM25 | `bm25` | inside every file below (must be identical across files) | |
| Hybrid ("audited") | `proposed_v3` | `_kgswap_kg_{silver,openie}.json` | `run_kg` of `eval/_kg_swap_2wiki.py` |
| PPR | `ppr` | `_kgswap_ppr_kg_{silver,openie}.json` | `run_kg(..., "ppr")` of `eval/_kg_swap_2wiki_r2.py` |
| Hybrid-PPR ("graph_centric") | `proposed_v5_graphcentric` | `_kgswap_proposed_v5_graphcentric_kg_{silver,openie}.json` | `run_kg(..., "proposed_v5_graphcentric")` of `eval/_kg_swap_2wiki_r2.py` |

The two existing drivers hard-code the split list and overwrite `results/kg_swap_2wiki*.json`, so
they are not run. A new driver, `eval/_kg_swap_2wiki_comp200.py`, imports their `run_kg` functions
unchanged and calls them for `paper/external/2wiki/comp200` only; it writes its statistics to
`results/twowiki_comp200/`. [omitted]

## 5. Metric and statistics

- hit@5 = 1 iff all gold passages of the question (two) are in the top five (`eval_independent.hit_at_k`).
- KG swap, per retriever: per-question paired difference aligned minus independent over the 200
  questions; mean; 10,000-resample percentile bootstrap 95% CI with Python `random`, re-seeded
  immediately before resampling with the seed of the driver that produced the existing statistic
  (20260608 for Hybrid as in `eval/_swap_stats.py`; 20260611 for PPR and Hybrid-PPR as in
  `eval/_kg_swap_2wiki_r2.py`; 20260918 for HippoRAG 2 as in `eval/_kg_swap_hipporag2_score.py`);
  exact two-sided sign test over discordant questions. The swap delta of `results/kg_swap_2wiki.json`
  (Delta aligned minus Delta independent) is also reported; it equals the mean paired difference.
- Graph over BM25, per retriever and KG arm: margin = graph hit@5 minus BM25 hit@5 on the same
  questions, exact two-sided sign test on the per-question pairs, verdict *beats* (margin > 0, p < 0.05),
  *below* (margin < 0, p < 0.05) or *n.s.*, uncorrected, using `paired`/`verdict` imported from
  `analysis/headline_margins.py` [omitted].
  Sensitivity: Holm over the comp200 cells of one arm.

## 6. End-to-end readers

`eval/e2e_qa_swap.py` unchanged (same prompt, temperature 0, seed 1, `num_predict` 48, answer
cleaning, alias-aware EM with the Wikidata aliases of `answer_id`, F1, cover-EM, paired statistics
with numpy bootstrap seed 20260918, B = 10,000, exact sign test), through a new wrapper
`eval/_e2e_twowiki_comp200.py` that, like `eval/_e2e_third_reader.py`, restricts the retrieval
conditions to the aligned/independent KG vectors (`kg_silver`, `kg_openie`) of every retriever present
in `external/2wiki/comp200/`, plus BM25, closed-book and gold-only (oracle), and passes
`--benchmarks 2wiki --splits comp200 --out results/twowiki_comp200/e2e_qa`. Readers: `llama3.1:8b`
(Llama-3.1-8B) and `qwen2.5:14b` (Qwen2.5-14B) via Ollama; each gets its own new cache under
`results/twowiki_comp200/e2e_qa/<tag>/`, so no existing reader file is touched. Reported per reader
and retriever: EM aligned / independent / BM25, paired answer swap (mean, CI, p), and the
graph-over-BM25 EM verdicts per arm as in Section 5. If the HippoRAG 2 vectors (Section 7) exist
when a reader runs, they are included as retriever `hipporag2`; if they are produced later, the
wrapper is re-run (cached prompts are not regenerated).

## 7. HippoRAG 2 (priority 3)

`eval/_kg_swap_hipporag2.py` unchanged, through a new wrapper `eval/_kg_swap_hipporag2_comp200.py`
that only (a) adds `comp200` to the accepted `--split` values and (b) points the run directory at
`results/twowiki_comp200/hipporag2/runs/`. Same configuration as Appendix C.7 / `results/hipporag2_swap/METHODS.md`:
conda env `rag-graphrag-py311`, `hipporag` 2.0.0a4 from the same clone, `PYTHONUTF8=1`,
`HF_HUB_OFFLINE=1`, `BaseConfig` defaults except `save_dir`/`llm_name`/`llm_base_url`/`embedding_model_name`,
BGE-M3 through the `Transformers/` backend (CPU, as in C.7), `llama3.1:8b` as the recognition-memory
filter LLM, node-provenance injection of each KG through HippoRAG's OpenIE cache, OpenIE stubbed to
raise, graph verification asserts, `retrieve(num_to_retrieve=20)`, top five scored, LLM gate on.
Scoring with `vectors`/`paired` of `eval/_kg_swap_hipporag2_score.py` (shared scorer, seed 20260918)
into `external/2wiki/comp200/_kgswap_hipporag2_kg_{silver,openie}.json`, statistics to
`results/twowiki_comp200/`. BM25 reference = the comp200 BM25 vector. No edge-provenance sensitivity run.

## 8. Sanity checks (reported)

Question-type counts; gold passages per question; corpus size and its overlap with the sub200 /
holdout200 pools; QA-id overlap (must be 0); aligned and independent KG node/edge counts, OpenIE
parse-failure rate; passages linked by each KG, linked gold, gold fraction of linked and gold
covered, computed with `counts` of `analysis/kg_passage_link_counts.py` [omitted]; BM25 hit@5 on comp200 against 0.365
(sub200) and 0.355 (holdout200); BM25 vector identical across all comp200 vector files; HippoRAG 2
dense-only ranking identical across its two conditions.

## 9. Scope and ordering

Priority: (1) data, both KGs and the retrieval swap for Hybrid, PPR, Hybrid-PPR and BM25;
(2) both readers; (3) HippoRAG 2. [omitted] No paid API is called. Not in scope: BGE-M3
dense-channel reruns, dense-only rankings and Kendall tau, dose / vocabulary variants, LLM judges.
[omitted]
