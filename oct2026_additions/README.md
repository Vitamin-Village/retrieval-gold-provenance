# Second part of the released materials

Paths below are relative to this folder, `oct2026_additions/`. The first part of the materials is under `aug2026_original/`, next to this folder.

`aug2026_original/` holds the August 2026 materials that support this paper, edited for this release; every reported number is unchanged.

Every number in the paper that is not in `aug2026_original/` is traceable to a file here. Documents and script docstrings (PROTOCOL.md, METHODS.md, DESIGN.md) cite paths of the authors' working tree, relative to `paper/`; see Working-tree layout. Names used in files are mapped to the paper's terms below.

## Names used in files

| Name in files | Term in the paper |
|---|---|
| `sub200`, `holdout200` | the first split and the held-out split (2WikiMultiHop, MetaQA, STaRK-Prime, GraphRAG-Bench) |
| `comp200` | the compositional split of 2WikiMultiHop (Appendix C.1, Question type; Table 23) |
| `kg_silver`, `silver` | the aligned KG |
| `kg_openie`, `openie` | the independent (OpenIE) KG |
| `kg_wikidata` | the KG fetched from live Wikidata, a realistic structure for 2WikiMultiHop (Table 29) |
| `kg_fullkb` | the full MetaQA KB restricted to the corpus, a realistic structure (Table 31) |
| `proposed_v3`, `audited` | the hybrid retriever |
| `proposed_v5_graphcentric`, `graph_centric` | the hybrid-PPR retriever |
| `ppr` | PPR |
| `hipporag2` | HippoRAG 2 |
| `bm25` | BM25 |
| `dense` | dense-only retrieval |
| `closedbook`, `oracle` (conditions) | closed-book and gold-only answering |
| `@bge`, `bge_kg_silver`, `bge_kg_openie`, `_dense_bge_m3` | the retrievers with a BGE-M3 dense channel, mean pooling: the mean-pooled variant reported in Appendix C.1 (in `results/e2e_qa_cls/` the same names are the [CLS]-pooled cells) |
| `_cls`, `_dense_bge_m3_cls` | the BGE-M3 dense channel with [CLS] pooling (Table 19) |
| `_kgswap_<system>_<kg>.json` | frozen per-question retrieval vectors of one retriever under one KG; `_kgswap_kg_<kg>.json` holds the hybrid retriever and BM25 |
| `dose_f00` to `dose_f100` | the aligned KG with a fraction f of its edges kept (Table 20, Figure 6) |
| `silver_dilute_g31`, `silver_dilute_g50`, `silver_dilute_g75` | gold-concentration controls: the aligned KG diluted to a gold fraction of 0.31, 0.50, 0.75 of its linked passages (Table 30) |
| `openie_goldonly` | the independent KG with only its links to gold passages (Table 30) |
| `kg_silver_noedges` | the aligned KG with every edge deleted (Table 21) |
| `kg_openie_svocab` | the independent KG with the aligned KG's entity inventory grafted in, gold node-to-passage mapping kept (Transplant, Table 24) |
| `kg_openie_svocab_noprov` | the same graft with that mapping emptied (Mapping emptied, Table 24) |
| `kg_openie_svocab_q25`, `_q50`, `_q75` | the graft with the mapping kept on a fraction q = 0.25, 0.50, 0.75 of the grafted nodes (Table 33) |
| `kg_openie_resorted` | the independent KG with its nodes and edges in the sorted order of the graft files, vocabulary unchanged (an order control written by the graft builder) |
| `qa_v4` | the upper calibration anchor (Ours, upper, in Table 1 and Table 8) |
| `OURSUP_*`, `OURSLOW_*` | the axes of the upper and the lower calibration anchor |
| primary LM rater A, B (`primary LM A/B`) | the two primary census LM raters (Table 9, Table 10) |
| condition A (in places `workbook`), condition B (`excerpt-only`) of `human_validation/llm_census_logged/` | the description condition and the excerpt-only condition of the logged LM raters (Appendix B.5) |
| R1, R2, R3 in `human_validation/v4/` | Humans 1, 2, 3 (Table 11) |
| `human_validation/v3/` | the rater materials of the logged LM raters (Appendix B.5) |
| `human_validation/llm_census_logged/` | the logged LM census raters (Appendix B.5) |
| `human_validation/v4/` | the three human census raters (Appendix B.4) |
| `human_validation/judge_spotcheck_v1/` | the first human rater of the GraphRAG-Bench judge (Appendix C.10, Judge validation) |
| `human_validation/judge_spotcheck_v4/` | the three human raters of the judge (Appendix C.10) |
| `human_validation/llm_spotcheck_v3/` | the labels of three Claude models for the same 100 judge verdicts (Appendix C.10); R1, R2, R3 = claude-opus-5, claude-sonnet-5, claude-haiku-4-5 |

## Files

| Path | Contents | Produced by |
|---|---|---|
| code/ | the scripts behind the numbers, tables and figures (Code layout, Reproduction); one LaTeX file, tab_quotes_full.tex (Tables 6 and 7), is typeset without a script | - |
| results/e2e_qa/<reader>/summary.md, summary.json | end-to-end QA: paired swap on EM/F1/cover, mediation, all cells (incl. closed-book, gold-only, dose, BGE-M3 conditions); readers llama3_1_8b, qwen2_5_14b, mistral_small_24b (third reader, core 2WikiMultiHop conditions only), and claude_sonnet_5_5 (proprietary reader, same conditions; PROTOCOL.md, batch_state.json, batch_results_raw.jsonl with every batch result object, pilot_raw.jsonl with the three single requests that checked the parameters before the batch and are not scored) | code/e2e_qa_swap.py; code/_e2e_third_reader.py for mistral_small_24b; code/_e2e_claude_reader.py for claude_sonnet_5_5 |
| results/e2e_qa/<reader>/per_question.jsonl | one scored row per (benchmark, split, condition, retriever, question) | code/e2e_qa_swap.py |
| results/e2e_qa/<reader>/generations.jsonl | raw reader outputs keyed by prompt hash (temperature 0) | code/e2e_qa_swap.py |
| results/e2e_qa/<reader>/judge_<judge>.md/.json | LM-judge accuracy on GraphRAG-Bench (primary judge mistral-small:24b; cross-reader second judge) | code/e2e_qa_judge.py |
| results/e2e_qa/judge_cache_<judge>.jsonl | judge verdicts keyed by (question, gold, prediction) | code/e2e_qa_judge.py |
| results/e2e_qa/<reader>/memorization_2wiki.jsonl, memorization_strata.md/.json | training-exposure probe: closed-book recall of every 2WikiMultiHop evidence triple, and the answer-level swap stratified by closed-book answer and evidence recall (all four benchmarks) | code/e2e_memorization_probe.py |
| results/e2e_qa/variant_pairs.md/.json | answer-level pairs for the realistic structures (live Wikidata, full MetaQA KB), Appendix C.3 | code/_e2e_variant_pairs.py |
| results/e2e_qa_cls/<reader>/summary.md/.json, per_question.jsonl, judge_mistral_small_24b.md/.json | end-to-end QA with the BGE-M3 cells re-run on the [CLS]-pooled vectors (all other cells are cache hits and identical to results/e2e_qa/); readers llama3_1_8b, qwen2_5_14b; judge verdicts in results/e2e_qa_cls/judge_cache_mistral_small_24b.jsonl. results/e2e_qa_cls/mistral_small_24b/summary.json is the third reader's summary, the same file as results/e2e_qa/mistral_small_24b/summary.json (that reader has no BGE-M3 cell), so that code/e2e_dose_across_cells.py --res results/e2e_qa_cls finds all three readers | code/e2e_qa_swap.py --bge-variant cls, code/e2e_qa_judge.py --res results/e2e_qa_cls |
| results/e2e_dose_across_cells.md/.json, e2e_dose_across_cells_cls.md/.json | answer-level swap against retrieval-level swap across every cell (mean-pooled and [CLS]-pooled BGE-M3 cells) | code/e2e_dose_across_cells.py (--res results/e2e_qa_cls --suffix _cls) |
| results/tables/e2e_numbers.json | numbers behind Table 3 and the prose | code/e2e_tex_tables.py |
| results/tables/e2e_multiple_comparison.json, dose_curves.json, tab_e2e_swap_em.md, tab_e2e_swap_f1.md, tab_decirc.md | Holm/BH corrections, dose curves, swap and de-circularization tables in Markdown | code/e2e_qa_figures.py, code/e2e_tex_tables.py |
| (not shipped) external/2wiki/sub200/_dose_work/ | the dose-curve KGs (`f00/` to `f100/`) and the per-question dose retrieval outputs (`eval_f00.json` to `eval_f100.json`) behind the cells `dose_f00` to `dose_f100`; `aug2026_original/swap/code/_swap_dose_response.py` (seed 20260608) regenerates them deterministically from the aligned KG and must run before `code/e2e_qa_swap.py` scores the dose cells (like every retrieval run, it needs the 2WikiMultiHop passage text) | aug2026_original/swap/code/_swap_dose_response.py |
| results/tables/*.tex | the LaTeX tables the manuscript inputs: tab_e2e_merged_v3.tex (Table 3) by code/_tab_e2e_v3.py; tab_memorization.tex and tab_memorization_main.tex by code/_tab_memorization_v3.py; tab_readerref.tex, tab_mediation.tex and tab_linkcounts.tex by code/_tab_appendix_v3.py. tab_third_reader.tex by code/larger_readers_table.py from results/e2e_qa/mistral_small_24b/ and claude_sonnet_5_5/summary.json. Typeset without a script: tab_quotes_full.tex (Tables 6 and 7), the verbatim construction quotes of the 29 census judgments from the cited benchmark papers | see Contents |
| manuscript/eacl2027_camera_ready.tex, eacl2027_camera_ready.aux | the LaTeX source of the published paper and the label file of its compile; code/headline_margins.py reads them for its Table 3 and Table 36 cross-checks | - |
| results/kg_passage_link_counts.json | passages linked by the aligned and the independent KG and their gold fraction, the first and held-out splits of every benchmark (Table 28; the compositional split is in results/twowiki_comp200_summary.*) | code/kg_passage_link_counts.py |
| results/census_kappa_ci_output.txt | the block of the console output behind the census agreement table (Table 9; all 29 judgments; qa_v4 is the upper anchor) | aug2026_original/census/reliability_kappa_ci.py |
| results/swap_samplesize_output.txt | console output behind the sample-size table (Table 22) | aug2026_original/swap/code/_swap_samplesize.py |
| external/metaqa/<split>/_kgswap_{kg,ppr_kg,proposed_v5_graphcentric_kg}_{silver,openie}.json | MetaQA frozen per-question retrieval vectors of the swap (Hybrid, PPR, Hybrid-PPR with the aligned and the independent KG), at their working-tree path; those of 2WikiMultiHop and STaRK are in aug2026_original/ (swap/data/, stark/data/) | aug2026_original/metaqa/code/_kg_swap_metaqa.py |
| results/graphragbench_swap_summary.md, kg_swap_graphragbench*.json, swap_coverage_graphragbench.json | fourth swap | code/_kg_swap_graphragbench.py |
| graphragbench_swap/ | design note, build report, per-split qa, silver-input relations, KG node/edge files, frozen retrieval vectors; corpus_chunks_sha256.jsonl (id, title, length and SHA-256 of every corpus chunk; the chunk text is not included, CORPUS_TEXT_REMOVED.md gives the rebuild steps); LICENSE_GraphRAG-Bench.txt | code/build_graphragbench_swap_data.py, code/build_openie_kg.py |
| results/dense_bge_swap_summary.md, kg_swap_dense_bge_m3.json; dense_bge/<bench>/<split>/*.json | the swap with a BGE-M3 dense channel, mean pooling: the mean-pooled variant reported in Appendix C.1; GraphRAG-Bench vectors in graphragbench_swap/<split>/_dense_bge_m3/ | code/_kg_swap_dense_bge.py --pooling mean |
| results/dense_bge_swap_summary_cls.md, kg_swap_dense_bge_m3_cls.json; dense_bge/<bench>/<split>/_dense_bge_m3_cls/ | the swap with a BGE-M3 dense channel, [CLS] pooling, the official BGE-M3 dense vector (Table 19), all four benchmarks | code/_kg_swap_dense_bge.py (--pooling cls, default) |
| results/bge_pooling_check.json, bge_pooling_check_stark_graphragbench.json | dense-only hit@5 of BGE-M3 under each pooling and input format | code/_bge_pooling_check.py |
| realistic_structure/, results/kg_fullkb_metaqa.json, fullkb_metaqa_summary.md | MetaQA full-KB KG and vectors | code/_kg_fullkb_metaqa.py |
| realistic_structure/2wiki/, results/kg_swap_variant_2wiki_kg_wikidata.json, variant_2wiki_kg_wikidata_summary.md | 2WikiMultiHop live-Wikidata KG and vectors | code/build_wikidata_kg_2wiki.py, code/_kg_swap_variant.py --bench 2wiki --variant kg_wikidata |
| concentration/, results/kg_concentration_2wiki.json, concentration_2wiki_summary.md | dilution and gold-only KGs and vectors | code/_kg_concentration_2wiki.py |
| zero_edge/, results/kg_zero_edge_2wiki.json, zero_edge_2wiki_summary.md | aligned 2WikiMultiHop KG with every edge deleted, all three retrievers, both splits | code/_kg_zero_edge_2wiki.py |
| hipporag2_swap/ | HippoRAG 2 under the KG swap (Appendix C.7): METHODS.md, summary.md/.json, frozen retrieval vectors for 2WikiMultiHop (2wiki/, with _kgswap_hipporag2_kg_openie_edgeprov.json, the edge-provenance sensitivity of METHODS.md) and MetaQA (metaqa/; full-KB vectors in realistic_structure/metaqa/), and the Qwen2.5-14B run on 2WikiMultiHop that METHODS.md describes (e2e_qa/qwen2_5_14b/); GraphRAG-Bench vectors in graphragbench_swap/<split>/. The answer-level results of HippoRAG 2 for all readers (system hipporag2) are in results/e2e_qa/<reader>/ | code/_kg_swap_hipporag2.py, code/_kg_swap_hipporag2_score.py, code/_e2e_qa_hipporag2.py |
| hipporag2_swap/runs/<run>/retrieval.jsonl | per-question retrieval of every HippoRAG 2 run (Appendix C.7), one row per question: ranked passage ids and scores, the dense-only ranking (`dpr_ranked_passage_ids`), the facts (triples of the injected KG) before and after the recognition-memory filter, the filter error if any, and `path` (`ppr`, or `dpr_fallback` when the filter kept no fact). <run> is <split>__<kg> for 2WikiMultiHop and <bench>__<split>__<kg> for MetaQA and GraphRAG-Bench; `__edgeprov` marks the edge-provenance sensitivity of METHODS.md | code/_kg_swap_hipporag2.py |
| external/2wiki/comp200/ | the compositional 2WikiMultiHop split (Appendix C.1, Question type): qa.jsonl, comp200_manifest.json (selection record), manifest.json (passage records with the text removed, PASSAGE_TEXT_REMOVED.md), the aligned (kg_silver/) and independent (kg_openie/) KG files, and the frozen per-question vectors _kgswap_*.json of the hybrid, PPR, hybrid-PPR and HippoRAG 2 retrievers (BM25 in each file of the first three) | code/make_comp200.py, code/build_openie_kg.py, code/_kg_swap_2wiki_comp200.py, code/_kg_swap_hipporag2_comp200.py, code/_kg_swap_hipporag2_comp200_score.py |
| results/twowiki_comp200/PROTOCOL.md | the protocol of the compositional split, fixed before the split was built (released copy, see Code layout) | - |
| results/twowiki_comp200/e2e_qa/<reader>/summary.md/.json, per_question.jsonl, generations.jsonl | end-to-end QA on the compositional split, readers llama3_1_8b and qwen2_5_14b | code/_e2e_twowiki_comp200.py (runs code/e2e_qa_swap.py) |
| results/twowiki_comp200/hipporag2/summary.md/.json | HippoRAG 2 on the compositional split | code/_kg_swap_hipporag2_comp200_score.py |
| results/twowiki_comp200_summary.md/.json | the compositional split: retrieval swap, graph-over-BM25 margins, answer-level swap, link counts and corpus checks, with the first and held-out splits for comparison | code/twowiki_comp200_summary.py |
| results/index_precision_summary.md/.json | index precision at question-matched KG nodes against controlled-swap magnitude | code/index_precision.py |
| results/headline_margins_summary.md/.json, results/figures/fig_headline_margins.pdf | graph-over-BM25 hit@5 margins under the aligned and the independent KG and, on 2WikiMultiHop and MetaQA, the realistic structures, verdict flips, retriever-ranking agreement, answer-level repeat; the paper's Figure 2 | code/headline_margins.py |
| results/published_pairs_classification.md, published_pairs_class_stats.json | the 30 published graph-versus-lexical pairs classified by the structure the graph system retrieved over, with quotes, and descriptive statistics per class | code/published_pairs_class_stats.py (classification by hand, quotes from the cited papers) |
| results/grade_stark_provenance/summary.md, summary.json, per_query_stark.jsonl | GRADE and STaRK provenance trace (deterministic, lexical, no model calls) | code/_grade_stark_provenance_trace.py |
| results/grade_stark_provenance/inputs/ | STaRK-Amazon first-release query ids | GitHub snap-stanford/stark dfafa99 / HF f721c636 |
| results/grade_stark_provenance/sources/ | quote sources and traced code: the public OpenReview discussion of STaRK (ids, dates, content only), HF file listings, copies of the traced GRADE (Apache-2.0) and STaRK (MIT) files with their licenses and THIRD_PARTY_NOTICE.md, and file:line references (with the quoted lines) for STaRK hybrid_qa.py at commit df59d09, which is not copied; no knowledge-base dump | - |
| results/openie_extractor_exposure/summary.md, summary.json, per_triple.jsonl | extractor-exposure check of the independent OpenIE KGs: gold-matching triples whose endpoint is absent from the source passage | code/_openie_extractor_exposure.py |
| results/index_precision_PREREG.md | the index-precision declaration (primary predictor, test and bootstrap), written before any index-precision number was computed | - |
| results/grb_provenance/summary.md, summary.json, per_question.jsonl | GraphRAG-Bench provenance trace (deterministic, lexical, no model calls) | code/_grb_provenance_trace.py |
| results/grb_provenance/inputs/PINNED_INPUTS.md | the three GraphRAG-Bench question files the trace reads, pinned by commit (e18981f, 0b8f46a), path, size and SHA-256, with a download command; the commits predate the repository's LICENSE file, so the files are not copied | GitHub GraphRAG-Bench/GraphRAG-Benchmark |
| results/grb_provenance/sources/ | the public OpenReview discussion (forum i9q9xDMjG7) quoted in the trace. The upstream construction scripts are third-party code and are not redistributed: GitHub GraphRAG-Bench/GraphRAG-Benchmark, commit 431e376 (1_kg_generate.py, 2_kg_post_process.py, 2_2_kg_extract_path.py; deleted in e6305f5) | - |
| human_validation/llm_census_logged/ | logged LM re-application of the census procedure (Appendix B.5; LM raters, not humans): PROTOCOL.md (released copy, see Code layout; fixed before the runs, deviations appended), raw.jsonl (every request and reply with model id, parameters, usage, timestamp), runs/, result.md/.json (agreement of the logged runs with the census, regenerated from raw.jsonl by the released code/llm_census_logged.py --score; the paper compares the raters with the human re-scorers in results/rater_validation_summary.* and results/rater_robustness_summary.*), condition-B excerpts, exact prompts, model parameters, pilot_raw.jsonl (parse check, not used in statistics) | code/llm_census_logged.py |
| human_validation/v3/ | the rater materials the logged LM raters used (Appendix B.5): instructions with worked examples, quiz key, answer key, blank workbooks (author fields blanked), package builder. The builder reads human_validation/items_29.json, reference_bins.json and items_partB.json and writes into its own folder: it reproduces answer_key_v3.json, QUIZ_KEY_v3.md and _worked_examples.md byte for byte and the Quiz, PartA and PartB sheets cell for cell, and also writes a Rater_info sheet (rater background and attestations) that the released blank workbooks, which the logged runs used, do not have, and build_manifest_v3.json (seed and row orders) | human_validation/v3/build_package_v3.py |
| human_validation/reference_bins.json, items_29.json | census reference bins and the 29 Part A items (same bins and items as aug2026_original/human_validation/) | - |
| human_validation/items_partB.json, lexical_rules.py | the 130 Part B units shown in the workbooks, and the deterministic lexical rules applied to them | - |
| results/multimodel_census_raw.json | headline-axis bins of the five cross-model LM raters (same data as aug2026_original/census/cross_model_census_raw.json) | aug2026_original/census/cross_model_census.py |
| (not released) | files cited by the protocols or scripts that hold no data used in the paper are not included (the released protocol copies mark such passages [omitted]); the scripts read API keys from a `.env` file in `paper/` or its parent folder, or from the environment, and no key is released | - |
| human_validation/v4/ | the three human raters of Appendix B.4: PROTOCOL_v4.md (released copy, see Code layout; fixed before any workbook was sent; deviations appended), INSTRUCTIONS_v4.md, WORKED_EXAMPLES_v4.md, CHECKLIST_v4.md, build_package_v4.py, SEALED_build.json (seeds, orders, SHA-256 of every sent file, item text, and locked cell; answer keys), to_send/ (the zips sent), score_v4.py and result_v4.md (integrity checks, eligibility, every statistic, alternative annotator test), independence_checks_v4.py/.md (follow-up on the scorer's warnings), export_public_v4.py and public/ (ratings, quiz answers, rater background and attestations, SHA-256 of the returned workbooks). The returned workbooks and result_v4.json are not released: their document properties and save paths name the raters' institution, and Rater_info holds the raters' initials. score_v4.py and independence_checks_v4.py therefore need the returned workbooks to run, and export_public_v4.py needs result_v4.json; the analyses run from public/ | human_validation/v4/score_v4.py |
| human_validation/rating_sheet_external.csv, external_rater_result.json | the short-prompt human re-scoring of the 29 census judgments, by a non-author blind to the census bins: filled sheet and scores | aug2026_original/human_validation/score_external_rater.py |
| results/rater_validation_summary.md/.json | agreement of every census rater (census, primary LM raters, logged LM raters, the short-prompt human) with the census, with each v4 human rater, and with their median | code/rater_validation.py |
| results/rater_robustness_summary.md/.json | census conclusions recomputed under each human and LM rater | code/rater_robustness.py |
| human_validation/llm_spotcheck_v3/ | LM labels (Appendix C.10) of the same 100 judge verdicts by claude-opus-5, claude-sonnet-5, claude-haiku-4-5 under the rater guide (LM labels, not human): raw replies with model ids and timestamps, filled sheets, scoring (re-scored offline by package/score_spotcheck_v3.py; see Reproduction) | code/llm_spotcheck_claude.py |
| human_validation/judge_spotcheck_v1/ | the first human rater of the judge (Appendix C.10): one non-author's labels of the 100 judge verdicts (the three raters are in judge_spotcheck_v4/), sealed verdicts, scoring (the scorer runs from inside the folder) | human_validation/judge_spotcheck_v1/score_spotcheck.py |
| human_validation/judge_spotcheck_v4/ | the three human raters of the judge, Appendix C.10: PROTOCOL_judge_v4.md (released copy, see Code layout; fixed before any workbook was sent; two deviations appended), GUIDE_v4.md (the guide sent to the raters), CHECKLIST_judge_v4.md (coordinator), build_package_judge_v4.py (deterministic builder; --check rebuilds in memory and compares), SEALED_judge_build.json (sheet password, seeds, row orders and codes, SHA-256 of every sent file, item text, locked cell and input; attention answers), to_send/ (the zips sent), score_judge_v4.py and result_judge_v4.md (integrity checks, eligibility, every statistic, comparisons with the first human rater and the LM labels), export_public_judge_v4.py and public/ (labels and notes, rater background and attestations, SHA-256 of the returned workbooks). The returned workbooks and result_judge_v4.json are not released: their document properties and save paths name the raters' institution, and Rater_info holds the raters' initials. score_judge_v4.py therefore needs the returned workbooks to run (its --exclude CODE=REASON option was not used in the final run), and export_public_judge_v4.py needs result_judge_v4.json; the scripts expect the working-tree layout (judge_spotcheck_v3/ and judge_spotcheck/, see Working-tree layout) | human_validation/judge_spotcheck_v4/score_judge_v4.py |

## Licenses and third-party data

- Author-written code, analyses and annotations: as stated in `aug2026_original/LICENSE.md` (code under the MIT License; annotations, ratings and results under CC BY 4.0).
- GraphRAG-Bench: the GitHub repository GraphRAG-Bench/GraphRAG-Benchmark and the Hugging Face dataset GraphRAG-Bench/GraphRAG-Bench declare the MIT License (Copyright (c) 2025 XMU-DeepLIT; text in `graphragbench_swap/LICENSE_GraphRAG-Bench.txt`). Included from it: the medical questions, answers, evidence sentences and evidence relations of the two swap splits (`graphragbench_swap/<split>/qa.jsonl`, `kg_silver_input.jsonl`), and question and answer texts in the end-to-end, judge and spot-check files. Not included: the medical corpus text, which is the text of the NCCN Guidelines (copyright NCCN) and which the benchmark's license does not clearly cover (`graphragbench_swap/CORPUS_TEXT_REMOVED.md`: chunk hashes and rebuild steps; the example passages quoted in `results/openie_extractor_exposure/summary.*` are replaced by a placeholder), and three question files taken at commits that predate the repository's LICENSE file (`results/grb_provenance/inputs/PINNED_INPUTS.md`). The KGs built from the corpus (`graphragbench_swap/<split>/kg_openie/`, OpenIE) hold the extracted entity and relation phrases, which are short spans of the corpus text.
- OpenReview: `results/grb_provenance/sources/openreview_forum_i9q9xDMjG7_notes.json` (GraphRAG-Bench) and `results/grade_stark_provenance/sources/openreview_forum_QSS5cGmKb1_notes.json` (STaRK) are the public discussions of the two submissions, reduced to note ids, dates and content (no signatures, readers, writers, invitations or profile ids), included as the quote sources of the provenance traces; their text remains its authors' under the OpenReview terms of use.
- GRADE and STaRK code: the traced files in `results/grade_stark_provenance/sources/` ship with their repositories' licenses (GRADE: Apache-2.0 with an MIT text; STaRK: MIT); see `THIRD_PARTY_NOTICE.md` there.
- Wikidata: the live-Wikidata KG of 2WikiMultiHop (`realistic_structure/2wiki/<split>/kg_wikidata/`, and the vectors and HippoRAG 2 runs built on it) is derived from Wikidata, whose structured data are available under CC0 1.0.
- MetaQA: the full-KB KG (`realistic_structure/metaqa/<split>/kg_fullkb/` and its vectors) is derived from the MetaQA knowledge base (kb.txt, built from WikiMovies); see `aug2026_original/DATA_LICENSES.md`, `aug2026_original/licenses/metaqa_acquisition.md` and `aug2026_original/licenses/WIKIMOVIES_CC_BY_3.0.txt`.
- The other 2WikiMultiHop-, MetaQA- and STaRK-derived files (KGs, splits, per-question vectors) follow `aug2026_original/DATA_LICENSES.md`; as in `aug2026_original/`, no 2WikiMultiHop/Wikipedia passage text is included (the 2WikiMultiHop example passages quoted in `results/openie_extractor_exposure/summary.*` are replaced by a placeholder; the OpenIE KG files hold extracted phrases, which are short spans of the passages).

## Working-tree layout

The scripts resolve every input and output from their own location in the authors' working tree, as `<repo>/paper/<eval|analysis|baselines>/<script>.py`. Here `<repo>` is a new folder of your choice, not the root of this release, and script docstrings call this layout the repository layout. To rerun a script, restore that layout and run the command below from `paper/`, reading `code/` as the script's working-tree folder:

- `code/` -> `paper/eval/`, `paper/eval/adapters/`, `paper/analysis/`, `paper/baselines/`, `paper/figures/` (Code layout); modules imported as `paper.eval.*`, `eval.*` and `prototype.*` (eval_independent, statistics, _kg_swap_metaqa, retrievers) are in `aug2026_original/`, placed at `<repo>/`.
- `results/` -> `paper/results/`, except `results/*_summary.*`, `results/e2e_dose_across_cells*`, `results/index_precision_*`, `results/published_pairs_*` and `results/rater_*_summary.*`, which live in `paper/analysis/`, `results/tables/`, which is `paper/tables/`, and `results/figures/`, which is `paper/figures/`.
- `external/` -> `paper/external/` (same paths); `manuscript/` -> `paper/manuscript/`.
- `human_validation/` -> `paper/human_validation/` (`judge_spotcheck_v1/` is `judge_spotcheck/`; `llm_spotcheck_v3/package/` is `judge_spotcheck_v3/`; the other folders keep their names).
- `hipporag2_swap/` -> `paper/results/hipporag2_swap/`; its vectors are `paper/external/<bench>/<split>/_kgswap_hipporag2_*.json`.
- `graphragbench_swap/` -> `paper/external/graphragbench/` (<split>/kg_silver_input.jsonl = <split>/kg_silver_input/relations_as_passages.jsonl; corpus_chunks_sha256.jsonl lists corpus/passages.jsonl by hash; DESIGN.md = eval/_graphragbench_swap_design.md).
- `realistic_structure/`, `concentration/`, `zero_edge/`, `dense_bge/` -> `paper/external/<bench>/<split>/` (kg_wikidata/, kg_fullkb/, _concentration/, _zero_edge/, _dense_bge_m3/, _dense_bge_m3_cls/).
- The August 2026 data map as `swap/data/<split>/` -> `paper/external/2wiki/<split>/`, `metaqa/data/<split>/` -> `paper/external/metaqa/<split>/`, `stark/data/<split>/` -> `paper/external/stark/<split>/`, and `swap/results/`, `metaqa/results/`, `stark/results/` -> `paper/results/`.

## Code layout

- `code/_bge_pooling_check.py` <- `eval/_bge_pooling_check.py`
- `code/_e2e_claude_reader.py` <- `eval/_e2e_claude_reader.py`
- `code/_e2e_qa_hipporag2.py` <- `eval/_e2e_qa_hipporag2.py`
- `code/_e2e_third_reader.py` <- `eval/_e2e_third_reader.py`
- `code/_e2e_twowiki_comp200.py` <- `eval/_e2e_twowiki_comp200.py`
- `code/_e2e_variant_pairs.py` <- `eval/_e2e_variant_pairs.py`
- `code/_grade_stark_provenance_trace.py` <- `eval/_grade_stark_provenance_trace.py`
- `code/_grb_provenance_trace.py` <- `eval/_grb_provenance_trace.py`
- `code/_kg_concentration_2wiki.py` <- `eval/_kg_concentration_2wiki.py`
- `code/_kg_fullkb_metaqa.py` <- `eval/_kg_fullkb_metaqa.py`
- `code/_kg_swap_2wiki_comp200.py` <- `eval/_kg_swap_2wiki_comp200.py`
- `code/_kg_swap_dense_bge.py` <- `eval/_kg_swap_dense_bge.py`
- `code/_kg_swap_graphragbench.py` <- `eval/_kg_swap_graphragbench.py`
- `code/_kg_swap_hipporag2.py` <- `eval/_kg_swap_hipporag2.py`
- `code/_kg_swap_hipporag2_comp200.py` <- `eval/_kg_swap_hipporag2_comp200.py`
- `code/_kg_swap_hipporag2_comp200_score.py` <- `eval/_kg_swap_hipporag2_comp200_score.py`
- `code/_kg_swap_hipporag2_score.py` <- `eval/_kg_swap_hipporag2_score.py`
- `code/_kg_swap_variant.py` <- `eval/_kg_swap_variant.py`
- `code/_kg_zero_edge_2wiki.py` <- `eval/_kg_zero_edge_2wiki.py`
- `code/_openie_extractor_exposure.py` <- `eval/_openie_extractor_exposure.py`
- `code/_tab_appendix_v3.py` <- `eval/_tab_appendix_v3.py`
- `code/_tab_e2e_v3.py` <- `eval/_tab_e2e_v3.py`
- `code/_tab_memorization_v3.py` <- `eval/_tab_memorization_v3.py`
- `code/build_graphragbench_swap_data.py` <- `eval/build_graphragbench_swap_data.py`
- `code/build_openie_kg.py` <- `baselines/build_openie_kg.py`
- `code/build_wikidata_kg_2wiki.py` <- `eval/build_wikidata_kg_2wiki.py`
- `code/e2e_dose_across_cells.py` <- `analysis/e2e_dose_across_cells.py`
- `code/e2e_memorization_probe.py` <- `eval/e2e_memorization_probe.py`
- `code/e2e_qa_figures.py` <- `eval/e2e_qa_figures.py`
- `code/e2e_qa_judge.py` <- `eval/e2e_qa_judge.py`
- `code/e2e_qa_swap.py` <- `eval/e2e_qa_swap.py`
- `code/e2e_tex_tables.py` <- `eval/e2e_tex_tables.py`
- `code/generate_fig3_census.py` <- `figures/generate_fig3_census.py`
- `code/headline_margins.py` <- `analysis/headline_margins.py`
- `code/index_precision.py` <- `analysis/index_precision.py`
- `code/kg_passage_link_counts.py` <- `analysis/kg_passage_link_counts.py`
- `code/larger_readers_table.py` <- `analysis/larger_readers_table.py`
- `code/llm_census_logged.py` <- `eval/llm_census_logged.py`
- `code/llm_spotcheck_claude.py` <- `eval/llm_spotcheck_claude.py`
- `code/make_comp200.py` <- `eval/adapters/make_comp200.py`
- `code/make_fig1_provenance.py` <- `figures/make_fig1_provenance.py`
- `code/make_fig_rubric_tree.py` <- `figures/make_fig_rubric_tree.py`
- `code/published_pairs_class_stats.py` <- `analysis/published_pairs_class_stats.py`
- `code/rater_robustness.py` <- `analysis/rater_robustness.py`
- `code/rater_validation.py` <- `analysis/rater_validation.py`
- `code/twowiki_comp200_summary.py` <- `analysis/twowiki_comp200_summary.py`

Shipped copies differ from the working-tree files only as follows. Absolute paths are made relative to `paper/`. `code/index_precision.py` defaults its cache to `analysis/_index_precision_cache`, cites its declaration as `analysis/index_precision_PREREG.md`, and names the secondary predictor "C_KG with GraphRAG-Bench at bin 2 (0.5)" `C_KG_grb_bin2`, as do `results/index_precision_summary.*` (values unchanged). `code/_grade_stark_provenance_trace.py` defaults its input folder to `external/grade_stark_trace/`. `code/llm_census_logged.py` reads `human_validation/items_partB.json`, and its `--score` reports agreement with the census only; the build re-runs it and checks that every number of `result.json` equals the result of the full scorer in the working tree. `human_validation/v3/build_package_v3.py` reads the released item files, and its docstring describes only the released package and the materials of the logged LM raters; it writes the reworded `_comment` of `answer_key_v3.json`. The `_comment` fields of `human_validation/reference_bins.json`, `items_29.json`, `v3/answer_key_v3.json` and `llm_spotcheck_v3/package/sealed_key_v3.json` are rewritten (contents otherwise unchanged). The sealed build records `human_validation/v4/SEALED_build.json` and `human_validation/judge_spotcheck_v4/SEALED_judge_build.json` list the SHA-256 of their inputs; every input ships with that hash except `human_validation/items_29.json`, where its `_comment` is rewritten (items unchanged), recorded as b4ab88534d1805b1d67f4b389d9fd9a64ec408e4cee59151713ce2be13a9371b; `human_validation/llm_spotcheck_v3/package/sealed_key_v3.json`, where its `_comment` is rewritten (keys unchanged), recorded as fa35e5f57732c627f7b5c2ce20c4096fdbda46ab78387f5e59ee287f7239bf55; `human_validation/llm_spotcheck_v3/result_v3.json`, where one key is renamed (numbers unchanged), recorded as 92508ebb4fb1c1e13ed8713eb4ce268982407a5eb683bba3e422c879e0aa35ad; `human_validation/reference_bins.json`, where its `_comment` is rewritten (bins unchanged), recorded as 22a0efbe7a4d65fa7e6424ccf8fa3303698dfc2098201c07ad62470b940afb61; `human_validation/v3/answer_key_v3.json`, where its `_comment` is rewritten (keys unchanged), recorded as 1b56171733f84f7f664a84cb8406aeae454643e317d8d94a8c610e722c09665e. `human_validation/llm_census_logged/PROTOCOL.md`, `human_validation/v4/PROTOCOL_v4.md`, `human_validation/judge_spotcheck_v4/PROTOCOL_judge_v4.md` and `results/twowiki_comp200/PROTOCOL.md` are released copies: the first line gives the SHA-256 of the protocol as written, each passage outside the scope of this paper is replaced by `[omitted]`, and the rest is verbatim. In `human_validation/llm_spotcheck_v3/`, the scorer and its outputs call the earlier human rater `v1 human rater` (numbers unchanged), `result_v3.md` has a title line naming the raters as LMs, and the docstrings of `package/score_spotcheck_v3.py` and `package/build_spotcheck_v3.py` describe the released package (the latter names its input file `per_question_20260924.jsonl` and writes the reworded `_comment` of `sealed_key_v3.json`), and `package/AUDIT_v1.md` is its item-by-item verification table with the scope path of the release. `pool_stats.json` is shipped as hashed in the sealed build record. Docstrings and comments of `code/_grb_provenance_trace.py` no longer name files that are not released. In `results/openie_extractor_exposure/summary.*` the quoted 2WikiMultiHop and GraphRAG-Bench example passages read `[passage text not included in the release]` (MetaQA quotes kept; `code/_openie_extractor_exposure.py` writes them in full from the passage files). In `human_validation/judge_spotcheck_v4/`, the docstring of `score_judge_v4.py` and one sentence it writes into `result_judge_v4.md` no longer name the plan of an earlier spot-check design, which is not released, and the scorer and its report call the v1 rater the first human rater (numbers unchanged). `results/census_kappa_ci_output.txt` holds the block of the console output behind Table 9. The blank workbooks of `human_validation/v3/sheets/` and `human_validation/llm_spotcheck_v3/package/sheets/` carry the English Office defaults in their document-property, style and theme parts (cell values unchanged). `external/2wiki/comp200/manifest.json` has its passage text removed (PASSAGE_TEXT_REMOVED.md there), and `hipporag2_swap/2wiki/<split>/_kgswap_hipporag2_kg_openie_edgeprov.json` is scored at release from the run outputs with the scorer's `vectors` function (checked against `hipporag2_swap/summary.json`). The sub-READMEs give paths and commands as they apply in this release. `code/headline_margins.py` and `results/headline_margins_summary.json` name the manuscript files as released here (`manuscript/eacl2027_camera_ready.tex` and `.aux`), and the script accepts the table path that source uses.

## Reproduction

Offline from this release (no model call, no data beyond this release, working-tree layout restored): `code/headline_margins.py`, `code/e2e_dose_across_cells.py` (both variants), `code/kg_passage_link_counts.py`, `code/rater_robustness.py`, `code/rater_validation.py`, `code/llm_census_logged.py --score`, the two spot-check scorers, and `human_validation/v3/build_package_v3.py`. Upstream raw data not in this release are needed by `code/index_precision.py` (the 2WikiMultiHop and GraphRAG-Bench passage texts), `code/_openie_extractor_exposure.py` (passage texts and the 2WikiMultiHop raw dev set), `code/_grb_provenance_trace.py` (the GraphRAG-Bench corpus, see `graphragbench_swap/CORPUS_TEXT_REMOVED.md`, the files in `results/grb_provenance/inputs/PINNED_INPUTS.md`, and the 2WikiMultiHop raw dev set and passages) and `code/_grade_stark_provenance_trace.py` (inputs listed in its docstring). The scorers and exports of the human studies (`human_validation/v4/score_v4.py`, `independence_checks_v4.py`, `export_public_v4.py`, `human_validation/judge_spotcheck_v4/score_judge_v4.py`, `export_public_judge_v4.py`) need the returned workbooks or the result JSON, which are not released; the analyses run from their `public/` exports. Every script that runs a retriever, an encoder, a reader, a judge, or an LM API also needs the models and the passage texts. The released outputs were written on Windows in text mode (CRLF line ends); the byte-for-byte statements below hold there, and elsewhere up to line ends.

```
python code/build_graphragbench_swap_data.py                 # GraphRAG-Bench splits + silver input (raw files: graphragbench_swap/CORPUS_TEXT_REMOVED.md)
python code/build_openie_kg.py --passages ... --out-dir ...  # kg_silver (from relations) and kg_openie (from corpus)
python code/_kg_swap_graphragbench.py                        # retrieval swap
python code/_kg_swap_dense_bge.py                            # BGE-M3 dense channel, [CLS] pooling (rag-bge-m3 env); --pooling mean for the mean-pooled variant
python code/_bge_pooling_check.py --benchmarks 2wiki --splits sub200   # dense-only hit@5 per pooling and input format
python code/_kg_fullkb_metaqa.py && python code/_kg_concentration_2wiki.py && python code/_kg_zero_edge_2wiki.py
python code/build_wikidata_kg_2wiki.py && python code/_kg_swap_variant.py --bench 2wiki --variant kg_wikidata
python code/_kg_swap_hipporag2.py --split sub200 --kg kg_silver   # HippoRAG 2 (rag-graphrag-py311 env); each bench/split/KG
python code/_kg_swap_hipporag2_score.py                      # HippoRAG 2 vectors + summary
python code/e2e_qa_swap.py --model llama3.1:8b               # end-to-end QA (ollama); repeat for qwen2.5:14b
python code/e2e_qa_swap.py --model llama3.1:8b --bge-variant cls --out results/e2e_qa_cls   # [CLS] cells; repeat for qwen2.5:14b
python code/_e2e_third_reader.py                             # mistral-small:24b, core 2WikiMultiHop conditions
python code/_e2e_claude_reader.py --submit && python code/_e2e_claude_reader.py --collect   # claude-sonnet-5-5, Batches API (key needed)
python code/_e2e_claude_reader.py --score && python code/larger_readers_table.py   # offline: truncation rule, scoring, Table 27
python code/_e2e_qa_hipporag2.py --model qwen2.5:14b --benchmarks 2wiki --out results/hipporag2_swap/e2e_qa
python code/e2e_qa_judge.py --reader llama3.1:8b --judge mistral-small:24b --benchmarks graphragbench   # add --res results/e2e_qa_cls for the [CLS] run
python code/e2e_tex_tables.py && python code/e2e_qa_figures.py
python code/e2e_dose_across_cells.py && python code/e2e_dose_across_cells.py --res results/e2e_qa_cls --suffix _cls   # offline
python code/e2e_memorization_probe.py --model qwen2.5:14b    # repeat for llama3.1:8b
python code/kg_passage_link_counts.py                        # offline; --check compares with results/kg_passage_link_counts.json
python code/_tab_e2e_v3.py && python code/_tab_memorization_v3.py && python code/_tab_appendix_v3.py
python code/_grb_provenance_trace.py                         # no model calls; needs upstream raw data (above)
python code/index_precision.py                               # needs upstream raw data (above)
SOURCE_DATE_EPOCH=<the 'generated' time of results/headline_margins_summary.json, in epoch seconds> python code/headline_margins.py
    # offline; reproduces results/headline_margins_summary.* byte for byte (without SOURCE_DATE_EPOCH only the time stamp differs)
python code/published_pairs_class_stats.py                   # no model calls
# the compositional 2WikiMultiHop split (from the folder that holds paper/; needs the 2WikiMultiHop raw dev set; make_comp200 also imports
#   paper/eval/adapters/twowiki_adapter.py and make_holdout200.py, which are not in this release, so its released output in external/2wiki/comp200/ is the starting point):
python -m paper.eval.adapters.make_comp200 --dev paper/external/2wiki/raw/data_ids/dev.json --out-dir paper/external/2wiki/comp200 --exclude-qa paper/external/2wiki/sub200/qa.jsonl paper/external/2wiki/holdout200/qa.jsonl --id-aliases paper/external/2wiki/raw/data_ids/id_aliases.json
python paper/baselines/build_openie_kg.py --passages paper/external/2wiki/comp200/passages.jsonl --out-dir paper/external/2wiki/comp200/kg_openie --model llama3.1:8b --synonyms surface --verbose
python paper/eval/_kg_swap_2wiki_comp200.py                  # hybrid, PPR, hybrid-PPR and BM25 vectors
python paper/eval/_kg_swap_hipporag2_comp200.py --split comp200 --kg kg_silver   # rag-graphrag-py311 env; then --kg kg_openie
python paper/eval/_kg_swap_hipporag2_comp200_score.py
python paper/eval/_e2e_twowiki_comp200.py --model llama3.1:8b   # repeat for qwen2.5:14b
python paper/analysis/twowiki_comp200_summary.py             # no model calls
python code/make_fig1_provenance.py && python code/make_fig_rubric_tree.py && python code/generate_fig3_census.py   # Figure 1, Figure 3, Figure 4
python code/_grade_stark_provenance_trace.py                 # inputs listed in its docstring (GS_TRACE_DIR); no model calls
python code/_openie_extractor_exposure.py                    # no model calls; needs upstream raw data (above)
# the rater analyses run directly in this folder after this one-line setup:
mkdir -p analysis eval && cp code/rater_robustness.py code/rater_validation.py analysis/ && cp code/llm_census_logged.py eval/
PYTHONIOENCODING=utf-8 python eval/llm_census_logged.py --score   # from raw.jsonl (--run re-queries the APIs); the variable is needed on Windows
python analysis/rater_robustness.py && python analysis/rater_validation.py   # write analysis/rater_*_summary.*
cd human_validation/judge_spotcheck_v1 && python score_spotcheck.py && cd ../..   # the human spot-check; reads its files by relative name
cd human_validation/llm_spotcheck_v3 && SPOT_RETURNED=returned SPOT_OUT=<empty folder> python package/score_spotcheck_v3.py && cd ../..
    # the LM spot-check, re-scored offline from the filled sheets (result_v3.md differs only in its title line)
python code/llm_spotcheck_claude.py                          # re-queries the three models (API key needed)
```

Models: readers llama3.1:8b, qwen2.5:14b, mistral-small:24b (third reader), claude-sonnet-5-5 (proprietary reader, Message Batches API, extended thinking off); judges mistral-small:24b (primary), cross-reader (secondary); OpenIE extractor llama3.1:8b; dense encoder BAAI/bge-m3. The open-weight models run locally (Ollama, transformers); claude-sonnet-5-5 and the LM raters below are called through their providers' APIs. LM census raters (logged API runs): claude-opus-5-5, claude-sonnet-5-5, claude-haiku-4-5, gemini-3.8-flash; LM labels of the judge sample: claude-opus-5, claude-sonnet-5, claude-haiku-4-5. Seeds and bootstrap settings are fixed in the scripts.

## Checksums (first 16 hex of SHA-256)

- `code/_bge_pooling_check.py` 1156776caeb7d6c7
- `code/_e2e_claude_reader.py` a57b9beb94ad94b9
- `code/_e2e_qa_hipporag2.py` 7ca366624e5e632d
- `code/_e2e_third_reader.py` 14c261497cf6bfa9
- `code/_e2e_twowiki_comp200.py` 2c15ade04338c803
- `code/_e2e_variant_pairs.py` 730faf5cd9bf0d88
- `code/_grade_stark_provenance_trace.py` 0f2245eaac25411a
- `code/_grb_provenance_trace.py` 247aa32c317ebb6b
- `code/_kg_concentration_2wiki.py` a9aba371aa8a17b0
- `code/_kg_fullkb_metaqa.py` 064cf4caa910f86d
- `code/_kg_swap_2wiki_comp200.py` a4f702b8125d6f8a
- `code/_kg_swap_dense_bge.py` c532753d833cf32e
- `code/_kg_swap_graphragbench.py` 13dda243b7f2a427
- `code/_kg_swap_hipporag2.py` 554f0a776b65c13f
- `code/_kg_swap_hipporag2_comp200.py` f6378ada0a6b3a2a
- `code/_kg_swap_hipporag2_comp200_score.py` 7b50142d64587e41
- `code/_kg_swap_hipporag2_score.py` c013d704b0ee64c0
- `code/_kg_swap_variant.py` 22f42cc98cd14278
- `code/_kg_zero_edge_2wiki.py` ca6aae66678a7ea3
- `code/_openie_extractor_exposure.py` b805ec014346a8f7
- `code/_tab_appendix_v3.py` 7d441ca4527c32fc
- `code/_tab_e2e_v3.py` c978cdb8a1af9dbd
- `code/_tab_memorization_v3.py` e9c5c7fac4172ffd
- `code/build_graphragbench_swap_data.py` 8375fd16c7be9950
- `code/build_openie_kg.py` bea5ac0ff059a6d9
- `code/build_wikidata_kg_2wiki.py` aa3e682e67e0e5a5
- `code/e2e_dose_across_cells.py` d8c99523b03a1cb8
- `code/e2e_memorization_probe.py` ebd98eeb0aa25959
- `code/e2e_qa_figures.py` f9ee7e27f0157add
- `code/e2e_qa_judge.py` a686aef60f27fe35
- `code/e2e_qa_swap.py` d239677429e15ca1
- `code/e2e_tex_tables.py` 80d357097d1f4118
- `code/generate_fig3_census.py` ee0ddeb711060846
- `code/headline_margins.py` a4806c2fb606d03b
- `code/index_precision.py` 350215658e9503f3
- `code/kg_passage_link_counts.py` 27e397d592a5873e
- `code/larger_readers_table.py` 542b40db96778767
- `code/llm_census_logged.py` 9e1955c02739a0c3
- `code/llm_spotcheck_claude.py` bdc1a4e2539e174a
- `code/make_comp200.py` 0f8694113a1f2b98
- `code/make_fig1_provenance.py` 4aa626756a3a4613
- `code/make_fig_rubric_tree.py` 3eb12e49c0720048
- `code/published_pairs_class_stats.py` cab7e1c2eab19251
- `code/rater_robustness.py` 825eb50735c77805
- `code/rater_validation.py` d6c614288d20935b
- `code/twowiki_comp200_summary.py` b7e7ba8b32599b74
- `concentration/2wiki/holdout200/_kgswap_ppr_openie_goldonly.json` 136344a798c41843
- `concentration/2wiki/holdout200/_kgswap_ppr_silver_dilute_g31.json` cca1df09c8305673
- `concentration/2wiki/holdout200/_kgswap_ppr_silver_dilute_g50.json` 56f517cee9b2ca13
- `concentration/2wiki/holdout200/_kgswap_ppr_silver_dilute_g75.json` 37f2bef41708cc55
- `concentration/2wiki/holdout200/_kgswap_proposed_v3_openie_goldonly.json` daae23b5850414b2
- `concentration/2wiki/holdout200/_kgswap_proposed_v3_silver_dilute_g31.json` 0ef467a1d0aafc7c
- `concentration/2wiki/holdout200/_kgswap_proposed_v3_silver_dilute_g50.json` 131032de86a289f9
- `concentration/2wiki/holdout200/_kgswap_proposed_v3_silver_dilute_g75.json` c2766703489dc11a
- `concentration/2wiki/holdout200/_kgswap_proposed_v5_graphcentric_openie_goldonly.json` 9ad50f1239ae465a
- `concentration/2wiki/holdout200/_kgswap_proposed_v5_graphcentric_silver_dilute_g31.json` 98f879f3b01a1ef1
- `concentration/2wiki/holdout200/_kgswap_proposed_v5_graphcentric_silver_dilute_g50.json` d9fb522aabb6e532
- `concentration/2wiki/holdout200/_kgswap_proposed_v5_graphcentric_silver_dilute_g75.json` 49cb270f2370770b
- `concentration/2wiki/holdout200/openie_goldonly/proposed_kg_edges.json` 429d0fba732d2956
- `concentration/2wiki/holdout200/openie_goldonly/proposed_kg_nodes.json` ea8858fa22e9a5af
- `concentration/2wiki/holdout200/openie_goldonly/variant_manifest.json` 4c149a9a53d59cd7
- `concentration/2wiki/holdout200/silver_dilute_g31/proposed_kg_edges.json` 6afc7402acf5b534
- `concentration/2wiki/holdout200/silver_dilute_g31/proposed_kg_nodes.json` b8681e714780a51f
- `concentration/2wiki/holdout200/silver_dilute_g31/variant_manifest.json` bf90bcd533ecb8ce
- `concentration/2wiki/holdout200/silver_dilute_g50/proposed_kg_edges.json` 6afc7402acf5b534
- `concentration/2wiki/holdout200/silver_dilute_g50/proposed_kg_nodes.json` bd8c6c8ae49da72b
- `concentration/2wiki/holdout200/silver_dilute_g50/variant_manifest.json` d847491d47f450e7
- `concentration/2wiki/holdout200/silver_dilute_g75/proposed_kg_edges.json` 6afc7402acf5b534
- `concentration/2wiki/holdout200/silver_dilute_g75/proposed_kg_nodes.json` a5443b4f655ee105
- `concentration/2wiki/holdout200/silver_dilute_g75/variant_manifest.json` 2029f2ba06a8a0f7
- `concentration/2wiki/sub200/_kgswap_ppr_openie_goldonly.json` 0326f0d9167e6e1c
- `concentration/2wiki/sub200/_kgswap_ppr_silver_dilute_g31.json` 881a732b78aab5fc
- `concentration/2wiki/sub200/_kgswap_ppr_silver_dilute_g50.json` fab8272e7bdd491d
- `concentration/2wiki/sub200/_kgswap_ppr_silver_dilute_g75.json` 39fcb7fd128c1bdc
- `concentration/2wiki/sub200/_kgswap_proposed_v3_openie_goldonly.json` ed75dbf620ce27c4
- `concentration/2wiki/sub200/_kgswap_proposed_v3_silver_dilute_g31.json` c9d4605ca001af78
- `concentration/2wiki/sub200/_kgswap_proposed_v3_silver_dilute_g50.json` 2168c2538a6eb6d1
- `concentration/2wiki/sub200/_kgswap_proposed_v3_silver_dilute_g75.json` 2ea1ffabadea14c8
- `concentration/2wiki/sub200/_kgswap_proposed_v5_graphcentric_openie_goldonly.json` 8da3e4f6b45cf526
- `concentration/2wiki/sub200/_kgswap_proposed_v5_graphcentric_silver_dilute_g31.json` a96162f7701975da
- `concentration/2wiki/sub200/_kgswap_proposed_v5_graphcentric_silver_dilute_g50.json` 01b38808c44e0af0
- `concentration/2wiki/sub200/_kgswap_proposed_v5_graphcentric_silver_dilute_g75.json` 6ffb4d28e40af12c
- `concentration/2wiki/sub200/openie_goldonly/proposed_kg_edges.json` 577f6ddb2fbb89b1
- `concentration/2wiki/sub200/openie_goldonly/proposed_kg_nodes.json` 348b36ce13af438d
- `concentration/2wiki/sub200/openie_goldonly/variant_manifest.json` 392749a98a45f9b3
- `concentration/2wiki/sub200/silver_dilute_g31/proposed_kg_edges.json` 50a3c04b97cb6ae8
- `concentration/2wiki/sub200/silver_dilute_g31/proposed_kg_nodes.json` 9d653241b0432075
- `concentration/2wiki/sub200/silver_dilute_g31/variant_manifest.json` bf90bcd533ecb8ce
- `concentration/2wiki/sub200/silver_dilute_g50/proposed_kg_edges.json` 50a3c04b97cb6ae8
- `concentration/2wiki/sub200/silver_dilute_g50/proposed_kg_nodes.json` fdc42d002e129ee3
- `concentration/2wiki/sub200/silver_dilute_g50/variant_manifest.json` d847491d47f450e7
- `concentration/2wiki/sub200/silver_dilute_g75/proposed_kg_edges.json` 50a3c04b97cb6ae8
- `concentration/2wiki/sub200/silver_dilute_g75/proposed_kg_nodes.json` 42096ae8043f8a1d
- `concentration/2wiki/sub200/silver_dilute_g75/variant_manifest.json` 2029f2ba06a8a0f7
- `dense_bge/2wiki/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` 99f1d83aeb280abc
- `dense_bge/2wiki/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` 37b96f5a722d1d02
- `dense_bge/2wiki/holdout200/_kgswap_all_kg_openie.json` 245d13fa0fd4d596
- `dense_bge/2wiki/holdout200/_kgswap_all_kg_silver.json` 47d2bbf79d0a00d8
- `dense_bge/2wiki/sub200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` 8efa1338687ac32a
- `dense_bge/2wiki/sub200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` 147096d800a96f9e
- `dense_bge/2wiki/sub200/_kgswap_all_kg_openie.json` 5772be0973b15060
- `dense_bge/2wiki/sub200/_kgswap_all_kg_silver.json` 6767592c75ef0e8b
- `dense_bge/graphragbench/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` adf6d3ff4f5f9fa8
- `dense_bge/graphragbench/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` b878e4e20e1ea37e
- `dense_bge/graphragbench/sub200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` e62f34be0d401990
- `dense_bge/graphragbench/sub200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` 0088c07032ba68f5
- `dense_bge/metaqa/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` 28a1b786dc4a8533
- `dense_bge/metaqa/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` 2c2cab6e5ca92acb
- `dense_bge/metaqa/holdout200/_kgswap_all_kg_openie.json` e0fb6e38d80a5024
- `dense_bge/metaqa/holdout200/_kgswap_all_kg_silver.json` 48fab80954350a66
- `dense_bge/metaqa/sub200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` 760fd83d7d1a2a34
- `dense_bge/metaqa/sub200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` 36242d4b722cde4a
- `dense_bge/metaqa/sub200/_kgswap_all_kg_openie.json` 2b5b4ecd7b8f1420
- `dense_bge/metaqa/sub200/_kgswap_all_kg_silver.json` 1734e3c1b5200bf6
- `dense_bge/stark/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` 53bb6783eb309972
- `dense_bge/stark/holdout200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` 2c113b6135402952
- `dense_bge/stark/holdout200/_kgswap_all_kg_openie.json` 0c452965b58fcd93
- `dense_bge/stark/holdout200/_kgswap_all_kg_silver.json` 3d90afac690122b0
- `dense_bge/stark/sub200/_dense_bge_m3_cls/_kgswap_all_kg_openie_cls.json` 7c85e94faea91afa
- `dense_bge/stark/sub200/_dense_bge_m3_cls/_kgswap_all_kg_silver_cls.json` 52a0f0d56925536b
- `dense_bge/stark/sub200/_kgswap_all_kg_openie.json` 1842d1338b19a958
- `dense_bge/stark/sub200/_kgswap_all_kg_silver.json` 02fb808e40a55d41
- `external/2wiki/comp200/_kgswap_hipporag2_kg_openie.json` c40dbc12a2910a8f
- `external/2wiki/comp200/_kgswap_hipporag2_kg_silver.json` 4725f8734ce262d8
- `external/2wiki/comp200/_kgswap_kg_openie.json` 34406333440788d1
- `external/2wiki/comp200/_kgswap_kg_silver.json` 1274cd38c8a7c6d0
- `external/2wiki/comp200/_kgswap_ppr_kg_openie.json` c1d2ff3e53630736
- `external/2wiki/comp200/_kgswap_ppr_kg_silver.json` 3314ba21b27d7876
- `external/2wiki/comp200/_kgswap_proposed_v5_graphcentric_kg_openie.json` aff1e49a0bece54b
- `external/2wiki/comp200/_kgswap_proposed_v5_graphcentric_kg_silver.json` 284307d61300a74c
- `external/2wiki/comp200/comp200_manifest.json` b8e3d8e7a88603d6
- `external/2wiki/comp200/kg_openie/build_summary.json` c088e3a4ce2316d4
- `external/2wiki/comp200/kg_openie/proposed_kg_edges.json` b2b85e422e9427ee
- `external/2wiki/comp200/kg_openie/proposed_kg_nodes.json` 3e1ada590b96f049
- `external/2wiki/comp200/kg_silver/proposed_kg_edges.json` de8b0805edf3cd41
- `external/2wiki/comp200/kg_silver/proposed_kg_nodes.json` 59b69709e51c50a0
- `external/2wiki/comp200/manifest.json` 49e37e2c71a18762
- `external/2wiki/comp200/PASSAGE_TEXT_REMOVED.md` c35d1095f3a07350
- `external/2wiki/comp200/qa.jsonl` 185cd21496bce73b
- `external/metaqa/holdout200/_kgswap_kg_openie.json` 59a947564ec619bf
- `external/metaqa/holdout200/_kgswap_kg_silver.json` 9a37e0ddce49e741
- `external/metaqa/holdout200/_kgswap_ppr_kg_openie.json` 343bce272ffacc88
- `external/metaqa/holdout200/_kgswap_ppr_kg_silver.json` 69675818e8f46d2c
- `external/metaqa/holdout200/_kgswap_proposed_v5_graphcentric_kg_openie.json` 74c63ff59d8de70b
- `external/metaqa/holdout200/_kgswap_proposed_v5_graphcentric_kg_silver.json` f6e3ad215fcb5c17
- `external/metaqa/sub200/_kgswap_kg_openie.json` 60be19cdba6f8794
- `external/metaqa/sub200/_kgswap_kg_silver.json` 16b05485566e3e61
- `external/metaqa/sub200/_kgswap_ppr_kg_openie.json` 3a45163ae92f5733
- `external/metaqa/sub200/_kgswap_ppr_kg_silver.json` ced82f894d7d30c5
- `external/metaqa/sub200/_kgswap_proposed_v5_graphcentric_kg_openie.json` 42afc30fb049cc0d
- `external/metaqa/sub200/_kgswap_proposed_v5_graphcentric_kg_silver.json` c942d2350d6fd20f
- `graphragbench_swap/build_report.json` e6635b5b861ade4a
- `graphragbench_swap/corpus_chunks_sha256.jsonl` eca48f0c9634ae44
- `graphragbench_swap/CORPUS_TEXT_REMOVED.md` 1af23c53b3228a36
- `graphragbench_swap/DESIGN.md` 571cbe32a28f4bc4
- `graphragbench_swap/holdout200/_dense_bge_m3/_kgswap_all_kg_openie.json` 984ecf97274d5aeb
- `graphragbench_swap/holdout200/_dense_bge_m3/_kgswap_all_kg_silver.json` 7d66ad716f30c2eb
- `graphragbench_swap/holdout200/_kgswap_hipporag2_kg_openie.json` e6be89533c3ec3a0
- `graphragbench_swap/holdout200/_kgswap_hipporag2_kg_silver.json` 9f27a07dbbeb03d9
- `graphragbench_swap/holdout200/_kgswap_kg_openie.json` 8aab3bc26e8a864d
- `graphragbench_swap/holdout200/_kgswap_kg_silver.json` db26101b45a972c1
- `graphragbench_swap/holdout200/_kgswap_ppr_kg_openie.json` 9d498ebf5d5d219d
- `graphragbench_swap/holdout200/_kgswap_ppr_kg_silver.json` 5948560109daf8f2
- `graphragbench_swap/holdout200/_kgswap_proposed_v5_graphcentric_kg_openie.json` df3a703d662f4b76
- `graphragbench_swap/holdout200/_kgswap_proposed_v5_graphcentric_kg_silver.json` 154644e7db411c0e
- `graphragbench_swap/holdout200/kg_openie/build_summary.json` a69f599d27820698
- `graphragbench_swap/holdout200/kg_openie/proposed_kg_edges.json` 052dd7970f2e3e77
- `graphragbench_swap/holdout200/kg_openie/proposed_kg_nodes.json` 7cd25fc9e94ef568
- `graphragbench_swap/holdout200/kg_silver/build_summary.json` 48fe83acea0efe28
- `graphragbench_swap/holdout200/kg_silver/proposed_kg_edges.json` f721e1931a50c969
- `graphragbench_swap/holdout200/kg_silver/proposed_kg_nodes.json` 2b853375f578947a
- `graphragbench_swap/holdout200/kg_silver_input.jsonl` beed3ac5c88f645d
- `graphragbench_swap/holdout200/qa.jsonl` 6cc9ad89c66009ba
- `graphragbench_swap/LICENSE_GraphRAG-Bench.txt` f607254fa7fd8fd9
- `graphragbench_swap/sub200/_dense_bge_m3/_kgswap_all_kg_openie.json` b5cfb09658235c4a
- `graphragbench_swap/sub200/_dense_bge_m3/_kgswap_all_kg_silver.json` d2d0fc9e60240d9a
- `graphragbench_swap/sub200/_kgswap_hipporag2_kg_openie.json` 94cfa991e8699740
- `graphragbench_swap/sub200/_kgswap_hipporag2_kg_silver.json` 134a3e7e8175b394
- `graphragbench_swap/sub200/_kgswap_kg_openie.json` d44c471401ed8b9d
- `graphragbench_swap/sub200/_kgswap_kg_silver.json` 6ecb003de2b1a17e
- `graphragbench_swap/sub200/_kgswap_ppr_kg_openie.json` 7c9ccd8f45e59acb
- `graphragbench_swap/sub200/_kgswap_ppr_kg_silver.json` 4966319c21b930b7
- `graphragbench_swap/sub200/_kgswap_proposed_v5_graphcentric_kg_openie.json` f2e75de3160d09e1
- `graphragbench_swap/sub200/_kgswap_proposed_v5_graphcentric_kg_silver.json` 55af68342d3b4d0e
- `graphragbench_swap/sub200/kg_openie/build_summary.json` a69f599d27820698
- `graphragbench_swap/sub200/kg_openie/proposed_kg_edges.json` 052dd7970f2e3e77
- `graphragbench_swap/sub200/kg_openie/proposed_kg_nodes.json` 7cd25fc9e94ef568
- `graphragbench_swap/sub200/kg_silver/build_summary.json` 615517c759e9c26a
- `graphragbench_swap/sub200/kg_silver/proposed_kg_edges.json` cfe65ab09ad42f30
- `graphragbench_swap/sub200/kg_silver/proposed_kg_nodes.json` ee2978da14dad9ca
- `graphragbench_swap/sub200/kg_silver_input.jsonl` 167a1eb75f8cea87
- `graphragbench_swap/sub200/qa.jsonl` 2fa5bd72ddf4ebba
- `hipporag2_swap/2wiki/holdout200/_kgswap_hipporag2_kg_openie.json` 6b75d345124fd122
- `hipporag2_swap/2wiki/holdout200/_kgswap_hipporag2_kg_openie_edgeprov.json` d76d2d8b08b64592
- `hipporag2_swap/2wiki/holdout200/_kgswap_hipporag2_kg_silver.json` 792efd1e51dcc180
- `hipporag2_swap/2wiki/holdout200/_kgswap_hipporag2_kg_wikidata.json` 404c23d72f107e2b
- `hipporag2_swap/2wiki/sub200/_kgswap_hipporag2_kg_openie.json` 5e8c02f65b3ea5ca
- `hipporag2_swap/2wiki/sub200/_kgswap_hipporag2_kg_openie_edgeprov.json` eeedc5e3d729c5de
- `hipporag2_swap/2wiki/sub200/_kgswap_hipporag2_kg_silver.json` 90fcd217b7af1705
- `hipporag2_swap/2wiki/sub200/_kgswap_hipporag2_kg_wikidata.json` 9939245b6248b35b
- `hipporag2_swap/e2e_qa/qwen2_5_14b/generations.jsonl` 96cf4dec261a4f60
- `hipporag2_swap/e2e_qa/qwen2_5_14b/per_question.jsonl` fc3cac46c8f64d0b
- `hipporag2_swap/e2e_qa/qwen2_5_14b/summary.json` c380741a19fce49b
- `hipporag2_swap/e2e_qa/qwen2_5_14b/summary.md` f31d885a87dae825
- `hipporag2_swap/metaqa/holdout200/_kgswap_hipporag2_kg_openie.json` a97f4e0cdeae0b46
- `hipporag2_swap/metaqa/holdout200/_kgswap_hipporag2_kg_silver.json` b281f0a58e396757
- `hipporag2_swap/metaqa/sub200/_kgswap_hipporag2_kg_openie.json` 2488786bdb5151a1
- `hipporag2_swap/metaqa/sub200/_kgswap_hipporag2_kg_silver.json` 5e66fab74ac02e06
- `hipporag2_swap/METHODS.md` 7422158375a26fd1
- `hipporag2_swap/runs/graphragbench__holdout200__kg_openie/retrieval.jsonl` 77fd9218e9abc4b8
- `hipporag2_swap/runs/graphragbench__holdout200__kg_silver/retrieval.jsonl` 1be1235dedac1f50
- `hipporag2_swap/runs/graphragbench__sub200__kg_openie/retrieval.jsonl` 75a849d7b24379bc
- `hipporag2_swap/runs/graphragbench__sub200__kg_silver/retrieval.jsonl` 8205a38fe1bea313
- `hipporag2_swap/runs/holdout200__kg_openie/retrieval.jsonl` 84d39e8e712b07bd
- `hipporag2_swap/runs/holdout200__kg_openie__edgeprov/retrieval.jsonl` b6948fd0164d77a9
- `hipporag2_swap/runs/holdout200__kg_silver/retrieval.jsonl` 5fbca0db264fb0ad
- `hipporag2_swap/runs/holdout200__kg_wikidata/retrieval.jsonl` 6e494374c13f0484
- `hipporag2_swap/runs/metaqa__holdout200__kg_fullkb/retrieval.jsonl` 14f2f738c67dd956
- `hipporag2_swap/runs/metaqa__holdout200__kg_openie/retrieval.jsonl` fd26bdeb1f67416b
- `hipporag2_swap/runs/metaqa__holdout200__kg_silver/retrieval.jsonl` dafa08fc1bd2a616
- `hipporag2_swap/runs/metaqa__sub200__kg_fullkb/retrieval.jsonl` 72e06338879014c8
- `hipporag2_swap/runs/metaqa__sub200__kg_openie/retrieval.jsonl` 1bf19dd86b028504
- `hipporag2_swap/runs/metaqa__sub200__kg_silver/retrieval.jsonl` 6743d7bb25587048
- `hipporag2_swap/runs/sub200__kg_openie/retrieval.jsonl` f2d1797e835e196a
- `hipporag2_swap/runs/sub200__kg_openie__edgeprov/retrieval.jsonl` 69b49f6fd86eea0b
- `hipporag2_swap/runs/sub200__kg_silver/retrieval.jsonl` d6bf23f2430063fa
- `hipporag2_swap/runs/sub200__kg_wikidata/retrieval.jsonl` 4f3295cf477f8368
- `hipporag2_swap/summary.json` 742a4713d1c3be98
- `hipporag2_swap/summary.md` 8f48aa3761941b67
- `human_validation/external_rater_result.json` f05ea64eaf1f31a5
- `human_validation/items_29.json` 735574ec3966b9d0
- `human_validation/items_partB.json` 7255b9ce58183169
- `human_validation/judge_spotcheck_v1/judge_spotcheck_sheet.csv` 217231c816cc0023
- `human_validation/judge_spotcheck_v1/judge_spotcheck_sheet_blank.csv` 53199398179d56ba
- `human_validation/judge_spotcheck_v1/README.md` 9338ed4956ec4cbe
- `human_validation/judge_spotcheck_v1/score_spotcheck.py` f0bd1f901c6946b8
- `human_validation/judge_spotcheck_v1/sealed_judge_verdicts.json` 3a1da7f8a76441c3
- `human_validation/judge_spotcheck_v1/spotcheck_result.json` 28b50fe86f8ed669
- `human_validation/judge_spotcheck_v4/build_package_judge_v4.py` fbac0c1b2e28c493
- `human_validation/judge_spotcheck_v4/CHECKLIST_judge_v4.md` 19e18dbd40f1cedf
- `human_validation/judge_spotcheck_v4/export_public_judge_v4.py` 5f5ef76e45fa3fa5
- `human_validation/judge_spotcheck_v4/GUIDE_v4.md` 630fd82a700867e3
- `human_validation/judge_spotcheck_v4/PROTOCOL_judge_v4.md` aa0e282aa879ecbf
- `human_validation/judge_spotcheck_v4/public/labels_judge_v4.csv` c6e333c2096c87da
- `human_validation/judge_spotcheck_v4/public/raters_judge_v4.csv` 5089f65e1af349fd
- `human_validation/judge_spotcheck_v4/public/README.md` 7a2ee67df6337893
- `human_validation/judge_spotcheck_v4/public/returned_sha256.txt` 543fc9193cd34ff5
- `human_validation/judge_spotcheck_v4/result_judge_v4.md` 2c6a2a0ca6cfeae0
- `human_validation/judge_spotcheck_v4/score_judge_v4.py` c09a1f193eac8356
- `human_validation/judge_spotcheck_v4/SEALED_judge_build.json` 30f03e38f1bfc133
- `human_validation/judge_spotcheck_v4/to_send/rater_1_judge_v4.zip` 4fed3f6d772e32bf
- `human_validation/judge_spotcheck_v4/to_send/rater_2_judge_v4.zip` 79c2806e747fdff2
- `human_validation/judge_spotcheck_v4/to_send/rater_3_judge_v4.zip` 4177a4a714a2ebfe
- `human_validation/lexical_rules.py` 18417b258c3cc33f
- `human_validation/llm_census_logged/excerpts.json` 6913e43578f7ab20
- `human_validation/llm_census_logged/model_params.json` 0597d74fa952bcea
- `human_validation/llm_census_logged/pilot_raw.jsonl` 659925af536beea2
- `human_validation/llm_census_logged/prompts/manifest.json` e01e3ab12e70b293
- `human_validation/llm_census_logged/prompts/partA_conditionA.jsonl` 7d66ae42487694d7
- `human_validation/llm_census_logged/prompts/partA_conditionB.jsonl` cdb00483d4e29b0f
- `human_validation/llm_census_logged/prompts/partB_conditionA.jsonl` 97efffec79384054
- `human_validation/llm_census_logged/prompts/quiz_both_conditions.jsonl` 1f894b7695cf1abd
- `human_validation/llm_census_logged/prompts/removed_sentences.md` b6144264e313b71f
- `human_validation/llm_census_logged/prompts/system_stage1.txt` 61ee1acfef19f433
- `human_validation/llm_census_logged/prompts/system_stage2.txt` de2b62c7e0594758
- `human_validation/llm_census_logged/PROTOCOL.md` e7bf98ecc974ddfe
- `human_validation/llm_census_logged/raw.jsonl` c027f65c24278755
- `human_validation/llm_census_logged/result.json` 8986d0ecb570e646
- `human_validation/llm_census_logged/result.md` f1332127a0dd8dfd
- `human_validation/llm_census_logged/runs/claude-haiku-4-5_A_1.json` d319f3d0797c25a5
- `human_validation/llm_census_logged/runs/claude-haiku-4-5_A_2.json` 70f6e135796870a9
- `human_validation/llm_census_logged/runs/claude-haiku-4-5_A_3.json` 41e00dfd3af04f1c
- `human_validation/llm_census_logged/runs/claude-haiku-4-5_B_1.json` 80a3a93f7d37e0e4
- `human_validation/llm_census_logged/runs/claude-haiku-4-5_B_2.json` 474e01e5f204fb14
- `human_validation/llm_census_logged/runs/claude-haiku-4-5_B_3.json` 71835069e0ba3eca
- `human_validation/llm_census_logged/runs/claude-opus-5-5_A_1.json` c4d9ab19188b58db
- `human_validation/llm_census_logged/runs/claude-opus-5-5_A_2.json` 4db35cdc11d60b50
- `human_validation/llm_census_logged/runs/claude-opus-5-5_A_3.json` 392282288d95a810
- `human_validation/llm_census_logged/runs/claude-opus-5-5_B_1.json` c640e280fa889ab6
- `human_validation/llm_census_logged/runs/claude-opus-5-5_B_2.json` 08b14128e4cbd05e
- `human_validation/llm_census_logged/runs/claude-opus-5-5_B_3.json` 5b9a87df1572f37d
- `human_validation/llm_census_logged/runs/claude-sonnet-5-5_A_1.json` c331bb1daad12d9b
- `human_validation/llm_census_logged/runs/claude-sonnet-5-5_A_2.json` 375cfd52a6506b2e
- `human_validation/llm_census_logged/runs/claude-sonnet-5-5_A_3.json` ab4f02e3452c8dec
- `human_validation/llm_census_logged/runs/claude-sonnet-5-5_B_1.json` 899ad662872c168c
- `human_validation/llm_census_logged/runs/claude-sonnet-5-5_B_2.json` 3df1b0d9ab648e1a
- `human_validation/llm_census_logged/runs/claude-sonnet-5-5_B_3.json` 7de8b2b053288846
- `human_validation/llm_census_logged/runs/gemini-3.8-flash_A_1.json` 703d40d430ea62aa
- `human_validation/llm_census_logged/runs/gemini-3.8-flash_A_2.json` 225ea02746148398
- `human_validation/llm_census_logged/runs/gemini-3.8-flash_A_3.json` 0c0155a949bf2337
- `human_validation/llm_census_logged/runs/gemini-3.8-flash_B_1.json` 688c5ae61afe3c9e
- `human_validation/llm_census_logged/runs/gemini-3.8-flash_B_2.json` 4a751abd5fc4a849
- `human_validation/llm_census_logged/runs/gemini-3.8-flash_B_3.json` 0b94ba9bcd248a9f
- `human_validation/llm_spotcheck_v3/agreement_with_v1_human.json` 82f82d4e34db69bb
- `human_validation/llm_spotcheck_v3/package/AUDIT_v1.md` 3670c6daedac0e1b
- `human_validation/llm_spotcheck_v3/package/build_spotcheck_v3.py` 18155a0f9b014e4f
- `human_validation/llm_spotcheck_v3/package/GUIDE_v3.md` 78344a75ed3936c1
- `human_validation/llm_spotcheck_v3/package/pool_stats.json` 3aff916f4d8c4973
- `human_validation/llm_spotcheck_v3/package/score_spotcheck_v3.py` 20b7b496ea534140
- `human_validation/llm_spotcheck_v3/package/sealed_key_v3.json` a74ada0326adfbf6
- `human_validation/llm_spotcheck_v3/package/sheets/rater_1_spotcheck.xlsx` 118c608ee119a79e
- `human_validation/llm_spotcheck_v3/package/sheets/rater_2_spotcheck.xlsx` b382c43c52432608
- `human_validation/llm_spotcheck_v3/package/sheets/rater_3_spotcheck.xlsx` 309d0aa904f9d637
- `human_validation/llm_spotcheck_v3/raw.jsonl` 5111ee37df671c53
- `human_validation/llm_spotcheck_v3/README.md` 5bfea4054a3aeec0
- `human_validation/llm_spotcheck_v3/result_v3.json` cf0275f2039ace8d
- `human_validation/llm_spotcheck_v3/result_v3.md` 30107a5bff7e91c9
- `human_validation/llm_spotcheck_v3/returned/rater_1_spotcheck.xlsx` 2c1761f4ae5131d6
- `human_validation/llm_spotcheck_v3/returned/rater_2_spotcheck.xlsx` d8cd1ed56d109cdb
- `human_validation/llm_spotcheck_v3/returned/rater_3_spotcheck.xlsx` a94410bb1e75fad1
- `human_validation/rating_sheet_external.csv` 25dc14727dccbff5
- `human_validation/reference_bins.json` 6c20f6ea4f2d1816
- `human_validation/v3/_worked_examples.md` 14c4dc3d2038273d
- `human_validation/v3/answer_key_v3.json` 8f8e574cc262ae36
- `human_validation/v3/build_package_v3.py` 8e9d22c8ebb13e20
- `human_validation/v3/INSTRUCTIONS_v3.md` 4fe0eb4701d3915b
- `human_validation/v3/QUIZ_KEY_v3.md` 3e0000b26667c071
- `human_validation/v3/sheets/rater_1_stage1_quiz.xlsx` e148c9a3b1d5cb1c
- `human_validation/v3/sheets/rater_1_stage2_main.xlsx` d854c9ebbe6c10f7
- `human_validation/v3/sheets/rater_2_stage1_quiz.xlsx` c04396fd2a38dc99
- `human_validation/v3/sheets/rater_2_stage2_main.xlsx` 90526ab11de00e3b
- `human_validation/v3/sheets/rater_3_stage1_quiz.xlsx` b91a83dd25bd69c1
- `human_validation/v3/sheets/rater_3_stage2_main.xlsx` 47db580429209ae9
- `human_validation/v4/build_package_v4.py` 55777cb2e58ecb3d
- `human_validation/v4/CHECKLIST_v4.md` c51e19cf9decb192
- `human_validation/v4/export_public_v4.py` bbe0c4de8eb614b8
- `human_validation/v4/independence_checks_v4.md` 5bcf15bc6aa47057
- `human_validation/v4/independence_checks_v4.py` eef24e996c588ec1
- `human_validation/v4/INSTRUCTIONS_v4.md` 26168d4bc1bf1602
- `human_validation/v4/PROTOCOL_v4.md` ab98a506154e275d
- `human_validation/v4/public/quiz_v4.csv` b55df1235495a93d
- `human_validation/v4/public/raters_v4.csv` a8c64b6b8fcc1891
- `human_validation/v4/public/ratings_v4.csv` 52aa21fbb058a1ad
- `human_validation/v4/public/README.md` dc934d45c89bfb1e
- `human_validation/v4/public/returned_sha256.txt` d9fa712c99fe6bee
- `human_validation/v4/result_v4.md` a13c4e9afcb3c163
- `human_validation/v4/score_v4.py` 0195fb0c067746b5
- `human_validation/v4/SEALED_build.json` 941b1d5e397bd97a
- `human_validation/v4/to_send/rater_1_v4.zip` bd0eddd713855be7
- `human_validation/v4/to_send/rater_2_v4.zip` 8c296c20cdbc8b7f
- `human_validation/v4/to_send/rater_3_v4.zip` 06b47d2e187fea11
- `human_validation/v4/WORKED_EXAMPLES_v4.md` 68da3e15c58c4cd8
- `manuscript/eacl2027_camera_ready.aux` 8c97b428c3832d1b
- `manuscript/eacl2027_camera_ready.tex` a13e8230d2c53da9
- `realistic_structure/2wiki/holdout200/_kgswap_hipporag2_kg_wikidata.json` 404c23d72f107e2b
- `realistic_structure/2wiki/holdout200/_kgswap_ppr_kg_wikidata.json` 2cec215470bd9b9e
- `realistic_structure/2wiki/holdout200/_kgswap_proposed_v3_kg_wikidata.json` 7e0469cfb887ec32
- `realistic_structure/2wiki/holdout200/_kgswap_proposed_v5_graphcentric_kg_wikidata.json` 120ced06cc6671a7
- `realistic_structure/2wiki/holdout200/kg_wikidata/build_summary.json` 17c3af90eea6206f
- `realistic_structure/2wiki/holdout200/kg_wikidata/proposed_kg_edges.json` e3cffad8b8f4df07
- `realistic_structure/2wiki/holdout200/kg_wikidata/proposed_kg_nodes.json` b503ce3c45e5ff22
- `realistic_structure/2wiki/sub200/_kgswap_hipporag2_kg_wikidata.json` 9939245b6248b35b
- `realistic_structure/2wiki/sub200/_kgswap_ppr_kg_wikidata.json` 68537f673465a1f4
- `realistic_structure/2wiki/sub200/_kgswap_proposed_v3_kg_wikidata.json` 08bf5a710c2a11a5
- `realistic_structure/2wiki/sub200/_kgswap_proposed_v5_graphcentric_kg_wikidata.json` 53027678353a554e
- `realistic_structure/2wiki/sub200/kg_wikidata/build_summary.json` 95c3a1168daad5ec
- `realistic_structure/2wiki/sub200/kg_wikidata/proposed_kg_edges.json` 7c3e1d2cc88e6328
- `realistic_structure/2wiki/sub200/kg_wikidata/proposed_kg_nodes.json` 33356edf8206221e
- `realistic_structure/metaqa/holdout200/_kgswap_hipporag2_kg_fullkb.json` e4537cdceebd9769
- `realistic_structure/metaqa/holdout200/_kgswap_ppr_kg_fullkb.json` a2043e88dc9ba175
- `realistic_structure/metaqa/holdout200/_kgswap_proposed_v3_kg_fullkb.json` f93421643309fcdf
- `realistic_structure/metaqa/holdout200/_kgswap_proposed_v5_graphcentric_kg_fullkb.json` 649c730a81e76633
- `realistic_structure/metaqa/holdout200/kg_fullkb/proposed_kg_edges.json` 6e401d70fe95bb32
- `realistic_structure/metaqa/holdout200/kg_fullkb/proposed_kg_nodes.json` 9fb6ac5c15cde5b0
- `realistic_structure/metaqa/sub200/_kgswap_hipporag2_kg_fullkb.json` 06c2ec82e5c9f5fd
- `realistic_structure/metaqa/sub200/_kgswap_ppr_kg_fullkb.json` 41cb07c89101d814
- `realistic_structure/metaqa/sub200/_kgswap_proposed_v3_kg_fullkb.json` ed6b530a4a148960
- `realistic_structure/metaqa/sub200/_kgswap_proposed_v5_graphcentric_kg_fullkb.json` c8e4e3abd2466721
- `realistic_structure/metaqa/sub200/kg_fullkb/proposed_kg_edges.json` c4093d0b62374e8e
- `realistic_structure/metaqa/sub200/kg_fullkb/proposed_kg_nodes.json` a8d3ac48f05d3ded
- `results/bge_pooling_check.json` 4035a4aa775d35ed
- `results/bge_pooling_check_stark_graphragbench.json` bb02c3781f860c39
- `results/census_kappa_ci_output.txt` 0f9860486b64ff37
- `results/concentration_2wiki_summary.md` 37196e6ba012b3b5
- `results/dense_bge_swap_summary.md` b0a44ca61134b7f8
- `results/dense_bge_swap_summary_cls.md` 43fd919d1a71376c
- `results/e2e_dose_across_cells.json` 422ec30a952be75a
- `results/e2e_dose_across_cells.md` 3c34846d3dd484a0
- `results/e2e_dose_across_cells_cls.json` bd1cc24b821774e8
- `results/e2e_dose_across_cells_cls.md` 4a8af588327174a2
- `results/e2e_qa/claude_sonnet_5_5/batch_results_raw.jsonl` f062a2f46b38857f
- `results/e2e_qa/claude_sonnet_5_5/batch_state.json` e40e4939d21c3a32
- `results/e2e_qa/claude_sonnet_5_5/generations.jsonl` e1df3378f48c11fb
- `results/e2e_qa/claude_sonnet_5_5/per_question.jsonl` 70a1430024f5e111
- `results/e2e_qa/claude_sonnet_5_5/pilot_raw.jsonl` cd53fdb8dacb7052
- `results/e2e_qa/claude_sonnet_5_5/PROTOCOL.md` 79a2c092a82e47d2
- `results/e2e_qa/claude_sonnet_5_5/summary.json` 3c58512bd7cba56f
- `results/e2e_qa/claude_sonnet_5_5/summary.md` 2fc318dfb581154a
- `results/e2e_qa/judge_cache_llama3_1_8b.jsonl` 0aea0a9ede99322c
- `results/e2e_qa/judge_cache_mistral_small_24b.jsonl` 7106589c148b1f9c
- `results/e2e_qa/judge_cache_qwen2_5_14b.jsonl` 6d1234aa32d1d487
- `results/e2e_qa/llama3_1_8b/generations.jsonl` cd7cd4474add9228
- `results/e2e_qa/llama3_1_8b/judge_mistral_small_24b.json` d1ca77e86f60d7fc
- `results/e2e_qa/llama3_1_8b/judge_mistral_small_24b.md` c57e49dcaaf79c35
- `results/e2e_qa/llama3_1_8b/judge_qwen2_5_14b.json` ed92d9df575eeecc
- `results/e2e_qa/llama3_1_8b/judge_qwen2_5_14b.md` 4097809c780eeaf6
- `results/e2e_qa/llama3_1_8b/memorization_2wiki.jsonl` eeb8029eb35680aa
- `results/e2e_qa/llama3_1_8b/memorization_strata.json` 6f1e4a1660c7ff7b
- `results/e2e_qa/llama3_1_8b/memorization_strata.md` 3289c523f51157f3
- `results/e2e_qa/llama3_1_8b/per_question.jsonl` d4c3ed434cb0bae3
- `results/e2e_qa/llama3_1_8b/summary.json` 9e7f9a63bccd3e4e
- `results/e2e_qa/llama3_1_8b/summary.md` 8df5e5ad02fbe86c
- `results/e2e_qa/mistral_small_24b/generations.jsonl` 2144acc21cce9064
- `results/e2e_qa/mistral_small_24b/per_question.jsonl` 26004ca02a62aaa6
- `results/e2e_qa/mistral_small_24b/summary.json` 8f2506f82839816a
- `results/e2e_qa/mistral_small_24b/summary.md` 1a5a772b689ea3dc
- `results/e2e_qa/qwen2_5_14b/generations.jsonl` 099322d7e18d7be9
- `results/e2e_qa/qwen2_5_14b/judge_llama3_1_8b.json` 2cd4161ec1c78e7c
- `results/e2e_qa/qwen2_5_14b/judge_llama3_1_8b.md` ee828b9d57b8eca8
- `results/e2e_qa/qwen2_5_14b/judge_mistral_small_24b.json` ed45247be303be8d
- `results/e2e_qa/qwen2_5_14b/judge_mistral_small_24b.md` a16ce95d50ce1187
- `results/e2e_qa/qwen2_5_14b/memorization_2wiki.jsonl` 51d1564de3b86d0e
- `results/e2e_qa/qwen2_5_14b/memorization_strata.json` 3963b556c460318b
- `results/e2e_qa/qwen2_5_14b/memorization_strata.md` 935dc9c362bb33fb
- `results/e2e_qa/qwen2_5_14b/per_question.jsonl` 2edd27288492b5b4
- `results/e2e_qa/qwen2_5_14b/summary.json` 62879d23b3716355
- `results/e2e_qa/qwen2_5_14b/summary.md` a1279dca2955c26b
- `results/e2e_qa/variant_pairs.json` b3a0cf7ed8c36731
- `results/e2e_qa/variant_pairs.md` 6d1918a8e766a191
- `results/e2e_qa_cls/judge_cache_mistral_small_24b.jsonl` 045a3f2945e2c0be
- `results/e2e_qa_cls/llama3_1_8b/judge_mistral_small_24b.json` 9cfd10df0a26e0ee
- `results/e2e_qa_cls/llama3_1_8b/judge_mistral_small_24b.md` 96113e88f89566e4
- `results/e2e_qa_cls/llama3_1_8b/per_question.jsonl` bf66af4b21d66eb5
- `results/e2e_qa_cls/llama3_1_8b/summary.json` 0727a08543d3239c
- `results/e2e_qa_cls/llama3_1_8b/summary.md` 7ef49619497649b4
- `results/e2e_qa_cls/mistral_small_24b/summary.json` 8f2506f82839816a
- `results/e2e_qa_cls/qwen2_5_14b/judge_mistral_small_24b.json` d3320c4e67663181
- `results/e2e_qa_cls/qwen2_5_14b/judge_mistral_small_24b.md` 800c7b6f69ba6fa6
- `results/e2e_qa_cls/qwen2_5_14b/per_question.jsonl` eb8f326e8e2079ee
- `results/e2e_qa_cls/qwen2_5_14b/summary.json` 6e654cc730939952
- `results/e2e_qa_cls/qwen2_5_14b/summary.md` 3c11eb47b3c89575
- `results/figures/fig_headline_margins.pdf` 040b161e82703244
- `results/fullkb_metaqa_summary.md` dcc842478ee4dda4
- `results/grade_stark_provenance/inputs/amazon_first_release_ids.csv` 593f95921bea5f5e
- `results/grade_stark_provenance/per_query_stark.jsonl` ac81f232bb9218ff
- `results/grade_stark_provenance/sources/grade_d4b16af_code__claims__generate_claim_from_sentence.py.txt` 1f060268f1ab723a
- `results/grade_stark_provenance/sources/grade_d4b16af_code__clustering__add_contextually_same_path.py.txt` e4bb05e04dc8413d
- `results/grade_stark_provenance/sources/grade_d4b16af_code__db__generate_db_from_chunk.py.txt` e4021517accd12b6
- `results/grade_stark_provenance/sources/grade_d4b16af_code__eval__retriever_correlation.py.txt` d1a43b09deef4717
- `results/grade_stark_provenance/sources/grade_d4b16af_code__eval__score_llm_eval.py.txt` 01ffcfef1305eb6c
- `results/grade_stark_provenance/sources/grade_d4b16af_code__graph__find_DAG_path_shortest.py.txt` a332a7449a56951b
- `results/grade_stark_provenance/sources/grade_d4b16af_code__ingestion__crawl_news.py.txt` 89d303e09f76d4e0
- `results/grade_stark_provenance/sources/grade_d4b16af_code__qa__final_validation.py.txt` 89f6cb4ba0326326
- `results/grade_stark_provenance/sources/grade_d4b16af_code__qa__qa_generation.py.txt` d6ddb7ed5a86bcaf
- `results/grade_stark_provenance/sources/grade_d4b16af_code__sampling__sample_path.py.txt` 27949491ae5e61b3
- `results/grade_stark_provenance/sources/grade_d4b16af_code__triples__extract_triples.py.txt` eba1ab832081f627
- `results/grade_stark_provenance/sources/hf_trees.json` bddac74581df5e99
- `results/grade_stark_provenance/sources/LICENSE_GRADE.txt` fe55d7059d661fab
- `results/grade_stark_provenance/sources/LICENSE_STaRK.txt` 749943c53fde2b13
- `results/grade_stark_provenance/sources/openreview_forum_QSS5cGmKb1_notes.json` ba142534b52a1711
- `results/grade_stark_provenance/sources/stark_df59d09_hybrid_qa_REFERENCES.md` 34556222c6891e1f
- `results/grade_stark_provenance/sources/stark_knowledge_base_py.txt` 2dfe053f072cb911
- `results/grade_stark_provenance/sources/THIRD_PARTY_NOTICE.md` 0722899334612914
- `results/grade_stark_provenance/summary.json` 1e7ce5f3b8bae9a4
- `results/grade_stark_provenance/summary.md` b72dd9118e25ff7a
- `results/graphragbench_swap_summary.md` 2628c5c1bcf554f8
- `results/grb_provenance/inputs/PINNED_INPUTS.md` 8ecdac892a4691ad
- `results/grb_provenance/per_question.jsonl` fcac215b96bfad7c
- `results/grb_provenance/sources/openreview_forum_i9q9xDMjG7_notes.json` 18f3059667b8a43c
- `results/grb_provenance/summary.json` 9b33a2175015765b
- `results/grb_provenance/summary.md` 6d4a41f6649dffb6
- `results/headline_margins_summary.json` d5df16b9ca92bd7f
- `results/headline_margins_summary.md` bb7058002da861fc
- `results/index_precision_PREREG.md` 1485b4d9db1b6f68
- `results/index_precision_summary.json` ac87d3b3a31c57ad
- `results/index_precision_summary.md` 714a638a9b6ae617
- `results/kg_concentration_2wiki.json` 358c1c5c2d4249e8
- `results/kg_fullkb_metaqa.json` 771e1fe58111df10
- `results/kg_passage_link_counts.json` e72cea7a5a4f3bfb
- `results/kg_swap_dense_bge_m3.json` b01c34fd52dcfc1e
- `results/kg_swap_dense_bge_m3_cls.json` 8141eb1215fe79c8
- `results/kg_swap_graphragbench.json` 08667991fb7752c4
- `results/kg_swap_graphragbench_ppr.json` 0392e25322fd5945
- `results/kg_swap_graphragbench_proposed_v5_graphcentric.json` b5482b66e6b5b234
- `results/kg_swap_variant_2wiki_kg_wikidata.json` b59e59087d2280af
- `results/kg_zero_edge_2wiki.json` 7cd6d80fe047f16e
- `results/multimodel_census_raw.json` 23d8191b42849a4f
- `results/openie_extractor_exposure/per_triple.jsonl` a71a4a6775401752
- `results/openie_extractor_exposure/summary.json` b79827d8cab92b41
- `results/openie_extractor_exposure/summary.md` 32cd8b582de185ad
- `results/published_pairs_class_stats.json` d5415473a9b84fbc
- `results/published_pairs_classification.md` f7ce21a7c5640038
- `results/rater_robustness_summary.json` 695491177f9c14ce
- `results/rater_robustness_summary.md` 25bb571a0d6372d8
- `results/rater_validation_summary.json` bff96849a3c89557
- `results/rater_validation_summary.md` d7945cdb6c5932b3
- `results/swap_coverage_graphragbench.json` a0d1d0bbbdd5b48e
- `results/swap_samplesize_output.txt` 7b41213373c583f4
- `results/tables/dose_curves.json` d1d9088315c91adc
- `results/tables/e2e_multiple_comparison.json` 322459cdb24802b7
- `results/tables/e2e_numbers.json` 42d9b6b9cf669b09
- `results/tables/tab_decirc.md` 81001bf7adfe9ec4
- `results/tables/tab_e2e_merged_v3.tex` ed7c9fc8d54cef14
- `results/tables/tab_e2e_swap_em.md` 9a6314f10731e620
- `results/tables/tab_e2e_swap_f1.md` 12c621fb1d700af6
- `results/tables/tab_linkcounts.tex` 7e6fb9978d25c6f5
- `results/tables/tab_mediation.tex` 0b74fd294a69c2c8
- `results/tables/tab_memorization.tex` abec4cca1de54634
- `results/tables/tab_memorization_main.tex` 91cdeb04d99421e2
- `results/tables/tab_quotes_full.tex` 9b2c68b925725882
- `results/tables/tab_readerref.tex` aa25922e1e8cb227
- `results/tables/tab_third_reader.tex` f5961f23958b2d90
- `results/twowiki_comp200/e2e_qa/llama3_1_8b/generations.jsonl` 727236124cde1938
- `results/twowiki_comp200/e2e_qa/llama3_1_8b/per_question.jsonl` 5fe3b4aaf948ec14
- `results/twowiki_comp200/e2e_qa/llama3_1_8b/summary.json` be613beb36cb0674
- `results/twowiki_comp200/e2e_qa/llama3_1_8b/summary.md` c6dcb781dc0e886c
- `results/twowiki_comp200/e2e_qa/qwen2_5_14b/generations.jsonl` 08e65237741929d4
- `results/twowiki_comp200/e2e_qa/qwen2_5_14b/per_question.jsonl` 0e5d8594b044a5b5
- `results/twowiki_comp200/e2e_qa/qwen2_5_14b/summary.json` d56e587952a5a1d0
- `results/twowiki_comp200/e2e_qa/qwen2_5_14b/summary.md` 60717838b70c68d1
- `results/twowiki_comp200/hipporag2/summary.json` c1047bafa33d5a47
- `results/twowiki_comp200/hipporag2/summary.md` e9ed0b79fd5d3861
- `results/twowiki_comp200/PROTOCOL.md` aaa62ea30225be80
- `results/twowiki_comp200_summary.json` aaf00bcf37a66b1b
- `results/twowiki_comp200_summary.md` 7f5b6ec94d5dca26
- `results/variant_2wiki_kg_wikidata_summary.md` 70d1910d7f892650
- `results/zero_edge_2wiki_summary.md` 5f6e49776a29de85
- `zero_edge/2wiki/holdout200/_kgswap_ppr_silver_noedges.json` 5dc2d639d39ec6fe
- `zero_edge/2wiki/holdout200/_kgswap_proposed_v3_silver_noedges.json` 3c7ac426a727fe70
- `zero_edge/2wiki/holdout200/_kgswap_proposed_v5_graphcentric_silver_noedges.json` 1747251fe128d640
- `zero_edge/2wiki/holdout200/kg_silver_noedges/proposed_kg_edges.json` 4f53cda18c2baa0c
- `zero_edge/2wiki/holdout200/kg_silver_noedges/proposed_kg_nodes.json` d83e9838bd0e9c92
- `zero_edge/2wiki/sub200/_kgswap_ppr_silver_noedges.json` 95272c83948024dd
- `zero_edge/2wiki/sub200/_kgswap_proposed_v3_silver_noedges.json` 269dc5b7ed9ed4dc
- `zero_edge/2wiki/sub200/_kgswap_proposed_v5_graphcentric_silver_noedges.json` 57f3494709bedc64
- `zero_edge/2wiki/sub200/kg_silver_noedges/proposed_kg_edges.json` 4f53cda18c2baa0c
- `zero_edge/2wiki/sub200/kg_silver_noedges/proposed_kg_nodes.json` d61b928ed7e77a70
