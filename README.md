# Retrieval-gold provenance audit: code and data

Code, data, and rater records for the paper "Auditing Retrieval-Gold Provenance in Knowledge-Grounded Question Answering Benchmarks" by Hyeonmin Kim and GyeongTaek Lee (Findings of EACL 2027).

The paper asks where the retrieval gold of knowledge-grounded question answering benchmarks comes from. It scores eleven benchmarks with a rubric (the census) and then measures, with controlled knowledge-graph swaps on four of them, what happens when a graph retriever ranks over the structure the gold was derived from. This repository holds the census records, the human and language-model ratings, the knowledge graphs, the frozen per-question retrieval and reader outputs, and the scripts behind the paper's tables and figures.

## Layout

The materials come in two parts. Each part has its own README with a file-by-file index and the commands that reproduce the reported numbers.

| Folder | Contents |
|---|---|
| `aug2026_original/` | The census (rubric, per-axis assignments, verbatim quotes, reliability scripts), the short-prompt human re-scoring, the reference benchmark, and the KG swaps on 2WikiMultiHop, MetaQA, and STaRK-Prime |
| `oct2026_additions/` | The GraphRAG-Bench swap, HippoRAG 2, the end-to-end reader runs, the realistic-structure and gold-concentration controls, the training-exposure probes, the three-rater human study, the logged LM raters, the judge validation, the source traces, and the scripts for the tables and figures |

`oct2026_additions/README.md` maps the names used in files (for example `kg_silver`, `sub200`, `proposed_v3`) to the terms of the paper and lists a checksum for every file in that folder.

## Where to find what the paper reports

| In the paper | Start here |
|---|---|
| Census, Table 1 and Appendix B | `aug2026_original/census/` |
| Human raters, Appendix B.4 | `oct2026_additions/human_validation/v4/public/` |
| Logged LM raters, Appendix B.5 | `oct2026_additions/human_validation/llm_census_logged/` |
| Source traces, Appendix B.10 | `oct2026_additions/results/grb_provenance/`, `oct2026_additions/results/grade_stark_provenance/` |
| Published pairs, Appendix B.11 | `aug2026_original/census/published_gains_*`, `oct2026_additions/results/published_pairs_*` |
| Controlled KG swap, Table 3 | `aug2026_original/swap/`, `aug2026_original/metaqa/`, `oct2026_additions/external/metaqa/`, `aug2026_original/stark/`, `oct2026_additions/graphragbench_swap/` |
| What carries the inflation, Table 4 | `oct2026_additions/zero_edge/`, `oct2026_additions/concentration/`, `oct2026_additions/realistic_structure/`, `oct2026_additions/hipporag2_swap/`, and the transplant results under `aug2026_original/swap/` |
| Index precision, Appendix C.5 | `oct2026_additions/results/index_precision_*` |
| Training exposure, Appendix C.6 | `oct2026_additions/results/e2e_qa/<reader>/memorization_strata.*`, `oct2026_additions/results/openie_extractor_exposure/` |
| Answer accuracy with fixed readers | `oct2026_additions/results/e2e_qa/` |
| HippoRAG 2, Appendix C.7 | `oct2026_additions/hipporag2_swap/` |
| Judge validation, Appendix C.10 | `oct2026_additions/human_validation/judge_spotcheck_v1/`, `oct2026_additions/human_validation/judge_spotcheck_v4/`, `oct2026_additions/human_validation/llm_spotcheck_v3/` |
| Reference benchmark, Appendix D.1 | `aug2026_original/benchmark/` |
| Tables as typeset in the paper | `oct2026_additions/results/tables/` |

## Checking a number

The census agreement needs only Python and the files here:

    cd aug2026_original/census
    python reliability_kappa_ci.py

It prints the agreement of the two primary raters over the 29 axis-level judgments: 25 exact, quadratically weighted kappa 0.977, ordinal Krippendorff alpha 0.938 (Table 9 of the paper). The two READMEs list the other commands and say which of them need models or upstream data.

## What is not included

Text that the authors could not confirm they may redistribute is left out, with identifiers, hashes, and rebuild steps in its place: the 2WikiMultiHop (Wikipedia) passage text, the GraphRAG-Bench medical corpus, and the passages of the reference benchmark (its papers are listed with their arXiv URLs in `aug2026_original/benchmark/corpus_manifest.json`). The one place that quotes benchmark text is the per-unit rater task, 130 gold units among which are 40 supporting sentences of 2WikiMultiHop, 20 of HotpotQA, and the openings of 20 MuSiQue paragraphs (see the third-party section of `oct2026_additions/README.md`). The workbooks returned by the human raters are not included because their document properties and save paths name the raters' institution and their `Rater_info` sheets hold the raters' initials; the ratings are released as exports without those fields. The filled sheets in `oct2026_additions/human_validation/llm_spotcheck_v3/returned/` are a different case: three language models filled them, not people. No API key is included.

## Licenses

Code written by the authors is under the MIT License (`LICENSE`); the license does not cover data files or the third-party code files copied under `oct2026_additions/results/grade_stark_provenance/sources/`, which keep their own licenses (`THIRD_PARTY_NOTICE.md` there). Annotations, ratings, census records, benchmark records, and result files written by the authors are under CC BY 4.0 (`LICENSE-DATA.md`). Third-party data keep their own terms, listed in `aug2026_original/DATA_LICENSES.md` and in the "Licenses and third-party data" section of `oct2026_additions/README.md`; files derived from those sources remain subject to them.

## Citation

```bibtex
@inproceedings{kim2027auditing,
  title     = {Auditing Retrieval-Gold Provenance in Knowledge-Grounded Question Answering Benchmarks},
  author    = {Kim, Hyeonmin and Lee, GyeongTaek},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```
