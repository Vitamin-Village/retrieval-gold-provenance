# First part of the released materials

First part of the materials released with "Auditing Retrieval-Gold
Provenance in Knowledge-Grounded Question Answering Benchmarks" (Findings of
EACL 2027). This folder holds the census records, the human validation, the
provenance-audited benchmark, and the 2WikiMultiHop KG-swap experiment, laid
out so that each claim in the paper points at a file here.

## Contents

Top-level `ENVIRONMENT.md`, `requirements*.txt`, `LICENSE.md`, `NOTICE.md`,
and `DATA_LICENSES.md` state the verified environment, dependency limits,
artifact scope, and third-party source/license status.

`census/` holds the scoring rubric (rubric.md, with the rater prompt verbatim),
the adjudicated per-axis assignments for all 29 axis-level judgments
(per_axis_assignments.json, with both raters' bins, the adjudicated C value,
and the derivation mechanism per row), rater A's complete scoring records with
the verbatim construction quote behind each assignment (rater_A_assignments.md
and rater_A_assignments_expansion.md), the quotes for the
benchmarks not covered by those records (quotes_remaining_benchmarks.md), the
rater-model identities and run dates for the later five-model sensitivity run
(model_identities.md), the primary-rater provenance limitation
(PRIMARY_RATER_PROVENANCE.md), the scripts that
compute every reported reliability statistic, and the cross-model census
script with its raw five-model matrix. The census-versus-published-gains
table of the paper's appendix is backed by published_gains_sources.md (the
thirty published graph-versus-lexical pairs with their source tables) and
published_gains_summary.py with its stats JSON (the per-benchmark medians and
the descriptive rank correlations). published_gains_sensitivity.py and its CSV
recompute those correlations when the further pairs of the cited tables are
added (the sensitivity stated in Appendix B.11).

Two notes on reading the census records next to the paper. First, the
`mechanism` field of per_axis_assignments.json has three labels.
`KG-derivation` and `independent` are "KG derivation" and "independent" in the
paper's full per-axis table (Table 8). `answer-surface` covers seven rows that
Table 8 labels more finely: "KG, answer string" (the answer axes of
2WikiMultiHop, GraphRAG-Bench, and the upper anchor), "KG-guided selection"
(GRADE passage, STaRK textual), "answer-guided selection" (Natural Questions
passage), and "answer-surface" (TriviaQA evidence). The bins and C values are
the same in both, and no script reads this field. Second, for GRADE and STaRK
the quotes in rater A's records follow arXiv v1, whose wording later versions
changed. Rater A's two records are left as written, and a note at the end of
each lists the strings in quote marks that are not word for word in the
source. In the quote column these are two GraphRAG-Bench quotes that differ
from the paper by one word each and one GRADE string that is not in the
paper; the others are shortened or reworded fragments in rater A's notes.
`oct2026_additions/human_validation/llm_census_logged/excerpts.json`
checks the census-record quotes that were shown to the later raters against
the source text (`workbook_excerpt_check`) and lists the description
statements that no source sentence states. The quotes printed in the paper's
Tables 6 and 7 are in `oct2026_additions/results/tables/tab_quotes_full.tex`.

`human_validation/` holds the blind human re-scoring of all 29 judgments by an
independent rater who is not an author: the rater instructions, the randomized
rating sheet (rating_sheet_external.csv), the sealed reference key, the
completed result (external_rater_result.json), and the scripts that build the
sheet and score it.

`benchmark/` contains the released C=0 benchmark records: the frozen
112-row single-hop file (111 distinct question strings), the frozen two-hop
slice, both data cards, the source-passage identifiers, the dual-verifier audit
trail for the two-hop slice, the two-hop leakage-gate report that reconciles
the 14 dual-kept candidates with the 13 released questions
(`twohop_leakage_report.json`: the gate removed `qa00001`), the frozen-112
re-evaluation, and the construction pipeline under benchmark/pipeline/ (the
generation, verification, leakage-gate, and content-blind freeze code; the
generation and verification prompts are in that code and in the paper's
appendix). The operative verification prompts and decision rule are in
benchmark/pipeline/verify_answer_papers.py and rater_code/cross_judge.py.
`benchmark/singlehop_verifier_spotcheck/` contains the 40-item blind
cross-model protocol spot-check behind the rounded kappa=0.79 claim, including
raw decisions and a standard-library-only recomputation. It does not imply
that all 112 frozen records were independently double-annotated. The two-hop
slice is a 13-item feasibility artifact and is domain-imbalanced (10 textile,
2 steel, 1 construction, 0 semiconductor), as recorded in its data card.
Because per-paper redistribution rights were not verified, source passage
text is excluded; exact end-to-end retrieval reruns require lawful
reacquisition using the released identifiers and URLs
(`benchmark/corpus_manifest.json` lists the 205 papers of the corpus with the
arXiv version that was downloaded and the hashes of its PDF and extracted text).

`rater_code/` carries the shared judge machinery and model backends used by
the benchmark verifiers; model identities and decoding settings are in
cross_judge_backends.py.

`paper/` and `prototype/` contain the real import-compatible support modules
used by the shipped benchmark and retrieval drivers. They are included so that
this folder does not depend on the authors' working tree.

`swap/` is the 2WikiMultiHop KG-swap experiment: the driver and analysis
scripts under swap/code/, the aggregated results under swap/results/, and the
audited data under swap/data/ for both 200-question splits (questions,
manifests, the KG variants kg_silver and kg_openie plus the
transplant variants kg_openie_svocab, kg_openie_svocab_noprov, and
kg_openie_resorted, and the per-question result vectors, the _kgswap_*.json
files, from which every swap statistic in the paper can be recomputed). This
release omits the two Wikipedia passage-text files because their
exact revision/license metadata was not available and, for the same reason,
ships both `manifest.json` files with the `abstract` text field emptied,
leaving per-record SHA-256 hashes and character counts for verification after
re-fetching (see `swap/data/PASSAGE_TEXT_REMOVED.md`); title/URL attribution and
upstream acquisition information remain under licenses/. The
inventory-transplant decomposition of the paper's appendix is
build_transplant_kgs.py (deterministic KG builder) plus
_kg_transplant_2wiki.py (paired driver), with aggregated results in
swap/results/kg_transplant_2wiki*.json.

`metaqa/` is the second causal point of Appendix C.8: the MetaQA KG-swap that
replicates the 2WikiMultiHop effect at a second census point (C_K=1.0).
metaqa/code/ holds the design note, the subset builder
(build_metaqa_swap_data.py, which regenerates the 200-question splits from the
public MetaQA and WikiMovies releases, cited in the paper), and the paired
swap driver (_kg_swap_metaqa.py); metaqa/results/ holds the per-retriever swap
statistics (kg_swap_metaqa*.json) and the gold-title coverage of each KG
(swap_vocab_align_metaqa.json); metaqa/data/ holds the built sub200 and
holdout200 splits (questions, passages, manifests, silver and OpenIE KGs;
heavy checkpoints and build logs are dropped, regenerable from the builder).
The intermediate-tick per-unit check of Appendix B.9 is
census/perunit_sentence_axis.py with its per-question result and summary.
`python census/perunit_sentence_axis.py` verifies the archived 400-item records
and both reported aggregates without external data. A full recomputation from
the upstream 2Wiki release is available with `--raw-zip`; the two audited QA
files default to their released `swap/data/` paths. The upstream raw ZIP is not
redistributed. The second check uses
census/perunit_graphragbench.py: its default command likewise validates all
archived per-question totals, while `--source` enables a full recomputation
from an independently obtained GraphRAG-Bench medical source file. Both full
recomputations write `.recomputed.*` siblings and preserve the frozen files.

`stark/` is the third swap of Appendix C.9: the STaRK-Prime KG-swap at the
non-maximal census point (C_K=0.75), reported honestly as an attenuated lower
bound, not a confirmed causal point. stark/code/ holds the design note, the
subset builder and the entity-id-keyed silver-KG rebuilder, and the paired
swap driver; stark/results/ holds the per-retriever swap statistics
(kg_swap_stark*.json, mostly non-significant) and the gold-name coverage of
each KG; stark/data/ holds the built sub200 and holdout200 splits (the
STaRK-Prime SKB raw release is not redistributed here; the builder regenerates
the subset from the public snap-stanford/stark release cited in the paper).

The system-side alignment analysis of the paper's appendix (coverage
stratification, inherited overlap, and the partial-mapping q-curve) lives in
swap/analysis/: coverage_stratification.py stratifies the released OpenIE
per-question vectors by gold-title coverage (results in
swap/results/coverage_stratification.json), inherited_overlap.py measures how
much of the gold entity inventory the self-built OpenIE KG inherits (results
in swap/results/inherited_overlap.json), and the q-curve conditions
kg_openie_svocab_q25/q50/q75 under swap/data/ are built by
build_transplant_kgs.py (deterministic nested selection, seed 20260612;
q=0 reproduces kg_openie_svocab_noprov and q=1 reproduces kg_openie_svocab)
and evaluated by _kg_transplant_2wiki.py --q-curve, with aggregated results in
swap/results/kg_transplant_q_2wiki*.json and the test suite in
swap/code/tests/test_transplant_kgs.py. The two analysis scripts are
post-processing only and run from the package layout directly.

## Reproducing the reported numbers

The commands below are run from this folder (`aug2026_original/`), which is
the package root.

Census reliability. From the package root:

    cd census
    python reliability_kappa_ci.py
    python reliability_confusion_clusterboot.py

The first prints, as its first block, the census agreement of Table 9
(n=29, exact 25/29, quadratic-weighted Cohen kappa 0.977 with bootstrap
95% CI, ordinal Krippendorff alpha 0.938). The second
prints the 5x5 confusion matrix (25 diagonal, 4 adjacent, none >=2 apart) and
the benchmark-level clustered bootstrap CI. Both are deterministic (fixed
seeds, standard library only).

Human validation. From the package root:

    cd human_validation
    python score_external_rater.py --rated-by "independent rater, not an author (blind to reference bins, randomized sheet order)"

This scores the external rater's sheet against the sealed reference bins and
prints exact agreement 20/29, quadratic-weighted Cohen kappa 0.908, and
ordinal Krippendorff alpha 0.879, rewriting external_rater_result.json.

Cross-model census. census/cross_model_census.py is the exact script that
produced census/cross_model_census_raw.json (the shipped five-model matrix).
Re-running it requires an Anthropic API key and a local Ollama daemon with the
three open-weight models pulled.

Published-gain summary. From the package root:

    python census/published_gains_summary.py

This rewrites `census/published_gains_summary_stats.json` and the corresponding
PNG/PDF. It reproduces the KG-only sparse-baseline benchmark correlation
($\rho=0.8563$) and the headline-C-hat sensitivity ($\rho=0.6831$) from the
30 transcribed pairs without network access.

Single-hop verifier reliability and acyclicity guard. From the package root:

    python benchmark/singlehop_verifier_spotcheck/recompute_kappa.py
    python -m pytest -q paper/gold_v2/tests/test_no_kg_imports.py

The first independently reconstructs kappa=0.788520 from 35 non-abstain pairs
in the 40-item blind cross-model protocol spot-check. The second statically
checks every shipped `paper/gold_v2` production module for forbidden KG-side
imports.

Inventory-transplant unit tests. From the package root:

    python -m pytest -q swap/code/tests/test_transplant_kgs.py

A run from a fresh copy of this folder yields 32 passes and 6 documented skips.
The skips are the deliberately unbuilt optional devocabulary condition and
checks that need the real-data layout of the authors' working tree; fixture-based builder,
determinism, statistical-helper, mapping, and q-curve tests execute normally.

KG swap. The import-compatible `paper/` and `prototype/` support modules are
shipped, so the retrieval and swap drivers import from a fresh copy of this folder.
The live drivers (the `_kg_swap_*` and `_kg_transplant_*` evaluation drivers
and `_swap_dose_response.py`) re-run retrieval and therefore need the withheld
passage text restored under the original `paper/external/...` layout (see
`swap/data/PASSAGE_TEXT_REMOVED.md`); their frozen per-question vectors are
shipped, so no reported number depends on re-running them. The post-processing
scripts -- `swap/code/_swap_stats.py`, `_swap_winpattern.py`,
`_swap_vocab_align.py`, `_swap_samplesize.py`, `swap/analysis/*.py`, and
`stark/analysis/stark_headroom_swap.py` -- read the released layout directly
and reproduce the shipped `swap/results/` and `stark/results/` JSONs from a
fresh copy of this folder. Set `RAG_V3_DENSE_BACKEND=deterministic`; install the
optional dependencies listed in `requirements-optional.txt` for non-offline
retrieval backends.

## Notes on the released files

Absolute filesystem paths in result-JSON metadata fields were rewritten to
paths relative to the authors' working tree. Non-ASCII characters in the 2Wiki-derived data
files (Wikipedia content, e.g. Korean person names) are stored as JSON
unicode escapes; the parsed content is unchanged. Third-party names,
affiliations, and emails inside the scraped source passages are dataset
content and are retained. Determinism: swap evaluations run with
RAG_V3_DENSE_BACKEND=deterministic and fixed seeds; bootstrap CIs are seeded
in the scripts.
