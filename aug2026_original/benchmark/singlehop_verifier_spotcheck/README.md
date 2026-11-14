# Single-hop verification-protocol spot-check

This directory supports the paper's rounded Cohen kappa value of 0.79 for the
single-hop answer-paper verification protocol. It is not a claim that all 112
frozen benchmark records were independently double-annotated.

Protocol recorded in `kappa_verdict.json`:

- generator: local `mistral-small:24b`;
- primary verifier: Google `gemini-2.5-flash`;
- blind second rater: Anthropic Claude Haiku 4.5, with the rubric in
  `rater_rubric_translation.md` (English translation of the Korean original);
- sampled tasks: 40;
- non-abstain matched pairs used for kappa: 35;
- exact matches among used pairs: 29;
- observed agreement among used pairs: 29/35 = 0.828571;
- expected agreement: 0.189388;
- Cohen kappa: 0.788520, reported as 0.79.

The five primary-verifier abstentions are excluded from Cohen kappa, following
the convention implemented in `verify_answer_papers.py`. Counting the 29 exact
matches over all 40 sampled rows gives 0.725, the `raw_agreement` stored in the
verdict. The CSV field name `openai_index` and the historical helper name
`compute_openai_human_kappa` are legacy names; the run-level verdict identifies
the primary verifier as Gemini. Every frozen accepted row also contains a
legacy `verifier: openai-gpt` value emitted by the then hard-coded output
writer. Raw provider responses are not available, so the exact provider
identity cannot be independently recovered from API evidence. See
`../SINGLEHOP_VERIFIER_PROVENANCE.md`. The second rater is a model, not a human.

Files:

- `agreement.csv`: all 80 pilot verifier decisions;
- `blind_rater_labels.csv`: the 40 blind second-model labels;
- `spotcheck_template.csv`: sampled tasks before the second labels were merged;
- `qa_candidates.jsonl` and `qa_labeled.jsonl`: pilot inputs and accepted rows;
- `rater_rubric_translation.md`: English translation of the second rater's
  instructions;
- `rate_spotcheck.py`: original model-rater driver;
- `recompute_kappa.py`: standard-library-only independent recomputation.

Run from the extracted package root:

    python benchmark/singlehop_verifier_spotcheck/recompute_kappa.py
