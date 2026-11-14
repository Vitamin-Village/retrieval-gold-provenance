# Provenance-Circularity Scoring Rubric

This file is the released form of the census rubric. The bin definitions and the
rater prompt are reproduced verbatim from the released code (the same prompt
string is in `cross_model_census.py` in this directory) and from the paper's
appendix; the step procedure is the rubric as stated in the paper.

## Rater prompt (verbatim; applied in the census and the cross-model validation)

```
You are scoring how a question-answering benchmark's retrieval GOLD was
constructed, on a five-level provenance-circularity scale. K = the evaluation
knowledge graph or base that a graph retriever ranks over. A = the answer
surface, the spans where the answer string appears.
bin 0: human or source-text gold, independent of K and A.
bin 1: human or text gold with KG-guided or answer-guided selection.
bin 2: auto-generated and then human-validated.
bin 3: auto-generated from KG triples or templates, OR evidence retained by
answer-string distant supervision.
bin 4: gold copied directly from the evaluation KG, OR selected solely because
it contains the answer string.

Benchmark: <name>
How its gold was constructed: <description>

Reply with ONLY one integer from 0 to 4 (the bin). No other text.
```

Normative clarification: the code block preserves the applied prompt verbatim.
Consistent with the paper's definition, bin 2 requires gold derived from K or
A by auto-generation and then human-validated; source-independent
auto-generation is bin 0. The applied prompt omitted that source qualifier.
No adjudicated primary-census row uses bin 2, so this clarification changes no
reported census value or reliability statistic.

## Step procedure (applied per gold-unit axis)

- **Step 0, gate.** The benchmark must have supporting gold that is scored by
  overlap such as hit@k or F1; win-rate and gold-free benchmarks are out of
  scope.
- **Step 1, axes.** Score every available gold-unit axis separately and report
  all axes: passage or sentence; answer; and entity, edge, or path.
- **Step 2, bins.** For each axis, classify how the gold was produced by
  derivation from the KG K or from the answer surface A:
  - bin 4, C = 1.0: gold copied directly from the evaluation KG K, or selected
    solely because it contains the answer string;
  - bin 3, C = 0.75: auto-generated from KG triples or templates, or evidence
    retained by answer-string distant supervision;
  - bin 2, C = 0.50: gold derived from K or A by auto-generation and then
    human-validated;
  - bin 1, C = 0.25: human or text gold with KG-guided or answer-guided
    selection;
  - bin 0, C = 0.0: human or source-text gold, independent of K and A.
- **Step 3, corrections.** KG augmentation, human refinement, and anti-leakage
  filtering each shift the bin one step lower; these shifts are applied at most
  twice and never below bin 0.
- **Step 4, evidence.** Attach a verbatim construction quote to each bin
  assignment.
- **Steps 5-6, adjudication and agreement.** Resolve one-bin disagreements to
  the lower bin, and adjudicate larger disagreements with a third rater; report
  Krippendorff alpha and quadratic-weighted Cohen kappa.

## Application

Two primary LM raters independently applied the rubric to each benchmark's
construction section, blind to each other's assignments. One-bin disagreements
were resolved to the lower bin; no disagreement reached the two-bin threshold
that invokes a third rater. The complete per-axis records are
`rater_A_assignments.md` and `rater_A_assignments_expansion.md` (rater A,
including the verbatim quote behind each assignment), the six broadening
judgments with both raters' bins in `quotes_remaining_benchmarks.md`, and the
full (rater A, rater B) bin pairs for all 29 judgments inlined as `ROWS` in
`reliability_kappa_ci.py`. The adjudicated per-axis values are in
`per_axis_assignments.json`.
