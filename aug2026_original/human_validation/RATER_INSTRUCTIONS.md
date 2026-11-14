# Human Rater Instructions — Provenance-Circularity Census Validation (29 items)

> Usage summary: open the rating sheet and write a single integer 0-4 in the `your_bin_0_to_4` cell of each row. Score from the rubric below only. **Never open `reference_bins.json` or the census tables of the paper (Table 1, Table 8) before submitting.** When every cell is filled, score the sheet as in Steps. The ideal rater is a non-author (a colleague).

## Who may rate

- Ideal: an independent rater who is **not an author** of the paper. The paper can then report independent human validation over all 29 judgments.
- Record who rated in the `rated_by` line at the bottom of this file after finishing.

## Blindness protocol (mandatory)

1. Do NOT open `reference_bins.json`.
2. Do NOT consult the paper, its census tables, its appendix, or any census analysis file while rating.
3. Rate ONLY from each row's `construction_description` plus the rubric below.
4. Rate all 29 rows in the order given (the order is randomized and intentionally uncorrelated with the paper's tables).
5. If you genuinely cannot decide between two adjacent bins, choose the LOWER bin (this mirrors the census adjudication rule).

## The rubric (identical to the released rater prompt)

You are scoring how a question-answering benchmark's retrieval GOLD was constructed, on a five-level provenance-circularity scale.
K = the evaluation knowledge graph or base that a graph retriever ranks over.
A = the answer surface, the spans where the answer string appears.

- **bin 0**: human or source-text gold, independent of K and A.
- **bin 1**: human or text gold with KG-guided or answer-guided selection.
- **bin 2**: auto-generated and then human-validated.
- **bin 3**: auto-generated from KG triples or templates, OR evidence retained by answer-string distant supervision.
- **bin 4**: gold copied directly from the evaluation KG, OR selected solely because it contains the answer string.

Corrections (apply after picking a base bin): if the description states KG augmentation beyond the original graph, human refinement of the gold, or anti-leakage filtering, shift the bin ONE step lower per stated mechanism, at most twice, never below bin 0.

## Steps

1. `python make_rating_sheet.py`  (writes the blank sheet `rating_sheet_blank.csv`)
2. Open the sheet, fill every `your_bin_0_to_4` cell with one integer 0-4.
3. Save it as `rating_sheet_external.csv`, then run `python score_external_rater.py --rated-by "..."`. It aborts if any cell is empty and otherwise writes `external_rater_result.json` with exact agreement, off-by-one, quadratic-weighted Cohen kappa, and ordinal Krippendorff alpha against the sealed census bins.
4. Fill in the line below and commit the sheet + result together.

rated_by (external, rating_sheet_external.csv): independent rater, not an author (blind, randomized order)  date: 2026-06-12  result: external_rater_result.json -- scored with `python score_external_rater.py --rated-by "..."` (exact 20/29, quadratic-weighted Cohen kappa 0.908, ordinal Krippendorff alpha 0.879)
