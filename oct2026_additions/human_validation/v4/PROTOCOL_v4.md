Released copy: passages outside the scope of this paper are omitted and marked [omitted]; SHA-256 of the full protocol as written: 689b5ad45d355f5994a1e128b1d534fe29bf2116e5105c86129550c68a5a8bf4.

# Human re-scoring of the census, v4: protocol

Written 2026-09-30T10:27Z [omitted], before any v4 workbook was generated or sent. Sections 1-8 are fixed from this point. Any later change is appended under "Deviations" at the end, each with its own timestamp; the text above that line is not edited.

## 1. Purpose

[omitted] The logged LM study (`../llm_census_logged/PROTOCOL.md`) added an excerpt-only condition (condition B) in which a rater sees no author-condensed description, only verbatim text from each benchmark's own paper, README, or generator source code.

v4 asks two or three independent human raters, who are not authors and were not involved in the project, to apply the census step procedure to the same 29 axis judgments using the same verified verbatim excerpts that condition B used. It then compares each human with the census, with each other, and with the logged LM raters.

## 2. Raters

- Two raters are needed and three are preferred. Each gets one workbook with a rater code (R1, R2, R3).
- Eligibility: not an author, not involved in the project in any way, background in NLP, IR, or knowledge graphs, able to read benchmark papers in English.
- Raters rate alone, use no AI assistant or language model in any way, do not search the web, do not discuss items with anyone, and use only the package materials. They attest to each point in the workbook (`Rater_info`), yes or no.
- The coordinator sends each rater only that rater's zip, answers only questions about how to use the workbook, and returns the raters' original attachments unrenamed with the message date (`CHECKLIST_v4.md`). No author proposes, discusses, or adjusts any bin. There is no adjudication round.

## 3. Materials (`build_package_v4.py`, deterministic)

Per rater, in `to_send/rater_k/` and zipped as `to_send/rater_k_v4.zip`: `rater_k_v4.xlsx`, `INSTRUCTIONS_v4.md`, `WORKED_EXAMPLES_v4.md`.

Workbook sheets, in this order:
1. `Read_me`: the short rules.
2. `Rater_info`: rater code (given), field/background, years in NLP/IR, relation to the authors (must be "not an author, not involved in the project"), session 1 start/end, session 2 start/end (optional), total minutes, three yes/no attestations ((a) rated alone, without discussing items with anyone; (b) used no AI assistant or language model at any point; (c) used only the package materials and did not look at the paper's results or any other rater's sheet), signature/initials, comments.
3. `Quiz`: the six v3 qualification items on invented benchmarks (`../v3/QUIZ_KEY_v3.md`), shown with their v3 construction text as the only excerpt, labelled "benchmark documentation". No key or feedback is sent.
4. `Census`: the 29 axis judgments of `../items_29.json` and the two v3 attention items (one clearly bin 0, one clearly bin 4; invented benchmarks, excluded from all statistics), 31 rows.

A Census row shows only: the benchmark name, the gold-unit axis name, a one-line neutral definition of that axis (what the unit is, never how it was produced, no rubric wording), and the verified verbatim excerpt(s) of `../llm_census_logged/excerpts.json` (`items[<id>].excerpts[*]`, every excerpt with verification EXACT, LOOSE, or LOOSE-INTERRUPTED), each labelled with its source kind (benchmark paper, README, or generator source code), version (arXiv or Anthology id, README commit, file MD5 prefix), and PDF page or code line. The benchmark and axis names are those of condition B (the two calibration anchors are "Anchor benchmark L" and "Anchor benchmark U"). No author-condensed description and no item id is shown; rows carry a per-rater random code. The attention items show their v3 description as their only excerpt, labelled "benchmark documentation", exactly as in condition B.

Input columns on Quiz and Census: `source_K_A_T` (drop-down K/A/T), `base_bin_0_4` (drop-down 0-4), `corrections` (free text), `final_bin_0_4` (drop-down 0-4), `deciding_phrase` (quote of the phrase that decided the bin), `confidence_1_3` (drop-down), `minutes` (optional).

Order: the Quiz and Census rows are shuffled independently per rater with `random.Random(seed)`, seed = 2026093000 + k for rater k; the same generator draws the row codes. Seeds, orders, and codes are in `SEALED_build.json`.

Instructions (`INSTRUCTIONS_v4.md`) adapt Section 4 of `../v3/INSTRUCTIONS_v3.md` (notation, Steps 1-6, base-bin table, corrections B3-1 to B3-4) with these changes only: "description" becomes "excerpts"; Part B, stage 1/stage 2, and the quiz-key hand-off are removed; the rules of Section 2 are stated explicitly, including "Do not use AI assistants or language models in any way; do not search the web for this task; do not discuss items with other raters"; the return procedure is stated (the unmodified filled workbook as an email or messenger attachment directly to the coordinator); the expected time is 60-100 minutes. No tie-breaking rule is given (the v3 statement that there is none is kept). Nothing about the census results or the paper's findings is included. The six worked examples of v3 are given unchanged in `WORKED_EXAMPLES_v4.md`.

Known differences from LM condition B, recorded here so they are not read as deviations later: the LM stage-2 prompt contained the quiz key, the human raters do not get it; the human rows add a one-line axis definition and the source version; each LM request held one row, the human sees the whole sheet.

## 4. Integrity controls

- Every sheet is protected; only the input cells and the `Rater_info` value cells are unlocked, and the workbook structure is locked. The password is stored only in `SEALED_build.json`. Excel's sheet password is a weak legacy hash; it prevents accidental edits and makes deliberate ones detectable, it is not a security boundary.
- Document properties are set by the build: creator "coordinator", lastModifiedBy empty, created = modified = the fixed build time. Zip entries carry a fixed timestamp so a rebuild reproduces every file byte for byte.
- `SEALED_build.json` stores, per rater: seed, Quiz and Census order, row codes, SHA-256 of every sent file (workbook, instructions, worked examples, zip), the hash of every locked cell, the list of input cells, and the sent document properties; per item: the SHA-256 of the displayed item text; and the answer keys (quiz key, attention expected bins) and the SHA-256 of `../reference_bins.json`. The census reference bins are never written to any rater file.
- `score_v4.py` reads the returned workbooks from `returned/` (file names unchanged) and checks each against `SEALED_build.json`:
  - hard flags (the workbook is not used): a sheet missing, added, renamed, or reordered; creator or created changed; a Census or Quiz item text hash that differs, an unknown row code, or a changed row order; any locked cell changed or any value outside the input cells; sheet protection or the structure lock removed; a filled workbook whose `modified` time and `app.xml` are identical to the sent file (it was filled by a script, not saved by a spreadsheet application);
  - warnings (reported, followed up by the coordinator, not excluding by themselves): lastModifiedBy empty, application other than Excel or LibreOffice, a changed protection hash, `modified` earlier than `created`, the same lastModifiedBy on two raters' workbooks, and a high share of identical deciding phrases between two raters.

## 5. Exclusion rules (fixed now)

A returned workbook enters the statistics only if all of the following hold:
1. no hard integrity flag;
2. every `Rater_info` value is filled (session 2 and comments optional), the relation is "not an author, not involved in the project", and all three attestations are "yes";
3. quiz: final bin correct on at least 4 of 6 items;
4. attention: not two or more bins away from the expected bin on both attention items;
5. every Census row has a final bin, base bin, source, and confidence.

Excluded workbooks are reported with the reason. A rater excluded under 3-5 may be replaced by a new rater with a fresh code (R4, R5, ...) built by the same script with the next seed; this is recorded as a deviation. No excluded rater is asked to re-rate.

## 6. Outcomes and analysis (`score_v4.py`, output `result_v4.{md,json}`)

All statistics use the 29 census items (the 23 external items are reported alongside where noted); attention and quiz items are excluded.

1. Each eligible human versus the census bins (`../reference_bins.json`): quadratically weighted kappa (QWK), exact agreement, within-one agreement, and agreement on derived (final bin >= 2) versus independent (<= 1); 29 and 23 items.
2. Human-human: pairwise QWK, exact, within-one, pairwise ordinal Krippendorff alpha, and the ordinal alpha over all eligible humans, each with a 95% item-bootstrap CI (10,000 resamples, seed 20260930).
3. Each human versus each logged LM model median, both conditions (`../llm_census_logged/result.json`, `per_model_median`), and versus the census and the primary LM raters A and B (`../../analysis/rater_robustness.py`, `LM_A_B`): QWK, exact, within-one, shown next to the human-human values.
4. Alternative annotator test (Calderon, Reichart and Dror, ACL 2025) for each LM rater of item 3 (logged medians of both conditions, the census, primary LM raters A and B) and for constant-bin controls 0-4, which must fail: leave one human out; alignment score -|bin difference| with one remaining human, -RMSE with two or more; winning indicators with ties counting for both; one-sided paired t-test of H0: mean(W_h - W_f) >= epsilon; Benjamini-Yekutieli correction over the left-out humans (q = 0.05); winning rate omega (pass = omega >= 0.5) and advantage probability rho; the full sweep epsilon = 0, 0.05, ..., 0.30 is reported, and epsilon = 0.15 is the value quoted in text, as in `analysis/rater_validation.py`. With two eligible humans the result is reported descriptively and labelled so (the test needs three or more annotators); with three or more it is reported as the standard test. The census is adjudicated from LM raters A and B, so these three are not independent of each other.
5. Per-benchmark KG-derived headline bin for each eligible human (maximum final bin over the benchmark's KG-derived axes, `KG_AXES` of `analysis/rater_robustness.py`), and whether GRADE, 2WikiMultiHop, MetaQA, and WebQSP are at bin >= 3 and GraphRAG-Bench and STaRK at bin >= 2. With three or more humans the item-wise median is reported as well.

Secondary: agreement on source (K/A/T) and base bin, mean confidence, reported minutes, and per-item bins.

Every outcome is reported whatever its value, together with the quiz, attention, attestation, and integrity results of every returned workbook, including excluded ones. If some but not all returned workbooks are eligible, the statistics over all returned workbooks with a complete Census sheet are reported as a labelled sensitivity analysis, never as the primary result.

## 7. What is not done

No author reads a returned workbook before `score_v4.py` has run on it. No bin is discussed with a rater or changed after return. Synthetic test workbooks are made only outside this folder and never placed in `returned/`.

## 8. Files

`PROTOCOL_v4.md` (this file), `build_package_v4.py`, `INSTRUCTIONS_v4.md` (source, copied into each rater folder), `WORKED_EXAMPLES_v4.md` (generated), `CHECKLIST_v4.md` (coordinator), `SEALED_build.json` (sealed), `to_send/` (what is sent), `returned/` (what comes back, unrenamed), `score_v4.py`, `result_v4.{md,json}`.

---

## Deviations

(none)
