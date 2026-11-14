Released copy: passages outside the scope of this paper are omitted and marked [omitted]; SHA-256 of the full protocol as written: 04ebe89dc7c5044270dc65461ed47f07cfe5cfbd377125fa3d6916de79a5744e.

# Judge spot-check v4 (GraphRAG-Bench LM judge): protocol

Written 2026-10-01T12:26Z [omitted], before any workbook of this package was generated or sent. Sections 1-8 are fixed from this point. Any later change is appended under "Deviations" at the end, each with its own timestamp; the text above that line is not edited.

## 1. Purpose

The end-to-end QA answers on GraphRAG-Bench are graded by an LM judge (mistral-small:24b). In `../judge_spotcheck/` (v1), one non-author human labelled 100 (question, reference answer, candidate answer) triples drawn from the judge's outputs. `../judge_spotcheck_v3/` fixed item ids for the same 100 triples (S001-S100, v1 sheet order, `sealed_key_v3.json`), explicit rater rules (`GUIDE_v3.md`), [omitted]. Three Claude models labelled the same items under `GUIDE_v3.md` (`../llm_spotcheck_v3/`); those are LM labels, not human labels.

v4 asks three independent human raters, who are not authors and were not involved in the project, to label the same 100 items under the v3 rules, with the integrity design of the census package `../v4/` (`../v4/PROTOCOL_v4.md`, Section 4). The v1 rater is not one of the three; the v1 labels are reported separately as a pilot rater. The v3 workbooks (`../judge_spotcheck_v3/sheets/`) are not used; v4 builds new ones.

## 2. Raters

- Three raters, codes R1, R2, R3, one workbook each.
- Eligibility: not an author, not involved in the project in any way, has not seen the judge's verdicts, the v1 labels, the Claude labels, or the paper's results; able to read English. No medical background is needed or used: the rules compare two texts.
- Raters rate alone, use no AI assistant or language model in any way, do not search the web, do not use outside medical knowledge, do not discuss items with anyone, and use only the package materials. They attest to the three v4 statements in the workbook (`Rater_info`), yes or no.
- The coordinator sends each rater only that rater's zip, answers only questions about how to use the workbook, and returns the raters' original attachments unrenamed with the message date and channel (`CHECKLIST_judge_v4.md`). No author proposes, discusses, or adjusts any label. There is no adjudication round.

## 3. Materials (`build_package_judge_v4.py`, deterministic)

Per rater, in `to_send/rater_k/` and zipped as `to_send/rater_k_judge_v4.zip`: `rater_k_judge_v4.xlsx` and `GUIDE_v4.md`.

Items: the 100 items S001-S100 of `../judge_spotcheck_v3/sealed_key_v3.json`. The question, reference answer, and candidate answer are taken verbatim from `../judge_spotcheck/judge_spotcheck_sheet_blank.csv` (row i is item S{i:03d}). The build checks each id's v1 key against `sealed_key_v3.json` and each text against the three v3 workbooks the Claude models labelled. Two attention items are added (invented, excluded from all statistics):

| id | question | reference | candidate | expected |
|---|---|---|---|---|
| CHECK_CORRECT | Which organ is affected first in Halvren disease? | Halvren disease affects the liver first; the spleen can be involved later in the course. | Halvren disease affects the liver first. | CORRECT (restates the key fact) |
| CHECK_INCORRECT | Which enzyme is deficient in patients with Ostrik syndrome? | Patients with Ostrik syndrome have a deficiency of the enzyme glucokinase. | Ostrik syndrome is caused by a deficiency of the enzyme lactase. | INCORRECT (names a different entity) |

Workbook sheets, in this order:
1. `Read_me`: the short rules.
2. `Rater_info`: the rows of `../v4/build_package_v4.py` (`INFO_ROWS`) with the same fields: rater code (given), field/background, years in NLP/IR, relation to the authors (must be "not an author, not involved in the project"), session 1 start/end, session 2 start/end (optional), total minutes, the three yes/no attestations of v4 ((a) rated alone, without discussing items with anyone; (b) used no AI assistant or language model at any point; (c) used only the package materials and did not look at the paper's results or any other rater's sheet), signature/initials, comments (optional).
3. `Items`: 102 rows, the 100 items and the two attention items. A row shows only a running number, a per-rater random row code, the question, the reference answer, and the candidate answer (model output, verbatim). No item id, v1 key, judge verdict, reader model, retrieval system, or condition is shown. Input columns: `verdict` (drop-down CORRECT / INCORRECT / UNSURE) and `note` (free text, optional).

Order: the 102 rows are shuffled per rater with `random.Random(seed)`, seed = 2026100100 + k for rater k; the same generator then draws the 4-character row codes. Seeds, orders, and codes are in `SEALED_judge_build.json`.

Guide (`GUIDE_v4.md`): rules 1-6 and the invented example table of `../judge_spotcheck_v3/GUIDE_v3.md`, copied verbatim; explicit conduct rules (rate alone; no AI assistant or language model in any way, including translators built on them; no web search and no outside medical knowledge; no discussion of items with anyone; only the guide and the workbook); how to fill `Rater_info`; the return procedure (the unmodified filled workbook, unrenamed, as an attachment directly to the coordinator); expected time 40-60 minutes. The Korean summary of GUIDE_v3 is replaced by a fixed Korean summary that adds the conduct and return rules. Nothing about the judge's verdicts, the sample composition, the v1 rater, the Claude labels, or the paper's findings appears in any rater-facing file. The build checks the workbook and the guide for item ids, v1 keys, the password, and a list of forbidden words.

Known differences from the earlier labels, recorded here so they are not read as deviations later: the v1 rater worked from a CSV sheet with the shorter rule of `../judge_spotcheck/README.md`; the Claude models had `GUIDE_v3.md` as the system prompt and one item per request; the v4 raters see the whole sheet in their own random order with two attention items, and their guide adds conduct rules and the new Korean summary.

## 4. Integrity controls

The controls of `../v4/PROTOCOL_v4.md`, Section 4, applied to this workbook:

- Every sheet is protected; only the Items input cells (`verdict`, `note`) and the `Rater_info` value cells are unlocked, and the workbook structure is locked. The password is stored only in `SEALED_judge_build.json`. Excel's sheet password is a weak legacy hash; it prevents accidental edits and makes deliberate ones detectable, it is not a security boundary.
- Document properties are set by the build: creator "coordinator", lastModifiedBy empty, created = modified = the fixed build time 2026-10-01T12:30:00Z. Zip entries carry a fixed timestamp, so a rebuild reproduces every file byte for byte (`build_package_judge_v4.py --check` rebuilds in memory and compares, writing nothing).
- `SEALED_judge_build.json` stores, per rater: seed, item order, row codes, SHA-256 of every sent file (workbook, guide, zip), the hash of every locked cell, the list of input cells, and the sent document properties; per item: the SHA-256 of the displayed item text; the attention items' expected verdicts; and the SHA-256 of `../judge_spotcheck_v3/sealed_key_v3.json` and of the other inputs. The judge verdicts are never written to any rater file or to `SEALED_judge_build.json`; the scorer reads them from `sealed_key_v3.json` after checking its hash.
- `score_judge_v4.py` reads the returned workbooks from `returned/` (file names unchanged) and checks each against `SEALED_judge_build.json`:
  - hard flags (the workbook is not used): a sheet missing, added, renamed, or reordered; creator or created changed; an item text hash that differs, an unknown row code, or a changed row order; any locked cell changed or any value outside the input cells; sheet protection or the structure lock removed; a filled workbook whose `modified` time and `app.xml` are identical to the sent file (it was filled by a script, not saved by a spreadsheet application); two raters returning byte-identical files;
  - warnings (reported, followed up by the coordinator, not excluding by themselves): lastModifiedBy empty, application other than Excel or LibreOffice, a changed protection hash, `modified` earlier than `created`, the same lastModifiedBy on two raters' workbooks, the same user-profile folder in the save path that Excel records in the workbook on two raters' workbooks, identical verdicts on all 100 items for two raters, identical non-empty notes on half or more of the items where both raters wrote a note (at least three such items), and identical signatures.
  - The script check recognises files written by a library such as openpyxl. It cannot recognise a spreadsheet application driven by automation; the attestations and the cross-workbook warnings address that case.
- No user name, signature, or file-system path taken from a returned file is written to `result_judge_v4.md`; the report states only whether such values are equal across workbooks (as `../v4/independence_checks_v4.py` does). `result_judge_v4.json` stores hashes in their place.

## 5. Exclusion rules (fixed now)

A returned workbook enters the statistics only if all of the following hold:
1. no hard integrity flag;
2. every `Rater_info` value is filled (session 2 and comments optional), the rater code is the one sent, the relation is "not an author, not involved in the project", and all three attestations are "yes";
3. attention: the verdict is not wrong on both attention items (wrong = anything other than the expected verdict, including UNSURE or blank);
4. every one of the 100 items has a verdict CORRECT, INCORRECT, or UNSURE (UNSURE counts as labelled).

Excluded workbooks are reported with the reason; a workbook returned unfilled is reported as not filled. A rater excluded under 2-4 may be replaced by a new rater with a fresh code (R4, R5, ...) built by the same script with the next seed; this is recorded as a deviation. No excluded rater is asked to re-rate.

## 6. Outcomes and analysis (`score_judge_v4.py`, output `result_judge_v4.{md,json}`)

All statistics use the 100 items; the attention items are excluded. Verdicts are read with surrounding spaces removed and in upper case. Judge verdicts, the reader model of each item, and the v1 labels come from `../judge_spotcheck_v3/sealed_key_v3.json`; p, the judge's CORRECT rate over the sampling pool, comes from `../judge_spotcheck_v3/pool_stats.json` (0.511), [omitted]. Confidence intervals are 95% percentile intervals from an item bootstrap: 10,000 resamples of the 100 items with replacement, seed 20261001.

1. Each eligible rater against the judge, UNSURE excluded per rater (count reported): agreement and Cohen's kappa with its bootstrap CI; secondary, the judge's precision and recall against that rater, as in `score_spotcheck_v3.py`.
2. Among the raters: Fleiss' kappa over the three raters, computed as in `score_spotcheck_v3.py` (per item over its non-UNSURE labels; items with fewer than two such labels are dropped), and pairwise Cohen's kappa and agreement over the items both raters labelled CORRECT or INCORRECT, each with its bootstrap CI.
3. The three-rater majority against the judge. The majority is taken over the non-UNSURE labels; a tie, or no label, is UNSURE, and UNSURE majority rows are excluded and counted. Reported: agreement, Cohen's kappa with its bootstrap CI, the judge's precision a = P(majority CORRECT | judge CORRECT), recall on the sample TP / (TP + FN), and recall weighted to the pool, recall_w = p a / (p a + (1 - p) b) with b = P(majority CORRECT | judge INCORRECT) [omitted].
4. Per reader: outcomes 1 and 3 restricted to the items whose candidate came from Qwen 2.5 14B (80 items) and from Llama 3.1 8B (20 items).
5. Comparisons, not primary: each rater and the majority against the v1 human, against each Claude model of `../llm_spotcheck_v3/raw.jsonl` (claude-opus-5, claude-sonnet-5, claude-haiku-4-5), and against the Claude majority (same majority rule): n, agreement, Cohen's kappa, over items where both labels are CORRECT or INCORRECT. The v1 rater's and each Claude model's own agreement and kappa with the judge are shown next to these, labelled "pilot human" and "LM labels"; the scorer checks that they reproduce `../judge_spotcheck/spotcheck_result.json` and `../llm_spotcheck_v3/result_v3.json`.

Secondary: label counts per rater, number of notes, reported total minutes and minutes from the session times, and a per-item table of all labels.

With fewer than three eligible raters, outcome 2 is computed over the eligible raters; with two, the "majority" is the label on which both agree (a disagreement is UNSURE) and is labelled so; with one, outcomes 1, 4, and 5 are reported for that rater only.

Every outcome is reported whatever its value, together with the attention, attestation, and integrity results of every returned workbook, including excluded ones. If some but not all returned workbooks with all 100 items labelled are eligible, the statistics over all of them are reported as a labelled sensitivity analysis, never as the primary result.

## 7. What is not done

No author reads a returned workbook before `score_judge_v4.py` has run on it. No label is discussed with a rater or changed after return. Synthetic test workbooks are made only outside this folder and never placed in `returned/`. The scorer's `--allow-script-fill` option exists only for those synthetic tests: it turns the script-fill hard flag into a warning, marks the whole report as a test, and refuses to write into this folder.

## 8. Files

`PROTOCOL_judge_v4.md` (this file), `build_package_judge_v4.py`, `GUIDE_v4.md` (source, copied into each rater folder), `CHECKLIST_judge_v4.md` (coordinator), `SEALED_judge_build.json` (sealed), `to_send/` (what is sent), `returned/` (what comes back, unrenamed), `score_judge_v4.py`, `result_judge_v4.{md,json}`.

---

## Deviations

1. 2026-10-01T12:46Z [omitted], after the build and before any workbook was sent; the sent files and `SEALED_judge_build.json` are unchanged. Section 4, hard flags: a filled workbook is also flagged as filled by a script when its `app.xml` names "Microsoft Excel" but its AppVersion is not of the form Excel writes (two digits, a dot, four digits, e.g. 16.0300) and it has no DocSecurity field. Reason: the check of Section 4 compares `app.xml` with the sent file, and openpyxl writes its own version into AppVersion, so a fill with another openpyxl version would not match the sent `app.xml` and would escape the check. Found in a code review of `score_judge_v4.py` before sending. In the same review, three reporting details were brought in line with the text above, not changed: unreadable session times are reported but do not exclude (Section 5 does not list them); the per-reader outcomes carry their bootstrap CIs (Section 6.4 restricts outcomes 1 and 3, which include CIs); file names of other files in `returned/` are counted, not printed, and the JSON uses keyed hashes (Section 4).

2. 2026-10-02T13:45Z, after the first scoring (2026-10-02T10:56Z). The workbook first returned for R3 (SHA-256 beginning 0d9379df516a314b) gave the same verdict as R2 on all 100 items, including the one UNSURE item, whereas R1 and R2 give the same verdict on 89; it cannot be treated as a rating independent of R2 and is not used. The coordinator then gave the unfilled R3 workbook, as sent, to a different rater, who returned it on 2026-10-02 (SHA-256 beginning fb9224d3d0a50bc1). This replacement reuses the R3 code and item order instead of a fresh code, and it was decided after the first scoring; both are deviations. The replacement workbook passed every check of Sections 4 and 5 and is R3 in every statistic of `result_judge_v4.md`.

3. 2026-10-09, correction to item 2. The coordinator had asked four raters; the fourth, a backup, received a separate blank copy of the R3 workbook and completed it on 2026-10-02 (session times 16:11-17:04 in the workbook), before the first scoring. After the first scoring showed the duplicate, the coordinator chose this workbook as R3. Item 2's "then gave ... to a different rater" misstates this order.
