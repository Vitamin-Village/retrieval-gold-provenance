# Provenance rating study (v4): rater instructions

## 1. What you are asked to do

The study asks one question about question-answering benchmarks: **where did a benchmark's gold come from?** In particular, was the gold derived from the knowledge graph (KG) that graph-based retrievers use, or from the answer string, or was it made independently from text by people?

Your workbook `rater_N_v4.xlsx` has four sheets: `Read_me`, `Rater_info`, `Quiz` (6 rows), and `Census` (31 rows). Each Quiz and Census row is one benchmark and one **axis**, that is, one kind of gold unit (a passage, a sentence, an answer, an entity, an edge, a path, or a subgraph). You decide how that gold unit was made, using the procedure in Section 5.

Expected time: **60-100 minutes**, in one sitting or in two sessions.

## 2. Rules

These rules are part of the study. Please follow them exactly.

- **Rate alone.** Do not discuss items with anyone, including the other raters and the coordinator. Do not look at any other rater's workbook, and do not show yours to anyone before you send it.
- **Do not use AI assistants or language models in any way.** This includes ChatGPT, Claude, Gemini, Copilot, AI features built into Excel, a browser, or an editor, and AI translation or summarising tools. Do not use them to read, translate, summarise, check, or fill anything in this package.
- **Do not search the web for this task.** Do not open the benchmark papers, their code, or any other source. Judge each row only from what the workbook shows.
- **Use only the materials in this package:** this file, `WORKED_EXAMPLES_v4.md`, and your workbook. Do not look at the results of the paper this study belongs to.
- Every judgment must be your own. There is no expected number of rows in any bin.
- The coordinator can answer questions about how to use the workbook (for example, which cell to type in). The coordinator cannot answer questions about any item.
- At the end, `Rater_info` asks you to confirm these rules with yes or no. Please answer truthfully. A "no" causes no problem for you; it only tells us not to use the workbook as an independent human rating.

## 3. Materials

- `INSTRUCTIONS_v4.md` (this file).
- `WORKED_EXAMPLES_v4.md`: six worked examples on invented benchmarks, with full answers. Read them before you start.
- `rater_N_v4.xlsx`: your workbook.

Open the workbook in desktop Microsoft Excel or LibreOffice Calc. Do not upload it to an online service (for example Google Sheets or Excel for the web), do not copy its contents into a new file, and do not rename it. The sheets are protected: only the yellow cells can be edited, which keeps the item text unchanged. You can widen columns and enlarge rows. To read a long excerpt, select its cell and read it in the formula bar, or enlarge the row.

## 4. Filling the workbook

Work in this order: `Read_me`, `Rater_info`, `Quiz`, `Census`.

**Rater_info.** Your rater code is already filled in. Fill the other yellow cells: your field or background, years working in NLP or IR, your relation to the authors (choose from the list), the start and end of each session as date and time (for example `2026-10-01 14:05`; session 2 only if you used a second session), the total minutes you spent rating, the three yes/no statements, and your initials. Use the comments row for anything you want to tell us, for example a value you could not correct.

**Quiz** (6 rows, invented benchmarks). Rate them with the procedure in Section 5, exactly like Census rows. There is no feedback; go on to the Census sheet when you are done.

**Census** (31 rows). Each row shows:
- `benchmark`: the benchmark's name;
- `axis` and `axis_definition`: the gold unit you rate, with a one-line definition of what that unit is;
- `verbatim_excerpts_with_source`: one or more passages copied word for word from the benchmark's own paper, README, or generator source code. Each is numbered and labelled with its source: the document kind, its version (arXiv or ACL Anthology id, README commit, or source-file checksum), and the PDF page or code line. PDF line-break hyphens are rejoined and footnote markers are dropped; nothing else is changed.

Input columns (Quiz and Census):

| column | what to enter |
|---|---|
| `source_K_A_T` | K, A, or T (drop-down), Step 2 |
| `base_bin_0_4` | 0-4 (drop-down), Step 3 |
| `corrections` | free text: `none`, or the codes you applied, for example `B3-2` or `B3-1 + B3-2`, Step 4 |
| `final_bin_0_4` | 0-4 (drop-down), Step 5 |
| `deciding_phrase` | quote the few words of the excerpt that decided your final bin; if you were torn, add the alternative bin, for example `alt 3` |
| `confidence_1_3` | 3 = sure, 2 = fairly sure, 1 = torn between two bins (drop-down) |
| `minutes` | optional: about how many minutes the row took |

## 5. The procedure

Rate the gold unit of **that row's axis only**.

Notation:
- **K**: the evaluation knowledge graph or knowledge base, the structure a graph retriever ranks over.
- **A**: the answer surface, the text spans where the answer string appears.
- **T**: human annotation of text, or the source text itself, with neither K nor A deciding the gold.

### Step 1. Find the gold unit of this axis
Ask what exactly is scored as gold on this axis. The way the question text was written matters only if it decided this gold unit. Automatic question generation is not circular in itself.

### Step 2. Decide the source (`source_K_A_T`)
- **K** if the gold unit is an element of K (a node, edge, triple, path, or subgraph), or was produced from K's content: filled from KG triples by a template, read off K by executing a query or traversing it, or located in text by matching a KG value.
- **A** if the gold unit was chosen because it contains the answer string.
- **T** if people annotated it on text, or it is source text chosen without K or A deciding it. Text produced automatically by a language model from source text alone is also T.
- Human involvement does not make gold T by itself: if annotators wrote a query and the gold answer is whatever K returns for it, the source is K.

### Step 3. Base bin (`base_bin_0_4`)

| base bin | K source | A source | T source |
|---|---|---|---|
| **4** | the unit is copied unchanged from K, or read off K by a query or traversal | the answer string alone decides the gold (e.g. every passage containing it is gold) | - |
| **3** | the unit is generated from K's content (template, program, or model), or located in text by matching a KG value | candidates are proposed by another process (e.g. search) and kept or labeled because they contain the answer string (distant supervision) | - |
| **2** | not used as a base bin: a final bin of 2 arises only after corrections (e.g. a base of 3 lowered by one step, see E4 in the worked examples) | same | - |
| **1** | K only proposed candidates; a human made the final choice from the text | the answer string only guided a human's choice | - |
| **0** | - | - | people or source text decided the gold, independent of K and A |

### Step 4. Corrections (`corrections`)
Apply each correction that the excerpts state explicitly. Each lowers the bin by one step; at most two steps in total; never below 0.
- **B3-1, KG augmentation**: links or facts not in the original K were added (e.g. by clustering) before the gold was produced. Lowers a base of 4 to 3; a base of 3 or lower stays.
- **B3-2, validation or refinement of the gold**: humans (or models that check the gold against text) confirmed, filtered, corrected, or removed gold units. One step down.
- **B3-3, output evaluation only**: people or models rated system outputs but did not change the gold. **No change** (record it so we can see you considered it).
- **B3-4, anti-leakage or disconnection filtering**: the benchmark actively removed answer-string shortcuts. One step down, only when the base was raised by an A source.

Write `none` if no correction applies.

### Step 5. Final bin (`final_bin_0_4`)
Final bin = base bin minus the corrections of Step 4.

### Step 6. Evidence and confidence
- `deciding_phrase`: quote the few words of the excerpt that decided your final bin. If you were torn, add the alternative bin, e.g. `alt 3`.
- `confidence_1_3`: 3 = sure, 2 = fairly sure, 1 = torn between two bins.
- There is no tie-breaking rule. Choose the bin that fits best.

## 6. Returning your work

1. Check that every Quiz and Census row has `source_K_A_T`, `base_bin_0_4`, `corrections`, `final_bin_0_4`, `deciding_phrase`, and `confidence_1_3`, and that `Rater_info` is complete.
2. Save the workbook in the same format (.xlsx) under the same name, `rater_N_v4.xlsx`.
3. Send that file, as it is, as an email or messenger attachment **directly to the coordinator**. Do not rename it, convert it (PDF, CSV, screenshots), paste its contents into a message, or upload it to a shared drive or online spreadsheet. Do not send it to other raters.

Thank you for your time.
