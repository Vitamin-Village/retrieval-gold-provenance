# Worked examples (invented benchmarks)

These six examples are invented benchmarks, not part of the quiz or the census. Each gives a construction statement and the answer that the procedure of INSTRUCTIONS_v4.md (Section 5) gives for it, with the reason.

**E1.** Crowd workers write questions about a Wikipedia paragraph and mark the answer span in it.

- source: T; base bin: 0; corrections: none; final bin: **0**
- why: Human annotation of source text; no K, no answer-string selection.

**E2.** Every corpus passage that contains the gold answer string is labeled a gold passage.

- source: A; base bin: 4; corrections: none; final bin: **4**
- why: The answer string alone decides the gold.

**E3.** In a product-catalog KG, a script picks a (product, made_by, company) triple, writes 'Who makes <product>?', and stores the company node as the gold answer.

- source: K; base bin: 4; corrections: none; final bin: **4**
- why: The gold answer is a KG node copied as-is; generating the question text does not change that.

**E4.** Supporting passages are written by a language model from KG triples, then crowd workers check each passage and delete those that misstate the triple.

- source: K; base bin: 3; corrections: B3-2; final bin: **2**
- why: Generated from K (base 3); human validation that removes gold lowers one step (B3-2).

**E5.** In a recipe KG, gold ingredient lists are copied from the KG for each dish, and annotators delete ingredients they judge wrong before release.

- source: K; base bin: 4; corrections: B3-2; final bin: **3**
- why: Copied from K (base 4); human refinement that removes or edits gold lowers one step (B3-2).

**E6.** Human evaluators rate the fluency of system answers; the gold answers were copied from the KG.

- source: K; base bin: 4; corrections: B3-3 (no change); final bin: **4**
- why: Evaluating system outputs does not touch the gold: B3-3, no change.
