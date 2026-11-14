| # | construction description | source | base | corrections | final | why |
|---|---|---|---|---|---|---|
| E1 | Crowd workers write questions about a Wikipedia paragraph and mark the answer span in it. | T | 0 | none | 0 | Human annotation of source text; no K, no answer-string selection. |
| E2 | Every corpus passage that contains the gold answer string is labeled a gold passage. | A | 4 | none | 4 | The answer string alone decides the gold. |
| E3 | In a product-catalog KG, a script picks a (product, made_by, company) triple, writes 'Who makes <product>?', and stores the company node as the gold answer. | K | 4 | none | 4 | The gold answer is a KG node copied as-is; generating the question text does not change that. |
| E4 | Supporting passages are written by a language model from KG triples, then crowd workers check each passage and delete those that misstate the triple. | K | 3 | B3-2 | 2 | Generated from K (base 3); human validation that removes gold lowers one step (B3-2). |
| E5 | In a recipe KG, gold ingredient lists are copied from the KG for each dish, and annotators delete ingredients they judge wrong before release. | K | 4 | B3-2 | 3 | Copied from K (base 4); human refinement that removes or edits gold lowers one step (B3-2). |
| E6 | Human evaluators rate the fluency of system answers; the gold answers were copied from the KG. | K | 4 | B3-3 (no change) | 4 | Evaluating system outputs does not touch the gold: B3-3, no change. |
