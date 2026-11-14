# Quiz key (send to a rater only AFTER their stage-1 quiz sheet is returned)

**Q1** (answer): source K, base bin 4, corrections none, final bin **4**. The answer is read off K by executing a query. Human-written parses do not make it independent; the humans wrote the query, K produced the answer.

**Q2** (supporting sentence): source K, base bin 3, corrections none, final bin **3**. A KG value is projected onto text by a matching heuristic: generated from K, not copied (the sentence is text), so bin 3.

**Q3** (evidence document): source A, base bin 3, corrections none, final bin **3**. Distant supervision: search proposes candidates, answer-string matching keeps them. Not 'answer string alone' (that would be bin 4).

**Q4** (reasoning path): source K, base bin 4, corrections B3-1, final bin **3**. Paths are copied from the graph (base 4), but the graph was augmented beyond K (B3-1 lowers 4 to 3).

**Q5** (answer): source T, base bin 0, corrections none, final bin **0**. Automatic generation from source text is provenance-independent. A review of an independent item cannot lower bin 0.

**Q6** (gold passage): source T, base bin 1, corrections none, final bin **1**. Human text gold whose candidates were guided by the KG: bin 1.
