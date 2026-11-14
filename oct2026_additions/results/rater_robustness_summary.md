# Census conclusions under each rater

Raters: 17. External benchmarks with C_K >= 0.75: 4-6. Flagged >= 0.75 by every rater: GRADE, 2WikiMultiHop, MetaQA, WebQSP. Flagged >= 0.5 by every rater: GRADE, STaRK, 2WikiMultiHop, MetaQA, WebQSP. Never flagged >= 0.5: MuSiQue, HotpotQA, MultiHop-RAG, Natural Questions, TriviaQA. Spearman rho (C_K vs published gain medians): 0.75-0.86.

| rater | scope | # C_K>=0.75 | # C_K>=0.5 | causal benchmarks >=0.5 | text-grounded flagged | rho |
|---|---|---:|---:|---|---|---:|
| census (adjudicated LM) | 29 axes | 6 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| primary LM rater A | 29 axes | 6 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| primary LM rater B | 29 axes | 6 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| human: non-author | 29 axes | 6 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.75 |
| LM logged: claude-opus-5-5 | 29 axes | 5 | 5 | 2WikiMultiHop, MetaQA, STaRK | - | 0.86 |
| LM logged: claude-sonnet-5-5 | 29 axes | 6 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| LM logged: gemini-3.8-flash | 29 axes | 6 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| LM logged: claude-haiku-4-5 | 29 axes | 4 | 5 | 2WikiMultiHop, MetaQA, STaRK | - | 0.86 |
| human v4: R1 | 29 axes | 5 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| human v4: R2 | 29 axes | 5 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| human v4: R3 | 29 axes | 5 | 5 | 2WikiMultiHop, MetaQA, STaRK | - | 0.86 |
| human v4: median | 29 axes | 5 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| LM: claude-opus-4-8 | headline axis | 5 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.86 |
| LM: claude-sonnet-4-5 | headline axis | 5 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.80 |
| LM: qwen2.5:14b | headline axis | 4 | 5 | 2WikiMultiHop, MetaQA, STaRK | - | 0.86 |
| LM: llama3.1:8b | headline axis | 5 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.77 |
| LM: mistral-small:24b | headline axis | 5 | 6 | 2WikiMultiHop, MetaQA, GraphRAG-Bench, STaRK | - | 0.80 |

Per-rater KG-derived bins:

| rater | GRADE | STaRK | GraphRAG-Bench | 2WikiMultiHop | MetaQA | WebQSP | MuSiQue | HotpotQA | MultiHop-RAG | Natural Questions | TriviaQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| census (adjudicated LM) | 3 | 3 | 3 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| primary LM rater A | 3 | 3 | 3 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| primary LM rater B | 3 | 3 | 3 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| human: non-author | 4 | 4 | 4 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM logged: claude-opus-5-5 | 3 | 3 | 0 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM logged: claude-sonnet-5-5 | 3 | 3 | 3 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM logged: gemini-3.8-flash | 3 | 3 | 3 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM logged: claude-haiku-4-5 | 3 | 2 | 0 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| human v4: R1 | 3 | 3 | 2 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| human v4: R2 | 3 | 3 | 2 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| human v4: R3 | 3 | 3 | 0 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| human v4: median | 3 | 3 | 2 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM: claude-opus-4-8 | 4 | 3 | 2 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM: claude-sonnet-4-5 | 4 | 3 | 2 | 3 | 3 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM: qwen2.5:14b | 3 | 2 | 1 | 3 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| LM: llama3.1:8b | 3 | 3 | 2 | 3 | 4 | 3 | 0 | 0 | 0 | 0 | 0 |
| LM: mistral-small:24b | 3 | 3 | 2 | 3 | 3 | 4 | 0 | 0 | 0 | 0 | 0 |
