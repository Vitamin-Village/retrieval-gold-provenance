# Published Graph-over-Lexical Retrieval Gains for Provenance-Circularity Scatter

Date: 2026-06-11. Purpose: observational scatter of census circularity score (C_KG, KG-derived axis) vs published graph-system advantage over a lexical baseline, same paper / same benchmark / retrieval or retrieval-proxy metric.

Inclusion rule: graph/KG-based retrieval system and a lexical (sparse) baseline evaluated on the SAME benchmark in the SAME paper. Where no sparse baseline exists in any graph paper, the nearest published pair is recorded and flagged. All rows were checked against the official paper PDFs on 2026-07-16. Per-pair deltas are rounded to one decimal percentage point before the displayed benchmark medians are computed.

## Master table

| # | Benchmark | C_KG | Graph system | Lexical baseline | Metric | Graph | Lexical | Delta (pp) | Paper (source table) |
|---|-----------|------|--------------|------------------|--------|-------|---------|------------|----------------------|
| 1 | MuSiQue | 0.00 | HippoRAG (ColBERTv2) | BM25 | recall@5 | 51.9 | 41.2 | +10.7 | Gutierrez et al. 2024, arXiv:2405.14831, Table 2 |
| 2 | MuSiQue | 0.00 | HippoRAG (ColBERTv2) | BM25 | recall@2 | 40.9 | 32.3 | +8.6 | Gutierrez et al. 2024, arXiv:2405.14831, Table 2 |
| 3 | MuSiQue | 0.00 | HippoRAG 2 | BM25 | recall@5 | 74.7 | 43.5 | +31.2 | Gutierrez et al. 2025, arXiv:2502.14802, Table 3 [verbatim] |
| 4 | MuSiQue | 0.00 | HopRAG (GPT-4o) | BM25 | EM (retrieval-proxy, same reader) | 42.20 | 13.80 | +28.4 | Liu et al. 2025, arXiv:2502.12442, Table 2 |
| 5 | HotpotQA | 0.00 | HippoRAG (ColBERTv2) | BM25 | recall@5 | 77.7 | 72.2 | +5.5 | Gutierrez et al. 2024, arXiv:2405.14831, Table 2 |
| 6 | HotpotQA | 0.00 | HippoRAG (ColBERTv2) | BM25 | recall@2 | 60.5 | 55.4 | +5.1 | Gutierrez et al. 2024, arXiv:2405.14831, Table 2 |
| 7 | HotpotQA | 0.00 | HippoRAG 2 | BM25 | recall@5 | 96.3 | 74.8 | +21.5 | Gutierrez et al. 2025, arXiv:2502.14802, Table 3 [verbatim] |
| 8 | MultiHop-RAG | 0.00 | KG-CQR + BM25 | BM25 | mAP | 0.250 | 0.241 | +0.9 | Bui et al. 2025, arXiv:2508.20417, Table 1 |
| 9 | MultiHop-RAG | 0.00 | KG-CQR + BM25 | BM25 | recall@10 | 0.372 | 0.353 | +1.9 | Bui et al. 2025, arXiv:2508.20417, Table 1 |
| 10 | MultiHop-RAG | 0.00 | KG-CQR + BM25 | BM25 | recall@25 | 0.532 | 0.486 | +4.6 | Bui et al. 2025, arXiv:2508.20417, Table 1 |
| 11 | Natural Questions | 0.00 | HippoRAG (reproduced) | BM25 | recall@5 | 44.4 | 56.1 | -11.7 | Gutierrez et al. 2025, arXiv:2502.14802, Table 3 [verbatim] |
| 12 | Natural Questions | 0.00 | HippoRAG 2 | BM25 | recall@5 | 78.0 | 56.1 | +21.9 | Gutierrez et al. 2025, arXiv:2502.14802, Table 3 [verbatim] |
| 13 | Natural Questions | 0.00 | GraphRetriever (ParReader++) | Text-match (TF-IDF article retrieval + BM25 passage ranking; ParReader++) | test EM (retrieval-proxy, same reader) | 33.5 | 28.7 | +4.8 | Min et al., arXiv:1911.03868v2 (revised 13 Apr 2020), Table 2 |
| 14 | TriviaQA | 0.00 (KG axis) | GraphRetriever (ParReader++) | Text-match (TF-IDF article retrieval + BM25 passage ranking; ParReader++) | test EM (retrieval-proxy, same reader) | 55.0 | 54.0 | +1.0 | Min et al., arXiv:1911.03868v2 (revised 13 Apr 2020), Table 2 |
| 15 | GRADE | 0.75 | — | — | — | — | — | no published graph-vs-lexical pair found | Lee et al. 2025, arXiv:2508.16994 (single embedding retriever only) |
| 16 | STaRK-MAG | 0.75 | MoR | BM25 | Hit@1 | 58.19 | 25.85 | +32.3 | Lei et al. 2025, arXiv:2502.20317, Table 1 |
| 17 | STaRK-Amazon | 0.75 | MoR | BM25 | Hit@1 | 52.19 | 44.94 | +7.3 | Lei et al. 2025, arXiv:2502.20317, Table 1 |
| 18 | STaRK-Prime | 0.75 | MoR | BM25 | Hit@1 | 36.41 | 12.75 | +23.7 | Lei et al. 2025, arXiv:2502.20317, Table 1 |
| 19 | STaRK-MAG | 0.75 | QAGNN (pure structural) | BM25 | Hit@1 | 12.88 | 25.85 | -13.0 | Lei et al. 2025, arXiv:2502.20317, Table 1 |
| 20 | GraphRAG-Bench (Novel) | 0.75 | HippoRAG | Vanilla RAG w/ rerank (dense; no BM25 in paper) | Evidence Recall, Complex Reasoning | 87.91 | 64.47 | +23.4 (flagged: dense baseline) | Xiang et al. 2026, arXiv:2506.05690v3, Table 4 |
| 21 | GraphRAG-Bench (Novel) | 0.75 | LightRAG | Vanilla RAG w/ rerank (dense; no BM25 in paper) | Evidence Recall, Complex Reasoning | 85.52 | 64.47 | +21.1 (flagged: dense baseline) | Xiang et al. 2026, arXiv:2506.05690v3, Table 4 |
| 22 | 2WikiMultiHopQA | 1.00 | HippoRAG (ColBERTv2) | BM25 | recall@5 | 89.1 | 61.9 | +27.2 | Gutierrez et al. 2024, arXiv:2405.14831, Table 2 |
| 23 | 2WikiMultiHopQA | 1.00 | HippoRAG (ColBERTv2) | BM25 | recall@2 | 70.7 | 51.8 | +18.9 | Gutierrez et al. 2024, arXiv:2405.14831, Table 2 |
| 24 | 2WikiMultiHopQA | 1.00 | HippoRAG 2 | BM25 | recall@5 | 90.4 | 65.3 | +25.1 | Gutierrez et al. 2025, arXiv:2502.14802, Table 3 [verbatim] |
| 25 | MetaQA 2-hop | 1.00 | PullNet (KB) | PullNet (text-only retrieval) | Hits@1 | 99.9 | 81.0 | +18.9 | Sun et al. 2019, arXiv:1904.09537, Table 3 of the EMNLP-IJCNLP 2019 version (Table 2 in arXiv v1) [verbatim] |
| 26 | MetaQA 3-hop | 1.00 | PullNet (KB) | PullNet (text-only retrieval) | Hits@1 | 91.4 | 78.2 | +13.2 | Sun et al. 2019, arXiv:1904.09537, Table 3 of the EMNLP-IJCNLP 2019 version (Table 2 in arXiv v1) [verbatim] |
| 27 | MetaQA 2-hop | 1.00 | GraftNet (KB) | GraftNet (text-only retrieval) | Hits@1 | 94.8 | 36.2 | +58.6 | Sun et al. 2019, arXiv:1904.09537, Table 3 of the EMNLP-IJCNLP 2019 version (Table 2 in arXiv v1) [verbatim, GraftNet rows reproduced therein] |
| 28 | MetaQA 3-hop | 1.00 | GraftNet (KB) | GraftNet (text-only retrieval) | Hits@1 | 77.7 | 40.2 | +37.5 | Sun et al. 2019, arXiv:1904.09537, Table 3 of the EMNLP-IJCNLP 2019 version (Table 2 in arXiv v1) [verbatim] |
| 29 | WebQSP | 1.00 | PullNet (KB) | PullNet (text-only retrieval) | Hits@1 | 68.1 | 24.8 | +43.3 | Sun et al. 2019, arXiv:1904.09537, Table 4 of the EMNLP-IJCNLP 2019 version (Table 5 in arXiv v1) [verbatim] |
| 30 | WebQSP | 1.00 | GraftNet (KB) | GraftNet (text-only retrieval) | Hits@1 | 66.4 | 24.9 | +41.5 | Sun et al. 2019, arXiv:1904.09537, Table 4 of the EMNLP-IJCNLP 2019 version (Table 5 in arXiv v1) [verbatim] |
| 31 | WebQSP | 1.00 | KV-Mem (KB) | KV-Mem (text-only retrieval) | Hits@1 | 46.7 | 23.2 | +23.5 | Sun et al. 2019, arXiv:1904.09537, Table 4 of the EMNLP-IJCNLP 2019 version (Table 5 in arXiv v1) [verbatim] |

## Per-benchmark source notes

### MuSiQue (C_KG = 0.00)
- HippoRAG (Gutierrez et al., NeurIPS 2024, arXiv:2405.14831), Table 2 "single-step retrieval": BM25 R@2 32.3 / R@5 41.2; Contriever 34.8/46.6; ColBERTv2 37.9/49.2; HippoRAG(Contriever) 41.0/52.1; HippoRAG(ColBERTv2) 40.9/51.9. Multi-step (Table 3): IRCoT+HippoRAG(ColBERTv2) R@2 45.3 / R@5 57.6.
- HippoRAG 2 (arXiv:2502.14802), Table 3 passage recall@5: BM25 43.5, HippoRAG(repro) 53.2, HippoRAG 2 74.7. Caption notes structure-augmented methods reproduced with same LLM/retriever.
- HopRAG (arXiv:2502.12442, Findings of ACL 2025), Tables 1-2: EM/F1 under identical reader; BM25 GPT-4o EM 13.80 vs HopRAG 42.20. EM is a retrieval-proxy here (same generator, retrieval varies).

### HotpotQA (C_KG = 0.00)
- Same three sources as MuSiQue. HippoRAG Table 2: BM25 R@2 55.4 / R@5 72.2; HippoRAG(ColBERTv2) 60.5/77.7 (note: HippoRAG with Contriever 59.0/76.2 barely beats plain ColBERTv2 64.7/79.3 — graph advantage over the strongest lexical-ish baseline is negative on HotpotQA in HippoRAG v1; vs BM25 it is +5.5 R@5).
- HippoRAG 2 Table 3: BM25 74.8 vs HippoRAG 2 96.3; HippoRAG(repro) 77.3 (+2.5 only).
- HopRAG Table 2 (GPT-4o): BM25 EM 41.20 vs HopRAG 62.00.

### MultiHop-RAG (C_KG = 0.00)
- Original benchmark paper (Tang & Yang 2024, arXiv:2401.15391) contains NO graph system and no BM25 result; it evaluates embedding models with MRR@10, MAP@10, Hits@10 and Hits@4.
- KG-CQR (arXiv:2508.20417), Table 1, evaluates on MultiHop-RAG with BM25 as one of its own baselines (mAP 0.241, R@5 0.261, R@10 0.353, R@25 0.486) and adds KG-derived contextual query enrichment: KG-CQR+BM25 mAP 0.250, R@5 0.267, R@10 0.372, R@25 0.532. Small graph gain (+0.9 mAP, +4.6pp R@25).
- No other graph-RAG paper found reporting BM25-paired retrieval metrics on MultiHop-RAG (HopRAG, AutoSchemaKG, T2RAG, Practical-GraphRAG checked — none evaluate MultiHop-RAG retrieval vs BM25).

### Natural Questions (C_KG = 0.00)
- HippoRAG 2 (arXiv:2502.14802), Table 3 [verbatim]: BM25 56.1; HippoRAG(repro) 44.4 (graph UNDERPERFORMS lexical by 11.7pp); HippoRAG 2 78.0 (+21.9; note HippoRAG 2's dense backbone NV-Embed-v2 alone reaches 75.4, so graph-specific increment over its own dense retriever is +2.6).
- GraphRetriever (arXiv:1911.03868v2, revised 13 Apr 2020), Table 2, same reader ParReader++: text-match test EM 28.7 vs GraphRetriever 33.5 (+4.8). The text-match pipeline uses TF-IDF article retrieval followed by BM25 passage ranking. GraphReader variant: 34.1-34.5 test. Version 2 is pinned because version 1 reports different values.

### TriviaQA (C_KG = 0.00 on KG axis; 0.75 is answer-surface axis)
- GraphRetriever (arXiv:1911.03868v2, revised 13 Apr 2020), Table 2, same reader ParReader++: text-match 54.0 test EM vs GraphRetriever 55.0 (+1.0). The text-match pipeline uses TF-IDF article retrieval followed by BM25 passage ranking. Smallest gain of its three datasets — consistent with TriviaQA's non-KG provenance.
- KG-FiD (arXiv:2110.04330) reports NQ/TriviaQA but has NO BM25/sparse baseline (FiD-large 68.7 vs KG-FiD-large 69.8 TQA EM; DPR-pipeline comparison only). Excluded.
- No modern graph-RAG system (HippoRAG/LightRAG/GraphRAG families) reports TriviaQA retrieval vs BM25 in the same paper.

### GRADE (C_KG = 0.75)
- GRADE paper (Lee, Kwon, Jin — EMNLP 2025 Findings, arXiv:2508.16994) evaluates a SINGLE retriever (OpenAI text-embedding-3-small, top-10 chunks) across GPT-4o/4o-mini/o1-mini; no BM25, no graph retriever, no retriever comparison at all. Datapoint: "no published graph-vs-lexical pair found." No external paper evaluating a graph system on GRADE was found (benchmark too recent/niche).

### STaRK (C_KG = 0.75)
- STaRK itself (Wu et al., NeurIPS 2024 D&B, arXiv:2404.13207, Table 6): BM25 Hit@1 44.94 (Amazon) / 25.85 (MAG) / 12.75 (Prime); no graph-traversal method in the original baseline set (BM25, ada-002, GritLM-7b, GPT-4 reranker).
- MoR (Lei et al. 2025, arXiv:2502.20317), Table 1, reuses identical BM25 numbers (cross-check passed: 44.94/25.85/12.75 exact match with STaRK paper) and adds structural baselines + MoR: MoR Hit@1 52.19/58.19/36.41; pure-structural QAGNN 26.56/12.88/8.85 (LOSES to BM25 on all three). MRR: BM25 55.30/34.91/19.84 vs MoR 62.24/67.14/46.92.
- Interpretive note: largest MoR gain is on MAG (citation-graph-derived, most KG-like; +32.3pp Hit@1), smallest on Amazon (review text; +7.3pp) — internally consistent with the circularity hypothesis even within one benchmark family.

### GraphRAG-Bench (C_KG = 0.75)
- GraphRAG-Bench (arXiv:2506.05690v3), Table 4 (retrieval): baseline is Vanilla RAG with dense retriever (+rerank), NOT BM25 — flagged as dense-baseline pair. Evidence Recall, Complex Reasoning task, Novel domain: RAG w/ rerank 64.47 vs HippoRAG 87.91 (+23.4), LightRAG 85.52 (+21.1), MS-GraphRAG 73.03 (+8.6). On simple Fact Retrieval the ordering flips: RAG w/ rerank 83.21 vs HippoRAG 80.44, MS-GraphRAG 61.04 (graph methods lose).
- Generation accuracy (Table 3, Novel): vanilla RAG 60.92 (fact) / 42.93 (complex) vs HippoRAG2 60.14 / 53.38. Paper's own finding: "Basic RAG matches GraphRAG in simple fact retrieval; GraphRAG excels in complex tasks."

### 2WikiMultiHopQA (C_KG = 1.00)
- HippoRAG (arXiv:2405.14831), Table 2: BM25 R@2 51.8 / R@5 61.9 vs HippoRAG(ColBERTv2) 70.7/89.1 — by far the largest gain among HippoRAG's three datasets (+27.2pp R@5 vs +10.7 MuSiQue, +5.5 HotpotQA). With IRCoT (Table 3): 75.8/93.9.
- HippoRAG 2 (arXiv:2502.14802), Table 3: BM25 65.3 vs HippoRAG(repro) 90.4 and HippoRAG 2 90.4 (+25.1).
- HopRAG Table 2 (GPT-4o, proxy): BM25 EM 40.30 vs HopRAG 61.10.

### MetaQA (C_KG = 1.00)
- PullNet (Sun et al., EMNLP 2019, arXiv:1904.09537), Table 3 of the EMNLP-IJCNLP 2019 version (Table 2 in arXiv v1) (Hits@1; columns per hop: KB / Text / 50%KB / 50%KB+Text) [verbatim]:
  - KV-Mem: 1-hop 96.2/75.4, 2-hop 82.7/7.0, 3-hop 48.9/19.5
  - GraftNet: 1-hop 97.0/82.5, 2-hop 94.8/36.2, 3-hop 77.7/40.2
  - PullNet: 1-hop 97.0/84.4, 2-hop 99.9/81.0, 3-hop 91.4/78.2
- "Text" = same model over text-only retrieval corpus (Wikipedia-derived movie text), the lexical-retrieval condition within the same paper. KB advantage grows with hop count and is enormous (GraftNet 2-hop +58.6pp).
- GraftNet's own paper (arXiv:1809.00782) does NOT include MetaQA (it uses WikiMovies-10K + WebQSP); MetaQA numbers for GraftNet come from PullNet's reproduction (marked * therein).

### WebQSP (C_KG = 1.00)
- PullNet (arXiv:1904.09537), Table 4 of the EMNLP-IJCNLP 2019 version (Table 5 in arXiv v1) (Hits@1, KB / Text / 50%KB / 50%KB+Text) [verbatim]: KV-Mem 46.7/23.2, GraftNet 66.4/24.9, PullNet 68.1/24.8. KB-over-text gains of +23.5 to +43.3pp.
- GraftNet's own paper (arXiv:1809.00782) corroborates: GN-EF 100%-KB Hits@1 67.8 (Table 2 there reports 66.7 GN-KB / 67.8 GN-EF) vs text-only 25.3 (Table 4) — same magnitude.

## Caveats for cross-checking
1. Final PDF audit completed 2026-07-16: HopRAG Table 2, MoR Table 1, GraphRAG-Bench v3 Table 4, and GraphRetriever v2 Table 2 were checked against the official PDFs. GraphRetriever is explicitly pinned to arXiv v2 because v1 reports different results.
2. BM25 cross-paper consistency check passed: STaRK vs MoR BM25 rows identical. The MultiHop-RAG benchmark paper reports no BM25 row, so KG-CQR's BM25 baseline has no counterpart to compare with.
3. Metric heterogeneity: recall@5 (HippoRAG/HippoRAG2), Hits@1 (PullNet/STaRK/MoR), mAP/recall@k (MultiHop-RAG), EM-as-retrieval-proxy (HopRAG, GraphRetriever; same reader, retrieval varies). For the scatter, normalize as relative gain or pp-delta per metric family, or plot pp-delta with metric-coded markers.
4. KB-vs-text in PullNet is a KB-answering vs text-retrieval contrast (not graph-retrieval-over-text-corpus); it is the standard published pair for MetaQA/WebQSP and exactly captures "structured advantage on KG-derived benchmarks."
5. GraphRAG-Bench pair uses a dense (not sparse) baseline; mark with an open symbol or footnote in the figure.
6. Negative datapoints exist and should be plotted: HippoRAG v1 on NQ (-11.7pp recall@5 vs BM25) and QAGNN on STaRK (-13.0pp Hit@1 vs BM25 on MAG).
