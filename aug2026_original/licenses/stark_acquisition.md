# STaRK-Prime Acquisition Record

Benchmark: STaRK (Semi-structured Retrieval on Textual and Relational
Knowledge Bases), PRIME (biomedical) subset.

## Source
- Dataset: snap-stanford/stark on the Hugging Face Hub.
- QA file (pre-staged): prime_qa.csv
  - URL family: https://huggingface.co/datasets/snap-stanford/stark
    (qa/prime/stark_qa/stark_qa.csv on the Hub).
  - 11,204 rows; columns: id, query, answer_ids (list of SKB node indices).
- SKB (downloaded):
  - URL: https://huggingface.co/datasets/snap-stanford/stark/resolve/main/skb/prime/processed.zip
  - Retrieved: 2026-06-15 (UTC), via curl -sL.
  - Archive: processed.zip, 28,013,696 bytes (~28 MB).
  - raw.zip (~205 MB) was deliberately NOT downloaded; processed.zip is
    sufficient to recover node_info / edges / type dicts.

## Extracted files (paper/external/stark/raw/processed/)
- node_info.pkl        70,747,199 bytes  (dict: 129,375 nodes)
- edge_index.pt       129,608,692 bytes  (tensor [2 x 8,100,498], int64)
- edge_types.pt        64,804,724 bytes  (tensor [8,100,498])
- node_types.pt         1,035,700 bytes  (tensor [129,375])
- node_type_dict.pkl          330 bytes  (10 entity types)
- edge_type_dict.pkl          181 bytes  (18 relation types)

File count: 1 archive + 6 extracted artifacts (+ prime_qa.csv pre-staged).

## Parsing
Parsed directly from the processed artifacts; the stark-qa Python package was
NOT required. node_info.pkl is a plain dict keyed by the same integer index
used in prime_qa.csv answer_ids (verified: all 12,828 distinct answer ids
resolve to nodes, 0 missing). Edges joined to relation strings via
edge_type_dict; graph treated as undirected for silver-KG construction.

## License
STaRK is released under CC-BY-4.0 (Creative Commons Attribution 4.0
International), per the snap-stanford/stark dataset card. Attribution:
Wu et al., "STaRK: Benchmarking LLM Retrieval on Textual and Relational
Knowledge Bases," NeurIPS 2024 Datasets and Benchmarks.

Note: dataset-card / repository (github / Hub identity) URLs are for the
manuscript citation only; they are not embedded in the released data
files (which contain only derived passages / qa / KG).
