# Third-party data sources and license status

This file records only claims supported by source files or official upstream
metadata inspected for the final audit. It is not legal advice.

## 2WikiMultiHopQA-derived material

- Shipped paths: `swap/data/**` except both `passages.jsonl` files, plus
  derived `swap/results/**`.
- Upstream: https://github.com/Alab-NII/2wikimultihop
- Local source: the upstream `data_ids/dev.json` archive under
  `external/2wiki/raw/`, sampled into two disjoint 200-question splits.
- Upstream repository license: Apache License 2.0.
- The package adds `licenses/2wiki_wikipedia_attribution.csv`, with one row per
  audited passage, its retained title, and a title-derived Wikipedia URL.
- Caveat: the local subset does not record exact Wikipedia revision IDs,
  dump date, or per-revision license metadata. The attribution manifest
  therefore improves traceability but does not establish the license of every
  source fragment; the repository license alone is not such proof. This
  release consequently excludes the Wikipedia passage text. The
  `manifest.json` files under `swap/data/` are shipped with the `abstract`
  text field emptied for the same reason; per-record SHA-256 hashes and
  character counts remain so a re-fetched corpus can be verified
  (`swap/data/PASSAGE_TEXT_REMOVED.md`).

## MetaQA and WikiMovies-derived material

- Shipped paths: `metaqa/data/**` and derived `metaqa/results/**`.
- MetaQA upstream: https://github.com/yuyuz/MetaQA
- Acquisition record: `licenses/metaqa_acquisition.md`.
- The MetaQA README states "Creative Commons Public License" but does not name
  a version in the inspected repository text.
- The shipped passage source is WikiMovies `wiki.txt`. The retained upstream
  license is Creative Commons Attribution 3.0 Unported; the complete notice is
  included as `licenses/WIKIMOVIES_CC_BY_3.0.txt`. Its README also states that
  portions derive from MovieLens, OMDb, and a 2015-08-05 Wikipedia dump.

## STaRK-Prime-derived material

- Shipped paths: `stark/data/**` and derived `stark/results/**`.
- Upstream: https://huggingface.co/datasets/snap-stanford/stark
- Snapshot acquisition record: `licenses/stark_acquisition.md`.
- The inspected official dataset card declares CC-BY-4.0. This release contains
  derived PRIME question/entity subsets rather than the raw full release.

## arXiv-paper corpus used by the independent benchmark

- This release excludes `benchmark/passages.jsonl`; derived
  QA/audit files remain under `benchmark/` with paper and passage identifiers.
- The corpus contains text extracted from 205 research papers identified by
  arXiv-style IDs. The local manifest did not retain a per-paper license field,
  and arXiv availability is not itself a blanket redistribution license.
- Redistribution permission for the complete passage corpus therefore remains
  unverified on a per-paper basis. The passage text is not redistributed; paper
  and passage identifiers are released instead. The exclusion avoids asserting
  redistribution rights that are not established.

## Author-created outputs

The census annotations, human rating sheets/results, derived KGs, analysis
scripts, and aggregate result files are author-created or transformed outputs.
They are released under the terms in `LICENSE.md`. Files derived from the
third-party sources above also remain subject to the upstream terms listed here.
