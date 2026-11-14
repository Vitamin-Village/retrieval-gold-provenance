# Artifact notice and scope

This folder contains author-written code and annotations together with derived
subsets of third-party datasets. Author-written material is licensed as stated
in `LICENSE.md`. Source and license status of the third-party material are
documented in `DATA_LICENSES.md`; no blanket license should be inferred for the
folder as a whole.

The following files are not included:

- `swap/data/{sub200,holdout200}/kg_openie/checkpoint.json` and the adjacent
  `build_log.jsonl` files, which record extraction progress and resume state. The
  frozen KG nodes, edges, and per-question result vectors are the operative
  artifacts.
- The 2WikiMultiHop (Wikipedia) passage text of the swap splits and
  `benchmark/passages.jsonl`, the passage text of the reference benchmark.
  `DATA_LICENSES.md` gives the reasons, and `swap/data/PASSAGE_TEXT_REMOVED.md`
  describes the hashes left in their place.
