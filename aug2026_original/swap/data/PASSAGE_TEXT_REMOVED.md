# Passage text removed from the released 2WikiMultiHop swap manifests

This release withholds 2WikiMultiHop/Wikipedia passage TEXT because the
exact per-passage revision and license metadata were not recorded
(DATA_LICENSES.md). The original `manifest.json` files in this directory
duplicated that text in each record's `abstract` field; in the released copies
the field is emptied and replaced by `abstract_sha256` (SHA-256 of the removed
UTF-8 text) and `abstract_n_chars`, so a re-fetched corpus can be verified
byte-for-byte.

To rebuild the corpus, fetch each title via the attribution list in
`licenses/2wiki_wikipedia_attribution.csv` and restore the text into
`abstract`. Every statistic reported in the paper recomputes WITHOUT the text,
from the frozen per-question vectors (`_kgswap_*.json`) in this directory.
Live re-evaluation (paper/eval/eval_independent.py and the swap/code drivers)
is the only path that needs the restored text.
