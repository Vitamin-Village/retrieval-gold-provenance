# Single-hop verifier provenance audit

The released artifacts contain two conflicting provider labels for the
single-hop verification run:

- `singlehop_verifier_spotcheck/kappa_verdict.json` records the primary
  verifier as Google `gemini-2.5-flash`.
- Every accepted row in the frozen single-hop JSONL files records
  `verifier: openai-gpt`.

The released writer explains the conflict: it wrote
`openai-gpt` as a constant regardless of the selected backend. The writer has
been corrected for future runs to require and record the actual backend ID.
The frozen rows are not rewritten because doing so would replace the recorded
metadata with an inference.

No raw provider response envelope or provider-side run receipt is available.
Consequently, the exact primary provider cannot be independently recovered;
the run-level verdict supports Gemini, while the row-level field is known to
be unreliable. This limitation does not change the released option indices,
prompts, blind second-rater labels, or the standard-library recomputation of
29 exact matches among 35 non-abstained pairs and Cohen kappa 0.788520.
