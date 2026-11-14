# Primary census rater provenance

This release preserves both primary raters' 29 ordinal bins in
`per_axis_assignments.json` and `reliability_kappa_ci.py`, plus Rater A's
quote-level records. These are sufficient to recompute the reported agreement
statistics.

However, no record identifies the exact
provider/model IDs, decoding parameters, API-client versions, or raw response
text for the two primary census raters. The five named models in
`model_identities.md` belong to the later cross-model sensitivity run and must
not be assumed to be the two primary raters.

Consequently, the numerical primary-rater agreement is reproducible from the
released bins, but the claim that the two primary raters were from different
model families cannot be independently verified from recorded provenance.
This is an unresolved provenance limitation; no model identity has been
inferred or invented.
