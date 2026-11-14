# Inherited Overlap Between Self-Built OpenIE KG and Gold Entity Inventory (2WikiMultiHop)

Script: `paper/analysis/inherited_overlap.py` | Results: `paper/results/inherited_overlap.json`

## Matching rule

Identical to `paper/eval/_swap_vocab_align.py`: a surface form is the lowercased,
whitespace-collapsed string of a node's `node_id` (minus the `n::` prefix), `name`,
or any entry in `aliases`; two surface forms match only on exact string equality
after this normalization. A silver node counts as *surface-inherited* if any of its
surface forms is also a surface form of some OpenIE node, and as *map-inherited* if,
additionally, at least one such surface-matched OpenIE node shares a `source_docs`
passage with the silver node (i.e., the self-built KG already carries the gold
node-to-passage link). Question-level mapping coverage requires the matched OpenIE
node to list the gold passage itself in its `source_docs`.

## Node-level inheritance (denominator: silver-KG nodes = gold entity inventory)

| Split | Silver nodes | OpenIE nodes | Omega_surface | Omega_map | Omega_map given surface match |
|---|---|---|---|---|---|
| sub200 | 589 | 9,949 | 0.730 (430/589) | 0.335 (197/589) | 0.458 |
| holdout200 | 593 | 9,629 | 0.809 (480/593) | 0.344 (204/593) | 0.425 |

## Question-level inheritance (per-question fraction of gold passage titles)

| Split | Surface mean | Surface fully covered | Mapping mean | Mapping fully covered |
|---|---|---|---|---|
| sub200 | 0.560 | 0.340 | 0.370 | 0.175 |
| holdout200 | 0.613 | 0.395 | 0.388 | 0.215 |

Reproduction check against `paper/results/swap_vocab_align.json` (OpenIE condition):
surface coverage matches the released aggregates exactly on both splits
(0.5600/0.3400 sub200, 0.6125/0.3950 holdout200) -- **PASS**.

## Argument

A self-built OpenIE KG extracted from 2Wiki passages is not vocabulary-independent
of the gold inventory: because the passages themselves originate from Wikidata, the
OpenIE extraction recovers 73.0-80.9 percent of the gold entity inventory's surface
forms and already carries the exact gold node-to-passage link for 33.5-34.4 percent
of gold nodes, without any access to the silver KG. At the question level this means
that for the average question, 56.0-61.3 percent of gold passage titles are already
in the self-built node inventory and 37.0-38.8 percent come with the gold passage
mapping attached for free. The assumption that systems building their own
OpenIE KG operate at alignment zero therefore does not hold on this
benchmark: they operate at a measurable partial-alignment point, intermediate
between zero and the census-flagged condition with full inventory alignment. Our
census flag does not assert that public results are fully inflated; it locates each
condition on this alignment axis, and the OpenIE condition sits at the partial
inheritance levels measured here. Since the PPR-on-OpenIE condition reproduces the
operating point of the HippoRAG-style mechanism (swap gains of +0.315/+0.310), the
inherited-overlap numbers transfer directly to that mechanism family.

## Appendix: feasibility of running official HippoRAG here (document-level assessment only; not installed)

The official HippoRAG repository requires an OpenAI API key for its OpenIE indexing
and query NER passes (per-corpus indexing cost scales with passage tokens) plus a
ColBERTv2 checkpoint download, and its local-LLM alternative routes through vLLM,
which does not support native Windows. Under this environment's constraints
(Windows, no-network analysis policy, paid API dependency), a faithful execution of
the official pipeline is not reproducible here, which is consistent with our choice
to re-create its operating point via the PPR-on-OpenIE swap instead.
