# -*- coding: utf-8 -*-
"""SPEC -- vocabulary-transplant KG builder for the 2WikiMultiHop swap (E2).

Scientific goal (preregistered)
-------------------------------
The existing KG swap (kg_silver vs kg_openie) is attackable because the two
KGs differ in quality / normalization, not only provenance. This builder
isolates the ENTITY-INVENTORY channel: graft the silver KG's entity
inventory into the OpenIE KG while keeping the OpenIE edge structure
untouched. The grafted inventory carries node ids, names, aliases, AND each
silver node's gold source_docs mapping, because those jointly constitute the
gold entity normalization -- the provenance property under test. Concretely,
on a node_id collision the silver record replaces the OpenIE record
wholesale (including source_docs, which differ on most collisions), and
silver-only nodes enter with their gold source_docs intact, injecting real
node->passage links into the otherwise independent graph. The T1b condition
below decomposes this deliberately bundled move: it empties source_docs on
every silver-sourced node, separating the lexical-seeding sub-channel
(names/aliases -> seed matching) from the passage-mapping sub-channel
(source_docs -> node->passage scoring). If the inventory alone moves hit@5
inside the noisy independent graph, the inventory channel -- not graph
quality -- carries the swap effect.

Inputs (read-only; never modified)
----------------------------------
For each split D in {paper/external/2wiki/sub200, paper/external/2wiki/holdout200}:
  D/kg_silver/proposed_kg_nodes.json   list of node dicts:
      {node_id "n::<lower title>", name, type, aliases [QID, surfaces...],
       source_docs [<lower gold passage title>], wikidata_ids}
  D/kg_silver/proposed_kg_edges.json   (loaded only for count reporting)
  D/kg_openie/proposed_kg_nodes.json   same schema minus wikidata_ids
                                       (aliases typically empty)
  D/kg_openie/proposed_kg_edges.json   list of edge dicts:
      {edge_id, source, target, source_id, target_id, relation,
       relation_type, source_doc, source_docs, reliability_score}
  D/passages.jsonl                     (sanity check only: every silver
                                       source_doc must be a corpus paper_id)

Outputs (NEW files only)
------------------------
T1  D/kg_openie_svocab/proposed_kg_nodes.json
      nodes = OpenIE nodes UNION silver nodes, deduplicated by node_id.
      On node_id collision the SILVER record wins (its name/aliases/
      source_docs/wikidata_ids replace the OpenIE record). Within each
      source list the FIRST occurrence of a node_id wins (ids are unique
      in the shipped files; the rule exists for determinism only).
      Node list is sorted ascending by node_id (Python sorted(), stable).
    D/kg_openie_svocab/proposed_kg_edges.json
      edges = OpenIE edges ONLY, content byte-identical per record,
      sorted ascending by edge_id (stable sort; duplicate edge_ids -- none
      in the shipped files -- would keep original relative order).
    D/kg_openie_svocab/transplant_manifest.json
      records the rule text, input paths, and the counts printed below.

T1b D/kg_openie_svocab_noprov/proposed_kg_nodes.json (+edges, +manifest)
      identical to T1 with ONE deterministic post-step applied AFTER the
      union/dedupe/silver-wins merge: every silver-sourced node (all
      collisions plus all silver-only nodes, i.e. node_id in the silver id
      set) gets source_docs = []. name/aliases/wikidata_ids/node_id/type
      are untouched; OpenIE-only nodes keep their OpenIE source_docs
      verbatim; edges identical to T1. T1b ~= T1 means the transferred
      gain is lexical (names/aliases seeding); T1b ~= 0 means it is the
      injected gold node->passage mapping. Built unconditionally.

    NOTE on ordering: the shipped kg_openie files are NOT id-sorted, so the
    transplant files differ from the originals in on-disk order as well as
    in vocabulary. The sort is preregistered (deterministic output). Because
    ProposedV3Retriever's legacy BFS pairs each discovered node with the
    edge that first reached it (adjacency insertion order = edge file
    order), a purely order-induced artifact is conceivable for proposed_v3.
    Pass --with-order-control to ALSO emit D/kg_openie_resorted (OpenIE
    nodes+edges only, same stable sort, zero vocabulary change) so the
    orchestrator can optionally quantify the pure-order effect. PPR-family
    systems (ppr, proposed_v5_graphcentric) are order-insensitive (set /
    normalized-mass arithmetic with sorted tie-breaks).

Tq  D/kg_openie_svocab_q25 / _q50 / _q75 (+edges, +manifest each)
      q-curve interpolation between T1b (q=0) and T1 (q=1), added 2026-06-12
      for the W1 defense. Same T1 union/dedupe/silver-wins merge; among the
      silver-sourced nodes (the same id set T1b empties) a fraction q KEEPS
      its silver source_docs and the remaining (1-q) gets source_docs = [].
      Selection is deterministic: sort the silver-sourced node_ids, shuffle
      once with random.Random(20260612), keep the first ceil(q*N). The kept
      sets are therefore NESTED across q (prefix rule): kept(0.25) is a
      subset of kept(0.5) is a subset of kept(0.75). q=1.0 reproduces T1
      exactly and q=0.0 reproduces T1b exactly (verified in-build via
      build_q_nodes and recorded in each q manifest; also exposed as
      verify_q_boundaries(split_dir) for post-hoc checks). The manifest
      records q, the seed, N, ceil(q*N), and the kept node_id list.

T2  kg_silver_devocab -- SKIPPED (deliberate; rationale printed at run time
    and recorded in the manifest). Summary of the judgment:
      (a) The prescribed counterpart rule (OpenIE node sharing a source_doc
          with the longest case-insensitive common substring of the silver
          name, ties lexicographic) is not an identity criterion: short
          accidental substrings pick semantically unrelated entities, so the
          output is "silver with noisy relabeling", not "silver minus gold
          normalization".
      (b) Replacing source_docs with the counterpart's source_docs perturbs
          the node->passage mapping channel, conflating exactly the two
          channels (vocabulary vs provenance mapping) the ablation must
          separate; keeping silver source_docs instead would deviate from
          the preregistered rule. Either variant is contestable.
      (c) The no-counterpart fallback (name = first token of the silver
          name) leaks gold vocabulary: a single title token still seed-
          matches via ProposedV3Retriever._surface_matches_query and
          PPRRetriever._seed_indices, biasing T2 toward a null result.
      (d) node_id itself retains the bare lowercase title; rewriting ids to
          remove it forces edge-endpoint remapping that can merge distinct
          entities (degraded names collide), changing graph topology -- a
          quality change, i.e. the very confound under test.
    T1 (vocab added to the independent graph) plus the existing zero-edge
    dose-response point (paper/eval/_swap_dose_response.py, f=0) already
    brackets the mechanism from both sides.

Printed per split (ASCII only)
------------------------------
  n_openie_nodes, n_silver_nodes, n_union, n_collisions, n_edges,
  plus sanity counts (silver source_doc refs missing from the passage pool,
  expected 0; silver-only nodes added).

Determinism
-----------
Pure file transforms; no RNG, no timestamps, no tuning. Re-running produces
byte-identical outputs. json.dump(ensure_ascii=True, indent=1, sort_keys
NOT applied -- record key order is preserved as loaded).

CLI
---
    python paper/eval/build_transplant_kgs.py [--splits sub200,holdout200]
                                              [--with-order-control]
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "paper" / "external" / "2wiki"
SPLITS = ("sub200", "holdout200")

# q-curve (partial provenance) conditions interpolating T1b (q=0) -> T1 (q=1)
Q_SEED = 20260612
Q_VALUES = (0.25, 0.5, 0.75)


def q_dir_name(q):
    return "kg_openie_svocab_q%02d" % int(round(q * 100))


Q_RULE = (
    "T1 union (silver wins on collision, sorted by node_id); among the "
    "silver-sourced node_ids (the same set T1b empties), sort the ids, "
    "shuffle once with random.Random(%d), and keep the silver source_docs "
    "on the first ceil(q*N) ids; the remaining (1-q) fraction gets "
    "source_docs=[]. OpenIE-only nodes keep their source_docs verbatim; "
    "edges = OpenIE edges only, sorted by edge_id. Kept sets are nested "
    "across q (prefix rule). q=1.0 is identical to kg_openie_svocab (T1), "
    "q=0.0 is identical to kg_openie_svocab_noprov (T1b)." % Q_SEED
)

T1_RULE = (
    "nodes = OpenIE nodes UNION silver nodes, deduped by node_id, silver "
    "record preferred on collision, sorted by node_id; edges = OpenIE edges "
    "only, record content unchanged, sorted by edge_id."
)
T2_SKIP_REASON = (
    "kg_silver_devocab SKIPPED: (a) longest-common-substring counterpart "
    "matching is not an identity criterion (noisy relabeling, not "
    "de-normalization); (b) replacing source_docs perturbs the "
    "node->passage mapping channel, conflating the two channels the "
    "ablation separates; (c) the first-token fallback name still leaks "
    "gold title vocabulary into seed matching; (d) removing the title from "
    "node_id forces endpoint remapping whose name collisions merge "
    "distinct entities (a topology/quality change). T1 plus the existing "
    "zero-edge dose-response (f=0 in _swap_dose_response.py) brackets the "
    "mechanism. NOTE: T1 itself also moves source_docs, deliberately -- "
    "the grafted inventory includes the gold node->passage mapping as part "
    "of the gold entity normalization under test -- and this bundled move "
    "is DECOMPOSED by the T1b condition (kg_openie_svocab_noprov: "
    "source_docs emptied on every silver-sourced node) rather than avoided."
)


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=True, indent=1)


def _node_id(node):
    return str(node.get("node_id") or node.get("id") or "")


def _edge_id(edge):
    return str(edge.get("edge_id") or "")


def _passage_paper_ids(split_dir):
    ids = set()
    with open(split_dir / "passages.jsonl", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                pid = json.loads(line).get("paper_id")
                if pid:
                    ids.add(str(pid))
    return ids


def q_kept_ids(silver_ids, q, seed=Q_SEED):
    """Deterministic kept-id selection for the q-curve: sort the silver-
    sourced node_ids, shuffle once with a seeded RNG, keep the first
    ceil(q*N). The prefix rule nests the kept sets across q values.
    (q values 0.25/0.5/0.75/1.0 are exact binary fractions, so q*N is
    exact and plain math.ceil implements the preregistered ceil(qN).)"""
    ids = sorted(silver_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_keep = int(math.ceil(q * len(ids)))
    return ids[:n_keep]


def build_q_nodes(merged, silver_ids, q, seed=Q_SEED):
    """Node list for the partial-provenance condition: silver-sourced nodes
    in the kept prefix retain their (silver) source_docs, the rest get
    source_docs=[]; OpenIE-only nodes pass through untouched. Returns
    (sorted node list, kept id set). Shared node dicts are copied before
    mutation so callers' inputs are never modified."""
    silver_in_merge = set(silver_ids) & set(merged)
    kept = set(q_kept_ids(silver_in_merge, q, seed))
    nodes = []
    for nid in sorted(merged):
        node = merged[nid]
        if nid in silver_in_merge and nid not in kept:
            node = dict(node)
            node["source_docs"] = []
        nodes.append(node)
    return nodes, kept


def verify_q_boundaries(split_dir):
    """Post-hoc boundary check from on-disk artifacts: build_q_nodes at
    q=1.0 must reproduce kg_openie_svocab (T1) and at q=0.0 must reproduce
    kg_openie_svocab_noprov (T1b), node-by-node including source_docs.
    Returns {'q1_matches_t1': bool, 'q0_matches_t1b': bool}."""
    split_dir = Path(split_dir)
    t1_nodes = _load(split_dir / "kg_openie_svocab" / "proposed_kg_nodes.json")
    t1b_nodes = _load(
        split_dir / "kg_openie_svocab_noprov" / "proposed_kg_nodes.json")
    silver_nodes = _load(split_dir / "kg_silver" / "proposed_kg_nodes.json")
    silver_ids = {_node_id(n) for n in silver_nodes if _node_id(n)}
    merged = {_node_id(n): n for n in t1_nodes}
    q1_nodes, _ = build_q_nodes(merged, silver_ids, 1.0)
    q0_nodes, _ = build_q_nodes(merged, silver_ids, 0.0)
    same = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(
        b, sort_keys=True)  # noqa: E731
    return {"q1_matches_t1": same(q1_nodes, t1_nodes),
            "q0_matches_t1b": same(q0_nodes, t1b_nodes)}


def build_split(split_dir, with_order_control=False):
    openie_nodes = _load(split_dir / "kg_openie" / "proposed_kg_nodes.json")
    openie_edges = _load(split_dir / "kg_openie" / "proposed_kg_edges.json")
    silver_nodes = _load(split_dir / "kg_silver" / "proposed_kg_nodes.json")
    silver_edges = _load(split_dir / "kg_silver" / "proposed_kg_edges.json")

    # ---- T1 union (dedupe by node_id; silver wins on collision) ---------
    merged = {}
    n_openie_internal_dup = 0
    for node in openie_nodes:
        nid = _node_id(node)
        if not nid:
            continue
        if nid in merged:
            n_openie_internal_dup += 1
            continue  # first occurrence wins
        merged[nid] = node
    openie_ids = set(merged)
    n_collisions = 0
    n_silver_internal_dup = 0
    silver_seen = set()
    for node in silver_nodes:
        nid = _node_id(node)
        if not nid:
            continue
        if nid in silver_seen:
            n_silver_internal_dup += 1
            continue  # first occurrence wins within silver
        silver_seen.add(nid)
        if nid in merged:
            n_collisions += 1
        merged[nid] = node  # silver record preferred
    union_nodes = [merged[nid] for nid in sorted(merged)]
    out_edges = sorted(openie_edges, key=_edge_id)

    # ---- sanity: silver source_docs must live in the passage paper pool.
    # Computed BEFORE any output is written so a failure here can never
    # leave a half-written kg_openie_svocab dir. passages.jsonl is not a
    # required input of the transform itself, so its absence downgrades
    # the check to "unmeasured" rather than aborting the build.
    if (split_dir / "passages.jsonl").exists():
        paper_ids = _passage_paper_ids(split_dir)
        n_srcdoc_missing = sum(
            1
            for node in silver_nodes
            for doc in (node.get("source_docs") or [])
            if str(doc) not in paper_ids
        )
    else:
        n_srcdoc_missing = None  # unmeasured: no passages.jsonl in split dir

    out_dir = split_dir / "kg_openie_svocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    _dump(union_nodes, out_dir / "proposed_kg_nodes.json")
    _dump(out_edges, out_dir / "proposed_kg_edges.json")

    counts = {
        "n_openie_nodes": len(openie_nodes),
        "n_silver_nodes": len(silver_nodes),
        "n_union": len(union_nodes),
        "n_collisions": n_collisions,
        # Direct set computation (not len(union)-len(openie)): immune to
        # internal duplicate node_ids in either source file.
        "n_silver_only_added": len(silver_seen - openie_ids),
        "n_edges": len(out_edges),
        "n_silver_edges_excluded": len(silver_edges),
        "n_openie_internal_dup_node_ids": n_openie_internal_dup,
        "n_silver_internal_dup_node_ids": n_silver_internal_dup,
        "n_silver_srcdoc_refs_missing_from_passages": n_srcdoc_missing,
        "openie_nodes_file_was_id_sorted": (
            [_node_id(n) for n in openie_nodes]
            == sorted(_node_id(n) for n in openie_nodes)
        ),
        "openie_edges_file_was_id_sorted": (
            [_edge_id(e) for e in openie_edges]
            == sorted(_edge_id(e) for e in openie_edges)
        ),
    }
    def _repo_relative(path):
        # Repo-relative when the split lives under ROOT (the real splits;
        # also keeps absolute local paths out of the artifact); plain name
        # chain otherwise (e.g. test fixtures under a temp dir).
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            # Location-independent (the fixture's temp parent must not leak
            # into the manifest, or two runs in different temp dirs differ).
            return str(Path(*path.parts[-2:]))

    manifest = {
        "ablation": "E2 vocabulary transplant (T1 kg_openie_svocab)",
        "rule": T1_RULE,
        "t2_kg_silver_devocab": {"built": False, "skip_reason": T2_SKIP_REASON},
        "inputs": {
            "openie_nodes": _repo_relative(
                split_dir / "kg_openie" / "proposed_kg_nodes.json"
            ),
            "openie_edges": _repo_relative(
                split_dir / "kg_openie" / "proposed_kg_edges.json"
            ),
            "silver_nodes": _repo_relative(
                split_dir / "kg_silver" / "proposed_kg_nodes.json"
            ),
        },
        "counts": counts,
        "determinism": "pure file transform; no RNG; rerun is byte-identical",
    }
    _dump(manifest, out_dir / "transplant_manifest.json")

    # ---- T1b: same union, source_docs EMPTIED on every silver-sourced
    # node (collisions + silver-only). Decomposes the lexical-seeding
    # sub-channel from the passage-mapping sub-channel. Shared node dicts
    # are copied before mutation so the T1 output above is unaffected.
    noprov_dir = split_dir / "kg_openie_svocab_noprov"
    noprov_dir.mkdir(parents=True, exist_ok=True)
    n_t1b_emptied = 0
    noprov_nodes = []
    for nid in sorted(merged):
        node = merged[nid]
        if nid in silver_seen:
            node = dict(node)
            node["source_docs"] = []
            n_t1b_emptied += 1
        noprov_nodes.append(node)
    counts["n_t1b_source_docs_emptied"] = n_t1b_emptied
    _dump(noprov_nodes, noprov_dir / "proposed_kg_nodes.json")
    _dump(out_edges, noprov_dir / "proposed_kg_edges.json")
    _dump(
        {
            "ablation": "E2 inventory transplant, provenance-neutralized "
                        "(T1b kg_openie_svocab_noprov)",
            "rule": "T1 union with source_docs=[] on every silver-sourced "
                    "node (node_id in the silver id set); OpenIE-only nodes "
                    "keep their source_docs; edges = OpenIE edges only, "
                    "sorted by edge_id.",
            "n_source_docs_emptied": n_t1b_emptied,
            "counts": counts,
            "determinism": "pure file transform; no RNG; rerun is "
                           "byte-identical",
        },
        noprov_dir / "transplant_manifest.json",
    )

    # ---- Tq: q-curve partial-provenance conditions (q in Q_VALUES).
    # Same union; a deterministic fraction q of the silver-sourced nodes
    # keeps its silver source_docs, the rest are emptied as in T1b.
    # In-build boundary check: q=1.0 must equal the T1 node list and
    # q=0.0 must equal the T1b node list written above (exact JSON match).
    q1_nodes, _ = build_q_nodes(merged, silver_seen, 1.0)
    q0_nodes, _ = build_q_nodes(merged, silver_seen, 0.0)
    _same = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(
        b, sort_keys=True)  # noqa: E731
    boundary_check = {"q1_matches_t1": _same(q1_nodes, union_nodes),
                      "q0_matches_t1b": _same(q0_nodes, noprov_nodes)}
    if not all(boundary_check.values()):
        raise AssertionError(
            "q-curve boundary check failed: %s" % boundary_check)
    counts["q_boundary_check"] = boundary_check
    q_built = {q: build_q_nodes(merged, silver_seen, q) for q in Q_VALUES}
    for q in Q_VALUES:
        counts["n_q%02d_source_docs_kept" % int(round(q * 100))] = len(
            q_built[q][1])
    for q in Q_VALUES:
        q_nodes, kept = q_built[q]
        q_dir = split_dir / q_dir_name(q)
        q_dir.mkdir(parents=True, exist_ok=True)
        _dump(q_nodes, q_dir / "proposed_kg_nodes.json")
        _dump(out_edges, q_dir / "proposed_kg_edges.json")
        _dump(
            {
                "ablation": "E2 inventory transplant, partial provenance "
                            "q=%s (%s)" % (q, q_dir_name(q)),
                "rule": Q_RULE,
                "q": q,
                "seed": Q_SEED,
                "n_silver_sourced": len(silver_seen),
                "n_source_docs_kept": len(kept),
                "n_source_docs_emptied": len(silver_seen) - len(kept),
                "kept_node_ids": sorted(kept),
                "boundary_check": boundary_check,
                "counts": counts,
                "determinism": "seeded RNG random.Random(%d) over sorted "
                               "silver node_ids, single shuffle, prefix "
                               "selection; rerun is byte-identical" % Q_SEED,
            },
            q_dir / "transplant_manifest.json",
        )

    if with_order_control:
        # Pure order control: OpenIE nodes+edges only, same stable sort,
        # ZERO vocabulary change. Lets the orchestrator separate any
        # adjacency-insertion-order effect from the vocabulary effect.
        ctrl_dir = split_dir / "kg_openie_resorted"
        ctrl_dir.mkdir(parents=True, exist_ok=True)
        ctrl_nodes = {}
        for node in openie_nodes:
            nid = _node_id(node)
            if nid and nid not in ctrl_nodes:
                ctrl_nodes[nid] = node
        _dump(
            [ctrl_nodes[nid] for nid in sorted(ctrl_nodes)],
            ctrl_dir / "proposed_kg_nodes.json",
        )
        _dump(out_edges, ctrl_dir / "proposed_kg_edges.json")
        _dump(
            {
                "ablation": "E2 order control (kg_openie_resorted)",
                "rule": "OpenIE nodes+edges only, sorted by node_id/edge_id;"
                        " no vocabulary change",
            },
            ctrl_dir / "transplant_manifest.json",
        )

    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--with-order-control", action="store_true")
    args = ap.parse_args(argv)

    print("T2 decision: SKIP -- " + T2_SKIP_REASON, flush=True)
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_dir = BASE / split
        if not split_dir.is_dir():
            print("ERROR: missing split dir %s" % split_dir, flush=True)
            return 1
        counts = build_split(split_dir, with_order_control=args.with_order_control)
        print(
            "%-12s n_openie_nodes=%d n_silver_nodes=%d n_union=%d "
            "n_collisions=%d n_edges=%d | silver_only_added=%d "
            "srcdoc_missing=%s openie_files_presorted=%s/%s"
            % (
                split,
                counts["n_openie_nodes"],
                counts["n_silver_nodes"],
                counts["n_union"],
                counts["n_collisions"],
                counts["n_edges"],
                counts["n_silver_only_added"],
                counts["n_silver_srcdoc_refs_missing_from_passages"],
                counts["openie_nodes_file_was_id_sorted"],
                counts["openie_edges_file_was_id_sorted"],
            ),
            flush=True,
        )
        print("wrote %s" % (split_dir / "kg_openie_svocab"), flush=True)
        print(
            "%-12s q-curve: %s | boundary q1==T1:%s q0==T1b:%s"
            % (
                split,
                " ".join(
                    "%s kept=%d" % (
                        q_dir_name(q),
                        counts["n_q%02d_source_docs_kept"
                               % int(round(q * 100))],
                    )
                    for q in Q_VALUES
                ),
                counts["q_boundary_check"]["q1_matches_t1"],
                counts["q_boundary_check"]["q0_matches_t1b"],
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
