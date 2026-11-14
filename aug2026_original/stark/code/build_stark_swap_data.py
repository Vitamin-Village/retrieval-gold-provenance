# -*- coding: utf-8 -*-
"""STaRK-Prime third-swap data builder (S1 subset + S2 silver KG).

SUPERSEDED -- do NOT run to (re)generate the on-disk artifacts. This script
keys passages / nodes by normalized entity NAME ("n::ttr"), which collides for
distinct entities that share a name and only mapped a subset of nodes. The
frozen on-disk build uses ENTITY-ID keying ("n::e<id>") via
paper/eval/rebuild_stark_silver_rr.py (corpus 1,250, round-robin neighbour
distractors -> 0 isolated gold). Use rebuild_stark_silver_rr.py for any
regeneration; this file is kept only for the S1 selection logic and design
reference. Running main() here would overwrite the disk artifacts with the
divergent name-keyed scheme.

Implements paper/eval/_stark_swap_design.md. Mirrors the MetaQA / 2Wiki
sub200 layout byte-for-byte in structure so the existing eval harness
(paper/eval/eval_independent.py) and the OpenIE builder
(paper/baselines/build_openie_kg.py) are reused with ZERO code changes.

Census point: STaRK-Prime answer-entity axis C_K = 0.75 (the gold answer
entity is SELECTED by KB relation traversal -- STaRK synthesizes each query
from a relational template over the precomputed knowledge base). This is a
NON-MAXIMAL circular point (2Wiki and MetaQA are both 1.0), which is the
whole value: it tests whether the effect needs a maximal circular point.

S1  From paper/external/stark/raw/prime_qa.csv keep questions whose gold
    answer-entity set (answer_ids) is small (size 1..3; size-1 preferred).
    Each gold entity is a STaRK SKB node (node_info index). corpus = all gold
    entities + deterministic distractor entities up to CORPUS_SIZE. One
    passage per entity (title = entity name, abstract = entity description
    assembled from STaRK ``details``; falls back to "<name> (<type>)" when no
    description text exists). Stratified round-robin sample over gold node
    TYPE (disease/drug/gene/...) so no single biomedical type dominates.
    Seeded (SEED); a disjoint holdout200 is frozen when the pool allows.

S2  Silver KG per split: the SKB subgraph induced by the selected questions'
    gold entities -- gold entity nodes plus their 1-hop SKB neighbours, with
    the actual SKB relation edges (both directions; STaRK treats the graph as
    undirected). Each entity node -> its own passage via source_docs. Asserts
    gold-entity coverage == 1.0 by construction.

Determinism: the whole build runs twice in memory and the script asserts the
two passes serialize identically before anything is written.

ASCII-only stdout. Run from repo root:
    python paper/eval/build_stark_swap_data.py

DOES NOT run OpenIE (S3) or evaluation (S4).
"""
from __future__ import annotations

import ast
import csv
import json
import pickle
import re
import random
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "paper" / "external" / "stark" / "raw"
PROC = RAW / "processed"
OUT_BASE = ROOT / "paper" / "external" / "stark"
PROGRESS = OUT_BASE / "_PROGRESS.log"

SEED = 20260615
N_PER_SPLIT = 200
CORPUS_SIZE = 1250
MAX_GOLD = 3          # accept answer sets of size 1..3 (1 preferred)
ABSTRACT_CHARS = 1200  # cap per-passage abstract length (deterministic)

_WS_RE = re.compile(r"\s+")

# Per-type ordered list of STaRK ``details`` fields that carry usable prose.
# First non-empty field(s) are concatenated into the passage abstract.
_TEXT_FIELDS = {
    "gene/protein": ["summary"],
    "drug": ["description", "indication", "mechanism_of_action",
             "pharmacodynamics"],
    "disease": ["mondo_definition", "umls_description", "orphanet_definition",
                "orphanet_clinical_description", "mayo_symptoms",
                "mayo_causes"],
    # pathway / molecular_function / biological_process / cellular_component /
    # anatomy / effect-phenotype / exposure carry no prose in processed.zip;
    # they fall back to name + type (handled below).
}


def norm(s: str) -> str:
    return _WS_RE.sub(" ", str(s).strip()).lower()


def is_text(v) -> bool:
    """True iff v is a usable non-empty string (filters NaN floats / None)."""
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    if s.lower() in ("nan", "none", "null", "n/a", "na"):
        return False
    return True


def log_progress(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write("%s %s\n" % (ts, msg))
    print("[progress] " + msg, flush=True)


# ---------------------------------------------------------------------------
# SKB loading (direct parse of processed.zip artifacts; no stark-qa package)
# ---------------------------------------------------------------------------
def load_skb():
    """Return (node_info, edge_type_dict, adj, nbr).

    node_info : dict[int] -> {id,type,name,source,details}
    adj       : dict[int] -> list[(neighbour_id, relation_str)]  (both dirs)
    nbr       : dict[int] -> set(neighbour_id)  (relation-agnostic, both dirs)
    """
    import torch
    with (PROC / "node_info.pkl").open("rb") as fh:
        node_info = pickle.load(fh)
    with (PROC / "edge_type_dict.pkl").open("rb") as fh:
        edge_type_dict = pickle.load(fh)
    ei = torch.load(PROC / "edge_index.pt", weights_only=False)
    et = torch.load(PROC / "edge_types.pt", weights_only=False)
    src = ei[0].tolist()
    dst = ei[1].tolist()
    etl = et.tolist()
    adj = defaultdict(list)
    nbr = defaultdict(set)
    seen = set()
    for s, d, r in zip(src, dst, etl):
        rel = edge_type_dict.get(r, str(r))
        # store undirected (both directions), dedup exact (s,rel,d)
        k1 = (s, rel, d)
        if k1 not in seen:
            seen.add(k1)
            adj[s].append((d, rel))
        k2 = (d, rel, s)
        if k2 not in seen:
            seen.add(k2)
            adj[d].append((s, rel))
        if s != d:
            nbr[s].add(d)
            nbr[d].add(s)
    return node_info, edge_type_dict, adj, nbr


def entity_abstract(node) -> str:
    """Assemble a passage abstract from a STaRK node's name + details prose.

    Always prefixed by the entity name + type so name-only entity types still
    yield a retrievable, self-describing passage."""
    name = str(node.get("name") or "").strip()
    ntype = str(node.get("type") or "entity").strip()
    head = "%s (%s)." % (name, ntype)
    details = node.get("details") or {}
    parts = []
    if isinstance(details, dict):
        # type-specific prose fields
        for fld in _TEXT_FIELDS.get(ntype, []):
            v = details.get(fld)
            if is_text(v):
                parts.append(v.strip())
        # gene/drug aliases add retrievable surface forms
        alias = details.get("alias")
        if isinstance(alias, list):
            al = [a for a in alias if is_text(a)]
            if al:
                parts.append("Also known as: " + ", ".join(al[:12]) + ".")
    body = " ".join(parts)
    out = head if not body else (head + " " + body)
    out = _WS_RE.sub(" ", out).strip()
    if len(out) > ABSTRACT_CHARS:
        out = out[:ABSTRACT_CHARS].rstrip() + " ..."
    return out


def entity_aliases(node) -> list:
    name = str(node.get("name") or "").strip()
    out = [name] if name else []
    details = node.get("details") or {}
    if isinstance(details, dict):
        alias = details.get("alias")
        if isinstance(alias, list):
            for a in alias:
                if is_text(a) and a.strip() not in out:
                    out.append(a.strip())
    return out


# ---------------------------------------------------------------------------
# QA parsing + passage id assignment
# ---------------------------------------------------------------------------
def parse_qa(node_info):
    """Return list of eligible QA dicts (gold size 1..MAX_GOLD, all gold ids
    present in node_info)."""
    rows = []
    with (RAW / "prime_qa.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                ids = [int(x) for x in ast.literal_eval(r["answer_ids"])]
            except Exception:
                continue
            ids = sorted(set(ids))
            if not ids or len(ids) > MAX_GOLD:
                continue
            if any(i not in node_info for i in ids):
                continue
            rows.append({"id": r["id"], "query": r["query"], "gold_ids": ids})
    return rows


def assign_passage_ids(node_ids, node_info):
    """Map each SKB node index -> a unique stable passage_id string.

    Base id = normalized entity name; collisions get the node index appended
    so the id stays human-readable yet unique and deterministic."""
    base = {}
    used = defaultdict(list)
    for nid in sorted(node_ids):
        nm = norm(node_info[nid].get("name") or "")
        if not nm:
            nm = "entity"
        base[nid] = nm
        used[nm].append(nid)
    pid = {}
    for nm, nids in used.items():
        if len(nids) == 1:
            pid[nids[0]] = nm
        else:
            for nid in nids:
                pid[nid] = "%s::n%d" % (nm, nid)
    return pid


# ---------------------------------------------------------------------------
# Subset selection (stratified by dominant gold-entity type, seeded RR)
# ---------------------------------------------------------------------------
def gold_stratum(qa, node_info):
    """Stratify by the type of the first gold entity (sorted id), so the
    subset spans STaRK's biomedical entity types rather than one type."""
    return str(node_info[qa["gold_ids"][0]].get("type") or "unknown")


def stratified_select(qa_rows, node_info, n_total):
    rng = random.Random(SEED)
    pools = defaultdict(list)
    for qa in qa_rows:
        pools[gold_stratum(qa, node_info)].append(qa)
    ordered = OrderedDict()
    for strat in sorted(pools):
        lst = list(pools[strat])
        # deterministic order then seeded shuffle
        lst.sort(key=lambda q: q["id"])
        rng.shuffle(lst)
        ordered[strat] = lst
    selected = []
    while len(selected) < n_total and any(ordered.values()):
        for strat in ordered:
            if ordered[strat]:
                selected.append(ordered[strat].pop(0))
                if len(selected) >= n_total:
                    break
    return selected


# ---------------------------------------------------------------------------
# Split artifacts
# ---------------------------------------------------------------------------
def select_distractors(gold_nodes, node_info, nbr):
    """Round-robin neighbour-priority distractor selection (deterministic).

    Each gold entity contributes its SKB 1-hop neighbours in turn so that
    EVERY gold gets at least one corpus neighbour before any gold gets a
    second -> the induced subgraph leaves no gold isolated. Remaining slots
    (if the gold neighbourhood is exhausted) are filled by a seeded random
    sample over the rest of the SKB. The corpus stays at CORPUS_SIZE so the
    silver KG is scale-comparable to the OpenIE KG (~1,250)."""
    gset = set(gold_nodes)
    rng = random.Random(SEED)
    # per-gold neighbour lists (exclude golds), seeded-shuffled for fair pick
    gold_nbrs = {}
    for g in sorted(gold_nodes):
        lst = sorted(nbr.get(g, ()) - gset)
        rng.shuffle(lst)
        gold_nbrs[g] = lst
    n_need = max(0, CORPUS_SIZE - len(gset))
    chosen = OrderedDict()
    pos = {g: 0 for g in gold_nbrs}
    progress = True
    while len(chosen) < n_need and progress:
        progress = False
        for g in sorted(gold_nbrs):
            lst, p = gold_nbrs[g], pos[g]
            while p < len(lst) and lst[p] in chosen:
                p += 1
            pos[g] = p
            if p < len(lst):
                chosen[lst[p]] = None
                pos[g] = p + 1
                progress = True
                if len(chosen) >= n_need:
                    break
    if len(chosen) < n_need:
        rest = sorted(set(node_info.keys()) - gset - set(chosen))
        for x in rng.sample(rest, min(n_need - len(chosen), len(rest))):
            chosen[x] = None
    return chosen


def build_split(split_qas, node_info, adj, nbr):
    # gold entity node ids in this split
    gold_nodes = set()
    for qa in split_qas:
        gold_nodes.update(qa["gold_ids"])

    # deterministic neighbour-priority distractors up to CORPUS_SIZE
    distractors = select_distractors(gold_nodes, node_info, nbr)
    corpus_nodes = sorted(gold_nodes | set(distractors))

    pid = assign_passage_ids(corpus_nodes, node_info)

    passages, manifest = [], []
    for nid in corpus_nodes:
        node = node_info[nid]
        text = entity_abstract(node)
        title = str(node.get("name") or pid[nid])
        passages.append({"passage_id": pid[nid], "paper_id": pid[nid],
                         "text": text, "title": title})
        manifest.append({"id": pid[nid], "paper_id": pid[nid], "title": title,
                         "abstract": text, "partition": "eval"})

    # qa rows in 2wiki schema
    qa_rows = []
    for qa in split_qas:
        gold_pids = [pid[i] for i in qa["gold_ids"]]
        hop = 2 if len(qa["gold_ids"]) >= 2 else 1
        # evidence_spans: the entity name span inside its own passage
        spans = []
        for i in qa["gold_ids"]:
            ptext = passages_lookup(passages, pid[i])
            nm = str(node_info[i].get("name") or "")
            lo = ptext.lower().find(nm.lower()) if nm else -1
            if lo >= 0:
                spans.append({"passage_id": pid[i], "span": [lo, lo + len(nm)]})
        qa_rows.append({
            "qa_id": "starkprime_" + str(qa["id"]),
            "question": qa["query"],
            "answer": str(node_info[qa["gold_ids"][0]].get("name") or ""),
            "answers": [str(node_info[i].get("name") or "")
                        for i in qa["gold_ids"]],
            "gold_paper_ids": gold_pids,
            "gold_entity_ids": qa["gold_ids"],
            "qtype": "stark_prime_entity",
            "hop_count": hop,
            "evidence_spans": spans,
            "source": "stark_prime",
        })

    # ---- S2 silver KG: the SKB subgraph INDUCED ON THE 1,250 CORPUS ENTITIES
    # Nodes are keyed by passage_id (n:: + normalized entity name), exactly the
    # 2Wiki/MetaQA convention, with source_docs = [own passage_id] so the graph
    # retriever can map every node to its corpus passage. Edges are restricted
    # to pairs whose BOTH endpoints are corpus entities -> the silver KG matches
    # the corpus scale (~1,250) instead of ballooning to the full SKB.
    corpus_set = set(corpus_nodes)
    nodes = OrderedDict()
    edges = OrderedDict()

    def node_key(nid):
        return "n::" + pid[nid]

    for nid in corpus_nodes:
        node = node_info[nid]
        key = node_key(nid)
        nodes[key] = {"node_id": key,
                      "name": str(node.get("name") or ""),
                      "type": str(node.get("type") or "entity"),
                      "aliases": entity_aliases(node),
                      "source_docs": [pid[nid]]}

    def add_edge(s_nid, rel, d_nid):
        s_key, d_key = node_key(s_nid), node_key(d_nid)
        eid = "%s|%s|%s" % (s_key, rel, d_key)
        if eid not in edges:
            edges[eid] = {"edge_id": eid, "source": s_key, "target": d_key,
                          "source_id": s_key, "target_id": d_key,
                          "relation": rel, "relation_type": rel,
                          "source_doc": pid[s_nid],
                          "reliability_score": 1.0}

    # induced subgraph: every SKB edge whose endpoints are both in the corpus,
    # materialized in both directions (STaRK PrimeKG is undirected).
    for s_nid in corpus_nodes:
        for (d_nid, rel) in sorted(adj.get(s_nid, []), key=lambda x: (x[1], x[0])):
            if d_nid in corpus_set and d_nid != s_nid:
                add_edge(s_nid, rel, d_nid)
                add_edge(d_nid, rel, s_nid)

    node_records = [nodes[k] for k in sorted(nodes)]
    edge_records = [edges[eid] for eid in sorted(edges)]

    # ---- coverage assert: every gold passage_id is covered by a silver node
    silver_pids = set()
    for n in node_records:
        for d in n["source_docs"]:
            silver_pids.add(d)
    for qa in qa_rows:
        for g in qa["gold_paper_ids"]:
            assert g in silver_pids, "silver coverage violated: %s" % g

    # ---- gold reachability stats over the induced subgraph
    silver_nbr = defaultdict(set)
    for e in edge_records:
        silver_nbr[e["source"]].add(e["target"])
    gold_nbr_counts = []
    isolated = 0
    for qa in qa_rows:
        for g in qa["gold_paper_ids"]:
            cnt = len(silver_nbr.get("n::" + g, set()))
            gold_nbr_counts.append(cnt)
            if cnt == 0:
                isolated += 1
    mean_nbr = (sum(gold_nbr_counts) / len(gold_nbr_counts)
                if gold_nbr_counts else 0.0)

    return {"qa": qa_rows, "passages": passages, "manifest": manifest,
            "kg_nodes": node_records, "kg_edges": edge_records,
            "n_gold_docs": len(gold_nodes), "n_corpus": len(corpus_nodes),
            "gold_isolated": isolated, "gold_mean_nbr": round(mean_nbr, 3),
            "n_gold_instances": len(gold_nbr_counts)}


def passages_lookup(passages, target_pid):
    for p in passages:
        if p["passage_id"] == target_pid:
            return p["text"]
    return ""


# ---------------------------------------------------------------------------
# Full build (pure modulo SKB load; run twice for determinism assert)
# ---------------------------------------------------------------------------
def build_all(node_info, adj, nbr):
    qa_rows = parse_qa(node_info)
    # size distribution of the eligible pool
    size_dist = defaultdict(int)
    for qa in qa_rows:
        size_dist[len(qa["gold_ids"])] += 1

    selected = stratified_select(qa_rows, node_info, 2 * N_PER_SPLIT)
    sub = selected[:N_PER_SPLIT]
    holdout = selected[N_PER_SPLIT:2 * N_PER_SPLIT]
    splits = {"sub200": build_split(sub, node_info, adj, nbr)}
    if len(holdout) >= N_PER_SPLIT:
        splits["holdout200"] = build_split(holdout, node_info, adj, nbr)

    strata = defaultdict(int)
    for qa in sub:
        strata[gold_stratum(qa, node_info)] += 1
    meta = {"pool": len(qa_rows), "seed": SEED, "max_gold": MAX_GOLD,
            "gold_size_dist_pool": dict(sorted(size_dist.items())),
            "sub200_stratum_dist": dict(sorted(strata.items())),
            "corpus_size": CORPUS_SIZE}
    return splits, meta


def serialize(splits):
    return json.dumps(splits, sort_keys=True, ensure_ascii=False)


def write_split(name, art):
    d = OUT_BASE / name
    (d / "kg_silver").mkdir(parents=True, exist_ok=True)
    with (d / "qa.jsonl").open("w", encoding="utf-8") as fh:
        for row in art["qa"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (d / "passages.jsonl").open("w", encoding="utf-8") as fh:
        for row in art["passages"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (d / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(art["manifest"], fh, ensure_ascii=False, indent=2)
    with (d / "kg_silver" / "proposed_kg_nodes.json").open("w", encoding="utf-8") as fh:
        json.dump(art["kg_nodes"], fh, ensure_ascii=False, indent=2)
    with (d / "kg_silver" / "proposed_kg_edges.json").open("w", encoding="utf-8") as fh:
        json.dump(art["kg_edges"], fh, ensure_ascii=False, indent=2)


def main():
    print("loading STaRK-Prime SKB (processed.zip artifacts) ...", flush=True)
    node_info, _etd, adj, nbr = load_skb()
    print("loaded %d nodes, adjacency for %d nodes" % (len(node_info), len(adj)),
          flush=True)

    print("building pass 1 ...", flush=True)
    splits1, meta = build_all(node_info, adj, nbr)
    print("building pass 2 (determinism check) ...", flush=True)
    splits2, _ = build_all(node_info, adj, nbr)
    assert serialize(splits1) == serialize(splits2), "non-deterministic build"
    print("determinism check passed", flush=True)

    for name, art in splits1.items():
        write_split(name, art)
        gsize = defaultdict(int)
        for qa in art["qa"]:
            gsize[len(qa["gold_paper_ids"])] += 1
        dist = " ".join("g%d=%d" % (k, v) for k, v in sorted(gsize.items()))
        print("%s: n_qa=%d gold_docs=%d corpus=%d kg_nodes=%d kg_edges=%d"
              % (name, len(art["qa"]), art["n_gold_docs"], art["n_corpus"],
                 len(art["kg_nodes"]), len(art["kg_edges"])), flush=True)
        print("  gold-size distribution: " + dist, flush=True)
        print("  gold reachability: instances=%d isolated=%d mean_silver_nbr=%.2f"
              % (art["n_gold_instances"], art["gold_isolated"],
                 art["gold_mean_nbr"]), flush=True)

    with (OUT_BASE / "build_report.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("pool=%d gold_size_dist=%s" % (meta["pool"],
          meta["gold_size_dist_pool"]), flush=True)
    print("sub200 stratum dist=%s" % meta["sub200_stratum_dist"], flush=True)

    log_progress("S1 subset built: pool=%d splits=%s seed=%d"
                 % (meta["pool"], ",".join(sorted(splits1)), SEED))
    log_progress("S2 silver KG REBUILT (corpus-induced, passage-keyed): "
                 + " ".join(
        "%s nodes=%d edges=%d isolated=%d mean_nbr=%.1f"
        % (n, len(a["kg_nodes"]), len(a["kg_edges"]), a["gold_isolated"],
           a["gold_mean_nbr"])
        for n, a in sorted(splits1.items()))
                 + " coverage=1.0 (asserted)")
    log_progress("READY FOR S3(OpenIE build via new_env)")
    return 0


if __name__ == "__main__":
    if "--force-name-keyed" not in sys.argv:
        print("REFUSING to run: this name-keyed builder is SUPERSEDED by "
              "rebuild_stark_silver_rr.py (entity-id-keyed). Pass "
              "--force-name-keyed only if you really want the legacy scheme.",
              flush=True)
        sys.exit(2)
    sys.exit(main())
