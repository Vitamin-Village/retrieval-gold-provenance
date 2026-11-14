# -*- coding: utf-8 -*-
"""Round-robin reachability rebuild of the STaRK-Prime KG-swap corpus + silver KG.

Refines build_stark_swap_data_v2.py: that version keyed everything by entity id
(node_id "n::e<id>", source_docs ["e<id>"]) -- which correctly maps every node
to its passage and matches the corpus scale (1,250) -- but drew distractors by
random sampling over the UNION of the gold entities' 1-hop neighbours, leaving
~80-90 gold entities with ZERO corpus neighbour (isolated in the induced
subgraph: 92 in sub200, 68 in holdout200). An isolated gold cannot be reached
by any graph walk, so the silver "circular" condition silently loses those
golds.

This script keeps v2's entity-id keying and every output schema byte-for-byte,
and changes ONLY the distractor selection to a deterministic PER-GOLD ROUND
ROBIN: each gold contributes its (seeded-shuffled) neighbours in turn so every
gold receives at least one corpus neighbour before any gold receives a second.
Result: 0 isolated golds in both splits at the same 1,250 corpus size.

The 200 selected questions per split are PRESERVED (read from the existing
qa.jsonl gold_entity_ids) so the subset stays frozen; only the corpus,
passages, manifest, and silver KG are rebuilt.

ASCII-only stdout. Run with the torch env:
    conda run -n new_env python paper/eval/rebuild_stark_silver_rr.py
"""
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
STARK = ROOT / "paper" / "external" / "stark"
PROC = STARK / "raw" / "processed"
PROGRESS = STARK / "_PROGRESS.log"
SEED = 20260615
CORPUS_N = 1250

# Type-aware prose fields (identical to v2) so passages stay clean biomedical
# prose suitable for OpenIE and retrieval; everything else is dropped.
PROSE_FIELDS = {
    "gene/protein": ["name", "summary"],
    "drug": ["description", "indication", "mechanism_of_action",
             "pharmacodynamics"],
    "disease": ["mondo_definition", "umls_description", "orphanet_definition"],
}


def pid(eid):
    return "e%d" % eid


def load_skb():
    ni = pickle.load(open(PROC / "node_info.pkl", "rb"))
    ei = torch.load(PROC / "edge_index.pt", weights_only=False)
    et = torch.load(PROC / "edge_types.pt", weights_only=False)
    etd = pickle.load(open(PROC / "edge_type_dict.pkl", "rb"))
    return ni, ei, et, etd


def build_neighbour_map(ei):
    """Relation-agnostic undirected neighbour sets for every node."""
    src, dst = ei[0].tolist(), ei[1].tolist()
    nbr = defaultdict(set)
    for s, d in zip(src, dst):
        if s != d:
            nbr[s].add(d)
            nbr[d].add(s)
    return nbr


def induced_edges(ei, et, etd, corpus_ids):
    cset = torch.tensor(sorted(corpus_ids), dtype=ei.dtype)
    src, dst = ei[0], ei[1]
    mask = torch.isin(src, cset) & torch.isin(dst, cset)
    s, d, t = src[mask].tolist(), dst[mask].tolist(), et[mask].tolist()
    rows, seen = [], set()
    for a, b, rt in zip(s, d, t):
        if a == b:
            continue
        rel = etd.get(rt, str(rt))
        key = "%s|%s|%s" % (a, rel, b)
        if key in seen:
            continue
        seen.add(key)
        rows.append((a, rel, b))
    return rows


def entity_text(node):
    name = str(node.get("name", "")).strip()
    typ = str(node.get("type", "")).strip()
    head = "%s (%s)." % (name, typ) if typ else "%s." % name
    det = node.get("details")
    parts = []
    if isinstance(det, dict):
        for f in PROSE_FIELDS.get(typ, []):
            v = det.get(f)
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() == "nan" or s == name:
                continue
            parts.append(s)
    body = " ".join(parts).strip()
    return (head + " " + body).strip() if body else head


def select_corpus(gold_set, nbr, all_ids, seed):
    """Per-gold round-robin neighbour allocation, then deterministic random fill.

    Guarantees every gold gets a corpus neighbour (when it has any SKB
    neighbour) before any gold gets a second -> 0 isolated golds."""
    rng = random.Random(seed)
    gold_sorted = sorted(gold_set)
    gold_nbrs = {}
    for g in gold_sorted:
        lst = sorted(nbr.get(g, set()) - gold_set)
        rng.shuffle(lst)
        gold_nbrs[g] = lst
    corpus = list(gold_sorted)  # all golds always in corpus
    chosen = set(corpus)
    n_need = CORPUS_N - len(corpus)
    pos = {g: 0 for g in gold_sorted}
    added = []
    progress = True
    while len(added) < n_need and progress:
        progress = False
        for g in gold_sorted:
            lst, p = gold_nbrs[g], pos[g]
            while p < len(lst) and lst[p] in chosen:
                p += 1
            pos[g] = p
            if p < len(lst):
                x = lst[p]
                chosen.add(x)
                added.append(x)
                pos[g] = p + 1
                progress = True
                if len(added) >= n_need:
                    break
    if len(added) < n_need:
        rest = [n for n in sorted(all_ids) if n not in chosen]
        rng.shuffle(rest)
        for x in rest:
            if len(added) >= n_need:
                break
            chosen.add(x)
            added.append(x)
    corpus = corpus + added
    corpus = corpus[:CORPUS_N]
    assert gold_set <= set(corpus), "gold not fully in corpus"
    return corpus


def build_split(split, ni, ei, et, etd, nbr, all_ids):
    d = STARK / split
    qa = [json.loads(l) for l in open(d / "qa.jsonl", encoding="utf-8")]
    gold_set = set()
    for q in qa:
        gold_set.update(int(g) for g in q["gold_entity_ids"])

    corpus = select_corpus(gold_set, nbr, all_ids, SEED)
    cset = set(corpus)

    # passages (clean prose, entity-id keyed) -- identical schema to v2
    with open(d / "passages.jsonl", "w", encoding="utf-8") as fh:
        for eid in corpus:
            node = ni[eid]
            rec = {"passage_id": pid(eid), "paper_id": pid(eid),
                   "entity_id": eid, "title": str(node.get("name", "")),
                   "text": entity_text(node)}
            fh.write(json.dumps(rec, ensure_ascii=True) + "\n")

    # qa rows preserved (gold ids re-emitted from frozen gold_entity_ids)
    with open(d / "qa.jsonl", "w", encoding="utf-8") as fh:
        for q in qa:
            gpaper = [pid(int(g)) for g in q["gold_entity_ids"]]
            rec = {"qa_id": q["qa_id"], "question": q["question"],
                   "answer": q.get("answer"), "answers": q.get("answers"),
                   "gold_paper_ids": gpaper,
                   "gold_entity_ids": [int(g) for g in q["gold_entity_ids"]],
                   "qtype": q.get("qtype", "stark_prime_entity"),
                   "hop_count": q.get("hop_count", 1), "source": "STaRK-Prime"}
            fh.write(json.dumps(rec, ensure_ascii=True) + "\n")

    manifest = []
    for eid in corpus:
        node = ni[eid]
        manifest.append({"id": pid(eid), "paper_id": pid(eid),
                         "title": str(node.get("name", "")),
                         "abstract": entity_text(node), "partition": "eval"})
    json.dump(manifest, open(d / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=True)

    kg = d / "kg_silver"
    kg.mkdir(parents=True, exist_ok=True)
    nodes = []
    for eid in corpus:
        node = ni[eid]
        name = str(node.get("name", ""))
        nodes.append({"node_id": "n::" + pid(eid), "name": name,
                      "type": str(node.get("type", "")),
                      "aliases": [name], "source_docs": [pid(eid)]})
    json.dump(nodes, open(kg / "proposed_kg_nodes.json", "w", encoding="utf-8"),
              ensure_ascii=True)

    edges = induced_edges(ei, et, etd, cset)
    erows = []
    for a, rel, b in edges:
        erows.append({"edge_id": "n::%s|%s|n::%s" % (pid(a), rel, pid(b)),
                      "source": "n::" + pid(a), "target": "n::" + pid(b),
                      "source_id": "n::" + pid(a), "target_id": "n::" + pid(b),
                      "relation": rel, "relation_type": rel,
                      "source_doc": "", "reliability_score": 1.0})
    json.dump(erows, open(kg / "proposed_kg_edges.json", "w", encoding="utf-8"),
              ensure_ascii=True)

    # reachability report (undirected silver degree per gold)
    deg = defaultdict(int)
    for e in erows:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    gold_deg = [deg.get("n::" + pid(g), 0) for g in sorted(gold_set)]
    iso = sum(1 for x in gold_deg if x == 0)
    mean_deg = sum(gold_deg) / max(1, len(gold_deg))
    print("%-10s corpus=%d silver_nodes=%d silver_edges=%d gold=%d "
          "gold_isolated=%d mean_gold_deg=%.2f" % (
              split, len(corpus), len(nodes), len(erows), len(gold_set),
              iso, mean_deg), flush=True)
    return {"corpus": len(corpus), "nodes": len(nodes), "edges": len(erows),
            "gold": len(gold_set), "isolated": iso, "mean_deg": round(mean_deg, 2)}


def main():
    ni, ei, et, etd = load_skb()
    print("loaded skb: nodes=%d edges=%d" % (len(ni), ei.shape[1]), flush=True)
    nbr = build_neighbour_map(ei)
    all_ids = list(ni.keys())
    rep = {}
    for split in ("sub200", "holdout200"):
        rep[split] = build_split(split, ni, ei, et, etd, nbr, all_ids)
    ts = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write("%s S2 silver KG RR-REBUILT (entity-id-keyed, corpus 1250, "
                 "round-robin distractors): %s coverage=1.0; READY FOR "
                 "S3(OpenIE via new_env)\n" % (ts, " ".join(
                     "%s nodes=%d edges=%d isolated=%d mean_deg=%.1f"
                     % (s, r["nodes"], r["edges"], r["isolated"], r["mean_deg"])
                     for s, r in sorted(rep.items()))))
    print("done", flush=True)


if __name__ == "__main__":
    main()
