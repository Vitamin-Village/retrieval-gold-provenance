# -*- coding: utf-8 -*-
"""Realistic aligned structure for 2WikiMultiHop: a Wikidata KG over the corpus
entities, NOT induced from the evaluated questions' evidence triples.

For each split: resolve every corpus passage title to a Wikidata QID
(wbsearchentities), fetch the entity's claims (wbgetentities), keep every
entity-valued statement, label object entities and properties, and write a KG in
the kg_silver schema. Node -> passage index: a corpus entity maps to its own passage;
an object entity maps to the corpus passages whose subjects link to it. The index
therefore covers all corpus passages (gold fraction ~0.31, like the OpenIE KG),
while the structure is the same Wikidata the benchmark's gold triples came from.

Cache: external/2wiki/_wikidata_cache/*.json (resumable). Polite rate: ~3 req/s, UA set.
Outputs: external/2wiki/<split>/kg_wikidata/{proposed_kg_nodes,proposed_kg_edges,build_summary}.json
Then run: python paper/eval/_kg_swap_variant.py --bench 2wiki --variant kg_wikidata
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "paper" / "external" / "2wiki"
CACHE = EXT / "_wikidata_cache"
UA = "provenance-audit/0.1 (academic research; contact via OpenReview)"
API = "https://www.wikidata.org/w/api.php"
_WS = re.compile(r"\s+")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def norm(s):
    return _WS.sub(" ", s.strip()).lower()


def get(params, retries=6):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            time.sleep(60 * (i + 1) if e.code == 429 else 5 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"wikidata failed: {last}")


def cached(name):
    p = CACHE / f"{name}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def save(name, obj):
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(CACHE / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False)


def resolve_titles(titles):
    res = cached("title2qid")
    todo = [t for t in titles if t not in res]
    for i, t in enumerate(todo, 1):
        d = get({"action": "wbsearchentities", "search": t, "language": "en", "limit": 3, "type": "item"})
        hit = None
        for h in d.get("search", []):
            if norm(h.get("label", "")) == norm(_PAREN.sub("", t)) or norm(h.get("label", "")) == norm(t):
                hit = h
                break
        if hit is None and d.get("search"):
            hit = d["search"][0]
        res[t] = {"qid": hit["id"], "label": hit.get("label", ""), "exact": bool(hit and norm(hit.get("label", "")) in (norm(t), norm(_PAREN.sub("", t))))} if hit else None
        if i % 50 == 0:
            save("title2qid", res)
            print(f"  resolved {i}/{len(todo)}", flush=True)
        time.sleep(1.0)
    save("title2qid", res)
    return res


def fetch_entities(qids, props):
    key = "entities_" + props.replace("|", "_")
    res = cached(key)
    todo = [q for q in qids if q not in res]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        d = get({"action": "wbgetentities", "ids": "|".join(batch), "props": props, "languages": "en"})
        for q, e in d.get("entities", {}).items():
            res[q] = e
        if (i // 50) % 10 == 0:
            save(key, res)
            print(f"  fetched {min(i + 50, len(todo))}/{len(todo)} ({props})", flush=True)
        time.sleep(1.0)
    save(key, res)
    return res


def build(split):
    d = EXT / split
    passages = [json.loads(l) for l in open(d / "passages.jsonl", encoding="utf-8")]
    titles = [p["title"] for p in passages]
    t2q = resolve_titles(titles)
    pid_of_qid = {}
    for p in passages:
        r = t2q.get(p["title"])
        if r:
            pid_of_qid.setdefault(r["qid"], p["passage_id"])
    subj_qids = sorted(pid_of_qid)
    ents = fetch_entities(subj_qids, "claims|labels|aliases")
    triples = []
    obj_qids, pids_used = set(), set()
    for q in subj_qids:
        e = ents.get(q, {})
        for pid, claims in (e.get("claims") or {}).items():
            for c in claims:
                ms = c.get("mainsnak", {})
                if ms.get("snaktype") != "value" or ms.get("datatype") != "wikibase-item":
                    continue
                o = ms["datavalue"]["value"]["id"]
                triples.append((q, pid, o))
                obj_qids.add(o)
                pids_used.add(pid)
    labels = fetch_entities(sorted(obj_qids - set(subj_qids)), "labels|aliases")
    plabels = fetch_entities(sorted(pids_used), "labels")

    def label(q):
        e = ents.get(q) or labels.get(q) or {}
        return (e.get("labels", {}).get("en", {}) or {}).get("value", q)

    def aliases(q):
        e = ents.get(q) or labels.get(q) or {}
        return [a["value"] for a in (e.get("aliases", {}).get("en", []) or [])][:10]

    nodes, edges = OrderedDict(), OrderedDict()
    links = defaultdict(set)
    for q in subj_qids:
        nodes[q] = {"node_id": "n::" + norm(label(q)), "name": label(q), "type": "entity", "aliases": [label(q)] + aliases(q), "source_docs": [pid_of_qid[q]], "qid": q}
    for s, p, o in triples:
        if o not in nodes:
            nodes[o] = {"node_id": "n::" + norm(label(o)), "name": label(o), "type": "entity", "aliases": [label(o)] + aliases(o), "source_docs": [], "qid": o}
        links[o].add(pid_of_qid[s])
        rel = (plabels.get(p, {}).get("labels", {}).get("en", {}) or {}).get("value", p)
        for a, b, r in ((nodes[s]["node_id"], nodes[o]["node_id"], rel), (nodes[o]["node_id"], nodes[s]["node_id"], rel + "_inv")):
            eid = f"{a}|{r}|{b}"
            if eid not in edges:
                edges[eid] = {"edge_id": eid, "source": a, "target": b, "source_id": a, "target_id": b, "relation": r, "relation_type": r, "source_doc": "", "reliability_score": 1.0}
    for q, n in nodes.items():
        if not n["source_docs"]:
            n["source_docs"] = sorted(links.get(q, ()))
    # merge nodes that share a node_id (same label) by unioning aliases/source_docs
    merged = OrderedDict()
    for n in nodes.values():
        m = merged.get(n["node_id"])
        if m is None:
            merged[n["node_id"]] = {k: (list(v) if isinstance(v, list) else v) for k, v in n.items() if k != "qid"}
        else:
            for k in ("aliases", "source_docs"):
                m[k] = sorted(set(m[k]) | set(n[k]))
    out = d / "kg_wikidata"
    out.mkdir(exist_ok=True)
    json.dump(list(merged.values()), open(out / "proposed_kg_nodes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(list(edges.values()), open(out / "proposed_kg_edges.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    summ = {"split": split, "n_passages": len(passages), "n_resolved": len(pid_of_qid), "n_exact_label_match": sum(1 for t in titles if t2q.get(t) and t2q[t]["exact"]),
            "n_triples": len(triples), "n_nodes": len(merged), "n_edges": len(edges), "n_object_entities": len(obj_qids)}
    json.dump(summ, open(out / "build_summary.json", "w", encoding="utf-8"), indent=1)
    print(split, summ, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for s in ("sub200", "holdout200"):
        build(s)
