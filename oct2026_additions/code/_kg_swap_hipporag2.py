# -*- coding: utf-8 -*-
"""
KG swap replicated with a published graph-RAG system: HippoRAG 2 (OSU-NLP-Group,
pip package ``hipporag`` 2.0.0a4) on 2WikiMultiHop, sub200 / holdout200.

Same questions, same candidate pool, same embedding model, same LLM, same HippoRAG
config; the ONLY change is the knowledge graph HippoRAG indexes:
  - kg_silver : Wikidata evidence triples the 2Wiki gold is derived from (aligned)
  - kg_openie : Llama-3.1-8B OpenIE KG over the passage text, no gold access (independent)

Injection. HippoRAG 2 ``index(docs)`` runs NER + triple extraction per passage and caches
the result in ``<save_dir>/openie_results_ner_<llm>.json`` keyed by
``compute_mdhash_id(passage_text, "chunk-")``; cached passages are never re-extracted.
We write that cache ourselves before ``index()``: for passage p the triples are every KG
edge (s, r, o) with p in source_docs(s) | source_docs(o) (node -> passage provenance, the
same construction for both KGs), the entities are the endpoint names of those triples.
Passages with no linked edge get empty lists. ``openie.batch_openie`` is replaced by a
function that raises, so any attempt to run LLM OpenIE aborts the run.

Everything downstream is HippoRAG's default retrieval: query-to-fact scoring, the DSPy
recognition-memory fact filter (LLM), phrase + passage node weighting (passage_node_weight
0.05), PPR (damping 0.5); dense-passage fallback when the filter keeps no fact.

Run in conda env rag-graphrag-py311 from paper/ (PYTHONUTF8=1, HF_HUB_OFFLINE=1):
    python eval/_kg_swap_hipporag2.py --split sub200 --kg kg_silver --limit 10
    python eval/_kg_swap_hipporag2.py --split sub200 --kg kg_silver
    python eval/_kg_swap_hipporag2.py --bench metaqa --split sub200 --kg kg_fullkb
Other benchmarks (--bench metaqa / graphragbench; kg_wikidata, kg_fullkb) use the identical
config and injection. Outputs under results/hipporag2_swap/runs/<split>__<kg>/ (2Wiki) or
runs/<bench>__<split>__<kg>/:
    save_dir/            HippoRAG state (openie cache, embeddings, graph.pickle, llm_cache)
    retrieval[_limN].jsonl  per-question ranked passage ids + fact-filter diagnostics
    run_meta[_limN].json    config, versions, injection + graph verification, timings
Scoring / the _kgswap_hipporag2_*.json vectors: eval/_kg_swap_hipporag2_score.py.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # paper/
OUT = ROOT / "results" / "hipporag2_swap" / "runs"
CLONE = ROOT.parent / "analysis" / "hipporag_py311_probe" / "HippoRAG"

# HippoRAG builds file names from llm_name and ':' is illegal in Windows paths, so HippoRAG gets
# a ':'-free label; every request is sent to the Ollama tag llama3.1:8b itself (LLM_API_MODEL).
# (A derived tag with its own num_ctx made Ollama swap runners with a concurrent llama3.1:8b job.)
LLM_NAME = "hipporag-llama3.1-8b"
LLM_API_MODEL = "llama3.1:8b"
LLM_BASE_URL = "http://localhost:11434/v1"
EMBEDDING = "Transformers/BAAI/bge-m3"


def load_jsonl(path: Path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def doc_string(p) -> str:
    # HippoRAG's own 2Wiki corpus format (main.py: f"{doc['title']}\n{doc['text']}")
    return f"{p['title']}\n{p['text']}"


def build_injection(passages, kg_dir: Path, provenance: str = "node"):
    """Per-passage triples. provenance="node" (the protocol, both KGs): edge (s, r, o) goes to
    every passage in source_docs(s) | source_docs(o). provenance="edge" (sensitivity, OpenIE KG
    only): edge goes only to the passages it was extracted from (edge source_docs); edges without
    extraction provenance (the KG builder's surface-synonym edges) are dropped."""
    nodes = {n["node_id"]: n for n in json.load(open(kg_dir / "proposed_kg_nodes.json", encoding="utf-8"))}
    edges = json.load(open(kg_dir / "proposed_kg_edges.json", encoding="utf-8"))
    pool = {p["passage_id"] for p in passages}
    triples = collections.defaultdict(list)
    for e in edges:
        s, o = nodes[e["source"]], nodes[e["target"]]
        if provenance == "node":
            where = set(s.get("source_docs") or []) | set(o.get("source_docs") or [])
        else:
            where = set(e.get("source_docs") or [])
        for pid in sorted(where):
            if pid in pool:
                triples[pid].append([s["name"], e["relation"], o["name"]])
    docs = []
    for p in passages:
        t = triples.get(p["passage_id"], [])
        ents = sorted({x for tr in t for x in (tr[0], tr[2])})
        docs.append({"passage_id": p["passage_id"], "passage": doc_string(p),
                     "extracted_entities": ents, "extracted_triples": t})
    return docs, {"kg_nodes": len(nodes), "kg_edges": len(edges), "provenance": provenance}


def write_openie_cache(path: Path, docs, mdhash):
    rows = [{"idx": mdhash(d["passage"], "chunk-"), "passage": d["passage"],
             "extracted_entities": d["extracted_entities"], "extracted_triples": d["extracted_triples"]}
            for d in docs]
    ents = [e for r in rows for e in r["extracted_entities"]]
    payload = {"docs": rows,
               "avg_ent_chars": round(sum(len(e) for e in ents) / len(ents), 4) if ents else 0,
               "avg_ent_words": round(sum(len(e.split()) for e in ents) / len(ents), 4) if ents else 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return payload


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def clone_state():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(CLONE), *a], capture_output=True, text=True,
                                  encoding="utf-8", timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"<git failed: {e}>"
    return {"path": str(CLONE), "head": git("rev-parse", "HEAD"),
            "modified_files": git("status", "--short", "--untracked-files=no").splitlines()}


def wait_for_llm_slot(log, poll=60, quiet_polls=3, max_hours=10.0):
    """Hold the (LLM-bound) retrieval while Ollama is actively serving another model. On the shared
    GPU llama3.1:8b cannot co-reside with the large reader models, so interleaving would make Ollama
    swap runners on every call. "Active" = another loaded model's expires_at (refreshed on every
    request) changed during the last `quiet_polls` polls. Returns the seconds waited."""
    import urllib.request
    t0, last, quiet = time.time(), {}, 0
    while time.time() - t0 < max_hours * 3600:
        try:
            with urllib.request.urlopen(LLM_BASE_URL.replace("/v1", "/api/ps"), timeout=30) as r:
                models = json.load(r).get("models", [])
        except Exception:  # noqa: BLE001
            models = []
        now = {m["name"]: m.get("expires_at") for m in models if m["name"] != LLM_API_MODEL}
        changed = any(last.get(k) != v for k, v in now.items() if k in last) or not set(now) <= set(last)
        quiet = 0 if (now and changed) else quiet + 1
        last = now
        if quiet >= quiet_polls:
            return time.time() - t0
        if quiet == 0 and int(time.time() - t0) % (30 * poll) < poll:
            log.info("waiting for Ollama: other active model(s) %s", sorted(now))
        time.sleep(poll)
    log.warning("LLM gate timed out after %.1f h; proceeding", max_hours)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="2wiki", choices=["2wiki", "metaqa", "graphragbench"])
    ap.add_argument("--split", required=True, choices=["sub200", "holdout200"])
    ap.add_argument("--kg", required=True, choices=["kg_silver", "kg_openie", "kg_wikidata", "kg_fullkb"])
    ap.add_argument("--limit", type=int, default=0, help="first N questions (quick test); index is always the full pool")
    ap.add_argument("--top-n", type=int, default=20, help="ranked passages stored per question (hit@5 uses the first 5)")
    ap.add_argument("--rebuild", action="store_true", help="delete an existing save_dir first")
    ap.add_argument("--provenance", choices=["node", "edge"], default="node",
                    help="node = protocol; edge = sensitivity (OpenIE triples only at their source passage)")
    ap.add_argument("--index-only", action="store_true", help="build + verify the index, then stop before retrieval")
    ap.add_argument("--no-llm-gate", action="store_true", help="do not wait for other Ollama models to go idle")
    ap.add_argument("--no-chunk-reuse", action="store_true", help="always encode passages instead of reusing the sibling condition's vectors")
    args = ap.parse_args()

    if args.provenance == "edge" and args.kg != "kg_openie":
        sys.exit("--provenance edge applies to kg_openie only (silver edges carry no extraction provenance)")
    cond = args.kg + ("" if args.provenance == "node" else "__edgeprov")
    # 2Wiki keeps its original run-dir names; other benchmarks are prefixed with the benchmark
    run_dir = OUT / (f"{args.split}__{cond}" if args.bench == "2wiki" else f"{args.bench}__{args.split}__{cond}")
    save_dir = run_dir / "save_dir"
    tag = f"_lim{args.limit}" if args.limit else ""
    if args.rebuild and save_dir.exists():
        shutil.rmtree(save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / f"hipporag{tag}.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("kgswap_hipporag2")

    import hipporag
    from importlib.metadata import version as pkg_version
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
    from hipporag.utils.misc_utils import compute_mdhash_id, text_processing
    from hipporag.utils.llm_utils import filter_invalid_triples

    d = ROOT / "external" / args.bench / args.split
    passages = load_jsonl(d / "passages.jsonl")
    qa = load_jsonl(d / "qa.jsonl")
    gold_all = {g for q in qa for g in q["gold_paper_ids"]}
    if args.limit:
        qa = qa[: args.limit]
    docs_inj, kg_info = build_injection(passages, d / args.kg, args.provenance)
    doc_texts = [x["passage"] for x in docs_inj]
    text_to_pid = {x["passage"]: x["passage_id"] for x in docs_inj}
    assert len(text_to_pid) == len(passages), "passage strings are not unique"

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name=LLM_NAME,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=EMBEDDING,
    )
    inj_fp = fingerprint({"docs": docs_inj, "llm": LLM_NAME, "emb": EMBEDDING})
    fp_path = save_dir / "injection_fingerprint.txt"
    openie_path = save_dir / f"openie_results_ner_{LLM_NAME.replace('/', '_')}.json"
    work = f"{LLM_NAME.replace('/', '_')}_{EMBEDDING.replace('/', '_')}"
    index_copied_from = None
    if not fp_path.exists() and not (save_dir.exists() and any(save_dir.iterdir())):
        # An identical injection already indexed for another split (same corpus + same KG file,
        # e.g. GraphRAG-Bench kg_openie): copy that built index instead of rebuilding it. The graph
        # verification below re-checks it against this run's injection.
        for other_fp in sorted(OUT.glob("*/save_dir/injection_fingerprint.txt")):
            other = other_fp.parent
            if other != save_dir and (other / "build_graph_info.json").exists() \
                    and other_fp.read_text(encoding="utf-8").strip() == inj_fp:
                shutil.copytree(other, save_dir, ignore=shutil.ignore_patterns("llm_cache"), dirs_exist_ok=True)
                index_copied_from = str(other)
                log.info("copied identical index from %s", other)
                break
    fresh = not fp_path.exists()
    chunk_reuse = None
    if not fresh:
        if fp_path.read_text(encoding="utf-8").strip() != inj_fp:
            sys.exit(f"{save_dir} was built from a different injection; rerun with --rebuild")
        if not (save_dir / "build_graph_info.json").exists():
            sys.exit(f"{save_dir} has a fingerprint but no completed build (another process may be building it)")
        log.info("reusing existing index in %s (fingerprint match)", save_dir)
    else:
        if save_dir.exists() and any(save_dir.iterdir()):
            sys.exit(f"{save_dir} exists without fingerprint; rerun with --rebuild")
        write_openie_cache(openie_path, docs_inj, compute_mdhash_id)
        fp_path.write_text(inj_fp, encoding="utf-8")
        # Passage vectors do not depend on the KG. If another run already encoded exactly this set
        # of passage strings with the same model, seed HippoRAG's chunk store with that parquet
        # (HippoRAG then encodes only missing rows = none). Saves ~20-30 min CPU per condition.
        if not args.no_chunk_reuse:
            import pandas as pd
            want = {compute_mdhash_id(t, "chunk-"): t for t in doc_texts}
            for sib in sorted(OUT.glob(f"*/save_dir/{work}/chunk_embeddings/vdb_chunk.parquet")):
                if sib.parents[3] == save_dir.parent:
                    continue
                df = pd.read_parquet(sib, columns=["hash_id", "content"])
                if len(df) == len(want) and dict(zip(df["hash_id"], df["content"])) == want:
                    dst = save_dir / work / "chunk_embeddings" / "vdb_chunk.parquet"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(sib, dst)
                    chunk_reuse = str(sib)
                    log.info("seeded chunk embeddings from %s", sib)
                    break

    rag = HippoRAG(global_config=config)
    assert Path(rag.openie_results_path) == openie_path, (rag.openie_results_path, openie_path)
    rag.llm_model.llm_config.generate_params["model"] = LLM_API_MODEL
    rag.rerank_filter.model_name = LLM_API_MODEL  # DSPyFilter passes model= explicitly

    def _forbidden_openie(*a, **k):
        raise RuntimeError("LLM OpenIE was invoked for a passage missing from the injected cache")
    rag.openie.batch_openie = _forbidden_openie

    # every pool passage must already be in the injected cache -> nothing left for LLM OpenIE
    cached = {r["idx"] for r in json.load(open(openie_path, encoding="utf-8"))["docs"]}
    missing = [t for t in doc_texts if compute_mdhash_id(t, "chunk-") not in cached]
    assert not missing, f"{len(missing)} passages missing from the injected OpenIE cache"

    t0 = time.time()
    rag.index(doc_texts)
    t_index = time.time() - t0

    # ---- verify the graph HippoRAG built contains exactly the injected triples ----------
    exp_by_pid = {}
    for x in docs_inj:
        tr = [text_processing(t) for t in filter_invalid_triples(x["extracted_triples"])]
        exp_by_pid[x["passage_id"]] = tr
    exp_facts = {tuple(t) for tr in exp_by_pid.values() for t in tr}
    exp_ents = {e for f in exp_facts for e in (f[0], f[2])}
    got_ents = set(rag.entity_embedding_store.get_all_texts())
    got_facts = set(rag.fact_embedding_store.get_all_texts())
    g = rag.graph
    names = g.vs["name"]
    contents = g.vs["content"]
    passage_keys = set(rag.chunk_embedding_store.get_all_ids())
    entity_keys = set(rag.entity_embedding_store.get_all_ids())
    # passage -> entity neighbours in the built graph vs the injection
    pe_mismatch, linked_pids = 0, set()
    for v in g.vs:
        if v["name"] not in passage_keys:
            continue
        nb = {contents[u] for u in g.neighbors(v.index) if names[u] in entity_keys}
        pid = text_to_pid[v["content"]]
        exp = {e for t in exp_by_pid[pid] for e in (t[0], t[2])}
        if nb:
            linked_pids.add(pid)
        pe_mismatch += nb != exp
    etype = collections.Counter()
    for e in g.es:
        a, b = names[e.source], names[e.target]
        etype["passage-entity" if (a in passage_keys) != (b in passage_keys) else
              ("passage-passage" if a in passage_keys else "entity-entity")] += 1
    ent_ent_fact_pairs = set()
    for f in exp_facts:
        a, b = compute_mdhash_id(f[0], "entity-"), compute_mdhash_id(f[2], "entity-")
        if a != b:
            ent_ent_fact_pairs.add(frozenset((a, b)))
    # get_graph_info() reads node_to_node_stats, which index() only fully rebuilds on a fresh
    # build (passage/synonymy edges are skipped for chunks already in graph.pickle): keep the
    # build-time numbers. The structural checks below read graph.pickle and are rerun every time.
    gi_path = save_dir / "build_graph_info.json"
    if fresh:
        graph_info = rag.get_graph_info()
        json.dump(graph_info, open(gi_path, "w", encoding="utf-8"), indent=1)
    else:
        graph_info = json.load(open(gi_path, encoding="utf-8"))
    verify = {
        "expected_unique_facts": len(exp_facts),
        "graph_fact_store_size": len(got_facts),
        "facts_equal": got_facts == {str(f) for f in exp_facts},
        "expected_unique_entities": len(exp_ents),
        "graph_entity_nodes": len(got_ents),
        "entities_equal": got_ents == exp_ents,
        "passage_nodes": len(passage_keys),
        "graph_vcount": g.vcount(),
        "graph_ecount": g.ecount(),
        "edges_by_type": dict(etype),
        "entity_pairs_from_facts": len(ent_ent_fact_pairs),
        "passages_linked_in_graph": len(linked_pids),
        "gold_passages_in_split": len(gold_all),
        "gold_passages_linked": len(linked_pids & gold_all),
        "gold_fraction_of_linked": len(linked_pids & gold_all) / len(linked_pids) if linked_pids else None,
        "passage_entity_neighbourhood_mismatches": pe_mismatch,
        "hipporag_graph_info": graph_info,
        "empty_entity_strings": sum(1 for e in exp_ents if e == ""),
    }
    log.info("verification: %s", json.dumps(verify))
    assert verify["facts_equal"] and verify["entities_equal"] and pe_mismatch == 0, "graph != injection"
    assert g.vcount() == len(passage_keys) + len(entity_keys)
    if args.index_only:
        log.info("INDEX-ONLY DONE %s %s %s", args.bench, args.split, cond)
        return
    llm_wait = 0.0 if args.no_llm_gate else wait_for_llm_slot(log)

    # ---- retrieval (default HippoRAG 2 pipeline), with fact-filter diagnostics ------------
    diag = []
    orig_rerank = rag.rerank_facts

    def rerank_logged(query, scores):
        idx, facts, rlog = orig_rerank(query, scores)
        diag.append({"facts_before_filter": [list(f) for f in rlog.get("facts_before_rerank", [])],
                     "facts_after_filter": [list(f) for f in facts], "filter_error": rlog.get("error")})
        return idx, facts, rlog
    rag.rerank_facts = rerank_logged

    t0 = time.time()
    sols = rag.retrieve(queries=[q["question"] for q in qa], num_to_retrieve=args.top_n)
    t_retr = time.time() - t0
    assert len(sols) == len(qa) == len(diag)
    # KG-independent reference: HippoRAG's own dense passage channel alone (same BGE-M3 vectors)
    dpr = rag.retrieve_dpr(queries=[q["question"] for q in qa], num_to_retrieve=args.top_n)

    rows = []
    for q, s, dg, ds in zip(qa, sols, diag, dpr):
        ids = [text_to_pid[t] for t in s.docs]
        rows.append({"qa_id": q["qa_id"], "question": q["question"], "gold_paper_ids": q["gold_paper_ids"],
                     "ranked_passage_ids": ids, "ranked_scores": [float(x) for x in s.doc_scores],
                     "dpr_ranked_passage_ids": [text_to_pid[t] for t in ds.docs],
                     "path": "ppr" if dg["facts_after_filter"] else "dpr_fallback", **dg})
    with open(run_dir / f"retrieval{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    cache_db = Path(rag.llm_model.cache_file_name)
    n_llm = sqlite3.connect(cache_db).execute("SELECT COUNT(*) FROM cache").fetchone()[0] if cache_db.exists() else 0
    hit5 = sum(all(gid in r["ranked_passage_ids"][:5] for gid in r["gold_paper_ids"]) for r in rows) / len(rows)
    cfg = {k: v for k, v in asdict(config).items()}
    meta = {
        "bench": args.bench, "split": args.split, "kg": args.kg, "provenance": args.provenance, "n_qa": len(qa), "n_passages": len(passages), "top_n_stored": args.top_n,
        "hipporag_version": pkg_version("hipporag"), "hipporag_module": hipporag.__file__, "clone": clone_state(),
        "python": sys.version, "llm": {"hipporag_label": LLM_NAME, "api_model": LLM_API_MODEL,
                                       "base_url": LLM_BASE_URL},
        "embedding": EMBEDDING, "config": cfg, "kg_info": kg_info,
        "chunk_embeddings_seeded_from": chunk_reuse if fresh else "(index reused; see the build run's meta)",
        "index_copied_from": index_copied_from,
        "injection": {
            "passages_with_triples": sum(1 for x in docs_inj if x["extracted_triples"]),
            "injected_triples_total": sum(len(x["extracted_triples"]) for x in docs_inj),
            "fingerprint": inj_fp,
        },
        "verification": verify,
        "llm_cache_rows_after_run": n_llm,
        "openie_llm_calls": 0,  # batch_openie was replaced by a raising stub and never fired
        "path_counts": dict(collections.Counter(r["path"] for r in rows)),
        "filter_errors": sum(1 for r in rows if r["filter_error"]),
        "hit_at_5_both_gold": hit5,
        "seconds": {"index": round(t_index, 1), "retrieve": round(t_retr, 1), "llm_gate_wait": round(llm_wait, 1)},
    }
    json.dump(meta, open(run_dir / f"run_meta{tag}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    log.info("DONE %s %s n=%d hit@5=%.3f paths=%s llm_cache_rows=%d index=%.0fs retrieve=%.0fs",
             args.split, args.kg, len(qa), hit5, meta["path_counts"], n_llm, t_index, t_retr)


if __name__ == "__main__":
    main()
