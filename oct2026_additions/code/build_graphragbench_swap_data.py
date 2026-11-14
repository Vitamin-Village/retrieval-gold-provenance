# -*- coding: utf-8 -*-
"""GraphRAG-Bench (medical) third-swap data builder. Implements S1 and the
S2 input of paper/eval/_graphragbench_swap_design.md.

Writes, under paper/external/graphragbench/:
    corpus/passages.jsonl, corpus/manifest.json      shared 120-word chunks
    <split>/qa.jsonl, passages.jsonl, manifest.json   (passages/manifest copied)
    <split>/kg_silver_input/relations_as_passages.jsonl
        pseudo-passages keyed by gold chunk id whose text is the concatenated
        evidence_relations of the split's questions; build_openie_kg.py turns
        these into kg_silver (gold-derived node-to-passage mapping).
    build_report.json

Run from repo root: python paper/eval/build_graphragbench_swap_data.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "paper" / "external" / "graphragbench" / "raw"
OUT_BASE = ROOT / "paper" / "external" / "graphragbench"

SEED = 20260918
N_PER_SPLIT = 200
CHUNK_WORDS = 120
MIN_RECALL = 0.6
MAX_GOLD = 2
TYPES = ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize")
STOP = set("a an the of in on for to and is are was were by with as at from that this it its be or can may which who what how".split())
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def toks(s: str) -> set:
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if t not in STOP and len(t) > 2}


def as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return list(ast.literal_eval(v))
        except Exception:
            return [v]
    return [v]


def chunk_corpus(text: str):
    sents = _SENT_RE.split(text)
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if n + w > CHUNK_WORDS and cur:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def best_sentence_span(chunk: str, evidence: str):
    """Char span of the chunk sentence with the highest token overlap with the evidence."""
    et = toks(evidence)
    best, pos = None, 0
    for s in _SENT_RE.split(chunk):
        i = chunk.find(s, pos)
        if i < 0:
            i = chunk.find(s)
        sc = len(et & toks(s)) / max(1, len(et))
        if best is None or sc > best[0]:
            best = (sc, [i, i + len(s)])
        pos = i + len(s)
    return best[1] if best else [0, min(len(chunk), 200)]


def build_all():
    corpus_text = json.load(open(RAW / "medical_corpus.json", encoding="utf-8"))[0]["context"]
    chunks = chunk_corpus(corpus_text)
    cids = ["c%04d" % i for i in range(len(chunks))]
    ctoks = [toks(c) for c in chunks]
    passages = [{"passage_id": cid, "paper_id": cid, "text": chunks[i], "title": "Medical guideline, segment %d" % i}
                for i, cid in enumerate(cids)]
    manifest = [{"id": cid, "paper_id": cid, "title": "Medical guideline, segment %d" % i, "abstract": chunks[i], "partition": "eval"}
                for i, cid in enumerate(cids)]

    questions = json.load(open(RAW / "medical_questions.json", encoding="utf-8"))
    eligible = defaultdict(list)
    stats = {"total": len(questions), "type_ok": 0, "all_evidence_mapped": 0, "gold_size_ok": 0}
    for x in questions:
        if x["question_type"] not in TYPES:
            continue
        stats["type_ok"] += 1
        ev = [s for s in as_list(x["evidence"]) if isinstance(s, str) and s.strip()]
        rel = [s for s in as_list(x["evidence_relations"]) if isinstance(s, str) and s.strip()]
        if not ev or not rel:
            continue
        golds, spans, ok = OrderedDict(), [], True
        for s in ev:
            t = toks(s)
            if not t:
                ok = False
                break
            best_i, best_r = -1, -1.0
            for i, ct in enumerate(ctoks):
                r = len(t & ct) / len(t)
                if r > best_r:
                    best_i, best_r = i, r
            if best_r < MIN_RECALL:
                ok = False
                break
            golds[cids[best_i]] = True
            spans.append({"passage_id": cids[best_i], "span": best_sentence_span(chunks[best_i], s), "recall": round(best_r, 3)})
        if not ok:
            continue
        stats["all_evidence_mapped"] += 1
        if not 1 <= len(golds) <= MAX_GOLD:
            continue
        stats["gold_size_ok"] += 1
        qa_id = "grb_" + hashlib.sha1((x["id"] + "\t" + x["question"]).encode("utf-8")).hexdigest()[:12]
        eligible[x["question_type"]].append({
            "qa_id": qa_id, "source_id": x["id"], "question": x["question"], "answer": x["answer"], "answers": [x["answer"]],
            "gold_paper_ids": list(golds), "qtype": x["question_type"], "hop_count": len(golds),
            "evidence_spans": spans, "evidence": ev, "evidence_relations": rel, "source": "GraphRAG-Bench medical",
        })

    # Per type: seeded shuffle, then alternate items between the two splits so both
    # splits draw from identical type pools; within each split, round-robin across
    # types until 200. Both splits therefore have the same type composition.
    rng = random.Random(SEED)
    halves = {"sub200": OrderedDict(), "holdout200": OrderedDict()}
    for t in TYPES:
        lst = list(eligible[t])
        rng.shuffle(lst)
        halves["sub200"][t] = lst[0::2]
        halves["holdout200"][t] = lst[1::2]
    splits = {}
    for name, pools in halves.items():
        sel = []
        while len(sel) < N_PER_SPLIT and any(pools.values()):
            for t in pools:
                if pools[t] and len(sel) < N_PER_SPLIT:
                    sel.append(pools[t].pop(0))
        splits[name] = sel
    selected = splits["sub200"] + splits["holdout200"]

    arts = {}
    for name, qas in splits.items():
        rel_by_chunk = OrderedDict()
        for qa in qas:
            for g in qa["gold_paper_ids"]:
                rel_by_chunk.setdefault(g, [])
                for r in qa["evidence_relations"]:
                    if r not in rel_by_chunk[g]:
                        rel_by_chunk[g].append(r)
        title = {p["passage_id"]: p["title"] for p in passages}
        silver_input = [{"passage_id": g, "paper_id": g, "title": title[g], "text": ". ".join(rs) + "."} for g, rs in rel_by_chunk.items()]
        arts[name] = {"qa": qas, "silver_input": silver_input,
                      "n_gold_chunks": len(rel_by_chunk), "qtype_counts": {t: sum(q["qtype"] == t for q in qas) for t in TYPES}}
    meta = {"seed": SEED, "n_chunks": len(chunks), "chunk_words": CHUNK_WORDS, "min_recall": MIN_RECALL, "stats": stats,
            "eligible_by_type": {t: len(v) for t, v in eligible.items()}, "n_selected": len(selected)}
    return passages, manifest, arts, meta


def main():
    p1, m1, a1, meta = build_all()
    p2, m2, a2, _ = build_all()
    assert json.dumps([p1, m1, a1], sort_keys=True) == json.dumps([p2, m2, a2], sort_keys=True), "non-deterministic build"
    (OUT_BASE / "corpus").mkdir(parents=True, exist_ok=True)
    with open(OUT_BASE / "corpus" / "passages.jsonl", "w", encoding="utf-8") as fh:
        for r in p1:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(m1, open(OUT_BASE / "corpus" / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for name, art in a1.items():
        d = OUT_BASE / name
        (d / "kg_silver_input").mkdir(parents=True, exist_ok=True)
        with open(d / "qa.jsonl", "w", encoding="utf-8") as fh:
            for r in art["qa"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(d / "passages.jsonl", "w", encoding="utf-8") as fh:
            for r in p1:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump(m1, open(d / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(d / "kg_silver_input" / "relations_as_passages.jsonl", "w", encoding="utf-8") as fh:
            for r in art["silver_input"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, "n_qa", len(art["qa"]), "gold chunks", art["n_gold_chunks"], art["qtype_counts"], flush=True)
    json.dump(meta, open(OUT_BASE / "build_report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
