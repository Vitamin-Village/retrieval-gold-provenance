# -*- coding: utf-8 -*-
"""Rater-free provenance trace for GraphRAG-Bench (Xiang et al., ICLR 2026),
with 2WikiMultiHop as the exact-provenance reference.

Purpose: replace a rater judgment with deterministic, per-unit facts about how
GraphRAG-Bench's released gold fields relate to the released structure
(`evidence_relations` / `evidence_triple`) and to the source corpus.

Everything is lexical and deterministic: no model calls, no sampling. SEED is
recorded for protocol completeness only.

Normalization `norm`: NFKC, lowercase, every non-[a-z0-9] run -> one space.
"Verbatim" means word-boundary containment of normalized strings.
Content tokens: normalized tokens of length > 1 not in STOP (fixed list below,
identical to analysis/perunit_graphragbench.py so the realization rule matches).

Data sets and structures
  grb_med     GraphRAG-Bench medical. Structure = released `evidence_relations`
              (natural-language relation strings). On the two 200-question
              swap splits the aligned KG is kg_silver (OpenIE triples extracted
              from those strings, keyed by gold chunk); its non-synonym edges
              give subject/object endpoints. Full raw set: string-level tests.
  grb_novel   GraphRAG-Bench novel, full set, pinned upstream commit 0b8f46a
              (results/grb_provenance/inputs). Structure = `evidence_triple`
              "(s, p, o)" strings, parsed only when unambiguous (exactly two
              commas); ambiguous triples are counted and excluded.
  w2          2WikiMultiHop swap splits. Structure = Wikidata `evidences`
              (s, r, o); aligned KG = kg_silver built from them.

Tests (per question; aggregated as question means and pooled unit fractions)
  (a) structure membership: each gold relation is present in the aligned KG;
      each evidence sentence realizes a released relation (>= 0.5 of the
      relation's content tokens in the sentence, the perunit rule) or, for
      triples, contains both endpoints.
  (b) answer axis: answer equals / contains / is contained in a released
      relation string or triple endpoint (exact after normalization);
      answer content-token recall by the relations vs by the best corpus chunk.
  (c) gold-passage recoverability from structure alone: every gold passage
      contains >= 1 relation endpoint; structure-only ranking of the whole
      candidate pool by the number of distinct endpoints contained
      (pessimistic tie rank); relation-string ranking by content-token recall.
  (d) text grounding of gold relations: subject and object both verbatim in a
      single evidence sentence / in a gold passage / in some corpus passage;
      relation string and evidence sentence verbatim in the corpus.
  (e) release order (GraphRAG-Bench git history, pinned inputs): whether the
      first released "raw evidence" equals the later `evidence_relations`, and
      whether questions/answers changed when `evidence` sentences were added.

Outputs: results/grb_provenance/{summary.json, summary.md, per_question.jsonl}
Run from anywhere:  python paper/eval/_grb_provenance_trace.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
EXT = PAPER / "external"
GRB = EXT / "graphragbench"
W2 = EXT / "2wiki"
OUT = PAPER / "results" / "grb_provenance"
INPUTS = OUT / "inputs"
SEED = 20260929  # no sampling anywhere; recorded for protocol completeness
SPLITS = ("sub200", "holdout200")
KS = (1, 5)
DERIV_FRAC = 0.5  # perunit realization rule

STOP = frozenset("""
a an the of for to in on at by with from into over under and or but nor so as
is are was were be been being am do does did have has had having will would
shall should can could may might must this that these those it its their his
her them they we you i he she who whom which what when where why how than then
not no yes such each any all both more most some few many much very also only
just about between among within without across per via vs versus include
includes including increase increases increased risk factor factors
""".split())

# Primary-source record (verbatim; locations checked 2026-09-29). The OpenReview notes are in
# results/grb_provenance/sources/; the deleted upstream scripts are cited by repository and commit (below).
SOURCES = [
    {"src": "arXiv 2506.05690v3 (ICLR 2026 version), App. A.3, p.18",
     "quote": "We begin by extracting information from corpora to build ontologies or knowledge graphs. Taking the Novel dataset as an "
              "example, we first construct a knowledge graph from the corpus, then form logic chains using relevant triples from this "
              "graph. Then, we control difficulty by adjusting the number of involved triples and the information span they cover."},
    {"src": "arXiv 2506.05690v3, App. C.2 Logic Mining, p.20",
     "quote": "we transform text into formalized ontologies using GPT-4.1 [...] This ontology acts as a ground-truth map of domain logic"},
    {"src": "arXiv 2506.05690v3, App. C.4 Question Generation, p.21",
     "quote": "Questions are generated by aligning their cognitive demands with the structural properties of the underlying evidence. "
              "Retrieval-focused questions target isolated subgraphs"},
    {"src": "arXiv 2506.05690v3, App. C.5 Check & Correct, pp.21-22",
     "quote": "given the original corpus and the constructed evidence, we assess whether the evidence can be logically derived from the "
              "corpus. Our validation criteria are strict: given a question, if no evidence triple can be inferred from the corpus, the "
              "corresponding question is discarded. Similarly, for answer correction, we check whether the provided answer can be "
              "logically inferred from the evidence. This verification and correction process is supported by advanced models combined "
              "with human checking for final confirmation."},
    {"src": "arXiv 2506.05690v3, App. C.6 Refinement, p.23",
     "quote": "for each question, we locate the corpus segment from which the evidence was derived and employ GPT-4.1 to refine the "
              "original question by integrating this segment."},
    {"src": "arXiv 2506.05690v3, Fig. 10 (Medical construction prompts), p.32",
     "quote": "Fact Retrieval: 'Create information-seeking questions that require direct factual retrieval from the relations list.' "
              "Contextual Summarize: 'should ideally cover **the majority of the ontology**' / 'Do NOT infer or invent unstated details or "
              "use any information not explicitly stated in the input or ontology.' Creative Generation: 'Use explicit entities from the triplets'"},
    {"src": "arXiv 2506.05690v3, Fig. 8 (Novel construction prompts), p.22",
     "quote": "Complex Reasoning: 'generate {num} high-quality English reasoning questions from a given context made up of multiple logical "
              "chains. Each question must span multiple triplets' / '2. The answer must be uniquely supported by the input triplets. "
              "3. Do not make up or infer anything that is not explicitly stated in the triplets.'"},
    {"src": "OpenReview forum i9q9xDMjG7, author response kD5Ux3zqga (to Reviewer zK5v, W3; Medical)",
     "quote": "for each specific disease section, we use GPT-4.1 to identify entities and relations, extracting structured triples to "
              "build an initial instance-level Knowledge Graph (KG). [...] we input both the extracted triples and their corresponding "
              "original text chunks into the LLM to induce a high-level ontology schema (classes and hierarchy). [...] we refine the "
              "generated ontology by cross-referencing it with the original text to resolve ambiguities"},
    {"src": "OpenReview forum i9q9xDMjG7, author response 6PlVCQOMWh (to Reviewer ebWh, W2; Novel)",
     "quote": "we first use the LLM (GPT-4.1) to iterate through each document and extract structured triples [...] building an initial "
              "knowledge graph (KG) for that document. [...] Finally, both the subgraphs and the extracted logic chains are represented "
              "and stored as lists of triples."},
    {"src": "GitHub GraphRAG-Bench/GraphRAG-Benchmark commit 431e376 (2025-05-16; deleted in e6305f5, 2025-06-09)",
     "quote": "1_kg_generate.py L23-27 `graph = await kg.generate(input_data=large_text, chunk_size=chunk_size, cluster=cluster)` "
              "(L74 chunk_size=5000; L60 model=\"moonshot-v1-auto\"); 2_kg_post_process.py L38 `subgraphs = "
              "graph.extract_largest_subgraphs(top_k=5)`, L44 `small_subgraphs = subgraph.split_graph(50)`; 2_2_kg_extract_path.py "
              "L43 `source_nodes = get_top_degree_nodes(G, top_k_sources)`, L52-57 simple paths with `len(path) - 1 >= k`, L108 "
              "`min_path_len = 5`. The scripts name pg19 ids test_30752/26183/29973/25646/2544 (commented file_list, L51), test_10146 (comment, L95) and test_10762 (live, 2_2 L91); all 7 are among the 20 released Novel sources."},
    {"src": "GitHub commits e18981f 'add raw evidence' (2025-06-25 01:55 +0800) and b068e72 'Add evidence' (2025-06-25 11:53 +0800)",
     "quote": "see release_order (computed from pinned inputs)"},
]

_NONALNUM = re.compile(r"[^a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def norm(s) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return " ".join(_NONALNUM.sub(" ", s).split())


def pad(s: str) -> str:
    return " " + s + " "


def has(hay_padded: str, needle_n: str) -> bool:
    """Word-boundary containment; hay must be pre-padded normalized text."""
    return bool(needle_n) and (" " + needle_n + " ") in hay_padded


def ctoks(s) -> set:
    return {t for t in norm(s).split() if len(t) > 1 and t not in STOP}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def frac(num, den):
    return {"num": num, "den": den, "frac": round(num / den, 4) if den else None}


def load_jsonl(p: Path):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_builder():
    sys.dont_write_bytecode = True  # keep eval/__pycache__ untouched
    spec = importlib.util.spec_from_file_location("grb_builder", PAPER / "eval" / "build_graphragbench_swap_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Pool:
    """Candidate passage pool with normalized text and a content-token index."""

    def __init__(self, ids, texts):
        self.ids = list(ids)
        self.pos = {pid: i for i, pid in enumerate(self.ids)}
        self.padded = [pad(norm(t)) for t in texts]
        self.ct = [ctoks(t) for t in texts]
        self.inv = defaultdict(list)
        for i, ts in enumerate(self.ct):
            for t in ts:
                self.inv[t].append(i)

    def recall_scores(self, tokens: set):
        c = Counter()
        for t in tokens:
            for i in self.inv.get(t, ()):
                c[i] += 1
        return c

    def best_recall(self, tokens: set) -> float:
        if not tokens:
            return None
        c = self.recall_scores(tokens)
        return round(max(c.values()) / len(tokens), 4) if c else 0.0

    def endpoint_scores(self, endpoints):
        eps = [e for e in endpoints if e]
        return [sum(1 for e in eps if has(p, e)) for p in self.padded]

    def any_contains_both(self, a: str, b: str) -> bool:
        return any(has(p, a) and has(p, b) for p in self.padded)


def pess_rank(scores, idx):
    """Pessimistic rank of item idx: count of items with score >= its score; score 0 -> unrecoverable."""
    s = scores[idx]
    if not s:
        return len(scores)
    return sum(1 for v in scores if v >= s)


def realizes(sentence: str, relation: str) -> bool:
    rt = ctoks(relation)
    return bool(rt) and len(rt & ctoks(sentence)) / len(rt) >= DERIV_FRAC


# --------------------------------------------------------------------------- GRB medical

def grb_med_records(builder):
    raw = json.load(open(GRB / "raw" / "medical_questions.json", encoding="utf-8"))
    corpus_raw = json.load(open(GRB / "raw" / "medical_corpus.json", encoding="utf-8"))[0]["context"]
    passages = load_jsonl(GRB / "corpus" / "passages.jsonl")
    pool = Pool([p["passage_id"] for p in passages], [p["text"] for p in passages])
    corpus_padded = pad(norm(corpus_raw))

    # builder-identical evidence->chunk mapping (for gold chunks on the full set)
    btoks = [builder.toks(p["text"]) for p in passages]
    binv = defaultdict(list)
    for i, ts in enumerate(btoks):
        for t in ts:
            binv[t].append(i)

    def builder_gold(ev):
        golds, ok = OrderedDict(), True
        for s in ev:
            t = builder.toks(s)
            if not t:
                return None
            c = Counter()
            for tok in t:
                for i in binv.get(tok, ()):
                    c[i] += 1
            best = max(c.values()) if c else 0
            best_i = min(i for i, v in c.items() if v == best) if c else 0
            if best / len(t) < builder.MIN_RECALL:
                return None
            golds[pool.ids[best_i]] = True
        return list(golds)

    split_q = {s: load_jsonl(GRB / s / "qa.jsonl") for s in SPLITS}
    split_of = {q["source_id"]: s for s, qs in split_q.items() for q in qs}
    split_gold = {q["source_id"]: q["gold_paper_ids"] for qs in split_q.values() for q in qs}

    # aligned KG per split: non-synonym edges with endpoint names
    kg = {}
    for s in SPLITS:
        nodes = json.load(open(GRB / s / "kg_silver" / "proposed_kg_nodes.json", encoding="utf-8"))
        edges = json.load(open(GRB / s / "kg_silver" / "proposed_kg_edges.json", encoding="utf-8"))
        name = {n["node_id"]: n["name"] for n in nodes}
        el = []
        for e in edges:
            if e.get("relation_type") == "synonym":
                continue
            el.append((norm(name.get(e["source"], e["source"][3:])), norm(e["relation"]), norm(name.get(e["target"], e["target"][3:])),
                       frozenset(e.get("source_docs") or ([e["source_doc"]] if e.get("source_doc") else []))))
        linked = {d for n in nodes for d in n.get("source_docs", [])}
        kg[s] = {"edges": el, "node_names": {norm(n["name"]) for n in nodes}, "linked": linked}

    recs, mapping_mismatch = [], 0
    for x in raw:
        rels = [r for r in builder.as_list(x["evidence_relations"]) if isinstance(r, str) and r.strip()]
        ev = [s for s in builder.as_list(x["evidence"]) if isinstance(s, str) and s.strip()]
        ans = x["answer"] or ""
        a_n, rels_n = norm(ans), [norm(r) for r in rels]
        split = split_of.get(x["id"])
        gold = builder_gold(ev) if ev else None
        if split is not None and gold != split_gold[x["id"]]:
            mapping_mismatch += 1
        rec = {"dataset": "grb_med", "split": split or "full_only", "id": x["id"], "qtype": x["question_type"],
               "n_rel": len(rels), "n_ev": len(ev), "gold": gold}
        # (a) evidence sentence realizes a released relation
        rec["a_ev_realizes_rel"] = [any(realizes(s, r) for r in rels) for s in ev]
        # (b) answer axis, string level
        rec["b_ans_eq_rel"] = any(a_n == r for r in rels_n)
        rec["b_ans_in_rel"] = any(has(pad(r), a_n) for r in rels_n)
        rec["b_rel_in_ans"] = any(has(pad(a_n), r) for r in rels_n)
        at, rt = ctoks(ans), set().union(*[ctoks(r) for r in rels]) if rels else set()
        rec["b_ans_recall_by_rels"] = round(len(at & rt) / len(at), 4) if at else None
        rec["b_ans_recall_by_best_chunk"] = pool.best_recall(at)
        if gold:
            gt = set().union(*[pool.ct[pool.pos[g]] for g in gold])
            rec["b_ans_recall_by_gold_chunks"] = round(len(at & gt) / len(at), 4) if at else None
        # (d) string-level grounding in the corpus
        rec["d_rel_verbatim_in_corpus"] = [has(corpus_padded, r) for r in rels_n]
        rec["d_rel_best_chunk_recall"] = [pool.best_recall(ctoks(r)) for r in rels]
        rec["d_ev_verbatim_in_corpus"] = [has(corpus_padded, norm(s)) for s in ev]
        rec["d_ev_best_chunk_recall"] = [pool.best_recall(ctoks(s)) for s in ev]
        rec["d_ev_recall_by_rels"] = [round(len(ctoks(s) & rt) / len(ctoks(s)), 4) if ctoks(s) else None for s in ev]
        # (c) relation-string-only recovery of gold chunks
        if gold:
            best_rank = {}
            for r in rels:
                t = ctoks(r)
                if not t:
                    continue
                c = pool.recall_scores(t)
                scores = [c.get(i, 0) for i in range(len(pool.ids))]
                for g in gold:
                    rk = pess_rank(scores, pool.pos[g])
                    best_rank[g] = min(best_rank.get(g, rk), rk)
            rec["c_rs_gold_rank"] = [best_rank.get(g, len(pool.ids)) for g in gold]
        # endpoint-level tests on the swap splits (aligned KG = kg_silver)
        if split is not None:
            gold = split_gold[x["id"]]
            K = kg[split]
            gset = set(gold)
            mapped = []
            for ri, r in enumerate(rels_n):
                pr = pad(r)
                m = [(s_, p_, o_) for (s_, p_, o_, docs) in K["edges"] if docs & gset and has(pr, s_) and has(pr, o_)]
                mapped.append(m)
            rec["a_rel_in_aligned_kg"] = [bool(m) for m in mapped]
            rec["a_gold_linked_by_aligned_kg"] = [g in K["linked"] for g in gold]
            eps = sorted({e for m in mapped for (s_, _, o_) in m for e in (s_, o_)})
            rec["n_endpoints"] = len(eps)
            rec["b_ans_eq_node"] = a_n in K["node_names"]
            rec["b_ans_eq_endpoint"] = a_n in set(eps)
            rec["b_ans_contains_endpoint"] = any(has(pad(a_n), e) for e in eps)
            gp = [pool.padded[pool.pos[g]] for g in gold]
            rec["c_gold_has_endpoint"] = [any(has(p, e) for e in eps) for p in gp]
            scores = pool.endpoint_scores(eps)
            rec["c_ep_gold_rank"] = [pess_rank(scores, pool.pos[g]) for g in gold]
            evp = [pad(norm(s)) for s in ev]
            trip = sorted({t for m in mapped for t in m})
            rec["n_triples"] = len(trip)
            rec["d_triple_both_in_ev_sentence"] = [any(has(p, s_) and has(p, o_) for p in evp) for (s_, _, o_) in trip]
            rec["d_triple_both_in_gold_chunk"] = [any(has(p, s_) and has(p, o_) for p in gp) for (s_, _, o_) in trip]
            rec["d_triple_both_in_some_chunk"] = [pool.any_contains_both(s_, o_) for (s_, _, o_) in trip]
        recs.append(rec)
    return recs, {"n_raw": len(raw), "split_gold_mapping_mismatch": mapping_mismatch,
                  "n_chunks": len(pool.ids)}


# --------------------------------------------------------------------------- GRB novel

_TRIPLE = re.compile(r"^\s*\((.*)\)\s*\.?\s*$", re.S)


def parse_triple(t):
    if not isinstance(t, str):
        return None
    m = _TRIPLE.match(t)
    if not m:
        return None
    parts = m.group(1).split(",")
    if len(parts) != 3:
        return None
    s, p, o = (norm(v) for v in parts)
    return (s, p, o) if s and o else None


def grb_novel_records():
    path = INPUTS / "novel_questions_0b8f46a.json"
    if not path.exists():
        return [], {"skipped": "missing " + str(path)}
    qs = json.load(open(path, encoding="utf-8"))
    corpus = {c["corpus_name"]: c["context"] for c in json.load(open(GRB / "raw" / "novel_corpus.json", encoding="utf-8"))}
    sents = {k: [pad(norm(s)) for s in _SENT.split(v) if s.strip()] for k, v in corpus.items()}
    full = {k: pad(norm(v)) for k, v in corpus.items()}
    recs, n_trip, n_amb = [], 0, 0
    for x in qs:
        trips_raw = x.get("evidence_triple") or []
        trips = []
        for t in trips_raw:
            n_trip += 1
            pt = parse_triple(t)
            if pt is None:
                n_amb += 1
            else:
                trips.append(pt)
        ev = [s for s in (x.get("evidence") or []) if isinstance(s, str) and s.strip()]
        evp = [pad(norm(s)) for s in ev]
        a_n = norm(x.get("answer") or "")
        src = x.get("source")
        S, F = sents.get(src, []), full.get(src, "")
        rec = {"dataset": "grb_novel", "split": "full", "id": x["id"], "qtype": x["question_type"],
               "n_triple_raw": len(trips_raw), "n_triple_parsed": len(trips), "n_ev": len(ev)}
        rec["a_ev_contains_triple_endpoints"] = [any(has(p, s_) and has(p, o_) for (s_, _, o_) in trips) for p in evp]
        objs, subs = {o_ for (_, _, o_) in trips}, {s_ for (s_, _, _) in trips}
        rec["b_ans_eq_object"] = a_n in objs
        rec["b_ans_eq_endpoint"] = a_n in (objs | subs)
        rec["b_ans_contains_object"] = any(has(pad(a_n), o_) for o_ in objs)
        rec["d_triple_both_in_ev_sentence"] = [any(has(p, s_) and has(p, o_) for p in evp) for (s_, _, o_) in trips]
        rec["d_triple_both_in_source_sentence"] = [any(has(p, s_) and has(p, o_) for p in S if (" " + o_ + " ") in p) for (s_, _, o_) in trips]
        rec["d_triple_s_and_o_in_source_text"] = [has(F, s_) and has(F, o_) for (s_, _, o_) in trips]
        rec["d_ev_verbatim_in_source"] = [has(F, norm(s)) for s in ev]
        recs.append(rec)
    return recs, {"n_questions": len(qs), "n_triples": n_trip, "n_triples_unparsed_or_ambiguous": n_amb, "input_md5": md5(path)}


# --------------------------------------------------------------------------- 2WikiMultiHop

def w2_records():
    dev = {r["_id"]: r for r in json.load(open(W2 / "raw" / "data_ids" / "dev.json", encoding="utf-8"))}
    aliases = {}
    with open(W2 / "raw" / "data_ids" / "id_aliases.json", encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            aliases[r["Q_id"]] = r.get("aliases", []) + r.get("demonyms", [])
    recs = []
    for split in SPLITS:
        qa = load_jsonl(W2 / split / "qa.jsonl")
        passages = load_jsonl(W2 / split / "passages.jsonl")
        pool = Pool([p["passage_id"] for p in passages], [p["text"] for p in passages])
        nodes = json.load(open(W2 / split / "kg_silver" / "proposed_kg_nodes.json", encoding="utf-8"))
        edges = json.load(open(W2 / split / "kg_silver" / "proposed_kg_edges.json", encoding="utf-8"))
        name = {n["node_id"]: n["name"] for n in nodes}
        edge_set = {(norm(name.get(e["source"], "")), norm(e["relation"]), norm(name.get(e["target"], ""))) for e in edges}
        linked = {d for n in nodes for d in n.get("source_docs", [])}
        for q in qa:
            r = dev[q["qa_id"]]
            ctx = {t: ss for t, ss in r["context"]}
            sup = [pad(norm(ctx[t][i])) for t, i in r["supporting_facts"] if t in ctx and i < len(ctx[t])]
            sup_title = [(norm(t), pad(norm(ctx[t][i]))) for t, i in r["supporting_facts"] if t in ctx and i < len(ctx[t])]
            trips = [(norm(s), norm(p), norm(o)) for s, p, o in r["evidences"]]
            ev_id = r.get("evidences_id") or []
            gold = q["gold_paper_ids"]
            a_n = norm(q["answer"])
            ans_alias = {norm(v) for v in aliases.get(r.get("answer_id") or "", [])} | {a_n}
            obj_alias = set()
            for i, (_, _, o) in enumerate(trips):
                obj_alias.add(o)
                if i < len(ev_id) and len(ev_id[i]) == 3:
                    obj_alias |= {norm(v) for v in aliases.get(ev_id[i][2], [])}
            eps = sorted({e for (s_, _, o_) in trips for e in (s_, o_)})
            gp = [pool.padded[pool.pos[g]] for g in gold if g in pool.pos]
            scores = pool.endpoint_scores(eps)
            rec = {"dataset": "w2", "split": split, "id": q["qa_id"], "qtype": q.get("qtype"), "n_triples": len(trips), "gold": gold}
            rec["a_rel_in_aligned_kg"] = [t in edge_set for t in trips]
            rec["a_gold_linked_by_aligned_kg"] = [g in linked for g in gold]
            rec["a_ev_contains_triple_endpoints"] = [any(has(p, s_) and has(p, o_) for (s_, _, o_) in trips) for p in sup]
            rec["b_ans_eq_object"] = a_n in {o for (_, _, o) in trips}
            rec["b_ans_eq_endpoint"] = a_n in set(eps)
            rec["b_ans_eq_object_alias_aware"] = bool(ans_alias & obj_alias)
            rec["c_gold_has_endpoint"] = [any(has(p, e) for e in eps) for p in gp]
            rec["c_ep_gold_rank"] = [pess_rank(scores, pool.pos[g]) for g in gold if g in pool.pos]
            rec["d_triple_both_in_ev_sentence"] = [any(has(p, s_) and has(p, o_) for p in sup) for (s_, _, o_) in trips]
            rec["d_triple_title_subject_object_in_sentence"] = [any(t == s_ and has(p, o_) for t, p in sup_title) for (s_, _, o_) in trips]
            rec["d_triple_both_in_gold_chunk"] = [any(has(p, s_) and has(p, o_) for p in gp) for (s_, _, o_) in trips]
            rec["d_triple_both_in_some_chunk"] = [pool.any_contains_both(s_, o_) for (s_, _, o_) in trips]
            recs.append(rec)
    return recs, {}


# --------------------------------------------------------------------------- release order

def release_order():
    out = {}
    local = {x["id"]: x for x in json.load(open(GRB / "raw" / "medical_questions.json", encoding="utf-8"))}
    p = INPUTS / "medical_questions_e18981f.json"
    if p.exists():
        early = {x["id"]: x for x in json.load(open(p, encoding="utf-8"))}
        ids = sorted(set(early) & set(local))
        out["medical"] = {
            "input_md5": md5(p), "n_common_ids": len(ids),
            "early_has_evidence_relations_field": any("evidence_relations" in early[i] for i in ids),
            "early_raw_evidence_equals_local_evidence_relations": sum(early[i]["evidence"] == local[i]["evidence_relations"] for i in ids),
            "early_raw_evidence_equals_local_evidence": sum(early[i]["evidence"] == local[i]["evidence"] for i in ids),
            "question_unchanged": sum(early[i]["question"] == local[i]["question"] for i in ids),
            "answer_unchanged": sum(early[i]["answer"] == local[i]["answer"] for i in ids),
        }
    p1, p2 = INPUTS / "novel_questions_e18981f.json", INPUTS / "novel_questions_0b8f46a.json"
    if p1.exists() and p2.exists():
        early = {x["id"]: x for x in json.load(open(p1, encoding="utf-8"))}
        late = {x["id"]: x for x in json.load(open(p2, encoding="utf-8"))}
        ids = sorted(set(early) & set(late))
        out["novel"] = {
            "input_md5": [md5(p1), md5(p2)], "n_common_ids": len(ids),
            "early_raw_evidence_equals_later_evidence_triple": sum(early[i]["evidence"] == late[i]["evidence_triple"] for i in ids),
            "early_raw_evidence_equals_later_evidence": sum(early[i]["evidence"] == late[i]["evidence"] for i in ids),
            "question_unchanged": sum(early[i]["question"] == late[i]["question"] for i in ids),
            "answer_unchanged": sum(early[i]["answer"] == late[i]["answer"] for i in ids),
        }
    return out


# --------------------------------------------------------------------------- aggregation

def agg_bool_list(recs, key):
    units = [v for r in recs for v in (r.get(key) or [])]
    qm = [sum(r[key]) / len(r[key]) for r in recs if r.get(key)]
    return {"pooled": frac(sum(units), len(units)), "question_mean": mean(qm), "n_questions": len(qm)}


def agg_bool(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return frac(sum(1 for v in vals if v), len(vals))


def agg_num(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return {"mean": mean(vals), "n": len(vals), "frac_eq_1": round(sum(1 for v in vals if v >= 1.0) / len(vals), 4) if vals else None}


def agg_num_list(recs, key):
    vals = [v for r in recs for v in (r.get(key) or []) if v is not None]
    return {"mean": mean(vals), "n_units": len(vals), "frac_eq_1": round(sum(1 for v in vals if v >= 1.0) / len(vals), 4) if vals else None,
            "frac_ge_0.9": round(sum(1 for v in vals if v >= 0.9) / len(vals), 4) if vals else None}


def agg_rank(recs, key):
    rs = [r for r in recs if r.get(key)]
    out = {"n_questions": len(rs)}
    for k in KS:
        out[f"all_gold_within_top{k}"] = frac(sum(1 for r in rs if max(r[key]) <= k), len(rs))
        units = [v for r in rs for v in r[key]]
        out[f"gold_units_within_top{k}"] = frac(sum(1 for v in units if v <= k), len(units))
    return out


def summarize(recs, keys):
    out = {"n_questions": len(recs)}
    for k, kind in keys:
        if not any(r.get(k) is not None for r in recs):
            continue
        out[k] = {"bl": agg_bool_list, "b": agg_bool, "n": agg_num, "nl": agg_num_list, "rk": agg_rank}[kind](recs, k)
    return out


MED_STR = [("a_ev_realizes_rel", "bl"), ("b_ans_eq_rel", "b"), ("b_ans_in_rel", "b"), ("b_rel_in_ans", "b"),
           ("b_ans_recall_by_rels", "n"), ("b_ans_recall_by_best_chunk", "n"), ("b_ans_recall_by_gold_chunks", "n"),
           ("d_rel_verbatim_in_corpus", "bl"), ("d_rel_best_chunk_recall", "nl"), ("d_ev_verbatim_in_corpus", "bl"),
           ("d_ev_best_chunk_recall", "nl"), ("d_ev_recall_by_rels", "nl"), ("c_rs_gold_rank", "rk")]
MED_EP = [("a_rel_in_aligned_kg", "bl"), ("a_gold_linked_by_aligned_kg", "bl"), ("b_ans_eq_node", "b"), ("b_ans_eq_endpoint", "b"),
          ("b_ans_contains_endpoint", "b"), ("c_gold_has_endpoint", "bl"), ("c_ep_gold_rank", "rk"),
          ("d_triple_both_in_ev_sentence", "bl"), ("d_triple_both_in_gold_chunk", "bl"), ("d_triple_both_in_some_chunk", "bl")]
NOV = [("a_ev_contains_triple_endpoints", "bl"), ("b_ans_eq_object", "b"), ("b_ans_eq_endpoint", "b"), ("b_ans_contains_object", "b"),
       ("d_triple_both_in_ev_sentence", "bl"), ("d_triple_both_in_source_sentence", "bl"), ("d_triple_s_and_o_in_source_text", "bl"),
       ("d_ev_verbatim_in_source", "bl")]
W2K = [("a_rel_in_aligned_kg", "bl"), ("a_gold_linked_by_aligned_kg", "bl"), ("a_ev_contains_triple_endpoints", "bl"),
       ("b_ans_eq_object", "b"), ("b_ans_eq_endpoint", "b"), ("b_ans_eq_object_alias_aware", "b"), ("c_gold_has_endpoint", "bl"),
       ("c_ep_gold_rank", "rk"), ("d_triple_both_in_ev_sentence", "bl"), ("d_triple_title_subject_object_in_sentence", "bl"),
       ("d_triple_both_in_gold_chunk", "bl"), ("d_triple_both_in_some_chunk", "bl")]


def build():
    builder = load_builder()
    med, med_meta = grb_med_records(builder)
    nov, nov_meta = grb_novel_records()
    w2, _ = w2_records()
    S = OrderedDict()
    S["meta"] = {"seed": SEED, "deterministic": True, "stopwords": "analysis/perunit_graphragbench.py STOP",
                 "grb_medical_raw_md5": md5(GRB / "raw" / "medical_questions.json"), **{"grb_med_" + k: v for k, v in med_meta.items()},
                 "grb_novel": nov_meta}
    S["sources"] = SOURCES
    S["release_order"] = release_order()
    S["grb_med_full"] = summarize(med, MED_STR)
    S["grb_med_full_by_type"] = {t: summarize([r for r in med if r["qtype"] == t], MED_STR) for t in sorted({r["qtype"] for r in med})}
    for sp in SPLITS:
        rs = [r for r in med if r["split"] == sp]
        S[f"grb_med_{sp}"] = summarize(rs, MED_STR + MED_EP)
    both = [r for r in med if r["split"] in SPLITS]
    S["grb_med_splits_pooled"] = summarize(both, MED_STR + MED_EP)
    S["grb_med_splits_pooled_by_type"] = {t: summarize([r for r in both if r["qtype"] == t], MED_STR + MED_EP) for t in sorted({r["qtype"] for r in both})}
    if nov:
        S["grb_novel_full"] = summarize(nov, NOV)
        S["grb_novel_full_by_type"] = {t: summarize([r for r in nov if r["qtype"] == t], NOV) for t in sorted({r["qtype"] for r in nov})}
    for sp in SPLITS:
        S[f"w2_{sp}"] = summarize([r for r in w2 if r["split"] == sp], W2K)
    S["w2_splits_pooled"] = summarize(w2, W2K)
    return S, med + nov + w2


def g(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


def fmt(x):
    if isinstance(x, dict) and "frac" in x:
        return f"{x['frac']:.3f} ({x['num']}/{x['den']})" if x["frac"] is not None else "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return "n/a" if x is None else str(x)


def write_md(S):
    L = []
    A = L.append
    A("# GraphRAG-Bench provenance trace (rater-free)\n")
    A("Generated by `eval/_grb_provenance_trace.py` (deterministic, lexical, no model calls). "
      "`norm` = NFKC + lowercase + non-alphanumerics to spaces; \"verbatim\" = word-boundary containment after `norm`. "
      "Pessimistic tie ranks over the whole candidate pool (GraphRAG-Bench: 1,584 chunks; 2Wiki: split pool).\n")
    A("## Construction record (primary sources, verbatim)\n")
    for s in S["sources"]:
        A(f"- {s['src']}: {s['quote']}")
    A("")
    A("Not released: question-generation code, any Medical construction code, and the ontology/KG itself. The deleted upstream "
      "scripts use KGGen with `moonshot-v1-auto` (cache path `llama-70b`), whereas the paper names GPT-4.1, so the extractor "
      "model of the released data cannot be pinned from code. The revised-manuscript pseudocode figure cited in the author "
      "responses is absent from arXiv v3 (Figures 1-10 only); the OpenReview PDF was not retrievable (HTTP 403).\n")
    A("## Release order in the upstream repository (pinned inputs in `inputs/`)\n")
    ro = S["release_order"]
    if "medical" in ro:
        m = ro["medical"]
        A(f"- Medical: the field first released as `evidence` in commit e18981f (\"add raw evidence\", 2025-06-25 01:55 +0800) equals the later "
          f"`evidence_relations` for {m['early_raw_evidence_equals_local_evidence_relations']}/{m['n_common_ids']} questions and the later `evidence` "
          f"sentences for {m['early_raw_evidence_equals_local_evidence']}/{m['n_common_ids']}. Questions unchanged {m['question_unchanged']}/{m['n_common_ids']}, "
          f"answers unchanged {m['answer_unchanged']}/{m['n_common_ids']} when the sentence-level `evidence` was added (commit b068e72, \"Add evidence\", 10 h later).")
    if "novel" in ro:
        m = ro["novel"]
        A(f"- Novel: first released `evidence` equals later `evidence_triple` for {m['early_raw_evidence_equals_later_evidence_triple']}/{m['n_common_ids']}; "
          f"equals later `evidence` for {m['early_raw_evidence_equals_later_evidence']}/{m['n_common_ids']}; questions unchanged {m['question_unchanged']}/{m['n_common_ids']}.")
    A("")

    def table(title, rows, cols):
        A(f"### {title}\n")
        A("| Test | " + " | ".join(c for c, _ in cols) + " |")
        A("|---|" + "---|" * len(cols))
        for label, path in rows:
            A(f"| {label} | " + " | ".join(fmt(g(S, c_key, *path)) for _, c_key in cols) + " |")
        A("")

    med_cols = [("sub200", "grb_med_sub200"), ("holdout200", "grb_med_holdout200"), ("full raw", "grb_med_full")]
    A("## GraphRAG-Bench medical\n")
    table("(a) structure membership", [
        ("gold relation present in aligned KG (pooled relations)", ("a_rel_in_aligned_kg", "pooled")),
        ("gold chunk linked by aligned KG (pooled chunks)", ("a_gold_linked_by_aligned_kg", "pooled")),
        ("evidence sentence realizes a released relation (pooled)", ("a_ev_realizes_rel", "pooled")),
        ("  same, question mean", ("a_ev_realizes_rel", "question_mean")),
    ], med_cols)
    table("(b) answer axis", [
        ("answer == a relation string (normalized)", ("b_ans_eq_rel",)),
        ("answer contained in a relation string", ("b_ans_in_rel",)),
        ("a relation string contained in answer", ("b_rel_in_ans",)),
        ("answer == aligned-KG node name", ("b_ans_eq_node",)),
        ("answer == relation endpoint", ("b_ans_eq_endpoint",)),
        ("answer contains a relation endpoint", ("b_ans_contains_endpoint",)),
        ("answer content-token recall by relations (mean)", ("b_ans_recall_by_rels", "mean")),
        ("  share with recall 1.0", ("b_ans_recall_by_rels", "frac_eq_1")),
        ("answer content-token recall by gold chunk(s) (mean)", ("b_ans_recall_by_gold_chunks", "mean")),
        ("  share with recall 1.0", ("b_ans_recall_by_gold_chunks", "frac_eq_1")),
        ("answer content-token recall by best corpus chunk (mean)", ("b_ans_recall_by_best_chunk", "mean")),
    ], med_cols)
    table("(c) gold-chunk recoverability from structure alone", [
        ("every gold chunk contains a relation endpoint (pooled chunks)", ("c_gold_has_endpoint", "pooled")),
        ("endpoint ranking: all gold chunks in top-1", ("c_ep_gold_rank", "all_gold_within_top1")),
        ("endpoint ranking: all gold chunks in top-5", ("c_ep_gold_rank", "all_gold_within_top5")),
        ("relation-string ranking: all gold chunks in top-1", ("c_rs_gold_rank", "all_gold_within_top1")),
        ("relation-string ranking: all gold chunks in top-5", ("c_rs_gold_rank", "all_gold_within_top5")),
    ], med_cols)
    table("(d) text grounding of gold relations", [
        ("triple s and o both verbatim in one evidence sentence", ("d_triple_both_in_ev_sentence", "pooled")),
        ("triple s and o both verbatim in a gold chunk", ("d_triple_both_in_gold_chunk", "pooled")),
        ("triple s and o both verbatim in some corpus chunk", ("d_triple_both_in_some_chunk", "pooled")),
        ("relation string verbatim in corpus", ("d_rel_verbatim_in_corpus", "pooled")),
        ("relation content-token recall, best chunk (mean)", ("d_rel_best_chunk_recall", "mean")),
        ("  share of relations with recall 1.0", ("d_rel_best_chunk_recall", "frac_eq_1")),
        ("evidence sentence verbatim in corpus", ("d_ev_verbatim_in_corpus", "pooled")),
        ("evidence sentence recall, best chunk (mean)", ("d_ev_best_chunk_recall", "mean")),
        ("evidence sentence recall by the question's relations (mean)", ("d_ev_recall_by_rels", "mean")),
    ], med_cols)
    if "grb_novel_full" in S:
        A("## GraphRAG-Bench novel (full set, explicit triples)\n")
        A(f"Triples: {S['meta']['grb_novel'].get('n_triples')} raw, {S['meta']['grb_novel'].get('n_triples_unparsed_or_ambiguous')} "
          "unparsed or ambiguous (not exactly two commas) and excluded.\n")
        types = sorted(S["grb_novel_full_by_type"])
        cols = [("all", "grb_novel_full")]
        for t in types:
            S["_nov_" + t] = S["grb_novel_full_by_type"][t]
            cols.append((t, "_nov_" + t))
        table("novel", [
            ("evidence sentence contains s and o of a triple", ("a_ev_contains_triple_endpoints", "pooled")),
            ("answer == triple object", ("b_ans_eq_object",)),
            ("answer == triple endpoint", ("b_ans_eq_endpoint",)),
            ("answer contains a triple object", ("b_ans_contains_object",)),
            ("triple s and o in one evidence sentence", ("d_triple_both_in_ev_sentence", "pooled")),
            ("triple s and o in one source-novel sentence", ("d_triple_both_in_source_sentence", "pooled")),
            ("triple s and o anywhere in source novel", ("d_triple_s_and_o_in_source_text", "pooled")),
            ("evidence sentence verbatim in source novel", ("d_ev_verbatim_in_source", "pooled")),
        ], cols)
        for t in types:
            S.pop("_nov_" + t)
    A("## 2WikiMultiHop reference (same code)\n")
    table("2Wiki", [
        ("evidence triple present in aligned KG", ("a_rel_in_aligned_kg", "pooled")),
        ("gold passage linked by aligned KG", ("a_gold_linked_by_aligned_kg", "pooled")),
        ("supporting sentence contains s and o of a triple", ("a_ev_contains_triple_endpoints", "pooled")),
        ("answer == triple object (normalized)", ("b_ans_eq_object",)),
        ("answer == triple object, alias-aware", ("b_ans_eq_object_alias_aware",)),
        ("every gold passage contains an endpoint", ("c_gold_has_endpoint", "pooled")),
        ("endpoint ranking: all gold in top-5", ("c_ep_gold_rank", "all_gold_within_top5")),
        ("triple s and o in one supporting sentence", ("d_triple_both_in_ev_sentence", "pooled")),
        ("title = s and o in supporting sentence", ("d_triple_title_subject_object_in_sentence", "pooled")),
        ("triple s and o in a gold passage", ("d_triple_both_in_gold_chunk", "pooled")),
    ], [("sub200", "w2_sub200"), ("holdout200", "w2_holdout200")])
    A("## By question type (GraphRAG-Bench medical, full raw)\n")
    A("| Type | n | ans==rel | rel in ans | ans recall by rels | ans recall by gold chunks | rel verbatim in corpus | ev verbatim in corpus | ev realizes rel |")
    A("|---|---|---|---|---|---|---|---|---|")
    for t, d in S["grb_med_full_by_type"].items():
        A(f"| {t} | {d['n_questions']} | {fmt(g(d,'b_ans_eq_rel'))} | {fmt(g(d,'b_rel_in_ans'))} | {fmt(g(d,'b_ans_recall_by_rels','mean'))} | "
          f"{fmt(g(d,'b_ans_recall_by_gold_chunks','mean'))} | {fmt(g(d,'d_rel_verbatim_in_corpus','pooled'))} | "
          f"{fmt(g(d,'d_ev_verbatim_in_corpus','pooled'))} | {fmt(g(d,'a_ev_realizes_rel','pooled'))} |")
    A("")
    A("Caveat: GraphRAG-Bench releases no passage-level gold; gold chunks are the swap builder's mapping of the "
      "released `evidence` sentences to 120-word chunks (content-token recall >= 0.6). The aligned KG on the splits is an "
      "OpenIE re-extraction from `evidence_relations`, not the benchmark's unreleased ontology graph, so endpoint-level "
      "tests (a, c, d) rest on that re-extraction.\n")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    S1, R1 = build()
    S2, R2 = build()
    h1 = hashlib.sha256(json.dumps([S1, R1], sort_keys=True).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps([S2, R2], sort_keys=True).encode()).hexdigest()
    assert h1 == h2, "non-deterministic trace"
    S1["meta"]["output_sha256"] = h1
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_question.jsonl", "w", encoding="utf-8") as fh:
        for r in R1:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(S1, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_md(S1)
    print(json.dumps({k: S1[k] for k in ("meta", "release_order")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
