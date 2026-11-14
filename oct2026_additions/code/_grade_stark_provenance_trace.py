# -*- coding: utf-8 -*-
"""Provenance trace for GRADE (Lee et al., Findings of EMNLP 2025) and STaRK
(Wu et al., NeurIPS 2024 D&B), following eval/_grb_provenance_trace.py.

Goal: move the census evidence for the KG-derived axes of both benchmarks from
"construction description + quote" (tier D) to "traced to primary sources" (T)
or "exact from released data/code" (E):
  GRADE  edge axis (census bin 3) and passage axis (bin 1)
  STaRK  answer-entity axis (bin 3) and textual axis (bin 1)

Everything is deterministic: no model calls. Null baselines use a seeded
numpy Generator (SEED) and are recomputed identically on every run.

Inputs (not in the repo; recreate under GS_TRACE_DIR, default =
external/grade_stark_trace/ under paper/; md5s are recorded in summary.json):
  grade/    git clone https://github.com/DaeyongKwon98/GRADE        (HEAD d4b16af; code added ab6c433)
  stark/    git clone https://github.com/snap-stanford/stark        (HEAD b246b94)
  papers/   arXiv 2508.16994v1, v2; ACL Anthology 2025.findings-emnlp.236.pdf;
            arXiv 2404.13207v1, v2, v3; NeurIPS 2024 D&B camera-ready (proceedings.neurips.cc)
  qa/       HF snap-stanford/stark main: qa/{amazon,mag,prime}/stark_qa/stark_qa.csv
  qa_hist/  GitHub stark@dfafa99 data/*/stark_qa/stark_qa.csv (first release, 2024-04-23);
            HF revisions f721c636 and 01ee1e10 of qa/amazon/stark_qa/stark_qa.csv
  skb/mag/processed/                          HF skb/mag/processed.zip
  skb/amazon/processed/cache/brand-category-color/   HF skb/amazon/processed.zip
  Prime SKB: paper/external/stark/raw/processed (HF skb/prime/processed.zip)
  openreview/stark_QSS5cGmKb1_notes.json   Wayback 20260430203713 capture of api2 notes
  hf_trees/  HF tree listings at revisions 8b07fb29..88269e23

Normalization and content tokens are imported from _grb_provenance_trace.py
(norm, ctoks, STOP), so lexical rules match the GraphRAG-Bench trace. `stem`
strips a plural suffix (ies->y, s) and is applied on both sides of every
token comparison below.

STaRK tests (per query; anchors are recovered from the query text because the
per-query template and anchor entities are not released):
  anchors   SKB nodes of a template start type whose name occurs in the query
            (normalized word-boundary phrase; gene/protein symbols
            case-sensitive; Prime also token-set match for names with >= 2
            content tokens). Gold answers are never anchors.
  R1 relaxed  all gold answers reachable from ONE detected anchor by ONE branch
            of a template listed in the paper appendix (A.2-A.4).
  R2 strict   (Prime, MAG) all gold answers satisfy every conjunct of one full
            listed template, anchors filling each slot (distinct anchors for
            two-slot templates), incl. the negation in Prime template 1.
  R3 1-hop    all gold answers adjacent (any relation) to one detected anchor.
  null      the same test with the gold set replaced by |gold| random nodes of
            the gold type (NULL_REPS seeded draws per query, same anchors).
  retention |gold| / |branch result of the tightest covering (anchor, branch)|
            (Amazon/MAG 1-hop branches and Prime 1-hop branches only).
  T1 textual residual query content tokens (minus detected anchor-name tokens
            and a fixed template-word list) recalled by the gold node's own
            text; percentile of each gold node among a seeded pool of POOL
            same-type nodes (share of pool with recall >= gold; lower = gold
            covers the query's text better than typical nodes).
  T2 seed   (Amazon only) the first release (GitHub dfafa99 / HF f721c636)
            keyed every query by a node id that is always one of its answers;
            recall of that node vs the other gold answers of the same query.

Outputs: results/grade_stark_provenance/{summary.json, summary.md,
per_query_stark.jsonl, sources/, inputs/}
Run: python paper/eval/_grade_stark_provenance_trace.py [--verify]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "results" / "grade_stark_provenance"
SRC_OUT = OUT / "sources"
INP_OUT = OUT / "inputs"
DEFAULT_TRACE = PAPER / "external" / "grade_stark_trace"  # inputs listed in the docstring; GS_TRACE_DIR overrides
TRACE = Path(os.environ.get("GS_TRACE_DIR", str(DEFAULT_TRACE)))
PRIME_DIR = PAPER / "external" / "stark" / "raw" / "processed"
MAG_DIR = TRACE / "skb" / "mag" / "processed"
AMZ_DIR = TRACE / "skb" / "amazon" / "processed" / "cache" / "brand-category-color"
SEED = 20260930
NULL_REPS = 10
POOL = 2000


def _load_grb():
    p = PAPER / "eval" / "_grb_provenance_trace.py"
    spec = importlib.util.spec_from_file_location("_grb_trace", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GRB = _load_grb()
norm, STOP = GRB.norm, GRB.STOP


def stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def sctoks(s) -> set:
    return {stem(t) for t in GRB.ctoks(s)}


def md5(p: Path, chunk=1 << 24) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def frac(num, den):
    return {"num": int(num), "den": int(den), "frac": round(num / den, 4) if den else None}


def rnd(x, k=4):
    return None if x is None else round(float(x), k)


def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


# ---------------------------------------------------------------------------
# Paper quotes: located per version (LOOSE = letters+digits containment, as in
# human_validation/llm_census_logged/excerpts.json); 1-indexed PDF pages.
# ---------------------------------------------------------------------------
PAPER_FILES = {
    "grade_v1": "grade_v1.pdf", "grade_v2": "grade_v2.pdf", "grade_findings": "grade_findings.pdf",
    "stark_v1": "stark_v1.pdf", "stark_v2": "stark_v2.pdf", "stark_v3": "stark_v3.pdf",
    "stark_neurips": "stark_neurips.pdf",
}
QUOTES = [
    # GRADE
    ("G-release-v1", "grade", "We release our code and data at https://github.com/anonymous/anonymous"),
    ("G-release-v2", "grade", "We release our code to facilitate reproducibility"),
    ("G-kg", "grade", "The knowledge graph is composed of entities and relationships extracted from factual claims."),
    ("G-augment", "grade", "we augmented the knowledge graph by adding mirrored inbound and outbound edges"),
    ("G-paths", "grade", "We first enumerated all shortest directed acyclic paths of length 2 to 5 within the knowledge graph"),
    ("G-validation", "grade", "an additional validation step was done to remove ambiguous QA pairs"),
    ("G-edgeset-v2", "grade", "all supporting facts are contained within the edge set P"),
    ("G-edgeset-v1", "grade", "all supporting facts are contained in the edge set P"),
    ("G-chunks", "grade", "Retriever-side difficulty was defined based on the semantic distance between a question q and its associated supporting chunks"),
    ("G-segment", "grade", "we first segmented all articles into overlapping chunks and embedded them using a frozen encoder"),
    ("G-human-limit", "grade", "Future work may benefit from integrating more robust validation mechanisms, such as human-in-the-loop verification"),
    ("G-human-A6", "grade", "we manually analyzed 30 randomly sampled entity groups from the augmentation phase"),
    # STaRK
    ("S-template-v1", "stark", "Initially, we sample a relation template, such as"),
    ("S-candidates-v1", "stark", "Each relational requirement yields a set of candidate entities that meet the requirement"),
    ("S-gold-v1", "stark", "Subsequently, we select one of the candidate nodes that satisfies the relational requirement, which is referred to as the gold answer."),
    ("S-extract-v1", "stark", "We extract textual properties that align with the interests of specific roles from the textual information of the gold answer."),
    ("S-filter-v1", "stark", "we assess whether each remaining candidate, excluding the gold answer used for extracting textual properties, meets the additional textual requirement. For the verification, we employ three Claude models."),
    ("S-exclude-gold", "stark", "excluding the gold answer used for extracting textual properties"),
    ("S-gold-v3", "stark", "We randomly sample a candidate entity from the first step, referred to as the gold answer, from which LLMs extract properties"),
    ("S-filter-v3", "stark", "we employ multiple LLMs to verify if the candidates from the first step meet the extracted textual properties. Only candidates passing all LLM verifications are included in the final ground truth set."),
    ("S-verify-rate", "stark", "86.6%, 98.9%"),
    ("S-human-gen", "stark", "We manually verified and filtered the ground truth answers to ensure the answer correctness."),
    ("S-human-eval", "stark", "rated the queries neutral or above in naturalness, diversity, and practicality"),
    ("S-prime-paraphrase", "stark", "Ensure that the query uses different descriptions than the original property descriptions while retaining similar meanings."),
    ("S-prime-relational-only", "stark", "the queries rely solely on relational data due to limited textual information"),
    ("S-task-def", "stark", "it satisfies the relational requirements imposed by the structure of G as specified in Q, and the associated documents Di satisfy the textual requirements specified in Q."),
    ("S-step4-llms", "stark", "claude-2.1, claude-2.0, claude-instant-1.2"),
]


def _loose(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def paper_quotes():
    import fitz  # PyMuPDF
    pages = {}
    meta = {}
    for key, fn in PAPER_FILES.items():
        p = TRACE / "papers" / fn
        doc = fitz.open(p)
        pages[key] = [_loose(pg.get_text()) for pg in doc]
        meta[key] = {"file": fn, "md5": md5(p), "pages": len(doc)}
    rows = []
    for qid, bench, q in QUOTES:
        lq = _loose(q)
        found = {}
        for key, pl in pages.items():
            if not key.startswith(bench):
                continue
            hits = [i + 1 for i, t in enumerate(pl) if lq in t]
            if not hits:  # quote broken across a page boundary
                hits = [f"{i + 1}-{i + 2}" for i in range(len(pl) - 1) if lq in pl[i] + pl[i + 1]]
            found[key] = hits
        rows.append({"id": qid, "quote": q, "pages": found})
    return rows, meta


# ---------------------------------------------------------------------------
# GRADE: static trace of the released construction code (no data released).
# Each fact is an exact code string located by file and line at a pinned commit.
# ---------------------------------------------------------------------------
GRADE_FACTS = [
    ("chunk_is_article", "code/ingestion/crawl_news.py",
     "df = df[(df['token_count']>=512) & (df['token_count']<=8192)]",
     "the unit written to chunk.csv is a whole news article of 512-8192 tokens"),
    ("chunk_is_article2", "code/ingestion/crawl_news.py",
     "df.to_csv(f\"{category}/chunk.csv\", index=False)", "articles saved as chunk.csv"),
    ("sentence_chunk_id", "code/ingestion/crawl_news.py",
     "chunk_id = idx  # use DataFrame index as chunk_id", "each sentence keeps the index of its source article"),
    ("claim_from_sentence", "code/claims/generate_claim_from_sentence.py",
     "user_prompt = f\"\"\"Context: {row['chunk']}", "claims are LLM rewrites of fact sentences, with the article as context"),
    ("triples_from_claims", "code/triples/extract_triples.py",
     "for i, sentence in enumerate(batch['claim']):", "triples are LLM-extracted from claims"),
    ("mirror_copy_row", "code/clustering/add_contextually_same_path.py",
     "new_row = row.to_dict()", "a mirrored edge copies the original row (claim, sentence, chunk_id)"),
    ("mirror_flag", "code/clustering/add_contextually_same_path.py",
     "new_row['is_added'] = True", "mirrored edges are flagged is_added"),
    ("graph_from_triples", "code/graph/find_DAG_path_shortest.py",
     "G.add_edge(row['source'], row['target'], relationship=row['relationship'])",
     "the KG is a DiGraph built from the (augmented, merged) triple table"),
    ("all_shortest_paths", "code/graph/find_DAG_path_shortest.py",
     "for path in nx.all_shortest_paths(G, start_node, end_node):", "all shortest paths are enumerated between every node pair"),
    ("edge_from_graph", "code/graph/find_DAG_path_shortest.py",
     "rel = G[src][tgt]['relationship']", "every path triple is read from the graph edge"),
    ("chunks_by_triple", "code/graph/find_DAG_path_shortest.py",
     "if key in triple_lookup:", "supporting chunks are looked up by the path's triples"),
    ("chunk_body", "code/graph/find_DAG_path_shortest.py",
     "body = chunk_id_to_body.get(info['chunk_id'])", "the chunk attached to a triple is its source article body"),
    ("hops_filter", "code/sampling/sample_path.py",
     "path = path[(path['hops']>=2) & (path['hops']<=5)]", "2-5 hop paths kept"),
    ("dedup_endpoints", "code/sampling/sample_path.py",
     "path.drop_duplicates(subset=['first_entity', 'last_entity'], inplace=True)", "one path per (start, end) pair"),
    ("sample_400", "code/sampling/sample_path.py",
     "one_sample = one.sample(n=400, random_state=42)", "400 paths per hop"),
    ("qa_input", "code/qa/qa_generation.py",
     "user_prompt = f\"Triples: {row['triples']}\\n\\nClaims: {row['claims']}\\n\\nContext: {row['chunks']}\"",
     "the QA generator sees the path triples, their claims and their source articles"),
    ("qa_from_triples", "code/qa/qa_generation.py",
     "- Do not add external knowledge or assumptions beyond the given triples.", "answers derived from the triples"),
    ("validation_llm", "code/qa/final_validation.py",
     "model=\"o1-mini\",", "the only post-generation filter is an LLM ambiguity check"),
    ("final_rows", "code/qa/final_validation.py",
     "selected_rows = pd.concat([metadata, qa], axis=1).loc[indices]",
     "released QA rows carry the path metadata (triples, claims, chunks) unchanged"),
    ("db_after", "code/db/generate_db_from_chunk.py",
     "chunk_size=256,", "the retrieval DB re-splits the articles into 256-token chunks (pipeline step 14, after QA)"),
    ("dr_uses_chunks", "code/eval/retriever_correlation.py",
     "for chunk_text in row['chunks']:", "retrieval difficulty is computed against the path's source articles"),
    ("dr_power_mean", "code/eval/retriever_correlation.py",
     "p_value = -3", "code uses a power mean (p=-3) where paper Eq. 1 uses the minimum"),
    ("judge_only", "code/eval/score_llm_eval.py",
     "user_prompt = f\"Question: {row['question']}\\nGround Truth Answer: {row['answer']}\\nResponse: {row['rag_answer']}\"",
     "evaluation scores answers by LLM judge; no overlap scoring of edges or chunks"),
]


def grade_trace():
    repo = TRACE / "grade"
    head = git(repo, "rev-parse", "HEAD").strip()
    log = git(repo, "log", "--format=%h|%ad|%s", "--date=iso").strip().splitlines()
    files = git(repo, "ls-files").split()
    code_commit = git(repo, "log", "--diff-filter=A", "--format=%h|%ad", "--date=iso", "--",
                      "code/graph/find_DAG_path_shortest.py").strip()
    facts = []
    for fid, rel, needle, meaning in GRADE_FACTS:
        lines = (repo / rel).read_text(encoding="utf-8").splitlines()
        hits = [i + 1 for i, l in enumerate(lines) if needle in l]
        if not hits:
            raise SystemExit(f"GRADE fact {fid} not found in {rel}: {needle!r}")
        facts.append({"id": fid, "file": rel, "lines": hits, "code": needle, "meaning": meaning})
    data_ext = {".csv", ".json", ".jsonl", ".tsv", ".parquet", ".pkl", ".zip", ".txt"}
    data_files = [f for f in files if Path(f).suffix.lower() in data_ext and not f.startswith("requirements")]
    # pin the traced code files
    SRC_OUT.mkdir(parents=True, exist_ok=True)
    for rel in sorted({f[1] for f in GRADE_FACTS}):
        body = (repo / rel).read_text(encoding="utf-8")
        (SRC_OUT / ("grade_" + head[:7] + "_" + rel.replace("/", "__") + ".txt")).write_text(
            f"# DaeyongKwon98/GRADE @ {head} :: {rel}\n" + body, encoding="utf-8")
    return {"repo": "https://github.com/DaeyongKwon98/GRADE", "head": head, "commits": log,
            "construction_code_added": code_commit, "tracked_files": len(files),
            "tracked_data_files": data_files, "facts": facts}


# ---------------------------------------------------------------------------
# STaRK: release history
# ---------------------------------------------------------------------------
def read_qa(p: Path):
    import pandas as pd
    df = pd.read_csv(p)
    return [(int(i), str(q), [int(x) for x in ast.literal_eval(a)]) for i, q, a in
            zip(df["id"], df["query"], df["answer_ids"])]


def stark_history():
    repo = TRACE / "stark"
    out = {"repo_head": git(repo, "rev-parse", "HEAD").strip()}
    all_files = sorted(set(git(repo, "log", "--all", "--name-only", "--format=").split()))
    gen_like = [f for f in all_files if re.search(r"generat|synth|templat|query_gen|construct", f, re.I)]
    out["files_ever_tracked"] = len(all_files)
    out["generation_like_paths"] = gen_like
    kb = (repo / "stark_qa" / "skb" / "knowledge_base.py").read_text(encoding="utf-8").splitlines()
    out["kb_sampler_lines"] = {n: [i + 1 for i, l in enumerate(kb) if l.strip().startswith(f"def {n}(")]
                               for n in ("sample_paths", "get_all_paths", "get_tuples")}
    first = git(repo, "show", "df59d09:src/benchmarks/semistruct/knowledge_base.py").splitlines()
    out["kb_sampler_lines_first_commit_df59d09"] = {
        n: [i + 1 for i, l in enumerate(first) if l.strip().startswith(f"def {n}(")]
        for n in ("sample_paths", "get_all_paths")}
    hq = git(repo, "show", "df59d09:src/benchmarks/qa_datasets/hybrid_qa.py").splitlines()
    out["first_loader_meta_info"] = [i + 1 for i, l in enumerate(hq) if "meta_info" in l]
    # pin the sampler source
    SRC_OUT.mkdir(parents=True, exist_ok=True)
    (SRC_OUT / "stark_knowledge_base_py.txt").write_text(
        f"# snap-stanford/stark @ {out['repo_head']} :: stark_qa/skb/knowledge_base.py\n" + "\n".join(kb),
        encoding="utf-8")
    (SRC_OUT / "stark_df59d09_hybrid_qa_py.txt").write_text(
        "# snap-stanford/stark @ df59d09 :: src/benchmarks/qa_datasets/hybrid_qa.py\n" + "\n".join(hq),
        encoding="utf-8")
    # HF tree listings: which files ever existed
    trees = {}
    for p in sorted((TRACE / "hf_trees").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        trees[p.stem] = sorted(x["path"] for x in d if x.get("type") == "file")
    out["hf_revisions"] = {k: len(v) for k, v in trees.items()}
    qa_files = sorted({f for v in trees.values() for f in v if f.startswith("qa/")})
    out["hf_qa_files_any_revision"] = qa_files
    (SRC_OUT / "hf_trees.json").write_text(json.dumps(trees, indent=1), encoding="utf-8")
    # QA files: columns and cross-release comparison
    qa = {}
    cmp_ = {}
    for sub, hist in (("amazon", "gh_dfafa99_amazon.csv"), ("mag", "gh_dfafa99_mag.csv"),
                      ("prime", "gh_dfafa99_primekg.csv")):
        cur = TRACE / "qa" / f"{sub}_stark_qa.csv"
        old = TRACE / "qa_hist" / hist
        header = cur.read_text(encoding="utf-8").splitlines()[0]
        a, b = read_qa(cur), read_qa(old)
        qa[sub] = a
        bq = {q: (i, ans) for i, q, ans in b}
        same_q = sum(q in bq for _, q, _ in a)
        same_ans = sum(q in bq and sorted(bq[q][1]) == sorted(ans) for _, q, ans in a)
        same_id = sum(q in bq and bq[q][0] == i for i, q, _ in a)
        id_in_ans_old = sum(i in ans for i, _, ans in b)
        id_in_ans_new = sum(i in ans for i, _, ans in a)
        cmp_[sub] = {"current_md5": md5(cur), "first_release_md5": md5(old), "columns": header,
                     "n": len(a), "same_query": frac(same_q, len(a)), "same_answer_set": frac(same_ans, len(a)),
                     "same_id": frac(same_id, len(a)),
                     "first_release_id_in_answers": frac(id_in_ans_old, len(b)),
                     "current_id_in_answers": frac(id_in_ans_new, len(a))}
    for rev in ("hf_f721c636_amazon.csv", "hf_01ee1e10_amazon.csv"):
        p = TRACE / "qa_hist" / rev
        r = read_qa(p)
        cmp_["amazon"][rev] = {"md5": md5(p), "id_in_answers": frac(sum(i in ans for i, _, ans in r), len(r))}
    out["qa"] = cmp_
    # Amazon seed map: current id -> first-release id (always an answer)
    old = {q: i for i, q, _ in read_qa(TRACE / "qa_hist" / "gh_dfafa99_amazon.csv")}
    seed = {i: old[q] for i, q, _ in qa["amazon"]}
    INP_OUT.mkdir(parents=True, exist_ok=True)
    with open(INP_OUT / "amazon_first_release_ids.csv", "w", encoding="utf-8") as fh:
        fh.write("id_current,id_first_release\n")
        for i in sorted(seed):
            fh.write(f"{i},{seed[i]}\n")
    pos = Counter(next(k for k, x in enumerate(ans) if x == seed[i]) for i, _, ans in qa["amazon"])
    out["amazon_seed_position_in_answer_list"] = {str(k): v for k, v in sorted(pos.items())[:6]}
    return out, qa, seed


def openreview_record():
    p = TRACE / "openreview" / "stark_QSS5cGmKb1_notes.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    SRC_OUT.mkdir(parents=True, exist_ok=True)
    (SRC_OUT / "openreview_forum_QSS5cGmKb1_notes.json").write_text(json.dumps(d, indent=1, ensure_ascii=False),
                                                                     encoding="utf-8")
    notes = d["notes"]
    kinds = Counter(n["invitations"][0].split("/-/")[-1] for n in notes)
    hits = []
    pat = re.compile(r"[^.\n]*(three LLMs to validate|human[- ]generated|manually|false negative|ground[- ]truth answer)[^.\n]*",
                     re.I)
    for n in notes:
        for k, v in n["content"].items():
            v = v.get("value") if isinstance(v, dict) else v
            if isinstance(v, str):
                for m in pat.finditer(v):
                    hits.append({"note": n["id"], "by": n["signatures"][0].split("/")[-1], "field": k,
                                 "text": m.group(0).strip()[:400]})
    return {"forum": "QSS5cGmKb1", "source": "web.archive.org capture 20260430203713 of api2.openreview.net notes",
            "md5": md5(p), "notes": len(notes), "kinds": dict(kinds), "matches": hits}


# ---------------------------------------------------------------------------
# SKB loading
# ---------------------------------------------------------------------------
class SKB:
    def __init__(self, d: Path, name: str, rel_short=None):
        import torch
        from scipy import sparse
        t0 = time.time()
        self.name = name
        self.edge_type_dict = pickle.load(open(d / "edge_type_dict.pkl", "rb"))
        self.node_type_dict = pickle.load(open(d / "node_type_dict.pkl", "rb"))
        self.ntype = torch.load(d / "node_types.pt").numpy().astype(np.int16)
        ei = torch.load(d / "edge_index.pt").numpy()
        et = torch.load(d / "edge_types.pt").numpy()
        n = len(self.ntype)
        self.N = n
        self.type_id = {v: k for k, v in self.node_type_dict.items()}
        rel_short = rel_short or {}
        self.rel_id = {rel_short.get(v, v): k for k, v in self.edge_type_dict.items()}
        # symmetrize (STaRK SKBs are used as undirected: knowledge_base.py to_undirected)
        src = np.concatenate([ei[0], ei[1]])
        dst = np.concatenate([ei[1], ei[0]])
        ety = np.concatenate([et, et])
        self.adj = {}
        for r in np.unique(ety):
            m = ety == r
            A = sparse.csr_matrix((np.ones(m.sum(), dtype=np.int8), (src[m], dst[m])), shape=(n, n))
            A.data[:] = 1
            A.sort_indices()
            self.adj[int(r)] = A
        Aall = sparse.csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
        Aall.sort_indices()
        self.all = Aall
        self.md5 = {f: md5(d / f) for f in ("edge_index.pt", "edge_types.pt", "node_types.pt",
                                            "edge_type_dict.pkl", "node_type_dict.pkl")}
        self._m1 = {}
        self.load_s = round(time.time() - t0, 1)

    def nbr(self, a, rel=None):
        A = self.all if rel is None else self.adj[self.rel_id[rel]]
        return A.indices[A.indptr[a]:A.indptr[a + 1]]

    def of_type(self, ids, tname):
        ids = np.asarray(ids, dtype=np.int64)
        return ids[self.ntype[ids] == self.type_id[tname]]

    def reach(self, a, branch, g):
        """g reachable from anchor a by a typed branch ((rel, type), ...), len 1 or 2."""
        if self.ntype[g] != self.type_id[branch[-1][1]]:
            return False
        if len(branch) == 1:
            nb = self.nbr(a, branch[0][0])
            i = np.searchsorted(nb, g)
            return bool(i < len(nb) and nb[i] == g)
        (r1, t1), (r2, _) = branch
        key = (a, r1, t1)
        m1 = self._m1.get(key)
        if m1 is None:
            m1 = self.of_type(self.nbr(a, r1), t1)
            if len(self._m1) > 200000:
                self._m1.clear()
            self._m1[key] = m1
        m2 = self.nbr(g, r2)
        if len(m1) == 0 or len(m2) == 0:
            return False
        return bool(np.intersect1d(m1, m2, assume_unique=True).size)

    def adjacent(self, a, g):
        nb = self.nbr(a)
        i = np.searchsorted(nb, g)
        return bool(i < len(nb) and nb[i] == g)

    def branch_size(self, a, branch):
        if len(branch) != 1:
            return None
        return int(len(self.of_type(self.nbr(a, branch[0][0]), branch[0][1])))


# Templates from the paper appendix (v3 A.2-A.4). Each template: list of slots
# (slot_name, start_type, branch); the answer is the branch end. `neg` is an
# extra constraint on the answer.
PRIME_T = {
    1: {"slots": [("x", "effect/phenotype", (("phenotype absent", "disease"),))], "neg": ("indication", "drug")},
    2: {"slots": [("x", "drug", (("contraindication", "disease"),)), ("y", "gene/protein", (("associated with", "disease"),))]},
    3: {"slots": [("x", "anatomy", (("expression present", "gene/protein"),)), ("y", "anatomy", (("expression absent", "gene/protein"),))]},
    4: {"slots": [("x", "anatomy", (("expression absent", "gene/protein"),)), ("y", "anatomy", (("expression absent", "gene/protein"),))]},
    5: {"slots": [("x", "drug", (("carrier", "gene/protein"),)), ("y", "drug", (("carrier", "gene/protein"),))]},
    6: {"slots": [("x", "anatomy", (("expression present", "gene/protein"), ("target", "drug")))]},
    7: {"slots": [("x", "drug", (("side effect", "effect/phenotype"), ("side effect", "drug")))]},
    8: {"slots": [("x", "drug", (("carrier", "gene/protein"), ("carrier", "drug")))]},
    9: {"slots": [("x", "anatomy", (("expression present", "gene/protein"), ("enzyme", "drug")))]},
    10: {"slots": [("x", "cellular_component", (("interacts with", "gene/protein"), ("carrier", "drug")))]},
    11: {"slots": [("x", "molecular_function", (("interacts with", "gene/protein"), ("target", "drug")))]},
    12: {"slots": [("x", "effect/phenotype", (("side effect", "drug"), ("synergistic interaction", "drug")))]},
    13: {"slots": [("x", "disease", (("indication", "drug"), ("contraindication", "disease")))]},
    14: {"slots": [("x", "disease", (("parent-child", "disease"), ("phenotype present", "effect/phenotype")))]},
    15: {"slots": [("x", "gene/protein", (("transporter", "drug"), ("side effect", "effect/phenotype")))]},
    16: {"slots": [("x", "drug", (("transporter", "gene/protein"), ("interacts with", "exposure")))]},
    17: {"slots": [("x", "pathway", (("interacts with", "gene/protein"), ("ppi", "gene/protein")))]},
    18: {"slots": [("x", "drug", (("synergistic interaction", "drug"), ("transporter", "gene/protein")))]},
    19: {"slots": [("x", "biological_process", (("interacts with", "gene/protein"), ("interacts with", "biological_process")))]},
    20: {"slots": [("x", "effect/phenotype", (("associated with", "gene/protein"), ("interacts with", "biological_process")))]},
    21: {"slots": [("x", "drug", (("transporter", "gene/protein"), ("expression present", "anatomy")))]},
    22: {"slots": [("x", "drug", (("target", "gene/protein"), ("interacts with", "cellular_component")))]},
    23: {"slots": [("x", "biological_process", (("interacts with", "gene/protein"), ("expression absent", "anatomy")))]},
    24: {"slots": [("x", "effect/phenotype", (("associated with", "gene/protein"), ("expression absent", "anatomy")))]},
    25: {"slots": [("x", "drug", (("indication", "disease"), ("indication", "drug"))), ("x", "drug", (("synergistic interaction", "drug"),))]},
    26: {"slots": [("x", "pathway", (("interacts with", "gene/protein"), ("interacts with", "pathway"))), ("x", "pathway", (("parent-child", "pathway"),))]},
    27: {"slots": [("x", "gene/protein", (("associated with", "disease"), ("associated with", "gene/protein"))), ("x", "gene/protein", (("ppi", "gene/protein"),))]},
    28: {"slots": [("x", "gene/protein", (("associated with", "effect/phenotype"), ("associated with", "gene/protein"))), ("x", "gene/protein", (("ppi", "gene/protein"),))]},
}
MAG_REL = {"author___affiliated_with___institution": "affiliated_with", "paper___cites___paper": "cites",
           "paper___has_topic___field_of_study": "has_topic", "author___writes___paper": "writes"}
MAG_T = {
    1: {"slots": [("x", "author", (("writes", "paper"),))]},
    2: {"slots": [("x", "paper", (("cites", "paper"),))]},
    3: {"slots": [("x", "field_of_study", (("has_topic", "paper"),))]},
    4: {"slots": [("x", "institution", (("affiliated_with", "author"), ("writes", "paper")))]},
    5: {"slots": [("x", "paper", (("writes", "author"), ("writes", "paper")))]},
    6: {"slots": [("x", "paper", (("writes", "author"), ("writes", "paper"))), ("x", "paper", (("has_topic", "field_of_study"), ("has_topic", "paper")))]},
    7: {"slots": [("x", "institution", (("affiliated_with", "author"), ("writes", "paper"))), ("y", "field_of_study", (("has_topic", "paper"),))]},
}
AMZ_T = {
    1: {"slots": [("x", "brand", (("has_brand", "product"),))]},
    2: {"slots": [("x", "product", (("also_buy", "product"),))]},
    3: {"slots": [("x", "product", (("also_view", "product"),))]},
    4: {"slots": [("x", "color", (("has_color", "product"),))]},
    5: {"slots": [("x", "category", (("has_category", "product"),))]},
}

TEMPLATE_WORDS = {
    "prime": """drug drugs medication medications medicine gene genes protein proteins disease diseases condition
      conditions disorder disorders pathway pathways anatomy anatomical phenotype phenotypes effect effects exposure
      exposures process processes component components function functions molecular biological cellular interact
      interacts interacting interaction interactions associated association expressed expression express target
      targets targeting targeted carrier carriers transporter transporters transport enzyme enzymes side synergistic
      indication indicated indications contraindication contraindicated related identify find list search looking need
      know could would please which what also both common shared share""",
    "mag": """paper papers publication publications published publish research researcher researchers study studies
      author authors authored coauthor coauthors co field fields cite cited cites citing citation institution
      university find list search looking work works article articles topic topics area areas related also
      similar same other some any show""",
    "amazon": """product products looking recommend recommendation suggest suggestion find need want something item
      items buy bought purchase brand color colour category similar also well good great like could would please
      help looking search""",
}
DOC_NOISE = {"reviewerid", "summary", "style", "reviewtext", "vote", "overall", "verified", "reviewtime", "nan",
             "true", "false", "questiontype", "answertype", "question", "answer", "answertime", "yes", "no"}


def build_name_index(names: dict, ok_types: set, skb: SKB, case_sensitive_types=()):
    """names: node -> raw name. Returns phrase index (normalized) and symbol index (raw, case-sensitive)."""
    phrase = defaultdict(list)
    symbol = defaultdict(list)
    tokset = defaultdict(list)  # rarest stemmed token -> [(frozenset, node)]
    df = Counter()
    entries = []
    for i, raw in names.items():
        tname = skb.node_type_dict[int(skb.ntype[i])]
        if tname not in ok_types or not raw:
            continue
        if tname in case_sensitive_types:
            if len(raw) >= 2:
                symbol[raw].append(i)
            continue
        n = norm(raw)
        if len(n) < 3 or n in STOP or n.isdigit():
            continue
        phrase[n].append(i)
        ts = frozenset(stem(t) for t in n.split() if len(t) > 1 and t not in STOP)
        if len(ts) >= 2:
            entries.append((ts, i))
            df.update(ts)
    for ts, i in entries:
        key = min(ts, key=lambda t: (df[t], t))
        tokset[key].append((ts, i))
    maxlen = max((len(k.split()) for k in phrase), default=1)
    return {"phrase": phrase, "symbol": symbol, "tokset": tokset, "maxlen": min(maxlen, 60)}


def find_anchors(q: str, idx, exclude: set, use_tokset: bool):
    toks = norm(q).split()
    found = {}
    L = len(toks)
    for i in range(L):
        for n in range(1, min(idx["maxlen"], L - i) + 1):
            ph = " ".join(toks[i:i + n])
            for node in idx["phrase"].get(ph, ()):
                if node not in exclude:
                    found[node] = "phrase"
    if idx["symbol"]:
        for t in set(re.split(r"[^A-Za-z0-9\-]+", q)):
            for node in idx["symbol"].get(t, ()):
                if node not in exclude:
                    found.setdefault(node, "symbol")
    if use_tokset:
        qs = {stem(t) for t in toks if len(t) > 1 and t not in STOP}
        for t in qs:
            for ts, node in idx["tokset"].get(t, ()):
                if node not in exclude and node not in found and ts <= qs:
                    found[node] = "tokset"
    return found


def eval_query(skb: SKB, T: dict, anchors: dict, gold, ans_type: str, names: dict):
    """Returns relaxed/strict/1-hop results for one gold set."""
    by_type = defaultdict(list)
    for a in anchors:
        by_type[skb.node_type_dict[int(skb.ntype[a])]].append(a)
    relaxed = []  # (template, anchor) covering all gold by one branch
    strict = []
    for tid, t in T.items():
        if t["slots"][-1][2][-1][1] != ans_type and t["slots"][0][2][-1][1] != ans_type:
            continue
        slot_cover = []
        for sname, stype, br in t["slots"]:
            cov = [a for a in by_type.get(stype, ()) if all(skb.reach(a, br, g) for g in gold)]
            for a in cov:
                relaxed.append((tid, a, br))
            slot_cover.append((sname, cov))
        # strict: every slot covered; same-name slots share the anchor; different names need distinct anchors
        names_ = [s for s, _ in slot_cover]
        ok = False
        if all(c for _, c in slot_cover):
            if len(set(names_)) == 1:
                common = set(slot_cover[0][1])
                for _, c in slot_cover[1:]:
                    common &= set(c)
                ok = bool(common)
            else:
                xs, ys = slot_cover[0][1], slot_cover[1][1]
                ok = any(x != y for x in xs for y in ys)
        if ok and "neg" in t:
            rel, _ = t["neg"]
            ok = all(len(skb.nbr(g, rel)) == 0 for g in gold)
        if ok:
            strict.append(tid)
    hop1 = [a for a in anchors if all(skb.adjacent(a, g) for g in gold)]
    tight = None
    sizes = [(skb.branch_size(a, br), tid, a) for tid, a, br in relaxed]
    sizes = [s for s in sizes if s[0]]
    if sizes:
        s, tid, a = min(sizes)
        tight = {"template": tid, "anchor": int(a), "anchor_name": names.get(a), "branch_size": s,
                 "retention": round(len(gold) / s, 4)}
    return {"relaxed": sorted({tid for tid, _, _ in relaxed}), "strict": strict, "hop1": len(hop1) > 0,
            "any": bool(relaxed) or len(hop1) > 0, "tight": tight}


class TextIndex:
    """Stemmed content-token sets for node texts, with a seeded same-type pool."""

    def __init__(self, text_fn, skb: SKB, types, rng, pool=POOL):
        self.text_fn = text_fn
        self.cache = {}
        self.pool = {}
        for t in types:
            ids = np.flatnonzero(skb.ntype == skb.type_id[t])
            sel = np.sort(rng.choice(ids, size=min(pool, len(ids)), replace=False))
            self.pool[t] = [int(x) for x in sel]
        from scipy import sparse
        self.sparse = sparse

    def toks(self, i):
        if i not in self.cache:
            self.cache[i] = frozenset(t for t in sctoks(self.text_fn(i)) if t not in DOC_NOISE)
        return self.cache[i]

    def build_pool_matrix(self):
        self.pool_mat = {}
        for t, ids in self.pool.items():
            vocab = {}
            rows, cols = [], []
            for r, i in enumerate(ids):
                for tok in self.toks(i):
                    c = vocab.setdefault(tok, len(vocab))
                    rows.append(r)
                    cols.append(c)
            M = self.sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                                       shape=(len(ids), max(len(vocab), 1))).tocsc()
            self.pool_mat[t] = (vocab, M, {j: k for k, j in enumerate(ids)})

    def pool_recalls(self, resid: set, t: str):
        vocab, M, _ = self.pool_mat[t]
        cols = [vocab[x] for x in resid if x in vocab]
        cnt = np.asarray(M[:, cols].sum(axis=1)).ravel() if cols else np.zeros(M.shape[0])
        return cnt / len(resid)

    def percentile(self, resid: set, i: int, t: str, pr=None):
        """share of pool nodes (excluding i) whose recall of resid >= node i's recall."""
        if not resid:
            return None, None
        rec = len(resid & self.toks(i)) / len(resid)
        if pr is None:
            pr = self.pool_recalls(resid, t)
        pos = self.pool_mat[t][2].get(i)
        if pos is not None:
            pr = np.delete(pr, pos)
        ge = float((pr >= rec - 1e-12).mean())          # pessimistic ties
        gt = float((pr > rec + 1e-12).mean())
        return rec, (ge, (ge + gt) / 2)                  # (pessimistic, mid-rank)


def best_cover(skb: SKB, nodes) -> float:
    arr = np.concatenate([skb.nbr(g) for g in nodes])
    if not len(arr):
        return 0.0
    return round(float(np.unique(arr, return_counts=True)[1].max()) / len(nodes), 4)


def run_subset(sub: str, skb: SKB, T: dict, qa, names: dict, text_fn, anchor_types: set, use_tokset: bool,
               case_sensitive_types=(), seed_map=None, tw=None, rng=None):
    t0 = time.time()
    idx = build_name_index(names, anchor_types, skb, case_sensitive_types)
    ans_types = sorted({skb.node_type_dict[int(skb.ntype[a[0]])] for _, _, a in qa})
    text = TextIndex(text_fn, skb, ans_types, rng)
    text.build_pool_matrix()
    type_ids = {t: np.flatnonzero(skb.ntype == skb.type_id[t]) for t in ans_types}
    tw = {stem(w) for w in tw.split()}
    recs = []
    for qid, q, gold in qa:
        gtypes = {skb.node_type_dict[int(skb.ntype[g])] for g in gold}
        at = sorted(gtypes)[0]
        anchors = find_anchors(q, idx, set(gold), use_tokset)
        r = {"subset": sub, "id": qid, "n_gold": len(gold), "answer_type": at, "gold_types": len(gtypes),
             "n_anchor": len(anchors), "n_anchor_phrase": sum(v != "tokset" for v in anchors.values())}
        res = eval_query(skb, T, anchors, gold, at, names) if anchors else \
            {"relaxed": [], "strict": [], "hop1": False, "any": False, "tight": None}
        r.update({k: res[k] for k in ("relaxed", "strict", "hop1", "any", "tight")})
        # null: same anchors, random same-type sets of the same size
        nr = ns = nh = na = 0
        if anchors:
            pool = type_ids[at]
            for _ in range(NULL_REPS):
                fake = [int(x) for x in rng.choice(pool, size=len(gold), replace=False)]
                e = eval_query(skb, T, {a: v for a, v in anchors.items() if a not in fake}, fake, at, names)
                nr += bool(e["relaxed"])
                ns += bool(e["strict"])
                nh += e["hop1"]
                na += e["any"]
        r["null_relaxed"] = nr / NULL_REPS
        r["null_strict"] = ns / NULL_REPS
        r["null_hop1"] = nh / NULL_REPS
        r["null_any"] = na / NULL_REPS
        # anchor-free: largest share of the gold set adjacent to one SKB node (multi-answer queries)
        r["gold_deg0"] = int(sum(len(skb.nbr(g)) == 0 for g in gold))
        if len(gold) >= 2:
            r["cover"] = best_cover(skb, gold)
            r["null_cover"] = [best_cover(skb, [int(x) for x in rng.choice(type_ids[at], size=len(gold), replace=False)])
                               for _ in range(NULL_REPS)]
        # textual
        anchor_toks = set()
        for a in anchors:
            anchor_toks |= sctoks(names.get(a, ""))
        resid = sctoks(q) - anchor_toks - tw
        r["n_resid"] = len(resid)
        pcts, mids, recs_ = [], [], []
        pr = text.pool_recalls(resid, at) if resid else None
        for g in gold:
            rec, pct = text.percentile(resid, g, at, pr)
            if rec is not None:
                pcts.append(pct[0])
                mids.append(pct[1])
                recs_.append(rec)
        r["gold_recall"] = [round(x, 4) for x in recs_]
        r["gold_pct"] = [round(x, 4) for x in pcts]
        r["gold_pct_mid"] = [round(x, 4) for x in mids]
        if seed_map is not None:
            s = seed_map[qid]
            r["seed"] = s
            r["seed_in_gold"] = s in gold
            rec, pct = text.percentile(resid, s, at, pr)
            r["seed_recall"] = rnd(rec)
            r["seed_pct"] = rnd(pct[0]) if pct else None
            r["seed_pct_mid"] = rnd(pct[1]) if pct else None
            others = [g for g in gold if g != s]
            orecs = [len(resid & text.toks(g)) / len(resid) for g in others] if resid else []
            r["other_recall_mean"] = rnd(np.mean(orecs)) if orecs else None
            r["other_recall_max"] = rnd(max(orecs)) if orecs else None
            r["seed_hop1"] = bool(anchors) and any(skb.adjacent(a, s) for a in anchors if a != s)
        recs.append(r)
    return recs, round(time.time() - t0, 1)


def summarize_subset(recs, has_seed=False):
    n = len(recs)
    wa = [r for r in recs if r["n_anchor"]]
    S = {"n": n, "with_anchor": frac(len(wa), n),
         "with_phrase_anchor": frac(sum(r["n_anchor_phrase"] > 0 for r in recs), n),
         "answer_types": dict(Counter(r["answer_type"] for r in recs).most_common()),
         "mixed_type_gold": sum(r["gold_types"] > 1 for r in recs)}
    for key, nk in (("relaxed", "null_relaxed"), ("strict", "null_strict"), ("hop1", "null_hop1"),
                    ("any", "null_any")):
        g_all = sum(bool(r[key]) for r in recs)
        g_wa = sum(bool(r[key]) for r in wa)
        S[key] = {"gold_all": frac(g_all, n), "gold_given_anchor": frac(g_wa, len(wa)),
                  "null_given_anchor": rnd(np.mean([r[nk] for r in wa])) if wa else None}
    by_n = {}
    for lab, f in (("1", lambda k: k == 1), ("2-5", lambda k: 2 <= k <= 5), (">5", lambda k: k > 5)):
        sub = [r for r in wa if f(r["n_gold"])]
        if sub:
            by_n[lab] = {"n": len(sub), "relaxed": rnd(np.mean([bool(r["relaxed"]) for r in sub])),
                         "null_relaxed": rnd(np.mean([r["null_relaxed"] for r in sub]))}
    S["relaxed_by_gold_size"] = by_n
    mc = [r for r in recs if "cover" in r]
    if mc:
        c = np.array([r["cover"] for r in mc])
        nc = np.array([x for r in mc for x in r["null_cover"]])
        S["anchor_free_cover"] = {
            "multi_answer_queries": len(mc),
            "all_gold_share_one_neighbour": frac(int((c >= 1 - 1e-9).sum()), len(mc)),
            "null_all": rnd((nc >= 1 - 1e-9).mean()),
            "cover_ge_0.75": rnd((c >= 0.75 - 1e-9).mean()), "null_ge_0.75": rnd((nc >= 0.75 - 1e-9).mean()),
            "cover_median": rnd(np.median(c)), "null_median": rnd(np.median(nc))}
    S["gold_with_no_skb_edge"] = frac(sum(r["gold_deg0"] for r in recs), sum(r["n_gold"] for r in recs))
    S["queries_with_an_edgeless_gold"] = frac(sum(r["gold_deg0"] > 0 for r in recs), n)
    def textual(rs):
        pct = np.array([p for r in rs for p in r["gold_pct"]])
        mid = np.array([p for r in rs for p in r["gold_pct_mid"]])
        rec = np.array([p for r in rs for p in r["gold_recall"]])
        if not len(pct):
            return {"gold_units": 0}
        return {"gold_units": int(len(pct)), "recall_mean": rnd(rec.mean()), "recall_zero": rnd((rec == 0).mean()),
                "pct_mid_median": rnd(np.median(mid)), "share_top1pct": rnd((pct <= 0.01).mean()),
                "share_top10pct": rnd((pct <= 0.10).mean())}

    by_t = {}
    for t in S["answer_types"]:
        sub = [r for r in wa if r["answer_type"] == t]
        if sub:
            by_t[t] = {"n_with_anchor": len(sub)}
            for key in ("relaxed", "strict", "hop1", "any"):
                by_t[t][key] = rnd(np.mean([bool(r[key]) for r in sub]))
                by_t[t]["null_" + key] = rnd(np.mean([r["null_" + key] for r in sub]))
        by_t.setdefault(t, {})["textual"] = textual([r for r in recs if r["answer_type"] == t])
    S["by_answer_type"] = by_t
    tmpl = Counter(t for r in recs for t in r["relaxed"])
    S["relaxed_template_hits"] = {str(k): v for k, v in sorted(tmpl.items())}
    tight = [r["tight"] for r in recs if r["tight"]]
    if tight:
        ret = np.array([t["retention"] for t in tight])
        S["retention_tightest_1hop_branch"] = {"n": len(tight), "median": rnd(np.median(ret)),
                                               "q25": rnd(np.quantile(ret, .25)), "q75": rnd(np.quantile(ret, .75)),
                                               "share_eq_1": rnd((ret >= 1 - 1e-9).mean())}
    S["textual"] = textual(recs)
    S["textual"]["queries_with_resid"] = frac(sum(r["n_resid"] > 0 for r in recs), n)
    if has_seed:
        sr = [r for r in recs if r.get("seed_recall") is not None]
        pr = [r for r in sr if r["other_recall_mean"] is not None]
        S["seed"] = {
            "seed_in_answers": frac(sum(r["seed_in_gold"] for r in recs), n),
            "seed_pct_mid_median": rnd(np.median([r["seed_pct_mid"] for r in sr])),
            "seed_top1pct": rnd(np.mean([r["seed_pct"] <= 0.01 for r in sr])),
            "seed_top10pct": rnd(np.mean([r["seed_pct"] <= 0.10 for r in sr])),
            "seed_recall_mean": rnd(np.mean([r["seed_recall"] for r in sr])),
            "multi_answer_queries": len(pr),
            "seed_minus_other_mean": rnd(np.mean([r["seed_recall"] - r["other_recall_mean"] for r in pr])),
            "seed_gt_other_mean": rnd(np.mean([r["seed_recall"] > r["other_recall_mean"] for r in pr])),
            "seed_ge_other_max": rnd(np.mean([r["seed_recall"] >= r["other_recall_max"] for r in pr])),
            "seed_hop1_given_anchor": rnd(np.mean([r["seed_hop1"] for r in recs if r["n_anchor"]])),
        }
    return S


# --- node text / names per SKB ---------------------------------------------------
def _flatten(v):
    if isinstance(v, dict):
        return " ".join(_flatten(x) for k, x in v.items() if k not in ("_id", "_score", "query", "genomic_pos"))
    if isinstance(v, (list, tuple)):
        return " ".join(_flatten(x) for x in v)
    if isinstance(v, str):
        return v
    return ""


def prime_setup():
    skb = SKB(PRIME_DIR, "prime")
    ni = pickle.load(open(PRIME_DIR / "node_info.pkl", "rb"))
    skb.md5["node_info.pkl"] = md5(PRIME_DIR / "node_info.pkl")
    names = {i: d.get("name") or "" for i, d in ni.items()}

    def text(i):
        d = ni[i]
        return (d.get("name") or "") + " " + _flatten(d.get("details", {}))
    return skb, names, text


def mag_setup():
    skb = SKB(MAG_DIR, "mag", MAG_REL)
    ni = pickle.load(open(MAG_DIR / "node_info.pkl", "rb"))
    skb.md5["node_info.pkl"] = md5(MAG_DIR / "node_info.pkl")
    names = {}
    for i, d in ni.items():
        t = d.get("type")
        if t == "paper":
            ti = str(d.get("title", ""))
            names[i] = ti if len(ti.split()) >= 4 and ti != "-1" else ""
        elif t == "author":
            dn = str(d.get("DisplayName", ""))
            names[i] = dn if len(dn.split()) >= 2 else ""
        else:
            names[i] = str(d.get("DisplayName", "") or "")

    def text(i):
        d = ni[i]
        return f"{d.get('title', '')} {d.get('abstract', '')}"
    return skb, names, text


def amazon_setup():
    skb = SKB(AMZ_DIR, "amazon")
    ni = pickle.load(open(AMZ_DIR / "node_info.pkl", "rb"))
    skb.md5["node_info.pkl"] = md5(AMZ_DIR / "node_info.pkl")
    names = {}
    for i, d in ni.items():
        if "brand_name" in d:
            names[i] = str(d["brand_name"])
        elif "category_name" in d:
            names[i] = str(d["category_name"])
        elif "color_name" in d:
            names[i] = str(d["color_name"])
        else:
            ti = str(d.get("title", ""))
            names[i] = ti if len(ti.split()) >= 4 else ""

    def text(i):
        d = ni[i]
        return " ".join(str(d.get(k, "")) for k in ("title", "feature", "description", "review", "qa"))
    return skb, names, text


# ---------------------------------------------------------------------------
ONLY = None


def build():
    rng = np.random.default_rng(SEED)
    S = {"meta": {"seed": SEED, "null_reps": NULL_REPS, "pool": POOL, "trace_dir": str(TRACE)}}
    S["quotes"], S["paper_files"] = paper_quotes()
    S["grade"] = grade_trace()
    hist, qa, seed_map = stark_history()
    S["stark_release"] = hist
    S["stark_openreview"] = openreview_record()
    per = []
    S["stark"] = {}
    for sub, setup, T, atypes, tok, cs, tw in (
            ("prime", prime_setup, PRIME_T,
             {"disease", "gene/protein", "molecular_function", "drug", "pathway", "anatomy", "effect/phenotype",
              "biological_process", "cellular_component", "exposure"}, True, ("gene/protein",), TEMPLATE_WORDS["prime"]),
            ("mag", mag_setup, MAG_T, {"author", "institution", "field_of_study", "paper"}, False, (),
             TEMPLATE_WORDS["mag"]),
            ("amazon", amazon_setup, AMZ_T, {"brand", "category", "color", "product"}, False, (),
             TEMPLATE_WORDS["amazon"])):
        if ONLY and sub not in ONLY:
            continue
        skb, names, text = setup()
        recs, secs = run_subset(sub, skb, T, qa[sub], names, text, atypes, tok, cs,
                                seed_map=seed_map if sub == "amazon" else None, tw=tw, rng=rng)
        S["stark"][sub] = summarize_subset(recs, has_seed=(sub == "amazon"))
        S["stark"][sub]["skb_md5"] = skb.md5
        S["stark"][sub]["runtime_s"] = {"load": skb.load_s, "tests": secs}
        per.extend(recs)
        del skb, names, text
    return S, per


def fmt(x):
    if isinstance(x, dict) and "frac" in x:
        return f"{x['frac']:.3f} ({x['num']:,}/{x['den']:,})" if x["frac"] is not None else "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return "n/a" if x is None else str(x)


def write_md(S):
    L = ["# GRADE and STaRK provenance trace", "",
         "Generated by `eval/_grade_stark_provenance_trace.py` (deterministic; no model calls; seeded nulls, "
         f"seed {S['meta']['seed']}, {S['meta']['null_reps']} draws per query; text pool {S['meta']['pool']} "
         "same-type nodes). Lexical helpers (`norm`, `ctoks`, `STOP`) are imported from `_grb_provenance_trace.py`.",
         "", "## Paper quotes located per version (1-indexed PDF pages; [] = absent)", "",
         "| id | quote | pages |", "|---|---|---|"]
    for r in S["quotes"]:
        pg = "; ".join(f"{k}: {v}" for k, v in r["pages"].items())
        L.append(f"| {r['id']} | {r['quote'][:150]}{'...' if len(r['quote']) > 150 else ''} | {pg} |")
    G = S["grade"]
    L += ["", "## GRADE: released construction code (no data released)", "",
          f"Repository {G['repo']} @ `{G['head'][:7]}`; construction code added in {G['construction_code_added']}; "
          f"{G['tracked_files']} tracked files, data files: {G['tracked_data_files'] or 'none'}. "
          "Copies of the traced files are in `sources/`.", "",
          "| fact | file:line | code | meaning |", "|---|---|---|---|"]
    for f in G["facts"]:
        code = f["code"].replace("|", "\\|")
        L.append(f"| {f['id']} | {f['file']}:{','.join(map(str, f['lines']))} | `{code[:110]}` | {f['meaning']} |")
    H = S["stark_release"]
    L += ["", "## STaRK: releases", "",
          f"GitHub snap-stanford/stark @ `{H['repo_head'][:7]}`: {H['files_ever_tracked']} files ever tracked; "
          f"paths matching generat/synth/templat/construct: {H['generation_like_paths'] or 'none'}. "
          f"`knowledge_base.py` typed metapath samplers `sample_paths` L{H['kb_sampler_lines']['sample_paths']}, "
          f"`get_all_paths` L{H['kb_sampler_lines']['get_all_paths']} (present since df59d09: "
          f"{H['kb_sampler_lines_first_commit_df59d09']}); the scripts that call them are not released. "
          f"The first loader (df59d09 hybrid_qa.py L{H['first_loader_meta_info']}) read per-query `meta_info` JSON; "
          f"no release contains it. HF QA files at any revision: {', '.join(H['hf_qa_files_any_revision'])}.", "",
          "| subset | columns | n | same query | same answer set | same id | first-release id in answers | current id in answers |",
          "|---|---|---|---|---|---|---|---|"]
    for sub, c in H["qa"].items():
        L.append(f"| {sub} | {c['columns']} | {c['n']:,} | {fmt(c['same_query'])} | {fmt(c['same_answer_set'])} | "
                 f"{fmt(c['same_id'])} | {fmt(c['first_release_id_in_answers'])} | {fmt(c['current_id_in_answers'])} |")
    a = H["qa"]["amazon"]
    L += ["", f"Amazon HF revisions: f721c636 id-in-answers {fmt(a['hf_f721c636_amazon.csv']['id_in_answers'])}; "
          f"01ee1e10 ('remapping') {fmt(a['hf_01ee1e10_amazon.csv']['id_in_answers'])}. Position of the first-release "
          f"id in the answer list: {H['amazon_seed_position_in_answer_list']}.", ""]
    O = S["stark_openreview"]
    L += ["OpenReview forum QSS5cGmKb1 (" + O["source"] + f"): {O['notes']} notes {O['kinds']}. Matches:", ""]
    for m in O["matches"]:
        L.append(f"- {m['note']} ({m['by']}, {m['field']}): {m['text']}")
    L += ["", "## STaRK: relational requirement (answer-entity axis)", "",
          "Anchors are recovered from the query text (template and anchor ids are not released). "
          "relaxed = one listed template branch from one anchor reaches every gold answer; strict = one full listed "
          "template (all conjuncts, negation) holds; 1-hop = one anchor adjacent to every gold answer. "
          "null = same test on random same-type sets of the same size, same anchors.", "",
          "| subset | n | anchor found | relaxed / anchor | null | strict / anchor | null | 1-hop / anchor | null | relaxed or 1-hop / anchor | null | relaxed or 1-hop (all queries) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sub, s in S["stark"].items():
        L.append(f"| {sub} | {s['n']:,} | {fmt(s['with_anchor'])} | "
                 f"{fmt(s['relaxed']['gold_given_anchor'])} | {fmt(s['relaxed']['null_given_anchor'])} | "
                 f"{fmt(s['strict']['gold_given_anchor'])} | {fmt(s['strict']['null_given_anchor'])} | "
                 f"{fmt(s['hop1']['gold_given_anchor'])} | {fmt(s['hop1']['null_given_anchor'])} | "
                 f"{fmt(s['any']['gold_given_anchor'])} | {fmt(s['any']['null_given_anchor'])} | "
                 f"{fmt(s['any']['gold_all'])} |")
    for sub, s in S["stark"].items():
        L += ["", f"### {sub}: by answer type (relational: queries with an anchor; textual: all gold units)", "",
              "| type | n | relaxed | null | strict | null | 1-hop | null | relaxed or 1-hop | null | text recall=0 | text pct (mid) | top 1% | top 10% |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, v in s["by_answer_type"].items():
            tx = v.get("textual", {})
            if "n_with_anchor" not in v:
                continue
            L.append(f"| {t} | {v['n_with_anchor']:,} | {fmt(v['relaxed'])} | {fmt(v['null_relaxed'])} | "
                     f"{fmt(v['strict'])} | {fmt(v['null_strict'])} | {fmt(v['hop1'])} | {fmt(v['null_hop1'])} | "
                     f"{fmt(v['any'])} | {fmt(v['null_any'])} | {fmt(tx.get('recall_zero'))} | "
                     f"{fmt(tx.get('pct_mid_median'))} | {fmt(tx.get('share_top1pct'))} | {fmt(tx.get('share_top10pct'))} |")
        L.append("")
        af = s.get("anchor_free_cover")
        if af:
            L.append(f"Anchor-free check (multi-answer queries, n={af['multi_answer_queries']:,}): all gold answers "
                     f"adjacent to one common SKB node {fmt(af['all_gold_share_one_neighbour'])} (null {fmt(af['null_all'])}); "
                     f">= 75% of them {fmt(af['cover_ge_0.75'])} (null {fmt(af['null_ge_0.75'])}); median best "
                     f"single-node coverage {fmt(af['cover_median'])} (null {fmt(af['null_median'])}). Gold answers with "
                     f"no SKB edge: {fmt(s['gold_with_no_skb_edge'])}; queries containing one: "
                     f"{fmt(s['queries_with_an_edgeless_gold'])}.")
            L.append("")
        L.append(f"By gold-set size: {json.dumps(s['relaxed_by_gold_size'])}. Template hits (relaxed): "
                 f"{s['relaxed_template_hits']}. Retention (gold / tightest covering 1-hop branch): "
                 f"{s.get('retention_tightest_1hop_branch')}.")
    L += ["", "## STaRK: textual requirement (textual axis)", "",
          "Residual query tokens = content tokens minus detected anchor-name tokens and template words; "
          "percentile = share of a seeded pool of same-type nodes whose text recalls at least as many residual "
          "tokens as the gold node (lower = the query's text matches the gold node's own text).", "",
          "| subset | gold units | recall mean | recall = 0 | percentile median (mid-rank) | top 1% (pessimistic) | top 10% (pessimistic) |",
          "|---|---|---|---|---|---|---|"]
    for sub, s in S["stark"].items():
        t = s["textual"]
        L.append(f"| {sub} | {t['gold_units']:,} | {fmt(t['recall_mean'])} | {fmt(t['recall_zero'])} | "
                 f"{fmt(t['pct_mid_median'])} | {fmt(t['share_top1pct'])} | {fmt(t['share_top10pct'])} |")
    sd = S["stark"].get("amazon", {}).get("seed")
    if sd:
        L += ["", "Amazon first-release id (seed) vs the other gold answers of the same query: "
              f"{json.dumps(sd)}", ""]
    L += verdict_lines(S)
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


RUBRIC = ("bin 4 (1.0): gold copied directly from the evaluation KG, or selected solely because it contains the answer "
          "string; bin 3 (0.75): auto-generated from KG triples or templates, or evidence retained by answer-string "
          "distant supervision; bin 2 (0.50): gold derived from K or A by auto-generation and then human-validated; "
          "bin 1 (0.25): human or text gold with KG-guided or answer-guided selection; bin 0: independent of K and A. "
          "Step 3: KG augmentation, human refinement, and anti-leakage filtering each shift the bin one step lower "
          "(at most twice).")


def verdict_lines(S):
    st = S.get("stark", {})
    g = {f["id"]: f for f in S["grade"]["facts"]}

    def ln(fid):
        f = g[fid]
        return f"{f['file']}:{','.join(map(str, f['lines']))}"
    L = ["", "## Rubric mapping (manuscript App. A.1)", "", "Rubric (verbatim): " + RUBRIC, ""]
    L += ["### GRADE edge axis (census bin 3)",
          f"- Order: articles -> fact sentences -> LLM claims -> LLM triples -> KG (DiGraph, {ln('graph_from_triples')}), "
          f"augmented by mirrored edges that copy the source row ({ln('mirror_copy_row')}, flag {ln('mirror_flag')}); "
          f"gold edge set P = a shortest path whose every triple is read from a graph edge ({ln('edge_from_graph')}); "
          f"2-5 hops, one path per endpoint pair, 400 per hop ({ln('hops_filter')}; {ln('dedup_endpoints')}; {ln('sample_400')}).",
          "- Base bin 4 (copied from the construction KG by code); Step 3 KG augmentation -1 -> bin 3. No human "
          f"refinement: the only post-generation filter is an o1-mini ambiguity check ({ln('validation_llm')}); the "
          "paper's human check audits 30 entity groups (App. A.6) and its Limitations list human-in-the-loop "
          "verification as future work. Census bin 3 stands.",
          "- Evidence: construction code (E by the manuscript's definition), but no generated data are released "
          "(repo has no data files; v1 promised 'code and data', v2/Findings say 'code'), so per-unit shares "
          "(e.g., gold paths using an augmented edge) cannot be computed.", ""]
    L += ["### GRADE passage axis (census bin 1)",
          f"- The supporting 'chunks' C_q are the source articles of the path's triples ({ln('chunks_by_triple')}, "
          f"{ln('chunk_body')}); a chunk.csv row is a whole 512-8192-token article ({ln('chunk_is_article')}); the "
          f"256-token retrieval chunks are cut later ({ln('db_after')}); C_q is used only for retrieval difficulty "
          f"({ln('dr_uses_chunks')}; power mean p=-3, {ln('dr_power_mean')}, vs min in Eq. 1); answers are LLM-judged "
          f"({ln('judge_only')}).",
          "- Source text selected entirely by the KG path: bin 1 (text gold with KG-guided selection) on the "
          "rubric's wording; no correction applies. Census bin 1 stands. Caveats: the selection is deterministic "
          "provenance lookup, the maximum degree of KG guidance (the census scores the analogous KG-to-text "
          "projection on 2WikiMultiHop's sentence axis at bin 3); the unit is an article, not a retrieval chunk; "
          "GRADE itself never scores either axis by overlap (Step 0 is met only for downstream use). Evidence: "
          "construction code (E), no data.", ""]
    if st:
        def g_(sub, k, f="gold_given_anchor"):
            return st[sub][k][f]["frac"] if isinstance(st[sub][k][f], dict) else st[sub][k][f]
        parts = []
        for sub in ("prime", "mag", "amazon"):
            if sub in st:
                af = st[sub].get("anchor_free_cover", {})
                parts.append(f"{sub}: listed-template-or-1-hop {g_(sub, 'any')} of anchored queries "
                             f"(null {g_(sub, 'any', 'null_given_anchor')}), strict template {g_(sub, 'strict')} "
                             f"(null {g_(sub, 'strict', 'null_given_anchor')}), anchors found "
                             f"{st[sub]['with_anchor']['frac']}; anchor-free: all gold of a multi-answer query "
                             f"adjacent to one node {af.get('all_gold_share_one_neighbour', {}).get('frac')} "
                             f"(null {af.get('null_all')}), median coverage {af.get('cover_median')} "
                             f"(null {af.get('null_median')}); edgeless gold {st[sub]['gold_with_no_skb_edge']['frac']}")
        L += ["### STaRK answer-entity axis (census bin 3)",
              "- Order (paper v1 pp.6-7, v2 p.6, v3/camera-ready p.5): sample an expert relation template, "
              "instantiate it with a sampled SKB entity, execute it over the SKB to get candidates, pick one "
              "candidate as the gold answer, extract textual properties from its document (LLM), fuse into a "
              "query (two LLMs), then keep the other candidates that three Claude models judge to meet the "
              "textual requirement. The seed gold answer is exempt from filtering (v1, v2) and is in the answer "
              "set of every Amazon query in the first release.",
              "- Released data: " + "; ".join(parts) + ". Retention of the tightest covering 1-hop branch is in "
              "the tables above.",
              "- Rubric: as described, every gold answer is an SKB node returned by executing a template over the "
              "same SKB the retrievers use (base bin 4, as for MetaQA and WebQSP), then filtered by LLMs; human "
              "answer verification covers only the 263 human-generated queries (v3 p.6) and the human study rates "
              "query naturalness, diversity, and practicality, so the census's stated correction (the 3-LLM "
              "filter read as human refinement) is not in Step 3. The released data do not support 'copied "
              "directly' either: on Amazon a majority of multi-answer gold sets contain an answer not adjacent to "
              "any node shared by the rest, and some gold answers have no SKB edge at all, so membership was "
              "partly decided outside the released SKB (the Amazon pipeline also LLM-filters by a generic "
              "category extracted from the seed's text, App. E). On MAG and Prime the relational requirement is "
              "met far above the null wherever the anchor can be recovered. Bin 3 (KG-template generation with "
              "an LLM-decided component) is the reading both the record and the data support; census bin stands. "
              "Tier T (per-query template and anchor are not released; checks use anchors recovered from the "
              "query and anchor-free coverage).",
              ""]
        L += ["### STaRK textual axis (census bin 1)",
              "- The textual requirement is extracted by an LLM from the seed gold answer's own document and "
              "paraphrased into the query (Prime prompt: 'uses different descriptions'); other gold documents "
              "enter through the LLM filter. The documents are source text (Amazon reviews/Q&A/descriptions, MAG "
              "abstracts, DrugBank/Mondo/NCBI text), not generated from the KG; which documents are gold is decided "
              "by the relational template. Bin 1 (text gold with KG-guided selection) stands; the LM raters' bin 3 "
              "would need text generated from K, which the sources contradict. Tier T.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="run twice and compare output hashes")
    ap.add_argument("--only", default="", help="comma list of STaRK subsets (debug; default all)")
    args = ap.parse_args()
    global ONLY
    ONLY = set(args.only.split(",")) if args.only else None
    if ONLY:
        global OUT, SRC_OUT, INP_OUT
        OUT = TRACE / "debug_out"
        SRC_OUT, INP_OUT = OUT / "sources", OUT / "inputs"
    OUT.mkdir(parents=True, exist_ok=True)
    S1, R1 = build()
    h1 = hashlib.sha256(json.dumps([{k: v for k, v in S1.items() if k != "stark"},
                                    {k: {kk: vv for kk, vv in v.items() if kk != "runtime_s"} for k, v in S1["stark"].items()},
                                    R1], sort_keys=True, default=str).encode()).hexdigest()
    if args.verify:
        S2, R2 = build()
        h2 = hashlib.sha256(json.dumps([{k: v for k, v in S2.items() if k != "stark"},
                                        {k: {kk: vv for kk, vv in v.items() if kk != "runtime_s"} for k, v in S2["stark"].items()},
                                        R2], sort_keys=True, default=str).encode()).hexdigest()
        assert h1 == h2, "non-deterministic trace"
        S1["meta"]["verified_twice"] = True
    S1["meta"]["output_sha256"] = h1
    with open(OUT / "per_query_stark.jsonl", "w", encoding="utf-8") as fh:
        for r in R1:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(S1, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    write_md(S1)
    print(json.dumps({k: S1["stark"][k]["relaxed"] for k in S1["stark"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
