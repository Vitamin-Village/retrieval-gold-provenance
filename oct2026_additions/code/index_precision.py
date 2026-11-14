# -*- coding: utf-8 -*-
"""Index precision at question-matched KG nodes vs. controlled-swap magnitude.

Observation: the census score C_KG does not predict the SIZE of the
controlled-swap effect (GraphRAG-Bench, C_KG 0.75, swaps more than MetaQA,
C_KG 1.0; STaRK-Prime, 0.75, shows almost none). Sec. 6 argues that the carrier
is a gold-precise entity-to-passage index. This script measures that index
system-side, for every KG condition and question, with each retriever's OWN
matcher, and relates it to the hit@5 gain over the OpenIE arm.

Per question q, KG K and matcher m:
  M_q  nodes the retriever's matcher returns for q
       m = "v3"  : ProposedV3Retriever._match_nodes  (proposed_v3 and
                   proposed_v5_graphcentric share it; both seed from it)
       m = "ppr" : PPRRetriever._seed_indices(_norm(q))
  L_q  union of those nodes' passage links restricted to the corpus, read with
       the accessor the retriever itself uses (_source_docs / node["source_docs"])
  G_q  the question's gold passages (eval_independent.gold_paper_ids)
  P_q = |L&G|/|L| (0 if L empty), R_q = |L&G|/|G|, F1_q = 2PR/(P+R) (0 if 0)
  1-hop variant: M_q plus every KG neighbour of M_q (same adjacency the
  retriever walks).
Question-agnostic: gold fraction of all KG-linked corpus passages (gold = union
of the split's gold).

Outcome per cell (bench, split, retriever, condition != OpenIE):
  gain = hit@5(condition KG) - hit@5(OpenIE KG), from the frozen per-question
  vectors _kgswap_*.json (no retrieval is re-run).

PRIMARY (declared before any index number was computed):
  predictor dF1 = mean_q F1_q(condition) - mean_q F1_q(OpenIE), same matcher/split;
  Spearman rho(dF1, gain) pooled over all cells of the three retrievers;
  95% CI from a two-stage cluster bootstrap (outer: the 4 benchmarks with
  replacement; inner: questions within each benchmark-split, jointly for all
  cells that share them), B=2000.

Read-only on all existing files. Writes:
  paper/analysis/index_precision_summary.json
  paper/analysis/index_precision_summary.md
  <cache-dir>/idx_<bench>_<split>_<kg>.json   (per-question index cache)

Run from anywhere:  python paper/analysis/index_precision.py [--workers 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("RAG_V3_DENSE_BACKEND", "deterministic")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
EXT = PAPER / "external"
RES = PAPER / "results"
sys.path.insert(0, str(ROOT))

DEFAULT_CACHE = PAPER / "analysis" / "_index_precision_cache"  # per-question index cache; --cache-dir overrides
SYSTEMS = ("proposed_v3", "proposed_v5_graphcentric", "ppr")
SYS_LABEL = {"proposed_v3": "Hybrid", "proposed_v5_graphcentric": "Hybrid-PPR", "ppr": "PPR"}
MATCHER = {"proposed_v3": "v3", "proposed_v5_graphcentric": "v3", "ppr": "ppr"}
SPLITS = ("sub200", "holdout200")
BENCHES = ("2wiki", "metaqa", "stark", "graphragbench")
C_KG = {"2wiki": 1.0, "metaqa": 1.0, "stark": 0.75, "graphragbench": 0.75}
# C_KG with GraphRAG-Bench at bin 2 (0.5) instead of bin 3 (0.75): a secondary sensitivity predictor
C_KG_GRB_BIN2 = {"2wiki": 1.0, "metaqa": 1.0, "stark": 0.75, "graphragbench": 0.5}
B_BOOT = 2000
SEED = 20260929


def conditions(bench):
    """(condition, kg dir rel. to split dir, vec prefix, vec stem)."""
    rows = [("openie", "kg_openie", "", "kg_openie"),
            ("aligned", "kg_silver", "", "kg_silver")]
    if bench == "2wiki":
        for g in ("g75", "g50", "g31"):
            rows.append(("dilute_" + g, "_concentration/silver_dilute_" + g,
                         "_concentration/", "silver_dilute_" + g))
        rows += [
            ("openie_goldonly", "_concentration/openie_goldonly", "_concentration/", "openie_goldonly"),
            ("zero_edge", "_zero_edge/kg_silver_noedges", "_zero_edge/", "silver_noedges"),
            ("wikidata_live", "kg_wikidata", "", "kg_wikidata"),
            ("T1_svocab", "kg_openie_svocab", "", "kg_openie_svocab"),
            ("T1b_svocab_noprov", "kg_openie_svocab_noprov", "", "kg_openie_svocab_noprov"),
            ("Tq25", "kg_openie_svocab_q25", "", "kg_openie_svocab_q25"),
            ("Tq50", "kg_openie_svocab_q50", "", "kg_openie_svocab_q50"),
            ("Tq75", "kg_openie_svocab_q75", "", "kg_openie_svocab_q75"),
            ("OC_resorted", "kg_openie_resorted", "", "kg_openie_resorted"),
        ]
    if bench == "metaqa":
        rows.append(("fullkb", "kg_fullkb", "", "kg_fullkb"))
    return rows


# ---------------------------------------------------------------------------
# Index computation (reuses the retrievers' own matching code)
# ---------------------------------------------------------------------------
class _NoBM25:
    """BM25 stub: only _match_nodes / _adjacency / paper_by_id are used."""

    def query(self, query, top_k=5):
        return []


def _index_job(args):
    bench, split, kgrel, cache_path = args
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.load(open(cache_path, encoding="utf-8"))
    from paper.eval import eval_independent as ev
    from paper.eval.ppr_retriever import PPRRetriever, _norm
    from prototype.baselines.proposed_v3_retriever import ProposedV3Retriever, _source_docs

    d = EXT / bench / split
    qa = ev.load_jsonl(d / "qa.jsonl")
    passages = ev.load_jsonl(d / "passages.jsonl")
    manifest = ev._load_manifest_records(d / "manifest.json")
    by_paper, _by_id = ev.build_passage_indexes(passages)
    papers = ev.build_candidate_papers(manifest, by_paper)
    kg = ev._load_kg(d / kgrel)
    if not kg.nodes:
        raise RuntimeError("empty KG: %s" % (d / kgrel))
    v3 = ProposedV3Retriever(papers=papers, bm25=_NoBM25(), dense=None,
                             nodes=kg.nodes, edges=kg.edges)
    ppr = PPRRetriever(papers, kg.nodes, kg.edges)

    def v3_links(node):
        return {p for p in _source_docs(node) if p in v3.paper_by_id}

    def ppr_links(i):
        return {p for p in ppr._nodes[i].get("source_docs", []) if p and p in ppr._paper_ids}

    split_gold = set()
    for q in qa:
        split_gold.update(ev.gold_paper_ids(q))
    linked_all = set()
    for node in kg.nodes:
        linked_all |= v3_links(node)
    agn = {"n_nodes": len(kg.nodes), "n_edges": len(kg.edges), "n_corpus": len(v3.paper_by_id),
           "n_gold": len(split_gold), "linked": len(linked_all),
           "linked_gold": len(linked_all & split_gold)}

    per_q = {}
    for q in qa:
        qid = ev._qa_id(q)
        gold = set(ev.gold_paper_ids(q))
        question = str(q.get("question") or q.get("query") or "")
        matched = v3._match_nodes(question)
        L = set()
        for m in matched:
            L |= v3_links(m.node)
        nb = set()
        for m in matched:
            for nid, _edge in v3._adjacency.get(m.node_id, []):
                nb.add(nid)
        L1 = set(L)
        for nid in nb:
            node = v3.nodes_by_id.get(nid)
            if node is not None:
                L1 |= v3_links(node)
        seeds = ppr._seed_indices(_norm(question))
        Lp = set()
        for i in seeds:
            Lp |= ppr_links(i)
        Lp1 = set(Lp)
        for i in seeds:
            for j in ppr._adj[i]:
                Lp1 |= ppr_links(j)
        per_q[qid] = {
            "g": len(gold),
            # [n_matched_nodes, n_linked, n_linked_gold]
            "v3": [len(matched), len(L), len(L & gold)],
            "v3_1hop": [len(matched) + len(nb), len(L1), len(L1 & gold)],
            "ppr": [len(seeds), len(Lp), len(Lp & gold)],
            "ppr_1hop": [len(seeds), len(Lp1), len(Lp1 & gold)],
        }
    out = {"bench": bench, "split": split, "kg": kgrel, "agnostic": agn,
           "qids": [ev._qa_id(q) for q in qa], "per_q": per_q}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache_path, "w", encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# Frozen per-question hit vectors
# ---------------------------------------------------------------------------
def vec_path(split_dir, system, prefix, stem):
    p = split_dir / ("%s_kgswap_%s_%s.json" % (prefix, system, stem))
    if p.exists():
        return p
    if system == "proposed_v3":
        p2 = split_dir / ("%s_kgswap_%s.json" % (prefix, stem))
        if p2.exists():
            return p2
    return None


def load_hits(path, system):
    d = json.load(open(path, encoding="utf-8"))
    res = d.get("results", {}).get(system)
    if not res or not res.get("records"):
        return None, None
    hits = {str(r["qa_id"]): float(r["metrics"]["hit_at_k"]) for r in res["records"]}
    bm25 = d["results"].get("bm25", {}).get("records")
    bm = {str(r["qa_id"]): float(r["metrics"]["hit_at_k"]) for r in bm25} if bm25 else None
    return hits, bm


# ---------------------------------------------------------------------------
# Per-question index arrays
# ---------------------------------------------------------------------------
def prf_arrays(idx, qids, key):
    g = np.array([idx["per_q"][q]["g"] for q in qids], dtype=float)
    nl = np.array([idx["per_q"][q][key][1] for q in qids], dtype=float)
    nh = np.array([idx["per_q"][q][key][2] for q in qids], dtype=float)
    P = np.divide(nh, nl, out=np.zeros_like(nh), where=nl > 0)
    R = np.divide(nh, g, out=np.zeros_like(nh), where=g > 0)
    F = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) > 0)
    return {"P": P, "R": R, "F1": F, "PR": P * R, "nl": nl, "ne": (nl > 0).astype(float)}


def _mean(a, w):
    """Plain mean (w None) or bootstrap-weighted mean (w = resample counts / n)."""
    return float(a @ w) if w is not None else float(a.mean())


class Cell:
    def __init__(self, bench, split, system, cond, hc, ho, ic, io, ic1, io1, agn_c, agn_o):
        self.bench, self.split, self.system, self.cond = bench, split, system, cond
        self.hc, self.ho = hc, ho          # per-question hit arrays (cond, openie)
        self.ic, self.io = ic, io          # index arrays (direct)
        self.ic1, self.io1 = ic1, io1      # index arrays (1-hop)
        self.agn_c, self.agn_o = agn_c, agn_o

    def outcome(self, key, ix=None):
        hc, ho = _mean(self.hc, ix), _mean(self.ho, ix)
        if key == "gain":
            return hc - ho
        if key == "gain_norm":
            return (hc - ho) / (1.0 - ho) if ho < 1.0 else float("nan")
        raise KeyError(key)

    def predictor(self, key, ix=None):
        if key in ("dF1", "dP", "dR", "dPR"):
            k = key[1:]
            return _mean(self.ic[k], ix) - _mean(self.io[k], ix)
        if key == "F1":
            return _mean(self.ic["F1"], ix)
        if key == "dF1_1hop":
            return _mean(self.ic1["F1"], ix) - _mean(self.io1["F1"], ix)
        if key == "dAgnGoldFrac":
            return self.agn_c - self.agn_o
        if key == "C_KG":
            return C_KG[self.bench]
        if key == "C_KG_grb_bin2":
            return C_KG_GRB_BIN2[self.bench]
        raise KeyError(key)


def _rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).correlation)


def spearman_boot(cells, pkey, okey, rng, cluster=True, resample_q=True, B=B_BOOT):
    point = _rho([c.predictor(pkey) for c in cells], [c.outcome(okey) for c in cells])
    by_b = {}
    for c in cells:
        by_b.setdefault(c.bench, []).append(c)
    benches = sorted(by_b)
    nq = {}
    for c in cells:
        nq[(c.bench, c.split)] = len(c.hc)
    rhos = []
    for _ in range(B):
        draw = list(rng.choice(benches, size=len(benches), replace=True)) if cluster else benches
        xs, ys = [], []
        for bn in draw:
            ixs = {}
            for c in by_b[bn]:
                key = (c.bench, c.split)
                if key not in ixs:
                    n = nq[key]
                    ixs[key] = (np.bincount(rng.integers(0, n, n), minlength=n) / n
                                if resample_q else None)
                ix = ixs[key]
                xs.append(c.predictor(pkey, ix))
                ys.append(c.outcome(okey, ix))
        rhos.append(_rho(xs, ys))
    r = np.array(rhos)
    ok = r[~np.isnan(r)]
    ci = [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else [None, None]
    return {"rho": None if np.isnan(point) else round(point, 3), "n_cells": len(cells),
            "n_bench": len(benches),
            "ci95": [None if v is None else round(v, 3) for v in ci],
            "frac_undefined_draws": round(1 - len(ok) / len(r), 3)}


# ---------------------------------------------------------------------------
# Cross-check against the shipped results JSONs
# ---------------------------------------------------------------------------
def crosscheck(cellmap):
    """Compare our per-question-vector gains to the numbers in results/*.json."""
    name2 = {"2Wiki": "2wiki", "MetaQA": "metaqa", "STaRK": "stark", "GraphRAG-Bench": "graphragbench"}
    checks = []

    def add(bench, split, system, cond, ref):
        c = cellmap.get((bench, split, system, cond))
        if c is None:
            checks.append((bench, split, system, cond, ref, None))
            return
        checks.append((bench, split, system, cond, ref, round(c.outcome("gain"), 4)))

    for f in sorted(RES.glob("kg_swap_*.json")):
        if "dense" in f.name or "variant" in f.name:
            continue
        for r in json.load(open(f, encoding="utf-8")):
            bname, split = r["name"].rsplit("-", 1)
            system = r.get("system", "proposed_v3")
            add(name2[bname], split, system, "aligned", r["silver"]["graph"] - r["openie"]["graph"])
    for r in json.load(open(RES / "kg_concentration_2wiki.json", encoding="utf-8"))["rows"]:
        cond = r["variant"].replace("silver_dilute_", "dilute_")
        add("2wiki", r["split"], r["system"], cond, r["diff_vs_openie"])
    for r in json.load(open(RES / "kg_zero_edge_2wiki.json", encoding="utf-8"))["rows"]:
        add("2wiki", r["split"], r["system"], "zero_edge", r["diff_vs_openie"])
    for r in json.load(open(RES / "kg_fullkb_metaqa.json", encoding="utf-8")):
        add("metaqa", r["split"], r["system"], "fullkb", r["diff_vs_openie"])
    for r in json.load(open(RES / "kg_swap_variant_2wiki_kg_wikidata.json", encoding="utf-8")):
        add("2wiki", r["split"], r["system"], "wikidata_live", r["diff_vs_openie"])
    cmap = {"T1_openie_svocab": "T1_svocab", "T1b_openie_svocab_noprov": "T1b_svocab_noprov",
            "OC_openie_resorted": "OC_resorted", "Tq_kg_openie_svocab_q25": "Tq25",
            "Tq_kg_openie_svocab_q50": "Tq50", "Tq_kg_openie_svocab_q75": "Tq75"}
    for f in sorted(RES.glob("kg_transplant*_2wiki*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            split = r["name"].rsplit("-", 1)[1]
            add("2wiki", split, r["system"], cmap[r["condition"]], r["delta_graph_vs_openie"])
    bad = [c for c in checks if c[5] is None or abs(c[5] - c[4]) > 1e-6]
    return {"n_checked": len(checks), "n_mismatch": len(bad),
            "mismatches": [list(b) for b in bad]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--boot", type=int, default=B_BOOT)
    args = ap.parse_args(argv)

    # 1. inventory + index jobs
    found, missing, jobs = [], [], []
    for b in BENCHES:
        for sp in SPLITS:
            sd = EXT / b / sp
            for cond, kgrel, prefix, stem in conditions(b):
                if not (sd / kgrel / "proposed_kg_nodes.json").exists():
                    missing.append("%s/%s/%s" % (b, sp, kgrel))
                    continue
                found.append((b, sp, cond, kgrel, prefix, stem))
                cp = args.cache_dir / ("idx_%s_%s_%s.json" % (b, sp, kgrel.replace("/", "__")))
                jobs.append((b, sp, kgrel, str(cp)))
    print("KG conditions found: %d, missing: %s" % (len(found), missing), flush=True)
    with Pool(processes=max(1, args.workers)) as pool:
        idx_list = pool.map(_index_job, jobs, chunksize=1)
    IDX = {(o["bench"], o["split"], o["kg"]): o for o in idx_list}

    # 2. index arrays per (bench, split, cond, matcher)
    qorder, arrays, agn = {}, {}, {}
    for b, sp, cond, kgrel, prefix, stem in found:
        o = IDX[(b, sp, kgrel)]
        qorder.setdefault((b, sp), o["qids"])
        assert o["qids"] == qorder[(b, sp)], "question order differs across KGs"
        assert len(set(o["qids"])) == len(o["qids"]), "duplicate qa_id"
        for m in ("v3", "ppr"):
            arrays[(b, sp, cond, m)] = prf_arrays(o, o["qids"], m)
            arrays[(b, sp, cond, m + "_1hop")] = prf_arrays(o, o["qids"], m + "_1hop")
        a = o["agnostic"]
        agn[(b, sp, cond)] = a["linked_gold"] / max(1, a["linked"])

    # 3. hits + cells
    hits, vec_used, bm25, bm25_identical = {}, {}, {}, True
    for b, sp, cond, kgrel, prefix, stem in found:
        for s in SYSTEMS:
            p = vec_path(EXT / b / sp, s, prefix, stem)
            if p is None:
                continue
            h, bm = load_hits(p, s)
            if h is None:
                continue
            qs = qorder[(b, sp)]
            assert set(h) == set(qs), "vector/question mismatch %s" % p
            hits[(b, sp, s, cond)] = np.array([h[q] for q in qs])
            if bm is not None:
                bv = np.array([bm[q] for q in qs])
                if (b, sp) in bm25 and not np.array_equal(bm25[(b, sp)], bv):
                    bm25_identical = False
                bm25.setdefault((b, sp), bv)
            vec_used[(b, sp, s, cond)] = str(p.relative_to(PAPER)).replace("\\", "/")
    cells, cellmap = [], {}
    for (b, sp, s, cond), hc in hits.items():
        if cond == "openie" or (b, sp, s, "openie") not in hits:
            continue
        m = MATCHER[s]
        c = Cell(b, sp, s, cond, hc, hits[(b, sp, s, "openie")],
                 arrays[(b, sp, cond, m)], arrays[(b, sp, "openie", m)],
                 arrays[(b, sp, cond, m + "_1hop")], arrays[(b, sp, "openie", m + "_1hop")],
                 agn[(b, sp, cond)], agn[(b, sp, "openie")])
        cells.append(c)
        cellmap[(b, sp, s, cond)] = c
    order = {b: i for i, b in enumerate(BENCHES)}
    cells.sort(key=lambda c: (order[c.bench], c.cond != "aligned", c.cond, c.split, SYSTEMS.index(c.system)))
    xcheck = crosscheck(cellmap)
    print("cells: %d; cross-check vs results/*.json: %d checked, %d mismatches"
          % (len(cells), xcheck["n_checked"], xcheck["n_mismatch"]), flush=True)

    # 4. statistics
    rng = np.random.default_rng(SEED)
    PRIMARY = ("dF1", "gain")
    out_stats = {}
    out_stats["primary_pooled"] = spearman_boot(cells, *PRIMARY, rng, True, True, args.boot)
    out_stats["primary_pooled_question_only_ci"] = spearman_boot(cells, *PRIMARY, rng, False, True, args.boot)
    preds = ("dF1", "dP", "dR", "dPR", "F1", "dF1_1hop", "dAgnGoldFrac", "C_KG", "C_KG_grb_bin2")
    subsets = {"pooled": cells}
    for s in SYSTEMS:
        subsets[s] = [c for c in cells if c.system == s]
    table = {}
    for okey in ("gain", "gain_norm"):
        for pkey in preds:
            for sname, sc in subsets.items():
                table["%s|%s|%s" % (okey, pkey, sname)] = spearman_boot(sc, pkey, okey, rng, True, True, args.boot)
    out_stats["predictor_table"] = table
    # leave-one-benchmark-out (point + cluster CI over the remaining 3)
    lobo = {}
    for b in BENCHES:
        rest = [c for c in cells if c.bench != b]
        lobo["drop_" + b] = {s: spearman_boot([c for c in rest if s == "pooled" or c.system == s],
                                              *PRIMARY, rng, True, True, args.boot)
                             for s in ("pooled",) + SYSTEMS}
    out_stats["leave_one_benchmark_out"] = lobo
    # within-benchmark (question bootstrap only)
    within = {}
    for b in BENCHES:
        bc = [c for c in cells if c.bench == b]
        within[b] = {s: spearman_boot([c for c in bc if s == "pooled" or c.system == s],
                                      *PRIMARY, rng, False, True, args.boot)
                     for s in ("pooled",) + SYSTEMS}
    out_stats["within_benchmark"] = within
    # cross-benchmark: aligned-vs-OpenIE cells only (the comparison across benchmarks)
    al = [c for c in cells if c.cond == "aligned"]
    out_stats["aligned_only"] = {
        "%s|%s|%s" % (ok, pk, s): spearman_boot([c for c in al if s == "pooled" or c.system == s],
                                                 pk, ok, rng, True, True, args.boot)
        for ok in ("gain", "gain_norm")
        for pk in ("dF1", "dP", "dR", "dAgnGoldFrac", "C_KG", "C_KG_grb_bin2")
        for s in ("pooled",) + SYSTEMS}
    # cross-benchmark ordering on aligned cells (split-averaged)
    ordering = {}
    for s in SYSTEMS:
        for b in BENCHES:
            cs = [c for c in al if c.system == s and c.bench == b]
            if cs:
                ordering["%s|%s" % (s, b)] = {
                    "gain": round(float(np.mean([c.outcome("gain") for c in cs])), 4),
                    "gain_norm": round(float(np.mean([c.outcome("gain_norm") for c in cs])), 4),
                    "dF1": round(float(np.mean([c.predictor("dF1") for c in cs])), 4),
                    "F1_aligned": round(float(np.mean([c.predictor("F1") for c in cs])), 4),
                    "dP": round(float(np.mean([c.predictor("dP") for c in cs])), 4),
                    "dR": round(float(np.mean([c.predictor("dR") for c in cs])), 4),
                    "C_KG": C_KG[b]}
    out_stats["aligned_ordering"] = ordering

    # 5. per-condition index table
    idx_table = []
    for b, sp, cond, kgrel, prefix, stem in found:
        a = IDX[(b, sp, kgrel)]["agnostic"]
        row = {"bench": b, "split": sp, "cond": cond, "kg_dir": kgrel,
               "n_nodes": a["n_nodes"], "n_edges": a["n_edges"],
               "agn_linked": a["linked"], "agn_gold_frac": round(a["linked_gold"] / max(1, a["linked"]), 3),
               "agn_gold_covered": round(a["linked_gold"] / max(1, a["n_gold"]), 3)}
        for m in ("v3", "ppr", "v3_1hop", "ppr_1hop"):
            A = arrays[(b, sp, cond, m)]
            ne = A["ne"] > 0
            row[m] = {"cov": round(float(A["ne"].mean()), 3),
                      "P_nonempty": round(float(A["P"][ne].mean()), 3) if ne.any() else None,
                      "P_micro": round(float((A["P"] * A["nl"]).sum() / max(1.0, A["nl"].sum())), 3),
                      "R": round(float(A["R"].mean()), 3), "F1": round(float(A["F1"].mean()), 3),
                      "n_linked_mean": round(float(A["nl"].mean()), 1),
                      "n_linked_median": float(np.median(A["nl"]))}
        idx_table.append(row)

    cell_rows = []
    for c in cells:
        cell_rows.append({
            "bench": c.bench, "split": c.split, "system": c.system, "cond": c.cond,
            "hit5_cond": round(float(c.hc.mean()), 4), "hit5_openie": round(float(c.ho.mean()), 4),
            "hit5_bm25": round(float(bm25[(c.bench, c.split)].mean()), 4),
            "gain": round(c.outcome("gain"), 4), "gain_norm": round(c.outcome("gain_norm"), 4),
            "F1_cond": round(c.predictor("F1"), 4), "dF1": round(c.predictor("dF1"), 4),
            "dP": round(c.predictor("dP"), 4), "dR": round(c.predictor("dR"), 4),
            "dPR": round(c.predictor("dPR"), 4), "dF1_1hop": round(c.predictor("dF1_1hop"), 4),
            "dAgnGoldFrac": round(c.predictor("dAgnGoldFrac"), 4),
            "C_KG": C_KG[c.bench], "vec": vec_used[(c.bench, c.split, c.system, c.cond)]})

    summary = {
        "preregistration": PREREG,
        "kg_conditions_found": ["%s/%s/%s (%s)" % (b, sp, kgrel, cond) for b, sp, cond, kgrel, _p, _s in found],
        "kg_conditions_missing": missing,
        "crosscheck_vs_results_json": xcheck,
        "bm25_vectors_identical_across_conditions": bm25_identical,
        "n_cells": len(cells),
        "stats": out_stats,
        "index_table": idx_table,
        "cells": cell_rows,
    }
    outj = PAPER / "analysis" / "index_precision_summary.json"
    json.dump(summary, open(outj, "w", encoding="utf-8"), indent=1)
    write_md(summary, PAPER / "analysis" / "index_precision_summary.md")
    print("wrote", outj, flush=True)
    return 0


PREREG = (
    "Declared before any index number or fit was computed (analysis/index_precision_PREREG.md, 2026-09-29). "
    "Cell = (benchmark, split, retriever, KG condition != OpenIE). Outcome = hit@5(condition) - "
    "hit@5(OpenIE), same retriever/split/questions, from frozen per-question vectors. "
    "PRIMARY predictor dF1 = mean_q F1_q(condition) - mean_q F1_q(OpenIE), where F1_q is the "
    "harmonic mean of index precision P_q=|L_q&G_q|/|L_q| (0 if L_q empty) and recall "
    "R_q=|L_q&G_q|/|G_q| over L_q = corpus passages linked from the nodes the retriever's own "
    "matcher returns for q. PRIMARY test = Spearman rho(dF1, gain) pooled over all cells of the three "
    "retrievers, 95% CI by two-stage cluster bootstrap (benchmarks, then questions within "
    "benchmark-split), B=2000. Everything else is secondary."
)


INTERPRETATION = """## Interpretation (written after viewing the results above; numbers refer to this run)

1. Primary. rho(dF1, gain) = +0.51 over 98 cells, but the two-stage cluster CI [-0.27, +0.82]
   spans zero; the question-only CI [+0.43, +0.55] is tight. The pooled relation is carried by
   2WikiMultiHop's 74 within-benchmark cells: dropping 2Wiki leaves rho = +0.07; within 2Wiki
   rho = +0.62 (Hybrid +0.82, Hybrid-PPR +0.71, PPR +0.45). Four benchmarks cannot establish a
   cross-benchmark relation at conventional confidence for any predictor.
2. C_KG. Constant within a benchmark, so it cannot rank the 2Wiki conditions whose gains run from
   -0.415 to +0.355 at C_KG = 1.0 (pooled rho -0.09). On the 24 aligned-vs-OpenIE cells it does not
   beat index precision on raw gain (C_KG +0.29, with GraphRAG-Bench at bin 2 +0.07; dF1 +0.50; question-agnostic gold
   fraction +0.61), and all CIs overlap. After headroom normalisation C_KG (+0.61) and dF1 (+0.59;
   1.00/0.98 for the two hybrids, +0.32 for PPR) order the benchmarks about equally well.
3. STaRK-Prime null: explained. At the nodes a question names, the aligned STaRK KG has F1
   0.06-0.08 (hybrid matcher) and 0.03-0.06 (PPR matcher, which reaches any passage for only
   28-29% of questions); dF1 = +0.00 to +0.06, the smallest of any benchmark. The aligned KG maps each
   entity to its own passage, questions name non-answer entities, and gold is the answer entity,
   so the index points away from the gold; its question-agnostic gold fraction equals OpenIE's
   (0.18-0.20).
4. GraphRAG-Bench > MetaQA: NOT explained by index precision. MetaQA has the largest dF1 for the
   hybrid matcher (+0.61 vs +0.09), and for the PPR matcher its OpenIE index is already precise
   (F1 0.88-0.92; dF1 +0.10 vs +0.18). The ordering comes from (a) headroom: MetaQA's OpenIE arm sits
   at 0.86-0.935 hit@5 (BM25 0.855-0.87) versus 0.20-0.52 on GraphRAG-Bench; as a share of headroom
   MetaQA's swap exceeds GraphRAG-Bench's for all three retrievers (0.36/0.62/1.00 vs 0.13/0.48/0.56);
   and (b) OpenIE-arm harm on GraphRAG-Bench: PPR and Hybrid-PPR with the OpenIE KG fall below BM25
   (0.20-0.245 and 0.40-0.405 vs 0.505-0.52), where the question-matched OpenIE index spans 360-690
   passages at F1 0.01-0.02. Against BM25 the aligned arms gain similar amounts on the two
   benchmarks for PPR (+0.12 to +0.17 vs +0.13 to +0.145). The aligned GraphRAG-Bench index is
   gold-precise only question-agnostically (every linked passage is gold for some question), not
   at the question's nodes (P 0.06-0.13 over 16-56 linked passages).
5. Where direct-link precision fails. (i) Diluting the aligned 2Wiki KG to g31 lowers the hybrid
   matcher's dF1 from +0.39 to +0.13 (first split) but leaves the hybrids' gains nearly intact
   (+0.160 to +0.130; +0.355 to +0.330); only PPR tracks it (+0.315 to -0.030). (ii) Deleting all
   edges leaves the index unchanged (same dF1) yet PPR falls to -0.40/-0.415 and Hybrid-PPR to 0:
   for bridge questions the second gold passage is reached through an edge, so the direct index is
   necessary but not sufficient. (iii) The two hybrids share a matcher and hence dF1, yet
   Hybrid-PPR's aligned gains are 2-5x the Hybrid's on 2Wiki, MetaQA and GraphRAG-Bench: graph
   reliance scales the effect.
6. Wording this supports: "C_a > 0 is necessary; within a benchmark, the size of the inflation
   tracks how precisely the structure maps the question's entities to the gold (rho 0.62-0.82 for
   the question-anchored retrievers on 2WikiMultiHop), scaled by the retriever's reliance on the
   graph and bounded by lexical headroom." Across benchmarks the data are consistent with index
   precision (it explains the STaRK-Prime null and separates it from GraphRAG-Bench at equal C_KG),
   but they do not show that it predicts magnitude better than C_KG, and the GraphRAG-Bench > MetaQA
   ordering in raw hit@5 comes from headroom and OpenIE-arm harm, not from index precision.
"""


def _f(v, nd=3, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return (("{:+.%df}" if sign else "{:.%df}") % nd).format(v)


def _ci(r):
    return "%s [%s, %s]" % (_f(r["rho"], 2, True), _f(r["ci95"][0], 2, True), _f(r["ci95"][1], 2, True))


def write_md(S, path):
    st = S["stats"]
    L = ["# Index precision at question-matched nodes vs. swap magnitude",
         "",
         "Generated by `paper/analysis/index_precision.py` (read-only on all existing results; "
         "no retrieval re-run). Machine-readable: `index_precision_summary.json`.",
         "",
         "## Pre-declared primary measure",
         "",
         S["preregistration"],
         "",
         "## Inputs",
         "",
         "KG conditions found (%d): " % len(S["kg_conditions_found"]) + "; ".join(S["kg_conditions_found"]) + ".",
         "Missing: %s." % (", ".join(S["kg_conditions_missing"]) or "none"),
         "Cross-check of gains recomputed from per-question vectors against `results/*.json`: "
         "%d values checked, %d mismatches." % (S["crosscheck_vs_results_json"]["n_checked"],
                                                 S["crosscheck_vs_results_json"]["n_mismatch"]),
         "",
         INTERPRETATION,
         "## Primary result",
         "",
         "Pooled Spearman rho(dF1, gain), %d cells: %s (two-stage cluster bootstrap over 4 benchmarks); "
         "question-only bootstrap CI [%s, %s]; undefined cluster draws %.1f%%." % (
             st["primary_pooled"]["n_cells"], _ci(st["primary_pooled"]),
             _f(st["primary_pooled_question_only_ci"]["ci95"][0], 2, True),
             _f(st["primary_pooled_question_only_ci"]["ci95"][1], 2, True),
             100 * st["primary_pooled"]["frac_undefined_draws"]),
         "",
         "## Predictors compared (Spearman rho, cluster-bootstrap 95% CI)",
         "",
         "Each entry draws its own bootstrap stream, so the gain|dF1|pooled CI differs from the "
         "primary CI above by Monte Carlo error only. With 4 clusters a percentile CI can exclude a "
         "tied point estimate (e.g. rho=1.00 over 8 cells).",
         "",
         "| outcome | predictor | pooled | Hybrid | Hybrid-PPR | PPR |",
         "|---|---|---|---|---|---|"]
    T = st["predictor_table"]
    for ok in ("gain", "gain_norm"):
        for pk in ("dF1", "dP", "dR", "dPR", "F1", "dF1_1hop", "dAgnGoldFrac", "C_KG", "C_KG_grb_bin2"):
            L.append("| %s | %s | %s |" % (ok, pk, " | ".join(
                _ci(T["%s|%s|%s" % (ok, pk, s)]) for s in ("pooled",) + SYSTEMS)))
    L += ["", "## Leave one benchmark out (primary)", "",
          "| dropped | pooled | Hybrid | Hybrid-PPR | PPR |", "|---|---|---|---|---|"]
    for k, v in st["leave_one_benchmark_out"].items():
        L.append("| %s | %s |" % (k[5:], " | ".join(_ci(v[s]) for s in ("pooled",) + SYSTEMS)))
    L += ["", "## Within benchmark (primary; question-bootstrap CI)", "",
          "| benchmark | pooled | Hybrid | Hybrid-PPR | PPR |", "|---|---|---|---|---|"]
    for k, v in st["within_benchmark"].items():
        L.append("| %s (n=%d) | %s |" % (k, v["pooled"]["n_cells"], " | ".join(_ci(v[s]) for s in ("pooled",) + SYSTEMS)))
    L += ["", "## Aligned-vs-OpenIE cells only (cross-benchmark, 24 cells)", "",
          "| outcome | predictor | pooled | Hybrid | Hybrid-PPR | PPR |", "|---|---|---|---|---|---|"]
    A = st["aligned_only"]
    for ok in ("gain", "gain_norm"):
        for pk in ("dF1", "dP", "dR", "dAgnGoldFrac", "C_KG", "C_KG_grb_bin2"):
            L.append("| %s | %s | %s |" % (ok, pk, " | ".join(_ci(A["%s|%s|%s" % (ok, pk, s)]) for s in ("pooled",) + SYSTEMS)))
    L += ["", "Aligned-arm ordering (split means):", "",
          "| retriever | benchmark | C_KG | gain | gain_norm | F1 aligned | dF1 | dP | dR |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in st["aligned_ordering"].items():
        s, b = k.split("|")
        L.append("| %s | %s | %.2f | %s | %s | %s | %s | %s | %s |" % (
            SYS_LABEL[s], b, v["C_KG"], _f(v["gain"], 3, True), _f(v["gain_norm"], 3, True),
            _f(v["F1_aligned"]), _f(v["dF1"], 3, True), _f(v["dP"], 3, True), _f(v["dR"], 3, True)))
    L += ["", "## Index at question-matched nodes, per KG condition", "",
          "cov = share of questions whose matched nodes link any corpus passage; P = mean precision over "
          "those questions; R = mean recall over all questions; |L| = mean linked passages; agn = "
          "question-agnostic gold fraction of all KG-linked passages.", "",
          "| bench | split | condition | v3 cov | v3 P | v3 R | v3 F1 | v3 abs L | ppr cov | ppr P | ppr R | ppr F1 | ppr abs L | agn |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in S["index_table"]:
        v, p = r["v3"], r["ppr"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %.1f | %s | %s | %s | %s | %.1f | %s |" % (
            r["bench"], r["split"], r["cond"], _f(v["cov"], 2), _f(v["P_nonempty"], 2), _f(v["R"], 2),
            _f(v["F1"], 2), v["n_linked_mean"], _f(p["cov"], 2), _f(p["P_nonempty"], 2), _f(p["R"], 2),
            _f(p["F1"], 2), p["n_linked_mean"], _f(r["agn_gold_frac"], 2)))
    L += ["", "## Plot-ready cell table", "",
          "| bench | split | retriever | condition | hit@5 cond | hit@5 OpenIE | hit@5 BM25 | gain | F1 cond | dF1 | dP | dR | dF1 1-hop | C_KG |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in S["cells"]:
        L.append("| %s | %s | %s | %s | %.3f | %.3f | %.3f | %s | %.3f | %s | %s | %s | %s | %.2f |" % (
            c["bench"], c["split"], SYS_LABEL[c["system"]], c["cond"], c["hit5_cond"], c["hit5_openie"],
            c["hit5_bm25"],
            _f(c["gain"], 3, True), c["F1_cond"], _f(c["dF1"], 3, True), _f(c["dP"], 3, True),
            _f(c["dR"], 3, True), _f(c["dF1_1hop"], 3, True), c["C_KG"]))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
