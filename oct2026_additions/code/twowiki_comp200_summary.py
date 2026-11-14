"""Summary of the KG swap on the 2WikiMultiHop compositional split (comp200).

Protocol: results/twowiki_comp200/PROTOCOL.md (fixed before any run on the split). Reads only frozen
per-question vectors and reader outputs; runs no retrieval or reader.

  (a) retrieval: hit@5 of Hybrid, PPR, Hybrid-PPR (and HippoRAG 2 when scored) under the aligned KG,
      the independent OpenIE KG and BM25; paired swap aligned minus independent (10,000-resample
      bootstrap CI re-seeded with each retriever's original driver seed, exact two-sided sign test);
      graph-over-BM25 margins and verdicts per KG arm with analysis/headline_margins.py's `paired`
      and `verdict` (uncorrected p<0.05; Holm over the comp200 cells of an arm as sensitivity);
  (b) end-to-end EM per reader and retriever (results/twowiki_comp200/e2e_qa/<reader>/), paired
      answer swap with the CI and p of eval/e2e_qa_swap.py, graph-over-BM25 EM verdicts;
  (c) HippoRAG 2 (results/twowiki_comp200/hipporag2/summary.json) when present;
  plus sanity checks and the existing splits' numbers for context (read from their frozen files).

Outputs: analysis/twowiki_comp200_summary.{json,md}
Usage (from paper/, env new_env): python analysis/twowiki_comp200_summary.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # paper/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT.parent))
import headline_margins as HM  # noqa: E402  (paired / verdict / sign_test of the paper's verdict analysis)
from kg_passage_link_counts import counts as link_counts  # noqa: E402
from paper.eval._kg_swap_2wiki_r2 import boot_ci  # noqa: E402  (python-random percentile bootstrap, B=10,000)

EXT = ROOT / "external" / "2wiki"
RES = ROOT / "results"
CDIR = RES / "twowiki_comp200"
SPLIT = "comp200"
OUT_JSON, OUT_MD = ROOT / "analysis" / "twowiki_comp200_summary.json", ROOT / "analysis" / "twowiki_comp200_summary.md"
ARMS = {"aligned": "silver", "independent": "openie"}
# key -> (label, vector file pattern, results key, e2e system alias, bootstrap seed of the original driver)
RETRIEVERS = {
    "hybrid": ("Hybrid", "_kgswap_kg_{kg}.json", "proposed_v3", "audited", 20260608),
    "ppr": ("PPR", "_kgswap_ppr_kg_{kg}.json", "ppr", "ppr", 20260611),
    "hybrid_ppr": ("Hybrid-PPR", "_kgswap_proposed_v5_graphcentric_kg_{kg}.json", "proposed_v5_graphcentric", "graph_centric", 20260611),
    "hipporag2": ("HippoRAG 2", "_kgswap_hipporag2_kg_{kg}.json", "hipporag2", "hipporag2", 20260918),
}
READERS = {"llama3_1_8b": "Llama-3.1-8B", "qwen2_5_14b": "Qwen2.5-14B"}
SPLITS_OLD = {"sub200": "first", "holdout200": "held-out"}

# Filled in by hand: every difference from PROTOCOL.md or from how the existing splits were run.
DEVIATIONS: list[str] = [
    "No departure from PROTOCOL.md in selection, builders, KG settings, retriever configurations, metrics, statistics or readers. "
    "The points below are differences from how the existing splits were run, all non-substantive or pre-declared.",
    "Drivers: the existing swap drivers hard-code sub200/holdout200 and overwrite results/kg_swap_2wiki*.json, so their run_kg "
    "functions were called from a new driver (eval/_kg_swap_2wiki_comp200.py); HippoRAG 2 and the readers ran through thin wrappers "
    "(split choice and output directory only).",
    "Hybrid swap CI: eval/_swap_stats.py seeds random once at import and draws sub200 then holdout200 from one stream; for comp200 the "
    "same seed (20260608) is set immediately before resampling, as eval/_kg_swap_2wiki_r2.py does (pre-declared).",
    "OpenIE build: same command and settings. Ollama is 0.34.3; the server version used for the May 2026 builds is not recorded, the "
    "model digest (46e0c10c039e) is the same. 16 of 1,155 passages (1.4%) returned no parsable JSON after the retry (0.4% and 0.8% on "
    "sub200/holdout200); as before they contribute no nodes.",
    "Corpus size 1,155 passages (existing splits 1,229-1,256): the rule takes each question's ten context paragraphs and deduplicates by title; "
    "compositional questions share more paragraphs. 48 / 49 comp200 passages also occur in the sub200 / holdout200 pools and 6 / 3 comp200 "
    "gold passages are also gold there (the existing pipeline enforces QA-id disjointness only, as does this split).",
    "HippoRAG 2: identical BaseConfig (no key differs from the configuration of the sub200 runs, results/hipporag2_swap/METHODS.md), "
    "same clone/commit and package; BGE-M3 on CPU with the default thread count. One OpenIE entity becomes an empty string after "
    "HippoRAG's text processing (empty_entity_strings = 1; 0 in the runs of the other splits); the graph-equals-injection checks pass.",
    "Readers: run first on the three retrievers, then re-run after HippoRAG 2 was scored to add its two conditions, as PROTOCOL.md "
    "Section 6 provides; the 1,709 earlier generations were reused from the cache, 396 new ones were generated per reader.",
]


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def vec(path: Path, key: str) -> dict[str, float]:
    d = json.load(open(path, encoding="utf-8"))
    return {r["qa_id"]: float(r["metrics"]["hit_at_k"]) for r in d["results"][key]["records"]}


def swap_stats(a: dict, b: dict, seed: int, sort_ids: bool) -> dict:
    """Aligned minus independent, the convention of the driver that produced the existing statistic:
    question order = record order (sorted ids for HippoRAG 2), random.seed(seed) right before resampling."""
    qs = sorted(q for q in a if q in b) if sort_ids else [q for q in a if q in b]
    diffs = [a[q] - b[q] for q in qs]
    random.seed(seed)
    lo, hi = boot_ci(diffs)
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"n": len(qs), "mean_paired_diff": sum(diffs) / len(diffs), "ci95": [lo, hi], "discordant_pos": pos,
            "discordant_neg": neg, "sign_test_p": HM.sign_test(pos, neg), "bootstrap_seed": seed}


# ----------------------------------------------------------------------------- sanity
def sanity() -> dict:
    d = EXT / SPLIT
    qa = jl(d / "qa.jsonl")
    pool = {json.loads(l)["passage_id"] for l in open(d / "passages.jsonl", encoding="utf-8")}
    gold = {g for q in qa for g in q["gold_paper_ids"]}
    out = {"n_questions": len(qa), "qtype_counts": dict(collections.Counter(q["qtype"] for q in qa)),
           "gold_passages_per_question": dict(collections.Counter(len(q["gold_paper_ids"]) for q in qa)),
           "n_passages": len(pool), "n_unique_gold_passages": len(gold),
           "manifest": json.load(open(d / "comp200_manifest.json", encoding="utf-8")), "overlap": {}}
    for sp in SPLITS_OLD:
        o_qa = jl(EXT / sp / "qa.jsonl")
        o_pool = {json.loads(l)["passage_id"] for l in open(EXT / sp / "passages.jsonl", encoding="utf-8")}
        o_gold = {g for q in o_qa for g in q["gold_paper_ids"]}
        out["overlap"][sp] = {"qa_ids": len({q["qa_id"] for q in qa} & {q["qa_id"] for q in o_qa}),
                              "passages_in_both_pools": len(pool & o_pool), "n_passages_other": len(o_pool),
                              "comp200_gold_in_other_pool": len(gold & o_pool), "comp200_gold_that_is_other_gold": len(gold & o_gold)}
    sil_n = json.load(open(d / "kg_silver" / "proposed_kg_nodes.json", encoding="utf-8"))
    sil_e = json.load(open(d / "kg_silver" / "proposed_kg_edges.json", encoding="utf-8"))
    out["kg_silver"] = {"nodes": len(sil_n), "nodes_with_source_docs": sum(1 for n in sil_n if n["source_docs"]), "edges": len(sil_e)}
    if (d / "kg_openie" / "build_summary.json").exists():
        out["kg_openie"] = json.load(open(d / "kg_openie" / "build_summary.json", encoding="utf-8"))
        old = {sp: json.load(open(EXT / sp / "kg_openie" / "build_summary.json", encoding="utf-8")) for sp in SPLITS_OLD}
        out["kg_openie_existing_splits"] = {sp: {k: v[k] for k in ("n_nodes", "n_edges", "n_synonym_edges", "n_relation_edges",
                                                                   "n_passages", "parse_fail_rate", "wall_clock_seconds")} for sp, v in old.items()}
        out["link_counts"] = link_counts(d)
        stored = json.load(open(RES / "kg_passage_link_counts.json", encoding="utf-8"))
        out["link_counts_existing_splits"] = [r for r in stored if r["bench"] == "2wiki"]
    # BM25 vector identical across every comp200 vector file
    files = sorted(d.glob("_kgswap_*.json"))
    bm = {}
    for f in files:
        r = json.load(open(f, encoding="utf-8"))["results"]
        if "bm25" in r:
            bm[f.name] = {x["qa_id"]: (x["metrics"]["hit_at_k"], x["retrieved_paper_ids"]) for x in r["bm25"]["records"]}
    ref = next(iter(bm.values()))
    out["bm25_identical_across_files"] = {"n_files": len(bm), "identical": all(v == ref for v in bm.values())}
    out["bm25_hit5"] = {SPLIT: sum(v[0] for v in ref.values()) / len(ref)}
    for sp in SPLITS_OLD:
        o = vec(EXT / sp / "_kgswap_kg_silver.json", "bm25")
        out["bm25_hit5"][sp] = sum(o.values()) / len(o)
    return out


# ----------------------------------------------------------------------------- retrieval
def retrieval() -> list[dict]:
    d = EXT / SPLIT
    bm25 = vec(d / "_kgswap_kg_silver.json", "bm25")
    qids = list(bm25)
    y = np.array([bm25[q] for q in qids])
    recs = []
    for r, (label, pat, key, _, seed) in RETRIEVERS.items():
        files = {arm: d / pat.format(kg=kg) for arm, kg in ARMS.items()}
        if not all(f.exists() for f in files.values()):
            continue
        h = {arm: vec(f, key) for arm, f in files.items()}
        for arm in ARMS:
            assert set(h[arm]) == set(qids), (r, arm)
        rec = {"retriever": r, "label": label, "n": len(qids), "bm25_hit5": float(y.mean())}
        for arm in ARMS:
            x = np.array([h[arm][q] for q in qids])
            rec[arm] = {"hit5": float(x.mean()), **HM.paired(x, y)}
            rec[arm]["verdict"] = HM.verdict(rec[arm]["margin"], rec[arm]["sign_test_p"])
        rec["swap"] = swap_stats(h["aligned"], h["independent"], seed, sort_ids=(r == "hipporag2"))
        rec["swap_delta_as_in_kg_swap_2wiki_json"] = (rec["aligned"]["hit5"] - rec["bm25_hit5"]) - (rec["independent"]["hit5"] - rec["bm25_hit5"])
        recs.append(rec)
    HM.add_holm(recs, tuple(ARMS))
    for rec in recs:
        rec["flip"] = rec["aligned"]["verdict"] == "beats" and rec["independent"]["verdict"] != "beats"
    return recs


def context_existing() -> list[dict]:
    """The existing 2WikiMultiHop splits' numbers, read from analysis/headline_margins_summary.json (frozen)."""
    hm = json.load(open(ROOT / "analysis" / "headline_margins_summary.json", encoding="utf-8"))
    out = []
    for c in hm["retrieval"]["cells"]:
        if c["bench"] == "2wiki":
            out.append({"split": c["split"], "retriever": c["retriever"], "bm25": c["bm25_hit5"],
                        "aligned": c["aligned"]["hit5"], "independent": c["independent"]["hit5"],
                        "verdict_aligned": c["aligned"]["verdict"], "verdict_independent": c["independent"]["verdict"],
                        "swap": c["swap"]["margin"], "swap_p": c["swap"]["sign_test_p"]})
    ans = {}
    for tag in READERS:
        ans[tag] = [{"split": c["split"], "retriever": c["retriever"], "bm25": c["bm25_acc"], "aligned": c["aligned"]["acc"],
                     "independent": c["independent"]["acc"], "verdict_aligned": c["aligned"]["verdict"],
                     "verdict_independent": c["independent"]["verdict"], "swap": c["swap"]["margin"], "swap_p": c["swap"]["sign_test_p"]}
                    for c in hm["answer"][tag]["cells"] if c["bench"] == "2wiki"]
    return {"retrieval": out, "answer": ans, "source": "analysis/headline_margins_summary.json", "generated": hm["generated"]}


# ----------------------------------------------------------------------------- answers
def answers(ret_recs: list[dict]) -> dict:
    out = {}
    ret = {r["retriever"]: r for r in ret_recs}
    for tag, lab in READERS.items():
        pq, sm = CDIR / "e2e_qa" / tag / "per_question.jsonl", CDIR / "e2e_qa" / tag / "summary.json"
        if not pq.exists():
            continue
        rows = jl(pq)
        summ = json.load(open(sm, encoding="utf-8"))
        sc = collections.defaultdict(dict)
        for r in rows:
            sc[(r["system"], r["condition"])][r["qa_id"]] = float(r["em"])
        if ("bm25", "none") not in sc:  # reader not run yet (or still generating)
            continue
        base = sc[("bm25", "none")]
        qids = list(base)
        y = np.array([base[q] for q in qids])
        cells = {(c["system"], c["condition"]): c for c in summ["cells"]}
        pair = {p["system"]: p for p in summ["paired"]}
        recs, checks = [], []
        for r, (label, _, _, alias, _) in RETRIEVERS.items():
            if (alias, "kg_silver") not in sc or (alias, "kg_openie") not in sc:
                continue
            rec = {"retriever": r, "label": label, "n": len(qids), "bm25_em": float(y.mean())}
            v = {}
            for arm, kg in ARMS.items():
                s = sc[(alias, f"kg_{kg}")]
                assert set(s) == set(qids)
                v[arm] = np.array([s[q] for q in qids])
                rec[arm] = {"em": float(v[arm].mean()), **HM.paired(v[arm], y)}
                rec[arm]["verdict"] = HM.verdict(rec[arm]["margin"], rec[arm]["sign_test_p"])
                checks.append(("EM equals e2e summary", cells[(alias, f"kg_{kg}")]["em"], rec[arm]["em"]))
                if r in ret:
                    checks.append(("e2e hit5 equals retrieval vector", cells[(alias, f"kg_{kg}")]["hit5"], ret[r][arm]["hit5"]))
            p = pair[alias]["em"]
            rec["swap"] = {"mean_paired_diff": p["mean_paired_diff"], "ci95": p["ci95"], "discordant_pos": p["discordant_pos"],
                           "discordant_neg": p["discordant_neg"], "sign_test_p": p["sign_test_p"],
                           "f1_diff": pair[alias]["f1"]["mean_paired_diff"], "mediation": pair[alias]["mediation"],
                           "ci_method": "eval/e2e_qa_swap.py: numpy default_rng(20260918), B=10,000, percentile"}
            chk = HM.paired(v["aligned"], v["independent"])
            checks.append(("swap mean recomputed", p["mean_paired_diff"], chk["margin"]))
            checks.append(("swap sign-test p recomputed", p["sign_test_p"], chk["sign_test_p"]))
            recs.append(rec)
        checks.append(("BM25 EM equals e2e summary", cells[("bm25", "none")]["em"], float(y.mean())))
        HM.add_holm(recs, tuple(ARMS))
        bad = [c for c in checks if abs(c[1] - c[2]) > 1e-9]
        assert not bad, bad
        out[tag] = {"reader": lab, "model": summ["model"], "n_rows": summ["n_rows"], "cells": recs,
                    "closedbook_em": cells[("none", "closedbook")]["em"], "oracle_em": cells[("none", "oracle")]["em"],
                    "bm25_em": cells[("bm25", "none")]["em"], "n_checks_ok": len(checks)}
    return out


# ----------------------------------------------------------------------------- markdown
def f3(x):
    return f"{x:.3f}"


def s3(x):
    return f"{x:+.3f}"


def pf(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def markdown(S: dict) -> str:
    san, ret = S["sanity"], S["retrieval"]
    L = ["# KG swap on a compositional 2WikiMultiHop split (comp200)", "",
         f"Generated {S['generated']} by `analysis/twowiki_comp200_summary.py` from frozen vectors and reader outputs. "
         "Protocol (fixed before any run on the split): `results/twowiki_comp200/PROTOCOL.md`. "
         "Every result below is reported whatever its value.", "",
         "Split: 200 `compositional` questions from 2WikiMultiHop dev (`select_subsample`, seed 20261002), QA-id-disjoint "
         "from the inference-type `sub200`/`holdout200`; corpus, gold, aligned KG (`kg_silver`, Wikidata evidence triples) "
         "and independent KG (`kg_openie`, Llama-3.1-8B OpenIE, no gold access) built by the unchanged builders; "
         "retrievers frozen, deterministic dense surrogate. hit@5 = both gold passages in the top five.", ""]
    if S["deviations"]:
        L += ["## Deviations", ""] + [f"- {x}" for x in S["deviations"]] + [""]
    # (a)
    L += ["## (a) Retrieval", "",
          "| retriever | hit@5 aligned | hit@5 indep. | hit@5 BM25 | swap aligned − indep. [95% CI] | disc. +/− | sign p | margin aligned (p, verdict) | margin indep. (p, verdict) |",
          "|---|---:|---:|---:|---|---:|---:|---|---|"]
    for r in ret["cells"]:
        a, i, s = r["aligned"], r["independent"], r["swap"]
        L.append(f"| {r['label']} | {f3(a['hit5'])} | {f3(i['hit5'])} | {f3(r['bm25_hit5'])} | {s3(s['mean_paired_diff'])} {ci(s['ci95'])} | "
                 f"+{s['discordant_pos']}/−{s['discordant_neg']} | {pf(s['sign_test_p'])} | {s3(a['margin'])} ({pf(a['sign_test_p'])}, {a['verdict']}) | "
                 f"{s3(i['margin'])} ({pf(i['sign_test_p'])}, {i['verdict']}) |")
    nb = {arm: sum(r[arm]["verdict"] == "beats" for r in ret["cells"]) for arm in ARMS}
    nbh = {arm: sum(r[arm]["verdict_holm"] == "beats" for r in ret["cells"]) for arm in ARMS}
    L += ["", f"Graph beats BM25 (sign test p<0.05, uncorrected): aligned {nb['aligned']}/{len(ret['cells'])}, independent "
          f"{nb['independent']}/{len(ret['cells'])}; flips beats→not: {sum(r['flip'] for r in ret['cells'])}. "
          f"Holm over the {len(ret['cells'])} comp200 cells of an arm: aligned {nbh['aligned']}, independent {nbh['independent']}.",
          "CI: 10,000-resample percentile bootstrap, Python `random` re-seeded right before resampling with the original driver's seed "
          "(Hybrid 20260608, PPR and Hybrid-PPR 20260611, HippoRAG 2 20260918 over sorted ids). Swap = Delta(aligned) − Delta(independent) = mean paired difference.", ""]
    L += ["### Context: the existing inference-type splits (frozen, `analysis/headline_margins_summary.json`)", "",
          "| split | retriever | aligned | indep. | BM25 | swap | swap p | verdict aligned / indep. |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for c in S["context_existing"]["retrieval"]:
        L.append(f"| {SPLITS_OLD[c['split']]} | {RETRIEVERS[c['retriever']][0]} | {f3(c['aligned'])} | {f3(c['independent'])} | {f3(c['bm25'])} | "
                 f"{s3(c['swap'])} | {pf(c['swap_p'])} | {c['verdict_aligned']} / {c['verdict_independent']} |")
    # (b)
    L += ["", "## (b) End-to-end answers (alias-aware EM; `eval/e2e_qa_swap.py` unchanged)", ""]
    if not S["answer"]:
        L += ["Not run.", ""]
    for tag, a in S["answer"].items():
        L += [f"### {a['reader']} (`{a['model']}`)", "",
              f"closed-book EM {f3(a['closedbook_em'])}, gold-only (oracle) EM {f3(a['oracle_em'])}, BM25 EM {f3(a['bm25_em'])}.", "",
              "| retriever | EM aligned | EM indep. | EM BM25 | answer swap [95% CI] | disc. +/− | sign p | margin aligned (p, verdict) | margin indep. (p, verdict) |",
              "|---|---:|---:|---:|---|---:|---:|---|---|"]
        for r in a["cells"]:
            x, y, s = r["aligned"], r["independent"], r["swap"]
            L.append(f"| {r['label']} | {f3(x['em'])} | {f3(y['em'])} | {f3(r['bm25_em'])} | {s3(s['mean_paired_diff'])} {ci(s['ci95'])} | "
                     f"+{s['discordant_pos']}/−{s['discordant_neg']} | {pf(s['sign_test_p'])} | {s3(x['margin'])} ({pf(x['sign_test_p'])}, {x['verdict']}) | "
                     f"{s3(y['margin'])} ({pf(y['sign_test_p'])}, {y['verdict']}) |")
        L += ["", "Mediation (from `e2e_qa_swap.py`): " + "; ".join(
            f"{r['label']}: EM diff {s3(r['swap']['mediation']['em_diff_on_retrieval_flips_pos']) if r['swap']['mediation']['em_diff_on_retrieval_flips_pos'] is not None else '-'} "
            f"on {r['swap']['mediation']['n_flips_pos']} retrieval flips to hit, "
            f"{s3(r['swap']['mediation']['em_diff_on_no_flip']) if r['swap']['mediation']['em_diff_on_no_flip'] is not None else '-'} on {r['swap']['mediation']['n_no_flip']} non-flips"
            for r in a["cells"]) + ".", ""]
    if S["answer"]:
        L += ["### Context: existing splits, answer swap (frozen)", "", "| reader | split | retriever | EM aligned | EM indep. | EM BM25 | swap | p |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for tag, rows in S["context_existing"]["answer"].items():
            for c in rows:
                L.append(f"| {READERS[tag]} | {SPLITS_OLD[c['split']]} | {RETRIEVERS[c['retriever']][0]} | {f3(c['aligned'])} | {f3(c['independent'])} | "
                         f"{f3(c['bm25'])} | {s3(c['swap'])} | {pf(c['swap_p'])} |")
        L.append("")
    # (c)
    L += ["## (c) HippoRAG 2", ""]
    h = S.get("hipporag2")
    if h is None:
        L += ["Not run.", ""]
    else:
        p = h["paired_aligned_minus_independent"]
        L += [f"hit@5 aligned {f3(h['hit5_aligned'])}, independent {f3(h['hit5_independent'])}, BM25 {f3(h['hit5_bm25'])}, "
              f"HippoRAG's own dense-only channel (BGE-M3) {f3(h['hit5_dense_only_bge_m3'])}; swap {s3(p['mean_paired_diff'])} {ci(p['ci95'])}, "
              f"+{p['discordant_pos']}/−{p['discordant_neg']}, p = {pf(p['sign_test_p'])}. Retrieval path (ppr / dense fallback): "
              + ", ".join(f"{k} {v.get('ppr', 0)}/{v.get('dpr_fallback', 0)}" for k, v in h["retrieval_path_counts"].items())
              + f". Dense-only ranking identical across conditions: {h['dpr_only_identical_across_conditions']}. Details: "
              "`results/twowiki_comp200/hipporag2/summary.md`.", ""]
    # sanity
    L += ["## Sanity checks", "",
          f"- Question types: {san['qtype_counts']}; gold passages per question: {san['gold_passages_per_question']}; "
          f"{san['n_unique_gold_passages']} unique gold passages.",
          f"- Corpus: {san['n_passages']} passages (sub200 1,256; holdout200 1,229). QA-id overlap with sub200 / holdout200: "
          f"{san['overlap']['sub200']['qa_ids']} / {san['overlap']['holdout200']['qa_ids']}. Passages also in the sub200 / holdout200 pool: "
          f"{san['overlap']['sub200']['passages_in_both_pools']} / {san['overlap']['holdout200']['passages_in_both_pools']}; comp200 gold passages "
          f"that are also gold there: {san['overlap']['sub200']['comp200_gold_that_is_other_gold']} / {san['overlap']['holdout200']['comp200_gold_that_is_other_gold']}.",
          f"- Aligned KG: {san['kg_silver']['nodes']} nodes ({san['kg_silver']['nodes_with_source_docs']} with source_docs), {san['kg_silver']['edges']} edges "
          "(sub200 589/391, holdout200 593/392)."]
    if "kg_openie" in san:
        o = san["kg_openie"]
        L.append(f"- Independent KG: {o['n_nodes']} nodes, {o['n_edges']} edges ({o['n_relation_edges']} extracted relations, {o['n_synonym_edges']} surface-synonym), "
                 f"parse failures {o['n_parse_fail']}/{o['n_passages']} ({o['parse_fail_rate']}), {o['wall_clock_seconds']:.0f} s "
                 "(sub200 9,949/9,497, holdout200 9,629/9,586).")
        L += ["", "| split | passages | gold | aligned: linked | linked gold | gold frac. of linked | gold covered | indep.: linked | linked gold | gold frac. of linked | gold covered |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in [san["link_counts"]] + san["link_counts_existing_splits"]:
            s_, o_ = r["kg_silver"], r["kg_openie"]
            L.append(f"| {r['split']} | {r['n_passages']} | {r['n_gold']} | {s_['linked']} | {s_['linked_gold']} | {s_['gold_frac_of_linked']} | {s_['gold_covered']} | "
                     f"{o_['linked']} | {o_['linked_gold']} | {o_['gold_frac_of_linked']} | {o_['gold_covered']} |")
        L.append("")
    b = san["bm25_hit5"]
    L.append(f"- BM25 hit@5: comp200 {f3(b[SPLIT])} (sub200 {f3(b['sub200'])}, holdout200 {f3(b['holdout200'])}); BM25 vector identical across all "
             f"{san['bm25_identical_across_files']['n_files']} comp200 vector files: {san['bm25_identical_across_files']['identical']}.")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ret = retrieval()
    S = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "script": "analysis/twowiki_comp200_summary.py",
         "protocol": "results/twowiki_comp200/PROTOCOL.md", "deviations": DEVIATIONS, "sanity": sanity(),
         "retrieval": {"cells": ret}, "answer": answers(ret), "context_existing": context_existing()}
    hp = CDIR / "hipporag2" / "summary.json"
    S["hipporag2"] = json.load(open(hp, encoding="utf-8")) if hp.exists() else None
    OUT_JSON.write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(markdown(S), encoding="utf-8")
    print(markdown(S))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
