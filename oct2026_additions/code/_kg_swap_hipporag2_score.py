# -*- coding: utf-8 -*-
"""
Score the HippoRAG 2 KG-swap runs (eval/_kg_swap_hipporag2.py) with the paper's shared
per-question scorer and write
  external/2wiki/<split>/_kgswap_hipporag2_<kg>.json   (same layout as _kgswap_ppr_*.json;
                                                        system name "hipporag2", top-5 ids)
  results/hipporag2_swap/summary.json / summary.md     (paired swap statistics + graph stats)

Paired statistics follow the hit@5 swap convention of eval/_kg_swap_graphragbench.py:
per-question hit@5 difference aligned - independent over sorted qa ids, 10,000-resample
bootstrap 95% CI (random.seed(20260918) right before resampling), exact two-sided sign
test over discordant questions. BM25 reference hit@5 is read from the frozen
_kgswap_ppr_kg_silver.json vectors (the BM25 run shared by every swap condition).

Run from paper/'s parent with the project python (base env):
    python paper/eval/_kg_swap_hipporag2_score.py            # full runs
    python paper/eval/_kg_swap_hipporag2_score.py --tag _lim10 --no-vectors   # quick test on --limit 10 runs
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402
from paper.eval._kg_swap_metaqa import boot_ci, sign_test  # noqa: E402

PAPER = ROOT / "paper"
RUNS = PAPER / "results" / "hipporag2_swap" / "runs"
OUTD = PAPER / "results" / "hipporag2_swap"
SPLITS = (("sub200", "first split"), ("holdout200", "held-out split"))
KGS = (("kg_silver", "aligned"), ("kg_openie", "independent"))
BOOT_SEED = 20260918
TOP_K = 5
METRIC_DEFS = {
    "hit_at_k": "single gold => contained in top-K; multi-hop => BOTH gold contained",
    "mrr": "single => 1/rank of gold; multi-hop => mean of the two golds' reciprocal ranks",
    "answer_span_recall_at_k": "fraction of evidence snippets found in top-K passage text (text-based, KG-independent)",
    "bridge_coverage_at_k": "multi-hop only: 1 iff both gold papers in top-K",
}


def run_dir(split, kg, bench="2wiki"):
    return RUNS / (f"{split}__{kg}" if bench == "2wiki" else f"{bench}__{split}__{kg}")


def load_run(split, kg, tag, bench="2wiki"):
    d = run_dir(split, kg, bench)
    rows = [json.loads(l) for l in open(d / f"retrieval{tag}.jsonl", encoding="utf-8")]
    meta = json.load(open(d / f"run_meta{tag}.json", encoding="utf-8"))
    return rows, meta


def vectors(split, rows, key="ranked_passage_ids", bench="2wiki"):
    d = PAPER / "external" / bench / split
    qa = ev.load_jsonl(d / "qa.jsonl")
    by_paper, by_id = ev.build_passage_indexes(ev.load_jsonl(d / "passages.jsonl"))
    top = {r["qa_id"]: r[key][:TOP_K] for r in rows}
    qa = [q for q in qa if q["qa_id"] in top]
    recs = [ev.score_record(q, top[q["qa_id"]], passages_by_paper=by_paper, passages_by_id=by_id) for q in qa]
    summary = ev.summarize_system(recs)
    summary["system"] = "hipporag2"
    payload = {"benchmark": str(d / "qa.jsonl"), "n_qa": len(qa), "top_k": TOP_K, "systems": ["hipporag2"],
               "metric_defs": METRIC_DEFS, "results": {"hipporag2": {"summary": summary, "records": recs}}}
    return payload


def hits(payload):
    return {r["qa_id"]: float(r["metrics"]["hit_at_k"]) for r in payload["results"]["hipporag2"]["records"]}


def bm25_ref(split, bench="2wiki"):
    p = json.load(open(PAPER / "external" / bench / split / "_kgswap_ppr_kg_silver.json", encoding="utf-8"))
    return {r["qa_id"]: float(r["metrics"]["hit_at_k"]) for r in p["results"]["bm25"]["records"]}


def frozen_hits(split, kg, system, bench="2wiki"):
    p = json.load(open(PAPER / "external" / bench / split / f"_kgswap_{kg}.json", encoding="utf-8"))
    return {r["qa_id"]: float(r["metrics"]["hit_at_k"]) for r in p["results"][system]["records"]}


def synonymy_edges(v):
    """Synonymy edges HippoRAG added to the graph. Fact links are stored in both directions, so
    entity-entity graph edges = 2 x (entity pairs with a fact) + synonymy-only edges. HippoRAG's own
    get_graph_info()["num_synonymy_triples"] subtracts *unique facts* instead and is not a count."""
    return v["edges_by_type"].get("entity-entity", 0) - 2 * v["entity_pairs_from_facts"]


def paired(a, b):
    qs = sorted(q for q in a if q in b)
    diffs = [a[q] - b[q] for q in qs]
    random.seed(BOOT_SEED)
    lo, hi = boot_ci(diffs)
    pos, neg, p = sign_test(diffs)
    return {"n": len(qs), "mean_paired_diff": sum(diffs) / len(diffs), "ci95": [lo, hi],
            "discordant_pos": pos, "discordant_neg": neg, "sign_test_p": p}


# Follow-up conditions, same HippoRAG config / embedding / LLM / injection. "key" = summary.json key.
EXTRA = [
    {"key": "2wiki_kg_wikidata", "bench": "2wiki", "title": "2WikiMultiHop, realistic structure: live-Wikidata KG (kg_wikidata)",
     "conds": [("kg_silver", "aligned"), ("kg_openie", "independent"), ("kg_wikidata", "wikidata (realistic)")],
     "pairs": [("kg_wikidata", "kg_openie"), ("kg_wikidata", "kg_silver")], "write": ["kg_wikidata"]},
    {"key": "metaqa", "bench": "metaqa", "title": "MetaQA (2-hop; gold = the two bridge documents)",
     "conds": [("kg_silver", "aligned (question-induced)"), ("kg_openie", "independent"), ("kg_fullkb", "full KB (realistic)")],
     "pairs": [("kg_silver", "kg_openie"), ("kg_fullkb", "kg_openie"), ("kg_fullkb", "kg_silver")],
     "write": ["kg_silver", "kg_openie", "kg_fullkb"]},
    {"key": "graphragbench", "bench": "graphragbench", "title": "GraphRAG-Bench (medical; gold = 1-2 chunks, hit@5 = all gold in top 5)",
     "conds": [("kg_silver", "aligned"), ("kg_openie", "independent")],
     "pairs": [("kg_silver", "kg_openie")], "write": ["kg_silver", "kg_openie"]},
]


def extra_benchmark(spec, write_vectors=True):
    bench = spec["bench"]
    block = {"title": spec["title"], "splits": []}
    for split, split_name in SPLITS:
        runs = {}
        for kg, _ in spec["conds"]:
            try:
                runs[kg] = load_run(split, kg, "", bench)
            except FileNotFoundError:
                pass
        needed = {k for pair in spec["pairs"] for k in pair}
        if not any(a in runs and b in runs for a, b in spec["pairs"]):
            continue
        payloads = {kg: vectors(split, rows, bench=bench) for kg, (rows, _) in runs.items()}
        if write_vectors:
            for kg in spec["write"]:
                if kg in payloads:
                    path = PAPER / "external" / bench / split / f"_kgswap_hipporag2_{kg}.json"
                    json.dump(payloads[kg], open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        h = {kg: hits(p) for kg, p in payloads.items()}
        any_rows = next(iter(runs.values()))[0]
        dense = {kg: vectors(split, rows, key="dpr_ranked_passage_ids", bench=bench) for kg, (rows, _) in runs.items()}
        dense_ids = {kg: [r["retrieved_paper_ids"] for r in p["results"]["hipporag2"]["records"]] for kg, p in dense.items()}
        first = next(iter(dense_ids.values()))
        dh = hits(next(iter(dense.values())))
        bm = bm25_ref(split, bench)
        qids = list(next(iter(h.values())))
        e = {"split": split, "split_name": split_name, "n": len(any_rows),
             "hit5": {kg: sum(v.values()) / len(v) for kg, v in h.items()},
             "hit5_bm25": sum(bm[q] for q in qids) / len(qids),
             "hit5_dense_only_bge_m3": sum(dh.values()) / len(dh),
             "dense_only_identical_across_conditions": all(v == first for v in dense_ids.values()),
             "paired": {f"{a} - {b}": paired(h[a], h[b]) for a, b in spec["pairs"] if a in h and b in h},
             "graph": {}, "paths": {}, "paper_ppr_hit5": {}, "missing_conditions": sorted(needed - set(runs))}
        for kg, (rows, meta) in runs.items():
            v = meta["verification"]
            e["graph"][kg] = {"entity_nodes": v["graph_entity_nodes"], "fact_edges": v["graph_fact_store_size"],
                              "synonymy_edges": synonymy_edges(v),
                              "passage_nodes": v["passage_nodes"], "passages_linked": v["passages_linked_in_graph"],
                              "gold_passages_linked": v.get("gold_passages_linked"),
                              "gold_passages_in_split": v.get("gold_passages_in_split"),
                              "gold_fraction_of_linked": v.get("gold_fraction_of_linked"),
                              "graph_ecount": v["graph_ecount"],
                              "injected_triples_total": meta["injection"]["injected_triples_total"]}
            e["paths"][kg] = meta["path_counts"]
            try:
                f = frozen_hits(split, f"ppr_{kg}", "ppr", bench)
                e["paper_ppr_hit5"][kg] = sum(f.values()) / len(f)
            except (FileNotFoundError, KeyError):
                pass
        block["splits"].append(e)
    return block


def extra_markdown(spec, block):
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    labels = dict(spec["conds"])
    conds = [kg for kg, _ in spec["conds"]]
    L = ["", f"## {block['title']}", "",
         "Same HippoRAG 2 configuration, BGE-M3 embeddings, llama3.1:8b filter and node-provenance injection as above. "
         "BM25 = the frozen BM25 vectors of this benchmark; dense-only = HippoRAG's own BGE-M3 passage channel "
         "(KG-independent); paper PPR = the paper's PPR retriever on the same KG (frozen vectors).", "",
         "| split | n | " + " | ".join(f"hit@5 {labels[k]}" for k in conds) + " | BM25 | dense-only | " +
         " | ".join(f"paper PPR {labels[k]}" for k in conds) + " |",
         "|---|---:|" + "---:|" * (2 * len(conds) + 2)]
    for e in block["splits"]:
        L.append(f"| {e['split']} ({e['split_name']}) | {e['n']} | " + " | ".join(f3(e["hit5"].get(k)) for k in conds) +
                 f" | {f3(e['hit5_bm25'])} | {f3(e['hit5_dense_only_bge_m3'])} | " +
                 " | ".join(f3(e["paper_ppr_hit5"].get(k)) for k in conds) + " |")
    L += ["", "| split | paired difference | mean [95% CI] | discordant +/- | sign-test p |", "|---|---|---|---:|---:|"]
    for e in block["splits"]:
        for name, p in e["paired"].items():
            a, b = name.split(" - ")
            L.append(f"| {e['split']} | {labels[a]} minus {labels[b]} | {p['mean_paired_diff']:+.3f} "
                     f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | +{p['discordant_pos']}/-{p['discordant_neg']} | {p['sign_test_p']:.2e} |")
    L += ["", "| split | condition | entity nodes | fact edges | synonymy edges | passages linked / pool | gold passages linked | "
          "gold fraction of linked | injected passage-triples | graph edges | ppr / dpr_fallback |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in block["splits"]:
        for k in conds:
            if k not in e["graph"]:
                continue
            g, pc = e["graph"][k], e["paths"][k]
            L.append(f"| {e['split']} | {labels[k]} | {g['entity_nodes']} | {g['fact_edges']} | {g['synonymy_edges']} | "
                     f"{g['passages_linked']} / {g['passage_nodes']} | {g['gold_passages_linked']} / {g['gold_passages_in_split']} | "
                     f"{f3(g['gold_fraction_of_linked'])} | {g['injected_triples_total']} | {g['graph_ecount']} | "
                     f"{pc.get('ppr', 0)} / {pc.get('dpr_fallback', 0)} |")
    notes = [f"{e['split']}: dense-only ranking identical across conditions = {e['dense_only_identical_across_conditions']}"
             + (f"; missing: {', '.join(e['missing_conditions'])}" if e["missing_conditions"] else "") for e in block["splits"]]
    L += ["", "Checks: " + "; ".join(notes) + "."]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="run file suffix, e.g. _lim10 for a --limit 10 test run")
    ap.add_argument("--no-vectors", action="store_true", help="do not write external/2wiki/*/_kgswap_hipporag2_*.json")
    args = ap.parse_args()

    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": args.tag,
           "system": "HippoRAG 2 (hipporag 2.0.0a4)", "bootstrap": {"B": 10000, "seed": BOOT_SEED}, "splits": []}
    for split, split_name in SPLITS:
        cond = {}
        for kg, kg_name in KGS:
            try:
                rows, meta = load_run(split, kg, args.tag)
            except FileNotFoundError:
                break
            payload = vectors(split, rows)
            dpr = vectors(split, rows, key="dpr_ranked_passage_ids") if "dpr_ranked_passage_ids" in rows[0] else None
            if not args.no_vectors:
                path = PAPER / "external" / "2wiki" / split / f"_kgswap_hipporag2_{kg}.json"
                json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            v = meta["verification"]
            cond[kg] = {
                "label": kg_name, "payload": payload, "dpr": dpr,
                "hit5": payload["results"]["hipporag2"]["summary"]["overall"]["hit_at_k"],
                "mrr": payload["results"]["hipporag2"]["summary"]["overall"]["mrr"],
                "paths": meta["path_counts"], "filter_errors": meta["filter_errors"],
                "graph": {"entity_nodes": v["graph_entity_nodes"], "fact_edges": v["graph_fact_store_size"],
                          "passage_nodes": v["passage_nodes"], "passages_linked": v["passages_linked_in_graph"],
                          "gold_passages_linked": v.get("gold_passages_linked"),
                          "gold_passages_in_split": v.get("gold_passages_in_split"),
                          "gold_fraction_of_linked": v.get("gold_fraction_of_linked"),
                          "synonymy_edges": synonymy_edges(v),
                          "graph_vcount": v["graph_vcount"], "graph_ecount": v["graph_ecount"],
                          "injected_triples_total": meta["injection"]["injected_triples_total"]},
                "llm_cache_rows": meta["llm_cache_rows_after_run"],
            }
        if len(cond) < 2:
            continue
        hs, ho = hits(cond["kg_silver"]["payload"]), hits(cond["kg_openie"]["payload"])
        bm = bm25_ref(split)
        bm_mean = sum(bm[q] for q in hs) / len(hs)
        entry = {"split": split, "split_name": split_name, "n": len(hs),
                 "hit5_aligned": cond["kg_silver"]["hit5"], "hit5_independent": cond["kg_openie"]["hit5"],
                 "hit5_bm25": bm_mean,
                 "delta_vs_bm25_aligned": cond["kg_silver"]["hit5"] - bm_mean,
                 "delta_vs_bm25_independent": cond["kg_openie"]["hit5"] - bm_mean,
                 "paired_aligned_minus_independent": paired(hs, ho),
                 "mrr_aligned": cond["kg_silver"]["mrr"], "mrr_independent": cond["kg_openie"]["mrr"],
                 "retrieval_path_counts": {k: cond[k]["paths"] for k, _ in KGS},
                 "filter_errors": {k: cond[k]["filter_errors"] for k, _ in KGS},
                 "graph": {k: cond[k]["graph"] for k, _ in KGS},
                 "llm_cache_rows": {k: cond[k]["llm_cache_rows"] for k, _ in KGS}}
        if cond["kg_silver"]["dpr"] is not None:
            ds, do = hits(cond["kg_silver"]["dpr"]), hits(cond["kg_openie"]["dpr"])
            entry["dpr_only_identical_across_conditions"] = ds == do and all(
                a["retrieved_paper_ids"] == b["retrieved_paper_ids"] for a, b in
                zip(cond["kg_silver"]["dpr"]["results"]["hipporag2"]["records"],
                    cond["kg_openie"]["dpr"]["results"]["hipporag2"]["records"]))
            entry["hit5_dense_only_bge_m3"] = sum(ds.values()) / len(ds)
            entry["paired_aligned_minus_dense_only"] = paired(hs, ds)
            entry["paired_independent_minus_dense_only"] = paired(ho, ds)
        # sensitivity (pre-declared): OpenIE triples only at their extraction passage
        try:
            rows_e, meta_e = load_run(split, "kg_openie__edgeprov", args.tag)
            he = hits(vectors(split, rows_e))
            v = meta_e["verification"]
            entry["sensitivity_edge_provenance"] = {
                "hit5_independent_edgeprov": sum(he.values()) / len(he),
                "paired_aligned_minus_independent_edgeprov": paired(hs, he),
                "paired_independent_edgeprov_minus_node": paired(he, ho),
                "paths": meta_e["path_counts"],
                "graph": {"entity_nodes": v["graph_entity_nodes"], "fact_edges": v["graph_fact_store_size"],
                          "passages_linked": v["passages_linked_in_graph"], "passage_nodes": v["passage_nodes"],
                          "gold_passages_linked": v["gold_passages_linked"],
                          "gold_fraction_of_linked": v["gold_fraction_of_linked"],
                          "synonymy_edges": synonymy_edges(v),
                          "injected_triples_total": meta_e["injection"]["injected_triples_total"]}}
        except FileNotFoundError:
            pass
        # context: the paper's own retrievers under the same two KGs (frozen vectors)
        ctx = {}
        for system, stem in (("ppr", "ppr_"), ("proposed_v3", "")):
            a, b = frozen_hits(split, stem + "kg_silver", system), frozen_hits(split, stem + "kg_openie", system)
            ctx[system] = {"hit5_aligned": sum(a.values()) / len(a), "hit5_independent": sum(b.values()) / len(b),
                           "paired": paired(a, b)}
        entry["paper_retrievers_same_kgs"] = ctx
        out["splits"].append(entry)

    suffix = args.tag
    json.dump(out, open(OUTD / f"summary{suffix}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    L = [f"# HippoRAG 2 KG swap on 2WikiMultiHop{' (' + suffix + ')' if suffix else ''}", "",
         f"Generated {out['generated']} by `eval/_kg_swap_hipporag2_score.py`. Methods: `METHODS.md`. "
         "hit@5 = both gold passages in the top 5. Paired difference = aligned (kg_silver) minus independent "
         f"(kg_openie), same questions; bootstrap B=10,000, seed {BOOT_SEED}; exact two-sided sign test on discordant questions.", "",
         "| split | n | hit@5 aligned | hit@5 independent | hit@5 BM25 | hit@5 dense-only | paired diff [95% CI] | discordant +/- | sign-test p |",
         "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for e in out["splits"]:
        p = e["paired_aligned_minus_independent"]
        L.append(f"| {e['split']} ({e['split_name']}) | {e['n']} | {f3(e['hit5_aligned'])} | {f3(e['hit5_independent'])} | "
                 f"{f3(e['hit5_bm25'])} | {f3(e.get('hit5_dense_only_bge_m3'))} | {p['mean_paired_diff']:+.3f} "
                 f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | +{p['discordant_pos']}/-{p['discordant_neg']} | {p['sign_test_p']:.2e} |")
    L += ["", "## Each condition against HippoRAG's own dense-only channel (BGE-M3, KG-independent)", "",
          "| split | condition | hit@5 | minus dense-only [95% CI] | discordant +/- | sign-test p |", "|---|---|---:|---|---:|---:|"]
    for e in out["splits"]:
        for key, lab, h in (("paired_aligned_minus_dense_only", "aligned", e["hit5_aligned"]),
                            ("paired_independent_minus_dense_only", "independent", e["hit5_independent"])):
            if key in e:
                p = e[key]
                L.append(f"| {e['split']} | {lab} | {f3(h)} | {p['mean_paired_diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | "
                         f"+{p['discordant_pos']}/-{p['discordant_neg']} | {p['sign_test_p']:.2e} |")
    L += ["", "## Graph HippoRAG built from each injected KG", "",
          "| split | condition | entity nodes | fact edges (unique triples) | synonymy edges | passages linked / pool | gold passages linked | gold fraction of linked | injected passage-triples |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in out["splits"]:
        for k, lab in KGS:
            g = e["graph"][k]
            L.append(f"| {e['split']} | {lab} | {g['entity_nodes']} | {g['fact_edges']} | {g['synonymy_edges']} | "
                     f"{g['passages_linked']} / {g['passage_nodes']} | {g['gold_passages_linked']} / {g['gold_passages_in_split']} | "
                     f"{f3(g['gold_fraction_of_linked'])} | {g['injected_triples_total']} |")
    L += ["", "## Sensitivity (pre-declared): independent KG with edge-level provenance", "",
          "OpenIE triples injected only at the passage they were extracted from (edge `source_docs`), which is what "
          "HippoRAG's own OpenIE step would produce; the 3.2k/3.4k provenance-free surface-synonym edges of the OpenIE KG "
          "are dropped (HippoRAG adds its own synonymy edges). Not the protocol condition; reported for robustness.", "",
          "| split | hit@5 independent (edge prov.) | aligned minus it [95% CI] | discordant +/- | p | edge minus node prov. [95% CI] | linked passages | gold fraction of linked |",
          "|---|---:|---|---:|---:|---|---:|---:|"]
    for e in out["splits"]:
        se = e.get("sensitivity_edge_provenance")
        if se:
            p, q, g = se["paired_aligned_minus_independent_edgeprov"], se["paired_independent_edgeprov_minus_node"], se["graph"]
            L.append(f"| {e['split']} | {f3(se['hit5_independent_edgeprov'])} | {p['mean_paired_diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | "
                     f"+{p['discordant_pos']}/-{p['discordant_neg']} | {p['sign_test_p']:.2e} | {q['mean_paired_diff']:+.3f} [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}] | "
                     f"{g['passages_linked']} / {g['passage_nodes']} | {f3(g['gold_fraction_of_linked'])} |")
    L += ["", "## Context: the paper's own retrievers under the same two KGs (frozen vectors, not re-run)", "",
          "| split | retriever | hit@5 aligned | hit@5 independent | paired diff [95% CI] | p |", "|---|---|---:|---:|---|---:|"]
    for e in out["splits"]:
        for system, c in e["paper_retrievers_same_kgs"].items():
            p = c["paired"]
            L.append(f"| {e['split']} | {system} | {f3(c['hit5_aligned'])} | {f3(c['hit5_independent'])} | "
                     f"{p['mean_paired_diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['sign_test_p']:.2e} |")
    L += ["", "## Retrieval path (recognition-memory filter output)", "",
          "`ppr` = the LLM fact filter kept at least one fact and HippoRAG ran PPR; `dpr_fallback` = no fact kept, "
          "HippoRAG returned its dense-passage ranking (default behaviour).", "",
          "| split | condition | ppr | dpr_fallback | filter exceptions | LLM calls cached |", "|---|---|---:|---:|---:|---:|"]
    for e in out["splits"]:
        for k, lab in KGS:
            pc = e["retrieval_path_counts"][k]
            L.append(f"| {e['split']} | {lab} | {pc.get('ppr', 0)} | {pc.get('dpr_fallback', 0)} | {e['filter_errors'][k]} | {e['llm_cache_rows'][k]} |")
    e2e = OUTD / "e2e_qa" / "qwen2_5_14b" / "summary.json"
    if e2e.exists() and not suffix:
        ed = json.load(open(e2e, encoding="utf-8"))
        rows = [x for x in ed["paired"] if x["system"] == "hipporag2"]
        out["e2e_qwen2_5_14b"] = rows
        json.dump(out, open(OUTD / f"summary{suffix}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        cells = {(c["split"], c["condition"]): c for c in ed["cells"]}
        L += ["", "## End-to-end QA (reader qwen2.5:14b; eval/e2e_qa_swap.py prompt and scoring; `e2e_qa/qwen2_5_14b/summary.md`)", "",
              "| split | EM aligned | EM independent | EM diff [95% CI] | p (sign) | F1 diff [95% CI] | EM diff on retrieval flips + (n) | EM diff on no flip (n) | EM closed-book / oracle |",
              "|---|---:|---:|---|---:|---|---:|---:|---:|"]
        for x in sorted(rows, key=lambda r: r["split"] != "sub200"):
            em, f1, m = x["em"], x["f1"], x["mediation"]
            cb, orc = cells[(x["split"], "closedbook")]["em"], cells[(x["split"], "oracle")]["em"]
            L.append(f"| {x['split']} | {em['silver']:.3f} | {em['openie']:.3f} | {em['mean_paired_diff']:+.3f} [{em['ci95'][0]:+.3f}, {em['ci95'][1]:+.3f}] | "
                     f"{em['sign_test_p']:.2e} | {f1['mean_paired_diff']:+.3f} [{f1['ci95'][0]:+.3f}, {f1['ci95'][1]:+.3f}] | "
                     f"{m['em_diff_on_retrieval_flips_pos']:+.3f} ({m['n_flips_pos']}) | {m['em_diff_on_no_flip']:+.3f} ({m['n_no_flip']}) | {cb:.3f} / {orc:.3f} |")
    if not suffix:
        out["benchmarks"] = {}
        for spec in EXTRA:
            block = extra_benchmark(spec, write_vectors=not args.no_vectors)
            if block["splits"]:
                out["benchmarks"][spec["key"]] = block
                L += extra_markdown(spec, block)
        json.dump(out, open(OUTD / f"summary{suffix}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(OUTD / f"summary{suffix}.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
