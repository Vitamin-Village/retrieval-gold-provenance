# -*- coding: utf-8 -*-
"""Score the HippoRAG 2 KG-swap runs on the 2WikiMultiHop compositional split (comp200).

Protocol: results/twowiki_comp200/PROTOCOL.md, Section 7.

Uses the functions of eval/_kg_swap_hipporag2_score.py unchanged (shared scorer
eval_independent.score_record / summarize_system via `vectors`; paired statistics via `paired`:
10,000-resample bootstrap with random.seed(20260918) right before resampling, sorted qa ids, exact
two-sided sign test), with its run directory pointed at results/twowiki_comp200/hipporag2/runs/.
That script's main() is not run: it would rewrite results/hipporag2_swap/summary.* and the
existing splits' vectors.

Writes
  external/2wiki/comp200/_kgswap_hipporag2_{kg_silver,kg_openie}.json   (same layout as the
                                                                          existing splits' files)
  results/twowiki_comp200/hipporag2/summary.{json,md}

Run from the repo root (parent of paper/) with env new_env:
    python paper/eval/_kg_swap_hipporag2_comp200_score.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import _kg_swap_hipporag2_score as S  # noqa: E402

SPLIT = "comp200"
S.RUNS = S.PAPER / "results" / "twowiki_comp200" / "hipporag2" / "runs"
OUTD = S.PAPER / "results" / "twowiki_comp200" / "hipporag2"
KGS = (("kg_silver", "aligned"), ("kg_openie", "independent"))


def main():
    cond = {}
    for kg, lab in KGS:
        rows, meta = S.load_run(SPLIT, kg, "")
        payload = S.vectors(SPLIT, rows)
        dpr = S.vectors(SPLIT, rows, key="dpr_ranked_passage_ids")
        path = S.PAPER / "external" / "2wiki" / SPLIT / f"_kgswap_hipporag2_{kg}.json"
        json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        v = meta["verification"]
        cond[kg] = {"label": lab, "payload": payload, "dpr": dpr,
                    "hit5": payload["results"]["hipporag2"]["summary"]["overall"]["hit_at_k"],
                    "mrr": payload["results"]["hipporag2"]["summary"]["overall"]["mrr"],
                    "paths": meta["path_counts"], "filter_errors": meta["filter_errors"],
                    "llm_cache_rows": meta["llm_cache_rows_after_run"], "seconds": meta["seconds"],
                    "graph": {"entity_nodes": v["graph_entity_nodes"], "fact_edges": v["graph_fact_store_size"],
                              "synonymy_edges": S.synonymy_edges(v), "passage_nodes": v["passage_nodes"],
                              "passages_linked": v["passages_linked_in_graph"],
                              "gold_passages_linked": v["gold_passages_linked"],
                              "gold_passages_in_split": v["gold_passages_in_split"],
                              "gold_fraction_of_linked": v["gold_fraction_of_linked"],
                              "graph_vcount": v["graph_vcount"], "graph_ecount": v["graph_ecount"],
                              "passage_entity_neighbourhood_mismatches": v["passage_entity_neighbourhood_mismatches"],
                              "facts_equal": v["facts_equal"], "entities_equal": v["entities_equal"],
                              "injected_triples_total": meta["injection"]["injected_triples_total"]},
                    "chunk_embeddings_seeded_from": meta.get("chunk_embeddings_seeded_from")}
    hs, ho = S.hits(cond["kg_silver"]["payload"]), S.hits(cond["kg_openie"]["payload"])
    bm = S.bm25_ref(SPLIT)
    assert set(bm) == set(hs) == set(ho)
    bm_mean = sum(bm.values()) / len(bm)
    ds, do = S.hits(cond["kg_silver"]["dpr"]), S.hits(cond["kg_openie"]["dpr"])
    dpr_same = ds == do and all(
        a["retrieved_paper_ids"] == b["retrieved_paper_ids"] for a, b in
        zip(cond["kg_silver"]["dpr"]["results"]["hipporag2"]["records"], cond["kg_openie"]["dpr"]["results"]["hipporag2"]["records"]))
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": "HippoRAG 2 (hipporag 2.0.0a4)", "split": SPLIT, "n": len(hs),
        "bootstrap": {"B": 10000, "seed": S.BOOT_SEED},
        "hit5_aligned": cond["kg_silver"]["hit5"], "hit5_independent": cond["kg_openie"]["hit5"], "hit5_bm25": bm_mean,
        "delta_vs_bm25_aligned": cond["kg_silver"]["hit5"] - bm_mean,
        "delta_vs_bm25_independent": cond["kg_openie"]["hit5"] - bm_mean,
        "paired_aligned_minus_independent": S.paired(hs, ho),
        "paired_aligned_minus_bm25": S.paired(hs, bm), "paired_independent_minus_bm25": S.paired(ho, bm),
        "hit5_dense_only_bge_m3": sum(ds.values()) / len(ds), "dpr_only_identical_across_conditions": dpr_same,
        "paired_aligned_minus_dense_only": S.paired(hs, ds), "paired_independent_minus_dense_only": S.paired(ho, ds),
        "mrr_aligned": cond["kg_silver"]["mrr"], "mrr_independent": cond["kg_openie"]["mrr"],
        "retrieval_path_counts": {k: cond[k]["paths"] for k, _ in KGS},
        "filter_errors": {k: cond[k]["filter_errors"] for k, _ in KGS},
        "llm_cache_rows": {k: cond[k]["llm_cache_rows"] for k, _ in KGS},
        "seconds": {k: cond[k]["seconds"] for k, _ in KGS},
        "chunk_embeddings_seeded_from": {k: cond[k]["chunk_embeddings_seeded_from"] for k, _ in KGS},
        "graph": {k: cond[k]["graph"] for k, _ in KGS},
    }
    OUTD.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTD / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    p = out["paired_aligned_minus_independent"]
    f3 = lambda x: f"{x:.3f}"
    L = ["# HippoRAG 2 KG swap on the 2WikiMultiHop compositional split (comp200)", "",
         f"Generated {out['generated']} by `eval/_kg_swap_hipporag2_comp200_score.py`; configuration as Appendix C.7 "
         "(`results/hipporag2_swap/METHODS.md`), protocol `results/twowiki_comp200/PROTOCOL.md`.", "",
         "| n | hit@5 aligned | hit@5 independent | hit@5 BM25 | hit@5 dense-only (BGE-M3) | aligned minus independent [95% CI] | discordant +/- | sign-test p |",
         "|---:|---:|---:|---:|---:|---|---:|---:|",
         f"| {out['n']} | {f3(out['hit5_aligned'])} | {f3(out['hit5_independent'])} | {f3(bm_mean)} | {f3(out['hit5_dense_only_bge_m3'])} | "
         f"{p['mean_paired_diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | +{p['discordant_pos']}/-{p['discordant_neg']} | {p['sign_test_p']:.2e} |", "",
         "| condition | minus BM25 [95% CI] | +/- | p | minus dense-only [95% CI] | +/- | p |", "|---|---|---:|---:|---|---:|---:|"]
    for k, lab in (("aligned", "aligned"), ("independent", "independent")):
        a, b = out[f"paired_{k}_minus_bm25"], out[f"paired_{k}_minus_dense_only"]
        L.append(f"| {lab} | {a['mean_paired_diff']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | +{a['discordant_pos']}/-{a['discordant_neg']} | {a['sign_test_p']:.2e} | "
                 f"{b['mean_paired_diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | +{b['discordant_pos']}/-{b['discordant_neg']} | {b['sign_test_p']:.2e} |")
    L += ["", "| condition | entity nodes | fact edges | synonymy edges | passages linked / pool | gold passages linked | gold fraction of linked | injected passage-triples | ppr / dpr_fallback | filter exceptions |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, lab in KGS:
        g, pc = out["graph"][k], out["retrieval_path_counts"][k]
        L.append(f"| {lab} | {g['entity_nodes']} | {g['fact_edges']} | {g['synonymy_edges']} | {g['passages_linked']} / {g['passage_nodes']} | "
                 f"{g['gold_passages_linked']} / {g['gold_passages_in_split']} | {f3(g['gold_fraction_of_linked'])} | {g['injected_triples_total']} | "
                 f"{pc.get('ppr', 0)} / {pc.get('dpr_fallback', 0)} | {out['filter_errors'][k]} |")
    L += ["", f"Checks: dense-only ranking identical across the two conditions = {dpr_same}; graph = injection "
          f"(facts, entities, 0 passage-neighbourhood mismatches) asserted by the driver for both conditions."]
    (OUTD / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
