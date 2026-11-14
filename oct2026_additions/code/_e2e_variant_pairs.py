"""Answer-level paired comparison of a variant KG condition against the independent OpenIE KG
(and against the aligned KG), from results/e2e_qa/<reader>/per_question.jsonl.
Used for the realistic-structure controls (2WikiMultiHop kg_wikidata, MetaQA kg_fullkb) and for
HippoRAG 2 (condition names hipporag2_kg_silver / hipporag2_kg_openie).
Usage (from paper/): python eval/_e2e_variant_pairs.py
Output: results/e2e_qa/variant_pairs.{json,md}"""
import collections
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_qa_swap import bootstrap_ci, sign_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e2e_qa"
READERS = ["llama3_1_8b", "qwen2_5_14b"]
# (bench, variant condition, variant system, reference condition, reference system, label)
PAIRS = [("2wiki", "kg_wikidata", s, "kg_openie", s, "live Wikidata vs OpenIE") for s in ("audited", "ppr", "graph_centric")]
PAIRS += [("metaqa", "kg_fullkb", s, "kg_openie", s, "full KB vs OpenIE") for s in ("audited", "ppr", "graph_centric")]
PAIRS += [("2wiki", "hipporag2_kg_silver", "hipporag2", "hipporag2_kg_openie", "hipporag2", "HippoRAG 2 aligned vs OpenIE")]
PAIRS += [("2wiki", "hipporag2_kg_silver", "hipporag2", "none", "bm25", "HippoRAG 2 aligned vs BM25")]
PAIRS += [("2wiki", "hipporag2_kg_openie", "hipporag2", "none", "bm25", "HippoRAG 2 OpenIE vs BM25")]

out, md = [], ["| reader | bench | split | comparison | system | n | EM a | EM b | Δ [95% CI] | +/- | p | hit@5 Δ |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for tag in READERS:
    rows = collections.defaultdict(dict)
    for l in open(RES / tag / "per_question.jsonl", encoding="utf-8"):
        r = json.loads(l)
        rows[(r["bench"], r["split"], r["condition"], r["system"])][r["qa_id"]] = r
    for bench, vc, vs, rc, rs, label in PAIRS:
        for split in ("sub200", "holdout200"):
            A, B = rows.get((bench, split, vc, vs), {}), rows.get((bench, split, rc, rs), {})
            ids = sorted(set(A) & set(B))
            if len(ids) < 150:
                continue
            d = np.array([A[i]["em"] - B[i]["em"] for i in ids])
            h = np.array([A[i]["retrieval_hit"] - B[i]["retrieval_hit"] for i in ids])
            pos, neg = int((d > 0).sum()), int((d < 0).sum())
            e = {"reader": tag, "bench": bench, "split": split, "comparison": label, "system": vs, "n": len(ids),
                 "em_a": float(np.mean([A[i]["em"] for i in ids])), "em_b": float(np.mean([B[i]["em"] for i in ids])),
                 "diff": float(d.mean()), "ci": bootstrap_ci(d), "pos": pos, "neg": neg, "p": sign_test(pos, neg),
                 "hit_diff": float(np.nanmean(h))}
            out.append(e)
            md.append(f"| {tag} | {bench} | {split} | {label} | {vs} | {e['n']} | {e['em_a']:.3f} | {e['em_b']:.3f} | "
                      f"{e['diff']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] | {pos}/{neg} | {e['p']:.2g} | {e['hit_diff']:+.3f} |")
json.dump(out, open(RES / "variant_pairs.json", "w", encoding="utf-8"), indent=1)
(RES / "variant_pairs.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
