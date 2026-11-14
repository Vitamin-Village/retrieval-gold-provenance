# -*- coding: utf-8 -*-
"""Retriever-robustness rerun of the KG swap with a real dense channel (BGE-M3)
instead of the deterministic surrogate used in the frozen runs.

For every benchmark/split with frozen swap data, re-run the two hybrid systems
(proposed_v3 = audited, proposed_v5_graphcentric) plus bm25 and a dense-only
retriever under both KG conditions, with the dense channel backed by
BAAI/bge-m3 (local cache, GPU if available). Outputs never overwrite the frozen
vectors: they go to <split>/_dense_bge_m3/_kgswap_<system>_<kg>.json.

Aborts if the BGE-M3 encoder cannot be loaded (no silent fallback).

Pooling (2026-09-29): the first rerun (2026-09-20) pooled BGE-M3 with an attention-mask
MEAN. BGE-M3's dense embedding is the L2-normalised [CLS] vector (FlagEmbedding
BGEM3FlagModel pooling_method='cls'; sentence-transformers 1_Pooling/config.json
pooling_mode_cls_token=true), so --pooling cls (default) is the correct run; --pooling mean
reproduces the 2026-09-20 outputs bit for bit (checked by eval/_bge_pooling_check.py).
Input text, max_length 512, truncation, and no query instruction are unchanged.

Run from repo root with the rag-bge-m3-py311 interpreter:
    python paper/eval/_kg_swap_dense_bge.py --benchmarks 2wiki,metaqa,stark,graphragbench
Outputs (cls):  <split>/_dense_bge_m3_cls/_kgswap_all_<kg>_cls.json,
                paper/results/kg_swap_dense_bge_m3_cls.json, paper/analysis/dense_bge_swap_summary_cls.md
Outputs (mean): <split>/_dense_bge_m3/_kgswap_all_<kg>.json,
                paper/results/kg_swap_dense_bge_m3.json, paper/analysis/dense_bge_swap_summary.md
The cls vectors live in a separate directory so the e2e_qa_swap.py / build_submission.py
globs over _dense_bge_m3/ never pick them up implicitly.
"""
import os
os.environ["RAG_V3_DENSE_BACKEND"] = "bge-m3"
os.environ.setdefault("RAG_V3_DENSE_LOCAL_FILES_ONLY", "1")
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402
from prototype.eval.run_retrieval_eval_v3 import DenseRetrieverConfig, _load_bge_m3_encoder  # noqa: E402
from paper.eval._kg_swap_metaqa import per_q_hits, per_q_ids, overall_hit, boot_ci, sign_test  # noqa: E402

BOOT_SEED = 20260919
SYSTEMS = ("bm25", "dense", "proposed_v3", "proposed_v5_graphcentric")
HYBRID = ("proposed_v3", "proposed_v5_graphcentric")

import prototype.eval.run_retrieval_eval_v3 as _rv  # noqa: E402


class CudaBgeEncoder:
    """BGE-M3 dense encoder (pooling 'cls' = official BGE-M3 dense vector, or 'mean' = the
    2026-09-20 rerun / the runner's transformers path) + L2 normalisation, on the GPU when
    available and with an embedding cache so the passage pool is encoded once per split
    instead of once per system."""
    backend_name = "bge-m3"
    model_name = "BAAI/bge-m3"

    def __init__(self, pooling="cls"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        assert pooling in ("cls", "mean"), pooling
        self.torch = torch
        self.pooling = pooling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device).eval()
        self.cache = {}
        print("BGE-M3 on", self.device, "pooling", pooling, flush=True)

    def encode(self, texts, *, batch_size):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        with self.torch.no_grad():
            for s in range(0, len(todo), batch_size):
                batch = todo[s:s + batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().tolist()
                for t, v in zip(batch, pooled):
                    self.cache[t] = v
        return [self.cache[t] for t in texts]


_ENC = None
_POOLING = "cls"


def _cached_encoder(config):
    global _ENC
    if _ENC is None:
        _ENC = CudaBgeEncoder(_POOLING)
    return _ENC


_rv._load_bge_m3_encoder = _cached_encoder


def main(argv=None):
    global _POOLING
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="2wiki,metaqa,stark,graphragbench")
    ap.add_argument("--splits", default="sub200,holdout200")
    ap.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    args = ap.parse_args(argv)
    _POOLING = args.pooling
    sfx = "_cls" if args.pooling == "cls" else ""
    cfg = DenseRetrieverConfig(backend="bge-m3", local_files_only=True, batch_size=16)
    _cached_encoder(cfg)  # raises if unavailable; no silent fallback allowed here
    print("BGE-M3 encoder loaded", flush=True)

    rows, cells = [], []
    for b in args.benchmarks.split(","):
        for s in args.splits.split(","):
            d = ROOT / "paper" / "external" / b / s
            if not (d / "kg_openie" / "proposed_kg_nodes.json").exists():
                print("SKIP", b, s, flush=True)
                continue
            out_dir = d / f"_dense_bge_m3{sfx}"
            payload = {}
            for kg in ("kg_silver", "kg_openie"):
                payload[kg] = ev.run(qa_path=d / "qa.jsonl", passages_path=d / "passages.jsonl", manifest_path=d / "manifest.json",
                                     kg_dir=d / kg, systems=SYSTEMS, out_path=out_dir / f"_kgswap_all_{kg}{sfx}.json", top_k=5, dense_config=cfg)
                for sysname in SYSTEMS:
                    cells.append({"bench": b, "split": s, "kg": kg, "system": sysname, "hit5": overall_hit(payload[kg], sysname)})
                    print(f"{b:14s} {s:10s} {kg:9s} {sysname:24s} hit@5={overall_hit(payload[kg], sysname):.3f}", flush=True)
            for base in ("bm25", "dense"):
                a, o = per_q_ids(payload["kg_silver"], base), per_q_ids(payload["kg_openie"], base)
                assert all(a[q] == o[q] for q in a), f"{base} differs across KG conditions ({b} {s})"
            for sysname in HYBRID:
                hs, ho = per_q_hits(payload["kg_silver"], sysname), per_q_hits(payload["kg_openie"], sysname)
                qs = sorted(q for q in hs if q in ho)
                diffs = [hs[q] - ho[q] for q in qs]
                random.seed(BOOT_SEED)
                lo, hi = boot_ci(diffs)
                pos, neg, p = sign_test(diffs)
                rows.append({"bench": b, "split": s, "system": sysname, "dense_backend": "bge-m3", "pooling": args.pooling, "n": len(qs),
                             "silver": overall_hit(payload["kg_silver"], sysname), "openie": overall_hit(payload["kg_openie"], sysname),
                             "bm25": overall_hit(payload["kg_silver"], "bm25"), "dense_only": overall_hit(payload["kg_silver"], "dense"),
                             "mean_paired_diff": sum(diffs) / len(diffs), "ci95": [lo, hi], "discordant_pos": pos, "discordant_neg": neg, "sign_test_p": p})
                print(f"  SWAP {sysname}: {rows[-1]['mean_paired_diff']:+.3f} [{lo:+.3f},{hi:+.3f}] +{pos}/-{neg} p={p:.2e}", flush=True)
    json.dump({"rows": rows, "cells": cells, "pooling": args.pooling, "generated": datetime.now(timezone.utc).isoformat()},
              open(ROOT / "paper" / "results" / f"kg_swap_dense_bge_m3{sfx}.json", "w", encoding="utf-8"), indent=1)
    title = "BGE-M3, [CLS] pooling" if args.pooling == "cls" else "BGE-M3"
    L = [f"# KG swap with a real dense channel ({title})", "",
         "| Benchmark | Split | System | BM25 | Dense-only | Graph, aligned KG | Graph, independent KG | Swap [95% CI] | +/- | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['bench']} | {r['split']} | {r['system']} | {r['bm25']:.3f} | {r['dense_only']:.3f} | {r['silver']:.3f} | {r['openie']:.3f} | "
                 f"{r['mean_paired_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | +{r['discordant_pos']}/-{r['discordant_neg']} | {r['sign_test_p']:.2e} |")
    old_path = ROOT / "paper" / "results" / "kg_swap_dense_bge_m3.json"
    if sfx and old_path.exists():
        old = {(r["bench"], r["split"], r["system"]): r for r in json.load(open(old_path, encoding="utf-8"))["rows"]}
        L += ["", "## Mean pooling (2026-09-20) vs [CLS] pooling", "",
              "Mean pooling = `results/kg_swap_dense_bge_m3.json` (attention-mask mean pooling); [CLS] pooling = this run. Same "
              "questions, pool, input text, max_length 512, seeds, bootstrap and sign test; only the pooling differs.", "",
              "| Benchmark | Split | System | Dense-only mean / [CLS] | Aligned mean / [CLS] | Independent mean / [CLS] | Swap mean | Swap [CLS] [95% CI] | p mean / [CLS] |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            o = old.get((r["bench"], r["split"], r["system"]))
            if o is None:
                continue
            L.append(f"| {r['bench']} | {r['split']} | {r['system']} | {o['dense_only']:.3f} / {r['dense_only']:.3f} | "
                     f"{o['silver']:.3f} / {r['silver']:.3f} | {o['openie']:.3f} / {r['openie']:.3f} | {o['mean_paired_diff']:+.3f} | "
                     f"{r['mean_paired_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {o['sign_test_p']:.2e} / {r['sign_test_p']:.2e} |")
    (ROOT / "paper" / "analysis" / f"dense_bge_swap_summary{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
