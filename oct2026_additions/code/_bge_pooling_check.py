# -*- coding: utf-8 -*-
"""Diagnostic: BGE-M3 dense-only hit@5 under different pooling / input formats.

The BGE-M3 dense-channel rerun (eval/_kg_swap_dense_bge.py, CudaBgeEncoder) pooled
BGE-M3 with an attention-mask MEAN. BGE-M3's dense embedding is the L2-normalised
[CLS] vector (FlagEmbedding BGEM3FlagModel pooling_method='cls'; the model's
sentence-transformers 1_Pooling/config.json has pooling_mode_cls_token=true).
This script re-encodes one split's passage pool and questions and scores dense-only
hit@5 for each variant, with the same ranking rule as the runner's
BgeM3DenseRetriever (dot product of L2-normalised vectors, score > 0, ties by id).

Variants
  runner-text  = run_retrieval_eval_v3._paper_text(paper): title + passage text x3
                 (eval_independent.build_candidate_papers copies the passage text into
                 abstract/content/text), tokenizer max_length 512. This is the exact
                 input of the frozen BGE rerun.
  hippo-text   = "{title}\\n{text}" (HippoRAG 2 corpus format), max_length 8192.
  pooling      = mean (frozen rerun) | cls (official)
  flag         = FlagEmbedding BGEM3FlagModel(use_fp16=False).encode(...)["dense_vecs"], runner text
  st           = SentenceTransformer('BAAI/bge-m3').encode(normalize_embeddings=True)
                 (the HippoRAG 2 'Transformers/BAAI/bge-m3' backend)

Run (from repo root, rag-bge-m3-py311):
    python paper/eval/_bge_pooling_check.py --benchmarks 2wiki --splits sub200
Output: paper/results/bge_pooling_check.json
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.eval import eval_independent as ev  # noqa: E402
from prototype.eval.run_retrieval_eval_v3 import _paper_text, _record_id  # noqa: E402

MODEL = "BAAI/bge-m3"


def load(bench, split):
    d = ROOT / "paper" / "external" / bench / split
    qa = ev.load_jsonl(d / "qa.jsonl")
    passages = ev.load_jsonl(d / "passages.jsonl")
    by_paper, _ = ev.build_passage_indexes(passages)
    papers = ev.build_candidate_papers(ev._load_manifest_records(d / "manifest.json"), by_paper)
    title_by_pid = {}
    for p in passages:
        title_by_pid.setdefault(str(p.get("paper_id")), str(p.get("title") or ""))
    return d, qa, papers, by_paper, title_by_pid


class HfEncoder:
    """One forward pass per text; returns both mean- and CLS-pooled L2-normalised vectors."""

    def __init__(self, torch, model, tok, device):
        self.torch, self.model, self.tok, self.device = torch, model, tok, device

    def encode(self, texts, max_length, batch_size=16, token_budget=16 * 512):
        """Batches of up to `batch_size` texts in input order (as the runner does), split
        further so that batch_len * padded_len <= token_budget (long pools at 8192)."""
        lens = [min(len(x), max_length) for x in self.tok(texts, add_special_tokens=True, truncation=True,
                                                            max_length=max_length)["input_ids"]]
        mean_v, cls_v = [], []
        with self.torch.no_grad():
            s = 0
            while s < len(texts):
                e = s + 1
                while e < len(texts) and e - s < batch_size and (e - s + 1) * max(lens[s:e + 1]) <= token_budget:
                    e += 1
                enc = self.tok(texts[s:e], padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).expand(h.size()).float()
                mean = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                cls = h[:, 0]
                mean_v.append(self.torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy())
                cls_v.append(self.torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy())
                s = e
        return np.concatenate(mean_v).astype(np.float64), np.concatenate(cls_v).astype(np.float64)


def rank_top5(P, Q, ids):
    """Runner rule: score = dot, keep score > 0, sort by (-score, id), top 5."""
    S = Q @ P.T
    out = []
    for row in S:
        cand = [(-float(sc), ids[j]) for j, sc in enumerate(row) if sc > 0.0]
        cand.sort()
        out.append([pid for _, pid in cand[:5]])
    return out


def hit5(qa, tops):
    return float(np.mean([ev.hit_at_k(t, ev.gold_paper_ids(q)) for q, t in zip(qa, tops)]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="2wiki")
    ap.add_argument("--splits", default="sub200")
    ap.add_argument("--no-flag", action="store_true")
    ap.add_argument("--no-st", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "paper" / "results" / "bge_pooling_check.json"))
    args = ap.parse_args(argv)

    # Import FlagEmbedding / sentence-transformers (-> datasets/pyarrow) before torch
    # touches CUDA: the reverse order segfaults in this Windows env.
    if not args.no_flag:
        from FlagEmbedding import BGEM3FlagModel
    if not args.no_st:
        from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).to(device).eval()
    hf = HfEncoder(torch, model, tok, device)
    flag = st = None
    if not args.no_flag:
        flag = BGEM3FlagModel(MODEL, use_fp16=False, device=device)
        print("FlagEmbedding pooling:", flag.model.sentence_pooling_method, flush=True)
    if not args.no_st:
        st = SentenceTransformer(MODEL, device=device)
        print("SentenceTransformer modules:", [type(m).__name__ for m in st], "pooling cls =",
              st[1].get_config_dict().get("pooling_mode_cls_token"), "max_seq_length =", st.max_seq_length, flush=True)

    report = {"generated": datetime.now(timezone.utc).isoformat(), "model": MODEL, "device": device,
              "torch": torch.__version__, "splits": []}
    for b in args.benchmarks.split(","):
        for s in args.splits.split(","):
            d, qa, papers, by_paper, title_by_pid = load(b, s)
            ids = [_record_id(p) for p in papers]
            questions = [str(q.get("question") or q.get("query") or "") for q in qa]
            runner_txt = [_paper_text(p) for p in papers]
            hippo_txt = [f"{title_by_pid.get(pid, p.get('title') or '')}\n{by_paper.get(pid, p.get('abstract') or '')}"
                         for pid, p in zip(ids, papers)]
            ntok = [len(x) for x in tok(runner_txt, add_special_tokens=True)["input_ids"]]
            ntok_h = [len(x) for x in tok(hippo_txt, add_special_tokens=True)["input_ids"]]
            ntok_q = [len(x) for x in tok(questions, add_special_tokens=True)["input_ids"]]
            entry = {"bench": b, "split": s, "n_q": len(qa), "n_pool": len(papers),
                     "tokens_runner_text": {"median": float(np.median(ntok)), "max": int(max(ntok)),
                                            "frac_truncated_512": float(np.mean([n > 512 for n in ntok]))},
                     "tokens_hippo_text": {"median": float(np.median(ntok_h)), "max": int(max(ntok_h)),
                                           "frac_over_512": float(np.mean([n > 512 for n in ntok_h]))},
                     "tokens_query_max": int(max(ntok_q)), "hit5": {}}
            Pm, Pc = hf.encode(runner_txt, 512)
            Qm, Qc = hf.encode(questions, 512, batch_size=1)  # runner encodes each query alone
            tops = {"hf_mean_runnertext": rank_top5(Pm, Qm, ids), "hf_cls_runnertext": rank_top5(Pc, Qc, ids)}
            Hm, Hc = hf.encode(hippo_txt, 8192, batch_size=8)
            tops["hf_mean_hippotext"] = rank_top5(Hm, Qm, ids)
            tops["hf_cls_hippotext"] = rank_top5(Hc, Qc, ids)
            if flag is not None:
                P = np.asarray(flag.encode(runner_txt, batch_size=16, max_length=512)["dense_vecs"], dtype=np.float64)
                Q = np.asarray(flag.encode(questions, batch_size=16, max_length=512)["dense_vecs"], dtype=np.float64)
                tops["flag_cls_runnertext"] = rank_top5(P, Q, ids)
                entry["flag_vs_hf_cls_max_abs_diff"] = float(np.abs(P - Pc).max())
            if st is not None:
                P = np.asarray(st.encode(hippo_txt, batch_size=16, normalize_embeddings=True), dtype=np.float64)
                Q = np.asarray(st.encode(questions, batch_size=16, normalize_embeddings=True), dtype=np.float64)
                tops["st_cls_hippotext"] = rank_top5(P, Q, ids)
                entry["st_vs_hf_cls_hippo_max_abs_diff"] = float(np.abs(P - Hc).max())
            for k, t in tops.items():
                entry["hit5"][k] = hit5(qa, t)
            # reproduction check against the frozen rerun's dense-only records
            frozen = d / "_dense_bge_m3" / "_kgswap_all_kg_silver.json"
            if frozen.exists():
                recs = {r["qa_id"]: r["retrieved_paper_ids"]
                        for r in json.load(open(frozen, encoding="utf-8"))["results"]["dense"]["records"]}
                same = sum(recs.get(ev._qa_id(q)) == t for q, t in zip(qa, tops["hf_mean_runnertext"]))
                same_cls = sum(recs.get(ev._qa_id(q)) == t for q, t in zip(qa, tops["hf_cls_runnertext"]))
                entry["frozen_dense_top5_identical"] = {"hf_mean_runnertext": same, "hf_cls_runnertext": same_cls,
                                                         "n": len(qa)}
            report["splits"].append(entry)
            print(json.dumps(entry, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
