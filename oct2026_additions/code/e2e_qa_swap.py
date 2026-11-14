"""End-to-end QA over the frozen KG-swap retrieval vectors.

For every stored retrieval condition (``_kgswap_*.json``, ``_dose_work/eval_f*.json``)
the top-5 passage ids per question are turned into a fixed reader prompt, answered by a
local Ollama model at temperature 0, and scored (EM / token-F1 / cover-EM) against the
benchmark gold answers. Two synthetic conditions bound the effect:

* ``closedbook`` -- no passages (measures the reader's parametric knowledge, i.e. the
  system-side training-exposure quantity that is held constant across swap arms);
* ``oracle``     -- the gold passages only (reader ceiling given perfect retrieval).

Everything except the KG file is identical across the aligned (silver) and independent
(openie) arms, so the paired difference in answer accuracy is attributable to provenance
alignment, exactly as for hit@5.

Usage (from paper/):
    python eval/e2e_qa_swap.py --model llama3.1:8b
    python eval/e2e_qa_swap.py --model qwen2.5:14b --benchmarks 2wiki
    python eval/e2e_qa_swap.py --model llama3.1:8b --score-only

Outputs (under results/e2e_qa/<model_tag>/):
    cache.jsonl          append-only generation cache (resumable)
    per_question.jsonl   one row per (bench, split, condition, system, qa_id)
    summary.json         cell means + paired swap statistics
    summary.md           human-readable tables
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import hashlib
import json
import math
import os
import re
import string
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # paper/
EXT = ROOT / "external"
OLLAMA = "http://127.0.0.1:11434/api/generate"  # not "localhost": on Windows urllib tries ::1 first and waits ~2 s per call
BOOT_B = 10000
BOOT_SEED = 20260918

SYSTEM_ALIASES = {
    "proposed_v3": "audited",
    "ppr": "ppr",
    "proposed_v5_graphcentric": "graph_centric",
    "bm25": "bm25",
    "dense": "dense",
    "hipporag2": "hipporag2",  # HippoRAG 2 with injected KGs (eval/_kg_swap_hipporag2.py)
}
PAIR_PREFIXES = ("", "bge_")  # frozen surrogate runs, BGE-M3 dense-channel runs
# Which BGE-M3 rerun backs the bge_* conditions: "" = _dense_bge_m3/_kgswap_all_kg_*.json (mean
# pooling, 2026-09-20); "cls" = _dense_bge_m3_cls/_kgswap_all_kg_*_cls.json ([CLS] pooling). Condition
# names stay bge_kg_silver / bge_kg_openie either way. Set by --bge-variant.
BGE_VARIANT = ""

PROMPT = (
    "Answer the question using the passages. Reply with the short answer only "
    "(an entity name, date, number, or short phrase) and no explanation. "
    "If several entities answer the question, give one of them.\n\n"
    "Passages:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)
PROMPT_CLOSED = (
    "Answer the question from your own knowledge. Reply with the short answer only "
    "(an entity name, date, number, or short phrase) and no explanation. "
    "If several entities answer the question, give one of them.\n\n"
    "Question: {q}\nAnswer:"
)


# ----------------------------------------------------------------------------- data
def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return list(ast.literal_eval(v))
        except Exception:
            return [v]
    return [v]


def load_split(bench: str, split: str):
    d = EXT / bench / split
    qa = [json.loads(l) for l in open(d / "qa.jsonl", encoding="utf-8")]
    passages = {}
    for l in open(d / "passages.jsonl", encoding="utf-8"):
        p = json.loads(l)
        passages[p["passage_id"]] = p
    golds = {}
    if bench == "2wiki":
        dev = {r["_id"]: r for r in json.load(open(EXT / "2wiki/raw/data_ids/dev.json", encoding="utf-8"))}
        aliases = {}
        for l in open(EXT / "2wiki/raw/data_ids/id_aliases.json", encoding="utf-8"):
            r = json.loads(l)
            aliases[r["Q_id"]] = r.get("aliases", []) + r.get("demonyms", [])
        for q in qa:
            g = [q["answer"]]
            rec = dev.get(q["qa_id"])
            if rec and rec.get("answer_id"):
                g += aliases.get(rec["answer_id"], [])
            golds[q["qa_id"]] = sorted(set(x for x in g if x))
    else:
        for q in qa:
            g = _as_list(q.get("answers", [])) or [q["answer"]]
            golds[q["qa_id"]] = sorted(set(str(x) for x in g if x))
    for q in qa:
        q["gold_paper_ids"] = _as_list(q["gold_paper_ids"])
    return qa, passages, golds


def condition_files(bench: str, split: str):
    """Yield (condition_name, path). Condition names: kg_silver, kg_openie,
    kg_openie_svocab*, dose_f25 ..."""
    d = EXT / bench / split
    out = []
    for f in sorted(glob.glob(str(d / "_kgswap_*.json"))):
        name = os.path.basename(f)[len("_kgswap_"):-len(".json")]
        for sysname in ("ppr_", "proposed_v5_graphcentric_", "proposed_v3_", "hipporag2_"):
            if name.startswith(sysname):
                name = name[len(sysname):]
        out.append((name, f))
    for f in sorted(glob.glob(str(d / "_dose_work" / "eval_f*.json"))):
        frac = os.path.basename(f)[len("eval_"):-len(".json")]
        out.append((f"dose_{frac}", f))
    # BGE-M3 dense-channel rerun (eval/_kg_swap_dense_bge.py): systems bm25, dense,
    # proposed_v3, proposed_v5_graphcentric under each KG condition
    sfx = f"_{BGE_VARIANT}" if BGE_VARIANT else ""
    for f in sorted(glob.glob(str(d / f"_dense_bge_m3{sfx}" / f"_kgswap_all_kg_*{sfx}.json"))):
        kg = os.path.basename(f)[len("_kgswap_all_"):-len(f"{sfx}.json")]
        out.append((f"bge_{kg}", f))
    return out


# ----------------------------------------------------------------------------- reader
def ollama_generate(model: str, prompt: str, num_predict: int = 48, retries: int = 3):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "60m",
        "options": {"temperature": 0, "seed": 1, "num_predict": num_predict},
    }).encode()
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.load(r)
            return resp["response"], resp.get("prompt_eval_count"), resp.get("eval_count")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5)
    raise RuntimeError(f"ollama failed: {last}")


def build_prompt(question: str, pids, passages):
    if not pids:
        return PROMPT_CLOSED.format(q=question)
    ctx = "\n\n".join(
        f"[{i + 1}] {passages[p]['title']}: {passages[p]['text']}" for i, p in enumerate(pids) if p in passages
    )
    return PROMPT.format(ctx=ctx, q=question)


def clean_prediction(text: str) -> str:
    t = text.strip().split("\n")[0].strip()
    t = re.sub(r"^(answer\s*:\s*)", "", t, flags=re.I)
    t = t.strip().strip('"').strip("'").rstrip(".").strip()
    return t


# ----------------------------------------------------------------------------- scoring
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def score(pred: str, golds) -> dict:
    np_ = normalize(pred)
    ng = [normalize(g) for g in golds]
    em = float(any(np_ == g for g in ng))
    f1 = max(token_f1(pred, g) for g in golds) if golds else 0.0
    cover = float(any(g and g in np_ for g in ng))
    return {"em": em, "f1": f1, "cover": cover}


# ----------------------------------------------------------------------------- stats
def bootstrap_ci(diff: np.ndarray, b: int = BOOT_B, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(b, n))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_test(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--benchmarks", default="2wiki,metaqa,stark")
    ap.add_argument("--splits", default="sub200,holdout200")
    ap.add_argument("--out", default=str(ROOT / "results" / "e2e_qa"))
    ap.add_argument("--score-only", action="store_true", help="do not call the model; score cached rows only")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N questions per split")
    ap.add_argument("--bge-variant", default="", choices=("", "cls"),
                    help="BGE-M3 rerun backing the bge_* conditions ('' = mean-pooled 2026-09-20 run, 'cls' = [CLS] rerun)")
    args = ap.parse_args()
    global BGE_VARIANT
    BGE_VARIANT = args.bge_variant

    tag = re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "cache.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:  # a line still being written by a concurrent run
                continue
            cache[r["key"]] = r

    # ---- build job list -------------------------------------------------------
    jobs = []  # (bench, split, cond, system, qa, pids, retrieval_hit)
    for bench in args.benchmarks.split(","):
        for split in args.splits.split(","):
            qa, passages, golds = load_split(bench, split)
            if args.limit:
                qa = qa[: args.limit]
            qmap = {q["qa_id"]: q for q in qa}
            for q in qa:
                q["_passages"] = passages
                q["_golds"] = golds[q["qa_id"]]
            seen_bm25 = False
            for cond, f in condition_files(bench, split):
                d = json.load(open(f, encoding="utf-8"))
                for system, block in d["results"].items():
                    if system == "bm25":
                        if seen_bm25:
                            continue
                        seen_bm25 = True
                        cname = "none"
                    elif system == "dense" and cond.endswith("kg_openie"):
                        continue  # dense-only retrieval is KG-independent; keep one copy (under kg_silver)
                    else:
                        cname = cond
                    for rec in block["records"]:
                        if rec["qa_id"] not in qmap:
                            continue
                        jobs.append((bench, split, cname, system, qmap[rec["qa_id"]],
                                     tuple(rec["retrieved_paper_ids"]), float(rec["metrics"]["hit_at_k"])))
            for q in qa:
                jobs.append((bench, split, "closedbook", "none", q, tuple(), float("nan")))
                jobs.append((bench, split, "oracle", "none", q, tuple(q["gold_paper_ids"]), 1.0))

    # ---- generation with cache ------------------------------------------------
    uniq = {}
    for j in jobs:
        q = j[4]
        prompt = build_prompt(q["question"], j[5], q["_passages"])
        key = hashlib.sha1((args.model + "\x00" + prompt).encode("utf-8")).hexdigest()
        uniq.setdefault(key, (prompt, j))
    todo = [k for k in uniq if k not in cache]
    print(f"[{args.model}] jobs={len(jobs)} unique prompts={len(uniq)} cached={len(uniq) - len(todo)} todo={len(todo)}", flush=True)
    if todo and not args.score_only:
        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as fh:
            for i, k in enumerate(todo, 1):
                prompt, j = uniq[k]
                resp, pc, ec = ollama_generate(args.model, prompt)
                row = {"key": k, "model": args.model, "bench": j[0], "split": j[1], "qa_id": j[4]["qa_id"],
                       "pids": list(j[5]), "response": resp, "prompt_tokens": pc, "gen_tokens": ec}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                cache[k] = row
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {el / i:.2f}s/call  eta {(len(todo) - i) * el / i / 60:.1f} min", flush=True)

    # ---- score ---------------------------------------------------------------
    rows = []
    for j in jobs:
        bench, split, cond, system, q, pids, rhit = j
        prompt = build_prompt(q["question"], pids, q["_passages"])
        key = hashlib.sha1((args.model + "\x00" + prompt).encode("utf-8")).hexdigest()
        if key not in cache:
            continue
        pred = clean_prediction(cache[key]["response"])
        sc = score(pred, q["_golds"])
        rows.append({"bench": bench, "split": split, "condition": cond, "system": SYSTEM_ALIASES.get(system, system),
                     "qa_id": q["qa_id"], "pred": pred, "golds": q["_golds"], "retrieval_hit": rhit, **sc})
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # cell means
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["bench"], r["split"], r["system"], r["condition"])].append(r)
    summary = {"model": args.model, "n_rows": len(rows), "cells": [], "paired": []}
    for (bench, split, system, cond), rs in sorted(cells.items()):
        summary["cells"].append({
            "bench": bench, "split": split, "system": system, "condition": cond, "n": len(rs),
            "em": float(np.mean([r["em"] for r in rs])), "f1": float(np.mean([r["f1"] for r in rs])),
            "cover": float(np.mean([r["cover"] for r in rs])),
            "hit5": float(np.nanmean([r["retrieval_hit"] for r in rs])) if cond not in ("closedbook",) else None,
        })

    # paired swap: silver - openie per (bench, split, system)
    by_q = collections.defaultdict(dict)
    for r in rows:
        by_q[(r["bench"], r["split"], r["system"], r["qa_id"])][r["condition"]] = r
    groups = collections.defaultdict(list)
    for (bench, split, system, qid), conds in by_q.items():
        for pre in PAIR_PREFIXES:
            if pre + "kg_silver" in conds and pre + "kg_openie" in conds:
                label = system if pre == "" else f"{system}@{pre.rstrip('_')}"
                groups[(bench, split, label)].append((conds[pre + "kg_silver"], conds[pre + "kg_openie"]))
    for (bench, split, system), pairs in sorted(groups.items()):
        entry = {"bench": bench, "split": split, "system": system, "n": len(pairs)}
        for m in ("em", "f1", "cover"):
            diff = np.array([s[m] - o[m] for s, o in pairs])
            pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
            entry[m] = {"silver": float(np.mean([s[m] for s, _ in pairs])), "openie": float(np.mean([o[m] for _, o in pairs])),
                        "mean_paired_diff": float(diff.mean()), "ci95": bootstrap_ci(diff),
                        "discordant_pos": pos, "discordant_neg": neg, "sign_test_p": sign_test(pos, neg)}
        # mediation through retrieval flips
        rh = np.array([s["retrieval_hit"] - o["retrieval_hit"] for s, o in pairs])
        em = np.array([s["em"] - o["em"] for s, o in pairs])
        entry["mediation"] = {
            "retrieval_hit_diff": float(rh.mean()),
            "em_diff_on_retrieval_flips_pos": float(em[rh > 0].mean()) if (rh > 0).any() else None,
            "n_flips_pos": int((rh > 0).sum()),
            "em_diff_on_retrieval_flips_neg": float(em[rh < 0].mean()) if (rh < 0).any() else None,
            "n_flips_neg": int((rh < 0).sum()),
            "em_diff_on_no_flip": float(em[rh == 0].mean()) if (rh == 0).any() else None,
            "n_no_flip": int((rh == 0).sum()),
        }
        summary["paired"].append(entry)

    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # markdown
    md = [f"# End-to-end QA over frozen swap vectors: reader `{args.model}`", ""]
    md.append("## Paired swap (aligned KG minus independent KG), same questions / gold / passages / reader")
    md.append("")
    md.append("| bench | split | system | n | hit@5 Δ | EM silver | EM openie | EM Δ [95% CI] | p (sign) | F1 Δ [95% CI] | cover Δ |")
    md.append("|---|---|---|---:|---:|---:|---:|---|---:|---|---:|")
    for e in summary["paired"]:
        em, f1, cv = e["em"], e["f1"], e["cover"]
        md.append(f"| {e['bench']} | {e['split']} | {e['system']} | {e['n']} | {e['mediation']['retrieval_hit_diff']:+.3f} | "
                  f"{em['silver']:.3f} | {em['openie']:.3f} | {em['mean_paired_diff']:+.3f} [{em['ci95'][0]:+.3f}, {em['ci95'][1]:+.3f}] | "
                  f"{em['sign_test_p']:.2e} | {f1['mean_paired_diff']:+.3f} [{f1['ci95'][0]:+.3f}, {f1['ci95'][1]:+.3f}] | {cv['mean_paired_diff']:+.3f} |")
    md.append("")
    md.append("## Mediation: EM Δ conditional on retrieval flip status")
    md.append("")
    md.append("| bench | split | system | flips + (n) | EM Δ on flips + | no flip (n) | EM Δ on no flip |")
    md.append("|---|---|---|---:|---:|---:|---:|")
    for e in summary["paired"]:
        m = e["mediation"]
        f = lambda x: "-" if x is None else f"{x:+.3f}"
        md.append(f"| {e['bench']} | {e['split']} | {e['system']} | {m['n_flips_pos']} | {f(m['em_diff_on_retrieval_flips_pos'])} | {m['n_no_flip']} | {f(m['em_diff_on_no_flip'])} |")
    md.append("")
    md.append("## All cells")
    md.append("")
    md.append("| bench | split | system | condition | n | hit@5 | EM | F1 | cover |")
    md.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for c in summary["cells"]:
        h = "-" if c["hit5"] is None or (isinstance(c["hit5"], float) and math.isnan(c["hit5"])) else f"{c['hit5']:.3f}"
        md.append(f"| {c['bench']} | {c['split']} | {c['system']} | {c['condition']} | {c['n']} | {h} | {c['em']:.3f} | {c['f1']:.3f} | {c['cover']:.3f} |")
    open(out / "summary.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}  rows={len(rows)}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
