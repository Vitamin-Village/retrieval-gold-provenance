"""LLM-judge correctness for end-to-end QA predictions whose gold is a sentence
(GraphRAG-Bench) or where EM is too strict. A cross-family local judge
(default mistral-small:24b, neither of the readers) grades each unique
(question, gold, prediction) triple once, at temperature 0, and the verdict is
cached. Paired swap statistics (aligned minus independent KG) are recomputed on
judge accuracy exactly as e2e_qa_swap.py does for EM.

Usage (after e2e_qa_swap.py --score-only has written per_question.jsonl):
    python eval/e2e_qa_judge.py --reader llama3.1:8b --benchmarks graphragbench
Outputs: results/e2e_qa/<reader_tag>/judge_<judge_tag>.{json,md}
Cache:   results/e2e_qa/judge_cache_<judge_tag>.jsonl (shared across readers)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_qa_swap import bootstrap_ci, sign_test, ollama_generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e2e_qa"

JUDGE_PROMPT = (
    "You are grading a question-answering system.\n\n"
    "Question: {q}\n"
    "Reference answer: {gold}\n"
    "Candidate answer: {pred}\n\n"
    "Decide whether the candidate answer is correct. It is CORRECT if it states the same key fact "
    "as the reference answer, or names the same entity, value, or conclusion, even if it is shorter "
    "or worded differently. It is INCORRECT if it is empty, contradicts the reference, names a "
    "different entity or value, or does not address the question.\n"
    "Reply with exactly one word: CORRECT or INCORRECT."
)


def tag(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--judge", default="mistral-small:24b")
    ap.add_argument("--benchmarks", default="graphragbench")
    ap.add_argument("--res", default=str(RES), help="results root holding <reader_tag>/per_question.jsonl and the judge cache")
    args = ap.parse_args()
    res = Path(args.res)
    rdir = res / tag(args.reader)
    benches = set(args.benchmarks.split(","))
    rows = [json.loads(l) for l in open(rdir / "per_question.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["bench"] in benches]
    if not rows:
        sys.exit("no rows for benchmarks %s in %s" % (benches, rdir))
    # question text is needed for the judge: pull from the split qa files
    qtext = {}
    for b in benches:
        for sp in ("sub200", "holdout200"):
            p = ROOT / "external" / b / sp / "qa.jsonl"
            if p.exists():
                for l in open(p, encoding="utf-8"):
                    q = json.loads(l)
                    qtext[(b, q["qa_id"])] = q["question"]

    cache_path = res / f"judge_cache_{tag(args.judge)}.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            cache[r["key"]] = r["verdict"]
    uniq = {}
    for r in rows:
        gold = r["golds"][0] if r["golds"] else ""
        key = hashlib.sha1("\x00".join([args.judge, r["bench"], r["qa_id"], gold, r["pred"]]).encode("utf-8")).hexdigest()
        r["_key"] = key
        if r["pred"].strip() == "":
            cache.setdefault(key, "INCORRECT")
            continue
        uniq.setdefault(key, (qtext.get((r["bench"], r["qa_id"]), ""), gold, r["pred"]))
    todo = [k for k in uniq if k not in cache]
    print(f"[judge {args.judge}] rows={len(rows)} unique={len(uniq)} todo={len(todo)}", flush=True)
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as fh:
        for i, k in enumerate(todo, 1):
            q, gold, pred = uniq[k]
            resp, _, _ = ollama_generate(args.judge, JUDGE_PROMPT.format(q=q, gold=gold, pred=pred), num_predict=4)
            v = "CORRECT" if re.match(r"\s*correct", resp, flags=re.I) else "INCORRECT"
            fh.write(json.dumps({"key": k, "judge": args.judge, "verdict": v, "raw": resp.strip()[:40]}) + "\n")
            fh.flush()
            cache[k] = v
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} {el / i:.2f}s/call eta {(len(todo) - i) * el / i / 60:.1f} min", flush=True)

    for r in rows:
        r["judge"] = 1.0 if cache.get(r["_key"]) == "CORRECT" else 0.0
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["bench"], r["split"], r["system"], r["condition"])].append(r["judge"])
    out = {"reader": args.reader, "judge": args.judge, "cells": [], "paired": []}
    for k, v in sorted(cells.items()):
        out["cells"].append({"bench": k[0], "split": k[1], "system": k[2], "condition": k[3], "n": len(v), "judge_acc": float(np.mean(v))})
    by_q = collections.defaultdict(dict)
    for r in rows:
        by_q[(r["bench"], r["split"], r["system"], r["qa_id"])][r["condition"]] = r
    groups = collections.defaultdict(list)
    for (b, sp, sy, qid), conds in by_q.items():
        for pre in ("", "bge_"):
            if pre + "kg_silver" in conds and pre + "kg_openie" in conds:
                label = sy if pre == "" else f"{sy}@{pre.rstrip('_')}"
                groups[(b, sp, label)].append((conds[pre + "kg_silver"], conds[pre + "kg_openie"]))
    for (b, sp, sy), pairs in sorted(groups.items()):
        diff = np.array([s["judge"] - o["judge"] for s, o in pairs])
        pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
        rh = np.array([s["retrieval_hit"] - o["retrieval_hit"] for s, o in pairs])
        out["paired"].append({"bench": b, "split": sp, "system": sy, "n": len(pairs),
                              "silver": float(np.mean([s["judge"] for s, _ in pairs])), "openie": float(np.mean([o["judge"] for _, o in pairs])),
                              "mean_paired_diff": float(diff.mean()), "ci95": bootstrap_ci(diff), "discordant_pos": pos, "discordant_neg": neg,
                              "sign_test_p": sign_test(pos, neg), "retrieval_hit_diff": float(rh.mean()),
                              "diff_on_flips_pos": float(diff[rh > 0].mean()) if (rh > 0).any() else None, "n_flips_pos": int((rh > 0).sum()),
                              "diff_on_no_flip": float(diff[rh == 0].mean()) if (rh == 0).any() else None, "n_no_flip": int((rh == 0).sum())})
    jt = tag(args.judge)
    json.dump(out, open(rdir / f"judge_{jt}.json", "w", encoding="utf-8"), indent=1)
    md = [f"# Judge accuracy (judge `{args.judge}`) for reader `{args.reader}`", "",
          "| bench | split | system | n | hit@5 Δ | acc aligned | acc indep. | Δ [95% CI] | p (sign) | Δ on flips+ (n) | Δ on no-flip (n) |",
          "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for e in out["paired"]:
        f = lambda x: "-" if x is None else f"{x:+.3f}"
        md.append(f"| {e['bench']} | {e['split']} | {e['system']} | {e['n']} | {e['retrieval_hit_diff']:+.3f} | {e['silver']:.3f} | {e['openie']:.3f} | "
                  f"{e['mean_paired_diff']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}] | {e['sign_test_p']:.2e} | {f(e['diff_on_flips_pos'])} ({e['n_flips_pos']}) | {f(e['diff_on_no_flip'])} ({e['n_no_flip']}) |")
    md += ["", "| bench | split | system | condition | n | judge acc |", "|---|---|---|---|---:|---:|"]
    for c in out["cells"]:
        md.append(f"| {c['bench']} | {c['split']} | {c['system']} | {c['condition']} | {c['n']} | {c['judge_acc']:.3f} |")
    open(rdir / f"judge_{jt}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
