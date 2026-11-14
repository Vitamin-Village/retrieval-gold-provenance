"""Training-exposure (memorization) probe for the end-to-end KG swap.

Could the swap effect come from the evaluated system having seen the gold structure during
training? Two measurements, both per reader:

1. Evidence-triple recall (2WikiMultiHop). Every gold evidence triple (s, r, o) of the
   400 audited questions is asked closed-book as "What is the <r> of <s>?" and scored
   against o and its Wikidata aliases. A question is *evidence-known* when the reader
   recalls every evidence triple, i.e. when it could have reconstructed the gold chain
   from parametric memory of the KG.
2. Closed-book answer (all four benchmarks), already produced by e2e_qa_swap.py.

The paired answer-level swap (aligned KG minus independent KG) is then stratified by
both properties. If the swap persists where the reader neither knows the answer nor the
evidence triples, training exposure cannot be what carries it.

Usage (from paper/):
    python eval/e2e_memorization_probe.py --model qwen2.5:14b            # probe + analysis
    python eval/e2e_memorization_probe.py --model llama3.1:8b --analyze-only
Outputs: results/e2e_qa/<model_tag>/memorization_2wiki.jsonl, memorization_strata.{json,md}
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
from e2e_qa_swap import (EXT, ROOT, bootstrap_ci, clean_prediction, normalize,  # noqa: E402
                         ollama_generate, sign_test)

PROBE = (
    "Answer from your own knowledge. Reply with the short answer only "
    "(an entity name, date, number, or short phrase) and no explanation.\n\n"
    "Question: What is the {r} of {s}?\nAnswer:"
)
JUDGE, JUDGE_TAG = "mistral-small:24b", "mistral_small_24b"
SPLITS = ("sub200", "holdout200")
SYSTEMS = ("audited", "graph_centric", "ppr")


def load_triples():
    dev = {r["_id"]: r for r in json.load(open(EXT / "2wiki/raw/data_ids/dev.json", encoding="utf-8"))}
    aliases = {}
    for l in open(EXT / "2wiki/raw/data_ids/id_aliases.json", encoding="utf-8"):
        r = json.loads(l)
        aliases[r["Q_id"]] = r.get("aliases", []) + r.get("demonyms", [])
    per_q = {}
    for split in SPLITS:
        for l in open(EXT / "2wiki" / split / "qa.jsonl", encoding="utf-8"):
            q = json.loads(l)
            rec = dev[q["qa_id"]]
            ev, ev_id = rec["evidences"], rec.get("evidences_id") or []
            trip = []
            for i, (s, r, o) in enumerate(ev):
                gold = [o]
                if i < len(ev_id) and len(ev_id[i]) == 3:
                    gold += aliases.get(ev_id[i][2], [])
                trip.append({"s": s, "r": r, "o": o, "golds": sorted(set(g for g in gold if g))})
            per_q[(split, q["qa_id"])] = trip
    return per_q


def recalled(pred: str, golds) -> bool:
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
        ng = normalize(g)
        if not ng:
            continue
        if p == ng or ng in p or (len(p) >= 4 and p in ng):
            return True
    return False


def probe(model: str, per_q, out_dir: Path):
    path = out_dir / "memorization_2wiki.jsonl"
    done = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            done[r["key"]] = r
    uniq = {}
    for trips in per_q.values():
        for t in trips:
            prompt = PROBE.format(r=t["r"], s=t["s"])
            key = hashlib.sha1((model + "\x00" + prompt).encode("utf-8")).hexdigest()
            uniq.setdefault(key, (prompt, t))
    todo = [k for k in uniq if k not in done]
    print(f"[{model}] unique triples={len(uniq)} cached={len(uniq) - len(todo)} todo={len(todo)}", flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fh:
        for i, k in enumerate(todo, 1):
            prompt, t = uniq[k]
            resp, _, _ = ollama_generate(model, prompt)
            pred = clean_prediction(resp)
            row = {"key": k, "model": model, "s": t["s"], "r": t["r"], "o": t["o"], "golds": t["golds"],
                   "pred": pred, "recalled": recalled(pred, t["golds"])}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[k] = row
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} {el / i:.2f}s/call", flush=True)
    return {k: v for k, v in done.items()}


def paired(rows_a, rows_b, ids):
    d = np.array([rows_a[i]["em"] - rows_b[i]["em"] for i in ids], dtype=float)
    if len(d) == 0:
        return {"n": 0}
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return {"n": len(d), "diff": float(d.mean()), "ci": bootstrap_ci(d) if len(d) > 1 else [float("nan")] * 2,
            "pos": pos, "neg": neg, "p": sign_test(pos, neg)}


def analyze(model: str, per_q, probe_rows, out_dir: Path):
    # GraphRAG-Bench gold answers are sentences: use the primary judge's cached verdict
    # (same key as eval/e2e_qa_judge.py) in place of EM.
    judge = {}
    jpath = ROOT / "results" / "e2e_qa" / f"judge_cache_{JUDGE_TAG}.jsonl"
    for l in open(jpath, encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        judge[r["key"]] = r["verdict"]
    em = collections.defaultdict(dict)  # (bench, split, cond, system) -> qa_id -> row
    for l in open(out_dir / "per_question.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["bench"] == "graphragbench":
            gold = r["golds"][0] if r["golds"] else ""
            key = hashlib.sha1("\x00".join([JUDGE, r["bench"], r["qa_id"], gold, r["pred"]]).encode("utf-8")).hexdigest()
            r["em"] = 1.0 if (r["pred"].strip() and judge.get(key) == "CORRECT") else 0.0
        em[(r["bench"], r["split"], r["condition"], r["system"])][r["qa_id"]] = r
    by_prompt = {}
    for r in probe_rows.values():
        by_prompt[(r["s"], r["r"])] = r["recalled"]

    out = {"model": model, "triple_recall": None, "strata": []}
    all_trips = [by_prompt[(t["s"], t["r"])] for trips in per_q.values() for t in trips]
    out["triple_recall"] = float(np.mean(all_trips))
    lines = [f"# Training-exposure strata: reader `{model}`", "",
             f"Evidence-triple closed-book recall on 2WikiMultiHop: {out['triple_recall']:.3f} "
             f"over {len(all_trips)} triple instances.", "",
             "| bench | split | system | stratum | n | EM swap | 95% CI | +/- | p |", "|---|---|---|---|---:|---:|---|---|---:|"]

    def add(bench, split, system, name, ids, a, b):
        s = paired(a, b, ids)
        s.update({"bench": bench, "split": split, "system": system, "stratum": name})
        out["strata"].append(s)
        if s["n"]:
            lines.append(f"| {bench} | {split} | {system} | {name} | {s['n']} | {s['diff']:+.3f} | "
                         f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['pos']}/{s['neg']} | {s['p']:.2g} |")

    for bench in ("2wiki", "metaqa", "stark", "graphragbench"):
        for system in SYSTEMS:
            pooled = collections.defaultdict(list)
            for split in SPLITS:
                a = em.get((bench, split, "kg_silver", system), {})
                b = em.get((bench, split, "kg_openie", system), {})
                cb = em.get((bench, split, "closedbook", "none"), {})
                ids = [i for i in a if i in b and i in cb]
                if not ids:
                    continue
                strata = {"all": ids,
                          "closed-book wrong": [i for i in ids if cb[i]["em"] == 0],
                          "closed-book right": [i for i in ids if cb[i]["em"] == 1]}
                if bench == "2wiki":
                    known = {i: all(by_prompt[(t["s"], t["r"])] for t in per_q[(split, i)]) for i in ids}
                    strata["evidence not recalled"] = [i for i in ids if not known[i]]
                    strata["evidence recalled"] = [i for i in ids if known[i]]
                    strata["neither (cb wrong, evidence not recalled)"] = [
                        i for i in ids if cb[i]["em"] == 0 and not known[i]]
                    none_known = {i: not any(by_prompt[(t["s"], t["r"])] for t in per_q[(split, i)]) for i in ids}
                    strata["strict (cb wrong, no evidence triple recalled)"] = [
                        i for i in ids if cb[i]["em"] == 0 and none_known[i]]
                for name, sids in strata.items():
                    add(bench, split, system, name, sids, a, b)
                    pooled[name].append((split, sids, a, b))
            for name, parts in pooled.items():
                ids, A, B = [], {}, {}
                for split, sids, a, b in parts:
                    for i in sids:
                        ids.append((split, i)); A[(split, i)] = a[i]; B[(split, i)] = b[i]
                add(bench, "pooled", system, name, ids, A, B)
    (out_dir / "memorization_strata.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (out_dir / "memorization_strata.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    tag = re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    out_dir = ROOT / "results" / "e2e_qa" / tag
    per_q = load_triples()
    if args.analyze_only:
        rows = {json.loads(l)["key"]: json.loads(l) for l in open(out_dir / "memorization_2wiki.jsonl", encoding="utf-8")}
    else:
        rows = probe(args.model, per_q, out_dir)
    analyze(args.model, per_q, rows, out_dir)


if __name__ == "__main__":
    main()
