"""Proprietary reader (Claude Sonnet 5.5) on the core end-to-end conditions of 2WikiMultiHop.

Protocol: results/e2e_qa/claude_sonnet_5_5/PROTOCOL.md (fixed before the first request).
Same conditions as eval/_e2e_third_reader.py (aligned and independent KG vectors of every retriever, BM25,
closed-book, gold-only; both 2WikiMultiHop splits), same prompt, scoring, and statistics as
eval/e2e_qa_swap.py. Generation goes through the Anthropic Message Batches API instead of Ollama; replies are
written to the reader cache in eval/e2e_qa_swap.py's format, and scoring is eval/e2e_qa_swap.py --score-only.

Usage (from paper/, Python with anthropic; key from ../.env):
  python eval/_e2e_claude_reader.py --plan              list the unique prompts and the cost estimate (no API call)
  python eval/_e2e_claude_reader.py --pilot 3           send 3 single requests to check the parameters
  python eval/_e2e_claude_reader.py --submit            submit one batch with every uncached prompt
  python eval/_e2e_claude_reader.py --collect           poll the batch until it ends and write the cache
  python eval/_e2e_claude_reader.py --score             apply the protocol's truncation rule to the cache, then score
                                                        with eval/e2e_qa_swap.py --score-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_qa_swap as E  # noqa: E402

MODEL = "claude-sonnet-5-5"
OUT = E.ROOT / "results" / "e2e_qa" / "claude_sonnet_5_5"
CACHE = OUT / "cache.jsonl"
STATE = OUT / "batch_state.json"
RAW = OUT / "batch_results_raw.jsonl"
PILOT = OUT / "pilot_raw.jsonl"
PARAMS = {"max_tokens": 256, "thinking": {"type": "between_tools"}}  # protocol: no temperature, no tools, no fallbacks
PRICE_IN, PRICE_OUT, BATCH_DISCOUNT = 2.0, 10.0, 0.5  # USD per 1M tokens (claude-sonnet-5-5), batch 50%

_orig = E.condition_files


def core_conditions(bench, split):
    return [(n, f) for n, f in _orig(bench, split) if n in ("kg_silver", "kg_openie")]


E.condition_files = core_conditions


def client():
    import anthropic
    env = (E.ROOT.parent / ".env").read_text(encoding="utf-8")
    key = re.search(r"^ANTHROPIC_API_KEY\s*=\s*['\"]?([^'\"\s]+)", env, re.M).group(1)
    return anthropic.Anthropic(api_key=key, max_retries=4)


def unique_prompts():
    """Rebuild the third reader's job list (eval/e2e_qa_swap.py main, 2WikiMultiHop) and its unique prompts."""
    jobs = []
    for split in ("sub200", "holdout200"):
        qa, passages, golds = E.load_split("2wiki", split)
        qmap = {q["qa_id"]: q for q in qa}
        seen_bm25 = False
        for cond, f in E.condition_files("2wiki", split):
            d = json.load(open(f, encoding="utf-8"))
            for system, block in d["results"].items():
                if system == "bm25":
                    if seen_bm25:
                        continue
                    seen_bm25 = True
                elif system == "dense" and cond.endswith("kg_openie"):
                    continue
                for rec in block["records"]:
                    if rec["qa_id"] in qmap:
                        jobs.append(("2wiki", split, qmap[rec["qa_id"]], tuple(rec["retrieved_paper_ids"]), passages))
        for q in qa:
            jobs.append(("2wiki", split, q, tuple(), passages))
            jobs.append(("2wiki", split, q, tuple(q["gold_paper_ids"]), passages))
    uniq = {}
    for bench, split, q, pids, passages in jobs:
        prompt = E.build_prompt(q["question"], pids, passages)
        key = hashlib.sha1((MODEL + "\x00" + prompt).encode("utf-8")).hexdigest()
        uniq.setdefault(key, {"prompt": prompt, "bench": bench, "split": split, "qa_id": q["qa_id"], "pids": list(pids)})
    return jobs, uniq


def cached_keys():
    if not CACHE.exists():
        return set()
    return {json.loads(l)["key"] for l in open(CACHE, encoding="utf-8") if l.strip()}


def request_params(prompt: str) -> dict:
    return {"model": MODEL, **PARAMS, "messages": [{"role": "user", "content": prompt}]}


def text_of(msg) -> str:
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def cache_row(key, meta, msg, source):
    u = msg.usage
    return {"key": key, "model": MODEL, "bench": meta["bench"], "split": meta["split"], "qa_id": meta["qa_id"],
            "pids": meta["pids"], "response": text_of(msg), "prompt_tokens": u.input_tokens, "gen_tokens": u.output_tokens,
            "model_returned": msg.model, "stop_reason": msg.stop_reason, "message_id": msg.id, "source": source}


def apply_truncation_rule() -> None:
    """Protocol: a truncated reply (stop_reason max_tokens) is scored as an empty answer. The cache row keeps the reply
    in response_raw and gets response = "" (idempotent; batch_results_raw.jsonl keeps every result object unchanged)."""
    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    n = 0
    for r in rows:
        if r.get("stop_reason") == "max_tokens" and "response_raw" not in r:
            r["response_raw"], r["response"], r["scored_as_empty"] = r["response"], "", "max_tokens"
        n += r.get("scored_as_empty") == "max_tokens"
    with open(CACHE, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"truncation rule: {n} of {len(rows)} replies stopped at max_tokens and are scored as empty answers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs, uniq = unique_prompts()
    have = cached_keys()
    todo = [k for k in uniq if k not in have]
    if args.plan or not (args.pilot or args.submit or args.collect or args.score):
        c = client() if args.plan else None
        n_tok = None
        if c is not None:  # count input tokens exactly on a sample, then scale
            sample = list(uniq)[:: max(1, len(uniq) // 40)][:40]
            n_tok = [c.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": uniq[k]["prompt"]}]).input_tokens
                     for k in sample]
        mean_in = (sum(n_tok) / len(n_tok)) if n_tok else None
        est = None if mean_in is None else len(todo) * (mean_in * PRICE_IN + 16 * PRICE_OUT) / 1e6 * BATCH_DISCOUNT
        print(json.dumps({"jobs": len(jobs), "unique_prompts": len(uniq), "cached": len(uniq) - len(todo), "todo": len(todo),
                          "mean_input_tokens_sample": mean_in, "estimated_batch_cost_usd": est}, indent=1))
    if args.pilot:
        c = client()
        keys = [k for k in todo][: args.pilot]
        with open(PILOT, "a", encoding="utf-8") as fh:
            for k in keys:
                msg = c.messages.create(**request_params(uniq[k]["prompt"]))
                row = cache_row(k, uniq[k], msg, "pilot")
                row["utc"] = datetime.now(timezone.utc).isoformat()
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(row["stop_reason"], row["model_returned"], row["prompt_tokens"], row["gen_tokens"], repr(row["response"][:80]))
        print("pilot replies are logged in pilot_raw.jsonl and are NOT written to the cache")
    if args.submit:
        if STATE.exists() and json.load(open(STATE, encoding="utf-8")).get("status") != "collected":
            sys.exit(f"a batch is already pending: {STATE}")
        c = client()
        reqs = [{"custom_id": k, "params": request_params(uniq[k]["prompt"])} for k in todo]
        batch = c.messages.batches.create(requests=reqs)
        json.dump({"batch_id": batch.id, "n_requests": len(reqs), "submitted_utc": datetime.now(timezone.utc).isoformat(),
                   "params": PARAMS, "model": MODEL, "status": "submitted"}, open(STATE, "w", encoding="utf-8"), indent=1)
        print(f"submitted batch {batch.id} with {len(reqs)} requests")
    if args.collect:
        c = client()
        st = json.load(open(STATE, encoding="utf-8"))
        while True:
            b = c.messages.batches.retrieve(st["batch_id"])
            if b.processing_status == "ended":
                break
            print(f"{datetime.now():%H:%M:%S} {b.processing_status} {b.request_counts}", flush=True)
            time.sleep(60)
        n_ok = n_bad = 0
        tot_in = tot_out = 0
        with open(RAW, "a", encoding="utf-8") as raw, open(CACHE, "a", encoding="utf-8") as cache:
            for r in c.messages.batches.results(st["batch_id"]):
                raw.write(r.to_json(indent=None) + "\n")
                if r.result.type == "succeeded" and r.custom_id in uniq:
                    msg = r.result.message
                    cache.write(json.dumps(cache_row(r.custom_id, uniq[r.custom_id], msg, st["batch_id"]), ensure_ascii=False) + "\n")
                    n_ok += 1
                    tot_in += msg.usage.input_tokens
                    tot_out += msg.usage.output_tokens
                else:
                    n_bad += 1
        st.update({"status": "collected", "succeeded": n_ok, "not_succeeded": n_bad, "input_tokens": tot_in, "output_tokens": tot_out,
                   "estimated_cost_usd": (tot_in * PRICE_IN + tot_out * PRICE_OUT) / 1e6 * BATCH_DISCOUNT,
                   "collected_utc": datetime.now(timezone.utc).isoformat()})
        json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1)
        print(json.dumps(st, indent=1))
    if args.score:
        apply_truncation_rule()
        left = [k for k in uniq if k not in cached_keys()]
        if left:
            print(f"WARNING: {len(left)} prompts have no reply; they are left out of the scored rows")
        sys.argv = [sys.argv[0], "--model", MODEL, "--benchmarks", "2wiki", "--score-only"]
        E.main()


if __name__ == "__main__":
    main()
