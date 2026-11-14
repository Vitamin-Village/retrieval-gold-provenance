# -*- coding: utf-8 -*-
"""Logged LM re-application of the census procedure (2026-09-29).

Logged, reproducible LM runs of the census step procedure. They test whether agreement with the census
depends on the author-condensed item descriptions (condition A, workbook) or survives with verbatim
source text only (condition B, excerpt-only). The design is fixed in
human_validation/llm_census_logged/PROTOCOL.md.

Every request is one workbook row in a fresh context, with the v3 rater instructions (two sentences
about the human Rater_info sheet removed) as the system prompt. No fallback model is ever used: a
refusal, truncation, or unparsable reply is stored as it is and counts as missing.

Usage (from paper/, Python with anthropic, httpx, openpyxl, numpy, scipy; keys in ../.env):
  python eval/llm_census_logged.py --prompts        write prompts/ only (no API call)
  python eval/llm_census_logged.py --pilot          parse check on 3 rows per model -> pilot_raw.jsonl
  python eval/llm_census_logged.py --run            all runs (resumable) -> raw.jsonl
  python eval/llm_census_logged.py --score          recompute everything from raw.jsonl
Outputs in human_validation/llm_census_logged/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # importing the census scripts must not leave caches outside this study's folder
PAPER = Path(__file__).resolve().parents[1]
HV = PAPER / "human_validation"
V3 = HV / "v3"
OUT = HV / "llm_census_logged"
RAW = OUT / "raw.jsonl"
PILOT = OUT / "pilot_raw.jsonl"
PARAMS = OUT / "model_params.json"
PROMPTS = OUT / "prompts"
RUNS = OUT / "runs"

GEMINI_MODEL = "gemini-3.8-flash"
MODELS = {"claude-opus-5-5": 1, "claude-sonnet-5-5": 2, "claude-haiku-4-5": 3, GEMINI_MODEL: 4}
CONDS = {"A": 1, "B": 2}
N_RUNS = 3
SEED_BASE = 20260929000
MAX_WORKERS = 6
BUDGET_USD = 150.0
CLAUDE_MAX_TOKENS = 16000
GEMINI_MAX_TOKENS = 16384
TEMPERATURE = 1.0
# USD per million tokens: input, output, cache read, cache write (5-minute TTL)
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
          "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25)}
CORRECTIONS = {"none", "B3-1", "B3-2", "B3-3", "B3-4", "B3-1+B3-2", "B3-2+B3-4", "B3-1+B3-4"}
REMOVED_SENTENCES = [  # address the human Rater_info sheet (background, consent, no-AI attestation)
    "Please fill the `Rater_info` sheet in each workbook.",
    "Record start and end times in `Rater_info`. Breaks are fine; note breaks longer than 10 minutes.",
]
FORMAT_APPENDIX = """

## 7. How you receive the rows in this run

Each message gives you one row of a workbook sheet (the quiz, Part A, or Part B) and names the stage and sheet. Messages are independent: rate the row on its own, using this document and the row only. For a quiz or Part A row, fill the fields `source_K_A_T`, `base_bin_0_4`, `corrections`, `final_bin_0_4`, `key_phrase`, and `confidence_1_3`. For a Part B row, fill `your_Q1`, `your_Q2`, and, only if the row itself looks wrong, `flag_data_problem`.

You may reason before answering. End your reply with exactly one line in the form below that fits the row, with valid JSON after the colon and nothing after it:

RATING: {"source_K_A_T": "<K, A, or T>", "base_bin_0_4": <integer 0-4>, "corrections": "<one of: none, B3-1, B3-2, B3-3, B3-4, B3-1+B3-2, B3-2+B3-4, B3-1+B3-4>", "final_bin_0_4": <integer 0-4>, "key_phrase": "<the deciding words copied from the row; add alt N if torn>", "confidence_1_3": <integer 1-3>}

PARTB: {"your_Q1": <integer on the row's Q1 scale>, "your_Q2": <integer on the row's Q2 scale>, "flag_data_problem": "<short note, or empty>"}
"""
RATING_FIELDS = "Fill source_K_A_T, base_bin_0_4, corrections, final_bin_0_4, key_phrase, and confidence_1_3 for this row. End your reply with the RATING line."
PARTB_FIELDS = "Answer your_Q1 and your_Q2 for this row (and flag_data_problem only if the row itself looks wrong). End your reply with the PARTB line."
COND_B_NOTE = ("In this version of the sheet a row gives the benchmark name, the axis, and verbatim excerpts from the benchmark's own "
               "paper, README, or generator source code; it gives no condensed construction description.")

LOCK = threading.Lock()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_key(name: str) -> str | None:
    for envp in (PAPER / ".env", PAPER.parent / ".env"):
        if envp.exists():
            for line in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().startswith(name):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(name)


# ----------------------------------------------------------------------------- instrument
def sheet_rows(path: Path, name: str) -> list[dict]:
    from openpyxl import load_workbook
    ws = load_workbook(path, data_only=True)[name]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:] if any(v not in (None, "") for v in r)]


def load_instrument() -> dict:
    quiz = sorted(sheet_rows(V3 / "sheets" / "rater_1_stage1_quiz.xlsx", "Quiz"), key=lambda r: r["qid"])
    part_a = sorted(sheet_rows(V3 / "sheets" / "rater_1_stage2_main.xlsx", "PartA"), key=lambda r: r["item_id"])
    part_b = sorted(sheet_rows(V3 / "sheets" / "rater_1_stage2_main.xlsx", "PartB"), key=lambda r: r["item_id"])
    assert len(quiz) == 6 and len(part_a) == 31 and len(part_b) == 130, (len(quiz), len(part_a), len(part_b))
    excerpts = json.load(open(OUT / "excerpts.json", encoding="utf-8"))
    return {"quiz": quiz, "A": part_a, "B": part_b, "excerpts": excerpts}


def system_prompts() -> dict:
    instr = (V3 / "INSTRUCTIONS_v3.md").read_text(encoding="utf-8")
    for s in REMOVED_SENTENCES:
        assert instr.count(s) == 1, s
        instr = instr.replace(" " + s, "") if (" " + s) in instr else instr.replace(s, "")
    quiz_key = (V3 / "QUIZ_KEY_v3.md").read_text(encoding="utf-8").strip()
    stage1 = instr.rstrip() + "\n" + FORMAT_APPENDIX
    stage2 = (instr.rstrip() + "\n\n## Stage-1 quiz key, as sent to raters before stage 2\n\n" + quiz_key + "\n" + FORMAT_APPENDIX)
    return {"stage1": stage1, "stage2": stage2}


def cell(v) -> str:
    return "" if v is None else str(v)


def source_label(e: dict) -> str:
    if e["source"] in ("upper", "lower", "lower_test"):
        return "benchmark generator source code"
    if e["source"] == "metaqa_readme":
        return "benchmark README"
    pl = e["pages_or_line"]
    pages = pl["pages"] if isinstance(pl, dict) else pl
    return f"benchmark paper, PDF p. {pages[0]}" if len(pages) == 1 else f"benchmark paper, PDF pp. {pages[0]}-{pages[-1]}"


def quiz_message(r: dict) -> str:
    return ("Stage 1, sheet Quiz, one row:\n\n"
            f"qid: {r['qid']}\nbenchmark: {r['benchmark']}\naxis: {r['axis']}\n"
            f"construction_description: {r['construction_description']}\n\n{RATING_FIELDS}")


def part_a_message(r: dict, pos: int, cond: str, excerpts: dict) -> str:
    if cond == "A":
        return ("Stage 2, sheet PartA, one row:\n\n"
                f"position: {pos}\nitem_id: {r['item_id']}\nbenchmark: {r['benchmark']}\naxis: {r['axis']}\n"
                f"construction_description: {cell(r['construction_description'])}\n"
                f"verbatim_excerpt_from_benchmark_paper: {cell(r['verbatim_excerpt_from_benchmark_paper'])}\n\n{RATING_FIELDS}")
    if r["item_id"].startswith("CHECK_"):  # invented attention items have no source paper
        exs = [{"label": "benchmark documentation", "text": r["construction_description"], "code": False}]
    else:
        exs = [{"label": source_label(e), "text": e["text"], "code": e["source"] in ("upper", "lower", "lower_test")}
               for e in excerpts["items"][r["item_id"]]["excerpts"]]
    lines = []
    for i, e in enumerate(exs, 1):
        if e["code"]:
            lines.append(f"[{i}] ({e['label']}):\n```\n{e['text']}\n```")
        else:
            lines.append(f"[{i}] ({e['label']}) \"{e['text']}\"")
    return (f"Stage 2, sheet PartA, one row. {COND_B_NOTE}\n\n"
            f"position: {pos}\nbenchmark: {r['benchmark']}\naxis: {r['axis']}\nverbatim_excerpts:\n" + "\n".join(lines) +
            f"\n\n{RATING_FIELDS}")


def part_b_message(r: dict, pos: int) -> str:
    keys = ["item_id", "benchmark", "question", "gold_answer", "unit_type", "unit_title", "unit_text", "structure_type",
            "structure_text", "Q1", "Q1_scale", "Q2", "Q2_scale"]
    body = "\n".join(f"{k}: {cell(r[k])}" for k in keys)
    return f"Stage 2, sheet PartB, one row:\n\nposition: {pos}\n{body}\n\n{PARTB_FIELDS}"


def run_seed(model: str, cond: str, run: int) -> int:
    return SEED_BASE + 100 * MODELS[model] + 10 * CONDS[cond] + run


def build_tasks(inst: dict, models: list[str]) -> list[dict]:
    tasks = []
    for model in models:
        for cond in CONDS:
            for run in range(1, N_RUNS + 1):
                seed = run_seed(model, cond, run)
                rng = random.Random(seed)
                quiz, part_a, part_b = list(inst["quiz"]), list(inst["A"]), list(inst["B"])
                rng.shuffle(quiz)
                rng.shuffle(part_a)
                rng.shuffle(part_b)
                base = {"model": model, "condition": cond, "run": run, "seed": seed}
                for pos, r in enumerate(quiz, 1):
                    tasks.append({**base, "stage": "quiz", "item_id": r["qid"], "position": pos, "system": "stage1",
                                  "user": quiz_message(r)})
                for pos, r in enumerate(part_a, 1):
                    tasks.append({**base, "stage": "partA", "item_id": r["item_id"], "position": pos, "system": "stage2",
                                  "user": part_a_message(r, pos, cond, inst["excerpts"])})
                if cond == "A":
                    for pos, r in enumerate(part_b, 1):
                        tasks.append({**base, "stage": "partB", "item_id": r["item_id"], "position": pos, "system": "stage2",
                                      "user": part_b_message(r, pos)})
    return tasks


def write_prompts(inst: dict, systems: dict) -> None:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    man = {"generated_utc": now(), "files": {}}
    for name, text in systems.items():
        (PROMPTS / f"system_{name}.txt").write_text(text, encoding="utf-8")
        man["files"][f"system_{name}.txt"] = sha(text)
    rem = "Sentences removed from ../v3/INSTRUCTIONS_v3.md for the LM system prompt (they address the human Rater_info sheet):\n\n" + \
          "\n".join(f"- {s}" for s in REMOVED_SENTENCES) + "\n"
    (PROMPTS / "removed_sentences.md").write_text(rem, encoding="utf-8")
    items = {"quiz_both_conditions.jsonl": [(r["qid"], quiz_message(r)) for r in inst["quiz"]],
             "partA_conditionA.jsonl": [(r["item_id"], part_a_message(r, 0, "A", inst["excerpts"])) for r in inst["A"]],
             "partA_conditionB.jsonl": [(r["item_id"], part_a_message(r, 0, "B", inst["excerpts"])) for r in inst["A"]],
             "partB_conditionA.jsonl": [(r["item_id"], part_b_message(r, 0)) for r in inst["B"]]}
    for fn, rows in items.items():
        with open(PROMPTS / fn, "w", encoding="utf-8") as f:
            for iid, text in rows:
                f.write(json.dumps({"item_id": iid, "text_with_position_0": text, "sha256": sha(text)}, ensure_ascii=False) + "\n")
        man["files"][fn] = sha((PROMPTS / fn).read_text(encoding="utf-8"))
    man["note"] = ("Item texts are shown with position 0; in each run the position is the row's place in that run's shuffled "
                   "sheet. The exact user text of every request is stored in raw.jsonl.")
    json.dump(man, open(PROMPTS / "manifest.json", "w", encoding="utf-8"), indent=1)


# ----------------------------------------------------------------------------- API calls
class Caller:
    def __init__(self, systems: dict, params: dict):
        self.systems = systems
        self.params = params
        self._claude = None
        self._gkey = None

    def claude(self):
        if self._claude is None:
            import anthropic
            key = env_key("ANTHROPIC_API_KEY")
            if not key:
                sys.exit("no ANTHROPIC_API_KEY in paper/.env, ../.env, or the environment")
            self._claude = anthropic.Anthropic(api_key=key, max_retries=5, timeout=900.0)
        return self._claude

    def request_params(self, model: str, system_name: str) -> dict:
        p = self.params.get(model, {})
        if model.startswith("claude"):
            d = {"api": "anthropic.messages.create", "model": model, "max_tokens": CLAUDE_MAX_TOKENS,
                 "system": [{"type": "text", "text": f"<{system_name}, sha256 {sha(self.systems[system_name])}>",
                             "cache_control": {"type": "ephemeral"}}],
                 "messages": [{"role": "user", "content": "<user text, stored in field user>"}],
                 "thinking": "not sent (API default)", "output_config": "not sent (API default)"}
            if p.get("send_temperature", True):
                d["temperature"] = TEMPERATURE
            else:
                d["temperature"] = "not sent (rejected by the model; API default)"
            return d
        return {"api": "generativelanguage v1beta generateContent (REST)", "model": model,
                "systemInstruction": f"<{system_name}, sha256 {sha(self.systems[system_name])}>",
                "contents": [{"role": "user", "parts": ["<user text, stored in field user>"]}],
                "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": GEMINI_MAX_TOKENS},
                "thinkingConfig": "not sent (API default)"}

    def call(self, model: str, system_name: str, user: str, send_temperature: bool | None = None) -> dict:
        system = self.systems[system_name]
        if model.startswith("claude"):
            if send_temperature is None:
                send_temperature = self.params.get(model, {}).get("send_temperature", True)
            kw = dict(model=model, max_tokens=CLAUDE_MAX_TOKENS,
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                      messages=[{"role": "user", "content": user}])
            if send_temperature:
                kw["temperature"] = TEMPERATURE
            msg = self.claude().messages.create(**kw)
            text = "".join(b.text for b in msg.content if b.type == "text")
            u = msg.usage
            stop_details = getattr(msg, "stop_details", None)
            return {"model_returned": msg.model, "stop_reason": msg.stop_reason, "text": text,
                    "stop_details": stop_details.model_dump() if hasattr(stop_details, "model_dump") else stop_details,
                    "usage": {"input": u.input_tokens, "output": u.output_tokens,
                              "cache_read": getattr(u, "cache_read_input_tokens", None) or 0,
                              "cache_write": getattr(u, "cache_creation_input_tokens", None) or 0},
                    "request_id": getattr(msg, "_request_id", None), "response_id": msg.id}
        return self.call_gemini(model, system, user)

    def call_gemini(self, model: str, system: str, user: str) -> dict:
        import httpx
        if self._gkey is None:
            self._gkey = env_key("GEMINI_API_KEY")
            if not self._gkey:
                raise RuntimeError("no GEMINI_API_KEY")
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": GEMINI_MAX_TOKENS}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        last = None
        for attempt in range(6):
            try:
                r = httpx.post(url, headers={"x-goog-api-key": self._gkey}, json=body, timeout=900)
            except httpx.HTTPError as e:
                last = repr(e)
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            d = r.json()
            cands = d.get("candidates") or []
            cand = cands[0] if cands else {}
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            stop = cand.get("finishReason") or (d.get("promptFeedback") or {}).get("blockReason") or "NO_CANDIDATE"
            um = d.get("usageMetadata") or {}
            return {"model_returned": d.get("modelVersion"), "stop_reason": stop, "text": text, "stop_details": None,
                    "usage": {"input": um.get("promptTokenCount"), "output": um.get("candidatesTokenCount"),
                              "thoughts": um.get("thoughtsTokenCount"), "cache_read": um.get("cachedContentTokenCount") or 0,
                              "cache_write": 0, "total": um.get("totalTokenCount")},
                    "request_id": None, "response_id": d.get("responseId")}
        raise RuntimeError(f"gave up after retries: {last}")


# ----------------------------------------------------------------------------- parsing
def _last_json_after(text: str, tag: str):
    idx = text.rfind(tag + ":")
    if idx < 0:
        return None
    brace = text.find("{", idx)
    if brace < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[brace:])
        return obj
    except json.JSONDecodeError:
        line_end = text.find("\n", brace)
        frag = text[brace:line_end if line_end > 0 else None].strip().rstrip("`").strip()
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            return None


def _as_int(v, lo, hi):
    try:
        x = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if lo <= x <= hi else None


def parse_reply(stage: str, text: str, stop_reason: str | None) -> dict:
    if stop_reason in ("refusal", "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "OTHER"):
        return {"ok": False, "why": f"refusal/{stop_reason}"}
    if stage == "partB":
        obj = _last_json_after(text or "", "PARTB")
        if not isinstance(obj, dict):
            return {"ok": False, "why": f"no PARTB line (stop {stop_reason})"}
        q1, q2 = _as_int(obj.get("your_Q1"), 0, 2), _as_int(obj.get("your_Q2"), 0, 1)
        if q1 is None or q2 is None:
            return {"ok": False, "why": "PARTB values out of range", "obj": obj}
        return {"ok": True, "Q1": q1, "Q2": q2, "flag": str(obj.get("flag_data_problem") or "").strip()}
    obj = _last_json_after(text or "", "RATING")
    if not isinstance(obj, dict):
        return {"ok": False, "why": f"no RATING line (stop {stop_reason})"}
    src = str(obj.get("source_K_A_T", "")).strip().upper()
    base, fin = _as_int(obj.get("base_bin_0_4"), 0, 4), _as_int(obj.get("final_bin_0_4"), 0, 4)
    conf = _as_int(obj.get("confidence_1_3"), 1, 3)
    if src not in ("K", "A", "T") or base is None or fin is None or conf is None:
        return {"ok": False, "why": "RATING fields invalid", "obj": obj}
    corr = str(obj.get("corrections", "")).strip()
    return {"ok": True, "source": src, "base": base, "final": fin, "confidence": conf, "corrections": corr,
            "corrections_in_list": corr in CORRECTIONS, "key_phrase": str(obj.get("key_phrase", ""))}


# ----------------------------------------------------------------------------- running
def rec_key(x: dict) -> tuple:
    return (x["model"], x["condition"], x["run"], x["stage"], x["item_id"])


def load_raw(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                if not x.get("error"):
                    done[rec_key(x)] = x
    return done


def cost_usd(rec: dict) -> float:
    p = PRICES.get(rec.get("model"))
    u = rec.get("usage") or {}
    if not p:
        return 0.0
    return (p[0] * (u.get("input") or 0) + p[1] * (u.get("output") or 0) + p[2] * (u.get("cache_read") or 0)
            + p[3] * (u.get("cache_write") or 0)) / 1e6


def execute(caller: Caller, tasks: list[dict], path: Path, spent0: float = 0.0) -> None:
    spent = [spent0]
    stop = threading.Event()
    n = [0]

    def one(t: dict):
        if stop.is_set():
            return
        rec = {k: t[k] for k in ("model", "condition", "run", "seed", "stage", "item_id", "position", "system")}
        rec["system_sha256"] = sha(caller.systems[t["system"]])
        rec["user"] = t["user"]
        rec["request"] = caller.request_params(t["model"], t["system"])
        rec["utc_sent"] = now()
        try:
            res = caller.call(t["model"], t["system"], t["user"])
            rec.update(res)
            rec["parsed"] = parse_reply(t["stage"], res["text"], res["stop_reason"])
            rec["error"] = None
        except Exception as e:  # logged; the key is retried on the next invocation
            rec["error"] = repr(e)[:1000]
        rec["utc"] = now()
        with LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            spent[0] += cost_usd(rec)
            n[0] += 1
            if spent[0] > BUDGET_USD:
                stop.set()
            if n[0] % 50 == 0 or rec.get("error"):
                print(f"  {n[0]}/{len(tasks)}  est. Claude cost ${spent[0]:.2f}  last: {rec['model']} {rec['condition']}"
                      f"{rec['run']} {rec['stage']} {rec['item_id']} err={rec.get('error')}", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(one, tasks))
    if stop.is_set():
        print(f"STOPPED: estimated cost ${spent[0]:.2f} exceeded the budget of ${BUDGET_USD}", flush=True)


def load_params() -> dict:
    return json.load(open(PARAMS, encoding="utf-8")) if PARAMS.exists() else {}


def pilot(models: list[str]) -> None:
    import anthropic
    inst, systems = load_instrument(), system_prompts()
    write_prompts(inst, systems)
    params = load_params()
    caller = Caller(systems, params)
    for model in models:
        tasks = [t for t in build_tasks(inst, [model]) if t["condition"] == "A" and t["run"] == 1]
        picks = [next(t for t in tasks if t["stage"] == s) for s in ("quiz", "partA", "partB")]
        entry = params.setdefault(model, {})
        for i, t in enumerate(picks):
            rec = {k: t[k] for k in ("model", "condition", "run", "seed", "stage", "item_id", "position", "system")}
            rec["user"], rec["utc_sent"] = t["user"], now()
            try:
                if model.startswith("claude") and i == 0:
                    try:
                        res = caller.call(model, t["system"], t["user"], send_temperature=True)
                        entry["send_temperature"] = True
                        entry["temperature_probe"] = "accepted temperature=1.0"
                    except anthropic.BadRequestError as e:
                        entry["send_temperature"] = False
                        entry["temperature_probe"] = f"rejected: {str(e)[:400]}"
                        res = caller.call(model, t["system"], t["user"], send_temperature=False)
                else:
                    res = caller.call(model, t["system"], t["user"])
                rec.update(res)
                rec["parsed"] = parse_reply(t["stage"], res["text"], res["stop_reason"])
                rec["error"] = None
            except Exception as e:
                rec["error"] = repr(e)[:1000]
            rec["request"] = caller.request_params(model, t["system"])
            rec["utc"] = now()
            with open(PILOT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            entry.setdefault("pilot", []).append({"stage": t["stage"], "item_id": t["item_id"], "error": rec.get("error"),
                                                  "model_returned": rec.get("model_returned"), "stop_reason": rec.get("stop_reason"),
                                                  "parsed": rec.get("parsed"), "usage": rec.get("usage")})
            print(model, t["stage"], t["item_id"], "->", rec.get("error") or (rec.get("model_returned"), rec.get("stop_reason"),
                                                                             rec.get("parsed"), rec.get("usage")), flush=True)
        entry["checked_utc"] = now()
    json.dump(params, open(PARAMS, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)


def run_all(models: list[str]) -> None:
    inst, systems = load_instrument(), system_prompts()
    write_prompts(inst, systems)
    params = load_params()
    for m in models:
        if m not in params:
            sys.exit(f"run --pilot for {m} first (model_params.json has no entry)")
    caller = Caller(systems, params)
    done = load_raw(RAW)
    tasks = [t for t in build_tasks(inst, models) if rec_key(t) not in done]
    spent0 = sum(cost_usd(x) for x in done.values())
    print(f"{len(tasks)} requests to run ({len(done)} already stored; est. Claude cost so far ${spent0:.2f})", flush=True)
    execute(caller, tasks, RAW, spent0)
    left = [t for t in build_tasks(inst, models) if rec_key(t) not in load_raw(RAW)]
    print(f"done; {len(left)} keys without a stored reply", flush=True)


# ----------------------------------------------------------------------------- statistics
def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def qwk(a, b, k=5):
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1.0
    ra = [sum(r) for r in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = sum(((i - j) ** 2) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(((i - j) ** 2) * ra[i] * rb[j] / n for i in range(k) for j in range(k))
    return 1.0 - num / den if den > 0 else float("nan")


def kalpha_ordinal(units, k=5):
    """Krippendorff's alpha, ordinal metric, via the coincidence matrix; units = per-item lists of values
    (missing values already dropped; items with fewer than two values carry no information)."""
    o = defaultdict(float)
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for i, j in itertools.permutations(range(m), 2):
            o[(u[i], u[j])] += 1.0 / (m - 1)
    nc = {c: sum(o[(c, d)] for d in range(k)) for c in range(k)}
    n = sum(nc.values())
    if n <= 1:
        return float("nan")

    def delta(c, d):
        lo, hi = min(c, d), max(c, d)
        return (sum(nc[g] for g in range(lo, hi + 1)) - (nc[lo] + nc[hi]) / 2.0) ** 2

    do = sum(o[(c, d)] * delta(c, d) for c in range(k) for d in range(k)) / n
    de = sum(nc[c] * nc[d] * delta(c, d) for c in range(k) for d in range(k)) / (n * (n - 1))
    return 1.0 - do / de if de > 0 else float("nan")


def pair_metrics(x: dict, y: dict, ids: list[str]) -> dict:
    common = [i for i in ids if x.get(i) is not None and y.get(i) is not None]
    a, b = [x[i] for i in common], [y[i] for i in common]
    n = len(common)
    if n == 0:
        return {"n": 0}
    return {"n": n, "kappa_qw": qwk(a, b), "exact": sum(p == q for p, q in zip(a, b)) / n,
            "within_1": sum(abs(p - q) <= 1 for p, q in zip(a, b)) / n,
            "binary_agree": sum((p >= 2) == (q >= 2) for p, q in zip(a, b)) / n,
            "alpha_ordinal": kalpha_ordinal([[p, q] for p, q in zip(a, b)])}


def median_low(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return v[(len(v) - 1) // 2]


METRICS = {"kappa_qw": lambda a, b: qwk(a, b), "exact": lambda a, b: float((a == b).mean()),
           "within_1": lambda a, b: float((abs(a - b) <= 1).mean()),
           "binary_agree": lambda a, b: float(((a >= 2) == (b >= 2)).mean())}


def boot_diff(pairs_1, pairs_2, metric, B=10000, seed=20260929):
    """Paired bootstrap over items of metric(pairs_1) - metric(pairs_2); each pairs_* is a list of
    (rater_vector, reference_vector) numpy arrays aligned on the same items; with several pairs the
    metric is averaged over them."""
    import numpy as np
    n = len(pairs_1[0][0])
    f = METRICS[metric]
    point = float(np.mean([f(a, b) for a, b in pairs_1]) - np.mean([f(a, b) for a, b in pairs_2]))
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for t in range(B):
        idx = rng.integers(0, n, n)
        diffs[t] = np.mean([f(a[idx], b[idx]) for a, b in pairs_1]) - np.mean([f(a[idx], b[idx]) for a, b in pairs_2])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci95": [float(lo), float(hi)], "B": B, "n_items": n,
            "n_boot_undefined": int(np.isnan(diffs).sum())}


def score(models_filter: list[str] | None = None) -> None:
    import numpy as np
    recs = load_raw(RAW)
    all_lines = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    models = [m for m in MODELS if any(k[0] == m for k in recs)]
    if models_filter:
        models = [m for m in models if m in models_filter]
    key3 = json.load(open(V3 / "answer_key_v3.json", encoding="utf-8"))
    census = json.load(open(HV / "reference_bins.json", encoding="utf-8"))["reference"]
    assert census == key3["partA_census"]
    items29 = json.load(open(HV / "items_29.json", encoding="utf-8"))["items"]
    ids = [x["item_id"] for x in items29]
    anchors = [x["item_id"] for x in items29 if x["benchmark"].startswith("Anchor benchmark")]
    external = [i for i in ids if i not in anchors]
    refs = {"census": census}
    rr = _import("rater_robustness", PAPER / "analysis" / "rater_robustness.py")

    # ---- parse every run (re-parsed from the stored reply text)
    RUNS.mkdir(parents=True, exist_ok=True)
    runs, qc = {}, {}
    for m in models:
        for c in CONDS:
            for r in range(1, N_RUNS + 1):
                sel = {k: v for k, v in recs.items() if k[:3] == (m, c, r)}
                if not sel:
                    continue
                parsed = {}
                for k, x in sel.items():
                    parsed[(k[3], k[4])] = {"parse": parse_reply(k[3], x.get("text", ""), x.get("stop_reason")), "rec": x}
                quiz = {i: p["parse"] for (s, i), p in parsed.items() if s == "quiz"}
                pa = {i: p["parse"] for (s, i), p in parsed.items() if s == "partA"}
                pb = {i: p["parse"] for (s, i), p in parsed.items() if s == "partB"}
                fin = {i: (pa[i]["final"] if pa.get(i, {}).get("ok") else None) for i in ids}
                quiz_ok = sum(1 for q, kq in key3["quiz"].items() if quiz.get(q, {}).get("ok") and quiz[q]["final"] == kq["final"])
                att = {a: (pa[a]["final"] if pa.get(a, {}).get("ok") else None) for a in key3["anchors"]}
                att_pass = {a: (v is not None and abs(v - key3["anchors"][a]) <= 1) for a, v in att.items()}
                # human exclusion rules, applied as flags: quiz >= 4/6; excluded if >= 2 bins off on BOTH attention items
                excluded = quiz_ok < 4 or all(v is not None and abs(v - key3["anchors"][a]) >= 2 for a, v in att.items())
                # a run cut short by API errors (no stored reply for some rows) is reported per run but is not
                # used in model medians, alpha, Part B, or the robustness table (deviation 10.2 of PROTOCOL.md)
                complete = len(sel) == (6 + 31 + (130 if c == "A" else 0))
                stops = Counter(p["rec"].get("stop_reason") for p in parsed.values())
                returned = sorted({str(p["rec"].get("model_returned")) for p in parsed.values()})
                fails = {f"{s}:{i}": p["parse"]["why"] for (s, i), p in parsed.items() if not p["parse"]["ok"]}
                corr_off = sorted(i for i, p in pa.items() if p.get("ok") and not p["corrections_in_list"])
                usage = defaultdict(int)
                for p in parsed.values():
                    for kk, vv in (p["rec"].get("usage") or {}).items():
                        usage[kk] += vv or 0
                cost = sum(cost_usd(p["rec"]) for p in parsed.values())
                name = f"{m}_{c}_{r}"
                runs[name] = {"model": m, "condition": c, "run": r, "seed": run_seed(m, c, r), "complete": complete, "final": fin,
                              "base": {i: (pa[i]["base"] if pa.get(i, {}).get("ok") else None) for i in ids},
                              "source": {i: (pa[i]["source"] if pa.get(i, {}).get("ok") else None) for i in ids},
                              "confidence": {i: (pa[i]["confidence"] if pa.get(i, {}).get("ok") else None) for i in ids},
                              "partB": {i: ({"Q1": p["Q1"], "Q2": p["Q2"], "flag": p["flag"]} if p.get("ok") else None) for i, p in pb.items()}}
                qc[name] = {"n_requests": len(sel), "quiz_correct_final": quiz_ok, "attention": att, "attention_pass": att_pass,
                            "excluded_by_human_rules": excluded, "complete": complete, "stop_reasons": dict(stops), "models_returned": returned,
                            "failed_parses": fails, "corrections_not_in_list": corr_off, "usage": dict(usage), "cost_usd": cost}
                out = {**runs[name], "qc": qc[name],
                       "partA_detail": {i: p for i, p in pa.items()}, "quiz_detail": {i: p for i, p in quiz.items()}}
                json.dump(out, open(RUNS / f"{name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    def run_names(m, c, include_excluded=False):
        return [f"{m}_{c}_{r}" for r in range(1, N_RUNS + 1) if f"{m}_{c}_{r}" in runs
                and qc[f"{m}_{c}_{r}"]["complete"]
                and (include_excluded or not qc[f"{m}_{c}_{r}"]["excluded_by_human_rules"])]

    def med(m, c, include_excluded=False):
        ns = run_names(m, c, include_excluded)
        return {i: median_low([runs[n]["final"][i] for n in ns]) for i in ids} if ns else None

    res = {"generated_utc": now(), "models": models, "n_items": len(ids), "anchors": anchors, "n_external": len(external),
           "qc": qc, "per_run": {}, "per_model_median": {}, "alpha": {}, "headline_KG": {},
           "condition_A_minus_B": {}, "partB": {}, "requests": {}}
    # request accounting (all lines, including errors that were later retried)
    for m in models:
        lines = [x for x in all_lines if x["model"] == m]
        res["requests"][m] = {"lines": len(lines), "api_errors_logged": sum(1 for x in lines if x.get("error")),
                              "stored_replies": sum(1 for k in recs if k[0] == m),
                              "unparsed_or_refused": sum(len(qc[n]["failed_parses"]) for n in qc if n.startswith(m + "_")),
                              "models_returned": sorted({str(x.get("model_returned")) for x in lines if not x.get("error")}),
                              "cost_usd_est": sum(cost_usd(x) for x in lines),
                              "tokens": {kk: sum((x.get("usage") or {}).get(kk) or 0 for x in lines)
                                         for kk in ("input", "output", "cache_read", "cache_write", "thoughts")}}

    def vs_refs(r):
        return {ref: {"all29": pair_metrics(r, rv, ids), "external23": pair_metrics(r, rv, external)} for ref, rv in refs.items()}

    for n, x in runs.items():
        res["per_run"][n] = vs_refs(x["final"])
    medians = {}
    for m in models:
        for c in CONDS:
            md = med(m, c)
            if md is None:
                continue
            medians[(m, c)] = md
            md_all = med(m, c, include_excluded=True)
            res["per_model_median"][f"{m}_{c}"] = {"runs_used": run_names(m, c), "bins": md, **vs_refs(md),
                                                   "with_excluded_runs": vs_refs(md_all) if md_all != md else "same"}

    # ---- alpha
    def alpha_of(names_or_dicts, item_ids):
        cols = [runs[n]["final"] if isinstance(n, str) else n for n in names_or_dicts]
        return kalpha_ordinal([[c[i] for c in cols if c.get(i) is not None] for i in item_ids])
    for c in CONDS:
        for m in models:
            ns = [f"{m}_{c}_{r}" for r in range(1, N_RUNS + 1) if f"{m}_{c}_{r}" in runs and qc[f"{m}_{c}_{r}"]["complete"]]
            if ns:
                res["alpha"][f"within_{m}_{c}"] = {"all29": alpha_of(ns, ids), "external23": alpha_of(ns, external)}
        meds = [medians[(m, c)] for m in models if (m, c) in medians]
        if len(meds) > 1:
            res["alpha"][f"across_model_medians_{c}"] = {"all29": alpha_of(meds, ids), "external23": alpha_of(meds, external),
                                                         "with_census": alpha_of(meds + [census], ids)}
        alln = [n for n in runs if runs[n]["condition"] == c and qc[n]["complete"]]
        if alln:
            res["alpha"][f"all_runs_{c}"] = {"n_runs": len(alln), "all29": alpha_of(alln, ids), "external23": alpha_of(alln, external),
                                             "with_census": alpha_of(alln + [census], ids)}

    # ---- headline C_KG per run / median
    def kg(r):
        out = {}
        for b, axes in rr.KG_AXES.items():
            vals = [r.get(a) for a in axes if r.get(a) is not None]
            out[b] = (max(vals) if vals else None) if axes else 0
        return out
    six = ["GRADE", "STaRK", "GraphRAG-Bench", "2WikiMultiHop", "MetaQA", "WebQSP"]
    for n, x in runs.items():
        k = kg(x["final"])
        res["headline_KG"][n] = {"bins": {b: k[b] for b in six}, "ge3": {b: (k[b] is not None and k[b] >= 3) for b in six},
                                 "n_external_ge3": sum(1 for b in rr.EXTERNAL if k[b] is not None and k[b] >= 3)}
    for (m, c), md in medians.items():
        k = kg(md)
        res["headline_KG"][f"{m}_{c}_median"] = {"bins": {b: k[b] for b in six}, "ge3": {b: (k[b] is not None and k[b] >= 3) for b in six},
                                                 "n_external_ge3": sum(1 for b in rr.EXTERNAL if k[b] is not None and k[b] >= 3)}
    for nm, rv in refs.items():
        k = kg(rv)
        res["headline_KG"][nm] = {"bins": {b: k[b] for b in six}, "ge3": {b: k[b] >= 3 for b in six}}

    # ---- condition A minus B (model medians; paired bootstrap over items)
    def vecs(d, item_ids):
        return np.array([d[i] for i in item_ids])
    for scope, item_ids in (("all29", ids), ("external23", external)):
        for metric in METRICS:
            both = [m for m in models if (m, "A") in medians and (m, "B") in medians]
            for m in both:
                A, B = medians[(m, "A")], medians[(m, "B")]
                common = [i for i in item_ids if A.get(i) is not None and B.get(i) is not None]
                res["condition_A_minus_B"].setdefault(f"{m}", {}).setdefault(scope, {})[metric] = boot_diff(
                    [(vecs(A, common), vecs(census, common))], [(vecs(B, common), vecs(census, common))], metric)
            if len(both) > 1:
                common = [i for i in item_ids if all(medians[(m, c)].get(i) is not None for m in both for c in CONDS)]
                res["condition_A_minus_B"].setdefault("mean_over_models", {}).setdefault(scope, {})[metric] = boot_diff(
                    [(vecs(medians[(m, "A")], common), vecs(census, common)) for m in both],
                    [(vecs(medians[(m, "B")], common), vecs(census, common)) for m in both], metric)
    # mean over runs (not medians), per model
    for m in models:
        na, nb = run_names(m, "A"), run_names(m, "B")
        if na and nb:
            common = [i for i in ids if all(runs[n]["final"].get(i) is not None for n in na + nb)]
            for metric in ("kappa_qw", "exact"):
                res["condition_A_minus_B"].setdefault(m, {}).setdefault("all29_mean_over_runs", {})[metric] = boot_diff(
                    [(vecs(runs[n]["final"], common), vecs(census, common)) for n in na],
                    [(vecs(runs[n]["final"], common), vecs(census, common)) for n in nb], metric)

    # ---- Part B (condition A)
    res["partB"] = score_part_b(runs, models)
    json.dump(res, open(OUT / "result.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    write_md(res, runs, medians, ids, external)
    print((OUT / "result.md").read_text(encoding="utf-8"))


def score_part_b(runs: dict, models: list[str]) -> dict:
    sys.path.insert(0, str(HV))
    from lexical_rules import contains, majority_token_rule  # noqa: E402
    key3 = json.load(open(V3 / "answer_key_v3.json", encoding="utf-8"))
    idmap = key3["partB_idmap"]
    items = {it["item_id"] + "\x00" + it["question"]: it for it in json.load(open(HV / "items_partB.json", encoding="utf-8"))["items"]}
    unit = {nid: items[m["v2_item_id"] + "\x00" + m["question"]] for nid, m in idmap.items()}
    out = {}
    for m in models:
        ns = [f"{m}_A_{r}" for r in range(1, N_RUNS + 1) if f"{m}_A_{r}" in runs and runs[f"{m}_A_{r}"].get("complete")]
        if not ns:
            continue
        e_m = {"runs": ns, "missing": {n: sum(1 for v in runs[n]["partB"].values() if v is None) + (130 - len(runs[n]["partB"])) for n in ns}}
        for part in ("B1", "B2", "B3", "B4", "B5"):
            pn = sorted(n for n in idmap if n.startswith(part))
            q1 = [[(runs[n]["partB"].get(i) or {}).get("Q1") for n in ns] for i in pn]
            q2 = [[(runs[n]["partB"].get(i) or {}).get("Q2") for n in ns] for i in pn]
            mq1 = [median_low(v) for v in q1]
            mq2 = [median_low(v) for v in q2]
            e = {"n": len(pn), "Q1_alpha_runs": kalpha_ordinal([[x for x in v if x is not None] for v in q1], k=3),
                 "Q1_majority": dict(Counter(mq1)), "Q2_majority_rate": sum(x for x in mq2 if x is not None) / max(1, sum(x is not None for x in mq2)),
                 "Q2_all_agree": sum(1 for v in q2 if None not in v and len(set(v)) == 1) / len(pn)}
            if part in ("B1", "B2", "B3", "B4"):
                if part == "B1":
                    rule = [int(contains(unit[i]["unit_text"], unit[i]["structure_text"].strip("()").rsplit(", ", 1)[-1])) for i in pn]
                    lm = [None if x is None else int(x >= 1) for x in mq1]
                    lm_q2 = mq2
                    e["Q2_majority_vs_object_rule"] = f"{sum(1 for a, b in zip(lm_q2, rule) if a is not None and a == b)}/{len(pn)}"
                elif part == "B2":
                    rule = [int(majority_token_rule(unit[i]["structure_text"], unit[i]["unit_text"])) for i in pn]
                    lm = [None if x is None else int(x >= 1) for x in mq1]
                else:
                    rule = [int(contains(unit[i]["unit_text"], unit[i]["answer"])) for i in pn]
                    lm = mq2
                e["rule_rate"] = sum(rule) / len(pn)
                e["majority_vs_rule"] = f"{sum(1 for a, b in zip(lm, rule) if a is not None and a == b)}/{len(pn)}"
            e_m[part] = e
        e_m["flags"] = {n: {i: v["flag"] for i, v in runs[n]["partB"].items() if v and v["flag"]} for n in ns}
        out[m] = e_m
    return out


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write_md(res, runs, medians, ids, external):
    L = ["# Logged LM re-application of the census procedure: results", "",
         f"Generated {res['generated_utc']} by `eval/llm_census_logged.py --score` from `raw.jsonl`. LM raters, not human raters. "
         "Design: PROTOCOL.md. κ = quadratically weighted Cohen's kappa; exact / within-1 = share of items; binary = agreement on derived "
         "(bin ≥ 2) versus independent.", "", "## Requests", "",
         "| model | returned ids | stored replies | API errors logged (retried) | unparsed or refused | input tok | output tok | cache read | cache write | est. cost USD |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, r in res["requests"].items():
        t = r["tokens"]
        L.append(f"| {m} | {', '.join(r['models_returned'])} | {r['stored_replies']} | {r['api_errors_logged']} | {r['unparsed_or_refused']} | "
                 f"{t['input']} | {t['output']} | {t['cache_read']} | {t['cache_write']} | {r['cost_usd_est']:.2f} |")
    L += ["", "## Quiz, attention items, parse failures", "", "| run | complete | quiz (final bin) | attention CHECK_TEXT / CHECK_KG | excluded by human rules | failed parses | stop reasons |",
          "|---|---|---:|---|---|---|---|"]
    for n, q in res["qc"].items():
        L.append(f"| {n} | {'yes' if q['complete'] else 'NO: ' + str(q['n_requests']) + ' replies (API errors)'} | {q['quiz_correct_final']}/6 | {q['attention'].get('CHECK_TEXT')} / {q['attention'].get('CHECK_KG')} | "
                 f"{'yes' if q['excluded_by_human_rules'] else 'no'} | {len(q['failed_parses'])} {list(q['failed_parses'])[:4]} | {q['stop_reasons']} |")
    L += ["", "## Agreement with the census (29 items; 23 external in brackets)", "",
          "| rater | n | κ census | exact | within-1 | binary | κ census [23 ext] |", "|---|---:|---:|---:|---:|---:|---:|"]

    def row(name, e):
        c = e["census"]["all29"]
        return (f"| {name} | {c['n']} | {fmt(c['kappa_qw'])} | {fmt(c['exact'])} | {fmt(c['within_1'])} | {fmt(c['binary_agree'])} | "
                f"{fmt(e['census']['external23']['kappa_qw'])} |")
    for c in CONDS:
        L.append(f"| **condition {c}** | | | | | | |")
        for m in res["models"]:
            for r in range(1, N_RUNS + 1):
                n = f"{m}_{c}_{r}"
                if n in res["per_run"]:
                    L.append(row(n, res["per_run"][n]))
            if f"{m}_{c}" in res["per_model_median"]:
                L.append(row(f"**{m} {c} median of {len(res['per_model_median'][f'{m}_{c}']['runs_used'])} run(s)**", res["per_model_median"][f"{m}_{c}"]))
    L += ["", "## Krippendorff's α (ordinal)", "", "| scope | 29 items | 23 external | with census as extra rater |", "|---|---:|---:|---:|"]
    for k, v in res["alpha"].items():
        if isinstance(v, dict):
            L.append(f"| {k} | {fmt(v.get('all29'))} | {fmt(v.get('external23'))} | {fmt(v.get('with_census'))} |")
        else:
            L.append(f"| {k} | {fmt(v)} | | |")
    L += ["", "## Condition A (workbook) minus condition B (excerpt-only), agreement with the census", "",
          "Differences of model medians; 95% paired bootstrap CI over items (10,000 resamples).", "",
          "| model | scope | Δκ | Δexact | Δwithin-1 | Δbinary |", "|---|---|---|---|---|---|"]
    for m, e in res["condition_A_minus_B"].items():
        for scope in ("all29", "external23"):
            if scope in e:
                cells = [f"{fmt(e[scope][k]['diff'])} [{fmt(e[scope][k]['ci95'][0])}, {fmt(e[scope][k]['ci95'][1])}]" for k in METRICS]
                L.append(f"| {m} | {scope} | " + " | ".join(cells) + " |")
        if "all29_mean_over_runs" in e:
            x = e["all29_mean_over_runs"]
            L.append(f"| {m} | all29, mean over runs | {fmt(x['kappa_qw']['diff'])} [{fmt(x['kappa_qw']['ci95'][0])}, {fmt(x['kappa_qw']['ci95'][1])}] | "
                     f"{fmt(x['exact']['diff'])} [{fmt(x['exact']['ci95'][0])}, {fmt(x['exact']['ci95'][1])}] | | |")
    L += ["", "## Headline KG-derived bins (Ĉ_KG = bin/4) of the six KG-constructed benchmarks", "",
          "| rater | GRADE | STaRK | GraphRAG-Bench | 2WikiMultiHop | MetaQA | WebQSP | # external at ≥3 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, e in res["headline_KG"].items():
        b = e["bins"]
        L.append(f"| {n} | {b['GRADE']} | {b['STaRK']} | {b['GraphRAG-Bench']} | {b['2WikiMultiHop']} | {b['MetaQA']} | {b['WebQSP']} | {e.get('n_external_ge3', '')} |")
    L += ["", "## Per-item bins (final bin; runs 1-3 and median)", "",
          "| item | census | " + " | ".join(f"{m} {c}" for m in res["models"] for c in CONDS) + " |",
          "|---|---:|" + "---|" * (len(res["models"]) * len(CONDS))]
    census = json.load(open(HV / "reference_bins.json", encoding="utf-8"))["reference"]
    for i in ids:
        cells = []
        for m in res["models"]:
            for c in CONDS:
                vals = [runs[f"{m}_{c}_{r}"]["final"][i] if f"{m}_{c}_{r}" in runs else None for r in range(1, N_RUNS + 1)]
                md = medians.get((m, c), {}).get(i)
                cells.append(f"{'/'.join('-' if v is None else str(v) for v in vals)} ({'-' if md is None else md})")
        L.append(f"| {i}{' (anchor)' if i not in external else ''} | {census[i]} | " + " | ".join(cells) + " |")
    L += ["", "## Part B (condition A)", "", "| model | part | n | Q1 α (3 runs) | Q1 majority | Q2 majority rate | Q2 all agree | rule rate | majority vs rule |",
          "|---|---|---:|---:|---|---:|---:|---:|---|"]
    for m, e in res["partB"].items():
        for part in ("B1", "B2", "B3", "B4", "B5"):
            x = e[part]
            L.append(f"| {m} | {part} | {x['n']} | {fmt(x['Q1_alpha_runs'])} | {x['Q1_majority']} | {fmt(x['Q2_majority_rate'])} | "
                     f"{fmt(x['Q2_all_agree'])} | {fmt(x.get('rule_rate'))} | {x.get('majority_vs_rule', '-')} |")
    (OUT / "result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.prompts:
        write_prompts(load_instrument(), system_prompts())
    if a.pilot:
        pilot(a.models)
    if a.run:
        run_all(a.models)
    if a.score:
        score(a.models)


if __name__ == "__main__":
    main()
