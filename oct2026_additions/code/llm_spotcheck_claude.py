# -*- coding: utf-8 -*-
"""AI spot-check of the GraphRAG-Bench LM judge (2026-09-29).

Three Claude models label the same 100 spot-check items as the human rater of Appendix C (judge
validation): question, reference answer, and candidate answer, one request per item, with the
rater guide human_validation/judge_spotcheck_v3/GUIDE_v3.md verbatim as the system prompt and no
other context. The labels are LM labels and are reported as such, never as human labels.

No server-side fallback is used, so every label comes from the model named in the output; a
refusal or an unparsable reply is recorded and scored as UNSURE.

Outputs in human_validation/llm_spotcheck_v3/:
  raw.jsonl                      one line per (model, item): model id returned by the API, stop
                                 reason, full reply text, parsed verdict, usage, UTC timestamp
  returned/rater_{1,2,3}_spotcheck.xlsx   the blank v3 sheets filled with each model's verdicts
                                 (rater 1 = claude-opus-5, 2 = claude-sonnet-5, 3 = claude-haiku-4-5)
  result_v3.json / result_v3.md  judge agreement, scored by ../judge_spotcheck_v3/score_spotcheck_v3.py
  agreement_with_v1_human.json   each model's agreement with the earlier human rater

Usage (from paper/, in a Python environment with the anthropic SDK and openpyxl; ANTHROPIC_API_KEY set):
  python eval/llm_spotcheck_claude.py
Re-running reuses raw.jsonl and only requests missing (model, item) pairs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from openpyxl import load_workbook

PAPER = Path(__file__).resolve().parents[1]
SPOT = PAPER / "human_validation" / "judge_spotcheck_v3"
OUT = PAPER / "human_validation" / "llm_spotcheck_v3"
RAW = OUT / "raw.jsonl"
MODELS = {1: "claude-opus-5", 2: "claude-sonnet-5", 3: "claude-haiku-4-5"}
VERDICT = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT|UNSURE)", re.I)
INSTRUCTION = ("Apply the guide to this row. Compare the two texts only. End your reply with exactly one line of the form "
               "`VERDICT: CORRECT`, `VERDICT: INCORRECT`, or `VERDICT: UNSURE`.")


def api_key() -> str | None:
    for envp in (PAPER / ".env", PAPER.parent / ".env"):
        if envp.exists():
            for line in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("ANTHROPIC_API_KEY")


def blank(r: int) -> Path:
    """Blank v3 sheet of rater r, under either of its two file names (same bytes)."""
    for name in (f"rater_{r}_spotcheck.xlsx", f"rater_{r}_spotcheck_og.xlsx"):
        if (SPOT / "sheets" / name).exists():
            return SPOT / "sheets" / name
    raise FileNotFoundError(f"no blank spot-check sheet for rater {r}")


def items() -> dict[str, dict]:
    ws = load_workbook(blank(1), data_only=True)["SpotCheck"]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    out = {}
    for r in rows[1:]:
        d = dict(zip(head, r))
        out[d["item_id"]] = {"question": d["question"], "reference": d["reference_answer"],
                             "candidate": d["candidate_answer (model output, verbatim)"]}
    assert len(out) == 100, len(out)
    return out


def main():
    key = api_key()
    if not key:
        sys.exit("no ANTHROPIC_API_KEY found in paper/.env, ../.env, or the environment")
    client = anthropic.Anthropic(api_key=key, max_retries=5)
    guide = (SPOT / "GUIDE_v3.md").read_text(encoding="utf-8")
    its = items()
    OUT.mkdir(parents=True, exist_ok=True)
    done = {}
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            done[(x["model_requested"], x["item_id"])] = x
    lock = threading.Lock()

    def run(model: str, iid: str):
        it = its[iid]
        user = (f"Question: {it['question']}\n\nReference answer: {it['reference']}\n\n"
                f"Candidate answer: {it['candidate']}\n\n{INSTRUCTION}")
        msg = client.messages.create(model=model, max_tokens=16000, system=guide,
                                     messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in msg.content if b.type == "text")
        m = VERDICT.findall(text)
        verdict = m[-1].upper() if (m and msg.stop_reason != "refusal") else "UNSURE"
        rec = {"item_id": iid, "model_requested": model, "model_returned": msg.model, "stop_reason": msg.stop_reason,
               "verdict": verdict, "parsed": bool(m), "text": text,
               "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens},
               "request_id": getattr(msg, "_request_id", None), "utc": datetime.now(timezone.utc).isoformat()}
        with lock:
            with open(RAW, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(model, iid)] = rec
        return rec

    todo = [(m, i) for m in MODELS.values() for i in its if (m, i) not in done]
    print(f"{len(todo)} requests to run", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, rec in enumerate(ex.map(lambda a: run(*a), todo), 1):
            if k % 25 == 0:
                print(f"  {k}/{len(todo)}", flush=True)

    # fill each model's copy of the blank sheet (its own item order), then score with the v3 scorer
    ret = OUT / "returned"
    ret.mkdir(exist_ok=True)
    for r, model in MODELS.items():
        wb = load_workbook(blank(r))
        ws = wb["SpotCheck"]
        head = [c.value for c in ws[1]]
        ci, cv, cn = head.index("item_id") + 1, head.index("your_verdict") + 1, head.index("note (optional)") + 1
        for row in range(2, ws.max_row + 1):
            rec = done[(model, ws.cell(row, ci).value)]
            ws.cell(row, cv).value = rec["verdict"]
            ws.cell(row, cn).value = None if rec["parsed"] and rec["stop_reason"] != "refusal" else f"unparsed/{rec['stop_reason']}"
        info = wb.create_sheet("Provenance")
        info.append(["field", "value"])
        info.append(["rater_type", "LM (not human)"])
        info.append(["model", model])
        info.append(["prompt", "GUIDE_v3.md as system prompt; one request per item; see eval/llm_spotcheck_claude.py"])
        wb.save(ret / f"rater_{r}_spotcheck.xlsx")
    env = dict(os.environ, SPOT_RETURNED=str(ret), SPOT_OUT=str(OUT))
    subprocess.run([sys.executable, str(SPOT / "score_spotcheck_v3.py")], env=env, check=True)

    sealed = json.load(open(SPOT / "sealed_key_v3.json", encoding="utf-8"))["items"]
    agree = {}
    for model in MODELS.values():
        pairs = [(done[(model, i)]["verdict"], sealed[i]["v1_human"]) for i in its if sealed[i].get("v1_human")]
        pairs = [(a, b) for a, b in pairs if a != "UNSURE" and b != "UNSURE"]
        agree[model] = {"n": len(pairs), "agreement_with_v1_human": sum(a == b for a, b in pairs) / len(pairs),
                        "unsure_or_unparsed": sum(done[(model, i)]["verdict"] == "UNSURE" for i in its),
                        "models_returned": sorted({done[(model, i)]["model_returned"] for i in its})}
    json.dump(agree, open(OUT / "agreement_with_v1_human.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(agree, indent=1))


if __name__ == "__main__":
    main()
