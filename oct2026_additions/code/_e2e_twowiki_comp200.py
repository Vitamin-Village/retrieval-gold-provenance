"""End-to-end QA on the 2WikiMultiHop compositional split (comp200) with one Ollama reader.

Protocol: results/twowiki_comp200/PROTOCOL.md, Section 6.

Runs eval/e2e_qa_swap.py UNCHANGED (same prompt, temperature 0, seed 1, alias-aware EM, paired
statistics), restricting the retrieval conditions -- as eval/_e2e_third_reader.py does -- to the
aligned/independent KG vectors (kg_silver, kg_openie) of every retriever present in
external/2wiki/comp200/ (hybrid, PPR, hybrid-PPR and, once scored, HippoRAG 2), plus BM25,
closed-book and gold-only. Output and generation cache go to a new directory,
results/twowiki_comp200/e2e_qa/<model_tag>/, so no existing reader file is read or written.

Usage (from paper/): python eval/_e2e_twowiki_comp200.py --model llama3.1:8b
                     python eval/_e2e_twowiki_comp200.py --model qwen2.5:14b
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_qa_swap as E  # noqa: E402

_orig = E.condition_files


def core_conditions(bench, split):
    return [(n, f) for n, f in _orig(bench, split) if n in ("kg_silver", "kg_openie")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("llama3.1:8b", "qwen2.5:14b"))
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    E.condition_files = core_conditions
    sys.argv = [sys.argv[0], "--model", a.model, "--benchmarks", "2wiki", "--splits", "comp200",
                "--out", str(E.ROOT / "results" / "twowiki_comp200" / "e2e_qa")] + (["--score-only"] if a.score_only else [])
    sys.stdout.reconfigure(encoding="utf-8")
    E.main()
