"""End-to-end QA for the HippoRAG 2 swap vectors only (external/2wiki/*/_kgswap_hipporag2_*.json).

Runs eval/e2e_qa_swap.py unchanged (same prompt, reader settings, cache keys, scoring and paired
statistics) with its condition list restricted to the hipporag2 vectors, into a separate output
directory, so the shared results/e2e_qa/<model>/summary.* files (all benchmarks) are not rewritten
while other e2e runs use them. Seed <out>/<model_tag>/cache.jsonl with a copy of the shared cache
first so closed-book / oracle prompts are not regenerated; the new rows can later be appended to
the shared cache (same keys).

Usage (from paper/):
    python eval/_e2e_qa_hipporag2.py --model qwen2.5:14b --benchmarks 2wiki --out results/hipporag2_swap/e2e_qa
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_qa_swap as E  # noqa: E402

_all_conditions = E.condition_files


def _hipporag2_only(bench, split):
    return [(c, f) for c, f in _all_conditions(bench, split) if os.path.basename(f).startswith("_kgswap_hipporag2_")]


E.condition_files = _hipporag2_only

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    E.main()
