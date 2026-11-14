"""Third, larger reader (Mistral-Small-24B) on the core end-to-end conditions of 2WikiMultiHop.

Runs eval/e2e_qa_swap.py unchanged (same prompt, temperature 0, alias-aware EM, paired statistics)
but restricts the retrieval conditions to the frozen aligned/independent KG vectors of every retriever
(hybrid, hybrid-PPR, PPR, HippoRAG 2), BM25, closed-book and gold-only, so that the reader-size
limitation can be checked without re-running the dose, vocabulary, and BGE-M3 variants.
GraphRAG-Bench is excluded because its answers are graded by the same Mistral-Small-24B judge.
Outputs go to results/e2e_qa/mistral_small_24b/ (a new reader directory; no other reader's files are touched).
Usage (from paper/): python eval/_e2e_third_reader.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_qa_swap as E  # noqa: E402

_orig = E.condition_files


def core_conditions(bench, split):
    return [(n, f) for n, f in _orig(bench, split) if n in ("kg_silver", "kg_openie")]


E.condition_files = core_conditions
sys.argv = [sys.argv[0], "--model", "mistral-small:24b", "--benchmarks", "2wiki"]
E.main()
