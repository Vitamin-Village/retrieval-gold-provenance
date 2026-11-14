# Reproduction environment

## What is and is not pinned

No lockfile was recorded for every experiment.
Accordingly, this release does not claim that the dependency versions below
are the exact versions used for every reported run. `requirements.txt` records
the exact core versions used for the final offline checks, while
`requirements-optional.txt` distinguishes locally verified optional versions
from dependencies whose exact version was not recorded.

## Final package verification environment

- OS: Microsoft Windows 11 Pro 64-bit, version 10.0.26200.
- Python: CPython 3.9.12.
- CPU: Intel Core i7-12700, 20 logical processors.
- RAM visible to the verifier: 95.7 GiB.
- GPU visible to the verifier: NVIDIA GeForce RTX 3090, 24,576 MiB, driver
  591.86.

These hardware facts describe the final package checks on 2026-07-15. They
do not imply that every experiment ran on this machine.
The reliability recomputations and acyclicity guard are CPU-only. The
deterministic retrieval mode is selected with
`RAG_V3_DENSE_BACKEND=deterministic`; learned dense backends require the
optional ML stack and may require a compatible GPU/CUDA installation.

## API and local-model runs

Cloud calls are lazy and require the relevant environment variable:
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`. Local open-weight
calls require an Ollama daemon and the model tag named by the corresponding
artifact. Hosted-model outputs may drift even when a model alias is unchanged.

The single-hop verifier spot-check records Gemini `gemini-2.5-flash` as the
primary verifier, Claude Haiku 4.5 as the blind second model, and
`mistral-small:24b` as the generator. API-client package versions and raw
second-rater response text are not available. The two-hop data card
records `gemini-2.5-flash` and `qwen2.5:14b` as the two verifiers. Exact model
IDs for the two primary census raters were not recorded; see
`census/PRIMARY_RATER_PROVENANCE.md`.

## Checks on a fresh copy

From this folder (`aug2026_original/`):

    python -m pip install -r requirements.txt
    python benchmark/singlehop_verifier_spotcheck/recompute_kappa.py
    python -m pytest -q paper/gold_v2/tests/test_no_kg_imports.py

All benchmark construction CLIs can be inspected without making API calls;
for example:

    python -m paper.gold_v2.generate_qa_candidates --help
    python -m paper.gold_v2.verify_answer_papers --help
    python -m paper.gold_v2.verify_multihop_bridge --help
    python -m paper.gold_v2.leakage_gate --help
    python -m paper.gold_v2.freeze_dataset --help
    python paper/gold_v2/scripts/run_twohop_slice.py --help

The full KG-swap rerun additionally needs the optional retrieval dependencies
and the released data paths described in `README.md` in this folder.
