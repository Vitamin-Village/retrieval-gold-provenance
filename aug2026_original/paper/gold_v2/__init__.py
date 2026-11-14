# -*- coding: utf-8 -*-
"""Independent (acyclic) QA benchmark build pipeline (gold_v2).

This package builds the benchmark gold set ONLY from raw corpus text and a
non-extractive LLM panel. It must NEVER import the proposed KG, the baseline
retrievers, or the cyclic qa_from_kg module. See
paper/analysis/independent_benchmark_design_2026-05-29.md (section 8 LOCKED
DECISIONS) for the single source of truth.
"""
