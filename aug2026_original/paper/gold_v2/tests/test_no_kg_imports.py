# -*- coding: utf-8 -*-
"""CI acyclicity guard: gold_v2 modules must NOT import KG-side symbols.

The independent benchmark's non-circularity rests on the generator and passage
builder never touching the proposed KG or the cyclic qa_from_kg pipeline. This
static test parses every gold_v2 module's AST and fails if any forbidden
module/symbol is imported, mechanically preventing circular regressions
(design section 0 + section 8 LOCKED DECISIONS).

Scan targets: ALL .py files under paper/gold_v2/ (excluding tests/).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Forbidden patterns.
#
# We match against the full dotted import name using three rules:
#   1. exact match          e.g. "qa_from_kg"
#   2. prefix + "."         e.g. "proposed_kg.loader"  (sub-module)
#   3. any segment equals   e.g. "paper.proposed_kg_v2.util" -- note: this
#                           would also catch "proposed_kg_v2" (suffix variant)
#
# Rule 3 is implemented as a segment split so "proposed_kg_v2" (a suffix
# variant like proposed_kg_anything) is also caught, while a module whose
# name merely CONTAINS the root string inside a longer word (e.g.
# "non_proposed_kg_adapter" -> segment "non_proposed_kg_adapter") is NOT a
# false positive because it fails the exact-segment equality check.
#
# Forbidden roots:
#   "prototype.baselines"  - old cyclic BM25/dense baseline package
#   "proposed_kg"          - proposed KG builder/loader AND any suffix variants
#                            (proposed_kg_v2, proposed_kg_anything, ...)
#   "qa_from_kg"           - the original cyclic gold builder
# ---------------------------------------------------------------------------

FORBIDDEN_ROOTS = ("prototype.baselines", "proposed_kg", "qa_from_kg")

# Path to the gold_v2 package root (parent of this tests/ directory).
GOLD_V2_ROOT = Path(__file__).resolve().parents[1]

# All gold_v2 production modules (exclude tests/ so we do not self-check).
def _gold_v2_modules() -> List[Path]:
    return [
        p
        for p in GOLD_V2_ROOT.rglob("*.py")
        if "tests" not in p.parts
    ]


# ---------------------------------------------------------------------------
# AST helpers.
# ---------------------------------------------------------------------------


def _imported_names(source: str):
    """Yield every dotted module name referenced by import statements."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            yield mod
            for alias in node.names:
                # capture `from X import qa_from_kg` style too.
                yield (mod + "." + alias.name) if mod else alias.name


def _is_forbidden(name: str) -> bool:
    """True if `name` matches any forbidden root, including suffix variants.

    Matching rules (applied per forbidden root R):
      (a) exact:   name == R
      (b) prefix:  name starts with R + "."   (sub-module of R)
      (c) segment: any dot-segment of name == a dot-segment prefix of R, OR
                   any segment starts with the leaf segment of R
                   e.g. "proposed_kg_v2" caught because segment "proposed_kg_v2"
                   starts with "proposed_kg" (the leaf of that root).

    Rule (c) prevents suffix variants like proposed_kg_v2 from slipping through
    while avoiding false positives on unrelated module names.
    """
    for root in FORBIDDEN_ROOTS:
        # (a) exact match
        if name == root:
            return True
        # (b) sub-module: name is within the root package
        if name.startswith(root + "."):
            return True
        # (c) segment-level: split both and check whether any segment of `name`
        #     starts with the LAST segment of `root` (catches suffix variants
        #     like proposed_kg_v2 where leaf="proposed_kg").
        root_leaf = root.split(".")[-1]
        for seg in name.split("."):
            if seg == root_leaf or seg.startswith(root_leaf + "_"):
                return True
        # Also catch `name` ending in ".root" (from X import root style after
        # module concatenation yields "X.root").
        if name.endswith("." + root):
            return True
    return False


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_all_gold_v2_modules_have_no_kg_imports():
    """Scan every production module under paper/gold_v2/ for KG imports."""
    modules = _gold_v2_modules()
    assert modules, "no gold_v2 production modules found -- check path"
    failures: List[str] = []
    for path in modules:
        source = path.read_text(encoding="utf-8")
        offenders = [n for n in _imported_names(source) if _is_forbidden(n)]
        if offenders:
            failures.append("{0}: {1}".format(path.name, offenders))
    assert not failures, "forbidden KG imports found:\n" + "\n".join(failures)


def test_guard_detects_forbidden_import_in_synthetic_source():
    bad_src = "from prototype.baselines import bm25\nimport os\n"
    offenders = [n for n in _imported_names(bad_src) if _is_forbidden(n)]
    assert "prototype.baselines" in offenders


def test_guard_detects_qa_from_kg_import():
    bad_src = "import qa_from_kg\n"
    offenders = [n for n in _imported_names(bad_src) if _is_forbidden(n)]
    assert "qa_from_kg" in offenders


def test_guard_detects_proposed_kg_suffix_variant():
    # proposed_kg_v2 is a suffix variant that the old guard missed.
    bad_src = "import proposed_kg_v2\nfrom proposed_kg_anything import loader\n"
    offenders = [n for n in _imported_names(bad_src) if _is_forbidden(n)]
    assert any("proposed_kg" in o for o in offenders), (
        "suffix variant proposed_kg_v2 not caught: {0}".format(offenders)
    )


def test_guard_no_false_positive_on_unrelated_name():
    # "non_kg_module" and "gold_v2_utils" must NOT be flagged.
    clean_src = "import non_kg_module\nfrom gold_v2_utils import helper\n"
    offenders = [n for n in _imported_names(clean_src) if _is_forbidden(n)]
    assert not offenders, "false positive: {0}".format(offenders)
