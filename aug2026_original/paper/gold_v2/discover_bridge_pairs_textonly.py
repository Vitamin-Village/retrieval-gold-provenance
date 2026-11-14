# -*- coding: utf-8 -*-
"""M0': TEXT-ONLY bridge-seeded cross-doc pair discovery (KG-free).

WHY this module (2026-06-01):
  The original M0 (paper/gold/discover_bridge_pairs.py) walks the proposed
  evaluation KG to choose which papers to pair and which bridge surface to
  seed. Under the provenance-circularity metric C(B) that makes every seeded
  multihop record KG-seeded (bridge == KG node), contradicting the benchmark's
  acyclic C=0 label; the 6 such records were removed from qa_independent_v1 by
  scripts/clean_remove_kg_seeded_multihop.py. This module replaces the KG
  sampling hint with corpus-text term mining so the ENTIRE discovery path is
  text-self-sufficient. It lives in paper/gold_v2/, so the test_no_kg_imports
  CI guard applies to it: it imports NO KG loader, ever.

WHAT it does:
  1. Mines candidate bridge surfaces from passage text only: token n-grams
     (1 to 4 tokens) over each paper's passage text.
  2. Counts the corpus fan-out of each candidate exactly as the original M0
     does -- the number of distinct corpus papers whose concatenated lowercase
     passage text contains the surface -- and keeps candidates whose fan-out
     lies in [MIN_FANOUT, MAX_FANOUT] (INV-5).
  3. Applies the same fixed structural validity rules as the original
     (denylist, minimum length), plus text-mining replacements for the KG-type
     pre-filter the original had (rules R1-R4 below).
  4. Emits (domain, paper_a, paper_b, bridge_surface, bridge_type, fanout)
     rows in the EXACT schema of paper/gold/bridge_pairs.jsonl, pairing papers
     within the same domain among the fan-out hits.

Deterministic: same passages.jsonl -> byte-identical output (stable sorts, no
randomness, no model call). ASCII-only console output (Windows cp949 safe).
DO NOT add hyperparameter tuning: every threshold below is a FIXED structural
rule with an inline justification, duplicated -- not imported -- from the
original module to respect the acyclicity boundary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASSAGES = PROJECT_ROOT / "paper" / "gold_v2" / "passages.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "gold_v2" / "bridge_pairs_textonly.jsonl"

# ---------------------------------------------------------------------------
# Fixed structural rules. MIN_FANOUT / MAX_FANOUT / MIN_BRIDGE_CHARS /
# BRIDGE_DENYLIST are duplicated VERBATIM from paper/gold/discover_bridge_pairs.py
# (which cannot be imported here: it imports the KG loaders, and this module
# must stay inside the gold_v2 no-KG-imports boundary).
# ---------------------------------------------------------------------------

# INV-5 fan-out ceiling (empirical cut: "computer vision" spans 10 papers,
# "defect detection" 6 -> both rejected; specific terms pass).
MAX_FANOUT = 4
# A genuine bridge must link AT LEAST two papers.
MIN_FANOUT = 2
# A 1-2 char string ("Fe") matches spuriously in the substring fan-out count.
MIN_BRIDGE_CHARS = 3

BRIDGE_DENYLIST = frozenset(
    {
        # NLP / language-method common words.
        "nlp", "natural language processing", "language model", "language models",
        "large language model", "large language models", "llm", "llms",
        # Classic ML / statistical-learning methods.
        "random forest", "random forests", "support vector machine",
        "support vector machines", "svm", "decision tree", "decision trees",
        "gradient boosting", "xgboost", "logistic regression",
        "linear regression", "k-means", "clustering", "regression",
        "classification", "machine learning", "deep learning",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "transfer learning",
        # Generic neural-network method words.
        "neural network", "neural networks", "deep neural network",
        "deep neural networks", "artificial neural network",
        "convolutional neural network", "convolutional neural networks", "cnn",
        "cnns", "recurrent neural network", "recurrent neural networks", "rnn",
        "lstm", "transformer", "transformers", "attention mechanism",
        "encoder", "decoder", "autoencoder", "gan",
        "generative adversarial network", "generative adversarial networks",
        # Generic computer-vision / detection method words.
        "computer vision", "image processing", "object detection",
        "defect detection", "anomaly detection", "image classification",
        "image segmentation", "semantic segmentation", "feature extraction",
        "pattern recognition", "data augmentation", "edge detection",
        # Generic robotics / sensing method words.
        "slam", "simultaneous localization and mapping", "sensor fusion",
        "point cloud", "lidar",
        # Generic process / abstraction nouns.
        "optimization", "simulation", "framework", "algorithm", "model",
        "method", "approach", "technique", "system", "analysis", "dataset",
        "accuracy", "performance", "efficiency", "automation",
    }
)

# R1: edge-token stopwords. The original M0 never produced stopword-edged
# bridges because KG node names are typed entity surfaces; text mining has no
# typing, so n-grams whose FIRST or LAST token is a function word ("of the
# process") are fragments, not entities. Fixed closed-class list, not tuned.
EDGE_STOPWORDS = frozenset(
    "a an the of in on at to for with by from and or as is are was were be been"
    " this that these those its their our your his her it we they i you he she"
    " which who whom whose where when while than then there here such same other"
    " each both all any some no not non more most less least very can could may"
    " might must shall should will would do does did done has have had having"
    " into onto over under between among during after before above below per"
    " via using used use based also however therefore thus hence et al e g i e".split()
)

# R2: at least one alphabetic token of length >= 4. A surface made only of
# short tokens ("x 2", "ab cd") carries no entity content; KG typing used to
# guarantee this, text mining must enforce it explicitly.
_ALPHA4 = re.compile(r"[a-z]{4,}")

# R3: single-token bridges must be >= 6 chars. The original accepted short
# typed tokens ("chromium" passed via KG type Material); without typing, short
# unigrams ("layer", "steel") are domain common words. 6 chars keeps specific
# technical unigrams ("austenite", "photoresist") and cuts the generic short
# ones. Fixed rule, mirrors the MIN_BRIDGE_CHARS rationale one level up.
MIN_UNIGRAM_CHARS = 6

# R4: purely numeric/punctuation surfaces are never entities.
_NON_ENTITY = re.compile(r"^[0-9 .,/-]+$")

_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]*")
MAX_NGRAM_TOKENS = 4


def load_passages(path: Path) -> List[dict]:
    """Load passages.jsonl (passage_id / paper_id / domain / text)."""
    rows: List[dict] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_paper_domain_index(passages: Sequence[dict]) -> Dict[str, str]:
    """paper_id -> domain (first passage's domain wins; all share it)."""
    out: Dict[str, str] = {}
    for p in passages:
        pid = str(p.get("paper_id") or "")
        if pid and pid not in out:
            out[pid] = str(p.get("domain") or "unknown")
    return out


def is_valid_bridge_surface(surface: str) -> bool:
    """Fixed structural validity rules for a mined bridge surface (R1-R4 +
    the duplicated MIN_BRIDGE_CHARS / BRIDGE_DENYLIST rules)."""
    s = surface.strip()
    if len(s) < MIN_BRIDGE_CHARS:
        return False
    if s in BRIDGE_DENYLIST:
        return False
    if _NON_ENTITY.match(s):
        return False
    toks = s.split(" ")
    if toks[0] in EDGE_STOPWORDS or toks[-1] in EDGE_STOPWORDS:
        return False
    if not _ALPHA4.search(s):
        return False
    if len(toks) == 1 and len(s) < MIN_UNIGRAM_CHARS:
        return False
    return True


def _paper_ngrams(text: str) -> FrozenSet[str]:
    """Unique candidate n-grams (1..MAX_NGRAM_TOKENS tokens) of one paper.

    Tokenization is line-local so n-grams never span passage boundaries (the
    per-paper text is passage texts joined by newlines). Cheap validity rules
    are applied here so the inverted index stays small.
    """
    seen = set()
    for line in text.split("\n"):
        toks = _TOKEN.findall(line)
        n_toks = len(toks)
        for i in range(n_toks):
            for n in range(1, MAX_NGRAM_TOKENS + 1):
                if i + n > n_toks:
                    break
                g = " ".join(toks[i:i + n])
                if g in seen:
                    continue
                if is_valid_bridge_surface(g):
                    seen.add(g)
    return frozenset(seen)


def mine_bridges(
    paper_text: Mapping[str, str],
) -> Dict[str, Tuple[str, ...]]:
    """Mine bridge surfaces with corpus fan-out in [MIN_FANOUT, MAX_FANOUT].

    Returns {surface: sorted tuple of paper_ids containing it}. The fan-out is
    counted from per-paper token n-gram sets, which equals the original M0's
    word-boundary containment count for token-aligned surfaces. Candidates that
    exceed MAX_FANOUT papers are marked dead immediately (memory guard; the
    verdict is identical to counting fully and filtering).
    """
    DEAD = ("",)  # sentinel: fan-out exceeded MAX_FANOUT
    index: Dict[str, tuple] = {}
    for pid in sorted(paper_text):
        for g in _paper_ngrams(paper_text[pid]):
            cur = index.get(g)
            if cur is DEAD:
                continue
            if cur is None:
                index[g] = (pid,)
            elif len(cur) < MAX_FANOUT:
                index[g] = cur + (pid,)
            else:
                index[g] = DEAD
    out: Dict[str, Tuple[str, ...]] = {}
    for g, pids in index.items():
        if pids is DEAD or len(pids) < MIN_FANOUT:
            continue
        out[g] = tuple(sorted(pids))
    return _prefer_maximal(out)


def _prefer_maximal(
    bridges: Dict[str, Tuple[str, ...]]
) -> Dict[str, Tuple[str, ...]]:
    """Maximal-phrase preference: drop a surface when a strictly longer surface
    with the SAME paper set contains it as a token subphrase ("layer
    deposition" is subsumed by "atomic layer deposition" when both occur in
    exactly the same papers). Fixed structural rule: the longer phrase is the
    more specific entity; the nested fragment adds no new pairing information.
    """
    by_papers: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for g, pids in bridges.items():
        by_papers[pids].append(g)
    keep: Dict[str, Tuple[str, ...]] = {}
    for pids, group in by_papers.items():
        group_sorted = sorted(group, key=lambda s: (-len(s), s))
        kept_group: List[str] = []
        for g in group_sorted:
            padded = " %s " % g
            if any((" %s " % k).find(padded) >= 0 for k in kept_group):
                continue
            kept_group.append(g)
        for g in kept_group:
            keep[g] = pids
    return keep


def emit_pairs(
    bridges: Mapping[str, Tuple[str, ...]],
    paper_domain: Mapping[str, str],
) -> List[dict]:
    """Emit same-domain unordered paper pairs for every surviving bridge, in
    the exact row schema of paper/gold/bridge_pairs.jsonl."""
    rows: List[dict] = []
    for surface in sorted(bridges):
        pids = bridges[surface]
        by_dom: Dict[str, List[str]] = defaultdict(list)
        for pid in pids:
            by_dom[paper_domain.get(pid, "unknown")].append(pid)
        for dom in sorted(by_dom):
            papers = sorted(by_dom[dom])
            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    rows.append(
                        {
                            "bridge_surface": surface,
                            "bridge_type": "TextTerm",
                            "domain": dom,
                            "fanout": len(pids),
                            "paper_a": papers[i],
                            "paper_b": papers[j],
                        }
                    )
    rows.sort(
        key=lambda r: (r["domain"], r["bridge_surface"], r["paper_a"], r["paper_b"])
    )
    return rows


def discover(passages: Sequence[dict]) -> List[dict]:
    """Full text-only discovery: passages -> bridge pair rows."""
    by_paper: Dict[str, List[str]] = defaultdict(list)
    for p in passages:
        pid = str(p.get("paper_id") or "")
        if pid:
            by_paper[pid].append(str(p.get("text") or ""))
    paper_text = {pid: "\n".join(parts).lower() for pid, parts in by_paper.items()}
    paper_domain = build_paper_domain_index(passages)
    bridges = mine_bridges(paper_text)
    return emit_pairs(bridges, paper_domain)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M0' text-only bridge-pair discovery (KG-free)"
    )
    parser.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    passages = load_passages(Path(args.passages))
    rows = discover(passages)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True, sort_keys=True))
            f.write("\n")
    by_dom: Dict[str, int] = defaultdict(int)
    n_bridges = len({r["bridge_surface"] for r in rows})
    for r in rows:
        by_dom[r["domain"]] += 1
    print(
        "M0' text-only done: {0} pair rows / {1} bridges -> {2}".format(
            len(rows), n_bridges, out
        )
    )
    for dom in sorted(by_dom):
        print("  {0}: {1}".format(dom, by_dom[dom]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
