"""Deterministic lexical rules used to check per-unit gold realizations (Appendix B.9 and B.5).

Identical to the helper functions of the rater-package builder; kept as a standalone module so the
per-unit scoring can be rerun from the released files alone.
"""
from __future__ import annotations

import re
import string


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def contains(hay: str, needle: str) -> bool:
    return bool(needle) and norm(needle) in norm(hay)


def content_tokens(s: str):
    stop = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "is", "are", "was", "were", "by", "with", "as", "at", "from", "that", "this", "it", "its", "be", "or"}
    return [t for t in norm(s).split() if t not in stop]


def majority_token_rule(relation: str, sentence: str) -> bool:
    toks = content_tokens(relation)
    if not toks:
        return False
    sent = set(norm(sentence).split())
    return sum(t in sent for t in toks) / len(toks) >= 0.5


def near_verbatim(relation: str, sentence: str) -> bool:
    r, s = norm(relation), norm(sentence)
    if not r or not s:
        return False
    if r in s:
        return True
    rt, st = set(r.split()), set(s.split())
    return len(rt & st) / max(1, len(rt | st)) >= 0.8
