# -*- coding: utf-8 -*-
"""S2: QA candidate generator for the independent (acyclic) QA benchmark.

Inputs : S1 passages (paper/gold_v2/passages.jsonl).
Process: a GENERATOR backend reads passage text and emits question / answer /
         evidence span. The live CLI defaults to local Mistral-Small-24B,
         matching the generator recorded in the frozen single-hop rows; Claude
         Sonnet remains an explicit alternative. The answer is derived ONLY
         from passage text; the KG is NEVER provided in the prompt.
Outputs: paper/gold_v2/qa_candidates.jsonl with rows:
         {question, answer, answer_paper_ids, evidence_spans, qtype,
          hop_count, generator}.

Four question types are forced (design section 3 tiers + D-E):
  - factual         (easy,   1 hop)
  - paraphrase_alias (medium, 1 hop)
  - comparison      (medium, 1 hop)
  - multihop_bridge (hard,   2 hop, two same-domain papers)

Per D-E: even easy/factual must paraphrase at least once; the answer surface
form must NOT appear verbatim in the question.

ACYCLICITY (CI guard, see test_no_kg_imports): this module must NOT import
prototype.baselines, proposed_kg, or qa_from_kg. The KG is never referenced.

NO real API call is made by the unit tests (a stub generator is injected).
Live generation requires the credentials or local service for the selected
backend and is run separately by the main operator. DO NOT add hyperparameter
tuning; structural rules only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

_PACKAGE_ROOT_FOR_IMPORT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT_FOR_IMPORT)

# Reuse S1's ASCII normaliser so model-emitted answer/evidence are normalised
# the SAME way as the passage text before any substring/word-boundary gate.
# Without this, unit-bearing answers (e.g. "227 uohm cm" emitted with Greek
# mu/ohm) never word-boundary-match the already-normalised passage, dropping
# every such candidate. build_passages is the S1 module (no KG import).
from paper.gold_v2.build_passages import ascii_normalize

# ---------------------------------------------------------------------------
# Intentionally NO import of:
#   prototype.baselines, proposed_kg, qa_from_kg
# A static CI test (test_no_kg_imports.py) asserts these symbols never appear.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASSAGES = PROJECT_ROOT / "paper" / "gold_v2" / "passages.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "gold_v2" / "qa_candidates.jsonl"

# Output-size budgets -- these are I/O capacity constants, NOT quality
# hyperparameters. Single-hop JSON has 3 fields (question/answer/evidence);
# 512 tokens is sufficient. Multihop JSON has 5 fields including two evidence
# spans and a bridge entity; 1024 tokens prevents mid-JSON truncation that
# would cause parse failure and candidate discard.
GEN_MAX_TOKENS_SINGLE = 512   # single-hop: 3-field JSON
GEN_MAX_TOKENS_MULTIHOP = 1024  # multihop: 5-field JSON (question/answer/bridge/evidence_a/evidence_b)
# Backward-compat alias used by the CLI and tests that do not distinguish.
GEN_MAX_TOKENS = GEN_MAX_TOKENS_SINGLE

# Question types (design section 3). Single-hop set + the multihop type.
QTYPE_FACTUAL = "factual"
QTYPE_PARAPHRASE = "paraphrase_alias"
QTYPE_COMPARISON = "comparison"
QTYPE_MULTIHOP = "multihop_bridge"

SINGLE_HOP_QTYPES = (QTYPE_FACTUAL, QTYPE_PARAPHRASE, QTYPE_COMPARISON)
ALL_QTYPES = SINGLE_HOP_QTYPES + (QTYPE_MULTIHOP,)

# hop_count per qtype (structural, not tunable).
QTYPE_HOP_COUNT = {
    QTYPE_FACTUAL: 1,
    QTYPE_PARAPHRASE: 1,
    QTYPE_COMPARISON: 1,
    QTYPE_MULTIHOP: 2,
}

# Hard-tier ratio for the per-domain cap (design section 3/4:
# easy:medium:hard = 0.4:0.4:0.2). This is the benchmark's STRUCTURAL tier
# mix, not a tuned hyperparameter -- it is fixed by the design doc. The
# per-domain hard cap (run_kappa_pilot --max-qa-per-domain) keeps multihop
# (hard tier) to this fraction of the cap, so that the O(n^2) multihop pair
# space does not dominate generation.
HARD_TIER_RATIO = 0.2

# Over-generation margin applied to the per-domain hard cap during GENERATION
# (not just post-hoc selection). When max_qa_per_domain is set, generation in
# a domain stops early once it has collected OVERGEN_FACTOR x the cap of each
# tier, so the post-cap selector still has a small surplus to choose from
# without paying for the full O(n^2) multihop pair space.
# Design spec (section 6) budgets "~1.6x overgen due to discard rate" (200
# target -> 320 generated). We round up to 2 for a clean integer multiplier
# and to give a small safety margin above the spec's 1.6x estimate -- a
# structural budget guard derived from the spec, not a tuned quality knob.
# It bounds the O(n^2) multihop pair space.
OVERGEN_FACTOR = 2


# ---------------------------------------------------------------------------
# Prompts (ASCII only; deterministic; NO KG provided).
#
# The prompts explicitly restrict questions to
# manufacturing-technical content (process / material / equipment / defect /
# parameter / measurement) and ban paper metadata and generic background.
# A few-shot BAD/GOOD block makes the constraint concrete for the LLM.
# ---------------------------------------------------------------------------

# Shared preamble injected into every system prompt so the content constraint
# applies to both single-hop and multihop generation paths.
_DOMAIN_CONSTRAINT = (
    "The passages come from manufacturing-domain research papers "
    "(semiconductor, construction, steel, or textile). "
    "Your question MUST be about manufacturing-technical content: "
    "a process step, material property, equipment specification, "
    "defect or failure mode, process parameter, or measured quantity. "
    "FORBIDDEN - do NOT write questions about: "
    "(a) paper metadata (author names, affiliations, corresponding author, "
    "venue, publication date, funding, acknowledgements); "
    "(b) generic background knowledge unrelated to the domain "
    "(e.g. AlexNet, ImageNet, generic ML architectures, mathematical "
    "definitions); "
    "(c) paper structure (section numbers, related work summaries). "
    "If the passage contains ONLY metadata or generic background with no "
    "domain-specific technical fact, output the single word SKIP and nothing "
    "else - do NOT produce a JSON object."
)

# Few-shot examples (ASCII; illustrative, not tuned).
_FEW_SHOT_BLOCK = (
    "Examples of FORBIDDEN questions (output SKIP for these passages):\n"
    "  BAD: 'Which institution is the corresponding author affiliated with?'"
    " -> answer would be an affiliation string - metadata, SKIP.\n"
    "  BAD: 'What neural network architecture was first introduced for"
    " large-size datasets?' -> answer is AlexNet - generic ML background,"
    " SKIP.\n"
    "  BAD: 'What does Section 3 describe?' -> paper structure, SKIP.\n"
    "Examples of GOOD questions (manufacturing-technical):\n"
    "  GOOD (factual): 'What property of the coating layer is measured in"
    " this study?' -> answer: 'corrosion resistance' from passage.\n"
    "  GOOD (paraphrase): 'What parameter governs the thickness uniformity"
    " of the deposited film?' -> answer surface form differs from question"
    " wording.\n"
    "  GOOD (comparison): 'How does the annealing temperature in Process A"
    " compare with that in Process B as described in the passage?'"
)

GENERATOR_SYSTEM_PROMPT = (
    "You are a careful benchmark author for a manufacturing-domain retrieval "
    "benchmark. You read a research paper passage and write a single QA item "
    "whose answer is derivable ONLY from the given passage text. "
    "You are given NO knowledge graph and must not invent facts.\n"
    + _DOMAIN_CONSTRAINT + "\n"
    + _FEW_SHOT_BLOCK + "\n"
    "Output rules when writing a question: "
    "(1) the answer must be a short span quoted verbatim from the passage; "
    "(2) the answer surface form must NOT appear verbatim in the question - "
    "paraphrase or use an alias/synonym instead; "
    "(3) reply with a single JSON object and nothing else, using exactly the "
    "keys: question, answer, evidence. "
    "'evidence' is the exact passage substring that supports the answer."
)

MULTIHOP_SYSTEM_PROMPT = (
    "You are a careful benchmark author writing a 2-hop bridge question for a "
    "manufacturing-domain retrieval benchmark. The question must require BOTH "
    "given passages (from two different papers in the same domain) to answer. "
    "The bridge entity must appear in both passages and must be a "
    "manufacturing-technical term (material, process, equipment, or property). "
    "You are given NO knowledge graph.\n"
    + _DOMAIN_CONSTRAINT + "\n"
    + _FEW_SHOT_BLOCK + "\n"
    "Output rules when writing a question: "
    "(1) the answer must be a short span quoted from one of the passages; "
    "(2) the bridge entity must NOT appear verbatim in the question; "
    "(3) reply with a single JSON object and nothing else, using exactly the "
    "keys: question, answer, bridge, evidence_a, evidence_b. "
    "'evidence_a'/'evidence_b' are exact substrings from passage A and B."
)


def build_single_hop_prompt(passage_text: str, qtype: str) -> str:
    """User prompt for a single-hop QA item of the given qtype."""
    instr = {
        QTYPE_FACTUAL: (
            "Write a factual question about a concrete fact in the passage. "
            "Paraphrase the answer term in the question (do not quote it)."
        ),
        QTYPE_PARAPHRASE: (
            "Write a question that refers to the target entity only by a "
            "paraphrase or alias, never by its surface form in the passage."
        ),
        QTYPE_COMPARISON: (
            "Write a comparison question contrasting two quantities, methods, "
            "or properties stated in the passage."
        ),
    }[qtype]
    lines = [
        "Question type: {0}".format(qtype),
        instr,
        "",
        "Passage:",
        passage_text.strip(),
        "",
        "Return JSON with keys: question, answer, evidence.",
    ]
    return "\n".join(lines)


def build_seeded_multihop_prompt(
    text_a: str, text_b: str, bridge_surface: str
) -> str:
    """User prompt for a SEEDED 2-hop bridge QA (M0 discovery seed injected).

    Unlike build_multihop_prompt (which lets the model pick a bridge), this
    injects the M0-discovered bridge surface string and the anchor/target roles:
    passage A is the ANCHOR (the question must be answerable-from / point-to A),
    passage B is the BRIDGE-TARGET (the answer lives in B). To make the QA
    lexical-defeating (design INV-2/INV-3, enforced later by M4):
      (a) the bridge string must NOT appear in the question;
      (b) the question must NOT reuse B's distinctive surface vocabulary (the
          model must paraphrase B's key nouns / method names);
      (c) the question anchors on A's fact and asks for B's fact THROUGH the
          (unstated) bridge -- so a lexical retriever recovers A but not B.

    The bridge is given ONLY as a hint to the model; it is a plain string from
    M0, carries no KG identifier (design section 3 boundary).
    """
    lines = [
        "Question type: {0}".format(QTYPE_MULTIHOP),
        "Write a 2-hop bridge question that REQUIRES BOTH passages to answer.",
        "Passage A is the ANCHOR; passage B is the ANSWER source.",
        "The two passages share this bridge concept: '{0}'.".format(bridge_surface),
        "STRICT rules to make the question genuinely 2-hop:",
        "  (1) Do NOT write the bridge concept '{0}' (or any obvious alias of "
        "it) anywhere in the question.".format(bridge_surface),
        "  (2) Do NOT reuse passage B's distinctive technical words verbatim. "
        "Paraphrase B's key nouns/methods so the question does NOT lexically "
        "match passage B.",
        "  (3) Anchor the question on a fact in passage A, and ask for a fact "
        "found in passage B that is reachable ONLY by following the shared "
        "bridge concept. The answer MUST be a span quoted from passage B.",
        # Self-containment (A1): the retriever reads the question alone.
        "The question MUST be self-contained: NEVER write 'Passage A', "
        "'Passage B', 'Passage 1', 'Passage 2', 'the first passage', "
        "'the second passage', or 'in this passage'. Refer to the subject "
        "matter directly.",
        "",
        "Passage A (anchor):",
        text_a.strip(),
        "",
        "Passage B (answer source):",
        text_b.strip(),
        "",
        "Return JSON with keys: question, answer, bridge, evidence_a, evidence_b.",
        "Set 'bridge' to '{0}'. 'answer' and 'evidence_b' come from passage B; "
        "'evidence_a' comes from passage A.".format(bridge_surface),
    ]
    return "\n".join(lines)


def build_multihop_prompt(text_a: str, text_b: str) -> str:
    """User prompt for a 2-hop bridge QA item over two passages."""
    lines = [
        "Question type: {0}".format(QTYPE_MULTIHOP),
        "Write a 2-hop bridge question requiring BOTH passages.",
        # Self-contained constraint (A1): the question is read in isolation by a
        # retriever that never sees these passage labels, so it must NOT refer to
        # the passages positionally. A post-filter discards any violation, but
        # instructing the generator up front reduces wasted generations.
        "The question MUST be self-contained: NEVER write 'Passage A', "
        "'Passage B', 'Passage 1', 'Passage 2', 'the first passage', "
        "'the second passage', or 'in this passage'. Refer to the subject "
        "matter directly, not to the passage positions.",
        "",
        "Passage A:",
        text_a.strip(),
        "",
        "Passage B:",
        text_b.strip(),
        "",
        "Return JSON with keys: question, answer, bridge, evidence_a, evidence_b.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator response parsing.
# ---------------------------------------------------------------------------


def parse_generator_json(raw: str) -> Optional[dict]:
    """Extract the first JSON object from a generator reply.

    Returns the parsed dict, or None if no JSON object can be recovered or
    the model replied with the single word SKIP (indicating the passage had
    no manufacturing-technical content worth a question).
    Tolerates surrounding prose / code fences by locating the outermost
    brace pair.
    """
    if not raw:
        return None
    text = raw.strip()
    # Model replied SKIP: passage contained only metadata / generic background.
    # Treat as no-candidate (silently dropped, consistent with other None paths).
    if re.match(r"^SKIP\s*$", text, re.IGNORECASE):
        return None
    # Strip a leading ```json / ``` fence if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Locate the first '{' and the matching last '}'.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Front-matter / references / generic-background passage filter (B).
# ---------------------------------------------------------------------------

# Structural heuristics for passages that contain NO manufacturing-technical
# content worth generating a question about.  Each rule is a structural signal
# (not a tuned threshold) justified inline.  Passages matching ANY rule are
# skipped before calling the generator, saving API cost and preventing the
# model from emitting metadata questions even without the SKIP instruction.

# Token budget: we only inspect the first N characters of each passage for
# efficiency.  Front-matter always falls within the first passage of a paper
# (p00000) and is typically < 1 000 chars; we use 1 200 chars to be safe.
# This is an I/O scan bound, not a quality threshold.
_FM_SCAN_CHARS = 1200

# Minimum number of whitespace-delimited tokens a passage must contain for it
# to plausibly hold a technical sentence.  A passage with fewer tokens is
# likely a stub, header, or pure reference list.  20 tokens ~ 3-4 short
# sentences; this catches near-empty chunks without a tunable cutoff.
# Structural rule: if there are fewer than 20 tokens there cannot be a
# complete technical claim to anchor a QA pair.
_MIN_CONTENT_TOKENS = 20

# Patterns whose presence in the FIRST _FM_SCAN_CHARS characters is a strong
# indicator that the passage is front-matter or a reference list, not body
# text with technical claims.  Each pattern is a structural document signal:
#   - "corresponding author" / "correspondence" / "@" near a domain-like
#     string: author-contact block (p00000 of most arXiv papers).
#   - Lines that are exclusively "[N]" citation markers: references section.
#   - Patterns like "University", "Institute", "Corporation" appearing with
#     country/city tokens: affiliation block.
# We use a small whitelist of NEGATIVE signals rather than a content classifier
# to keep this rule-based (no tuning, no model call).
_FRONTMATTER_SIGNALS_RE = re.compile(
    r"(?:"
    r"corresponding\s+author"               # affiliation/contact block
    r"|correspondence\s*:"                  # same
    r"|e-?mail\s*:"                         # author email
    r"|\[\s*\d+\s*\]\s+[A-Z][a-z]"         # numbered reference entry
    r"|^\s*acknowledgem"                    # acknowledgements section
    r"|^\s*(?:received|accepted|published)" # journal submission dates
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def is_skippable_passage(passage: dict) -> bool:
    """Return True if this passage should be excluded from QA generation.

    Two structural rules (not tuned):
    (1) Token count < _MIN_CONTENT_TOKENS: the passage is too short to contain
        a complete technical claim; generating from it produces trivial or
        metadata-only questions.
    (2) Front-matter signal in first _FM_SCAN_CHARS chars: the passage text
        contains markers of author/affiliation blocks or reference lists.
        The regex matches structural document patterns (corresponding author,
        email lines, numbered reference entries, acknowledgements, dates) that
        appear only in non-body sections.

    Passes through all passages for which neither rule fires, so body passages
    with legitimate technical content are never filtered.
    """
    text = passage.get("text", "")
    # Rule 1: too few tokens to hold a technical claim.
    if len(text.split()) < _MIN_CONTENT_TOKENS:
        return True
    # Rule 2: front-matter / reference-list signal in the first scan window.
    scan_window = text[:_FM_SCAN_CHARS]
    if _FRONTMATTER_SIGNALS_RE.search(scan_window):
        return True
    return False


# ---------------------------------------------------------------------------
# Static validators (no model).
# ---------------------------------------------------------------------------


def answer_in_passage(answer: str, passage_text: str) -> bool:
    """True if the answer token sequence appears in the passage (word-boundary).

    Two structural rules -- not tuned thresholds:
    (1) Minimum 3-character answer: prevents accidental matches on 1-2 char
        tokens like "AI", "Fe", "Cu" that are too short to be meaningful spans.
        This is a structural grounding rule (design S2 gate), not a threshold.
    (2) Word-boundary match (\b): "Fe" in "different" must not fire.
        We match the full answer phrase with \b at start and end so it must
        begin and end on a word boundary in the passage.
    """
    if not answer or not passage_text:
        return False
    ans = answer.strip()
    # Structural rule: answers shorter than 3 chars are too short to be
    # unambiguous evidence spans (single-letter / 2-char tokens like "AI",
    # "Fe" match spuriously inside longer words).
    if len(ans) < 3:
        return False
    # Word-boundary search, case-insensitive.
    pattern = r"\b" + re.escape(ans.lower()) + r"\b"
    return bool(re.search(pattern, passage_text.lower()))


def answer_surface_absent_from_question(answer: str, question: str) -> bool:
    """Per D-E: the answer surface form must not appear verbatim in question."""
    if not answer or not question:
        return True
    return answer.strip().lower() not in question.lower()


def find_span(needle: str, haystack: str) -> Optional[List[int]]:
    """Return [start, end] char offsets of needle in haystack (ci), else None."""
    if not needle:
        return None
    idx = haystack.lower().find(needle.strip().lower())
    if idx == -1:
        return None
    return [idx, idx + len(needle.strip())]


# ---------------------------------------------------------------------------
# Multihop bridge-quality gates (A2/A3) -- structural rules, NOT thresholds.
# ---------------------------------------------------------------------------

# Positional-reference patterns a self-contained question must NOT contain (A1
# post-filter). A retriever reads the question in isolation, so a reference to
# "Passage A/B" / "the first passage" / "in this passage" is unanswerable and
# reveals the synthetic two-passage construction. Each alternative is a literal
# document-position phrase, not a tuned cutoff. The \b boundaries prevent
# matching inside longer words.
_POSITIONAL_REF_RE = re.compile(
    r"\b(?:"
    r"passage\s+[ab]"               # "Passage A" / "Passage B"
    r"|passage\s+[12]"              # "Passage 1" / "Passage 2"
    r"|first\s+passage"            # "the first passage"
    r"|second\s+passage"          # "the second passage"
    r"|in\s+this\s+passage"       # "in this passage"
    r")\b",
    re.IGNORECASE,
)

# Generic single-token words that are too unspecific to be a genuine 2-hop
# bridge entity. A real bridge is a multi-word technical term (e.g. "atomic
# layer deposition") OR a typed single-token entity (e.g. "chromium"). A lone
# common/process verb-noun like "automated" or "sensing" links nothing
# domain-specific, so such single-token bridges are rejected. This is a fixed
# structural denylist (pure-grammar function words + the recurring generic
# process/ML nouns that are not bridges), NOT a tuned threshold. Multi-word
# bridges always pass; single-token bridges pass UNLESS they appear here.
_BRIDGE_GENERIC_SINGLE_TOKENS = frozenset(
    # Pure-grammar English function words (language-level stop-words).
    {
        "a", "an", "the", "and", "or", "of", "as", "for", "to", "in",
        "on", "with", "by", "is", "are", "was", "were", "be", "this",
        "that", "these", "those", "it", "its",
    }
    # Generic process / ML / sensing nouns that are not bridges.
    | {
        "automated", "automation", "sensing", "detection", "object",
        "transformer", "transformers", "model", "models", "method",
        "methods", "system", "systems", "process", "processes",
        "approach", "approaches", "technique", "techniques", "algorithm",
        "algorithms", "network", "networks", "framework", "data",
        "feature", "features", "analysis", "result", "results",
    }
)


def is_valid_bridge(bridge: str) -> bool:
    """True if `bridge` is specific enough to anchor a genuine 2-hop link (A2).

    Structural rule (NOT a tuned threshold):
      - A multi-word bridge (>= 2 whitespace tokens) is a compound technical
        term (e.g. "atomic layer deposition") and always passes.
      - A single-token bridge passes ONLY if it is not in the fixed generic
        denylist (_BRIDGE_GENERIC_SINGLE_TOKENS). A lone generic word
        ("automated", "sensing", "object") links nothing domain-specific and is
        rejected; a typed single-token entity ("chromium") passes.
    """
    tokens = bridge.split()
    if len(tokens) >= 2:
        return True
    if not tokens:
        return False
    return tokens[0].lower() not in _BRIDGE_GENERIC_SINGLE_TOKENS


# ---------------------------------------------------------------------------
# Candidate construction from a parsed generator object.
# ---------------------------------------------------------------------------


def make_single_hop_candidate(
    obj: dict,
    passage: dict,
    qtype: str,
    generator_id: str,
) -> Optional[dict]:
    """Validate + assemble a single-hop candidate row, or None if invalid."""
    # Normalise model output the SAME way as the passage text (S1) so unit /
    # symbol-bearing answers compare correctly against the normalised passage.
    question = ascii_normalize((obj.get("question") or "").strip())
    answer = ascii_normalize((obj.get("answer") or "").strip())
    evidence = ascii_normalize((obj.get("evidence") or "").strip())
    if not question or not answer:
        return None
    ptext = passage["text"]
    # Gate: answer must be grounded in the passage text.
    if not answer_in_passage(answer, ptext):
        return None
    # Gate D-E: answer surface form must not be in the question.
    if not answer_surface_absent_from_question(answer, question):
        return None
    # Evidence span: prefer the model's evidence quote, but fall back to the
    # answer span when the evidence is a paraphrase (not a verbatim substring).
    # answer_in_passage already guarantees find_span(answer) succeeds, so this
    # only drops candidates whose answer truly is not a span -- not candidates
    # whose evidence quote happens to be non-verbatim (the dominant failure).
    span = find_span(evidence, ptext) or find_span(answer, ptext)
    if span is None:
        return None
    return {
        "question": question,
        "answer": answer,
        "answer_paper_ids": [passage["paper_id"]],
        "evidence_spans": [
            {"paper_id": passage["paper_id"], "passage_id": passage["passage_id"], "span": span}
        ],
        "qtype": qtype,
        "hop_count": QTYPE_HOP_COUNT[qtype],
        "generator": generator_id,
        "domain": passage["domain"],  # C1: propagate domain from source passage
    }


def make_multihop_candidate(
    obj: dict,
    passage_a: dict,
    passage_b: dict,
    generator_id: str,
) -> Optional[dict]:
    """Validate + assemble a multihop bridge candidate, or None if invalid."""
    # Normalise model output the SAME way as the passage text (S1).
    question = ascii_normalize((obj.get("question") or "").strip())
    answer = ascii_normalize((obj.get("answer") or "").strip())
    bridge = ascii_normalize((obj.get("bridge") or "").strip())
    ev_a = ascii_normalize((obj.get("evidence_a") or "").strip())
    ev_b = ascii_normalize((obj.get("evidence_b") or "").strip())
    if not question or not answer or not bridge:
        return None
    # Gate A1: the question must be self-contained. Discard any candidate that
    # refers to the passages positionally ("Passage A/B", "the first passage",
    # "in this passage"), since the retriever reads the question in isolation.
    if _POSITIONAL_REF_RE.search(question):
        return None
    # Gate A2: the bridge must be a genuine technical link -- a multi-word term
    # or a typed single-token entity, never a lone generic word.
    if not is_valid_bridge(bridge):
        return None
    ta, tb = passage_a["text"], passage_b["text"]
    # Gate: answer grounded in at least one passage. (A3) passage_b is the
    # cross-paper partner where the bridge is followed to the second hop, so we
    # check it for grounding too -- a genuine bridge question is answered from
    # the partner paper, not the seed. The OR keeps yield non-zero when the
    # answer legitimately lives on the seed (passage_a) side.
    if not (answer_in_passage(answer, ta) or answer_in_passage(answer, tb)):
        return None
    # Gate: bridge must appear in BOTH passages (design S4 precondition).
    if not (answer_in_passage(bridge, ta) and answer_in_passage(bridge, tb)):
        return None
    # Gate D-E: neither the bridge NOR the answer surface form may appear in
    # the question. The answer check matches assert_candidate_invariants (which
    # checks the answer for ALL qtypes); without it a multihop candidate whose
    # answer leaks into the question passes make_* but trips the batch invariant
    # and aborts the whole run instead of being dropped here.
    if not answer_surface_absent_from_question(bridge, question):
        return None
    if not answer_surface_absent_from_question(answer, question):
        return None
    # Evidence spans: prefer the model's evidence quote, fall back to the bridge
    # (guaranteed in both passages by the gate above) when evidence is a non-
    # verbatim paraphrase.
    span_a = find_span(ev_a, ta) or find_span(bridge, ta)
    span_b = find_span(ev_b, tb) or find_span(bridge, tb)
    if span_a is None or span_b is None:
        return None
    return {
        "question": question,
        "answer": answer,
        "answer_paper_ids": [passage_a["paper_id"], passage_b["paper_id"]],
        "evidence_spans": [
            {"paper_id": passage_a["paper_id"], "passage_id": passage_a["passage_id"], "span": span_a},
            {"paper_id": passage_b["paper_id"], "passage_id": passage_b["passage_id"], "span": span_b},
        ],
        "qtype": QTYPE_MULTIHOP,
        "hop_count": QTYPE_HOP_COUNT[QTYPE_MULTIHOP],
        "generator": generator_id,
        "bridge": bridge,
        "domain": passage_a["domain"],  # C1: both passages share the same domain
    }


# ---------------------------------------------------------------------------
# Passage loading + grouping.
# ---------------------------------------------------------------------------


def load_passages(path: Path) -> List[dict]:
    """Load passages.jsonl into a list of dicts."""
    rows: List[dict] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_domain(passages: Sequence[dict]) -> Dict[str, List[dict]]:
    """Group passages by domain tag (preserving input order)."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for p in passages:
        out[p["domain"]].append(p)
    return dict(out)


# ---------------------------------------------------------------------------
# Generation driver (backend injectable; tests pass a stub).
# ---------------------------------------------------------------------------


def cap_candidates_per_domain(
    rows: Sequence[dict], passages: Sequence[dict], max_qa_per_domain: int
) -> List[dict]:
    """Hard-cap the candidate count per domain (pilot cost guard).

    Uncapped, multihop generation can dominate the budget, so this enforces a STRUCTURAL hard cap of `max_qa_per_domain` total QA per
    domain. Within each domain the cap is split by the design tier mix
    (HARD_TIER_RATIO for multihop/hard; the remainder for single-hop), so a
    domain never spends its whole budget on hard multihop items.

    Selection is deterministic (input order preserved) -- NO random sampling,
    NO scoring -- so the cap is a pure structural truncation, not tuning.
    """
    if max_qa_per_domain is None or max_qa_per_domain < 0:
        return list(rows)
    domain_of_passage = {p["passage_id"]: p["domain"] for p in passages}

    def _row_domain(r: dict) -> str:
        # A row's domain is the domain of its first evidence passage.
        spans = r.get("evidence_spans") or []
        if spans:
            pid = spans[0].get("passage_id")
            if pid in domain_of_passage:
                return domain_of_passage[pid]
        return "unknown"

    # Per-domain hard-tier quota (multihop) vs single-hop quota.
    hard_quota = int(round(max_qa_per_domain * HARD_TIER_RATIO))
    single_quota = max_qa_per_domain - hard_quota

    kept: List[dict] = []
    taken_single: Dict[str, int] = defaultdict(int)
    taken_hard: Dict[str, int] = defaultdict(int)
    for r in rows:
        dom = _row_domain(r)
        is_hard = r.get("qtype") == QTYPE_MULTIHOP
        if is_hard:
            if taken_hard[dom] >= hard_quota:
                continue
            taken_hard[dom] += 1
        else:
            if taken_single[dom] >= single_quota:
                continue
            taken_single[dom] += 1
        kept.append(r)
    return kept


def generate_candidates(
    passages: Sequence[dict],
    backend,
    *,
    per_domain: Optional[int] = None,
    max_qa_per_domain: Optional[int] = None,
    max_tokens: int = GEN_MAX_TOKENS_SINGLE,
) -> List[dict]:
    """Generate QA candidates from passages using the injected backend.

    Single-hop: for each source passage one attempt per single-hop qtype.
    Multihop: for each source passage, pair it with ONE representative passage
    from each OTHER paper in the same domain (cross-paper, not cross-boundary).
    `per_domain` caps the number of source passages per domain (pilot mode).
    `max_qa_per_domain` (pilot cost guard) hard-caps the TOTAL accepted QA per
    domain after generation via cap_candidates_per_domain (tier-split). Invalid
    / ungrounded generations are silently dropped.

    `backend` must expose chat(system, user, *, max_tokens) -> str and a
    `name` attribute (the generator id recorded on each row).
    """
    generator_id = getattr(backend, "name", "unknown-generator")
    by_dom = group_by_domain(passages)
    out: List[dict] = []

    # Per-tier generation budgets when a hard cap is active (cost guard). None
    # means unbounded (full build). We stop generating each tier for a domain
    # once OVERGEN_FACTOR x quota valid candidates are collected so the post-cap
    # selector still has surplus without paying for the full pair space.
    if max_qa_per_domain is not None:
        hard_q = int(round(max_qa_per_domain * HARD_TIER_RATIO))
        single_budget = (max_qa_per_domain - hard_q) * OVERGEN_FACTOR
        multihop_budget = hard_q * OVERGEN_FACTOR
    else:
        single_budget = None
        multihop_budget = None

    for domain in sorted(by_dom.keys()):
        dom_passages = by_dom[domain]
        # Apply the front-matter/reference/background skip filter BEFORE the
        # per_domain cap. Otherwise --per-domain N selects the FIRST N passages
        # of each paper-ordered domain list -- which are the title/author/
        # abstract front-matter (all skippable) -- and every selected source is
        # then dropped, yielding 0 candidates. Capping on the technical subset
        # guarantees N real content passages. The pool for multihop partners is
        # the same technical subset so bridges are never built on front-matter.
        technical = [p for p in dom_passages if not is_skippable_passage(p)]
        if per_domain is not None:
            sources = technical[:per_domain]
        else:
            sources = technical

        # Single-hop: one attempt per qtype per source passage.
        # Skip front-matter / reference / generic-background passages (B)
        # before calling the generator to save API cost and prevent metadata
        # questions from entering the pipeline.
        n_single = 0
        for passage in sources:
            if single_budget is not None and n_single >= single_budget:
                break
            if is_skippable_passage(passage):
                continue
            for qtype in SINGLE_HOP_QTYPES:
                user = build_single_hop_prompt(passage["text"], qtype)
                try:
                    raw = backend.chat(
                        GENERATOR_SYSTEM_PROMPT, user, max_tokens=GEN_MAX_TOKENS_SINGLE
                    )
                except Exception:  # noqa: BLE001 - skip failed generations
                    continue
                obj = parse_generator_json(raw)
                if obj is None:
                    continue
                cand = make_single_hop_candidate(obj, passage, qtype, generator_id)
                if cand is not None:
                    out.append(cand)
                    n_single += 1

        # Multihop: pair each source passage with one passage from each OTHER
        # paper in the same domain. This produces cross-paper pairs regardless
        # of passage ordering, fixing the "consecutive boundary only" bug where
        # every pair happened to be a topic-boundary pair with no shared bridge
        # entity, causing the bridge-in-both gate to reject all multihop items.
        # `per_domain` caps multihop pair count via the source list size; the
        # multihop_budget early-stop bounds generation cost under a hard cap.
        # Skip front-matter passages on both sides of the pair (B).
        n_multi = 0
        for a, b in _cross_paper_pairs(sources, technical, per_domain=per_domain):
            if multihop_budget is not None and n_multi >= multihop_budget:
                break
            if is_skippable_passage(a) or is_skippable_passage(b):
                continue
            user = build_multihop_prompt(a["text"], b["text"])
            try:
                raw = backend.chat(
                    MULTIHOP_SYSTEM_PROMPT, user, max_tokens=GEN_MAX_TOKENS_MULTIHOP
                )
            except Exception:  # noqa: BLE001
                continue
            obj = parse_generator_json(raw)
            if obj is None:
                continue
            cand = make_multihop_candidate(obj, a, b, generator_id)
            if cand is not None:
                out.append(cand)
                n_multi += 1

    if max_qa_per_domain is not None:
        out = cap_candidates_per_domain(out, passages, max_qa_per_domain)
    return out


# ---------------------------------------------------------------------------
# M0-seeded multihop generation (design section 4, M2).
#
# Instead of the random cross-paper pairing above, this path consumes the
# bridge_pairs.jsonl SEEDS produced by paper/gold/discover_bridge_pairs.py and
# generates ONE genuine multihop QA per (A, B, bridge) seed. The seed carries
# ONLY strings (paper ids + bridge surface) -- no KG identifier -- so reading it
# here keeps the gold pipeline KG-free (the gold_v2 acyclicity CI still holds:
# this module imports no KG; bridge_pairs.jsonl is plain text data).
# ---------------------------------------------------------------------------


def load_bridge_pairs(path: Path) -> List[dict]:
    """Load M0 seed rows: {domain, paper_a, paper_b, bridge_surface, ...}."""
    rows: List[dict] = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _representative_passage(
    paper_id: str, passages_by_paper: Dict[str, List[dict]], bridge: str
) -> Optional[dict]:
    """Pick the passage of `paper_id` to feed the generator for a seed.

    Prefer the FIRST passage whose text actually contains the bridge surface
    string (so the bridge -- guaranteed in both papers by M0 -- is visible to
    the generator and the make_multihop bridge-in-both gate can pass). Fall back
    to the paper's first technical passage when no single passage holds the
    bridge (the bridge may span the concatenated paper text).
    """
    candidates = passages_by_paper.get(paper_id, [])
    technical = [p for p in candidates if not is_skippable_passage(p)] or candidates
    for p in technical:
        if answer_in_passage(bridge, p.get("text", "")):
            return p
    return technical[0] if technical else None


def generate_from_seed_pairs(
    seed_pairs: Sequence[dict],
    passages: Sequence[dict],
    backend,
    *,
    max_per_domain: Optional[int] = None,
) -> List[dict]:
    """Generate genuine multihop candidates from M0 (A, B, bridge) seeds.

    For each seed we pick a representative passage from A and from B (preferring
    one that contains the bridge), inject the bridge into a B-vocabulary-avoiding
    prompt, and validate via make_multihop_candidate (same structural gates as
    the random path: bridge in both, self-contained, answer grounded, bridge/
    answer not leaked into the question). `max_per_domain` caps accepted
    candidates per domain (cost guard, deterministic input-order truncation).

    The KEPT rows carry gold_paper_ids=[A, B] and gold_bridge_entities=[bridge]
    (the bridge as a STRING only -- no KG id), hop_count=2, matching the design
    output schema. Invalid generations are silently dropped.
    """
    generator_id = getattr(backend, "name", "unknown-generator")
    by_paper: Dict[str, List[dict]] = defaultdict(list)
    for p in passages:
        by_paper[p["paper_id"]].append(p)

    out: List[dict] = []
    taken: Dict[str, int] = defaultdict(int)
    for seed in seed_pairs:
        domain = str(seed.get("domain") or "unknown")
        if max_per_domain is not None and taken[domain] >= max_per_domain:
            continue
        pa_id = str(seed.get("paper_a") or "")
        pb_id = str(seed.get("paper_b") or "")
        bridge = str(seed.get("bridge_surface") or "").strip()
        if not pa_id or not pb_id or not bridge:
            continue
        passage_a = _representative_passage(pa_id, by_paper, bridge)
        passage_b = _representative_passage(pb_id, by_paper, bridge)
        if passage_a is None or passage_b is None:
            continue
        user = build_seeded_multihop_prompt(
            passage_a["text"], passage_b["text"], bridge
        )
        try:
            raw = backend.chat(
                MULTIHOP_SYSTEM_PROMPT, user, max_tokens=GEN_MAX_TOKENS_MULTIHOP
            )
        except Exception:  # noqa: BLE001 - skip failed generations
            continue
        obj = parse_generator_json(raw)
        if obj is None:
            continue
        # Force the seed bridge string onto the candidate so make_multihop's
        # bridge-in-both gate checks the DISCOVERED bridge (the model is told to
        # echo it, but we do not trust the echo).
        obj["bridge"] = bridge
        cand = make_multihop_candidate(obj, passage_a, passage_b, generator_id)
        if cand is None:
            continue
        # Stamp the design gold fields (strings only -- NO KG identifier).
        cand["gold_paper_ids"] = [pa_id, pb_id]
        cand["gold_bridge_entities"] = [bridge]
        cand["seed_fanout"] = int(seed.get("fanout") or 0)
        out.append(cand)
        taken[domain] += 1
    return out


def _cross_paper_pairs(
    sources: Sequence[dict],
    domain_pool: Sequence[dict],
    *,
    per_domain: Optional[int] = None,
) -> Iterator[Tuple[dict, dict]]:
    """Yield (source_passage, partner_passage) across DIFFERENT papers.

    For each passage in `sources`, find one representative passage from every
    other paper present in `domain_pool`. "Representative" = the first passage
    of that paper in pool order (deterministic, no random sampling).

    When `per_domain` is set the source list is already capped by the caller;
    we additionally cap partner papers to `per_domain` distinct paper_ids so
    the multihop pair count stays proportional to the pilot budget.
    """
    # Build a map: paper_id -> first passage in pool order (representative).
    first_passage_of: Dict[str, dict] = {}
    for p in domain_pool:
        pid = p["paper_id"]
        if pid not in first_passage_of:
            first_passage_of[pid] = p

    for src in sources:
        src_pid = src["paper_id"]
        partners = [
            first_passage_of[pid]
            for pid in first_passage_of
            if pid != src_pid
        ]
        # Cap partner count when per_domain is active, to avoid O(n^2) blowup
        # in full-corpus runs while keeping pairs manageable for the pilot.
        if per_domain is not None:
            partners = partners[:per_domain]
        for partner in partners:
            yield src, partner


# ---------------------------------------------------------------------------
# Output gate + writer.
# ---------------------------------------------------------------------------


def assert_candidate_invariants(
    rows: List[dict],
    passage_index: Dict[str, dict],
    *,
    require_all_qtypes: bool = True,
) -> None:
    """Raise ValueError if any candidate violates the S2 static gate.

    Gate (design S2): every evidence span is a real substring of its passage,
    qtype is one of the forced four, hop_count matches the qtype, and all four
    qtypes are present at least once across the batch.

    `require_all_qtypes=False` relaxes only the all-qtypes-present check; it is
    used by the capped pilot, where a small per-domain hard cap can legitimately
    drop a qtype (the per-row validity checks still apply). The full build keeps
    the default (True).
    """
    seen_qtypes = set()
    for r in rows:
        qt = r.get("qtype")
        if qt not in ALL_QTYPES:
            raise ValueError("bad qtype: {0}".format(qt))
        if r.get("hop_count") != QTYPE_HOP_COUNT[qt]:
            raise ValueError("hop_count mismatch for qtype {0}".format(qt))
        if answer_surface_absent_from_question(r["answer"], r["question"]) is False:
            raise ValueError("answer surface leaked into question")
        for ev in r.get("evidence_spans", []):
            pid = ev["passage_id"]
            passage = passage_index.get(pid)
            if passage is None:
                raise ValueError("evidence passage not found: {0}".format(pid))
            s, e = ev["span"]
            sub = passage["text"][s:e]
            # span must be non-empty and lie within the passage text.
            if not sub or e > len(passage["text"]) or s < 0:
                raise ValueError("evidence span out of range: {0}".format(pid))
        seen_qtypes.add(qt)
    if require_all_qtypes:
        missing = set(ALL_QTYPES) - seen_qtypes
        if missing:
            raise ValueError("missing forced qtypes: {0}".format(sorted(missing)))


def write_candidates_jsonl(rows: List[dict], out_path: Path) -> None:
    """Write candidates as one JSON object per line (ASCII-safe)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True))
            f.write("\n")


# ---------------------------------------------------------------------------
# CLI (live generation; backend constructed lazily so tests never import SDKs).
# ---------------------------------------------------------------------------


def _build_default_backend():
    """Construct the Sonnet generator backend (live use only)."""
    from paper.eval.cross_judge_backends import ClaudeSonnetBackend

    return ClaudeSonnetBackend()


def _build_mistral_backend(model_tag: str = "mistral-small:24b"):
    """Construct the local Ollama (mistral) generator (M2 design: local, free)."""
    from paper.eval.cross_judge_backends import OllamaBackend

    return OllamaBackend(model_tag)


def main_seeded(argv: Optional[List[str]] = None) -> int:
    """M2 live driver: generate genuine multihop QA from M0 bridge seeds.

    Reads bridge_pairs.jsonl (A, B, bridge strings -- no KG id), generates one
    candidate per seed with the local mistral generator (default), and writes
    multihop candidates carrying gold_paper_ids / gold_bridge_entities. No SDK is
    imported until this function runs, so unit tests never touch it.
    """
    parser = argparse.ArgumentParser(
        description="M2 seeded multihop generator (from M0 bridge_pairs.jsonl)"
    )
    parser.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    parser.add_argument(
        "--bridge-pairs",
        default=str(PROJECT_ROOT / "paper" / "gold" / "bridge_pairs.jsonl"),
        help="M0 seed file (A, B, bridge surface strings -- no KG id)",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "paper" / "gold_v2" / "qa_multihop_candidates.jsonl"),
    )
    parser.add_argument(
        "--backend",
        choices=["mistral", "sonnet"],
        default="mistral",
        help="generator backend (default mistral: local + free, design M2)",
    )
    parser.add_argument("--model-tag", default="mistral-small:24b")
    parser.add_argument(
        "--max-per-domain",
        type=int,
        default=None,
        help="cap accepted multihop candidates per domain (cost guard)",
    )
    args = parser.parse_args(argv)

    passages = load_passages(Path(args.passages))
    seed_pairs = load_bridge_pairs(Path(args.bridge_pairs))
    backend = (
        _build_mistral_backend(args.model_tag)
        if args.backend == "mistral"
        else _build_default_backend()
    )
    rows = generate_from_seed_pairs(
        seed_pairs, passages, backend, max_per_domain=args.max_per_domain
    )
    write_candidates_jsonl(rows, Path(args.out))
    by_dom: Dict[str, int] = defaultdict(int)
    for r in rows:
        by_dom[r.get("domain", "unknown")] += 1
    print(
        "M2 seeded done: {0} multihop candidates from {1} seeds -> {2}".format(
            len(rows), len(seed_pairs), args.out
        )
    )
    for dom in sorted(by_dom):
        print("  {0}: {1}".format(dom, by_dom[dom]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S2 QA candidate generator")
    parser.add_argument("--passages", default=str(DEFAULT_PASSAGES))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--backend",
        choices=["mistral", "sonnet"],
        default="mistral",
        help="generator backend (default: local Mistral, matching frozen rows)",
    )
    parser.add_argument(
        "--model-tag",
        default="mistral-small:24b",
        help="Ollama model tag used when --backend=mistral",
    )
    parser.add_argument(
        "--per-domain",
        type=int,
        default=None,
        help="pilot mode: cap source passages per domain (e.g. 20)",
    )
    parser.add_argument(
        "--max-qa-per-domain",
        type=int,
        default=None,
        help=(
            "pilot cost guard: hard-cap TOTAL accepted QA per domain "
            "(tier-split; multihop limited to HARD_TIER_RATIO of the cap)"
        ),
    )
    args = parser.parse_args(argv)

    passages = load_passages(Path(args.passages))
    backend = (
        _build_mistral_backend(args.model_tag)
        if args.backend == "mistral"
        else _build_default_backend()
    )
    rows = generate_candidates(
        passages,
        backend,
        per_domain=args.per_domain,
        max_qa_per_domain=args.max_qa_per_domain,
    )
    passage_index = {p["passage_id"]: p for p in passages}
    assert_candidate_invariants(rows, passage_index)
    write_candidates_jsonl(rows, Path(args.out))
    print(
        "S2 done: {0} candidates -> {1} (per_domain={2})".format(
            len(rows), args.out, args.per_domain
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
