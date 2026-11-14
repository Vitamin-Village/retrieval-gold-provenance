"""Sub-method (D36 / H11): entity-type-aware answer requirements.

Per the project-design answer-schema method, retrieval answers are formatted to expose
the *fields a reader needs* for that entity's question class. This module
maps a top-candidate's KG entity type to a small ``AnswerSpec`` that the
downstream answer-formatting block can use as a label scaffold. No LLM
call is made -- if the upstream NLI/QA returned a plain string, the
scaffold is applied to the structured dict only.

    Process    -> (outputs, conditions)
    Equipment  -> (parameters, specs)
    Material   -> (properties, alternatives)
    Issue      -> (root_causes, solutions)

The OFF flag (cell C12, ``generic``) returns the single-field
``("answer",)`` schema, which reproduces today's plain-text answer
behaviour.

Reference: project design; MASTER_PLAN W7; DECISIONS.md D36.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


EntityType = str  # "Process" | "Equipment" | "Material" | "Issue" | "generic"


_KNOWN_TYPES = ("Process", "Equipment", "Material", "Issue")


@dataclass(frozen=True)
class AnswerSpec:
    """The set of fields a structured answer should expose for an entity type."""

    entity_type: EntityType
    fields: tuple[str, ...]


_SCHEMAS: dict[EntityType, AnswerSpec] = {
    "Process": AnswerSpec("Process", ("outputs", "conditions")),
    "Equipment": AnswerSpec("Equipment", ("parameters", "specs")),
    "Material": AnswerSpec("Material", ("properties", "alternatives")),
    "Issue": AnswerSpec("Issue", ("root_causes", "solutions")),
    "generic": AnswerSpec("generic", ("answer",)),
}


_QUERY_TIE_BREAK = (
    (re.compile(r"\b(defect|issue|problem|fail(ure|s|ed)?|error)\b"), "Issue"),
    (re.compile(r"\b(machine|equipment|tool|chamber|reactor)\b"), "Equipment"),
    (re.compile(r"\b(material|substrate|wafer|gas|chemical)\b"), "Material"),
    (re.compile(r"\b(process|step|method|procedure)\b"), "Process"),
)


def entity_type_of(
    query: str,
    candidate_entity_types: Iterable[str],
) -> EntityType:
    """Pick the most frequent {Process,Equipment,Material,Issue} type.

    Ties are broken by the query's lexical signal (e.g., "defect" -> Issue).
    If neither the candidate set nor the query mentions a known type the
    function returns ``"generic"`` so the caller can fall back to the
    single-field schema.
    """
    counts: dict[str, int] = {t: 0 for t in _KNOWN_TYPES}
    for raw in candidate_entity_types:
        normalised = _normalise_type(str(raw))
        if normalised in counts:
            counts[normalised] += 1

    nonzero = {key: value for key, value in counts.items() if value > 0}
    if not nonzero:
        # No KG signal -> defer entirely to the query.
        for pattern, type_label in _QUERY_TIE_BREAK:
            if pattern.search(query.lower()):
                return type_label
        return "generic"

    top = max(nonzero.values())
    winners = sorted(t for t, v in nonzero.items() if v == top)
    if len(winners) == 1:
        return winners[0]

    # Tie -> query lexical signal breaks it. If none matches, alphabetical.
    for pattern, type_label in _QUERY_TIE_BREAK:
        if pattern.search(query.lower()) and type_label in winners:
            return type_label
    return winners[0]


def _normalise_type(raw: str) -> str:
    """Map free-form type strings to one of the four canonical labels."""
    key = raw.strip().lower()
    for label in _KNOWN_TYPES:
        if key == label.lower():
            return label
    return raw


def schema_for(entity_type: EntityType) -> AnswerSpec:
    """Return the ``AnswerSpec`` for a (possibly unknown) entity type."""
    return _SCHEMAS.get(entity_type, _SCHEMAS["generic"])


def format_answer(
    answer_text: str,
    spec: AnswerSpec,
) -> dict[str, Any]:
    """Apply ``spec`` as a label scaffold over ``answer_text``.

    No content synthesis: every spec field is initialised to ``""`` and
    the upstream string is parked under ``"answer"`` so callers can still
    surface the original prose alongside the structured shell.
    """
    out: dict[str, Any] = {field: "" for field in spec.fields}
    out["answer"] = answer_text
    out["entity_type"] = spec.entity_type
    return out
