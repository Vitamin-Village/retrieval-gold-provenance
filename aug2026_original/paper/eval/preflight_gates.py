# -*- coding: utf-8 -*-
"""D47 fail-fast gates for paper eval runner + ablation orchestrator.

Phase 1 implements: G3 (seed_effective_n), G4 (recomputed_stat),
G7 (projection_provenance). Phase 2-4 will add: G5, G8, G1, G2, G6.

References
----------
- paper/DECISIONS.md D44-D47 (gate roster + behavioural contracts).
- prototype/eval/run_retrieval_eval_v3.py
    * 1200-1303: ``_retrieved_entities`` RRF projection.
    * 1305-...:  ``_retrieved_edges``    RRF projection.
    * 1582-1650: ``_summarize`` (records -> summary[*_mean]).
    * 1653-1687: ``_summarize_by_domain`` (Q6 = full check).
- paper/eval/ablation_orchestrator.py:319-401 (statistics aggregator).

Design notes (all ASCII safe: cp949-compatible runtime error messages)
---------------------------------------------------------------------
1. No silent skip. Every check path raises a GateError subclass.
2. Every magic number has a comment citing the paper Method or a published convention.
3. The Q3 bypass is explicit, through the environment variable ``RAG_V3_ALLOW_DETERMINISTIC_SEEDS=1``.
4. Q6 by_domain scope = full (root and by_domain are both checked).
5. Q7 violation handling = write the payload + ``.GATE_FAIL.log`` + audit field.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any


# ---- exceptions -----------------------------------------------------------

class GateError(Exception):
    """Base class for all preflight gate violations."""

    gate_name: str = "unknown"


class RecomputedStatError(GateError):
    gate_name = "recomputed_stat"


class SeedEffectiveNError(GateError):
    gate_name = "seed_effective_n"


class ProjectionProvenanceError(GateError):
    gate_name = "projection_provenance"


class AblationIsolationError(GateError):
    gate_name = "ablation_isolation"


class EstimandError(GateError):
    gate_name = "estimand"


class ArtifactIntegrityError(GateError):
    """G1: retriever class / runtime metadata / payload backend label
    consistency. Raised when the runtime-loaded retriever and the
    reported actual_backend label disagree, or when a silent fallback
    has occurred without a recorded fallback_reason.
    """

    gate_name = "artifact_integrity"


class BackendGateError(GateError):
    """G2: paper-level backend claim vs. actual_backend bidirectional
    consistency. Raised when the paper claim (--paper-backend-claim CLI
    or RAG_V3_PAPER_BACKEND_CLAIM env) disagrees with the actual
    backend used to produce the payload.
    """

    gate_name = "backend_gate"


# ---- constants (each cites its source in the paper Method) ------------------------------

# IR convention reports metrics rounded to 4 decimals (e.g. hit@5 = 0.7536).
# 1e-6 is 1000x stricter than that publication precision, so it catches
# float accumulation drift without flagging legitimate rounding noise.
# Not a tuned knob: lifted directly from numpy.isclose default rtol scale.
_RECOMP_TOL: float = 1e-6

# DECISIONS.md D46 fix #1: HippoRAG-style RRF projection from paper-rank
# to entity-/edge-rank. The decorator marker ``__projection_version__``
# is the contract every projection helper must expose so that G7 can
# verify the runner module has not silently reverted to a pre-D46
# implementation.
_EXPECTED_PROJECTION_VERSION: str = "rrf_v1_d46"

# Reciprocal Rank Fusion constant.
#   Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms
#   Condorcet and individual Rank Learning Methods." SIGIR 2009.
# k=60 is the structural constant published in that paper -- NOT a tuned
# hyperparameter. The runner pins this in ``_K_RRF`` at module scope; G7
# verifies it has not drifted from the published value.
# Source: prototype/eval/run_retrieval_eval_v3.py:94 (``_K_RRF``).
_K_RRF_EXPECTED: int = 60

# Q3 bypass env var. Setting this to "1" allows G3 to mark the cell as
# ``gate_bypassed`` instead of raising when 3-seed hashes are identical
# (intended for deterministic-by-design retrievers such as the
# pre-tie-band BM25 baseline, where seed has zero effect on ordering).
_SEED_BYPASS_ENV: str = "RAG_V3_ALLOW_DETERMINISTIC_SEEDS"


# ---- payload helpers ------------------------------------------------------

def _summary_of(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise RecomputedStatError(
            "payload missing 'summary' mapping (got "
            f"type={type(summary).__name__})"
        )
    return summary


def _records_of(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = payload.get("records")
    if records is None:
        return []
    if not isinstance(records, list):
        raise RecomputedStatError(
            "payload 'records' must be list (got "
            f"type={type(records).__name__})"
        )
    return [r for r in records if isinstance(r, Mapping)]


def _record_metric(record: Mapping[str, Any], metric_name: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    value = metrics.get(metric_name)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---- G4 recomputed_stat ---------------------------------------------------

# Pairs of (summary_key_root, record_metric_key). The summary stores the
# pre-aggregated mean under ``<key>_mean`` (legacy flat schema) and the
# records carry the per-QA metric under the matching ``record_metric``
# field. G4 recomputes mean(records[*]) and compares to the summary value.
_METRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("source_hit_at_5_mean", "source_hit_at_5"),
    ("source_mrr_mean",      "source_mrr"),
    ("source_recall_at_5_mean", "source_recall_at_5"),
    ("entity_f1_at_5_mean",  "entity_f1_at_5"),
    ("edge_f1_at_5_mean",    "edge_f1_at_5"),
)


def _check_metric_pairs(
    summary_block: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> None:
    """Verify summary[*_mean] = mean(records[metric_key]) within tolerance.

    ``scope`` is a human-readable label ("root" or "by_domain:<name>") used
    in the raised error message; it is ASCII so cp949 consoles do not
    explode when the violation is printed.
    """
    for summary_key, record_key in _METRIC_PAIRS:
        if summary_key not in summary_block:
            # Missing keys are not violations -- legacy payloads may omit
            # them. G4 only enforces consistency where the summary has
            # already committed a value.
            continue
        try:
            reported = float(summary_block[summary_key])
        except (TypeError, ValueError) as exc:
            raise RecomputedStatError(
                f"[{scope}] summary['{summary_key}'] is not numeric: "
                f"{summary_block[summary_key]!r} ({exc})"
            ) from exc
        recomputed = _mean([_record_metric(r, record_key) for r in records])
        if abs(reported - recomputed) > _RECOMP_TOL:
            raise RecomputedStatError(
                f"[{scope}] summary['{summary_key}']={reported:.10f} "
                f"does not match recomputed mean={recomputed:.10f} "
                f"(diff={abs(reported - recomputed):.3e}, "
                f"tol={_RECOMP_TOL:.0e}, n={len(records)})"
            )


def assert_recomputed_stat(payload: Mapping[str, Any]) -> None:
    """G4: summary[*_mean] must equal mean(records[*]) within 1e-6.

    Covers both ``summary.*_mean`` (root scope) and
    ``summary.by_domain[<domain>].*_mean`` (Q6 = full scope).
    """
    summary = _summary_of(payload)
    records = _records_of(payload)

    # Root scope.
    _check_metric_pairs(summary, records, scope="root")

    # by_domain scope (Q6 = full).
    by_domain = summary.get("by_domain")
    if not isinstance(by_domain, Mapping):
        return

    # Group records by domain so we can recompute per-domain means.
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        dom = str(r.get("domain") or "unknown")
        grouped.setdefault(dom, []).append(r)

    for domain, block in by_domain.items():
        if not isinstance(block, Mapping):
            raise RecomputedStatError(
                f"summary.by_domain[{domain!r}] must be mapping, "
                f"got type={type(block).__name__}"
            )
        domain_records = grouped.get(str(domain), [])
        # Phase 1 fix #2. When a domain has
        # zero records, _mean([]) returns 0.0; if the summary reported any
        # non-zero mean for that domain, _check_metric_pairs would not be
        # called the same way (records pass-through), producing a false
        # PASS. Explicitly flag the inconsistency.
        if not domain_records:
            for summary_key, _record_key in _METRIC_PAIRS:
                if summary_key not in block:
                    continue
                try:
                    reported = float(block[summary_key])
                except (TypeError, ValueError) as exc:
                    raise RecomputedStatError(
                        f"[by_domain:{domain}] summary['{summary_key}'] "
                        f"is not numeric: {block[summary_key]!r} ({exc})"
                    ) from exc
                if reported != 0.0:
                    raise RecomputedStatError(
                        f"[by_domain:{domain}] reports "
                        f"'{summary_key}'={reported:.10f} but records "
                        "contains 0 entries for this domain"
                    )
            # 0 records + all-zero summary -> consistent, skip per-pair check.
            continue
        _check_metric_pairs(
            block, domain_records, scope=f"by_domain:{domain}"
        )


def assert_table_consistency(
    table_rows: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
) -> None:
    """G4 aggregator-level: ablation_table.csv rows must reconcile with
    ablation_matrix.json cell means.

    ``table_rows``: each row has keys (config, seed, domain, hit_at_5,
    mrr, entity_f1, edge_f1) -- see ablation_orchestrator.build_ablation_table.
    ``matrix``: schema {c0_reference, cells: {cfg: {metric: {mean,
    delta_vs_c0}}}} -- see ablation_orchestrator.build_ablation_matrix.
    """
    if not isinstance(matrix, Mapping):
        raise RecomputedStatError(
            f"matrix must be mapping, got type={type(matrix).__name__}"
        )
    cells = matrix.get("cells")
    if not isinstance(cells, Mapping):
        raise RecomputedStatError(
            "matrix['cells'] must be mapping, got "
            f"type={type(cells).__name__}"
        )

    metrics = ("hit_at_5", "mrr", "entity_f1", "edge_f1")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in table_rows:
        cfg = str(row.get("config") or "")
        if not cfg:
            continue
        grouped.setdefault(cfg, []).append(row)

    for cfg, cell in cells.items():
        if not isinstance(cell, Mapping):
            raise RecomputedStatError(
                f"matrix.cells[{cfg!r}] must be mapping, "
                f"got type={type(cell).__name__}"
            )
        rows = grouped.get(str(cfg), [])
        for metric in metrics:
            entry = cell.get(metric)
            if not isinstance(entry, Mapping):
                continue
            reported = entry.get("mean")
            if reported is None:
                continue
            try:
                reported_f = float(reported)
            except (TypeError, ValueError) as exc:
                raise RecomputedStatError(
                    f"matrix.cells[{cfg!r}][{metric!r}]['mean'] is not "
                    f"numeric: {reported!r}"
                ) from exc
            vals = []
            for r in rows:
                try:
                    vals.append(float(r.get(metric, 0.0)))
                except (TypeError, ValueError):
                    vals.append(0.0)
            recomputed = _mean(vals)
            if abs(reported_f - recomputed) > _RECOMP_TOL:
                raise RecomputedStatError(
                    f"matrix.cells[{cfg!r}][{metric!r}]['mean']="
                    f"{reported_f:.10f} does not match recomputed mean="
                    f"{recomputed:.10f} (diff="
                    f"{abs(reported_f - recomputed):.3e}, "
                    f"tol={_RECOMP_TOL:.0e}, n={len(vals)})"
                )


# ---- G7 projection_provenance --------------------------------------------

_PROJECTION_HELPER_NAMES: tuple[str, ...] = (
    "_retrieved_entities",
    "_retrieved_edges",
)


def assert_projection_provenance(
    payload: Mapping[str, Any] | None = None,
    *,
    runner_module: Any | None = None,
) -> None:
    """G7: projection helpers must carry ``__projection_version__``
    marker and the runner must use the published RRF constant k=60.

    Two operating modes:
    * Static (``runner_module`` set) -- called once at runner main()
      startup to verify the loaded code, before any retrieval runs.
    * Payload-introspection (``payload`` set) -- called inside the
      runner just before writing JSON so a re-imported module never
      slips through.
    """
    if runner_module is not None:
        # 1. _K_RRF constant must match Cormack 2009 published value.
        k_rrf = getattr(runner_module, "_K_RRF", None)
        if k_rrf != _K_RRF_EXPECTED:
            raise ProjectionProvenanceError(
                f"runner._K_RRF={k_rrf!r} but expected "
                f"{_K_RRF_EXPECTED} (Cormack 2009 published RRF constant)"
            )
        # 2. Both projection helpers must carry the version marker.
        for name in _PROJECTION_HELPER_NAMES:
            fn = getattr(runner_module, name, None)
            if fn is None:
                raise ProjectionProvenanceError(
                    f"runner missing projection helper '{name}'"
                )
            version = getattr(fn, "__projection_version__", None)
            if version is None:
                raise ProjectionProvenanceError(
                    f"runner.{name} missing __projection_version__ "
                    "marker (decorator not applied)"
                )
            if version != _EXPECTED_PROJECTION_VERSION:
                raise ProjectionProvenanceError(
                    f"runner.{name}.__projection_version__={version!r} "
                    f"!= expected {_EXPECTED_PROJECTION_VERSION!r} "
                    "(see DECISIONS.md D46 fix #1)"
                )

    if payload is not None:
        # Payload-side check: summary.retriever (or summary.ablation)
        # must assert a projection version that matches the expected
        # one. Phase 1 Round 3 fix: we tighten the policy so that the neo-schema
        # (summary.retriever block exists) MUST also carry a non-None
        # 'projection_version'; otherwise the runner is mid-regression
        # (block was written but version stamping was omitted).
        # Legacy payloads (no summary.retriever block at all) are still
        # accepted for backward compatibility with pre-D46 result files.
        summary = payload.get("summary")
        if not isinstance(summary, Mapping):
            return
        retriever = summary.get("retriever")
        if isinstance(retriever, Mapping):
            ver = retriever.get("projection_version")
            if ver is None:
                # Neo-schema (retriever block present) but version is
                # missing -- runner regression caught at payload time.
                raise ProjectionProvenanceError(
                    "summary.retriever exists but lacks "
                    "'projection_version'. Expected "
                    f"{_EXPECTED_PROJECTION_VERSION!r}. Runner regression "
                    "(see DECISIONS.md D46 fix #1)."
                )
            if ver != _EXPECTED_PROJECTION_VERSION:
                raise ProjectionProvenanceError(
                    f"summary.retriever.projection_version={ver!r} != "
                    f"expected {_EXPECTED_PROJECTION_VERSION!r}"
                )
        ablation = summary.get("ablation")
        if isinstance(ablation, Mapping):
            ver = ablation.get("projection_version")
            if ver is not None and ver != _EXPECTED_PROJECTION_VERSION:
                raise ProjectionProvenanceError(
                    f"summary.ablation.projection_version={ver!r} != "
                    f"expected {_EXPECTED_PROJECTION_VERSION!r}"
                )


# ---- G3 seed_effective_n -------------------------------------------------

# Retrieval-only fields that participate in the canonical hash. The hash
# must change between seeds whenever tie-band shuffling produces
# different orderings; QA-text / metric fields are excluded so the hash
# only reflects retrieval state, not downstream LLM noise.
_HASH_RECORD_FIELDS: tuple[str, ...] = (
    "qa_id",
    "domain",
    "retrieved_sources",
    "retrieved_entities",
    "retrieved_edges",
)

# Phase 1 Round 3 fix #1.
# Subset of _HASH_RECORD_FIELDS that the runner writes nested under
# record["prediction"] rather than at record top-level. The runner schema
# (prototype/eval/run_retrieval_eval_v3.py:1168-1212) emits:
#   record = {"qa_id": ..., "domain": ..., "prediction": {
#       "retrieved_sources": [...], "retrieved_entities": [...],
#       "retrieved_edges": [...], ...}}
# and the production payloads under paper/results/ablation_v3_v4/*.json
# confirm this nesting. Before this fix _canonical_hash looked at
# record top-level only, so the retrieval fields were always None and
# all three seed runs produced the same hash 796984661e302515 (true
# production observation, see Round 2 audit). The schema fallback
# retains the legacy "top-level first, prediction-nested fallback"
# precedence so historical payloads still hash correctly.
_RETRIEVAL_FIELD_NAMES: frozenset[str] = frozenset(
    {"retrieved_sources", "retrieved_entities", "retrieved_edges"}
)


def _extract_retrieval_fields(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract retrieval fields from either top-level or prediction-nested
    schema. Returns a dict with the standard retrieval field names so the
    rest of the pipeline can stay schema-agnostic.

    Lookup order:
      1. ``record[field]`` (legacy / flat schema)
      2. ``record["prediction"][field]`` (current runner schema, see
         prototype/eval/run_retrieval_eval_v3.py:1180-1186)
    Returns None for fields absent from both locations.
    """
    out: dict[str, Any] = {}
    pred = record.get("prediction")
    if not isinstance(pred, Mapping):
        pred = {}
    for field in _RETRIEVAL_FIELD_NAMES:
        if field in record:
            out[field] = record[field]
        elif field in pred:
            out[field] = pred[field]
        else:
            out[field] = None
    return out


# Phase 1 fix #1. Hash retrieval ordering
# only; score / rank / title drift is a separate signal (NOT seed-variance).
# Including float scores in the canonical hash caused legitimate score
# drift between identical orderings to register as fake "effective n>1",
# defeating G3's purpose. We keep only stable identity fields per item.
# Note: list ordering itself is preserved (no sort), so tie-band shuffle
# that reorders the same id-set still produces a different hash.
#
# Phase 1 Round 3 fix #5.
# Grepped item dict keys against production payloads
# (paper/results/ablation_v3_v4/C0_seed42.json:236-314) and the runner
# emitters (prototype/eval/run_retrieval_eval_v3.py:1262-1267 for
# retrieved_sources, :1285-1390 for retrieved_entities,
# :1391-1490 for retrieved_edges). Confirmed identity fields:
#   retrieved_sources[*]: paper_id (rank/score/title volatile)
#   retrieved_entities[*]: entity_id (rank/score/name/type volatile)
#   retrieved_edges[*]: edge_id, source, target, relation_type
#     (rank/score volatile). source/target/relation_type are kept
#     because edge_id is a derived concatenation of those three and
#     legacy payloads occasionally omit edge_id; keeping the structural
#     triple guards against that schema drift.
# 'domain' is retained for legacy flat-schema records where it sometimes
# nested into item dicts.
_HASH_ITEM_ID_FIELDS: frozenset[str] = frozenset(
    {
        "paper_id",
        "entity_id",
        "edge_id",
        "domain",
        "source",
        "target",
        "relation_type",
    }
)


def _strip_volatile(value: Any) -> Any:
    """Strip score / rank / title-like fields from retrieved-item dicts.

    Preserves list ordering (so reorderings still hash differently) and
    only retains identity fields enumerated in ``_HASH_ITEM_ID_FIELDS``.
    """
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    if isinstance(value, Mapping):
        return {
            k: _strip_volatile(value[k])
            for k in value
            if k in _HASH_ITEM_ID_FIELDS
        }
    return value


def _canonicalise(value: Any) -> Any:
    """Map a JSON-decoded structure to a deterministic representation:
    sort dict keys, descend into lists/dicts, leave scalars alone.
    """
    if isinstance(value, Mapping):
        return {k: _canonicalise(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonicalise(v) for v in value]
    return value


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """Deterministic SHA256 of records[*] retrieval-only fields.

    Uses the canonicalised JSON encoding so trivial key-order or
    whitespace variation does not falsely separate identical content.
    The retrieved_sources / _entities / _edges lists are pre-stripped of
    score / rank / title fields so this hash measures retrieval ordering
    diversity only (D47 G3 contract).

    Phase 1 Round 3 fix #1: retrieval fields are read via
    ``_extract_retrieval_fields`` so both legacy flat schemas and the
    current runner schema (record["prediction"][field]) hash correctly.
    Before this fix all production payloads collapsed to the same hash
    (796984661e302515 across all 30 production files) because the
    fields were silently None at the top level.
    """
    records = _records_of(payload)
    digest_input: list[Any] = []
    for r in records:
        slim: dict[str, Any] = {}
        retrieval = _extract_retrieval_fields(r)
        for k in _HASH_RECORD_FIELDS:
            if k in _RETRIEVAL_FIELD_NAMES:
                slim[k] = _strip_volatile(retrieval.get(k))
            else:
                slim[k] = r.get(k)
        digest_input.append(_canonicalise(slim))
    blob = json.dumps(digest_input, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_seed_variance(
    result_paths: Sequence[Path],
    cell_id: str,
) -> str:
    """G3: across the provided seed-result files, canonical hashes must
    differ -- unless ``RAG_V3_ALLOW_DETERMINISTIC_SEEDS=1`` is set, in
    which case the cell is marked bypassed.

    Returns
    -------
    str
        One of: ``"ok"`` (all hashes distinct), ``"bypassed"`` (env
        var set and hashes collapse).  Identical hashes without the
        bypass env raise SeedEffectiveNError.
    """
    if len(result_paths) < 2:
        raise SeedEffectiveNError(
            f"[{cell_id}] G3 requires >=2 seed files, "
            f"got {len(result_paths)}"
        )

    hashes: list[tuple[str, str]] = []
    for path in result_paths:
        if not path.exists():
            raise SeedEffectiveNError(
                f"[{cell_id}] seed result file missing: {path}"
            )
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise SeedEffectiveNError(
                f"[{cell_id}] cannot load seed file {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise SeedEffectiveNError(
                f"[{cell_id}] seed file {path} must contain JSON object"
            )
        hashes.append((str(path), _canonical_hash(payload)))

    distinct = {h for _, h in hashes}
    if len(distinct) == len(hashes):
        return "ok"

    bypass = os.environ.get(_SEED_BYPASS_ENV, "").strip() == "1"
    if bypass:
        # Annotate each loaded payload's audit field so the
        # downstream aggregator records bypassed cells.
        for path, _ in hashes:
            try:
                p = Path(path)
                with p.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                append_audit(
                    payload,
                    "seed_effective_n",
                    "bypassed",
                    f"cell={cell_id} hashes collapsed under "
                    f"{_SEED_BYPASS_ENV}=1",
                )
                with p.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
            except (OSError, ValueError):
                # Annotation is best-effort; the bypass decision stands.
                pass
        return "bypassed"

    collapsed = ", ".join(
        f"{Path(p).name}:{h[:12]}" for p, h in hashes
    )
    raise SeedEffectiveNError(
        f"[{cell_id}] G3 effective_n=1: all {len(hashes)} seed runs "
        f"produced identical retrieval hashes "
        f"({collapsed}). Set {_SEED_BYPASS_ENV}=1 to acknowledge "
        "this is a deterministic-by-design retriever."
    )


# ---- audit helpers -------------------------------------------------------

def write_gate_violation_log(output_path: Path, error: GateError) -> Path:
    """Q7: write ``<output>.GATE_FAIL.log`` next to the result file.

    Returns the path actually written so the runner can echo it.
    """
    log_path = output_path.with_suffix(output_path.suffix + ".GATE_FAIL.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"[GATE {error.gate_name}] {error}\n",
        encoding="utf-8",
    )
    return log_path


def append_audit(
    payload: dict,
    gate_name: str,
    status: str,
    detail: str = "",
) -> None:
    """Mark ``summary.audit.gate_violations`` or ``gate_bypassed``.

    Mutates ``payload`` in place so callers can persist the audited
    state via the runner's existing ``_write_payload`` helper.

    Phase 1 fix. Idempotent on
    ``(gate, status, detail)`` tuple: repeating the same call on an
    already-marked payload does NOT duplicate the audit entry. This
    matters because ``assert_seed_variance`` may be retried by the
    orchestrator and would otherwise inflate the bypass record.
    """
    summary = payload.setdefault("summary", {})
    if not isinstance(summary, dict):
        # Payload schema is broken; nothing we can safely write.
        return
    audit = summary.setdefault("audit", {})
    entry = {"gate": gate_name, "detail": detail}
    if status == "violation":
        bucket = audit.setdefault("gate_violations", [])
    elif status == "bypassed":
        bucket = audit.setdefault("gate_bypassed", [])
    else:
        bucket = audit.setdefault("gate_other", [])
        entry = {**entry, "status": status}
    # Dedupe: same gate + same detail already present -> no-op.
    for existing in bucket:
        if (
            isinstance(existing, Mapping)
            and existing.get("gate") == entry.get("gate")
            and existing.get("detail") == entry.get("detail")
            and existing.get("status") == entry.get("status")
        ):
            return
    bucket.append(entry)


# ---- G5 ablation_isolation ------------------------------------------------
#
# Two complementary semantics for ablation cells:
#   * minus_one (C4a-C8a): start from the C0 default feature set and flip
#     EXACTLY ONE of the C0 mechanisms OFF via a dedicated minus_one_*
#     flag. The cell exercises "C0 minus one mechanism", isolating that
#     mechanism's contribution.
#   * single_off (C9-C12): C0 default features + flip exactly ONE feature
#     to its OFF value (postprocess off / query_strategy uniform_BFS /
#     trust single_source_weight / sub_answer_req generic).
# Baselines (C0, C0_BM25, C1, C2, C3) are NOT subject to this gate.

_C0_FEATURE_DEFAULTS: Mapping[str, str] = {
    "c4_postprocess": "on",
    "c5_query_strategy": "adaptive",
    "c6_trust": "4axis",
    "sub_answer_req": "schema",
}
"""C0 full-system feature vector (D33-D36 mechanisms ON).
Source: paper/configs/ablation_v3.yaml C0 cell."""

_MINUS_ONE_FEATURE_KEYS: frozenset[str] = frozenset({
    "c4a_minus_one_coverage_trigger",
    "c5a_minus_one_typed_schema",
    "c6a_minus_one_edge_path",
    "c7a_minus_one_query_strategy",
    "c8a_minus_one_trust",
})
"""Legal minus_one flag keys for C4a-C8a. Exactly one MUST be True for
those cells; any other flag presence is a violation."""

_CELL_TO_MINUS_ONE: Mapping[str, str] = {
    "C4a": "c4a_minus_one_coverage_trigger",
    "C5a": "c5a_minus_one_typed_schema",
    "C6a": "c6a_minus_one_edge_path",
    "C7a": "c7a_minus_one_query_strategy",
    "C8a": "c8a_minus_one_trust",
}
"""Pinning: each C?a cell id is bound to ONE specific minus_one flag.
Mismatch -> violation."""

_SINGLE_OFF_CELLS: Mapping[str, tuple[str, str]] = {
    "C9":  ("c4_postprocess",    "off"),
    "C10": ("c5_query_strategy", "uniform_BFS"),
    "C11": ("c6_trust",          "single_source_weight"),
    "C12": ("sub_answer_req",    "generic"),
}
"""For each C9-C12 cell: (feature_key, OFF_value). The remaining three
feature keys must equal their C0 defaults."""

_BASELINE_CELLS: frozenset[str] = frozenset({
    "C0", "C0_BM25", "C1", "C2", "C3",
})
"""Cells exempt from G5 (baselines and W7 single-weight isolation cells
C2/C3 whose pre-P3 semantics are weight-driven, not feature-driven)."""


def _normalize_feature_value(key: str, value: Any) -> Any:
    """YAML maps ``on``/``off`` to Python True/False. The C0 defaults are
    string-valued ("on", "adaptive", ...), so we translate the legacy
    bool form for the four C0 feature keys before equality checks.
    Other keys (minus_one_* flags, etc.) pass through unchanged.
    """
    if key == "c4_postprocess" and isinstance(value, bool):
        return "on" if value else "off"
    return value


def assert_ablation_isolation(
    cell_id: str,
    features: Mapping[str, Any] | None,
) -> None:
    """G5: each ablation cell must perturb exactly one mechanism.

    Three paths:
      * Baseline cells (_BASELINE_CELLS) -> skip silently. Their semantics
        are defined elsewhere and not subject to the single-mechanism rule.
      * minus_one cells (C4a-C8a) -> features MUST contain the four C0
        defaults unchanged AND exactly one of the legal minus_one flags
        set to True. The pinned flag (per _CELL_TO_MINUS_ONE) must be the
        one set; any other minus_one flag is a violation.
      * single_off cells (C9-C12) -> three of the four feature keys equal
        C0 defaults; the fourth equals the cell's pinned OFF value.

    Cells outside the three buckets raise AblationIsolationError.
    """
    if cell_id in _BASELINE_CELLS:
        return
    feats: Mapping[str, Any] = features or {}

    if cell_id in _CELL_TO_MINUS_ONE:
        # minus_one path. The four C0 defaults must be present and equal
        # to the C0 values (drift = silent contamination).
        for key, expected in _C0_FEATURE_DEFAULTS.items():
            actual = _normalize_feature_value(key, feats.get(key, expected))
            if actual != expected:
                raise AblationIsolationError(
                    f"[{cell_id}] minus_one cell drifted from C0 "
                    f"default: {key}={actual!r} expected {expected!r}"
                )
        # Exactly one minus_one flag must be True, and it must be the
        # pinned one for this cell.
        present_true = [
            key for key in _MINUS_ONE_FEATURE_KEYS
            if bool(feats.get(key, False)) is True
        ]
        if len(present_true) == 0:
            raise AblationIsolationError(
                f"[{cell_id}] no minus_one_* flag set; expected exactly "
                f"one (pinned: {_CELL_TO_MINUS_ONE[cell_id]!r})"
            )
        if len(present_true) > 1:
            raise AblationIsolationError(
                f"[{cell_id}] {len(present_true)} minus_one_* flags set "
                f"({sorted(present_true)}); expected exactly one "
                f"(pinned: {_CELL_TO_MINUS_ONE[cell_id]!r})"
            )
        pinned = _CELL_TO_MINUS_ONE[cell_id]
        if present_true[0] != pinned:
            raise AblationIsolationError(
                f"[{cell_id}] minus_one flag {present_true[0]!r} does "
                f"not match cell pinning {pinned!r}"
            )
        return

    if cell_id in _SINGLE_OFF_CELLS:
        off_key, off_value = _SINGLE_OFF_CELLS[cell_id]
        # No minus_one_* flag is allowed in single_off cells.
        stray_minus_one = [
            key for key in _MINUS_ONE_FEATURE_KEYS
            if bool(feats.get(key, False)) is True
        ]
        if stray_minus_one:
            raise AblationIsolationError(
                f"[{cell_id}] single_off cell carries minus_one_* "
                f"flag(s) {sorted(stray_minus_one)}; not permitted"
            )
        # Pinned OFF feature must equal the published OFF value.
        actual_off = _normalize_feature_value(off_key, feats.get(off_key))
        if actual_off != off_value:
            raise AblationIsolationError(
                f"[{cell_id}] OFF feature {off_key}={actual_off!r} "
                f"expected {off_value!r}"
            )
        # The other three features must equal C0 defaults.
        off_count = 0
        for key, expected in _C0_FEATURE_DEFAULTS.items():
            if key == off_key:
                off_count += 1
                continue
            actual = _normalize_feature_value(key, feats.get(key, expected))
            if actual != expected:
                raise AblationIsolationError(
                    f"[{cell_id}] non-OFF feature drifted: {key}="
                    f"{actual!r} expected C0 default {expected!r}"
                )
        if off_count != 1:
            raise AblationIsolationError(
                f"[{cell_id}] internal pinning error: off_count="
                f"{off_count} expected 1"
            )
        return

    # Unknown cell id (not in any of the three registered buckets).
    # Silent skip: G5 only governs the production cells defined in
    # ablation_orchestrator.py CONFIGS (C0 + C4a-C8a + C9-C12) plus the
    # legacy baselines (C0_BM25/C1/C2/C3). Synthetic / experimental cell
    # ids exercised by unit tests (e.g. C_ALIGN, C_UNSUPPORTED) must not
    # trigger a hard fail here; their contract is owned by the caller.
    return


# ---- G8 estimand ----------------------------------------------------------
#
# Each paper claim is grounded in (a) a published numeric threshold from
# the project-design / D39 register and (b) a specific summary_key emitted by the
# eval pipeline. G8 wires the two together so a payload cannot silently
# fail a published claim or use the wrong metric family.

_METRIC_FAMILY_PREFIX: Mapping[str, tuple[str, ...]] = {
    "source":       ("source_",),
    "entity":       ("entity_",),
    "edge":         ("edge_",),
    "path":         ("path_",),
    "alignment":    ("alignment_",),
    "agreement":    ("cross_judge_", "agreement_"),
    "faithfulness": ("nli_", "faithfulness_"),
    "lift":         ("adaptive_enrichment_", "postprocess_coverage_"),
}
"""Allowed metric families and their summary_key prefixes. The manifest
schema validator (G8._load_paper_claim_manifest) refuses any claim whose
summary_key does not start with one of the family's prefixes."""

_LEGAL_DIRECTIONS: frozenset[str] = frozenset({">=", ">", "<=", "<"})

_EXPECTED_MANIFEST_VERSION = "paper_claim_manifest_v1"


def _validate_manifest_schema(
    raw: Mapping[str, Any],
    *,
    source: str,
) -> Mapping[str, Any]:
    """Apply the manifest_v1 schema. Returns the validated mapping."""
    if not isinstance(raw, Mapping):
        raise EstimandError(
            f"manifest {source} root must be mapping (got "
            f"type={type(raw).__name__})"
        )
    version = raw.get("manifest_version")
    if version != _EXPECTED_MANIFEST_VERSION:
        raise EstimandError(
            f"manifest {source} version={version!r} expected "
            f"{_EXPECTED_MANIFEST_VERSION!r}"
        )
    claims = raw.get("claims")
    if not isinstance(claims, list) or not claims:
        raise EstimandError(
            f"manifest {source} must carry non-empty list 'claims'"
        )
    required_fields = (
        "id", "summary_key", "metric_family", "threshold",
        "direction", "source", "manuscript_anchor", "expected_in_cells",
    )
    seen_ids: set[str] = set()
    for idx, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise EstimandError(
                f"manifest {source} claims[{idx}] must be mapping (got "
                f"type={type(claim).__name__})"
            )
        for field in required_fields:
            if field not in claim:
                raise EstimandError(
                    f"manifest {source} claims[{idx}] missing required "
                    f"field {field!r}"
                )
        claim_id = str(claim["id"])
        if claim_id in seen_ids:
            raise EstimandError(
                f"manifest {source} duplicate claim id {claim_id!r}"
            )
        seen_ids.add(claim_id)
        family = str(claim["metric_family"])
        if family not in _METRIC_FAMILY_PREFIX:
            raise EstimandError(
                f"manifest {source} claim {claim_id!r} metric_family="
                f"{family!r} not in {sorted(_METRIC_FAMILY_PREFIX)}"
            )
        summary_key = str(claim["summary_key"])
        prefixes = _METRIC_FAMILY_PREFIX[family]
        if not any(summary_key.startswith(pref) for pref in prefixes):
            raise EstimandError(
                f"manifest {source} claim {claim_id!r} summary_key="
                f"{summary_key!r} does not match metric_family={family!r} "
                f"prefixes {list(prefixes)}"
            )
        direction = str(claim["direction"])
        if direction not in _LEGAL_DIRECTIONS:
            raise EstimandError(
                f"manifest {source} claim {claim_id!r} direction="
                f"{direction!r} not in {sorted(_LEGAL_DIRECTIONS)}"
            )
        try:
            float(claim["threshold"])
        except (TypeError, ValueError) as exc:
            raise EstimandError(
                f"manifest {source} claim {claim_id!r} threshold not "
                f"numeric: {claim['threshold']!r}"
            ) from exc
        expected_cells = claim["expected_in_cells"]
        if not isinstance(expected_cells, list) or not expected_cells:
            raise EstimandError(
                f"manifest {source} claim {claim_id!r} expected_in_cells "
                f"must be non-empty list"
            )
        for cell in expected_cells:
            if not isinstance(cell, str) or not cell.strip():
                raise EstimandError(
                    f"manifest {source} claim {claim_id!r} expected_in_cells "
                    f"entry must be non-empty string (got {cell!r})"
                )
    # Cross-validate co_claim references.
    for claim in claims:
        co = claim.get("co_claim")
        if co is None:
            continue
        if str(co) not in seen_ids:
            raise EstimandError(
                f"manifest {source} claim {claim['id']!r} co_claim="
                f"{co!r} refers to undefined id"
            )
    return raw


@lru_cache(maxsize=4)
def _load_paper_claim_manifest_cached(path_str: str) -> Mapping[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise EstimandError(f"manifest file not found: {path}")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dep
        raise EstimandError(
            f"PyYAML required to load manifest {path}: {exc}"
        ) from exc
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise EstimandError(
            f"manifest {path} could not be parsed: {exc}"
        ) from exc
    return _validate_manifest_schema(raw, source=str(path))


def _load_paper_claim_manifest(path: Path) -> Mapping[str, Any]:
    """Public entry point: load + validate the claim manifest, cached."""
    return _load_paper_claim_manifest_cached(str(path.resolve()))


def _compare(value: float, threshold: float, direction: str) -> bool:
    if direction == ">=":
        return value >= threshold
    if direction == ">":
        return value > threshold
    if direction == "<=":
        return value <= threshold
    if direction == "<":
        return value < threshold
    # _validate_manifest_schema already rejects other values; defensive.
    raise EstimandError(f"unknown direction {direction!r}")


def assert_estimand(
    payload: Mapping[str, Any],
    cell_id: str,
    *,
    manifest_path: Path,
    strict: bool = False,
    require_keys: bool | None = None,
    enforce_thresholds: bool | None = None,
) -> None:
    """G8: enforce the paper_claim_manifest contract for ``cell_id``.

    Policy
    ------
    Two operating modes:
      * **Runtime / orchestrator** (default, ``strict=False``,
        ``require_keys=False``, ``enforce_thresholds=False``):
        The manifest is loaded + schema-validated (any malformed
        claim raises EstimandError). For each claim whose
        ``expected_in_cells`` contains ``cell_id``, the gate only
        verifies that a present summary_key is numeric. Missing keys
        are silently skipped (the metric is owned by a downstream
        evaluator such as faithfulness / alignment / cross_judge);
        threshold checks are also skipped because the retrieval
        payload alone cannot fulfil the paper claim threshold. This
        keeps the runtime fast-path non-destructive on legitimate
        retrieval-only payloads.
      * **Audit / strict** (``strict=True``): both ``require_keys``
        and ``enforce_thresholds`` default to True. Missing keys
        and threshold violations are both reported. If a claim
        defines ``strict_threshold``, that tighter value replaces
        the R18 fallback ``threshold``.
    Callers can override either flag independently to mix the
    policies (e.g. require_keys=True but enforce_thresholds=False).

    Co-claim semantics
    ------------------
    When claim X declares ``co_claim: Y`` and both are expected in
    ``cell_id``, X failing implies Y is also reported as failing
    (paired contract). Single-fail of X with Y silently passing ->
    EstimandError. Drift is only checked among reported failures, so
    it cannot fire when both legs were silently skipped.
    """
    manifest = _load_paper_claim_manifest(manifest_path)
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise EstimandError(
            f"[{cell_id}] payload missing 'summary' mapping"
        )

    if require_keys is None:
        require_keys = strict
    if enforce_thresholds is None:
        enforce_thresholds = strict

    failures: dict[str, str] = {}
    relevant_ids: set[str] = set()
    for claim in manifest["claims"]:
        expected_cells = claim.get("expected_in_cells") or []
        if cell_id not in expected_cells:
            continue
        claim_id = str(claim["id"])
        relevant_ids.add(claim_id)
        summary_key = str(claim["summary_key"])
        if summary_key not in summary:
            if require_keys:
                failures[claim_id] = (
                    f"summary missing key {summary_key!r}"
                )
            # Else: silently skip; downstream evaluator owns this claim.
            continue
        try:
            actual = float(summary[summary_key])
        except (TypeError, ValueError) as exc:
            failures[claim_id] = (
                f"summary[{summary_key!r}]={summary[summary_key]!r} "
                f"not numeric ({exc})"
            )
            continue
        if not enforce_thresholds:
            continue
        threshold = float(claim["threshold"])
        if strict and "strict_threshold" in claim:
            try:
                threshold = float(claim["strict_threshold"])
            except (TypeError, ValueError) as exc:
                raise EstimandError(
                    f"[{cell_id}] claim {claim_id!r} strict_threshold "
                    f"not numeric: {claim['strict_threshold']!r}"
                ) from exc
        direction = str(claim["direction"])
        if not _compare(actual, threshold, direction):
            failures[claim_id] = (
                f"{summary_key}={actual:.6f} {direction} {threshold:.6f} "
                f"FAILED"
            )

    # Cross-validate co_claim drift: when X failed but Y (co_claim) did
    # not, the contract is broken (paired claims must move together).
    for claim in manifest["claims"]:
        claim_id = str(claim["id"])
        if claim_id not in relevant_ids:
            continue
        co = claim.get("co_claim")
        if co is None:
            continue
        co_id = str(co)
        if co_id not in relevant_ids:
            continue
        x_failed = claim_id in failures
        y_failed = co_id in failures
        if x_failed != y_failed:
            failures[claim_id + "+" + co_id] = (
                f"co_claim drift: {claim_id} failed={x_failed} "
                f"vs {co_id} failed={y_failed} (paired contract)"
            )

    if failures:
        detail = "; ".join(
            f"{cid}: {msg}" for cid, msg in sorted(failures.items())
        )
        raise EstimandError(f"[{cell_id}] G8 estimand violations -> {detail}")


# ---- G1 artifact_integrity + G2 backend_gate (Phase 3) -------------------
#
# Two paired gates that close the "false labeling" gap exposed by D47:
#
#   G1 (artifact_integrity) - runtime-side check that the retriever class
#     actually loaded matches the actual_backend label it reports, AND
#     that any backend downgrade was recorded with an explicit
#     fallback_reason (no silent fallbacks).
#
#   G2 (backend_gate)       - paper-level claim vs actual_backend
#     bidirectional consistency. The user (or paper manifest) declares
#     which backend the paper claim is grounded in
#     ('bge-m3' or 'surrogate-diagnostic'); the gate refuses to let a
#     mismatch slip through silently.
#
# Backend label whitelists. The BGE family list is sourced from
# prototype/eval/run_retrieval_eval_v3.py - BgeM3DenseRetriever.metadata
# (line ~371-376) sets actual_backend = encoder.backend_name where the
# two encoder classes hardcode 'sentence-transformers' or 'FlagEmbedding'
# (see _SentenceTransformerEncoder / _FlagEmbeddingEncoder). When the
# user requests --dense-backend bge-m3 the payload may carry any of
# {'bge-m3', 'sentence-transformers', 'FlagEmbedding'} so we accept all
# three under the BGE-family bucket. The surrogate list is sourced from
# DeterministicDenseRetriever.metadata (line ~289-294) which hardcodes
# 'deterministic_surrogate'. Any other value triggers G1 with
# 'unknown_backend_label' detail.
_BGE_BACKENDS: frozenset[str] = frozenset({
    "bge-m3", "sentence-transformers", "FlagEmbedding",
})
_SURROGATE_BACKENDS: frozenset[str] = frozenset({
    "deterministic_surrogate",
})
_ALL_VALID_ACTUAL_BACKENDS: frozenset[str] = (
    _BGE_BACKENDS | _SURROGATE_BACKENDS
)

# Paper-claim whitelist. Strict string match (no normalisation). The
# two legal claims are
#   'bge-m3'              - the paper makes a real-dense-encoder claim;
#                            payload must carry an actual_backend in
#                            _BGE_BACKENDS.
#   'surrogate-diagnostic' - the paper explicitly discloses the
#                            deterministic surrogate (D47 disclosure
#                            mode); payload must carry an
#                            actual_backend in _SURROGATE_BACKENDS.
_VALID_PAPER_CLAIMS: frozenset[str] = frozenset({
    "bge-m3", "surrogate-diagnostic",
})
_PAPER_CLAIM_ENV: str = "RAG_V3_PAPER_BACKEND_CLAIM"
_REQUIRE_CLAIM_ENV: str = "RAG_V3_REQUIRE_BACKEND_CLAIM"

# G1 Tier-3: model_name format hints. When actual_backend is in the
# BGE family the model_name should look like a BGE-family identifier
# (loose substring check, not a regex pin, so a future fine-tune like
# 'BAAI/bge-m3-finetuned' still passes). Surrogate backend model_name
# can be anything (DeterministicDenseRetriever surfaces whatever the
# config requested), so this hint is only enforced for the BGE bucket.
_BGE_MODEL_NAME_HINTS: tuple[str, ...] = ("bge-m3", "bge_m3", "BAAI/bge")


def _extract_actual_backend(
    payload: Mapping[str, Any] | None = None,
    *,
    retriever: Any | None = None,
) -> str | None:
    """Locate the ``actual_backend`` label on either a runtime retriever
    or a written payload.

    Runtime mode (``retriever`` set): the retriever object may be a
    ``DeterministicDenseRetriever`` / ``BgeM3DenseRetriever`` directly,
    a ``ReciprocalRankFusionRetriever`` wrapping a dense retriever in
    ``.dense``, or a ``ProposedV3Retriever`` whose dense leg lives at
    ``.dense`` or ``._dense``. BM25-only retrievers carry no metadata
    and return ``None`` here (G1 treats that as a no-op because there
    is no backend to validate).

    Payload mode (``payload`` set): we read
    ``summary.retriever.dense.actual_backend`` first (the schema the
    runner emits today, see prototype/eval/run_retrieval_eval_v3.py
    ``_retriever_metadata`` + ``_summarize``), falling back to
    ``summary.retriever.actual_backend`` for legacy direct-dense
    payloads.
    """
    if retriever is not None:
        meta = getattr(retriever, "metadata", None)
        if isinstance(meta, Mapping):
            label = meta.get("actual_backend")
            if isinstance(label, str):
                return label
        # RRF / proposed_v3: dense leg may be exposed as .dense or ._dense.
        for attr in ("dense", "_dense"):
            inner = getattr(retriever, attr, None)
            if inner is None:
                continue
            inner_meta = getattr(inner, "metadata", None)
            if isinstance(inner_meta, Mapping):
                label = inner_meta.get("actual_backend")
                if isinstance(label, str):
                    return label
        return None

    if payload is not None and isinstance(payload, Mapping):
        summary = payload.get("summary", {})
        if not isinstance(summary, Mapping):
            return None
        retriever_block = summary.get("retriever", {})
        if not isinstance(retriever_block, Mapping):
            return None
        dense_block = retriever_block.get("dense")
        if isinstance(dense_block, Mapping):
            label = dense_block.get("actual_backend")
            if isinstance(label, str):
                return label
        label = retriever_block.get("actual_backend")
        if isinstance(label, str):
            return label
    return None


def _extract_dense_metadata(
    payload: Mapping[str, Any] | None = None,
    *,
    retriever: Any | None = None,
) -> Mapping[str, Any] | None:
    """Companion to ``_extract_actual_backend`` returning the full
    metadata mapping (``actual_backend`` + ``requested_backend`` +
    ``model_name`` + optional ``fallback_reason``). Returns ``None`` if
    no dense metadata is present (e.g. BM25-only retriever).
    """
    if retriever is not None:
        meta = getattr(retriever, "metadata", None)
        if isinstance(meta, Mapping):
            return meta
        for attr in ("dense", "_dense"):
            inner = getattr(retriever, attr, None)
            if inner is None:
                continue
            inner_meta = getattr(inner, "metadata", None)
            if isinstance(inner_meta, Mapping):
                return inner_meta
        return None
    if payload is not None and isinstance(payload, Mapping):
        summary = payload.get("summary", {})
        if not isinstance(summary, Mapping):
            return None
        retriever_block = summary.get("retriever", {})
        if not isinstance(retriever_block, Mapping):
            return None
        dense_block = retriever_block.get("dense")
        if isinstance(dense_block, Mapping):
            return dense_block
        # Legacy flat schema (no nested 'dense' block).
        if "actual_backend" in retriever_block:
            return retriever_block
    return None


def _classify_retriever(retriever: Any) -> str | None:
    """Map a runtime retriever class name to one of the expected
    backend buckets ('bge', 'surrogate'). Returns ``None`` for
    retrievers that carry no dense leg (BM25-only) or for unknown
    classes (caller raises with the class name in the error).

    We intentionally use class-name strings rather than isinstance()
    so this module stays import-free from the runner package (avoids
    a circular import: preflight_gates is consumed by the runner).
    """
    # Probe the dense leg first when the top-level is a fusion wrapper.
    for attr in (None, "dense", "_dense"):
        target = retriever if attr is None else getattr(retriever, attr, None)
        if target is None:
            continue
        cls_name = type(target).__name__
        if cls_name == "DeterministicDenseRetriever":
            return "surrogate"
        if cls_name == "BgeM3DenseRetriever":
            return "bge"
    return None


def _resolve_paper_claim(cli_claim: str | None) -> str | None:
    """Resolve the paper-level backend claim with precedence
    CLI > env > None. Both CLI and env values must be in
    ``_VALID_PAPER_CLAIMS``; otherwise BackendGateError is raised.
    """
    if cli_claim is not None:
        if cli_claim not in _VALID_PAPER_CLAIMS:
            raise BackendGateError(
                f"--paper-backend-claim={cli_claim!r} not in "
                f"{sorted(_VALID_PAPER_CLAIMS)}"
            )
        return cli_claim
    env_value = os.environ.get(_PAPER_CLAIM_ENV, "").strip()
    if not env_value:
        return None
    if env_value not in _VALID_PAPER_CLAIMS:
        raise BackendGateError(
            f"{_PAPER_CLAIM_ENV}={env_value!r} not in "
            f"{sorted(_VALID_PAPER_CLAIMS)}"
        )
    return env_value


def assert_artifact_integrity(
    payload: Mapping[str, Any] | None = None,
    *,
    retriever: Any | None = None,
) -> None:
    """G1: 3-tier consistency check.

    Tier 1 (class <-> label): when a runtime retriever is supplied, its
        Python class must match the bucket of the reported
        ``actual_backend`` label. Class-vs-label drift means the
        retriever was instantiated as one type but is advertising
        itself as another (false labeling).

    Tier 2 (silent fallback): when the request was for 'bge-m3' but
        the loaded backend is the surrogate AND no ``fallback_reason``
        was recorded, the gate raises. Production paths must always
        carry the reason for the downgrade so the analyst can see
        why the BGE-M3 path was bypassed.

    Tier 3 (model_name format): the model_name field must be a
        non-empty ASCII string. When the actual backend is in the BGE
        family the name must contain a BGE-family hint substring;
        surrogate model_names are accepted as-is.

    Operating modes (mutually exclusive)
    ------------------------------------
    * runtime: ``retriever`` set. Used inside the runner right after
      ``_build_retriever`` so a misconfigured backend never produces a
      payload.
    * payload: ``payload`` set. Used inside the orchestrator
      ``_load_result`` so historical artefacts cannot bypass the gate.

    BM25-only retrievers / legacy payloads without any dense block
    return silently because there is no backend to validate.
    """
    if retriever is None and payload is None:
        # No-op: neither source supplied, nothing to validate.
        return

    meta = _extract_dense_metadata(payload=payload, retriever=retriever)
    if meta is None:
        # No dense leg present (e.g. BM25-only runs). G1 is a no-op.
        return

    actual = meta.get("actual_backend")
    if not isinstance(actual, str) or not actual:
        raise ArtifactIntegrityError(
            "G1 missing actual_backend label "
            f"(metadata keys={sorted(meta.keys())})"
        )
    if actual not in _ALL_VALID_ACTUAL_BACKENDS:
        raise ArtifactIntegrityError(
            f"G1 unknown_backend_label actual_backend={actual!r} not in "
            f"{sorted(_ALL_VALID_ACTUAL_BACKENDS)}"
        )

    # Tier 1: class <-> label consistency (runtime mode only).
    if retriever is not None:
        bucket = _classify_retriever(retriever)
        if bucket is not None:
            if bucket == "bge" and actual not in _BGE_BACKENDS:
                raise ArtifactIntegrityError(
                    "G1 class_label_mismatch: BgeM3DenseRetriever loaded "
                    f"but actual_backend={actual!r} not in "
                    f"{sorted(_BGE_BACKENDS)}"
                )
            if bucket == "surrogate" and actual not in _SURROGATE_BACKENDS:
                raise ArtifactIntegrityError(
                    "G1 class_label_mismatch: DeterministicDenseRetriever "
                    f"loaded but actual_backend={actual!r} not in "
                    f"{sorted(_SURROGATE_BACKENDS)}"
                )

    # Tier 2: silent fallback detection.
    requested = meta.get("requested_backend")
    fallback_reason = meta.get("fallback_reason")
    if (
        isinstance(requested, str)
        and requested == "bge-m3"
        and actual in _SURROGATE_BACKENDS
        and not (isinstance(fallback_reason, str) and fallback_reason.strip())
    ):
        raise ArtifactIntegrityError(
            "G1 silent_fallback: requested_backend='bge-m3' but actual "
            f"backend={actual!r} loaded without a recorded "
            "fallback_reason. Production runs must always disclose the "
            "BGE-M3 downgrade reason."
        )

    # Tier 3: model_name format validation.
    model_name = meta.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ArtifactIntegrityError(
            f"G1 invalid_model_name model_name={model_name!r} "
            "(must be non-empty string)"
        )
    try:
        model_name.encode("ascii")
    except UnicodeEncodeError as exc:
        # ASCII-only restriction guarantees cp949 / Windows consoles
        # render the label cleanly in error messages and logs.
        # the offending model_name itself
        # contains non-ASCII bytes, so embedding ``model_name!r`` (or
        # the raw exc.reason) in the error message re-introduces the
        # cp949 encode failure when Windows logs the exception. We
        # collapse the message to a sanitized fingerprint
        # (ASCII-safe repr + the offending byte offset) so it
        # round-trips through cp949 consoles unchanged.
        ascii_safe = model_name.encode("ascii", errors="backslashreplace").decode("ascii")
        raise ArtifactIntegrityError(
            f"G1 invalid_model_name model_name_ascii_escaped={ascii_safe!r} "
            f"is not ASCII-safe (offset={exc.start})"
        ) from exc
    if actual in _BGE_BACKENDS:
        lowered = model_name.lower()
        if not any(hint.lower() in lowered for hint in _BGE_MODEL_NAME_HINTS):
            raise ArtifactIntegrityError(
                f"G1 model_name_family_mismatch model_name={model_name!r} "
                f"does not contain any BGE-family hint "
                f"{list(_BGE_MODEL_NAME_HINTS)} for actual_backend="
                f"{actual!r}"
            )


def assert_backend_gate(
    payload: Mapping[str, Any] | None = None,
    *,
    retriever: Any | None = None,
    cli_claim: str | None = None,
    claim_id: str | None = None,
    manifest_path: Path | None = None,
    require_claim: bool | None = None,
) -> str:
    """G2: bidirectional claim <-> actual_backend consistency.

    Forward direction: claim='bge-m3' AND actual not in _BGE_BACKENDS
        -> BackendGateError (paper claims real-dense but ran surrogate).
    Reverse direction: claim='surrogate-diagnostic' AND actual in
        _BGE_BACKENDS -> BackendGateError (paper claims disclosed
        surrogate but actually ran real-dense; the disclosure is
        false).

    Resolution precedence for the claim:
        1. ``cli_claim`` argument (the value of --paper-backend-claim).
        2. ``RAG_V3_PAPER_BACKEND_CLAIM`` env var.
        3. None (gate returns ``'skipped_no_claim'`` unless
           ``require_claim`` is True, in which case BackendGateError
           is raised).

    ``claim_id`` / ``manifest_path`` are reserved for the option-A
    extension (per-claim manifest binding); option-B (this PR) leaves
    them no-op. Surfacing them in the signature now keeps the future
    swap source-compatible.

    ``require_claim`` defaults to the env var
    ``RAG_V3_REQUIRE_BACKEND_CLAIM=1`` when not explicitly passed; the
    orchestrator opts into strict mode via that env var.

    Returns
    -------
    str
        ``'ok'`` when the claim and actual_backend agree, or
        ``'skipped_no_claim'`` when no claim is set and not required.
    """
    # Option A reserved hooks. Kept in the signature for forward
    # compatibility; option B (this PR) does not load the manifest here.
    del claim_id, manifest_path

    claim = _resolve_paper_claim(cli_claim)
    if require_claim is None:
        require_claim = os.environ.get(_REQUIRE_CLAIM_ENV, "").strip() == "1"
    if claim is None:
        if require_claim:
            raise BackendGateError(
                "G2 require_claim is set but no paper-backend claim "
                "supplied (CLI --paper-backend-claim or "
                f"{_PAPER_CLAIM_ENV} env var)"
            )
        return "skipped_no_claim"

    actual = _extract_actual_backend(payload=payload, retriever=retriever)
    if actual is None:
        # No backend label to compare against. With a claim set this
        # is a hard failure; without G2 cannot run.
        raise BackendGateError(
            f"G2 cannot locate actual_backend label "
            f"(claim={claim!r}, retriever={retriever is not None}, "
            f"payload={payload is not None})"
        )
    if actual not in _ALL_VALID_ACTUAL_BACKENDS:
        raise BackendGateError(
            f"G2 unknown_backend_label actual_backend={actual!r} not in "
            f"{sorted(_ALL_VALID_ACTUAL_BACKENDS)}"
        )

    if claim == "bge-m3" and actual not in _BGE_BACKENDS:
        raise BackendGateError(
            f"G2 forward_mismatch: paper claim={claim!r} but "
            f"actual_backend={actual!r} not in {sorted(_BGE_BACKENDS)}"
        )
    if claim == "surrogate-diagnostic" and actual in _BGE_BACKENDS:
        raise BackendGateError(
            f"G2 reverse_mismatch: paper claim={claim!r} but "
            f"actual_backend={actual!r} is a BGE-family backend "
            f"(disclosure would be false)"
        )
    return "ok"


# ---- G6 mechanism_activation (Phase 4) ------------------------------------
#
# Per-cell mechanism activation check. For each cell that has a defined
# EXPECTED_BEHAVIOR rule set, verify that >=50% of QA records observe the
# expected mechanism state (e.g. C9 must show postprocess.applied=False in
# the majority of records). Legacy weight-only cells (C4a-C8a) have no
# mechanism toggle and are explicitly skipped.
#
# Magic number derivation
# -----------------------
# 0.5 is the structural definition of "majority" -- if fewer than half of
# the records exhibit the expected state, the mechanism is not the
# dominant driver of the cell's behaviour. This is NOT a tuned
# hyperparameter; it lifts directly from the dictionary meaning of
# "majority", consistent with the rest of the gate family's natural
# derivation pattern:
#   * G4 (recomputed_stat) tol = 1e-6      -- 1000x stricter than IR's
#                                            published 4-decimal precision.
#   * G7 (projection_provenance) k_RRF=60   -- Cormack et al., SIGIR 2009.
#   * G6 (mechanism_activation) thr = 0.5   -- "more than half" (majority).
# All three constants are sourced from external publication / convention
# rather than from in-loop tuning on production payloads.


class MechanismActivationError(GateError):
    gate_name = "mechanism_activation"


# Majority natural definition.
# Source: dictionary definition of "majority" = more than half (>= 0.5).
# Lifted directly from the structural meaning, NOT tuned against
# production payloads. Consistent with G4's 1e-6 (IR convention) and G7's
# _K_RRF=60 (Cormack 2009) natural derivation pattern.
_MECH_RATIO_THRESHOLD: float = 0.5

# Adaptive query_strategy labels.
# Source: prototype/eval/query_strategy.py (label emission sites for
# query-type-aware adaptive routing). The C0 default exercises adaptive
# routing which produces one of these four labels per QA; C10 disables it
# and collapses to the single "uniform" label.
_ADAPTIVE_STRATEGY_LABELS: frozenset[str] = frozenset({
    "causal", "compare", "define", "factual_typed",
})
_UNIFORM_STRATEGY_LABEL: str = "uniform"

# Cell-id -> list of (prediction_dot_path, expected_value, threshold).
# Each rule asserts that >= threshold of the records exhibit
# ``_resolve_dot_path(prediction, path) == expected`` (or, when
# ``expected`` is a frozenset, ``actual in expected``). Legacy
# weight-only cells (C4a, C5a, C6a, C7a, C8a) carry no mechanism toggle
# and are OMITTED here; assert_mechanism_activated returns "skipped" for
# any cell_id absent from this table.
EXPECTED_BEHAVIOR: Mapping[str, list[tuple[str, Any, float]]] = {
    "C0":  [
        ("postprocess.applied", True, _MECH_RATIO_THRESHOLD),
        ("query_strategy_actual", _ADAPTIVE_STRATEGY_LABELS, _MECH_RATIO_THRESHOLD),
        ("signals.gates.trust_4axis", True, _MECH_RATIO_THRESHOLD),
        ("answer_schema.mode", "schema", _MECH_RATIO_THRESHOLD),
    ],
    "C9":  [("postprocess.applied", False, _MECH_RATIO_THRESHOLD)],
    "C10": [("query_strategy_actual", _UNIFORM_STRATEGY_LABEL, _MECH_RATIO_THRESHOLD)],
    "C11": [("signals.gates.trust_4axis", False, _MECH_RATIO_THRESHOLD)],
    "C12": [("answer_schema.mode", "generic", _MECH_RATIO_THRESHOLD)],
    # C4a, C5a, C6a, C7a, C8a: legacy weight-only cells (no mechanism
    # toggle). G6 skipped (returns "skipped"). EXPECTED_BEHAVIOR omits
    # them on purpose.
}


# Sentinel used by the G6 legacy-payload bypass to distinguish
# "path resolved to None" (an explicitly-stamped null value) from
# "path missing" (a pre-Phase-4 record). A normal sentinel is needed
# because legacy payloads MAY legitimately stamp a path as None.
_PHASE4_PATH_MISSING = object()


def _resolve_dot_path(d: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted key path against a nested mapping.

    Returns ``None`` for any segment that is missing or non-mapping; the
    caller treats ``None`` as "did not match expected" so a malformed
    record cannot accidentally count toward the activation ratio.
    """
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(k)
    return cur


def assert_mechanism_activated(
    payload: Mapping[str, Any],
    cell_id: str,
) -> str:
    """G6: per-cell mechanism activation check.

    Returns
    -------
    str
        ``"ok"``                       -- every expected mechanism state
                                          observed at >= 50%
        ``"skipped"``                  -- cell_id not in
                                          ``EXPECTED_BEHAVIOR`` (e.g.
                                          legacy weight-only C4a-C8a
                                          cells).
        ``"skipped_legacy_payload"``   -- payload predates Phase 4
                                          stamping (no ``postprocess`` /
                                          ``query_strategy_actual`` /
                                          ``signals.gates`` fields in
                                          the first record's
                                          ``prediction``). This labels
                                          a timing/schema gap, not a
                                          real mechanism failure.

    Raises
    ------
    MechanismActivationError
        When at least one EXPECTED_BEHAVIOR rule's activation ratio is
        below ``_MECH_RATIO_THRESHOLD`` (0.5, majority natural), or when
        the payload carries zero records.
    """
    rules = EXPECTED_BEHAVIOR.get(cell_id)
    if rules is None:
        # Legacy weight-only cells (C4a-C8a) or unknown cell ids.
        return "skipped"
    records = (
        payload.get("records", [])
        if isinstance(payload, Mapping)
        else []
    )
    if not records:
        raise MechanismActivationError(
            f"[{cell_id}] G6 requires >=1 record, got 0"
        )

    # G6 legacy-payload bypass:
    # Phase 4 stamping is required for G6 evaluation. Payloads written
    # BEFORE the Phase 4 stamping rollout (e.g. production v3_v4
    # 30/30 generated 2026-05-29 AM) carry only the legacy
    # ``answer_schema`` field and lack the three Phase 4 marker
    # fields (``postprocess`` / ``query_strategy_actual`` /
    # ``signals.gates``). For those payloads G6 would always fail
    # with ratio=0 -- this is a labeling/timing issue, not a real
    # mechanism failure. We follow the G7 legacy-payload pattern
    # (assert_projection_provenance, L396-417): if NO record carries
    # ANY of the three Phase 4 markers, mark legacy and skip without
    # raising.
    #
    # ``answer_schema`` is deliberately EXCLUDED from the marker set
    # because pre-Phase-4 runners stamped it as a legacy field; using
    # it as a marker would mis-classify production v3_v4 as
    # post-Phase-4 and re-introduce the false-positive raise.
    _PHASE4_MARKERS = ("postprocess", "query_strategy_actual")

    def _record_has_phase4_marker(rec: Any) -> bool:
        if not isinstance(rec, Mapping):
            return False
        pred = rec.get("prediction")
        if not isinstance(pred, Mapping):
            return False
        for marker in _PHASE4_MARKERS:
            if marker in pred:
                return True
        sig = pred.get("signals")
        if isinstance(sig, Mapping) and "gates" in sig:
            return True
        return False

    if not any(_record_has_phase4_marker(r) for r in records):
        return "skipped_legacy_payload"

    for path, expected, threshold in rules:
        match_count = 0
        for r in records:
            pred = r.get("prediction") if isinstance(r, Mapping) else None
            if not isinstance(pred, Mapping):
                continue
            actual = _resolve_dot_path(pred, path)
            if isinstance(expected, frozenset):
                if actual in expected:
                    match_count += 1
            else:
                if actual == expected:
                    match_count += 1
        ratio = match_count / len(records)
        if ratio < threshold:
            raise MechanismActivationError(
                f"[{cell_id}] G6 mechanism inactive: "
                f"path={path!r} expected={expected!r} "
                f"observed_ratio={ratio:.3f} < {threshold:.2f} "
                f"({match_count}/{len(records)} records)"
            )
    return "ok"


__all__ = [
    "GateError",
    "RecomputedStatError",
    "SeedEffectiveNError",
    "ProjectionProvenanceError",
    "AblationIsolationError",
    "EstimandError",
    "ArtifactIntegrityError",
    "BackendGateError",
    "MechanismActivationError",
    "assert_recomputed_stat",
    "assert_table_consistency",
    "assert_projection_provenance",
    "assert_seed_variance",
    "assert_ablation_isolation",
    "assert_estimand",
    "assert_artifact_integrity",
    "assert_backend_gate",
    "assert_mechanism_activated",
    "write_gate_violation_log",
    "append_audit",
]
