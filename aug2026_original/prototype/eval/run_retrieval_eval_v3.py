"""V3 retrieval evaluation runner for source/entity/edge diagnostics."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from prototype.baselines.proposed_v3_retriever import (
    C4PostprocessMode,
    C5QueryStrategyMode,
    C6TrustMode,
    FusionMode,
    ProposedV3Retriever,
    SubAnswerReqMode,
    V3Weights,
)
from prototype.eval.answer_schema import (
    entity_type_of as _answer_entity_type_of,
    format_answer as _answer_format,
    schema_for as _answer_schema_for,
)
from prototype.eval.metrics_v3 import (
    edge_metrics,
    entity_metrics,
    path_edge_sequence_recall,
    path_endpoint_recall,
    path_exact_match,
    source_metrics,
)
from prototype.eval.split_policy import ManifestIndex


C4_POSTPROCESS_CHOICES: tuple[str, ...] = ("on", "off")
C5_QUERY_STRATEGY_CHOICES: tuple[str, ...] = ("adaptive", "uniform_BFS")
C6_TRUST_CHOICES: tuple[str, ...] = ("4axis", "single_source_weight")
SUB_ANSWER_REQ_CHOICES: tuple[str, ...] = ("schema", "generic")


@dataclass(frozen=True)
class OptionBToggles:
    """W7 D33-D36 4-mechanism toggle bundle. Defaults preserve current
    behaviour: all four mechanisms ON (proposed_v3 full system unchanged).
    """

    c4_postprocess: C4PostprocessMode = "on"
    c5_query_strategy: C5QueryStrategyMode = "adaptive"
    c6_trust: C6TrustMode = "4axis"
    sub_answer_req: SubAnswerReqMode = "schema"


@dataclass(frozen=True)
class CliToggleOverrides:
    """Sparse CLI override bundle for the W7 D33-D36 toggles.

    Each field is ``None`` when the user did NOT pass the corresponding CLI
    flag. The previous ``OptionBToggles``-as-CLI-bundle approach silently
    substituted argparse defaults (``"on"``, ``"adaptive"``, ``"4axis"``,
    ``"schema"``) for missing flags, which then masked the YAML-supplied
    OFF values for C9-C12 cells. Carrying explicit None preserves the
    "CLI omitted = no override" semantics ``_resolve_toggles`` relies on.
    """

    c4_postprocess: C4PostprocessMode | None = None
    c5_query_strategy: C5QueryStrategyMode | None = None
    c6_trust: C6TrustMode | None = None
    sub_answer_req: SubAnswerReqMode | None = None


SUPPORTED_SYSTEMS = {"bm25", "dense", "rrf", "dense_rrf", "proposed_v3"}
UNSUPPORTED_SYSTEMS: set[str] = set()
ALL_SYSTEMS = SUPPORTED_SYSTEMS | UNSUPPORTED_SYSTEMS
DEFAULT_TOP_K = 5

# Reciprocal Rank Fusion constant. Published in:
#   Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms
#   Condorcet and individual Rank Learning Methods." SIGIR 2009.
# k=60 is the standard structural constant from that paper -- it is
# NOT a tuned hyperparameter. Used here to project paper-level ranks
# onto entity-level and edge-level ranks (HippoRAG-style projection),
# applied identically to every retriever (BM25 / Dense / RRF /
# proposed_v3) so entity/edge-level metrics are evaluator-symmetric
# and not biased toward systems whose papers happen to rank early.
_K_RRF: int = 60
DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
DENSE_BACKENDS = {"deterministic", "bge-m3", "auto"}
DENSE_BACKEND_ENV = "RAG_V3_DENSE_BACKEND"
DENSE_MODEL_ENV = "RAG_V3_DENSE_MODEL"
DENSE_BATCH_SIZE_ENV = "RAG_V3_DENSE_BATCH_SIZE"
DENSE_LOCAL_FILES_ONLY_ENV = "RAG_V3_DENSE_LOCAL_FILES_ONLY"


class DenseBackendUnavailable(RuntimeError):
    """Raised when the optional BGE-M3 backend cannot be loaded."""

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class LoadedKg:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    system: str | None
    status: str
    weights: Mapping[str, Any]
    disable: tuple[str, ...]
    raw: Mapping[str, Any]


@dataclass
class RrfCandidate:
    paper: dict[str, Any]
    rrf_total: float = 0.0
    bm25_rrf: float = 0.0
    dense_rrf: float = 0.0
    bm25_raw: float = 0.0
    dense_raw: float = 0.0
    best_rank: int | None = None


@dataclass(frozen=True)
class DenseRetrieverConfig:
    backend: str = "deterministic"
    model_name: str = DEFAULT_DENSE_MODEL
    batch_size: int = 16
    local_files_only: bool = True

    def __post_init__(self) -> None:
        backend = _normalize_dense_backend(self.backend)
        object.__setattr__(self, "backend", backend)
        if self.batch_size <= 0:
            raise ValueError(f"dense batch_size must be positive; got {self.batch_size}")
        if not self.model_name.strip():
            raise ValueError("dense model_name must not be empty")

    @classmethod
    def from_env(
        cls,
        *,
        backend: str | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
        local_files_only: bool | None = None,
    ) -> "DenseRetrieverConfig":
        env_batch_size = os.environ.get(DENSE_BATCH_SIZE_ENV)
        resolved_batch_size = (
            batch_size
            if batch_size is not None
            else _parse_positive_int_env(DENSE_BATCH_SIZE_ENV, env_batch_size, default=16)
        )
        return cls(
            backend=backend or os.environ.get(DENSE_BACKEND_ENV, "deterministic"),
            model_name=model_name or os.environ.get(DENSE_MODEL_ENV, DEFAULT_DENSE_MODEL),
            batch_size=resolved_batch_size,
            local_files_only=(
                local_files_only
                if local_files_only is not None
                else _parse_bool_env(
                    DENSE_LOCAL_FILES_ONLY_ENV,
                    os.environ.get(DENSE_LOCAL_FILES_ONLY_ENV),
                    default=True,
                )
            ),
        )


class DenseTextEncoder(Protocol):
    backend_name: str
    model_name: str

    def encode(self, texts: list[str], *, batch_size: int) -> Sequence[Sequence[float]]:
        """Return one dense vector per input text."""
        ...


class LexicalBm25Retriever:
    """Small deterministic BM25-style retriever used by the eval runner.

    The project baseline can still be run separately. This runner-local
    implementation keeps Task 7 tests and ablations offline and dependency-free.
    """

    def __init__(self, papers: list[dict[str, Any]]) -> None:
        self.papers = list(papers)
        self._documents = [_tokenize(_paper_text(paper)) for paper in self.papers]
        self._avg_doc_len = (
            sum(len(document) for document in self._documents) / len(self._documents)
            if self._documents
            else 0.0
        )
        self._idf = self._build_idf(self._documents)

    def query(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        if not query_tokens or top_k <= 0:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for paper, document in zip(self.papers, self._documents):
            score = self._score(query_tokens, document)
            if score > 0.0:
                scored.append((score, paper))

        scored.sort(key=lambda item: (-item[0], _record_id(item[1])))
        max_score = max((score for score, _paper in scored), default=0.0)
        results: list[dict[str, Any]] = []
        for rank, (score, paper) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "paper": paper,
                    "rank": rank,
                    "score": _safe_score(score / max_score if max_score else 0.0),
                }
            )
        return results

    def _score(self, query_tokens: list[str], document: list[str]) -> float:
        if not document:
            return 0.0

        frequencies: dict[str, int] = {}
        for token in document:
            frequencies[token] = frequencies.get(token, 0) + 1

        k1 = 1.5
        b = 0.75
        score = 0.0
        doc_len = len(document)
        for token in set(query_tokens):
            term_frequency = frequencies.get(token, 0)
            if term_frequency == 0:
                continue
            idf = self._idf.get(token, 0.0)
            denominator = term_frequency + k1 * (
                1.0 - b + b * doc_len / max(self._avg_doc_len, 1.0)
            )
            score += idf * (term_frequency * (k1 + 1.0)) / denominator
        return score

    @staticmethod
    def _build_idf(documents: list[list[str]]) -> dict[str, float]:
        doc_count = len(documents)
        if doc_count == 0:
            return {}

        document_frequency: dict[str, int] = {}
        for document in documents:
            for token in set(document):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        return {
            token: math.log(1.0 + (doc_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }


class DeterministicDenseRetriever:
    """Dependency-free dense surrogate for offline eval wiring.

    This intentionally avoids model downloads. It uses a normalized token and
    character n-gram vector so dense/RRF ablation cells can exercise the same
    diagnostics path in tests and offline test runs.
    """

    def __init__(
        self,
        papers: list[dict[str, Any]],
        *,
        config: DenseRetrieverConfig | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        self.papers = list(papers)
        dense_config = config or DenseRetrieverConfig()
        self.metadata: dict[str, Any] = {
            "requested_backend": dense_config.backend,
            "actual_backend": "deterministic_surrogate",
            "model_name": dense_config.model_name,
            "local_files_only": dense_config.local_files_only,
        }
        if fallback_reason:
            self.metadata["fallback_reason"] = fallback_reason
        self._documents = [_dense_terms(_paper_text(paper)) for paper in self.papers]
        self._idf = self._build_idf(self._documents)
        self._vectors = [self._build_vector(document) for document in self._documents]

    def query(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []

        query_vector = self._build_vector(_dense_terms(query))
        if not query_vector:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for paper, vector in zip(self.papers, self._vectors):
            score = _cosine_similarity(query_vector, vector)
            if score > 0.0:
                scored.append((score, paper))

        scored.sort(key=lambda item: (-item[0], _record_id(item[1])))
        return [
            {
                "paper": paper,
                "rank": rank,
                "score": _safe_score(score),
                "signals": {"dense": _safe_score(score)},
            }
            for rank, (score, paper) in enumerate(scored[:top_k], start=1)
        ]

    def _build_vector(self, terms: list[str]) -> dict[str, float]:
        frequencies: dict[str, int] = {}
        for term in terms:
            frequencies[term] = frequencies.get(term, 0) + 1

        weighted = {
            term: frequency * self._idf.get(term, 0.0)
            for term, frequency in frequencies.items()
            if self._idf.get(term, 0.0) > 0.0
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm <= 0.0:
            return {}
        return {term: value / norm for term, value in weighted.items()}

    @staticmethod
    def _build_idf(documents: list[list[str]]) -> dict[str, float]:
        doc_count = len(documents)
        if doc_count == 0:
            return {}

        document_frequency: dict[str, int] = {}
        for document in documents:
            for term in set(document):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        return {
            term: math.log(1.0 + (doc_count + 1.0) / (frequency + 1.0))
            for term, frequency in document_frequency.items()
        }


class BgeM3DenseRetriever:
    """Optional real BGE-M3 dense retriever used only when explicitly selected."""

    def __init__(
        self,
        papers: list[dict[str, Any]],
        *,
        encoder: DenseTextEncoder,
        config: DenseRetrieverConfig,
    ) -> None:
        self.papers = list(papers)
        self.encoder = encoder
        self.config = config
        self.metadata: dict[str, Any] = {
            "requested_backend": config.backend,
            "actual_backend": encoder.backend_name,
            "model_name": encoder.model_name,
            "local_files_only": config.local_files_only,
        }
        texts = [_paper_text(paper) for paper in self.papers]
        encoded = encoder.encode(texts, batch_size=config.batch_size)
        self._vectors = [_normalize_dense_vector(vector) for vector in encoded]
        if len(self._vectors) != len(self.papers):
            raise DenseBackendUnavailable(
                "BGE-M3 encoder returned "
                f"{len(self._vectors)} vectors for {len(self.papers)} papers"
            )

    def query(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []

        encoded = self.encoder.encode([query], batch_size=1)
        query_vector = _normalize_dense_vector(encoded[0]) if encoded else []
        if not query_vector:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for paper, vector in zip(self.papers, self._vectors):
            score = _dense_dot(query_vector, vector)
            if score > 0.0:
                scored.append((score, paper))

        scored.sort(key=lambda item: (-item[0], _record_id(item[1])))
        return [
            {
                "paper": paper,
                "rank": rank,
                "score": _safe_score(score),
                "signals": {"dense": _safe_score(score)},
            }
            for rank, (score, paper) in enumerate(scored[:top_k], start=1)
        ]


class ReciprocalRankFusionRetriever:
    """Fuse BM25 and deterministic dense rankings with standard RRF."""

    def __init__(
        self,
        *,
        bm25: LexicalBm25Retriever,
        dense: Any,
        rrf_k: float = 60.0,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def query(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []

        pool_k = max(top_k * 4, top_k)
        candidates: dict[str, RrfCandidate] = {}
        for channel, hits in (
            ("bm25", self.bm25.query(query, top_k=pool_k)),
            ("dense", self.dense.query(query, top_k=pool_k)),
        ):
            self._add_hits(candidates, channel=channel, hits=hits)

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -item.rrf_total,
                item.best_rank if item.best_rank is not None else pool_k + 1,
                _record_id(item.paper),
            ),
        )
        max_score = max((candidate.rrf_total for candidate in ordered), default=0.0)
        results: list[dict[str, Any]] = []
        for rank, candidate in enumerate(ordered[:top_k], start=1):
            score = candidate.rrf_total / max_score if max_score else 0.0
            results.append(
                {
                    "paper": candidate.paper,
                    "rank": rank,
                    "score": _safe_score(score),
                    "signals": {
                        "rrf_bm25": _safe_score(candidate.bm25_rrf),
                        "rrf_dense": _safe_score(candidate.dense_rrf),
                        "rrf_total": _safe_score(candidate.rrf_total),
                        "bm25": _safe_score(candidate.bm25_raw),
                        "dense": _safe_score(candidate.dense_raw),
                    },
                }
            )
        return results

    def _add_hits(
        self,
        candidates: dict[str, RrfCandidate],
        *,
        channel: str,
        hits: list[dict[str, Any]],
    ) -> None:
        for fallback_rank, hit in enumerate(hits, start=1):
            paper = hit.get("paper")
            if not isinstance(paper, dict):
                continue
            paper_id = _record_id(paper)
            if not paper_id:
                continue

            candidate = candidates.setdefault(paper_id, RrfCandidate(paper=paper))
            rank = _as_int(hit.get("rank")) or fallback_rank
            rank_score = 1.0 / (self.rrf_k + rank)
            raw_score = _as_float(hit.get("score"))
            candidate.rrf_total += rank_score
            candidate.best_rank = (
                rank
                if candidate.best_rank is None
                else min(candidate.best_rank, rank)
            )
            if channel == "bm25":
                candidate.bm25_rrf += rank_score
                candidate.bm25_raw = max(candidate.bm25_raw, raw_score)
            elif channel == "dense":
                candidate.dense_rrf += rank_score
                candidate.dense_raw = max(candidate.dense_raw, raw_score)


def run_eval(
    *,
    manifest_path: Path,
    qa_path: Path,
    kg_dir: Path | None,
    system: str,
    candidate_partition: str,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
    ablation_config_path: Path | None = None,
    ablation_id: str | None = None,
    dense_config: DenseRetrieverConfig | None = None,
    toggles: "CliToggleOverrides | OptionBToggles | None" = None,
    paper_backend_claim: str | None = None,
) -> dict[str, Any]:
    """Run a V3 retrieval evaluation and write ``summary`` plus ``records``.

    ``paper_backend_claim``: D47 G2 paper-level backend claim. CLI > env
    > None precedence (env var: ``RAG_V3_PAPER_BACKEND_CLAIM``). When
    set, the runner enforces bidirectional consistency between the
    claim and the actual loaded backend after ``_build_retriever``.
    """
    _validate_top_k(top_k)

    manifest_records = _load_manifest_records(manifest_path)
    manifest_index = ManifestIndex.from_manifest(manifest_records)
    qa_records = _load_jsonl(qa_path)

    _assert_qa_source_targets_in_partition(
        qa_records,
        manifest_index,
        candidate_partition=candidate_partition,
    )

    candidate_papers = [
        _paper_from_manifest_record(record)
        for record in manifest_records
        if _record_partition(record) == candidate_partition
    ]
    kg = _load_kg(kg_dir)
    ablation = _load_ablation_spec(ablation_config_path, ablation_id)

    # D47 G5 ablation_isolation: fail-fast on cell-level perturbation
    # contract violations BEFORE we burn a retrieval run. Raises so the
    # CLI / orchestrator cannot silently emit a contaminated payload.
    if ablation is not None:
        from paper.eval.preflight_gates import (
            AblationIsolationError as _G5Error,
            assert_ablation_isolation as _g5,
        )
        try:
            _g5(ablation.ablation_id, ablation.raw.get("features") or {})
        except _G5Error as _g5_err:
            raise SystemExit(f"[GATE G5 FAILED] {_g5_err}") from _g5_err

    normalized_system = _resolve_system(system, ablation)
    effective_weights = _effective_v3_weights(ablation)
    if _is_unsupported_ablation(ablation):
        payload = _unsupported_payload(
            system=normalized_system,
            candidate_partition=candidate_partition,
            candidate_count=len(candidate_papers),
            qa_count=len(qa_records),
            top_k=top_k,
            ablation=ablation,
            effective_weights=effective_weights,
        )
        _write_payload(output_path, payload)
        return payload
    if normalized_system not in SUPPORTED_SYSTEMS:
        raise ValueError(f"unsupported system: {system!r}")

    effective_toggles = _resolve_toggles(toggles, ablation)
    retriever = _build_retriever(
        system=normalized_system,
        candidate_papers=candidate_papers,
        kg=kg,
        weights=effective_weights,
        ablation=ablation,
        dense_config=dense_config,
        toggles=effective_toggles,
    )

    # D47 Phase 3 G1 (artifact_integrity) + G2 (backend_gate) runtime
    # checks. Both fail-closed BEFORE the (expensive) per-QA loop so a
    # misconfigured backend never produces a payload at all. G1 catches
    # class-vs-label drift and silent fallbacks; G2 enforces the paper
    # claim <-> actual_backend bidirectional contract when the user (or
    # env) declares which backend the paper is grounded in.
    from paper.eval.preflight_gates import (
        ArtifactIntegrityError as _G1Error,
        BackendGateError as _G2Error,
        assert_artifact_integrity as _g1,
        assert_backend_gate as _g2,
    )
    try:
        _g1(retriever=retriever)
    except _G1Error as _g1_err:
        # Fail-closed: do not write a falsely-labeled payload.
        raise SystemExit(f"[GATE G1 FAILED] {_g1_err}") from _g1_err
    try:
        _g2_status = _g2(retriever=retriever, cli_claim=paper_backend_claim)
        print(f"[GATE G2] {_g2_status}")
    except _G2Error as _g2_err:
        raise SystemExit(f"[GATE G2 FAILED] {_g2_err}") from _g2_err

    records = [
        _evaluate_record(
            qa=qa,
            retriever=retriever,
            manifest_index=manifest_index,
            candidate_partition=candidate_partition,
            kg=kg,
            top_k=top_k,
            toggles=effective_toggles,
        )
        for qa in qa_records
    ]
    payload = {
        "summary": _summarize(
            records,
            system=normalized_system,
            candidate_partition=candidate_partition,
            candidate_count=len(candidate_papers),
            qa_count=len(qa_records),
            top_k=top_k,
            ablation=ablation,
            effective_weights=effective_weights,
            retriever_metadata=_retriever_metadata(retriever),
            toggles=effective_toggles,
        ),
        "records": records,
    }

    # D47 G4 + G7 runtime gates. On violation: write the payload anyway
    # (with audit field), drop a sibling .GATE_FAIL.log, then re-raise so
    # the caller (CLI / orchestrator) knows the cell did not pass.
    from paper.eval.preflight_gates import (
        GateError as _GateError,
        append_audit as _append_audit,
        assert_projection_provenance as _assert_projection_provenance,
        assert_recomputed_stat as _assert_recomputed_stat,
        write_gate_violation_log as _write_gate_violation_log,
    )
    try:
        _assert_recomputed_stat(payload)
        _assert_projection_provenance(payload=payload)
    except _GateError as _gate_err:
        _append_audit(
            payload, _gate_err.gate_name, "violation", str(_gate_err)
        )
        # Phase 1 fix #5. Mark the payload
        # so downstream consumers (orchestrators, aggregators, paper
        # analysis scripts) can fast-reject the partial artifact instead
        # of silently consuming a gate-violating result.
        #
        # Phase 1 Round 3 fix #6: two markers, two distinct contracts.
        #   * ``gate_failed``: a technical gate raised. ANY gate
        #     violation sets this. Tools that inspect payloads for
        #     debugging purposes (e.g. cross_judge backends, leakage
        #     audits) can still consume the payload but should record
        #     the gate violation in their own audit trail.
        #   * ``invalid_for_analysis``: downstream-contract flag. The
        #     ablation_orchestrator / paper-table / stat_tests pipeline
        #     MUST drop this cell. Today this is set identically to
        #     gate_failed, but the split is preserved so a future "G3
        #     bypass for diagnostic re-run" scenario can set
        #     gate_failed=True while leaving invalid_for_analysis=False.
        # See paper/eval/ablation_orchestrator.py:_is_gate_failed for
        # the consumer-side check.
        _summary_block = payload.setdefault("summary", {})
        if isinstance(_summary_block, dict):
            _summary_block["gate_failed"] = True
            _summary_block["invalid_for_analysis"] = True
        log_path = _write_gate_violation_log(output_path, _gate_err)
        print(f"[GATE VIOLATION] {_gate_err}\nlog: {log_path}")
        _write_payload(output_path, payload)
        raise

    _write_payload(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V3 retrieval evaluation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--kg-dir", type=Path, default=None)
    parser.add_argument("--system", required=True, choices=sorted(ALL_SYSTEMS))
    parser.add_argument("--candidate-partition", default="eval")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ablation-config", type=Path, default=None)
    parser.add_argument("--ablation-id", default=None)
    parser.add_argument(
        "--dense-backend",
        choices=sorted(DENSE_BACKENDS),
        default=None,
        help=(
            "Dense backend for dense/rrf systems. Defaults to "
            f"${DENSE_BACKEND_ENV} or deterministic."
        ),
    )
    parser.add_argument(
        "--dense-model",
        default=None,
        help=f"Dense model name. Defaults to ${DENSE_MODEL_ENV} or {DEFAULT_DENSE_MODEL}.",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=None,
        help=f"Dense embedding batch size. Defaults to ${DENSE_BATCH_SIZE_ENV} or 16.",
    )
    parser.add_argument(
        "--dense-allow-download",
        action="store_true",
        help=(
            "Allow the optional BGE-M3 backend to download model files. "
            "By default it only uses locally cached files."
        ),
    )
    # W7 D33-D36 4-mechanism toggles. Argparse defaults are None (sentinel)
    # so a missing CLI flag does NOT override a YAML feature value. The final
    # default (when neither CLI nor YAML supplies a value) is restored inside
    # _resolve_toggles via the OptionBToggles dataclass defaults.
    parser.add_argument(
        "--c4-postprocess",
        choices=C4_POSTPROCESS_CHOICES,
        default=None,
        help="C4 D33 H8: 7-step saturation postprocess. Default on.",
    )
    parser.add_argument(
        "--c5-query-strategy",
        choices=C5_QUERY_STRATEGY_CHOICES,
        default=None,
        help="C5 D34 H9: query-type-adaptive multi-hop. Default adaptive.",
    )
    parser.add_argument(
        "--c6-trust",
        choices=C6_TRUST_CHOICES,
        default=None,
        help="C6 D35 H10: 4-axis evidence-grounded trust. Default 4axis.",
    )
    parser.add_argument(
        "--sub-answer-req",
        choices=SUB_ANSWER_REQ_CHOICES,
        default=None,
        help="Sub-method D36 H11: entity-type-aware answer schema. Default schema.",
    )
    # D47 G3: explicit seed flag for paired-bootstrap with effective n>1.
    # When set, the value is exported via RAG_V3_SEED and consumed by the
    # tie-band shuffler at retriever build time. ``None`` preserves the
    # pre-D47 deterministic behaviour.
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for tie-band shuffle (post-RRF ranking). "
            "Required for paired-bootstrap with effective n>1."
        ),
    )
    # D47 G2 (backend_gate): paper-level backend claim with CLI > env >
    # None precedence. Whitelist matches paper.eval.preflight_gates
    # _VALID_PAPER_CLAIMS. When env RAG_V3_PAPER_BACKEND_CLAIM is set
    # this CLI flag overrides it.
    parser.add_argument(
        "--paper-backend-claim",
        choices=["bge-m3", "surrogate-diagnostic"],
        default=None,
        help=(
            "D47 G2: paper-level backend claim. CLI > env > None "
            "priority. If env RAG_V3_PAPER_BACKEND_CLAIM is set, this "
            "CLI flag wins."
        ),
    )
    args = parser.parse_args(argv)

    # D47 G3: propagate --seed to the runner-internal random seed env var
    # (paper/results/ablation_v3_v4/run_v4_ablation.py:71-73 already sets
    # RAG_V3_SEED externally; this wires up the same env var from the CLI
    # so both code paths converge on a single seeding contract).
    if args.seed is not None:
        os.environ["RAG_V3_SEED"] = str(args.seed)
        # Seed Python's random module so any tie-band shuffle that imports
        # ``random`` from prototype.baselines.* sees a deterministic state.
        import random as _random
        _random.seed(int(args.seed))

    # D47 G7 static check: verify the loaded runner module carries the
    # expected RRF projection version marker BEFORE any retrieval runs.
    # Fails closed (exit code 2) so misconfigured environments cannot
    # silently degrade to a pre-D46 projection.
    from paper.eval.preflight_gates import (
        GateError as _GateError,
        assert_projection_provenance as _assert_projection_provenance,
    )
    import sys as _sys
    _runner_self = _sys.modules[__name__]
    try:
        _assert_projection_provenance(runner_module=_runner_self)
    except _GateError as _gate_err:
        print(f"[PRE-FLIGHT GATE FAILED] {_gate_err}")
        return 2

    try:
        dense_config = None
        if (
            args.dense_backend is not None
            or args.dense_model is not None
            or args.dense_batch_size is not None
            or args.dense_allow_download
        ):
            dense_config = DenseRetrieverConfig.from_env(
                backend=args.dense_backend,
                model_name=args.dense_model,
                batch_size=args.dense_batch_size,
                local_files_only=False if args.dense_allow_download else None,
            )
        # CliToggleOverrides carries only the user-supplied (non-None) flags so
        # that missing CLI options never clobber YAML values inside
        # _resolve_toggles. Pass None for any flag the user did not provide.
        toggles = CliToggleOverrides(
            c4_postprocess=args.c4_postprocess,
            c5_query_strategy=args.c5_query_strategy,
            c6_trust=args.c6_trust,
            sub_answer_req=args.sub_answer_req,
        )
        payload = run_eval(
            manifest_path=args.manifest,
            qa_path=args.qa,
            kg_dir=args.kg_dir,
            system=args.system,
            candidate_partition=args.candidate_partition,
            output_path=args.output,
            top_k=args.top_k,
            ablation_config_path=args.ablation_config,
            ablation_id=args.ablation_id,
            dense_config=dense_config,
            toggles=toggles,
            paper_backend_claim=args.paper_backend_claim,
        )
    except (OSError, TypeError, ValueError) as err:
        parser.error(str(err))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


def _build_retriever(
    *,
    system: str,
    candidate_papers: list[dict[str, Any]],
    kg: LoadedKg,
    weights: V3Weights,
    ablation: AblationSpec | None = None,
    dense_config: DenseRetrieverConfig | None = None,
    toggles: OptionBToggles | None = None,
) -> Any:
    bm25 = LexicalBm25Retriever(candidate_papers)
    if system == "bm25":
        return bm25
    if system == "dense":
        return _build_dense_retriever(
            candidate_papers,
            config=dense_config or DenseRetrieverConfig.from_env(),
        )
    if system in {"rrf", "dense_rrf"}:
        dense = _build_dense_retriever(
            candidate_papers,
            config=dense_config or DenseRetrieverConfig.from_env(),
        )
        return ReciprocalRankFusionRetriever(bm25=bm25, dense=dense)
    if system == "proposed_v3":
        dense = None
        if weights.dense > 0.0:
            dense = _build_dense_retriever(
                candidate_papers,
                config=dense_config or DenseRetrieverConfig.from_env(),
            )
        resolved_toggles = toggles or OptionBToggles()
        # Phase 2 (D47 G5) wiring: thread the minus_one_* kwargs derived
        # from the ablation YAML's features block into the retriever. The
        # retriever's __init__ enforces at-most-one-True and surgically
        # disables the targeted mechanism via init-time configuration
        # mutation; D46 retrieval body (query/scoring/RRF) untouched.
        minus_one_kwargs = _minus_one_kwargs(ablation)
        return ProposedV3Retriever(
            papers=candidate_papers,
            bm25=bm25,
            dense=dense,
            nodes=kg.nodes,
            edges=kg.edges,
            weights=weights,
            coverage_trigger_threshold=_coverage_trigger_threshold(ablation),
            c4_postprocess=resolved_toggles.c4_postprocess,
            c5_query_strategy=resolved_toggles.c5_query_strategy,
            c6_trust=resolved_toggles.c6_trust,
            sub_answer_req=resolved_toggles.sub_answer_req,
            mode=_resolve_mode(ablation),
            **minus_one_kwargs,
        )
    raise ValueError(f"unsupported system: {system!r}")


def _resolve_mode(ablation: "AblationSpec | None") -> "FusionMode":
    """Pick the V3 fusion mode for the resolved ablation cell.

    Precedence:
      1. Explicit ``mode:`` field in the ablation YAML cell.
      2. ``additive_legacy`` for the W7 single-weight isolation cells
         (C2/C3/C5a/C6a/C7a/C8a) so their pre-P3 weight-driven semantics
         survive untouched.
      3. The retriever default (``schema_centric_gated``).
    """
    if ablation is None:
        return "schema_centric_gated"
    raw_mode = ablation.raw.get("mode")
    if isinstance(raw_mode, str):
        candidate = raw_mode.strip().lower()
        if candidate in {"additive_legacy", "schema_centric_gated"}:
            return candidate  # type: ignore[return-value]
    # Phase 2 D47 G5: C5a/C6a/C7a/C8a are now minus_one cells that share
    # the C0 schema-centric fusion mode and surgically remove ONE
    # mechanism via _apply_minus_one_overrides. Only C2/C3 retain the
    # pre-P3 weight-driven semantics (W7 single-weight isolation cells).
    legacy_cells = {"C2", "C3"}
    if ablation.ablation_id in legacy_cells:
        return "additive_legacy"
    return "schema_centric_gated"


def _resolve_toggles(
    overrides: "CliToggleOverrides | OptionBToggles | None",
    ablation: AblationSpec | None,
) -> OptionBToggles:
    """Combine ablation-config feature toggles with CLI overrides.

    Precedence (highest -> lowest):
      1. CLI override that the user explicitly passed (non-None field of
         ``CliToggleOverrides``).
      2. YAML feature value under ``features.c4_postprocess`` etc.
      3. OptionBToggles default (all four mechanisms ON).

    The previous implementation treated argparse defaults (``"on"`` /
    ``"adaptive"`` / ``"4axis"`` / ``"schema"``) as user overrides and
    therefore stomped YAML's OFF values for C9-C12 cells. Carrying
    ``CliToggleOverrides`` with explicit None preserves the
    "CLI omitted = no override" semantics. The legacy ``OptionBToggles``
    shape is still accepted for back-compat with callers that pre-resolved
    their toggles.
    """
    base = OptionBToggles()
    if ablation is not None:
        features = ablation.raw.get("features")
        if isinstance(features, Mapping):
            c4 = features.get("c4_postprocess")
            c5 = features.get("c5_query_strategy")
            c6 = features.get("c6_trust")
            sub = features.get("sub_answer_req")
            base = OptionBToggles(
                c4_postprocess=(
                    "on" if c4 is True else ("off" if c4 is False else c4 if isinstance(c4, str) else base.c4_postprocess)
                ),
                c5_query_strategy=(
                    c5 if isinstance(c5, str) else base.c5_query_strategy
                ),
                c6_trust=(
                    c6 if isinstance(c6, str) else base.c6_trust
                ),
                sub_answer_req=(
                    sub if isinstance(sub, str) else base.sub_answer_req
                ),
            )
    if overrides is None:
        return base
    if isinstance(overrides, CliToggleOverrides):
        return OptionBToggles(
            c4_postprocess=(
                overrides.c4_postprocess
                if overrides.c4_postprocess is not None
                else base.c4_postprocess
            ),
            c5_query_strategy=(
                overrides.c5_query_strategy
                if overrides.c5_query_strategy is not None
                else base.c5_query_strategy
            ),
            c6_trust=(
                overrides.c6_trust
                if overrides.c6_trust is not None
                else base.c6_trust
            ),
            sub_answer_req=(
                overrides.sub_answer_req
                if overrides.sub_answer_req is not None
                else base.sub_answer_req
            ),
        )
    # Legacy OptionBToggles overrides: all four fields are non-None strings
    # by construction, so they act as full overrides (back-compat path).
    return OptionBToggles(
        c4_postprocess=overrides.c4_postprocess,
        c5_query_strategy=overrides.c5_query_strategy,
        c6_trust=overrides.c6_trust,
        sub_answer_req=overrides.sub_answer_req,
    )


def _build_dense_retriever(
    papers: list[dict[str, Any]],
    *,
    config: DenseRetrieverConfig,
) -> Any:
    if config.backend == "deterministic":
        return DeterministicDenseRetriever(papers, config=config)
    try:
        encoder = _load_bge_m3_encoder(config)
        return BgeM3DenseRetriever(papers, encoder=encoder, config=config)
    except Exception as err:  # noqa: BLE001
        return DeterministicDenseRetriever(
            papers,
            config=config,
            fallback_reason=str(err),
        )


@dataclass(frozen=True)
class _SentenceTransformerEncoder:
    model: Any
    model_name: str
    backend_name: str = "sentence-transformers"

    def encode(self, texts: list[str], *, batch_size: int) -> Sequence[Sequence[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return _coerce_embedding_matrix(vectors)


@dataclass(frozen=True)
class _FlagEmbeddingEncoder:
    model: Any
    model_name: str
    backend_name: str = "FlagEmbedding"

    def encode(self, texts: list[str], *, batch_size: int) -> Sequence[Sequence[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
        )
        if isinstance(vectors, Mapping):
            vectors = vectors.get("dense_vecs", [])
        return _coerce_embedding_matrix(vectors)


class _TransformersEncoder:
    backend_name = "bge-m3"

    def __init__(self, *, model: Any, tokenizer: Any, torch_module: Any, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.model_name = model_name

    def encode(self, texts: list[str], *, batch_size: int) -> Sequence[Sequence[float]]:
        vectors: list[list[float]] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                output = self.model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())
        return vectors


def _load_bge_m3_encoder(config: DenseRetrieverConfig) -> DenseTextEncoder:
    errors: list[str] = []
    try:
        return _load_transformers_encoder(config)
    except Exception as err:  # noqa: BLE001
        errors.append(f"transformers: {err}")

    try:
        return _load_sentence_transformer_encoder(config)
    except Exception as err:  # noqa: BLE001
        errors.append(f"sentence-transformers: {err}")

    try:
        return _load_flag_embedding_encoder(config)
    except Exception as err:  # noqa: BLE001
        errors.append(f"FlagEmbedding: {err}")

    raise DenseBackendUnavailable("; ".join(errors))


def _load_transformers_encoder(config: DenseRetrieverConfig) -> _TransformersEncoder:
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as err:  # noqa: BLE001
        raise DenseBackendUnavailable(f"transformers/torch unavailable: {err}") from err

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        model = AutoModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
    except Exception as err:  # noqa: BLE001
        mode = "local cache" if config.local_files_only else "download/load"
        raise DenseBackendUnavailable(
            f"{config.model_name} unavailable via {mode}: {err}"
        ) from err
    model.eval()
    return _TransformersEncoder(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        model_name=config.model_name,
    )


def _load_sentence_transformer_encoder(
    config: DenseRetrieverConfig,
) -> _SentenceTransformerEncoder:
    from sentence_transformers import SentenceTransformer  # type: ignore

    try:
        model = SentenceTransformer(
            config.model_name,
            local_files_only=config.local_files_only,
        )
    except TypeError as err:
        if config.local_files_only:
            raise DenseBackendUnavailable(
                "installed sentence-transformers does not support local_files_only"
            ) from err
        model = SentenceTransformer(config.model_name)
    return _SentenceTransformerEncoder(model=model, model_name=config.model_name)


def _load_flag_embedding_encoder(config: DenseRetrieverConfig) -> _FlagEmbeddingEncoder:
    from FlagEmbedding import BGEM3FlagModel  # type: ignore

    with _temporary_hf_offline(config.local_files_only):
        model = BGEM3FlagModel(config.model_name, use_fp16=False)
    return _FlagEmbeddingEncoder(model=model, model_name=config.model_name)


@contextmanager
def _temporary_hf_offline(enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _coverage_trigger_threshold(ablation: AblationSpec | None) -> float:
    if ablation is None:
        return 0.0
    features = ablation.raw.get("features")
    if isinstance(features, Mapping) and features.get("coverage_trigger") is False:
        return 0.0
    trigger = ablation.raw.get("trigger")
    if not isinstance(trigger, Mapping):
        return 0.0
    return _as_float(trigger.get("low_entity_coverage_threshold"))


# Phase 2 (D47 G5) minus_one feature key -> ProposedV3Retriever ctor kwarg.
# Each YAML feature flag (set True for exactly one C4a-C8a cell) maps to a
# dedicated ctor kwarg; the retriever then mutates init-time configuration
# in _apply_minus_one_overrides. D46 retrieval body NEVER touched.
_MINUS_ONE_KWARG_BY_FEATURE: dict[str, str] = {
    "c4a_minus_one_coverage_trigger": "minus_one_coverage_trigger",
    "c5a_minus_one_typed_schema":     "minus_one_typed_schema",
    "c6a_minus_one_edge_path":        "minus_one_edge_path",
    "c7a_minus_one_query_strategy":   "minus_one_query_strategy",
    "c8a_minus_one_trust":            "minus_one_trust",
}


def _minus_one_kwargs(ablation: AblationSpec | None) -> dict[str, bool]:
    """Return the minus_one_* kwargs to pass into ProposedV3Retriever.

    Reads ``features.c?a_minus_one_*`` flags from the ablation YAML; any
    True flag becomes ``minus_one_*=True`` on the retriever. The retriever
    enforces at-most-one-True at construction time (ValueError otherwise).
    Returns an empty dict when no flag is set or no ablation is loaded.
    """
    if ablation is None:
        return {}
    features = ablation.raw.get("features")
    if not isinstance(features, Mapping):
        return {}
    out: dict[str, bool] = {}
    for feature_key, kwarg in _MINUS_ONE_KWARG_BY_FEATURE.items():
        if features.get(feature_key) is True:
            out[kwarg] = True
    return out


def _evaluate_record(
    *,
    qa: dict[str, Any],
    retriever: Any,
    manifest_index: ManifestIndex,
    candidate_partition: str,
    kg: LoadedKg,
    top_k: int,
    toggles: OptionBToggles | None = None,
) -> dict[str, Any]:
    question = str(qa.get("question") or qa.get("query") or "")
    hits = retriever.query(question, top_k=top_k)
    retrieved_sources = _retrieved_sources(hits)
    retrieved_source_ids = [item["paper_id"] for item in retrieved_sources]

    retrieved_entities = _retrieved_entities(
        kg.nodes,
        retrieved_sources,
        hits=hits,
        top_k=top_k,
    )
    retrieved_entity_ids = [item["entity_id"] for item in retrieved_entities]
    retrieved_edges = _retrieved_edges(
        kg.edges,
        retrieved_sources,
        retrieved_entity_ids,
        hits=hits,
        top_k=top_k,
    )
    retrieved_edge_ids = [item["edge_id"] for item in retrieved_edges]

    gold_source_ids = _metric_gold_source_ids(qa)
    evidence_source_ids = _evidence_source_ids(qa)
    gold_entity_ids = _string_list(qa.get("canonical_entity_ids"))
    gold_edge_ids = _string_list(qa.get("canonical_edge_ids"))
    predicted_paths = _reconstruct_retrieved_paths(
        kg=kg,
        retrieved_entity_ids=retrieved_entity_ids,
        retrieved_edges=retrieved_edges,
        top_k=top_k,
    )
    gold_paths = _path_list(qa.get("gold_paths") or qa.get("canonical_paths"))

    source_result = source_metrics(retrieved_source_ids, gold_source_ids, top_k)
    entity_result = entity_metrics(retrieved_entity_ids, gold_entity_ids, top_k)
    edge_result = edge_metrics(retrieved_edge_ids, gold_edge_ids, top_k)

    answer_schema = _build_answer_schema(
        question=question,
        retrieved_entities=retrieved_entities,
        toggles=toggles or OptionBToggles(),
    )

    # Phase 4 G6 (mechanism_activation) signal propagation.
    # The retriever stamps three diagnostic fields on hits[0].diagnostics; this
    # block mirrors them into the per-record prediction dict so the
    # payload-side G6 gate (paper/eval/preflight_gates.py
    # assert_mechanism_activated) can score the >=50% majority-natural
    # activation ratio without reaching back into hit objects. The
    # majority threshold (0.5) is the structural definition of "majority"
    # -- not a tuned hyperparameter -- consistent with G4's 1e-6 (IR
    # convention) and G7's _K_RRF=60 (Cormack 2009) natural derivation
    # pattern. D46 retrieval body is NEVER touched: this is read-only
    # mirroring of diagnostics already produced upstream.
    hits_top1 = hits[0] if hits else {}
    diag = (
        hits_top1.get("diagnostics")
        if isinstance(hits_top1, Mapping)
        else None
    )
    if not isinstance(diag, Mapping):
        diag = {}
    postprocess_info = diag.get("postprocess")
    if not isinstance(postprocess_info, Mapping):
        postprocess_info = {"applied": False, "bonus_applied_count": 0}
    query_strategy_actual = diag.get("query_strategy_actual")
    gated_signals = diag.get("gated_signals")
    if not isinstance(gated_signals, Mapping):
        gated_signals = {}

    return {
        "qa_id": str(qa.get("qa_id") or qa.get("id") or ""),
        "domain": _record_domain(qa, manifest_index, gold_source_ids),
        "query_type": str(qa.get("query_type") or "unknown"),
        "mismatch_type": str(qa.get("mismatch_type") or "none"),
        "gold": {
            "answer_source": qa.get("answer_source") or gold_source_ids,
            "evidence_source_ids": evidence_source_ids,
            "canonical_entity_ids": gold_entity_ids,
            "canonical_edge_ids": gold_edge_ids,
            "gold_paths": gold_paths,
        },
        "prediction": {
            "retrieved_sources": retrieved_sources,
            "retrieved_entities": retrieved_entities,
            "retrieved_edges": retrieved_edges,
            "retrieved_paths": predicted_paths,
            "answer_schema": answer_schema,
            # Phase 4 G6 signal mirrors (the retriever stamps, the runner mirrors).
            "postprocess": dict(postprocess_info),
            "query_strategy_actual": query_strategy_actual,
            "signals": {"gates": dict(gated_signals)},
        },
        "metrics": {
            "source_hit_at_5": float(source_result["hit_at_k"]),
            "source_mrr": float(source_result["mrr"]),
            "source_recall_at_5": float(source_result["recall_at_k"]),
            "source_matched_rank": source_result["matched_rank"],
            "entity_precision_at_5": float(entity_result["precision_at_k"]),
            "entity_recall_at_5": float(entity_result["recall_at_k"]),
            "entity_f1_at_5": float(entity_result["f1_at_k"]),
            "entity_hit_at_5": float(entity_result["hit_at_k"]),
            "edge_precision_at_5": float(edge_result["precision_at_k"]),
            "edge_recall_at_5": float(edge_result["recall_at_k"]),
            "edge_f1_at_5": float(edge_result["f1_at_k"]),
            "edge_hit_at_5": float(edge_result["hit_at_k"]),
            "path_exact_match": path_exact_match(predicted_paths, gold_paths),
            "path_endpoint_recall": path_endpoint_recall(predicted_paths, gold_paths),
            "path_edge_sequence_recall": path_edge_sequence_recall(
                predicted_paths, gold_paths
            ),
        },
        "signals": _merge_signals(hits),
        "partition_audit": _partition_audit(
            retrieved_source_ids=retrieved_source_ids,
            manifest_index=manifest_index,
            candidate_partition=candidate_partition,
        ),
    }


def _build_answer_schema(
    *,
    question: str,
    retrieved_entities: list[dict[str, Any]],
    toggles: OptionBToggles,
) -> dict[str, Any]:
    """Apply the entity-type-aware answer requirement scaffold (D36 / H11).

    When ``sub_answer_req`` is ``"schema"`` (default), inspect the top
    retrieved entities' types and emit the matching ``AnswerSpec``. The
    OFF flag ``"generic"`` collapses to the single-field
    ``("answer",)`` schema. No content synthesis happens here -- this is
    a label scaffold the downstream answer-formatting block can populate.
    """
    if toggles.sub_answer_req == "generic":
        spec = _answer_schema_for("generic")
        return {
            "mode": "generic",
            "entity_type": spec.entity_type,
            "fields": list(spec.fields),
        }
    candidate_types = [
        str(entity.get("type") or "") for entity in retrieved_entities
    ]
    chosen_type = _answer_entity_type_of(question, candidate_types)
    spec = _answer_schema_for(chosen_type)
    return {
        "mode": "schema",
        "entity_type": spec.entity_type,
        "fields": list(spec.fields),
    }


def _retrieved_sources(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    max_score = max((_as_float(hit.get("score")) for hit in hits), default=0.0)
    for fallback_rank, hit in enumerate(hits, start=1):
        paper = hit.get("paper")
        if not isinstance(paper, Mapping):
            continue
        paper_id = _record_id(paper)
        if not paper_id:
            continue
        raw_score = _as_float(hit.get("score"))
        score = raw_score / max_score if max_score > 1.0 else raw_score
        sources.append(
            {
                "paper_id": paper_id,
                "rank": _as_int(hit.get("rank")) or fallback_rank,
                "score": _safe_score(score),
                "title": str(paper.get("title") or ""),
            }
        )
    return sources


def _projection_versioned(version: str):
    """Decorator that pins the published RRF projection contract version
    onto a helper. D47 G7 reads ``__projection_version__`` at runner
    startup and at payload-write time to detect silent regression to a
    pre-D46 implementation. See paper/DECISIONS.md D46 fix #1.
    """

    def _decorator(fn):
        fn.__projection_version__ = version
        return fn

    return _decorator


@_projection_versioned("rrf_v1_d46")
def _retrieved_entities(
    nodes: list[dict[str, Any]],
    retrieved_sources: list[dict[str, Any]],
    *,
    top_k: int,
    hits: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # HippoRAG-style RRF projection from paper-rank to entity-rank.
    # Every hit that admits a node (via source_docs overlap OR via the
    # retriever's diagnostics.matched_nodes / alias_nodes / neighbor_nodes)
    # contributes 1 / (_K_RRF + paper_rank) to that node's score. The
    # entity rank is then the dense rank over RRF scores, with a
    # deterministic node_id tie break. This projection is applied
    # identically to every supported system (BM25 / Dense / RRF /
    # proposed_v3) so entity-level metrics measure KG coverage rather
    # than rewarding evaluator-side optimism for retrievers that happen
    # to rank the gold paper at position 1.
    #
    # Backward compatibility:
    #   * Without diagnostics (BM25/Dense/RRF hits) only source_docs
    #     overlap admits a node, matching the previous behaviour for
    #     the candidate set.
    #   * Without ``hits`` (legacy call sites) the retrieved_sources
    #     list is treated as the hit stream so source-doc overlap still
    #     admits nodes.
    source_rank = {
        item["paper_id"]: int(item["rank"]) for item in retrieved_sources
    }
    nodes_by_id = {_node_id(node): node for node in nodes if _node_id(node)}
    docs_by_node = {
        _node_id(node): set(_source_docs(node))
        for node in nodes
        if _node_id(node)
    }

    scores: dict[str, float] = {}
    chosen_node: dict[str, dict[str, Any]] = {}

    def _accumulate(node_id: str, paper_rank: int) -> None:
        node = nodes_by_id.get(node_id)
        if node is None:
            return
        # paper_rank is 1-indexed; _K_RRF + paper_rank is always positive.
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (
            _K_RRF + max(paper_rank, 1)
        )
        chosen_node.setdefault(node_id, node)

    hit_stream: list[dict[str, Any]] = list(hits or [])
    if hit_stream:
        for hit in hit_stream:
            paper = hit.get("paper") if isinstance(hit.get("paper"), Mapping) else {}
            paper_id = _record_id(paper) if isinstance(paper, Mapping) else ""
            hit_paper_rank = source_rank.get(paper_id) or (
                _as_int(hit.get("rank")) or 1
            )
            # (1) source_docs overlap: every node whose source_docs include
            # this hit's paper gets the hit's RRF contribution.
            if paper_id:
                for node_id, docs in docs_by_node.items():
                    if paper_id in docs:
                        _accumulate(node_id, hit_paper_rank)
            # (2) diagnostics union (proposed_v3 matched / alias / neighbor):
            # canonical nodes whose source_docs do not include the retrieved
            # paper still get the hit's RRF contribution.
            diag = (
                hit.get("diagnostics")
                if isinstance(hit.get("diagnostics"), Mapping)
                else {}
            )
            if diag:
                for key in ("matched_nodes", "alias_nodes", "neighbor_nodes"):
                    for node_id in _string_list(diag.get(key)):
                        _accumulate(node_id, hit_paper_rank)
    else:
        # Legacy call: treat retrieved_sources as the hit stream so
        # source-doc overlap still admits nodes.
        for source in retrieved_sources:
            paper_id = str(source.get("paper_id") or "")
            paper_rank = int(source.get("rank") or 1)
            if not paper_id:
                continue
            for node_id, docs in docs_by_node.items():
                if paper_id in docs:
                    _accumulate(node_id, paper_rank)

    # Deterministic ordering: RRF score descending, node_id ascending on tie.
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[dict[str, Any]] = []
    for rank, (node_id, score) in enumerate(items, start=1):
        node = chosen_node[node_id]
        out.append(
            {
                "entity_id": node_id,
                "rank": rank,
                "score": _safe_score(score),
                "name": str(node.get("name") or ""),
                "type": str(node.get("type") or node.get("node_type") or ""),
            }
        )
        if len(out) >= top_k:
            break
    return out


@_projection_versioned("rrf_v1_d46")
def _retrieved_edges(
    edges: list[dict[str, Any]],
    retrieved_sources: list[dict[str, Any]],
    retrieved_entity_ids: list[str],
    *,
    top_k: int,
    hits: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # Mirror of _retrieved_entities: HippoRAG-style RRF projection from
    # paper-rank to edge-rank, applied identically to every supported
    # system. An edge accumulates 1 / (_K_RRF + paper_rank) from each
    # hit that admits it via:
    #   (a) source_docs overlap (the hit's paper appears in the edge's
    #       source documents);
    #   (b) endpoint diagnostics union -- the hit's
    #       diagnostics.matched_nodes / alias_nodes / neighbor_nodes
    #       contains BOTH endpoints of the edge (so the edge is in the
    #       expanded candidate set the retriever surfaced for the
    #       paper);
    #   (c) endpoint-set backstop -- both endpoints are already in the
    #       projected entity_set even without diagnostics support
    #       (legacy backward-compat with BM25-only hits).
    # The edge rank is then dense rank over RRF scores with an
    # edge_id ascending tie break for determinism.
    source_rank = {
        item["paper_id"]: int(item["rank"]) for item in retrieved_sources
    }
    entity_set = set(retrieved_entity_ids)

    # Collect, per hit, which papers + which diagnostics-admitted node
    # sets contribute the hit's RRF weight to edges.
    hit_stream: list[dict[str, Any]] = list(hits or [])
    # Each entry: (paper_rank_used_for_rrf, paper_id_or_empty, diag_node_set).
    hit_entries: list[tuple[int, str, set[str]]] = []
    if hit_stream:
        for hit in hit_stream:
            paper = (
                hit.get("paper")
                if isinstance(hit.get("paper"), Mapping)
                else {}
            )
            paper_id = (
                _record_id(paper) if isinstance(paper, Mapping) else ""
            )
            hit_paper_rank = source_rank.get(paper_id) or (
                _as_int(hit.get("rank")) or 1
            )
            diag = (
                hit.get("diagnostics")
                if isinstance(hit.get("diagnostics"), Mapping)
                else {}
            )
            diag_nodes: set[str] = set()
            if diag:
                for key in ("matched_nodes", "alias_nodes", "neighbor_nodes"):
                    diag_nodes.update(_string_list(diag.get(key)))
            hit_entries.append((hit_paper_rank, paper_id or "", diag_nodes))
    else:
        # Legacy call without hits: synthesise a stream from
        # retrieved_sources so source-doc overlap still admits edges.
        for source in retrieved_sources:
            paper_id = str(source.get("paper_id") or "")
            paper_rank = int(source.get("rank") or 1)
            hit_entries.append((paper_rank, paper_id, set()))

    scores: dict[str, float] = {}
    chosen_edge: dict[str, dict[str, Any]] = {}

    for edge in edges:
        edge_id = _edge_id(edge)
        if not edge_id:
            continue
        edge_docs = set(_source_docs(edge))
        endpoints = _edge_endpoints(edge)
        admitted_by_endpoint_set = bool(endpoints) and endpoints <= entity_set
        contributed = False
        for hit_paper_rank, paper_id, diag_nodes in hit_entries:
            # (a) source_docs overlap.
            doc_match = bool(paper_id) and paper_id in edge_docs
            # (b) endpoints fully covered by this hit's diagnostics union.
            diag_match = bool(endpoints) and endpoints <= diag_nodes
            # (c) legacy backstop: edge endpoints already in the projected
            # entity set. Counted exactly once (use the top hit's rank).
            if doc_match or diag_match:
                weight = 1.0 / (_K_RRF + max(hit_paper_rank, 1))
                scores[edge_id] = scores.get(edge_id, 0.0) + weight
                chosen_edge.setdefault(edge_id, edge)
                contributed = True
        if not contributed and admitted_by_endpoint_set:
            # Backstop: legacy BM25-style behaviour where neither
            # source_docs nor diagnostics admitted the edge but both
            # endpoints survived the entity projection. Award the
            # rank-1 RRF weight so the edge is still surfaced.
            scores[edge_id] = scores.get(edge_id, 0.0) + 1.0 / (_K_RRF + 1)
            chosen_edge.setdefault(edge_id, edge)

    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[dict[str, Any]] = []
    for rank, (edge_id, _score) in enumerate(items, start=1):
        edge = chosen_edge[edge_id]
        out.append(
            {
                "edge_id": edge_id,
                "rank": rank,
                "score": _safe_score(
                    _as_float(
                        edge.get("reliability_score")
                        or edge.get("weight")
                        or 1.0
                    )
                ),
                "source": str(
                    edge.get("source_id") or edge.get("source")
                    or edge.get("head") or edge.get("from") or ""
                ),
                "target": str(
                    edge.get("target_id") or edge.get("target")
                    or edge.get("tail") or edge.get("to") or ""
                ),
                "relation_type": str(
                    edge.get("relation_type") or edge.get("relation") or ""
                ),
            }
        )
        if len(out) >= top_k:
            break
    return out


def _reconstruct_retrieved_paths(
    *,
    kg: LoadedKg,
    retrieved_entity_ids: list[str],
    retrieved_edges: list[dict[str, Any]],
    top_k: int,
) -> list[list[str]]:
    seed_entity_ids = _path_seed_entity_ids(
        retrieved_entity_ids=retrieved_entity_ids,
        retrieved_edges=retrieved_edges,
        top_k=top_k,
    )
    if len(seed_entity_ids) < 2 or not kg.edges:
        return []

    adjacency = _build_directed_path_adjacency(kg.edges)
    paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for source_id in seed_entity_ids:
        for target_id in seed_entity_ids:
            if source_id == target_id:
                continue
            path = _shortest_path(source_id, target_id, adjacency)
            if path is None:
                continue
            path_key = tuple(path)
            if path_key in seen:
                continue
            seen.add(path_key)
            paths.append(path)
    return paths


def _path_seed_entity_ids(
    *,
    retrieved_entity_ids: list[str],
    retrieved_edges: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    seed_ids = list(retrieved_entity_ids[:top_k])
    for edge in retrieved_edges[:top_k]:
        endpoints = _edge_endpoint_pair(edge)
        if endpoints is None:
            continue
        source_id, target_id = endpoints
        seed_ids.extend([source_id, target_id])
    return _unique_in_order(seed_ids)


def _build_directed_path_adjacency(
    edges: list[dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        endpoints = _edge_endpoint_pair(edge)
        if endpoints is None:
            continue
        source_id, target_id = endpoints
        adjacency.setdefault(source_id, []).append((target_id, _edge_id(edge)))

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (item[0], item[1]))
    return adjacency


def _shortest_path(
    source_id: str,
    target_id: str,
    adjacency: Mapping[str, Sequence[tuple[str, str]]],
) -> list[str] | None:
    frontier: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
    visited = {source_id}
    while frontier:
        current_id, path = frontier.popleft()
        for neighbor_id, edge_id in adjacency.get(current_id, ()):
            if neighbor_id in visited:
                continue
            next_path = path + [edge_id, neighbor_id]
            if neighbor_id == target_id:
                return next_path
            visited.add(neighbor_id)
            frontier.append((neighbor_id, next_path))
    return None


def _edge_endpoint_pair(edge: Mapping[str, Any]) -> tuple[str, str] | None:
    source_id = str(
        edge.get("source_id") or edge.get("source")
        or edge.get("head") or edge.get("from") or ""
    ).strip()
    target_id = str(
        edge.get("target_id") or edge.get("target")
        or edge.get("tail") or edge.get("to") or ""
    ).strip()
    if not source_id or not target_id:
        return None
    return source_id, target_id


def _merge_signals(hits: list[dict[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for hit in hits:
        signals = hit.get("signals")
        if not isinstance(signals, Mapping):
            diagnostics = hit.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                signals = diagnostics.get("signals")
        if not isinstance(signals, Mapping):
            continue
        for key, value in signals.items():
            merged[str(key)] = max(merged.get(str(key), 0.0), _as_float(value))
    return merged


def _partition_audit(
    *,
    retrieved_source_ids: list[str],
    manifest_index: ManifestIndex,
    candidate_partition: str,
) -> dict[str, Any]:
    partitions: dict[str, str] = {}
    out_of_partition: list[str] = []
    for source_id in retrieved_source_ids:
        record = manifest_index.get(source_id)
        partition = _record_partition(record or {})
        partitions[source_id] = partition
        if partition != candidate_partition:
            out_of_partition.append(source_id)
    return {
        "passed": not out_of_partition,
        "candidate_partition": candidate_partition,
        "retrieved_partitions": partitions,
        "retrieved_out_of_partition": out_of_partition,
    }


def _retriever_metadata(retriever: Any) -> dict[str, Any]:
    metadata = getattr(retriever, "metadata", None)
    if isinstance(metadata, Mapping):
        return {"dense": dict(metadata)}

    dense = getattr(retriever, "dense", None)
    dense_metadata = getattr(dense, "metadata", None)
    if isinstance(dense_metadata, Mapping):
        return {"dense": dict(dense_metadata)}
    return {}


def _summarize(
    records: list[dict[str, Any]],
    *,
    system: str,
    candidate_partition: str,
    candidate_count: int,
    qa_count: int,
    top_k: int,
    ablation: AblationSpec | None,
    effective_weights: V3Weights,
    retriever_metadata: dict[str, Any] | None = None,
    toggles: OptionBToggles | None = None,
) -> dict[str, Any]:
    summary = {
        "system": system,
        "status": "ok",
        "candidate_partition": candidate_partition,
        "candidate_count": candidate_count,
        "qa_count": qa_count,
        "n_records": len(records),
        "top_k": top_k,
        "retriever": retriever_metadata or {},
        "source": {
            "hit_at_5_mean": _mean_metric(records, "source_hit_at_5"),
            "mrr_mean": _mean_metric(records, "source_mrr"),
            "recall_at_5_mean": _mean_metric(records, "source_recall_at_5"),
        },
        "by_domain": _summarize_by_domain(records),
        "entity": {
            "precision_at_5_mean": _mean_metric(records, "entity_precision_at_5"),
            "recall_at_5_mean": _mean_metric(records, "entity_recall_at_5"),
            "f1_at_5_mean": _mean_metric(records, "entity_f1_at_5"),
            "hit_at_5_mean": _mean_metric(records, "entity_hit_at_5"),
        },
        "edge": {
            "precision_at_5_mean": _mean_metric(records, "edge_precision_at_5"),
            "recall_at_5_mean": _mean_metric(records, "edge_recall_at_5"),
            "f1_at_5_mean": _mean_metric(records, "edge_f1_at_5"),
            "hit_at_5_mean": _mean_metric(records, "edge_hit_at_5"),
        },
        "path": {
            "exact_match_mean": _mean_metric(records, "path_exact_match"),
            "endpoint_recall_mean": _mean_metric(records, "path_endpoint_recall"),
            "edge_sequence_recall_mean": _mean_metric(
                records, "path_edge_sequence_recall"
            ),
        },
        "source_hit_at_5_mean": _mean_metric(records, "source_hit_at_5"),
        "source_mrr_mean": _mean_metric(records, "source_mrr"),
        "source_recall_at_5_mean": _mean_metric(records, "source_recall_at_5"),
        "entity_f1_at_5_mean": _mean_metric(records, "entity_f1_at_5"),
        "edge_f1_at_5_mean": _mean_metric(records, "edge_f1_at_5"),
    }
    if ablation is not None:
        summary["ablation"] = _ablation_summary(ablation, effective_weights)
    else:
        summary["ablation"] = {}
    if toggles is not None:
        # Persist the resolved Option B (W7 D33-D36) toggle states so each
        # output JSON is self-describing for downstream H8-H11 analysis.
        summary["ablation"]["effective_features"] = {
            "c4_postprocess": toggles.c4_postprocess,
            "c5_query_strategy": toggles.c5_query_strategy,
            "c6_trust": toggles.c6_trust,
            "sub_answer_req": toggles.sub_answer_req,
        }
    # Phase 1 Round 3 fix #3.
    # Shallow-copy retriever_metadata before assignment so the runtime
    # mutation a few lines below ("projection_version") does not leak
    # back into the caller's retriever object. Previously
    # ``summary["retriever"] = retriever_metadata`` shared the dict
    # reference, so attaching projection_version mutated the original
    # _retriever_metadata(retriever) return value -- a latent bug only
    # avoided because no caller currently re-reads that object.
    if retriever_metadata:
        summary["retriever"] = (
            dict(retriever_metadata)
            if isinstance(retriever_metadata, Mapping)
            else retriever_metadata
        )
    # Phase 1 fix #4. Bind the runtime RRF
    # projection version into the payload so preflight_gates.G7 payload-
    # side check can verify it (otherwise payload-G7 was always-PASS).
    # See paper/DECISIONS.md D46 fix #1.
    #
    # Phase 1 Round 3 fix #4.
    # If the decorator marker is somehow missing the runner is mid-
    # regression -- static G7 at startup should have caught it, but as a
    # belt-and-braces guard we raise here instead of silently writing
    # projection_version=None (which would then be flagged by the
    # tightened payload-side G7 anyway, but with a noisier diagnostic).
    retriever_block = summary.setdefault("retriever", {})
    if isinstance(retriever_block, dict):
        version = getattr(_retrieved_entities, "__projection_version__", None)
        if version is None:
            raise RuntimeError(
                "_retrieved_entities lost __projection_version__ marker "
                "(D46 regression). Static G7 should have caught this at "
                "startup; runner refusing to write a payload that would "
                "fail payload-side G7."
            )
        retriever_block["projection_version"] = version
    return summary


def _summarize_by_domain(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        domain = str(record.get("domain") or "unknown")
        grouped.setdefault(domain, []).append(record)

    by_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted(grouped):
        domain_records = grouped[domain]
        by_domain[domain] = {
            "n_records": len(domain_records),
            "source": {
                "hit_at_5_mean": _mean_metric(domain_records, "source_hit_at_5"),
                "mrr_mean": _mean_metric(domain_records, "source_mrr"),
                "recall_at_5_mean": _mean_metric(domain_records, "source_recall_at_5"),
            },
            "entity": {
                "precision_at_5_mean": _mean_metric(domain_records, "entity_precision_at_5"),
                "recall_at_5_mean": _mean_metric(domain_records, "entity_recall_at_5"),
                "f1_at_5_mean": _mean_metric(domain_records, "entity_f1_at_5"),
                "hit_at_5_mean": _mean_metric(domain_records, "entity_hit_at_5"),
            },
            "edge": {
                "precision_at_5_mean": _mean_metric(domain_records, "edge_precision_at_5"),
                "recall_at_5_mean": _mean_metric(domain_records, "edge_recall_at_5"),
                "f1_at_5_mean": _mean_metric(domain_records, "edge_f1_at_5"),
                "hit_at_5_mean": _mean_metric(domain_records, "edge_hit_at_5"),
            },
            "source_hit_at_5_mean": _mean_metric(domain_records, "source_hit_at_5"),
            "source_mrr_mean": _mean_metric(domain_records, "source_mrr"),
            "source_recall_at_5_mean": _mean_metric(domain_records, "source_recall_at_5"),
            "entity_f1_at_5_mean": _mean_metric(domain_records, "entity_f1_at_5"),
            "edge_f1_at_5_mean": _mean_metric(domain_records, "edge_f1_at_5"),
        }
    return by_domain


def _unsupported_payload(
    *,
    system: str,
    candidate_partition: str,
    candidate_count: int,
    qa_count: int,
    top_k: int,
    ablation: AblationSpec | None,
    effective_weights: V3Weights,
) -> dict[str, Any]:
    summary = {
        "summary": {
            "system": system,
            "status": "unsupported",
            "reason": (
                "This ablation is marked unsupported by its ablation config."
            ),
            "supported_systems": sorted(SUPPORTED_SYSTEMS),
            "candidate_partition": candidate_partition,
            "candidate_count": candidate_count,
            "qa_count": qa_count,
            "n_records": 0,
            "top_k": top_k,
        },
        "records": [],
    }
    if ablation is not None:
        summary["summary"]["ablation"] = _ablation_summary(ablation, effective_weights)
    return summary


def _mean_metric(records: list[dict[str, Any]], metric_name: str) -> float:
    if not records:
        return 0.0
    return sum(_as_float(record["metrics"].get(metric_name)) for record in records) / len(records)


def _validate_top_k(top_k: int) -> None:
    if top_k != DEFAULT_TOP_K:
        raise ValueError(
            f"top_k is locked to {DEFAULT_TOP_K} for V3 diagnostics schema compatibility; "
            f"got {top_k}"
        )


def _assert_qa_source_targets_in_partition(
    qa_records: Iterable[dict[str, Any]],
    manifest_index: ManifestIndex,
    *,
    candidate_partition: str,
) -> None:
    issues: list[str] = []
    for qa in qa_records:
        qa_id = str(qa.get("qa_id") or qa.get("id") or "?")
        checked_any = False
        for source_field in ("answer_source", "evidence_source_ids"):
            source_ids = _string_list(qa.get(source_field))
            if not source_ids:
                continue
            checked_any = True
            for source_id in source_ids:
                record = manifest_index.get(source_id)
                if record is None:
                    issues.append(
                        f"QA id={qa_id} {source_field}='{source_id}' missing from manifest"
                    )
                    continue
                partition = _record_partition(record)
                if partition != candidate_partition:
                    actual = partition if partition else "<missing>"
                    issues.append(
                        f"QA id={qa_id} {source_field}='{source_id}' "
                        f"partition='{actual}' expected='{candidate_partition}'"
                    )
        if not checked_any:
            issues.append(f"QA id={qa_id} missing answer_source/evidence_source_ids")
    if issues:
        raise ValueError("QA source partition violation(s): " + "; ".join(issues))


def _load_ablation_spec(
    config_path: Path | None,
    ablation_id: str | None,
) -> AblationSpec | None:
    if config_path is None and ablation_id is None:
        return None
    if config_path is None or not ablation_id:
        raise ValueError("--ablation-config and --ablation-id must be provided together")

    with config_path.open(encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"ablation config must be a mapping: {config_path}")
    ablations = payload.get("ablations")
    if not isinstance(ablations, list):
        raise ValueError(f"ablation config missing list 'ablations': {config_path}")

    for cell in ablations:
        if not isinstance(cell, Mapping):
            continue
        if str(cell.get("id") or "") != ablation_id:
            continue
        weights = cell.get("weights") if isinstance(cell.get("weights"), Mapping) else {}
        return AblationSpec(
            ablation_id=ablation_id,
            system=str(cell.get("system")).lower() if cell.get("system") else None,
            status=str(cell.get("status") or "ok"),
            weights=weights,
            disable=tuple(_string_list(cell.get("disable"))),
            raw=dict(cell),
        )
    raise ValueError(f"ablation id not found in {config_path}: {ablation_id}")


def _resolve_system(system: str, ablation: AblationSpec | None) -> str:
    resolved = (ablation.system if ablation and ablation.system else system).lower()
    if resolved not in ALL_SYSTEMS:
        raise ValueError(f"unsupported system: {resolved!r}")
    return resolved


def _is_unsupported_ablation(ablation: AblationSpec | None) -> bool:
    if ablation is None:
        return False
    return ablation.status.lower().startswith("unsupported")


def _effective_v3_weights(ablation: AblationSpec | None) -> V3Weights:
    base = V3Weights()
    values: dict[str, float] = {
        "bm25": base.bm25,
        "dense": base.dense,
        "alignment": base.alignment,
        "alias": base.alias,
        "schema": base.schema,
        "edge_path": base.edge_path,
        "query_strategy": base.query_strategy,
        "trust": base.trust,
    }
    if ablation is None:
        return V3Weights(**values)

    for raw_key, raw_value in ablation.weights.items():
        key = _weight_key(raw_key)
        if key in values:
            values[key] = _as_float(raw_value)

    features = ablation.raw.get("features")
    if isinstance(features, Mapping):
        for feature, enabled in features.items():
            if enabled is False:
                for key in _disable_weight_keys(str(feature)):
                    values[key] = 0.0

    for disabled in ablation.disable:
        for key in _disable_weight_keys(disabled):
            values[key] = 0.0

    return V3Weights(**values)


def _weight_key(raw_key: object) -> str:
    aliases = {
        "typed_schema": "schema",
        "edges": "edge_path",
        "edge": "edge_path",
        "aliases": "alias",
    }
    key = str(raw_key).strip().lower()
    return aliases.get(key, key)


def _disable_weight_keys(toggle: str) -> tuple[str, ...]:
    key = _weight_key(toggle)
    if key == "dense_rrf":
        return ("dense",)
    if key in {
        "bm25",
        "dense",
        "alignment",
        "alias",
        "schema",
        "edge_path",
        "query_strategy",
        "trust",
    }:
        return (key,)
    return ()


def _ablation_summary(
    ablation: AblationSpec,
    effective_weights: V3Weights,
) -> dict[str, Any]:
    return {
        "id": ablation.ablation_id,
        "system": ablation.system,
        "status": ablation.status,
        "disable": list(ablation.disable),
        "effective_weights": {
            "bm25": effective_weights.bm25,
            "dense": effective_weights.dense,
            "alignment": effective_weights.alignment,
            "alias": effective_weights.alias,
            "schema": effective_weights.schema,
            "edge_path": effective_weights.edge_path,
            "query_strategy": effective_weights.query_strategy,
            "trust": effective_weights.trust,
        },
    }


def _load_manifest_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    if isinstance(manifest, list):
        return [record for record in manifest if isinstance(record, dict)]
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary or list of records")
    for key in ("files", "papers", "records", "sources", "documents", "items"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    if _record_id(manifest):
        return [manifest]
    records: list[dict[str, Any]] = []
    for key, value in manifest.items():
        if isinstance(value, dict):
            record = dict(value)
            record.setdefault("id", str(key))
            records.append(record)
    return records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as err:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {err}") from err
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            records.append(record)
    return records


def _load_kg(kg_dir: Path | None) -> LoadedKg:
    if kg_dir is None:
        return LoadedKg(nodes=[], edges=[])
    return LoadedKg(
        nodes=_load_json_array(kg_dir / "proposed_kg_nodes.json"),
        edges=_load_json_array(kg_dir / "proposed_kg_edges.json"),
    )


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [record for record in payload if isinstance(record, dict)]


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _paper_from_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    paper = dict(record)
    paper.setdefault("id", _record_id(record))
    paper.setdefault("title", str(record.get("title") or ""))
    paper.setdefault("abstract", _paper_body(record))
    return paper


def _paper_text(paper: Mapping[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in (
            paper.get("title"),
            paper.get("abstract"),
            paper.get("content"),
            paper.get("text"),
        )
    )


def _paper_body(record: Mapping[str, Any]) -> str:
    for key in ("abstract", "content", "text", "summary"):
        value = record.get(key)
        if value:
            return str(value)
    text_path = record.get("text_path")
    if isinstance(text_path, str) and text_path.strip():
        path = Path(text_path)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _dense_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in _tokenize(text):
        terms.append(f"tok:{token}")
        padded = f" {token} "
        max_n = min(5, len(padded))
        for ngram_size in range(3, max_n + 1):
            for index in range(0, len(padded) - ngram_size + 1):
                terms.append(f"chr:{padded[index:index + ngram_size]}")
    return terms


def _cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def _normalize_dense_backend(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    aliases = {
        "bge": "bge-m3",
        "baai/bge-m3": "bge-m3",
        "bge-m3-dense": "bge-m3",
        "real": "bge-m3",
        "sentence-transformers": "bge-m3",
        "flagembedding": "bge-m3",
        "deterministic-surrogate": "deterministic",
        "surrogate": "deterministic",
        "offline": "deterministic",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in DENSE_BACKENDS:
        raise ValueError(
            f"unsupported dense backend: {value!r}; "
            f"expected one of {sorted(DENSE_BACKENDS)}"
        )
    return normalized


def _parse_positive_int_env(name: str, value: str | None, *, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError(f"{name} must be an integer; got {value!r}") from err
    if parsed <= 0:
        raise ValueError(f"{name} must be positive; got {parsed}")
    return parsed


def _parse_bool_env(name: str, value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value; got {value!r}")


def _coerce_embedding_matrix(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DenseBackendUnavailable("encoder output must be a sequence of vectors")
    if not value:
        return []

    if _is_number(value[0]):
        return [[float(item) for item in value]]

    matrix: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise DenseBackendUnavailable("encoder output rows must be numeric vectors")
        matrix.append([float(item) for item in row])
    return matrix


def _normalize_dense_vector(vector: Sequence[float]) -> list[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return []
    return [value / norm for value in values]


def _dense_dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("id", "paper_id", "openalex_id", "source_id", "doc_id", "arxiv_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _record_partition(record: Mapping[str, Any]) -> str:
    for key in ("partition", "cutoff_partition", "split", "dataset_split"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _record_domain(
    qa: Mapping[str, Any],
    manifest_index: ManifestIndex,
    gold_source_ids: list[str],
) -> str:
    qa_domain = _domain_value(qa)
    if qa_domain:
        return qa_domain
    for source_id in gold_source_ids:
        record = manifest_index.get(source_id)
        if record is None:
            continue
        source_domain = _domain_value(record)
        if source_domain:
            return source_domain
    return "unknown"


def _domain_value(record: Mapping[str, Any]) -> str:
    for key in ("domain", "domain_tag"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _metric_gold_source_ids(qa: Mapping[str, Any]) -> list[str]:
    return _unique_in_order(
        _string_list(qa.get("answer_source"))
        + _string_list(qa.get("evidence_source_ids"))
    )


def _evidence_source_ids(qa: Mapping[str, Any]) -> list[str]:
    evidence_sources = _string_list(qa.get("evidence_source_ids"))
    return evidence_sources if evidence_sources else _string_list(qa.get("answer_source"))


def _source_docs(item: Mapping[str, Any]) -> list[str]:
    # Edges in proposed_kg_v3_strict_combined_v2 use the singular
    # `source_doc` (one string per edge). Nodes still use the plural
    # `source_docs` list. Coalesce both forms here so the source-rank
    # lookup in _retrieved_edges / _retrieved_entities is non-empty for
    # KG edges; otherwise no edge passes the rank gate and edge_f1
    # collapses to 0 structurally.
    raw_docs = (
        item.get("source_docs")
        or item.get("paper_ids")
        or item.get("sources")
        or item.get("source_doc")
        or []
    )
    return _string_list(raw_docs)


def _node_id(node: Mapping[str, Any]) -> str:
    return str(node.get("node_id") or node.get("id") or "").strip()


def _edge_id(edge: Mapping[str, Any]) -> str:
    """Canonical edge identifier.

    W3 D1 (D43) amendment: emit `{source}--{relation}-->{target}` rather
    than the legacy `{source}|{relation}|{target}` so that ids match the
    qa_from_kg / auto_label_pipeline canonical_edge_ids format and the
    edge_f1 metric stops reading 0% structurally. The legacy pipe format
    can still appear in the input edge dict via `edge_id` / `id`; that
    overrides everything else (back-compat).
    """
    explicit = str(edge.get("edge_id") or edge.get("id") or "").strip()
    if explicit:
        return explicit
    source = str(
        edge.get("source_id") or edge.get("source")
        or edge.get("head") or edge.get("from") or ""
    ).strip()
    relation = str(edge.get("relation_type") or edge.get("relation") or "related_to").strip()
    target = str(
        edge.get("target_id") or edge.get("target")
        or edge.get("tail") or edge.get("to") or ""
    ).strip()
    return "{0}--{1}-->{2}".format(source, relation, target)


def _edge_endpoints(edge: Mapping[str, Any]) -> set[str]:
    return {
        value
        for value in (
            str(
                edge.get("source_id") or edge.get("source")
                or edge.get("head") or edge.get("from") or ""
            ).strip(),
            str(
                edge.get("target_id") or edge.get("target")
                or edge.get("tail") or edge.get("to") or ""
            ).strip(),
        )
        if value
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _path_list(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    paths: list[list[str]] = []
    for item in value:
        if isinstance(item, list):
            path = [str(part).strip() for part in item if str(part).strip()]
            if path:
                paths.append(path)
    return paths


def _safe_score(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BgeM3DenseRetriever",
    "DenseBackendUnavailable",
    "DenseRetrieverConfig",
    "DeterministicDenseRetriever",
    "run_eval",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
