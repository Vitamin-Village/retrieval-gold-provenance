"""Strict corpus split policy checks for gold and evaluation records."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PartitionPolicy:
    """Expected source partitions for gold labels and evaluation QA."""

    gold_partition: str = "gold"
    eval_partition: str = "eval"


@dataclass(frozen=True)
class ManifestIndex:
    """Index manifest source records by paper/source id."""

    records_by_id: Mapping[str, dict]

    @classmethod
    def from_manifest(cls, manifest: dict | list[dict]) -> "ManifestIndex":
        """Build an index from a manifest dictionary or a list of records."""
        records = _extract_manifest_records(manifest)
        indexed: dict[str, dict] = {}
        for record in records:
            paper_id = _record_id(record)
            if paper_id:
                indexed[paper_id] = record
        return cls(records_by_id=indexed)

    @classmethod
    def from_path(cls, path: str | Path) -> "ManifestIndex":
        """Load a JSON manifest from disk and build an index."""
        manifest_path = Path(path)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        return cls.from_manifest(manifest)

    def get(self, paper_id: str) -> dict | None:
        """Return a manifest record for ``paper_id`` if present."""
        return self.records_by_id.get(str(paper_id))

    def __contains__(self, paper_id: object) -> bool:
        return str(paper_id) in self.records_by_id


def assert_answer_sources_in_partition(
    qa_records: Iterable[dict],
    manifest_index: ManifestIndex,
    policy: PartitionPolicy,
) -> None:
    """Require every QA answer source to exist in the eval partition."""
    issues: list[str] = []
    for qa in qa_records:
        qa_id = str(qa.get("qa_id") or qa.get("id") or "?")
        source_ids = _source_ids(qa.get("answer_source"))
        if not source_ids:
            issues.append(f"QA id={qa_id} missing answer_source")
            continue
        for source_id in source_ids:
            issues.extend(
                _partition_issues(
                    record_kind="QA",
                    record_id=qa_id,
                    source_field="answer_source",
                    source_id=source_id,
                    manifest_index=manifest_index,
                    expected_partition=policy.eval_partition,
                )
            )
    if issues:
        raise ValueError(
            "QA answer_source partition violation(s): " + "; ".join(issues)
        )


def assert_gold_sources_in_partition(
    entity_records: Iterable[dict],
    edge_records: Iterable[dict],
    manifest_index: ManifestIndex,
    policy: PartitionPolicy,
) -> None:
    """Require every entity/edge source paper to exist in the gold partition."""
    issues: list[str] = []
    for entity in entity_records:
        entity_id = str(entity.get("id") or "?")
        issues.extend(
            _source_partition_issues(
                record_kind="Entity",
                record_id=entity_id,
                source_field="source_paper",
                source_value=entity.get("source_paper"),
                manifest_index=manifest_index,
                expected_partition=policy.gold_partition,
            )
        )
    for edge in edge_records:
        edge_id = str(edge.get("edge_id") or edge.get("id") or "?")
        issues.extend(
            _source_partition_issues(
                record_kind="Edge",
                record_id=edge_id,
                source_field="source_paper",
                source_value=edge.get("source_paper"),
                manifest_index=manifest_index,
                expected_partition=policy.gold_partition,
            )
        )
    if issues:
        raise ValueError(
            "Gold source_paper partition violation(s): " + "; ".join(issues)
        )


def find_duplicate_ids(records: Iterable[dict]) -> list[list[dict]]:
    """Return groups of records sharing the same source id."""
    return _duplicate_groups(records, lambda record: _record_id(record))


def find_duplicate_titles(records: Iterable[dict]) -> list[list[dict]]:
    """Return groups of records with the same normalized title."""
    return _duplicate_groups(records, lambda record: _normalize_title(record.get("title")))


def find_duplicate_dois(records: Iterable[dict]) -> list[list[dict]]:
    """Return groups of records with the same normalized DOI."""
    return _duplicate_groups(records, lambda record: _normalize_doi(record.get("doi")))


def _source_partition_issues(
    *,
    record_kind: str,
    record_id: str,
    source_field: str,
    source_value: object,
    manifest_index: ManifestIndex,
    expected_partition: str,
) -> list[str]:
    source_ids = _source_ids(source_value)
    if not source_ids:
        return [f"{record_kind} id={record_id} missing {source_field}"]

    issues: list[str] = []
    for source_id in source_ids:
        issues.extend(
            _partition_issues(
                record_kind=record_kind,
                record_id=record_id,
                source_field=source_field,
                source_id=source_id,
                manifest_index=manifest_index,
                expected_partition=expected_partition,
            )
        )
    return issues


def _partition_issues(
    *,
    record_kind: str,
    record_id: str,
    source_field: str,
    source_id: str,
    manifest_index: ManifestIndex,
    expected_partition: str,
) -> list[str]:
    source_record = manifest_index.get(source_id)
    if source_record is None:
        return [
            f"{record_kind} id={record_id} {source_field}='{source_id}' missing from manifest"
        ]

    partition = _record_partition(source_record)
    if partition != expected_partition:
        actual = partition if partition else "<missing>"
        return [
            f"{record_kind} id={record_id} {source_field}='{source_id}' "
            f"partition='{actual}' expected='{expected_partition}'"
        ]
    return []


def _extract_manifest_records(manifest: dict | list[dict]) -> list[dict]:
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

    records: list[dict] = []
    for key, value in manifest.items():
        if isinstance(value, dict):
            record = dict(value)
            record.setdefault("id", str(key))
            records.append(record)
    return records


def _record_id(record: Mapping[str, object]) -> str:
    for key in ("id", "paper_id", "openalex_id", "source_id", "doc_id", "arxiv_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _record_partition(record: Mapping[str, object]) -> str:
    for key in ("partition", "cutoff_partition", "split", "dataset_split"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _source_ids(source_value: object) -> list[str]:
    if source_value is None:
        return []
    if isinstance(source_value, str):
        source_id = source_value.strip()
        return [source_id] if source_id else []
    if isinstance(source_value, Sequence) and not isinstance(source_value, (bytes, bytearray)):
        return [str(item).strip() for item in source_value if str(item).strip()]
    source_id = str(source_value).strip()
    return [source_id] if source_id else []


def _duplicate_groups(records: Iterable[dict], key_func) -> list[list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = key_func(record)
        if key:
            grouped[key].append(record)
    return [group for group in grouped.values() if len(group) > 1]


def _normalize_title(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _normalize_doi(value: object) -> str:
    if value is None:
        return ""
    doi = str(value).strip().casefold()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()
