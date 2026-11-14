"""Shared corpus loader for W4 baselines.

Loads the 960-paper corpus manifest and lazily fetches/caches abstracts via the
underlying OA sources (OpenAlex / S2 / PMC / arXiv).  Caches abstracts on disk
under ``paper/baselines/abstract_cache/{paper_id}.json`` so repeat baseline
runs do not re-hit the external APIs.

Public API
----------
- ``load_manifest(path)``             -> list[dict]
- ``filter_papers(papers, ...)``      -> list[dict]
- ``fetch_abstract(paper, cache_dir)``-> str
- ``load_corpus_with_abstracts(...)`` -> list[dict] (id, title, abstract, ...)
- ``load_qa(qa_path)``                -> list[dict]
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import requests  # type: ignore
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


logger = logging.getLogger(__name__)

DEFAULT_MAILTO = "anonymous@example.org"
DEFAULT_USER_AGENT = (
    f"GraphRAG-Baselines/0.1 (mailto:{DEFAULT_MAILTO})"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "paper" / "corpus" / "manifest.json"
DEFAULT_QA = REPO_ROOT / "paper" / "gold" / "qa_100.jsonl"
DEFAULT_CACHE_DIR = REPO_ROOT / "paper" / "baselines" / "abstract_cache"

VALID_PARTITIONS = {"gold", "eval"}
VALID_DOMAINS = {"semiconductor", "construction", "textile", "steel"}


# ---------------------------------------------------------------------------
# Manifest / QA loaders
# ---------------------------------------------------------------------------

def load_manifest(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return paper records from the legacy or arXiv full-text manifest."""
    path = Path(path) if path is not None else DEFAULT_MANIFEST
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if isinstance(manifest, list):
        records = manifest
    else:
        records = manifest.get("files") or manifest.get("records") or []
    if not isinstance(records, list):
        raise ValueError(f"manifest records must be a list, got {type(records).__name__}")
    return records


def filter_papers(
    papers: Iterable[Dict[str, Any]],
    *,
    domain: Optional[str] = None,
    partition: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Apply domain / partition / limit filters and return a list."""
    out: List[Dict[str, Any]] = []
    for p in papers:
        if domain and p.get("domain_tag") != domain:
            continue
        if partition and p.get("cutoff_partition") != partition:
            continue
        out.append(p)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_qa(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read ``qa_100.jsonl`` and return one dict per line."""
    path = Path(path) if path is not None else DEFAULT_QA
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Abstract fetching (with disk cache)
# ---------------------------------------------------------------------------

class AbstractFetcher:
    """Cache-first abstract fetcher dispatching to the right OA endpoint.

    Each abstract lands at ``{cache_dir}/{sanitized_id}.json`` so repeated
    baseline runs share the cache.  Network errors fall back to the title
    only — the baseline still has *something* to retrieve over.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        mailto: str = DEFAULT_MAILTO,
        timeout: float = 15.0,
        rate_limit: float = 0.5,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mailto = mailto
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._last_call = 0.0

    # ---- public ----------------------------------------------------------

    def fetch(self, paper: Dict[str, Any]) -> str:
        """Return abstract text for ``paper`` (cached, may be empty)."""
        cache_path = self.cache_dir / f"{self._safe_id(paper['id'])}.json"
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                return cached.get("abstract") or ""
            except (json.JSONDecodeError, OSError):
                pass

        abstract = ""
        try:
            if not _REQUESTS_OK:
                abstract = ""
            elif paper["source"] == "openalex":
                abstract = self._fetch_openalex(paper["id"])
            elif paper["source"] == "s2":
                abstract = self._fetch_s2(paper["id"])
            elif paper["source"] == "pmc":
                abstract = self._fetch_pmc(paper["id"])
            elif paper["source"] == "arxiv":
                abstract = self._fetch_arxiv(paper["id"])
        except Exception as exc:  # pragma: no cover - network noise
            logger.warning("Abstract fetch failed for %s: %s", paper.get("id"), exc)
            abstract = ""

        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump({"id": paper["id"], "abstract": abstract}, fh, ensure_ascii=False)
        return abstract

    # ---- per-source fetchers --------------------------------------------

    def _fetch_openalex(self, paper_id: str) -> str:
        # id format: openalex:1234567890
        work_id = paper_id.split(":", 1)[1]
        url = f"https://api.openalex.org/works/W{work_id}"
        payload = self._get(url, params={"mailto": self.mailto})
        if payload is None:
            return ""
        return self._rebuild_inverted_index(payload.get("abstract_inverted_index"))

    def _fetch_s2(self, paper_id: str) -> str:
        s2_id = paper_id.split(":", 1)[1]
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}"
        payload = self._get(url, params={"fields": "abstract"})
        if payload is None:
            return ""
        return (payload.get("abstract") or "").strip()

    def _fetch_pmc(self, paper_id: str) -> str:
        pmc_id = paper_id.split(":", 1)[1]
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pmc&id={pmc_id}&rettype=abstract&retmode=text"
        )
        text = self._get_text(url)
        if not text:
            return ""
        # Strip headers / footers heuristically
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]

    def _fetch_arxiv(self, paper_id: str) -> str:
        arxiv_id = paper_id.split(":", 1)[1]
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        text = self._get_text(url)
        if not text:
            return ""
        m = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        if not m:
            return ""
        return re.sub(r"\s+", " ", m.group(1)).strip()

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _safe_id(paper_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)

    def _rate_limit_sleep(self) -> None:
        delta = time.time() - self._last_call
        if delta < self.rate_limit:
            time.sleep(self.rate_limit - delta)
        self._last_call = time.time()

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        self._rate_limit_sleep()
        try:
            r = requests.get(  # type: ignore[union-attr]
                url,
                params=params or {},
                headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def _get_text(self, url: str) -> str:
        self._rate_limit_sleep()
        try:
            r = requests.get(  # type: ignore[union-attr]
                url,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return ""
            return r.text
        except Exception:
            return ""

    @staticmethod
    def _rebuild_inverted_index(inverted: Optional[Dict[str, List[int]]]) -> str:
        if not inverted:
            return ""
        positions: Dict[int, str] = {}
        for token, pos_list in inverted.items():
            if not isinstance(pos_list, list):
                continue
            for pos in pos_list:
                if isinstance(pos, int):
                    positions[pos] = token
        if not positions:
            return ""
        return " ".join(positions[i] for i in sorted(positions))


def load_corpus_with_abstracts(
    manifest_path: Optional[Path] = None,
    *,
    domain: Optional[str] = None,
    partition: Optional[str] = "gold",
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    fetch: bool = True,
    progress_every: int = 50,
) -> List[Dict[str, Any]]:
    """Return ``[{id, title, abstract, domain_tag, ...}]`` with cached abstracts."""
    papers = filter_papers(
        load_manifest(manifest_path),
        domain=domain,
        partition=partition,
        limit=limit,
    )

    fetcher = AbstractFetcher(cache_dir=cache_dir) if fetch else None
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(papers):
        abstract = p.get("abstract") or ""
        if fetcher is not None:
            abstract = fetcher.fetch(p) or abstract
        record = {
            "id": p["id"],
            "source": p.get("source"),
            "title": p.get("title") or "",
            "abstract": abstract,
            "domain_tag": p.get("domain_tag"),
            "doi": p.get("doi"),
            "published_date": p.get("published_date"),
            "raw_url": p.get("raw_url"),
            "cutoff_partition": p.get("cutoff_partition"),
        }
        for key in (
            "text_path",
            "chunk_path",
            "pdf_path",
            "parse_status",
            "char_count",
            "arxiv_id",
            "categories",
        ):
            if key in p:
                record[key] = p[key]
        out.append(record)
        if progress_every and (i + 1) % progress_every == 0:
            logger.info("loaded %d/%d (%s)", i + 1, len(papers), domain or "all")
    return out


__all__ = [
    "AbstractFetcher",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MANIFEST",
    "DEFAULT_QA",
    "filter_papers",
    "load_corpus_with_abstracts",
    "load_manifest",
    "load_qa",
]
