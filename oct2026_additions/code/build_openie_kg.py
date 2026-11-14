# -*- coding: utf-8 -*-
"""Realistic (gold-free) OpenIE KG builder for the 2Wiki public-multihop bench.

Plan option B [public_multihop_integration_plan_2026-05-30.md, sec 3.2]: build
a schema-less open KG from the 1,256 sub-corpus passages with a LOCAL LLM,
WITHOUT ever touching the gold ``evidences`` / ``supporting_facts``. This is the
HippoRAG-style pipeline (NER -> OpenIE -> synonym edge) adapted to single-
paragraph Wikipedia passages.

Pipeline per passage (one Ollama round-trip, single JSON-parse retry):

    S1 NER + OpenIE in one prompt
        -> {entities:[...], triples:[[subj, relation, obj], ...]}
        Open relation phrases (NOT restricted to a fixed schema). Both subj and
        obj must be among the extracted entities (or are added if missing).

Cross-passage assembly:

    * Node id  = "n::" + normalize(surface). Entities sharing a normalized
      surface across passages merge into one node. ``source_docs`` = the set of
      passage_ids the surface was extracted from -> this is the B-passage key
      graph-recall walks to.
    * Edge carries ``source``/``target`` node ids (the keys ProposedV3Retriever
      ._build_adjacency reads) plus ``source_id``/``target_id`` mirrors and the
      open ``relation``/``relation_type`` phrase. ``source_doc`` = the passage
      the triple was extracted from.
    * Synonym/alias edges: BGE-M3 cosine >= tau when available, else a surface-
      normalisation fallback (token-set containment) so alias linking degrades
      gracefully with zero extra deps. Synonym edges let graph-recall hop across
      surface variants of the same entity (HippoRAG synonymy edges).

Output format is byte-compatible with the silver KG
(``proposed_kg_nodes.json`` / ``proposed_kg_edges.json``) so ``_load_kg`` and
the whole eval path are reused with ZERO retriever/eval code changes.

CLI
---
    python paper/baselines/build_openie_kg.py \
        --passages paper/external/2wiki/sub200/passages.jsonl \
        --out-dir  paper/external/2wiki/sub200/kg_openie \
        --model    llama3.1:8b \
        --synonyms surface
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger("build_openie_kg")

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Surfaces too generic to ever be a node or to drive alias matching.
_STOP_SURFACES = frozenset(
    {
        "he",
        "she",
        "it",
        "they",
        "him",
        "her",
        "his",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "who",
        "which",
        "what",
        "none",
        "null",
        "n/a",
        "unknown",
        "",
    }
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_surface(text: str) -> str:
    return _WS_RE.sub(" ", str(text).strip()).lower()


def node_id_for(surface: str) -> str:
    return "n::" + normalize_surface(surface)


def _content_tokens(surface: str) -> set[str]:
    return {
        tok
        for tok in _TOKEN_RE.findall(normalize_surface(surface))
        if tok not in _STOP_SURFACES and len(tok) > 1
    }


def _strip_json(text: str) -> str:
    """Extract the first balanced JSON object from ``text`` (fences tolerated)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


# ---------------------------------------------------------------------------
# Extraction prompt (schema-less open IE)
# ---------------------------------------------------------------------------
OPENIE_PROMPT = """You are an information extraction system. Read ONE Wikipedia passage and extract a knowledge graph from it.

Return ONLY a JSON object with two keys:
  "entities": a list of distinct named entities mentioned (people, places, organizations, works, dates as written). Use the surface form exactly as it appears. No pronouns.
  "triples": a list of [subject, relation, object] facts stated in the passage. The subject and object MUST be entity surfaces. The relation is a short verb phrase (1-4 words, lowercase), e.g. "born in", "directed by", "father", "located in", "member of", "spouse". Only extract facts explicitly stated in the passage.

Rules:
- Do NOT invent facts not in the passage.
- Keep entity surfaces verbatim (preserve casing).
- 3 to 12 triples is typical; extract what the passage states.

PASSAGE:
{passage_text}

Return ONLY the JSON object, nothing else."""

JSON_RETRY_SUFFIX = (
    "\n\nReturn ONLY a single valid JSON object with keys \"entities\" and "
    '"triples". No prose, no markdown fences.'
)


# ---------------------------------------------------------------------------
# Extracted records
# ---------------------------------------------------------------------------
@dataclass
class _NodeAccum:
    name: str
    source_docs: "OrderedDict[str, None]" = field(default_factory=OrderedDict)
    aliases: "OrderedDict[str, None]" = field(default_factory=OrderedDict)


# ---------------------------------------------------------------------------
# Ollama extractor (schema-less)
# ---------------------------------------------------------------------------
class OpenIEExtractor:
    """Single-call NER+OpenIE extractor over one passage.

    Tests inject a mock via the ``ollama_client`` setter, mirroring
    ThreeStageLLMExtractor so no network is required in unit tests.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        num_predict: int = 700,
        seed: int = 0,
        log_sink=None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.seed = seed
        self.log_sink = log_sink
        self._ollama = None
        try:
            import ollama  # type: ignore

            self._ollama = ollama
        except ImportError:
            self._ollama = None

    @property
    def ollama_client(self):
        return self._ollama

    @ollama_client.setter
    def ollama_client(self, value) -> None:
        self._ollama = value

    def extract(self, passage_id: str, text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
        """Return (entities, triples) for one passage. Empty on any failure."""
        prompt = OPENIE_PROMPT.format(passage_text=text[:3500])
        parsed = self._chat_json(prompt, passage_id=passage_id)
        if not parsed:
            return [], []
        entities: list[str] = []
        for ent in parsed.get("entities") or []:
            if isinstance(ent, str) and ent.strip():
                entities.append(ent.strip())
            elif isinstance(ent, dict):
                name = str(ent.get("name") or ent.get("entity") or "").strip()
                if name:
                    entities.append(name)
        triples: list[tuple[str, str, str]] = []
        for tri in parsed.get("triples") or []:
            if isinstance(tri, (list, tuple)) and len(tri) == 3:
                s, r, o = (str(tri[0]).strip(), str(tri[1]).strip(), str(tri[2]).strip())
            elif isinstance(tri, dict):
                s = str(tri.get("subject") or tri.get("subj") or tri.get("head") or "").strip()
                r = str(tri.get("relation") or tri.get("rel") or tri.get("predicate") or "").strip()
                o = str(tri.get("object") or tri.get("obj") or tri.get("tail") or "").strip()
            else:
                continue
            if s and r and o and normalize_surface(s) != normalize_surface(o):
                if normalize_surface(s) in _STOP_SURFACES or normalize_surface(o) in _STOP_SURFACES:
                    continue
                triples.append((s, r, o))
        return entities, triples

    # ---- internals -------------------------------------------------------
    def _chat_once(self, prompt: str) -> str:
        if self._ollama is None:
            raise RuntimeError("Ollama client not available")
        response = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        )
        try:
            return str(response["message"]["content"])
        except (TypeError, KeyError):
            return str(getattr(getattr(response, "message", None), "content", ""))

    def _chat_json(self, prompt: str, *, passage_id: str) -> Optional[dict[str, Any]]:
        t0 = time.time()
        try:
            text = self._chat_once(prompt)
        except Exception as exc:  # noqa: BLE001
            self._log(passage_id, f"call_error:{type(exc).__name__}", 0, t0)
            return None
        raw = _strip_json(text)
        parsed: Optional[dict[str, Any]] = None
        status = "ok"
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                status = "parse_fail"
        else:
            status = "no_json"
        if parsed is None:
            try:
                text2 = self._chat_once(prompt + JSON_RETRY_SUFFIX)
            except Exception as exc:  # noqa: BLE001
                self._log(passage_id, f"retry_call_error:{type(exc).__name__}", len(text), t0)
                return None
            raw2 = _strip_json(text2)
            if raw2:
                try:
                    parsed = json.loads(raw2)
                    status = "ok_after_retry"
                except json.JSONDecodeError:
                    status = "parse_fail_after_retry"
            else:
                status = "no_json_after_retry"
            text = text2
        if not isinstance(parsed, dict):
            self._log(passage_id, status if parsed is None else "not_object", len(text), t0)
            return None
        self._log(passage_id, status, len(text), t0)
        return parsed

    def _log(self, passage_id: str, status: str, n_out: int, t0: float) -> None:
        if self.log_sink is None:
            return
        try:
            self.log_sink(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "passage_id": passage_id,
                    "parse_status": status,
                    "n_chars_out": n_out,
                    "elapsed_s": round(time.time() - t0, 3),
                }
            )
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Synonym / alias edges
# ---------------------------------------------------------------------------
def surface_synonym_pairs(
    node_ids: Sequence[str],
    names_by_id: Mapping[str, str],
    *,
    min_tokens: int = 2,
) -> list[tuple[str, str]]:
    """Token-containment synonym pairs (zero-dependency fallback).

    Two nodes are synonyms when one's content-token set is a non-trivial
    subset of the other's (e.g. "Gerald Ford" vs "Gerald Rudolff Ford"). The
    smaller set must have >= ``min_tokens`` tokens to avoid linking on a single
    generic first name. Deterministic, no embeddings.
    """
    toks: dict[str, set[str]] = {nid: _content_tokens(names_by_id.get(nid, "")) for nid in node_ids}
    ids = [nid for nid in node_ids if len(toks[nid]) >= min_tokens]
    ids.sort()
    pairs: list[tuple[str, str]] = []
    for i in range(len(ids)):
        a = ids[i]
        ta = toks[a]
        for j in range(i + 1, len(ids)):
            b = ids[j]
            tb = toks[b]
            if ta == tb:
                continue
            smaller, larger = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
            if len(smaller) >= min_tokens and smaller <= larger:
                pairs.append((a, b))
    return pairs


def embedding_synonym_pairs(
    node_ids: Sequence[str],
    names_by_id: Mapping[str, str],
    *,
    tau: float = 0.8,
) -> list[tuple[str, str]]:
    """BGE-M3 cosine synonym pairs (HippoRAG synonymy edges, tau=0.8).

    Returns [] (caller falls back to surface mode) if the model is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        logger.warning("sentence-transformers/BGE-M3 unavailable; no embedding synonyms")
        return []
    ids = [nid for nid in node_ids if names_by_id.get(nid)]
    if len(ids) < 2:
        return []
    try:
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(
            [names_by_id[nid] for nid in ids],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("BGE-M3 encode failed (%s); no embedding synonyms", exc)
        return []
    embs = np.asarray(embs, dtype="float32")
    sims = embs @ embs.T
    pairs: list[tuple[str, str]] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sims[i, j]) >= tau:
                pairs.append((ids[i], ids[j]))
    return pairs


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class OpenIEKGBuilder:
    def __init__(
        self,
        passages_path: Path,
        out_dir: Path,
        *,
        model: str = "llama3.1:8b",
        synonyms: str = "surface",
        synonym_tau: float = 0.8,
        extractor: Optional[OpenIEExtractor] = None,
        max_passages: Optional[int] = None,
    ) -> None:
        self.passages_path = Path(passages_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.synonyms = synonyms
        self.synonym_tau = synonym_tau
        self.max_passages = max_passages

        self._nodes_path = self.out_dir / "proposed_kg_nodes.json"
        self._edges_path = self.out_dir / "proposed_kg_edges.json"
        self._log_path = self.out_dir / "build_log.jsonl"
        self._checkpoint_path = self.out_dir / "checkpoint.json"
        self._summary_path = self.out_dir / "build_summary.json"

        self.extractor = extractor or OpenIEExtractor(
            model=model, log_sink=self._append_log
        )
        if self.extractor.log_sink is None:
            self.extractor.log_sink = self._append_log

        # Accumulators (rebuilt on resume from checkpoint).
        self._nodes: "OrderedDict[str, _NodeAccum]" = OrderedDict()
        self._edges: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._processed: set[str] = set()
        self._n_parse_fail = 0
        self._n_calls = 0

    # ---- public ----------------------------------------------------------
    def build(self, *, checkpoint_every: int = 50, verbose: bool = False) -> dict[str, Any]:
        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        t0 = time.time()
        passages = self._load_passages()
        if self.max_passages is not None:
            passages = passages[: self.max_passages]
        self._load_checkpoint()

        total = len(passages)
        n_new = 0
        for idx, p in enumerate(passages):
            pid = str(p.get("passage_id") or p.get("paper_id") or p.get("id") or "")
            if not pid or pid in self._processed:
                continue
            text = str(p.get("text") or p.get("abstract") or "")
            self._n_calls += 1
            entities, triples = self.extractor.extract(pid, text)
            if not entities and not triples:
                self._n_parse_fail += 1
            self._absorb(pid, entities, triples)
            self._processed.add(pid)
            n_new += 1
            if verbose and n_new % 25 == 0:
                logger.info(
                    "progress %d/%d (new=%d) nodes=%d edges=%d",
                    idx + 1,
                    total,
                    n_new,
                    len(self._nodes),
                    len(self._edges),
                )
            if checkpoint_every and n_new % checkpoint_every == 0:
                self._dump_partial()
                self._write_checkpoint()

        # Synonym edges over the final node set.
        n_syn = self._add_synonym_edges()

        self._dump_partial()
        self._write_checkpoint()
        summary = {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "n_synonym_edges": n_syn,
            "n_relation_edges": len(self._edges) - n_syn,
            "n_passages": total,
            "n_passages_processed": len(self._processed),
            "n_parse_fail": self._n_parse_fail,
            "parse_fail_rate": round(self._n_parse_fail / self._n_calls, 4) if self._n_calls else 0.0,
            "nodes_with_source_docs": sum(1 for a in self._nodes.values() if a.source_docs),
            "model": self.model,
            "synonyms": self.synonyms,
            "wall_clock_seconds": round(time.time() - t0, 2),
            "kg_option": "B_openie_realistic",
        }
        with self._summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        return summary

    # ---- assembly --------------------------------------------------------
    def _absorb(
        self,
        passage_id: str,
        entities: Sequence[str],
        triples: Sequence[tuple[str, str, str]],
    ) -> None:
        def _touch(surface: str) -> Optional[str]:
            ns = normalize_surface(surface)
            if not ns or ns in _STOP_SURFACES:
                return None
            nid = "n::" + ns
            accum = self._nodes.get(nid)
            if accum is None:
                accum = _NodeAccum(name=surface)
                self._nodes[nid] = accum
            accum.source_docs.setdefault(passage_id, None)
            return nid

        for ent in entities:
            _touch(ent)
        for subj, relation, obj in triples:
            sid = _touch(subj)
            oid = _touch(obj)
            if not sid or not oid or sid == oid:
                continue
            rel_norm = normalize_surface(relation) or "related to"
            # Guard against the LLM dumping a whole clause as the relation: keep
            # a relation-like phrase (first 6 tokens). The edge endpoints --
            # which are what graph-recall walks -- are unaffected.
            rel_tokens = rel_norm.split()
            if len(rel_tokens) > 6:
                rel_norm = " ".join(rel_tokens[:6])
            edge_key = f"{sid}|{rel_norm}|{oid}"
            existing = self._edges.get(edge_key)
            if existing is None:
                self._edges[edge_key] = {
                    "edge_id": edge_key,
                    "source": sid,
                    "target": oid,
                    "source_id": sid,
                    "target_id": oid,
                    "relation": rel_norm,
                    "relation_type": rel_norm,
                    "source_doc": passage_id,
                    "source_docs": [passage_id],
                    "reliability_score": 0.85,
                }
            else:
                docs = existing.setdefault("source_docs", [])
                if passage_id not in docs:
                    docs.append(passage_id)

    def _add_synonym_edges(self) -> int:
        node_ids = list(self._nodes.keys())
        names = {nid: a.name for nid, a in self._nodes.items()}
        pairs: list[tuple[str, str]] = []
        if self.synonyms == "bge-m3":
            pairs = embedding_synonym_pairs(node_ids, names, tau=self.synonym_tau)
            if not pairs:
                logger.warning("BGE-M3 produced no pairs; falling back to surface synonyms")
                pairs = surface_synonym_pairs(node_ids, names)
                self.synonyms = "surface_fallback"
        elif self.synonyms == "surface":
            pairs = surface_synonym_pairs(node_ids, names)
        # "none" -> no synonym edges
        n = 0
        for a, b in pairs:
            edge_key = f"{a}|synonym|{b}"
            if edge_key in self._edges:
                continue
            self._edges[edge_key] = {
                "edge_id": edge_key,
                "source": a,
                "target": b,
                "source_id": a,
                "target_id": b,
                "relation": "synonym",
                "relation_type": "synonym",
                "source_doc": "",
                "source_docs": [],
                "reliability_score": 0.7,
            }
            n += 1
        return n

    # ---- IO --------------------------------------------------------------
    def _load_passages(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.passages_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _node_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for nid, accum in self._nodes.items():
            out.append(
                {
                    "node_id": nid,
                    "name": accum.name,
                    "type": "entity",
                    "aliases": list(accum.aliases.keys()),
                    "source_docs": list(accum.source_docs.keys()),
                }
            )
        return out

    def _dump_partial(self) -> None:
        self._atomic_json(self._nodes_path, self._node_records())
        self._atomic_json(self._edges_path, list(self._edges.values()))

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _append_log(self, record: Mapping[str, Any]) -> None:
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_checkpoint(self) -> None:
        tmp = self._checkpoint_path.with_suffix(".json.tmp")
        payload = {
            "processed_ids": sorted(self._processed),
            "nodes": [
                {
                    "node_id": nid,
                    "name": a.name,
                    "source_docs": list(a.source_docs.keys()),
                    "aliases": list(a.aliases.keys()),
                }
                for nid, a in self._nodes.items()
            ],
            "edges": list(self._edges.values()),
            "n_parse_fail": self._n_parse_fail,
            "n_calls": self._n_calls,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, self._checkpoint_path)

    def _load_checkpoint(self) -> None:
        if not self._checkpoint_path.exists():
            return
        try:
            with self._checkpoint_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        self._processed = set(data.get("processed_ids") or [])
        for n in data.get("nodes") or []:
            accum = _NodeAccum(name=n.get("name") or "")
            for d in n.get("source_docs") or []:
                accum.source_docs.setdefault(d, None)
            for al in n.get("aliases") or []:
                accum.aliases.setdefault(al, None)
            self._nodes[n["node_id"]] = accum
        for e in data.get("edges") or []:
            # Skip synonym edges on resume; they are recomputed at the end.
            if e.get("relation_type") == "synonym":
                continue
            self._edges[e["edge_id"]] = e
        self._n_parse_fail = int(data.get("n_parse_fail") or 0)
        self._n_calls = int(data.get("n_calls") or 0)
        logger.info(
            "resumed: %d processed, %d nodes, %d edges",
            len(self._processed),
            len(self._nodes),
            len(self._edges),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Realistic OpenIE KG builder (2Wiki option B).")
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument(
        "--synonyms",
        choices=["surface", "bge-m3", "none"],
        default="surface",
        help="alias/synonym edge source",
    )
    parser.add_argument("--synonym-tau", type=float, default=0.8)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    builder = OpenIEKGBuilder(
        passages_path=args.passages,
        out_dir=args.out_dir,
        model=args.model,
        synonyms=args.synonyms,
        synonym_tau=args.synonym_tau,
        max_passages=args.max_passages,
    )
    summary = builder.build(checkpoint_every=args.checkpoint_every, verbose=args.verbose)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
