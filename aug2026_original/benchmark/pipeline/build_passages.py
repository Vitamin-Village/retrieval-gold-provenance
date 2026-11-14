# -*- coding: utf-8 -*-
"""S1: Passage pool builder for the independent (acyclic) QA benchmark.

Inputs : corpus manifest.json (fields: id, text_path, domain_tag, char_count,
         chunk_path) -> the full text of all 205 papers.
Outputs: paper/gold_v2/passages.jsonl with one row per passage:
         {passage_id, paper_id, domain, text, char_span}.

NO model is called here (S1 is model-free). Each passage is attributed to
exactly one paper_id. Text is ASCII-normalized for Windows cp949 console
safety.

Acyclicity: this module imports ONLY the corpus manifest text fields. It does
NOT import the proposed KG, baseline retrievers, or qa_from_kg. See design
doc section 8 (LOCKED DECISIONS).

DO NOT add hyperparameter tuning. The two chunking constants below are derived
from a standard tokens<->characters ratio, not tuned, and are justified in
comments.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Paths (ASCII-safe).
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "paper"
    / "corpus"
    / "arxiv_fulltext_strict_combined_v1"
    / "manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "gold_v2" / "passages.jsonl"

# ---------------------------------------------------------------------------
# Chunking constants.
#
# Design S1 requires 600-1000 token chunks. The corpus has no usable
# chunk_path on disk (verified: 0/205 chunk files exist), so we approximate
# token boundaries from characters. We use the widely-cited ~4 characters per
# English token ratio (OpenAI tokenizer rule-of-thumb) to map the token target
# to a character target. This is a documented standard ratio, NOT a tuned
# hyperparameter:
#   800 tokens (midpoint of 600-1000) * 4 chars/token = 3200 chars target.
#   600 tokens * 4 = 2400 chars min; 1000 tokens * 4 = 4000 chars max.
# We chunk on whitespace-token counts (split on whitespace) so the actual
# bound is in tokens, and only fall back to the char target for very long
# whitespace-free runs.
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4  # standard English tokenizer rule-of-thumb (not tuned)
TARGET_TOKENS = 800  # midpoint of the 600-1000 design window
MIN_TOKENS = 600
MAX_TOKENS = 1000
# Hard character ceiling guards against pathological whitespace-free text so a
# single "token" cannot exceed the design's char budget.
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN  # 4000


# ---------------------------------------------------------------------------
# ASCII normalization.
# ---------------------------------------------------------------------------


def ascii_normalize(text: str) -> str:
    """Return an ASCII-only version of text (Windows cp949 safety).

    Unicode is NFKD-decomposed and non-ASCII codepoints are dropped after the
    decomposition strips accents. Whitespace is collapsed to single spaces but
    paragraph structure is preserved as single newlines.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_bytes = decomposed.encode("ascii", "ignore")
    out = ascii_bytes.decode("ascii")
    # Normalize newlines, then collapse runs of spaces/tabs but keep newlines.
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# Manifest loading.
# ---------------------------------------------------------------------------


def load_manifest_records(manifest_path: Path) -> List[dict]:
    """Load the manifest and return its list of file records.

    Raises ValueError if the manifest is missing the expected 'files' list or
    any record lacks the required fields.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest missing non-empty 'files' list")
    required = ("id", "text_path", "domain_tag")
    for rec in files:
        for field in required:
            if field not in rec:
                raise ValueError(
                    "manifest record missing field {0}: id={1}".format(
                        field, rec.get("id", "<unknown>")
                    )
                )
    return files


def _resolve_text_path(text_path: str) -> Path:
    """Resolve a manifest text_path (may be relative + Windows backslashes)."""
    p = Path(text_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Chunking (whitespace-token windows over a single paper's text).
# ---------------------------------------------------------------------------


def chunk_text(text: str) -> List[Dict[str, int]]:
    """Split ascii-normalized text into 600-1000 (approx) token chunks.

    Returns a list of dicts {start, end, text} where start/end are character
    offsets into the input text (char_span). Tokens are whitespace-delimited.
    The last chunk may be shorter than MIN_TOKENS (tail remainder is kept so
    no text is dropped); it is merged into the previous chunk only if it is
    tiny and a previous chunk exists.

    A token run longer than MAX_CHARS characters is split on the char ceiling
    to honor the design's upper budget.
    """
    if not text:
        return []
    # token matches with their char spans.
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return []

    chunks: List[Dict[str, int]] = []
    cur_start_char: Optional[int] = None
    cur_end_char = 0
    cur_token_count = 0

    def _flush() -> None:
        nonlocal cur_start_char, cur_end_char, cur_token_count
        if cur_start_char is None:
            return
        chunks.append(
            {
                "start": cur_start_char,
                "end": cur_end_char,
                "text": text[cur_start_char:cur_end_char],
            }
        )
        cur_start_char = None
        cur_end_char = 0
        cur_token_count = 0

    for m in matches:
        tok_start, tok_end = m.start(), m.end()
        if cur_start_char is None:
            cur_start_char = tok_start
        cur_end_char = tok_end
        cur_token_count += 1
        char_len = cur_end_char - cur_start_char
        # Close the chunk once we reach the token target OR the char ceiling.
        if cur_token_count >= TARGET_TOKENS or char_len >= MAX_CHARS:
            _flush()

    # Flush the tail. Merge a tiny tail into the previous chunk so we do not
    # emit a sub-MIN chunk when a fuller one is available.
    if cur_start_char is not None:
        tail = {
            "start": cur_start_char,
            "end": cur_end_char,
            "text": text[cur_start_char:cur_end_char],
        }
        if chunks and cur_token_count < MIN_TOKENS:
            prev = chunks[-1]
            # Only merge if the combined char length stays within ceiling.
            if (tail["end"] - prev["start"]) <= MAX_CHARS:
                prev["end"] = tail["end"]
                prev["text"] = text[prev["start"]:prev["end"]]
            else:
                chunks.append(tail)
        else:
            chunks.append(tail)

    return chunks


# ---------------------------------------------------------------------------
# Passage building.
# ---------------------------------------------------------------------------


def build_passages_for_record(rec: dict) -> List[dict]:
    """Build passage rows for a single manifest record.

    Reads the paper text, ASCII-normalizes it, chunks it, and emits one
    passage dict per chunk. Each passage is attributed to exactly this
    record's id (single paper_id guarantee).
    """
    paper_id = rec["id"]
    domain = rec["domain_tag"]
    text_path = _resolve_text_path(rec["text_path"])
    with open(text_path, "r", encoding="utf-8") as f:
        raw = f.read()
    norm = ascii_normalize(raw)
    chunks = chunk_text(norm)
    rows: List[dict] = []
    for i, ch in enumerate(chunks):
        rows.append(
            {
                "passage_id": "{0}::p{1:05d}".format(paper_id, i),
                "paper_id": paper_id,
                "domain": domain,
                "text": ch["text"],
                "char_span": [ch["start"], ch["end"]],
            }
        )
    return rows


def build_all_passages(records: Iterable[dict]) -> List[dict]:
    """Build passage rows across all manifest records."""
    all_rows: List[dict] = []
    for rec in records:
        all_rows.extend(build_passages_for_record(rec))
    return all_rows


# ---------------------------------------------------------------------------
# Gate: every passage attributes to exactly one paper_id and is ASCII.
# ---------------------------------------------------------------------------


def assert_passage_invariants(rows: List[dict]) -> None:
    """Raise ValueError if any passage violates the S1 build gate.

    Gate (design S1): every passage has a non-empty single paper_id, text is
    pure ASCII, and char_span is a valid [start, end] pair.
    """
    for r in rows:
        pid = r.get("paper_id")
        if not pid or not isinstance(pid, str):
            raise ValueError("passage missing paper_id: {0}".format(r.get("passage_id")))
        text = r.get("text", "")
        try:
            text.encode("ascii")
        except UnicodeEncodeError as e:
            raise ValueError(
                "passage text not ASCII: {0}".format(r.get("passage_id"))
            ) from e
        span = r.get("char_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(x, int) for x in span)
            or span[0] >= span[1]
        ):
            raise ValueError(
                "passage bad char_span: {0}".format(r.get("passage_id"))
            )


# ---------------------------------------------------------------------------
# Writer.
# ---------------------------------------------------------------------------


def write_passages_jsonl(rows: List[dict], out_path: Path) -> None:
    """Write passages as one JSON object per line (ASCII-safe)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True))
            f.write("\n")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="S1 passage pool builder")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    records = load_manifest_records(Path(args.manifest))
    rows = build_all_passages(records)
    assert_passage_invariants(rows)
    write_passages_jsonl(rows, Path(args.out))
    print(
        "S1 done: {0} passages from {1} papers -> {2}".format(
            len(rows), len(records), args.out
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
