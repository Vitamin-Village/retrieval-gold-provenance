"""Passages linked by the aligned and the independent KG of every swap split, and their gold fraction.

For each benchmark x split, a passage counts as linked by a KG if it appears in the source_docs of at
least one node of that KG (kg_silver = aligned, kg_openie = independent OpenIE KG). Gold passages are the
union of the split's gold_paper_ids. Per KG:
  linked               number of linked passages
  linked_gold          linked passages that are gold for some question of the split
  gold_frac_of_linked  linked_gold / linked (3 decimals)
  gold_covered         linked_gold / n_gold (3 decimals)
n_passages is the size of the split's passage pool (passages.jsonl, or the ids of manifest.json when the
passage text is not available). Deterministic; no retrieval or model call.

Inputs (repository layout): external/<bench>/<split>/{qa.jsonl, passages.jsonl or manifest.json,
kg_silver/proposed_kg_nodes.json, kg_openie/proposed_kg_nodes.json}; for GraphRAG-Bench the split's pool
falls back to external/graphragbench/corpus/manifest.json, then to the released chunk list
external/graphragbench/corpus_chunks_sha256.jsonl.
Output: results/kg_passage_link_counts.json (the passage link-count table of the appendix, typeset by
eval/_tab_appendix_v3.py)
Usage (from paper/): python analysis/kg_passage_link_counts.py [--check]
  --check  recompute and compare with results/kg_passage_link_counts.json; writes nothing
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external"
OUT = ROOT / "results" / "kg_passage_link_counts.json"
BENCHES = ("2wiki", "metaqa", "stark", "graphragbench")
SPLITS = ("sub200", "holdout200")


def as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return [v]
    return [v]


def passage_ids(d: Path) -> list[str]:
    if (d / "passages.jsonl").exists():
        return [json.loads(line)["passage_id"] for line in open(d / "passages.jsonl", encoding="utf-8") if line.strip()]
    for m in (d / "manifest.json", d.parent / "corpus" / "manifest.json"):
        if m.exists():
            return [r["id"] for r in json.load(open(m, encoding="utf-8"))]
    h = d.parent / "corpus_chunks_sha256.jsonl"  # released GraphRAG-Bench chunk list (ids and hashes, no text)
    if h.exists():
        return [json.loads(line)["passage_id"] for line in open(h, encoding="utf-8") if line.strip()]
    raise SystemExit(f"no passages.jsonl or manifest.json for {d}")


def counts(d: Path) -> dict:
    pool = passage_ids(d)
    gold = set()
    for line in open(d / "qa.jsonl", encoding="utf-8"):
        if line.strip():
            gold |= set(as_list(json.loads(line)["gold_paper_ids"]))
    row = {"bench": d.parent.name, "split": d.name, "n_passages": len(pool), "n_gold": len(gold)}
    for kg in ("kg_silver", "kg_openie"):
        nodes = json.load(open(d / kg / "proposed_kg_nodes.json", encoding="utf-8"))
        linked = {p for n in nodes for p in n.get("source_docs", [])}
        lg = len(linked & gold)
        row[kg] = {"linked": len(linked), "linked_gold": lg, "gold_frac_of_linked": round(lg / len(linked), 3),
                   "gold_covered": round(lg / len(gold), 3)}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the stored file; write nothing")
    args = ap.parse_args()
    rows = [counts(EXT / b / sp) for b in BENCHES for sp in SPLITS]
    if args.check:
        stored = json.load(open(OUT, encoding="utf-8"))
        if stored != rows:
            sys.exit(f"MISMATCH with {OUT.relative_to(ROOT).as_posix()}")
        print(f"{OUT.relative_to(ROOT).as_posix()}: all {len(rows)} rows reproduced")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
