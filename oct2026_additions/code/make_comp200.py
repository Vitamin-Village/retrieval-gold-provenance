# -*- coding: utf-8 -*-
"""Build a 200-question 2Wiki split of the COMPOSITIONAL question type (comp200).

Protocol: results/twowiki_comp200/PROTOCOL.md (fixed before this split was built).

The existing splits (sub200, seed 20260530; holdout200, seed 20260531) both hold
inference-type questions only. This driver reuses the twowiki_adapter build functions
UNCHANGED, exactly as make_holdout200.py does; the only differences are the pool
(compositional rows only, minus the qa_ids of sub200 and holdout200) and the seed.
Like make_holdout200.py it enforces QA-id disjointness, not passage disjointness.

Usage (from the repo root, env new_env):
    python -m paper.eval.adapters.make_comp200 \
        --dev paper/external/2wiki/raw/data_ids/dev.json \
        --out-dir paper/external/2wiki/comp200 \
        --exclude-qa paper/external/2wiki/sub200/qa.jsonl paper/external/2wiki/holdout200/qa.jsonl \
        --id-aliases paper/external/2wiki/raw/data_ids/id_aliases.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from paper.eval.adapters.make_holdout200 import _load_exclude_qa_ids
from paper.eval.adapters.twowiki_adapter import (
    build_passages_and_manifest,
    build_qa,
    build_silver_kg,
    load_dev,
    load_id_aliases,
    select_subsample,
    _write_json,
    _write_jsonl,
)

QTYPE = "compositional"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the 2Wiki compositional split comp200.")
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-qa", type=Path, nargs="+", required=True,
                        help="qa.jsonl files whose qa_ids must NOT appear in the new split.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--id-aliases", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"{args.out_dir} is not empty; refusing to overwrite")

    exclude: set[str] = set()
    for p in args.exclude_qa:
        exclude |= _load_exclude_qa_ids(p)
    rows_all = load_dev(args.dev)
    rows_type = [r for r in rows_all if str(r.get("type") or "") == QTYPE]
    rows_pool = [r for r in rows_type if str(r.get("_id") or r.get("id") or "") not in exclude]

    rows = select_subsample(rows_pool, limit=args.limit, seed=args.seed, preferred_types=(QTYPE,))
    passages, manifest = build_passages_and_manifest(rows)
    passages_by_id = {p["passage_id"]: p["text"] for p in passages}
    qa_rows = build_qa(rows, passages_by_id)
    id_aliases = load_id_aliases(args.id_aliases)
    nodes, edges = build_silver_kg(rows, id_aliases=id_aliases)

    out_dir = args.out_dir
    kg_dir = out_dir / "kg_silver"
    _write_jsonl(out_dir / "passages.jsonl", passages)
    _write_jsonl(out_dir / "qa.jsonl", qa_rows)
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(kg_dir / "proposed_kg_nodes.json", nodes)
    _write_json(kg_dir / "proposed_kg_edges.json", edges)

    selected_ids = {q["qa_id"] for q in qa_rows}
    leak = len(selected_ids & exclude)
    comp_manifest = {
        "protocol": "results/twowiki_comp200/PROTOCOL.md",
        "source_dev": str(args.dev),
        "question_type": QTYPE,
        "excluded_from": [str(p) for p in args.exclude_qa],
        "n_excluded_qa_ids": len(exclude),
        "seed": args.seed,
        "limit": args.limit,
        "n_dev": len(rows_all),
        "n_dev_of_type": len(rows_type),
        "n_pool_after_exclude": len(rows_pool),
        "n_subsample": len(rows),
        "n_passages": len(passages),
        "n_qa": len(qa_rows),
        "qtype_counts": dict(collections.Counter(q["qtype"] for q in qa_rows)),
        "gold_per_question": dict(collections.Counter(len(q["gold_paper_ids"]) for q in qa_rows)),
        "qa_overlap_with_excluded": leak,
        "n_silver_nodes": len(nodes),
        "n_silver_nodes_with_source_docs": sum(1 for n in nodes if n["source_docs"]),
        "n_silver_edges": len(edges),
    }
    _write_json(out_dir / "comp200_manifest.json", comp_manifest)
    assert leak == 0, f"comp200 leaks {leak} qa_ids from {args.exclude_qa}"
    assert set(comp_manifest["qtype_counts"]) == {QTYPE}, comp_manifest["qtype_counts"]
    print(json.dumps(comp_manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
