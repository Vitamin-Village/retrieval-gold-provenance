# -*- coding: utf-8 -*-
"""
Inherited-overlap analysis for the 2WikiMultiHop KG swap (W1 defense).

Claim under test: because 2Wiki passages are themselves derived from Wikidata,
a self-built OpenIE KG extracted from those passages partially inherits the
gold entity inventory (the silver KG). We quantify two inheritance levels:

  Omega_surface  fraction of silver-KG nodes whose surface forms (node_id,
                 name, aliases; normalization identical to
                 paper/eval/_swap_vocab_align.py: lowercase + whitespace
                 collapse, exact string match) also occur as a surface form
                 of at least one OpenIE node. Denominator = silver nodes.

  Omega_map      fraction of silver-KG nodes for which, additionally, at
                 least one surface-matched OpenIE node shares a source_docs
                 passage with the silver node, i.e. the self-built KG already
                 carries the gold node -> gold passage link for free.
                 Reported over all silver nodes and conditional on matched.

Question level: for each question's gold_paper_ids (gold passage titles),
  (a) surface coverage  -- title is a surface form of some OpenIE node
      (exact reproduction of swap_vocab_align.json kg_openie numbers);
  (b) mapping coverage  -- additionally some such OpenIE node lists that very
      passage in its source_docs.

Deterministic, offline, post-processing of released artifacts only.
ASCII-only output (Windows cp949 safe).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Reproduction targets from paper/results/swap_vocab_align.json (kg_openie).
TARGETS = {
    "sub200": {"mean": 0.56, "fully": 0.34},
    "holdout200": {"mean": 0.6125, "fully": 0.395},
}


def norm(s):
    """Normalization rule identical to paper/eval/_swap_vocab_align.py."""
    return " ".join(s.lower().split())


def load_nodes(kg_dir):
    return json.load(open(kg_dir / "proposed_kg_nodes.json", encoding="utf-8"))


def surfaces(node):
    """Surface-form set of a node under the shared normalization rule."""
    out = set()
    nid = node.get("node_id", "")
    if nid.startswith("n::"):
        out.add(norm(nid[3:]))
    if node.get("name"):
        out.add(norm(node["name"]))
    for a in node.get("aliases", []) or []:
        out.add(norm(a))
    out.discard("")
    return out


def openie_index(nodes):
    """surface form -> set of normalized source_docs across matching nodes."""
    idx = {}
    for n in nodes:
        docs = {norm(d) for d in n.get("source_docs", []) or []}
        for s in surfaces(n):
            idx.setdefault(s, set()).update(docs)
    return idx


def run(split):
    d = ROOT / "swap" / "data" / split
    silver = load_nodes(d / "kg_silver")
    openie = load_nodes(d / "kg_openie")
    idx = openie_index(openie)

    # ---- node-level inheritance (denominator: silver nodes) ----
    n_silver = len(silver)
    n_surface = 0
    n_map = 0
    for sn in silver:
        sf = surfaces(sn)
        hit = [s for s in sf if s in idx]
        if not hit:
            continue
        n_surface += 1
        sdocs = {norm(x) for x in sn.get("source_docs", []) or []}
        odocs = set()
        for s in hit:
            odocs |= idx[s]
        if sdocs & odocs:
            n_map += 1

    omega_surface = n_surface / n_silver
    omega_map = n_map / n_silver
    omega_map_given_surface = (n_map / n_surface) if n_surface else 0.0

    # ---- question-level inheritance ----
    qa = [json.loads(l) for l in
          open(d / "qa.jsonl", encoding="utf-8") if l.strip()]
    cov_a, cov_b = [], []
    for q in qa:
        gs = [norm(g) for g in q.get("gold_paper_ids", [])]
        if not gs:
            continue
        a = sum(1 for g in gs if g in idx)
        b = sum(1 for g in gs if g in idx and g in idx[g])
        cov_a.append(a / len(gs))
        cov_b.append(b / len(gs))

    mean_a = sum(cov_a) / len(cov_a)
    full_a = sum(1 for c in cov_a if c == 1.0) / len(cov_a)
    mean_b = sum(cov_b) / len(cov_b)
    full_b = sum(1 for c in cov_b if c == 1.0) / len(cov_b)

    t = TARGETS[split]
    repro_ok = (round(mean_a, 4) == t["mean"] and round(full_a, 4) == t["fully"])

    res = {
        "split": split,
        "n_silver_nodes": n_silver,
        "n_openie_nodes": len(openie),
        "n_questions": len(cov_a),
        "node_level": {
            "omega_surface": round(omega_surface, 4),
            "omega_map": round(omega_map, 4),
            "omega_map_given_surface": round(omega_map_given_surface, 4),
            "n_surface_matched": n_surface,
            "n_map_inherited": n_map,
        },
        "question_level": {
            "surface": {"mean_gold_title_coverage": round(mean_a, 4),
                        "frac_questions_fully_covered": round(full_a, 4)},
            "mapping": {"mean_gold_title_coverage": round(mean_b, 4),
                        "frac_questions_fully_covered": round(full_b, 4)},
        },
        "reproduction_check": {
            "target_mean": t["mean"], "target_fully": t["fully"],
            "got_mean": round(mean_a, 4), "got_fully": round(full_a, 4),
            "passed": repro_ok,
        },
    }

    print("[%s] silver=%d openie=%d questions=%d" %
          (split, n_silver, len(openie), len(cov_a)), flush=True)
    print("  Omega_surface = %.4f (%d/%d)" %
          (omega_surface, n_surface, n_silver), flush=True)
    print("  Omega_map     = %.4f (%d/%d) | given surface match: %.4f" %
          (omega_map, n_map, n_silver, omega_map_given_surface), flush=True)
    print("  Q-level surface: mean=%.4f fully=%.4f | mapping: mean=%.4f "
          "fully=%.4f" % (mean_a, full_a, mean_b, full_b), flush=True)
    print("  reproduction vs swap_vocab_align: %s (target %.4f/%.4f)" %
          ("PASS" if repro_ok else "FAIL", t["mean"], t["fully"]), flush=True)
    return res


if __name__ == "__main__":
    rows = [run("sub200"), run("holdout200")]
    out = ROOT / "swap" / "results" / "inherited_overlap.json"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, indent=2)
    print("wrote %s" % out, flush=True)
