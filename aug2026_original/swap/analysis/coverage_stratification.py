# -*- coding: utf-8 -*-
"""
Coverage-stratified per-question analysis of the 2WikiMultiHop KG-swap under
the independent OpenIE KG. For each split and each graph retriever we compute
the per-question hit@5 delta (graph - BM25) and stratify questions by their
gold-title coverage in the OpenIE node inventory (full / partial / none).
Gold-title <-> node matching reuses the exact rule of
paper/eval/_swap_vocab_align.py (exact match on normalized node id, name, or
alias); the script first re-derives that script's aggregate coverage numbers
and aborts if they do not reproduce. A 10,000-resample bootstrap (fixed seed
20260612) gives a 95% CI for the difference in mean paired delta between the
fully-covered stratum and the rest. Win-flip questions (graph beats BM25) are
counted per stratum. Post-processing of released artifacts only; deterministic;
no network. ASCII-only output (cp949 console).
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SEED = 20260612
N_BOOT = 10000

SYSTEMS = [
    ("ppr", "_kgswap_ppr_kg_openie.json", "ppr"),
    ("proposed_v3", "_kgswap_kg_openie.json", "proposed_v3"),
    ("proposed_v5_graphcentric",
     "_kgswap_proposed_v5_graphcentric_kg_openie.json",
     "proposed_v5_graphcentric"),
]

SPLITS = [
    ("sub200", "swap/data/sub200",
     {"mean_gold_title_coverage": 0.56, "frac_questions_fully_covered": 0.34}),
    ("holdout200", "swap/data/holdout200",
     {"mean_gold_title_coverage": 0.6125,
      "frac_questions_fully_covered": 0.395}),
]


# ---- matching rule: identical to paper/eval/_swap_vocab_align.py ----

def norm(s):
    return " ".join(s.lower().split())


def vocab(kg_dir):
    nodes = json.load(open(kg_dir / "proposed_kg_nodes.json",
                           encoding="utf-8"))
    v = set()
    for n in nodes:
        nid = n.get("node_id", "")
        if nid.startswith("n::"):
            v.add(norm(nid[3:]))
        if n.get("name"):
            v.add(norm(n["name"]))
        for a in n.get("aliases", []) or []:
            v.add(norm(a))
    return v


# ---- helpers ----

def load_records(path, sys_key):
    p = json.load(open(path, encoding="utf-8"))
    g = {r["qa_id"]: r for r in p["results"][sys_key]["records"]}
    b = {r["qa_id"]: r for r in p["results"]["bm25"]["records"]}
    assert set(g) == set(b), "qa_id mismatch between %s and bm25" % sys_key
    return g, b


def coverage_map(split_dir, v):
    """qa_id -> (coverage fraction, stratum). Gold lists from any eval file."""
    p = json.load(open(split_dir / "_kgswap_kg_silver.json", encoding="utf-8"))
    recs = p["results"]["proposed_v3"]["records"]
    out = {}
    for r in recs:
        gs = [norm(g) for g in r["gold_paper_ids"]]
        if not gs:
            continue
        cov = sum(1 for g in gs if g in v) / len(gs)
        stratum = "full" if cov == 1.0 else ("none" if cov == 0.0
                                             else "partial")
        out[r["qa_id"]] = (cov, stratum)
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_diff_ci(full_d, rest_d, rng):
    """95% percentile CI for mean(full deltas) - mean(rest deltas)."""
    diffs = []
    nf, nr = len(full_d), len(rest_d)
    for _ in range(N_BOOT):
        bf = [full_d[rng.randrange(nf)] for _ in range(nf)]
        br = [rest_d[rng.randrange(nr)] for _ in range(nr)]
        diffs.append(mean(bf) - mean(br))
    diffs.sort()
    lo = diffs[int(0.025 * N_BOOT)]
    hi = diffs[int(0.975 * N_BOOT) - 1]
    return lo, hi


def main():
    results = []
    for split_name, rel, expected in SPLITS:
        d = ROOT / rel
        v = vocab(d / "kg_openie")
        covmap = coverage_map(d, v)

        # --- reproduce _swap_vocab_align.py aggregates (validation gate) ---
        covs = [c for c, _ in covmap.values()]
        mean_cov = round(mean(covs), 4)
        frac_full = round(sum(1 for c in covs if c == 1.0) / len(covs), 4)
        exp_cov = expected["mean_gold_title_coverage"]
        exp_full = expected["frac_questions_fully_covered"]
        print("[%s] reproduce alignment: mean_cov=%.4f (exp %.4f), "
              "frac_full=%.4f (exp %.4f)"
              % (split_name, mean_cov, exp_cov, frac_full, exp_full),
              flush=True)
        assert abs(mean_cov - exp_cov) < 1e-9, "mean coverage mismatch"
        assert abs(frac_full - exp_full) < 1e-9, "fully-covered mismatch"

        split_out = {"split": split_name, "n_questions": len(covmap),
                     "validation": {"mean_gold_title_coverage": mean_cov,
                                    "frac_questions_fully_covered": frac_full},
                     "systems": {}}

        for sys_name, fname, sys_key in SYSTEMS:
            g, b = load_records(d / fname, sys_key)
            strata = {"full": [], "partial": [], "none": []}
            for qid, (_, st) in covmap.items():
                dg = g[qid]["metrics"]["hit_at_k"]
                db = b[qid]["metrics"]["hit_at_k"]
                strata[st].append((dg, db))

            sys_out = {"strata": {}, "win_flip": {}}
            for st in ("full", "partial", "none"):
                pairs = strata[st]
                deltas = [a - c for a, c in pairs]
                wins = sum(1 for x in deltas if x > 0)
                sys_out["strata"][st] = {
                    "n": len(pairs),
                    "graph_hit_at_5": round(mean([a for a, _ in pairs]), 4)
                    if pairs else None,
                    "bm25_hit_at_5": round(mean([c for _, c in pairs]), 4)
                    if pairs else None,
                    "paired_delta": round(mean(deltas), 4) if pairs else None,
                    "n_win_flip": wins,
                }
                sys_out["win_flip"][st] = wins

            full_d = [a - c for a, c in strata["full"]]
            rest_d = [a - c for a, c in strata["partial"] + strata["none"]]
            rng = random.Random(SEED)
            lo, hi = bootstrap_diff_ci(full_d, rest_d, rng)
            obs = mean(full_d) - mean(rest_d)
            sys_out["full_vs_rest"] = {
                "n_full": len(full_d), "n_rest": len(rest_d),
                "delta_full": round(mean(full_d), 4),
                "delta_rest": round(mean(rest_d), 4),
                "diff": round(obs, 4),
                "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
                "n_boot": N_BOOT, "seed": SEED,
            }
            split_out["systems"][sys_name] = sys_out

            print("[%s][%s] full n=%d d=%+.4f | partial n=%d d=%+.4f | "
                  "none n=%d d=%+.4f | full-rest diff=%+.4f "
                  "CI95=[%+.4f, %+.4f] | win-flips f/p/n = %d/%d/%d"
                  % (split_name, sys_name,
                     sys_out["strata"]["full"]["n"],
                     sys_out["strata"]["full"]["paired_delta"] or 0,
                     sys_out["strata"]["partial"]["n"],
                     sys_out["strata"]["partial"]["paired_delta"] or 0,
                     sys_out["strata"]["none"]["n"],
                     sys_out["strata"]["none"]["paired_delta"] or 0,
                     obs, lo, hi,
                     sys_out["win_flip"]["full"],
                     sys_out["win_flip"]["partial"],
                     sys_out["win_flip"]["none"]), flush=True)

        results.append(split_out)

    out_path = ROOT / "swap" / "results" / "coverage_stratification.json"
    payload = {
        "description": "Coverage-stratified per-question hit@5 delta "
                       "(graph - BM25) under the independent OpenIE "
                       "KG; matching rule identical to _swap_vocab_align.py.",
        "metric": "hit_at_k (hit@5)",
        "seed": SEED,
        "n_boot": N_BOOT,
        "splits": results,
    }
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
