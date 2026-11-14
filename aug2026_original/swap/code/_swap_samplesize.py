# -*- coding: utf-8 -*-
# Sample-size sensitivity of the 2WikiMultiHop KG-swap paired effect.
# Post-processing of existing per-question hit@5 vectors (no new experiment).
# For prefixes N in {50,100,150,200} (first N questions in stable id order),
# report mean paired diff (silver hit5 - openie hit5) and a 10000-resample
# bootstrap 95% CI seeded deterministically with numpy default_rng(0).
import json
import numpy as np

SILVER = "swap/data/sub200/_kgswap_kg_silver.json"
OPENIE = "swap/data/sub200/_kgswap_kg_openie.json"


def load_hits(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    recs = d["results"]["proposed_v3"]["records"]
    return [(r["qa_id"], float(r["metrics"]["hit_at_k"])) for r in recs]


def main():
    silver = load_hits(SILVER)
    openie = load_hits(OPENIE)
    sids = [q for q, _ in silver]
    oids = [q for q, _ in openie]
    assert sids == oids, "id order mismatch"
    diff = np.array([s - o for (_, s), (_, o) in zip(silver, openie)], dtype=float)

    print("N\tmean_diff\tCI_low\tCI_high")
    for N in (50, 100, 150, 200):
        sub = diff[:N]
        mean = float(sub.mean())
        rng = np.random.default_rng(0)
        boot = np.empty(10000, dtype=float)
        n = len(sub)
        for b in range(10000):
            idx = rng.integers(0, n, n)
            boot[b] = sub[idx].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print("%d\t%.4f\t%.4f\t%.4f" % (N, mean, lo, hi))


if __name__ == "__main__":
    main()
