# -*- coding: utf-8 -*-
"""
Census inter-rater reliability over the broadened set (11 external + 2 anchors).
Each row = (benchmark, axis, raterA_bin, raterB_bin). Computes linear-weighted
Cohen kappa + ordinal Krippendorff alpha + bootstrap 95% CI for the weighted kappa.
ASCII-only output (Windows cp949 safe).
"""
import random

random.seed(20260605)  # deterministic; Date/random pinned for reproducibility
CATS = (0, 1, 2, 3, 4)

# (label, raterA, raterB)  -- supporting-gold axis rows scored by BOTH raters
ROWS = [
    ("MuSiQue/passage", 0, 0), ("MuSiQue/answer", 0, 0),
    ("HotpotQA/sentence", 0, 0), ("HotpotQA/answer", 0, 0),
    ("2Wiki/edge", 4, 4), ("2Wiki/sentence", 3, 3), ("2Wiki/answer", 4, 3),
    ("GraphRAG-Bench/subgraph", 3, 3), ("GraphRAG-Bench/answer", 3, 3),
    ("GRADE/answer-edge", 3, 3), ("GRADE/passage", 1, 1),
    ("qa_v4/edge", 4, 4), ("qa_v4/entity", 4, 4), ("qa_v4/path", 4, 4), ("qa_v4/answer", 4, 3),
    ("qa_indep/passage", 0, 0), ("qa_indep/answer", 0, 0),
    ("MultiHop-RAG/evidence", 0, 0), ("MultiHop-RAG/answer", 0, 0),
    ("STaRK/answer-entity", 3, 3), ("STaRK/textual", 1, 1),
    ("MetaQA/answer", 4, 4), ("MetaQA/path", 4, 4),
    # --- broadening (3 new benchmarks, 6 axis rows) ---
    ("NQ/passage", 1, 1), ("NQ/answer", 0, 1),
    ("TriviaQA/evidence", 4, 3), ("TriviaQA/answer", 0, 0),
    ("WebQSP/answer-entity", 4, 4), ("WebQSP/path", 4, 4),
]


def weighted_kappa(pairs, weight="quadratic"):
    k = len(CATS); idx = {c: i for i, c in enumerate(CATS)}; n = len(pairs)
    if n == 0:
        return float("nan")
    O = [[0] * k for _ in range(k)]; r1 = [0] * k; r2 = [0] * k
    for a, b in pairs:
        O[idx[a]][idx[b]] += 1; r1[idx[a]] += 1; r2[idx[b]] += 1
    maxd = k - 1; num = den = 0.0
    for i in range(k):
        for j in range(k):
            d = abs(i - j) / maxd
            w = d * d if weight == "quadratic" else d
            num += w * O[i][j]; den += w * (r1[i] * r2[j] / n)
    return 1.0 if den == 0 else 1 - num / den


def linear_weighted_kappa(pairs):
    return weighted_kappa(pairs, "linear")


def krippendorff_alpha_ordinal(units):
    k = len(CATS); idx = {c: i for i, c in enumerate(CATS)}
    o = [[0.0] * k for _ in range(k)]
    for u in units:
        vals = [v for v in u if v is not None]; m = len(vals)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[vals[a]]][idx[vals[b]]] += 1.0 / (m - 1)
    n_c = [sum(o[i]) for i in range(k)]; n = sum(n_c)
    if n == 0:
        return float("nan")

    def d2(i, j):
        lo, hi = (i, j) if i <= j else (j, i)
        s = sum(n_c[g] for g in range(lo, hi + 1)) - (n_c[i] + n_c[j]) / 2.0
        return s * s
    do = de = 0.0
    for i in range(k):
        for j in range(k):
            do += o[i][j] * d2(i, j); de += n_c[i] * n_c[j] * d2(i, j)
    return 1.0 if de == 0 else 1 - (n - 1) * do / de


def summarize(rows):
    pairs = [(a, b) for _, a, b in rows]
    units = [[a, b] for _, a, b in rows]
    n = len(pairs)
    exact = sum(1 for a, b in pairs if a == b)
    off1 = sum(1 for a, b in pairs if abs(a - b) == 1)
    off2 = sum(1 for a, b in pairs if abs(a - b) >= 2)
    kap_q = weighted_kappa(pairs, "quadratic")
    kap_l = weighted_kappa(pairs, "linear")
    alpha = krippendorff_alpha_ordinal(units)
    # bootstrap CI for quadratic-weighted kappa
    B = 10000; boot = []
    for _ in range(B):
        samp = [pairs[random.randrange(n)] for _ in range(n)]
        boot.append(weighted_kappa(samp, "quadratic"))
    boot.sort()
    lo = boot[int(0.025 * B)]; hi = boot[int(0.975 * B)]
    print("n axis-level judgments        : %d" % n)
    print("exact agreement              : %d of %d (%.1f%%)" % (exact, n, 100.0 * exact / n))
    print("off-by-one                   : %d" % off1)
    print("|diff|>=2                    : %d" % off2)
    print("quadratic-weighted Cohen k   : %.4f  (95%% CI [%.3f, %.3f])" % (kap_q, lo, hi))
    print("linear-weighted Cohen kappa  : %.4f" % kap_l)
    print("ordinal Krippendorff alpha   : %.4f" % alpha)
    disag = [(l, a, b) for (l, a, b) in rows if a != b]
    print("disagreements (%d):" % len(disag))
    for l, a, b in disag:
        print("   %-26s A=bin%d B=bin%d" % (l, a, b))


if __name__ == "__main__":
    print("=== BROADENED census (n=29: 11 external benchmarks + 2 anchors) ===")
    summarize(ROWS)
    print()
    # Manuscript's restricted figure: the eleven external benchmarks only,
    # i.e. the 29 judgments minus the 6 anchor rows (qa_v4 x4 + qa_indep x2).
    print("=== Eleven external benchmarks only (n=23: 29 minus 6 anchor rows) ===")
    ext = [row for row in ROWS
           if not (row[0].startswith("qa_v4") or row[0].startswith("qa_indep"))]
    summarize(ext)
    print()
    # Original 8-benchmark census (pre-broadening); includes the 2 anchors.
    # Distinct from the external-only n=23 above despite the same size.
    print("=== ORIGINAL 8-benchmark census (n=23: pre-broadening, incl. 2 anchors) ===")
    summarize(ROWS[:23])
