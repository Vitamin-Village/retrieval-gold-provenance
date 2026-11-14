# -*- coding: utf-8 -*-
"""Scatter: census circularity score (C_KG) vs published graph-over-lexical
retrieval gain. Data hand-transcribed from
paper/analysis/scatter_published_gains_2026-06-11.md (31-row master table;
sources verified there). Produces a publication-grade PNG/PDF and a JSON
stats file with Spearman correlations at two granularities:

  * row-level     : every published (graph, lexical) pair (n=30 usable rows
                    from the 31-row master table; the GRADE row has no
                    published pair and is excluded by construction)
  * benchmark-level: median pp-delta per benchmark family (primary analysis;
                    avoids over-weighting benchmarks with many rows)

Sensitivity variants reported: excluding the two dense-baseline-flagged
GraphRAG-Bench rows, and excluding EM-as-retrieval-proxy rows.

Deterministic; no network; ASCII-only output.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = Path(__file__).stem
OUT_PNG = OUTPUT_DIR / (OUTPUT_STEM + ".png")
OUT_PDF = OUTPUT_DIR / (OUTPUT_STEM + ".pdf")
OUT_JSON = OUTPUT_DIR / (OUTPUT_STEM + "_stats.json")

# (benchmark_family, C_KG, delta_pp, metric_family, flags)
# metric_family: recall | hit1 | map | em_proxy
# flags: '' | 'dense_baseline' (GraphRAG-Bench rows 20-21)
ROWS = [
    ("MuSiQue",        0.00, 10.7, "recall",   ""),
    ("MuSiQue",        0.00,  8.6, "recall",   ""),
    ("MuSiQue",        0.00, 31.2, "recall",   ""),
    ("MuSiQue",        0.00, 28.4, "em_proxy", ""),
    ("HotpotQA",       0.00,  5.5, "recall",   ""),
    ("HotpotQA",       0.00,  5.1, "recall",   ""),
    ("HotpotQA",       0.00, 21.5, "recall",   ""),
    ("MultiHop-RAG",   0.00,  0.9, "map",      ""),
    ("MultiHop-RAG",   0.00,  1.9, "recall",   ""),
    ("MultiHop-RAG",   0.00,  4.6, "recall",   ""),
    ("NaturalQuestions", 0.00, -11.7, "recall",   ""),
    ("NaturalQuestions", 0.00,  21.9, "recall",   ""),
    ("NaturalQuestions", 0.00,   4.8, "em_proxy", ""),
    ("TriviaQA",       0.00,  1.0, "em_proxy", ""),
    ("STaRK-MAG",      0.75, 32.3, "hit1",     ""),
    ("STaRK-Amazon",   0.75,  7.3, "hit1",     ""),
    ("STaRK-Prime",    0.75, 23.7, "hit1",     ""),
    ("STaRK-MAG",      0.75, -13.0, "hit1",    ""),
    ("GraphRAG-Bench", 0.75, 23.4, "recall",   "dense_baseline"),
    ("GraphRAG-Bench", 0.75, 21.1, "recall",   "dense_baseline"),
    ("2WikiMultiHopQA", 1.00, 27.2, "recall",  ""),
    ("2WikiMultiHopQA", 1.00, 18.9, "recall",  ""),
    ("2WikiMultiHopQA", 1.00, 25.1, "recall",  ""),
    ("MetaQA",         1.00, 18.9, "hit1",     ""),
    ("MetaQA",         1.00, 13.2, "hit1",     ""),
    ("MetaQA",         1.00, 58.6, "hit1",     ""),
    ("MetaQA",         1.00, 37.5, "hit1",     ""),
    ("WebQSP",         1.00, 43.3, "hit1",     ""),
    ("WebQSP",         1.00, 41.5, "hit1",     ""),
    ("WebQSP",         1.00, 23.5, "hit1",     ""),
]


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None, None
    rho = num / (dx * dy)
    # permutation p-value (exact for small n via all-pair Monte Carlo would
    # need RNG; use the standard t approximation instead -- reported as
    # approximate, two-sided)
    import math
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # two-sided p from t distribution via incomplete beta (survival approx)
    df = n - 2
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return rho, p


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b) via continued fraction."""
    import math
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    val = front * (f - 1.0)
    if x < (a + 1) / (a + b + 2):
        return min(max(val, 0.0), 1.0)
    return min(max(1.0 - val, 0.0), 1.0)


def bench_medians(rows):
    by = {}
    for name, c, d, _m, _f in rows:
        by.setdefault((name, c), []).append(d)
    out = []
    for (name, c), ds in sorted(by.items()):
        ds = sorted(ds)
        n = len(ds)
        med = ds[n // 2] if n % 2 else 0.5 * (ds[n // 2 - 1] + ds[n // 2])
        out.append((name, c, med, n))
    return out


def load_headline_scores():
    """Load benchmark-level headline C-hat in archive or development layout."""
    candidates = (
        OUTPUT_DIR / "per_axis_assignments.json",
        OUTPUT_DIR.parent / "supplementary" / "static" / "per_axis_assignments.json",
    )
    pa_path = next((path for path in candidates if path.exists()), candidates[0])
    try:
        with open(pa_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else payload.get("rows")
        if not isinstance(rows, list):
            raise TypeError("expected a list or an object with a rows list")
        aliases = {
            "Natural Questions": "NaturalQuestions",
            "2WikiMultiHop": "2WikiMultiHopQA",
        }
        headline = {}
        for row in rows:
            benchmark = aliases.get(row["benchmark"], row["benchmark"])
            headline[benchmark] = max(
                headline.get(benchmark, 0.0), float(row["census_bin"])
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            "cannot reproduce headline C-hat: invalid %s (%s)"
            % (pa_path, error)
        )

    for subbenchmark in ("STaRK-Prime", "STaRK-MAG", "STaRK-Amazon"):
        if "STaRK" not in headline:
            raise RuntimeError("cannot reproduce headline C-hat: STaRK row is missing")
        headline[subbenchmark] = headline["STaRK"]

    required = {row[0] for row in ROWS if row[4] != "dense_baseline"}
    missing = sorted(required - set(headline))
    if missing:
        raise RuntimeError(
            "cannot reproduce headline C-hat: missing benchmarks " + ", ".join(missing)
        )
    return headline


def main():
    # Headline-Chat sensitivity (paper: "Under the headline Chat ... rho=0.68"):
    # per-benchmark headline Chat = max adjudicated census bin over that
    # benchmark's axes, read from the released census/per_axis_assignments.json
    # (answer-surface axes included, unlike the KG-derived C_KG column above).
    headline = load_headline_scores()
    variants = {
        "all_rows": ROWS,
        "headline_chat_sparse_baseline": [
            (row[0], headline[row[0]], row[2], row[3], row[4])
            for row in ROWS if row[4] != "dense_baseline"
        ],
        "bm25_paired_only": [r for r in ROWS if r[4] != "dense_baseline"],
        "bm25_no_em_proxy": [r for r in ROWS
                             if r[4] != "dense_baseline"
                             and r[3] != "em_proxy"],
    }
    stats = {}
    for key, rows in variants.items():
        xs = [r[1] for r in rows]
        ys = [r[2] for r in rows]
        rho_r, p_r = spearman(xs, ys)
        meds = bench_medians(rows)
        rho_b, p_b = spearman([m[1] for m in meds], [m[2] for m in meds])
        score_name = "c_hat" if key == "headline_chat_sparse_baseline" else "c_kg"
        stats[key] = {
            "score_name": score_name,
            "n_rows": len(rows),
            "row_level": {"spearman_rho": round(rho_r, 4),
                          "p_approx_two_sided": round(p_r, 6)},
            "n_benchmarks": len(meds),
            "benchmark_median_level": {"spearman_rho": round(rho_b, 4),
                                       "p_approx_two_sided": round(p_b, 6)},
            "benchmark_medians": [
                {"benchmark": m[0], score_name: m[1], "median_delta_pp": m[2],
                 "n_pairs": m[3]} for m in meds],
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=True)
    print("wrote %s" % OUT_JSON, flush=True)
    for key, s in stats.items():
        print("%-18s rows n=%d rho=%.3f p~%.4f | bench n=%d rho=%.3f p~%.4f"
              % (key, s["n_rows"], s["row_level"]["spearman_rho"],
                 s["row_level"]["p_approx_two_sided"], s["n_benchmarks"],
                 s["benchmark_median_level"]["spearman_rho"],
                 s["benchmark_median_level"]["p_approx_two_sided"]),
              flush=True)

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed (requirements-optional.txt); "
              "stats JSON written, plots skipped")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker_by_metric = {"recall": "o", "hit1": "s", "map": "D",
                        "em_proxy": "^"}
    label_by_metric = {"recall": "recall@k", "hit1": "Hits@1 / Hit@1",
                       "map": "mAP", "em_proxy": "EM (same-reader proxy)"}
    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=200)
    import random as _r  # jitter for overlapping x; seeded, cosmetic only
    _r.seed(20260612)
    seen = set()
    for name, c, d, metric, flag in ROWS:
        x = c + _r.uniform(-0.022, 0.022)
        face = "none" if flag == "dense_baseline" else "C0"
        lab = label_by_metric[metric] if metric not in seen else None
        seen.add(metric)
        ax.scatter(x, d, marker=marker_by_metric[metric], s=26,
                   facecolors=face, edgecolors="C0", linewidths=0.9,
                   alpha=0.75, label=lab, zorder=3)
    meds = bench_medians(variants["bm25_paired_only"])
    ax.scatter([m[1] for m in meds], [m[2] for m in meds], marker="_",
               s=480, color="C3", linewidths=2.2, zorder=4,
               label="benchmark median (BM25-paired)")
    ax.axhline(0.0, color="gray", linewidth=0.7, zorder=1)
    # No title and no p-value on the figure itself: the presentation is
    # observational by design (panel constraint on n=11 inference); the
    # caption carries the descriptive rho with its caveats.
    ax.set_xlabel(r"census circularity score $\hat{C}_{\mathrm{KG}}$")
    ax.set_ylabel("published graph $-$ lexical gain (pp)")
    ax.set_xticks([0.0, 0.75, 1.0])
    ax.legend(fontsize=6.0, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    pdf_timestamp = datetime(2026, 6, 12, tzinfo=timezone.utc)
    fig.savefig(
        OUT_PDF,
        metadata={"CreationDate": pdf_timestamp, "ModDate": pdf_timestamp},
    )
    print("wrote %s" % OUT_PNG, flush=True)
    print("wrote %s" % OUT_PDF, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
