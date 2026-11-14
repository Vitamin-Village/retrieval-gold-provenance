"""Statistical testing utilities for paired retrieval-eval comparisons.

Per MASTER_PLAN §5.3: bootstrap CI, Wilcoxon signed-rank, Holm-Bonferroni
correction, Cohen's d effect size, and power analysis.

All implementations use numpy/scipy only (no statsmodels).
Bootstrap n=10000 follows the standard IR evaluation convention (Efron &
Tibshirani, 1993).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the paired mean difference b - a.

    Args:
        a: Baseline per-sample scores.
        b: Candidate per-sample scores.  Must be same length as *a*.
        n_boot: Number of bootstrap resamples (default 10000 — IR convention).
        ci: Nominal coverage (default 0.95).
        seed: RNG seed for reproducibility.

    Returns:
        (mean_delta, ci_low, ci_high) where mean_delta = mean(b) - mean(a).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape or a_arr.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length")
    if len(a_arr) == 0:
        raise ValueError("a and b must not be empty")

    rng = np.random.default_rng(seed)
    diffs = b_arr - a_arr
    mean_delta = float(np.mean(diffs))

    # Percentile bootstrap on the paired differences
    indices = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_means, 100.0 * alpha))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_delta, ci_low, ci_high


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples.

    Args:
        a: Baseline per-sample scores.
        b: Candidate per-sample scores.  Must be same length as *a*.

    Returns:
        (statistic, p_value) two-sided.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) != len(b_arr):
        raise ValueError("a and b must have the same length")
    if len(a_arr) == 0:
        raise ValueError("a and b must not be empty")

    result = _scipy_stats.wilcoxon(b_arr - a_arr, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def holm_bonferroni(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni step-down multiple-comparison correction.

    Args:
        p_values: Mapping from label to raw p-value.
        alpha: Family-wise error rate (default 0.05).

    Returns:
        Mapping from label to (adjusted_p, significant) where adjusted_p is
        the Holm-corrected p-value and significant is True when it is < alpha.
    """
    if not p_values:
        return {}

    labels = list(p_values.keys())
    raw = [p_values[label] for label in labels]
    m = len(raw)

    # Sort ascending by raw p-value
    order = sorted(range(m), key=lambda i: raw[i])
    adjusted: list[float] = [0.0] * m

    # Holm step-down: adjusted_p[k] = max(p[0]*m, p[1]*(m-1), ..., p[k]*(m-k))
    running_max = 0.0
    for step, idx in enumerate(order):
        corrected = raw[idx] * (m - step)
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)

    return {
        labels[i]: (adjusted[i], adjusted[i] < alpha)
        for i in range(m)
    }


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d effect size: mean(d) / std(d) where d = b - a.

    Returns 0.0 when all differences are identical (zero variance).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) != len(b_arr):
        raise ValueError("a and b must have the same length")
    if len(a_arr) == 0:
        raise ValueError("a and b must not be empty")

    diffs = b_arr - a_arr
    std_d = float(np.std(diffs, ddof=1))
    if std_d == 0.0:
        return 0.0
    return float(np.mean(diffs) / std_d)


def power_analysis(n: int, effect: float, alpha: float = 0.05) -> float:
    """Approximate power for a paired two-sided t-test.

    Uses the non-central t-distribution: power = P(|T| > t_crit | ncp).

    Args:
        n: Number of paired observations.
        effect: Expected Cohen's d effect size.
        alpha: Significance level (default 0.05).

    Returns:
        Estimated power in [0, 1].
    """
    if n <= 1:
        raise ValueError("n must be > 1")
    df = n - 1
    ncp = effect * (n ** 0.5)
    t_crit = _scipy_stats.t.ppf(1.0 - alpha / 2.0, df)
    # P(T > t_crit) + P(T < -t_crit) under non-central t
    power = (
        1.0 - _scipy_stats.nct.cdf(t_crit, df, ncp)
        + _scipy_stats.nct.cdf(-t_crit, df, ncp)
    )
    return float(np.clip(power, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Hero table renderer
# ---------------------------------------------------------------------------


def hero_table(rows: list[dict]) -> str:
    """Render a Markdown comparison table per MASTER_PLAN §5.3.

    Each row dict must contain:
        metric, mode_a, mode_b, delta, ci_low, ci_high,
        p_wilcoxon, p_holm, significant, cohens_d

    Returns a Markdown table string with columns:
        Metric | ModeA | ModeB | Delta | 95% CI | p(Wilcoxon) | p(Holm) | Cohen's d
    """
    header = (
        "| Metric | ModeA | ModeB | Δ | 95% CI | p(Wilcoxon) | p(Holm) | Cohen's d |"
    )
    separator = "|---|---|---|---|---|---|---|---|"
    lines = [header, separator]
    for row in rows:
        sig_marker = "*" if row.get("significant") else ""
        ci_str = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        lines.append(
            f"| {row['metric']} "
            f"| {row['mode_a']:.4f} "
            f"| {row['mode_b']:.4f} "
            f"| {row['delta']:+.4f} "
            f"| {ci_str} "
            f"| {row['p_wilcoxon']:.4f} "
            f"| {row['p_holm']:.4f}{sig_marker} "
            f"| {row['cohens_d']:.3f} |"
        )
    return "\n".join(lines)


__all__ = [
    "paired_bootstrap_ci",
    "wilcoxon_signed_rank",
    "holm_bonferroni",
    "cohens_d_paired",
    "power_analysis",
    "hero_table",
]
