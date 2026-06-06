"""
Statistical validation for the redesigned method.

Two levels (paper, Sec. Validacion estadistica):

* **Per descriptor** -- Friedman omnibus over ``G(f,s), G(f,m), G(f,l)`` fold by
  fold (+ Nemenyi post-hoc): "do the scales differ?". For IRMAS (K=2) Friedman
  degenerates, so it is replaced by the Wilcoxon signed-rank test (two paired
  samples). NOTE: the "temporally exploitable" label is governed by the *payoff
  CI + gate* in ``tsi.py``, NOT by Friedman significance.

* **Between fusion strategies** -- paired Wilcoxon per fold over the 5 strategies
  with Bonferroni correction for ``C(5,2)=10`` comparisons (alpha_corr=0.005);
  effect size via Cliff's delta (0.147 / 0.33 / 0.474 thresholds).

* **Consistency** -- Spearman ``rho`` (bootstrap CI) between TSI *rankings* across
  datasets / classifiers (rankings, not raw G).

The legacy "Wilcoxon best-vs-worst" comparison is intentionally absent: comparing
only ``G(f,k*)`` against ``G(f,k_min)`` (argmax/argmin on the same data) is the
selection-biased test the paper explicitly rejects in favor of Friedman over all
scales.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Dict, List, Sequence, Optional

# Cliff's delta magnitude thresholds (Romano et al.)
CLIFF_SMALL, CLIFF_MEDIUM, CLIFF_LARGE = 0.147, 0.33, 0.474


# --------------------------------------------------------------------------- #
# Per-descriptor: Friedman (K>=3) / Wilcoxon (K=2) over per-fold gains
# --------------------------------------------------------------------------- #
def scale_difference_test(
    per_fold_gains: Dict[str, Sequence[float]],
    scales: Sequence[str] = ("short", "medium", "long"),
) -> Dict:
    """Test whether discriminability differs across scales, fold by fold.

    Parameters
    ----------
    per_fold_gains : dict
        ``{scale: [G per fold]}`` for one descriptor (aligned by fold).
    scales : sequence
        Scales to include (length 2 -> Wilcoxon signed-rank; >=3 -> Friedman).

    Returns
    -------
    dict with ``test`` ('friedman'|'wilcoxon'), ``statistic``, ``p_value``,
    ``significant`` (alpha=0.05) and, for Friedman, ``nemenyi`` (pairwise p-value
    matrix as nested dict).
    """
    scales = [s for s in scales if s in per_fold_gains]
    mats = [np.asarray(per_fold_gains[s], dtype=float) for s in scales]
    n_folds = len(mats[0])

    if len(scales) == 2:
        from scipy.stats import wilcoxon
        a, b = mats
        if np.allclose(a, b):
            return {"test": "wilcoxon", "scales": scales, "statistic": 0.0,
                    "p_value": 1.0, "significant": False}
        stat, p = wilcoxon(a, b)
        return {"test": "wilcoxon", "scales": scales, "statistic": float(stat),
                "p_value": float(p), "significant": bool(p < 0.05)}

    from scipy.stats import friedmanchisquare
    stat, p = friedmanchisquare(*mats)
    out = {"test": "friedman", "scales": scales, "n_folds": n_folds,
           "statistic": float(stat), "p_value": float(p),
           "significant": bool(p < 0.05)}
    if p < 0.05:
        out["nemenyi"] = _nemenyi(mats, scales)
    return out


def _nemenyi(mats: List[np.ndarray], scales: Sequence[str]) -> Dict:
    """Nemenyi post-hoc p-values from per-fold rankings across scales."""
    try:
        import scikit_posthocs as sp  # optional dependency
        data = np.column_stack(mats)
        pvals = sp.posthoc_nemenyi_friedman(data)
        return {scales[i]: {scales[j]: float(pvals.iloc[i, j])
                            for j in range(len(scales))}
                for i in range(len(scales))}
    except Exception:
        # Fallback: critical-difference via studentized range approximation.
        from scipy.stats import rankdata
        data = np.column_stack(mats)  # (folds, k)
        ranks = np.array([rankdata(row) for row in data])
        avg_rank = ranks.mean(axis=0)
        k, n = len(scales), data.shape[0]
        se = np.sqrt(k * (k + 1) / (6.0 * n))
        from scipy.stats import norm
        out = {}
        for i in range(k):
            out[scales[i]] = {}
            for j in range(k):
                if i == j:
                    out[scales[i]][scales[j]] = 1.0
                    continue
                z = abs(avg_rank[i] - avg_rank[j]) / se
                # two-sided, conservative Bonferroni-style scaling by #pairs
                p = min(1.0, 2 * (1 - norm.cdf(z)) * (k * (k - 1) / 2))
                out[scales[i]][scales[j]] = float(p)
        return out


# --------------------------------------------------------------------------- #
# Cliff's delta
# --------------------------------------------------------------------------- #
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta dominance effect size in ``[-1, 1]``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (len(a) * len(b)))


def cliffs_magnitude(delta: float) -> str:
    """Map |delta| to a magnitude label."""
    d = abs(delta)
    if d < CLIFF_SMALL:
        return "negligible"
    if d < CLIFF_MEDIUM:
        return "small"
    if d < CLIFF_LARGE:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Between fusion strategies: pairwise Wilcoxon + Bonferroni
# --------------------------------------------------------------------------- #
def run_pairwise_comparisons(
    strategy_scores: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> List[Dict]:
    """Paired Wilcoxon over per-fold scores for every strategy pair, Bonferroni.

    Parameters
    ----------
    strategy_scores : dict
        ``{strategy_name: [score per fold]}`` (aligned by fold), e.g. the 5
        strategies. ``alpha_corr = alpha / C(n,2)``.

    Returns
    -------
    list of dict, one per pair, with ``a``, ``b``, ``p_value``,
    ``alpha_corrected``, ``significant`` (after Bonferroni), ``cliffs_delta`` and
    ``magnitude``.
    """
    from scipy.stats import wilcoxon
    names = list(strategy_scores)
    pairs = list(itertools.combinations(names, 2))
    n_comp = len(pairs)
    alpha_corr = alpha / n_comp if n_comp else alpha

    results = []
    for a, b in pairs:
        sa = np.asarray(strategy_scores[a], dtype=float)
        sb = np.asarray(strategy_scores[b], dtype=float)
        if np.allclose(sa, sb):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(sa, sb)
            except ValueError:
                p = 1.0
        delta = cliffs_delta(sa, sb)
        results.append({
            "a": a, "b": b,
            "median_a": float(np.median(sa)), "median_b": float(np.median(sb)),
            "p_value": float(p),
            "alpha_corrected": float(alpha_corr),
            "significant": bool(p < alpha_corr),
            "cliffs_delta": float(delta),
            "magnitude": cliffs_magnitude(delta),
        })
    return results


# --------------------------------------------------------------------------- #
# Consistency: Spearman over TSI rankings with bootstrap CI
# --------------------------------------------------------------------------- #
def spearman_with_bootstrap_ci(
    ranking_a: Sequence[float],
    ranking_b: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict:
    """Spearman ``rho`` between two TSI score vectors (over descriptors) + CI.

    The inputs are per-descriptor TSI values (or ranks) in the SAME descriptor
    order; ``rho`` is computed on rankings. Returns ``rho``, ``ci`` and
    ``consistent`` (convention ``rho > 0.7``).
    """
    from scipy.stats import spearmanr
    a = np.asarray(ranking_a, dtype=float)
    b = np.asarray(ranking_b, dtype=float)
    rho, _ = spearmanr(a, b)
    rng = np.random.RandomState(seed)
    n = len(a)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(a[idx])) < 2 or len(np.unique(b[idx])) < 2:
            continue
        r, _ = spearmanr(a[idx], b[idx])
        if np.isfinite(r):
            boot.append(r)
    if boot:
        lo = float(np.percentile(boot, 100 * alpha / 2))
        hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    else:
        lo = hi = float(rho)
    return {"rho": float(rho), "ci": (lo, hi),
            "consistent": bool(np.isfinite(rho) and rho > 0.7)}
