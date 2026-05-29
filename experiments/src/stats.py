"""
Statistical validation utilities.

Implements:
- Wilcoxon signed-rank test with Bonferroni correction.
- Cliff's delta effect size.
- Spearman correlation with bootstrap CI.
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import wilcoxon, spearmanr


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    n_comparisons: int = 6
) -> Dict:
    """
    Wilcoxon signed-rank test with Bonferroni correction.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Paired scores (e.g., accuracy per fold or AP per tag).
    alpha : float
        Significance level before correction.
    n_comparisons : int
        Number of comparisons for Bonferroni (default 6 for pairwise
        comparisons between 4 strategies).

    Returns
    -------
    Dict with keys: statistic, p_value, p_corrected, significant, alpha_corrected.
    """
    alpha_corrected = alpha / n_comparisons

    # Handle identical arrays
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return {
            'statistic': 0,
            'p_value': 1.0,
            'p_corrected': 1.0,
            'significant': False,
            'alpha_corrected': alpha_corrected,
        }

    stat, p_value = wilcoxon(scores_a, scores_b, alternative='two-sided')

    return {
        'statistic': stat,
        'p_value': p_value,
        'p_corrected': min(p_value * n_comparisons, 1.0),
        'significant': p_value < alpha_corrected,
        'alpha_corrected': alpha_corrected,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's delta effect size.

    Thresholds (Romano et al., 2006):
    - Small: |delta| >= 0.147
    - Medium: |delta| >= 0.33
    - Large: |delta| >= 0.474

    Parameters
    ----------
    x, y : np.ndarray
        Two samples to compare.

    Returns
    -------
    Tuple of (delta_value, magnitude_label).
    """
    n_x, n_y = len(x), len(y)
    dominance = 0

    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1
            elif xi < yj:
                dominance -= 1

    delta = dominance / (n_x * n_y)

    # Classify magnitude
    abs_delta = abs(delta)
    if abs_delta >= 0.474:
        magnitude = 'large'
    elif abs_delta >= 0.33:
        magnitude = 'medium'
    elif abs_delta >= 0.147:
        magnitude = 'small'
    else:
        magnitude = 'negligible'

    return delta, magnitude


def spearman_with_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Dict:
    """
    Compute Spearman correlation with bootstrap confidence interval.

    Parameters
    ----------
    x, y : np.ndarray
        Paired values (e.g., TSI rankings from two configurations).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level.

    Returns
    -------
    Dict with keys: rho, p_value, ci_lower, ci_upper, consistent (rho > 0.7).
    """
    rho, p_value = spearmanr(x, y)

    rng = np.random.RandomState(random_state)
    boot_rhos = []
    n = len(x)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rhos.append(r)

    boot_rhos = np.array(boot_rhos)
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_rhos, alpha * 100)
    ci_upper = np.percentile(boot_rhos, (1 - alpha) * 100)

    return {
        'rho': rho,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'consistent': rho > 0.7,
    }


def tsi_bootstrap_ci(
    per_fold_info_gain: Dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict:
    """
    Bootstrap confidence interval for a descriptor's TSI by resampling folds.

    The TSI is range_k of the per-scale mean information gain. Resampling the
    folds (rows) and recomputing the range yields its sampling distribution.

    Parameters
    ----------
    per_fold_info_gain : Dict[str, np.ndarray]
        Mapping scale_name -> array of per-fold information gain G(f, k).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level.

    Returns
    -------
    Dict with keys: tsi (point estimate), tsi_std, ci_lower, ci_upper.
    """
    scales = list(per_fold_info_gain.keys())
    matrix = np.column_stack([np.asarray(per_fold_info_gain[s]) for s in scales])
    n_folds = matrix.shape[0]

    point_means = matrix.mean(axis=0)
    point_tsi = float(point_means.max() - point_means.min())

    if n_folds < 2:
        return {'tsi': point_tsi, 'tsi_std': 0.0,
                'ci_lower': point_tsi, 'ci_upper': point_tsi}

    rng = np.random.RandomState(random_state)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_folds, size=n_folds, replace=True)
        means = matrix[idx].mean(axis=0)
        boot[b] = means.max() - means.min()

    tail = (1 - ci) / 2
    return {
        'tsi': point_tsi,
        'tsi_std': float(boot.std(ddof=1)),
        'ci_lower': float(np.percentile(boot, tail * 100)),
        'ci_upper': float(np.percentile(boot, (1 - tail) * 100)),
    }


def tsi_scale_significance(
    info_gain_best: np.ndarray,
    info_gain_worst: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Test whether a descriptor's best and worst scales differ significantly.

    Paired Wilcoxon signed-rank test on the per-fold information gain of the
    best vs. worst scale. A descriptor is only "temporally sensitive" when this
    is significant -- i.e. the scale gap exceeds fold-level noise.

    Returns
    -------
    Dict with keys: statistic, p_value, significant.
    """
    best = np.asarray(info_gain_best)
    worst = np.asarray(info_gain_worst)
    diff = best - worst

    if len(diff) < 1 or np.all(diff == 0):
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}

    stat, p_value = wilcoxon(best, worst, alternative='two-sided')
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
    }


def run_pairwise_comparisons(
    results: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> List[Dict]:
    """
    Run all pairwise Wilcoxon tests between strategies.

    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Mapping strategy_name -> array of per-fold/per-tag scores.
    alpha : float
        Base significance level.

    Returns
    -------
    List of comparison results.
    """
    strategies = list(results.keys())
    n_comparisons = len(strategies) * (len(strategies) - 1) // 2
    comparisons = []

    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            name_a = strategies[i]
            name_b = strategies[j]

            test_result = wilcoxon_test(
                results[name_a], results[name_b],
                alpha=alpha, n_comparisons=n_comparisons
            )

            delta, magnitude = cliffs_delta(results[name_a], results[name_b])

            comparisons.append({
                'strategy_a': name_a,
                'strategy_b': name_b,
                **test_result,
                'cliffs_delta': delta,
                'effect_magnitude': magnitude,
            })

    return comparisons
