"""
Feature-importance analysis -- CORROBORATIVE only.

Per ``paper/proposal.tex`` (Sec. Analisis de importancia), this produces a 7x3
*importance* matrix (descriptor x scale) from the multivariate model, distinct
from the univariate-calibrated *gain* matrix that governs the TSI. Importance
*complements and corroborates* the gain analysis; it does NOT govern the fusion
strategies.

Methods (decreasing statistical reliability):
1. **Permutation Importance (PI)** -- primary. Drop in F1-macro when each feature
   group is permuted (30 repeats), on the best classifier per task.
2. **Mean Decrease in Impurity (MDI)** -- secondary, from Random Forest.
3. **Scale-conditioned importance** -- group early-fusion (576-d) columns by their
   scale of origin and aggregate, with bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Sequence, Callable

from .features import FEATURE_DIMS, descriptor_slices, TRACK_DIM
from .tsi import SCALE_ORDER


def _f1_macro(y_true, proba, task_type):
    if task_type == "multilabel":
        from sklearn.metrics import f1_score
        pred = (np.asarray(proba) >= 0.5).astype(int)
        return float(f1_score(y_true, pred, average="macro", zero_division=0))
    from sklearn.metrics import f1_score
    pred = np.asarray(proba).argmax(axis=1)
    return float(f1_score(y_true, pred, average="macro"))


def permutation_importance_grouped(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    groups: Dict[str, np.ndarray],
    task_type: str = "multiclass",
    n_repeats: int = 30,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Group-wise permutation importance = drop in F1-macro when a group permutes.

    ``groups`` maps a name -> column indices permuted together (e.g. one
    descriptor, or one descriptor-at-one-scale). Returns per group ``{'mean':,
    'std':}`` over ``n_repeats`` permutations.
    """
    rng = np.random.RandomState(seed)
    base = _f1_macro(y, clf.predict_proba(X), task_type)
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(X.shape[0])
            Xp[:, cols] = Xp[perm][:, cols]
            drops.append(base - _f1_macro(y, clf.predict_proba(Xp), task_type))
        out[name] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return out


def mdi_by_group(feature_importances: np.ndarray,
                 groups: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Sum of MDI importances within each column group."""
    fi = np.asarray(feature_importances, dtype=float)
    return {name: float(fi[cols].sum()) for name, cols in groups.items()}


def early_fusion_groups(scales: Sequence[str] = tuple(SCALE_ORDER)
                        ) -> Dict[str, np.ndarray]:
    """Column groups for the 576-d early-fusion vector, keyed ``"descriptor@scale"``.

    Scale ``s_i`` occupies columns ``[i*192, (i+1)*192)``; within that block the
    descriptor slices follow the canonical 192-d layout.
    """
    scales = list(scales)
    base = descriptor_slices()
    groups = {}
    for i, s in enumerate(scales):
        offset = i * TRACK_DIM
        for f in FEATURE_DIMS:
            groups[f"{f}@{s}"] = base[f] + offset
    return groups


def aggregate_importance_by_descriptor_and_scale(
    importance_by_group: Dict[str, Dict[str, float]],
    scales: Sequence[str] = tuple(SCALE_ORDER),
) -> Dict[str, Dict[str, float]]:
    """Reshape ``"descriptor@scale"`` importances into a 7x3 nested matrix.

    Returns ``{descriptor: {scale: mean_importance}}``.
    """
    scales = list(scales)
    matrix = {f: {s: 0.0 for s in scales} for f in FEATURE_DIMS}
    for key, val in importance_by_group.items():
        if "@" not in key:
            continue
        f, s = key.split("@", 1)
        if f in matrix and s in matrix[f]:
            matrix[f][s] = float(val["mean"] if isinstance(val, dict) else val)
    return matrix


def bootstrap_importance_ci(
    per_fold_importance: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Bootstrap CI for an importance value aggregated across folds/partitions."""
    vals = np.asarray(per_fold_importance, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    boot = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))
