"""
Feature-fusion strategies (5) for the redesigned method.

Per ``paper/proposal.tex`` (Sec. Agregacion / Validacion entre estrategias), the
five strategies compared are:

1. **single-scale**      -- best single scale, selected on INNER folds (one entry).
2. **early fusion**      -- concat of the 3 scales -> 576-d, one classifier.
3. **late fusion**       -- uniform 1/3 average of the 3 per-scale classifiers.
4. **TSI-guided scale selection** -- each descriptor at its nested k*(f) -> 192-d.
5. **TSI-weighted late fusion**   -- y = sum_k w_k p_k, weights LEARNED on inner
   folds by non-negative logistic regression (w_k >= 0, sum_k w_k = 1); the TSI is
   only an optional *prior* initializer (w_k ~ eps + sum_{f:k*(f)=k} TSI(f)).

The legacy *multiplicative TSI weighting* is intentionally absent: here the
late-fusion weights are LEARNED, with the TSI used at most to initialize them.

``k*(f)`` is taken from :func:`tsi.select_k_star` on the inner-fold gains, so the
TSI-guided representation shares the exact same optimal-scale definition as the
TSI itself (no information leakage; no second notion of k*).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Callable, Sequence

from .features import FEATURE_DIMS, extract_descriptor, TRACK_DIM
from .tsi import select_k_star, SCALE_ORDER, BASELINE
from .evaluation import task_log_loss

EPS_PRIOR = 1e-3


# --------------------------------------------------------------------------- #
# Representation builders
# --------------------------------------------------------------------------- #
def early_fusion(features: Dict[str, np.ndarray],
                 scales: Sequence[str] = tuple(SCALE_ORDER)) -> np.ndarray:
    """Concatenate per-scale 192-d vectors -> ``(n, 192*len(scales))``."""
    scales = [s for s in scales if s in features]
    return np.concatenate([features[s] for s in scales], axis=1)


def tsi_guided_features(features: Dict[str, np.ndarray],
                        k_star_map: Dict[str, str]) -> np.ndarray:
    """Each descriptor's sub-vector taken from its optimal scale -> ``(n, 192)``.

    ``k_star_map`` maps descriptor -> scale (its nested k*). Columns are placed in
    the canonical 192-d layout, so the result is a drop-in 192-d representation.
    """
    n = next(iter(features.values())).shape[0]
    out = np.zeros((n, TRACK_DIM), dtype=float)
    from .features import descriptor_slices
    cols = descriptor_slices()
    for f in FEATURE_DIMS:
        k = k_star_map[f]
        out[:, cols[f]] = extract_descriptor(features[k], f)
    return out


# --------------------------------------------------------------------------- #
# Late-fusion weight learning (non-negative, sum-to-one simplex)
# --------------------------------------------------------------------------- #
def tsi_prior_weights(tsi_by_descriptor: Dict[str, float],
                      k_star_map: Dict[str, str],
                      scales: Sequence[str] = tuple(SCALE_ORDER),
                      eps: float = EPS_PRIOR) -> np.ndarray:
    """Optional prior init: ``w_k ~ eps + sum_{f: k*(f)=k} TSI(f)`` (normalized)."""
    scales = list(scales)
    w = np.full(len(scales), eps, dtype=float)
    for i, k in enumerate(scales):
        w[i] += sum(tsi_by_descriptor.get(f, 0.0)
                    for f in FEATURE_DIMS if k_star_map.get(f) == k)
    return w / w.sum()


def learn_late_fusion_weights(
    probas: List[np.ndarray],
    y_true: np.ndarray,
    task_type: str,
    n_classes: Optional[int] = None,
    prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Learn convex-combination weights minimizing log-loss of the mixture.

    Solves ``min_w  L( sum_k w_k p_k )`` s.t. ``w_k >= 0`` and ``sum_k w_k = 1``
    (non-negative logistic-style mixture on the simplex). Falls back to the prior
    / uniform weights if SciPy is unavailable or optimization fails.

    Parameters
    ----------
    probas : list of np.ndarray
        Per-scale probability matrices on a held-out (inner-val) set, each
        ``(n, C)`` (multiclass) or ``(n, T)`` (multilabel).
    prior : np.ndarray, optional
        Initialization (e.g. from :func:`tsi_prior_weights`). Defaults to uniform.

    Returns
    -------
    np.ndarray
        Weights of length ``len(probas)`` with ``w_k >= 0`` and ``sum w_k = 1``.
    """
    K = len(probas)
    if K == 1:
        return np.ones(1)
    w0 = np.asarray(prior, dtype=float) if prior is not None else np.full(K, 1.0 / K)
    w0 = np.clip(w0, 0, None)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(K, 1.0 / K)

    P = np.stack(probas, axis=0)  # (K, n, C)

    def loss(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        w = w / s if s > 0 else np.full(K, 1.0 / K)
        mix = np.tensordot(w, P, axes=(0, 0))  # (n, C)
        return task_log_loss(y_true, mix, task_type, n_classes=n_classes)

    try:
        from scipy.optimize import minimize
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * K
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9})
        w = np.clip(res.x, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(K, 1.0 / K)
        return w
    except Exception:
        return w0


def combine_late(probas: List[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted average of per-scale probability matrices."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    P = np.stack(probas, axis=0)
    return np.tensordot(w, P, axes=(0, 0))


# --------------------------------------------------------------------------- #
# Primary metric for the fusion comparison
# --------------------------------------------------------------------------- #
def primary_metric(y_true: np.ndarray, proba: np.ndarray, task_type: str) -> float:
    """F1-macro for multiclass; mean Average Precision (mAP) for multilabel."""
    if task_type == "multilabel":
        from sklearn.metrics import average_precision_score
        y_true = np.asarray(y_true)
        # tags present in y_true are required for AP; skip degenerate tags
        aps = []
        for t in range(y_true.shape[1]):
            if y_true[:, t].sum() == 0:
                continue
            aps.append(average_precision_score(y_true[:, t], proba[:, t]))
        return float(np.mean(aps)) if aps else 0.0
    from sklearn.metrics import f1_score
    pred = np.asarray(proba).argmax(axis=1)
    return float(f1_score(y_true, pred, average="macro"))


# --------------------------------------------------------------------------- #
# Full per-fold evaluation of the 5 strategies
# --------------------------------------------------------------------------- #
def _per_scale_inner_proba(clf_factory, X_full_by_scale, y, inner_splits,
                           task_type, n_classes, scales):
    """Stacked inner-val probabilities per scale (for weight learning / selection)."""
    val_idx = np.concatenate([iv for _, iv in inner_splits])
    probas = {}
    for k in scales:
        parts = []
        for itr, iv in inner_splits:
            clf = clf_factory(input_dim=X_full_by_scale[k].shape[1],
                              n_classes=n_classes, task_type=task_type)
            clf.fit(X_full_by_scale[k][itr], y[itr])
            parts.append(clf.predict_proba(X_full_by_scale[k][iv]))
        probas[k] = np.concatenate(parts, axis=0)
    return probas, val_idx


def evaluate_fusion_cv(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    outer_folds: List,
    clf_factory: Callable,
    task_type: str,
    n_classes: int,
    fold_gains: Dict[str, List[Dict]],
    tsi_by_descriptor: Optional[Dict[str, float]] = None,
    scales: Sequence[str] = tuple(SCALE_ORDER),
    n_inner: int = 3,
    seed: int = 42,
    use_tsi_prior: bool = True,
    progress: bool = False,
    desc: Optional[str] = None,
    checkpoint_path=None,
) -> Dict[str, List[float]]:
    """Per-fold primary-metric scores for each of the 5 strategies (+ references).

    ``fold_gains`` (from :func:`tsi.compute_fold_gains`) supplies the inner-fold
    gains used to pick each descriptor's nested k* per fold -- the SAME definition
    the TSI uses. Returns ``{strategy: [score_per_fold...]}`` for:
    ``single_scale``, ``early``, ``late_uniform``, ``tsi_guided``,
    ``tsi_weighted_lf`` and the references ``learned_lf`` (no prior, upper bound)
    and ``late_uniform`` (lower bound is the same uniform LF).

    Set ``progress=True`` for a tqdm bar over outer folds (each fold trains the 5
    strategies); ``desc`` labels it. Default is silent.

    Pass ``checkpoint_path`` to make the sweep crash-safe: each fold's six scores
    are saved as soon as the fold finishes and a re-run resumes from disk (each
    fold seeds its inner splits with ``seed + fold_i``, so resumed folds match).
    """
    from .tsi import _inner_splits
    from .progress import progress_iter
    from .checkpoint import load_progress, save_progress
    scales = [s for s in scales if s in features]
    y = np.asarray(y)
    tsi_by_descriptor = tsi_by_descriptor or {}

    strategy_names = ["single_scale", "early", "late_uniform", "tsi_guided",
                      "tsi_weighted_lf", "learned_lf"]

    label = desc or "fusion"
    folds = list(enumerate(outer_folds))
    # checkpoint store: {str(fold_i): {strategy: score}}
    meta = {"kind": "evaluate_fusion_cv", "scales": scales, "n_inner": n_inner,
            "seed": seed, "n_folds": len(folds), "use_tsi_prior": use_tsi_prior,
            "n_classes": n_classes, "strategies": strategy_names}
    store = load_progress(checkpoint_path, meta)

    for fold_i, (train_idx, eval_idx) in progress_iter(
            folds, progress, desc=f"{label} | fusion [folds]", total=len(folds)):
        if str(fold_i) in store:
            continue  # fold fully cached on a previous run
        results = {k: [] for k in strategy_names}
        train_idx, eval_idx = np.asarray(train_idx), np.asarray(eval_idx)
        inner = _inner_splits(y, train_idx, n_inner, task_type, seed + fold_i)

        # per-scale full 192-d inner probabilities (selection + weight learning)
        inner_probas, val_idx = _per_scale_inner_proba(
            clf_factory, features, y, inner, task_type, n_classes, scales)
        y_val = y[val_idx]

        # --- (1) single-scale: best scale by inner primary metric ---
        inner_scores = {k: primary_metric(y_val, inner_probas[k], task_type)
                        for k in scales}
        best_scale = max(inner_scores, key=inner_scores.get)

        # per-scale classifiers trained on full outer-train, eval on held-out fold
        eval_probas = {}
        for k in scales:
            clf = clf_factory(input_dim=features[k].shape[1],
                              n_classes=n_classes, task_type=task_type)
            clf.fit(features[k][train_idx], y[train_idx])
            eval_probas[k] = clf.predict_proba(features[k][eval_idx])
        results["single_scale"].append(
            primary_metric(y[eval_idx], eval_probas[best_scale], task_type))

        # --- (2) early fusion ---
        Xe = early_fusion(features, scales)
        clf = clf_factory(input_dim=Xe.shape[1], n_classes=n_classes, task_type=task_type)
        clf.fit(Xe[train_idx], y[train_idx])
        results["early"].append(
            primary_metric(y[eval_idx], clf.predict_proba(Xe[eval_idx]), task_type))

        # --- (3) late fusion uniform ---
        uni = combine_late([eval_probas[k] for k in scales],
                           np.full(len(scales), 1.0 / len(scales)))
        results["late_uniform"].append(primary_metric(y[eval_idx], uni, task_type))

        # --- (4) TSI-guided scale selection (per-fold nested k*) ---
        k_star_map = {f: select_k_star(fold_gains[f][fold_i]["inner"])
                      for f in FEATURE_DIMS}
        Xg = tsi_guided_features(features, k_star_map)
        clf = clf_factory(input_dim=Xg.shape[1], n_classes=n_classes, task_type=task_type)
        clf.fit(Xg[train_idx], y[train_idx])
        results["tsi_guided"].append(
            primary_metric(y[eval_idx], clf.predict_proba(Xg[eval_idx]), task_type))

        # --- (5) TSI-weighted late fusion (learned weights, TSI prior init) ---
        prior = None
        if use_tsi_prior and tsi_by_descriptor:
            prior = tsi_prior_weights(tsi_by_descriptor, k_star_map, scales)
        w_tsi = learn_late_fusion_weights(
            [inner_probas[k] for k in scales], y_val, task_type,
            n_classes=n_classes, prior=prior)
        mix_tsi = combine_late([eval_probas[k] for k in scales], w_tsi)
        results["tsi_weighted_lf"].append(primary_metric(y[eval_idx], mix_tsi, task_type))

        # reference: learned weights WITHOUT prior (upper bound)
        w_ref = learn_late_fusion_weights(
            [inner_probas[k] for k in scales], y_val, task_type,
            n_classes=n_classes, prior=None)
        mix_ref = combine_late([eval_probas[k] for k in scales], w_ref)
        results["learned_lf"].append(primary_metric(y[eval_idx], mix_ref, task_type))

        store[str(fold_i)] = {k: float(results[k][0]) for k in strategy_names}
        save_progress(checkpoint_path, meta, store)

    # assemble {strategy: [score per fold]} in fold order (the public return shape)
    return {k: [store[str(i)][k] for i in range(len(folds))] for k in strategy_names}
