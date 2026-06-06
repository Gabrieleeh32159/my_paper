"""
Evaluation primitives for the redesigned TSI method.

Implements the *normalized information gain* that underlies the Temporal
Sensitivity Index (TSI), as specified in ``paper/proposal.tex`` (Sec. TSI):

    G(f, k) = 1 - L(f, k) / L_chance

where

* ``L(f, k)`` is the log-loss of a *calibrated* classifier trained on descriptor
  ``f`` alone at scale ``k``, and
* ``L_chance`` is the log-loss of the **train base-rate** predictor, evaluated on
  the *same* evaluation fold as ``L(f, k)``.

Multiclass: base rate = train class frequencies ``p_c``.
Multilabel (MTAT): base rate = per-tag train prevalence ``pi_t``; the chance
log-loss is the mean over tags of the binary cross-entropy of ``pi_t``.

This module also provides the Expected Calibration Error (ECE) and reliability
curve data, since calibration quality governs whether ``G`` is interpretable as
information (the paper requires reporting ECE + reliability diagrams).

NOTE: ``G in (-inf, 1]``. The floor ``0`` is an *interpretive* convention
("no useful information"), not a guaranteed property; imperfect calibration can
yield ``L > L_chance`` and hence ``G < 0``. Use :func:`truncate_gain` for the
reporting convention; selection / CI machinery operates on the raw ``G``.
"""

from __future__ import annotations

import warnings

import numpy as np
from typing import Dict, Optional

# Probability clip to keep log-loss finite (mirrors sklearn's eps handling).
EPS = 1e-15
# Prevalence clip for the multilabel base rate: guards against tags with 0 or 1
# prevalence in a given train fold (which would make the chance log-loss diverge).
PREVALENCE_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Log-loss
# --------------------------------------------------------------------------- #
def log_loss_multiclass(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: Optional[int] = None,
) -> float:
    """Mean per-sample cross-entropy for a multiclass problem.

    Parameters
    ----------
    y_true : np.ndarray
        Integer class labels, shape ``(n,)``.
    proba : np.ndarray
        Predicted class probabilities, shape ``(n, C)`` (rows sum to 1).
    n_classes : int, optional
        Total number of classes ``C``. Defaults to ``proba.shape[1]``.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be 2-D (n_samples, n_classes)")
    C = n_classes if n_classes is not None else proba.shape[1]
    proba = np.clip(proba, EPS, 1.0)
    # renormalize after clipping
    proba = proba / proba.sum(axis=1, keepdims=True)
    idx = np.arange(len(y_true))
    p_true = proba[idx, y_true]
    return float(-np.mean(np.log(p_true)))


def log_loss_multilabel(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean binary cross-entropy over tags and samples (multilabel).

    Parameters
    ----------
    y_true : np.ndarray
        Binary tag matrix, shape ``(n, T)``.
    proba : np.ndarray
        Predicted per-tag probabilities, shape ``(n, T)``.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0 - EPS)
    bce = -(y_true * np.log(proba) + (1.0 - y_true) * np.log(1.0 - proba))
    return float(np.mean(bce))


def task_log_loss(y_true, proba, task_type: str, n_classes: Optional[int] = None) -> float:
    """Dispatch log-loss by task type."""
    if task_type == "multilabel":
        return log_loss_multilabel(y_true, proba)
    return log_loss_multiclass(y_true, proba, n_classes=n_classes)


# --------------------------------------------------------------------------- #
# Base-rate (chance) predictor and its log-loss
# --------------------------------------------------------------------------- #
def base_rate(y_train: np.ndarray, task_type: str, n_classes: Optional[int] = None) -> np.ndarray:
    """Constant base-rate probability vector estimated on the training fold.

    Returns
    -------
    np.ndarray
        Multiclass: class-frequency vector ``p`` of length ``C``.
        Multilabel: per-tag prevalence vector ``pi`` of length ``T``.
    """
    if task_type == "multilabel":
        y_train = np.asarray(y_train, dtype=float)
        # clip prevalences off 0/1 so a tag absent (or saturated) in this train
        # fold yields a finite chance log-loss (paper: H(pi_t) base rate per tag).
        return np.clip(y_train.mean(axis=0), PREVALENCE_EPS, 1.0 - PREVALENCE_EPS)
    y_train = np.asarray(y_train).astype(int)
    C = n_classes if n_classes is not None else int(y_train.max()) + 1
    counts = np.bincount(y_train, minlength=C).astype(float)
    return counts / counts.sum()


def chance_log_loss(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    task_type: str,
    n_classes: Optional[int] = None,
) -> float:
    """Log-loss of the train base-rate predictor, evaluated on the eval fold.

    This is ``L_chance`` in ``G = 1 - L / L_chance``. Evaluating it on the same
    eval fold as ``L`` (rather than using the closed form ``-sum p_c log p_c``)
    keeps ``G`` correctly bounded when the eval distribution differs slightly
    from training (paper, Eq. L_chance).
    """
    if task_type == "multilabel":
        pi = base_rate(y_train, task_type)
        n = np.asarray(y_eval).shape[0]
        proba = np.tile(pi, (n, 1))
        return log_loss_multilabel(y_eval, proba)
    C = n_classes
    if C is None:
        C = int(max(np.max(y_train), np.max(y_eval))) + 1
    p = base_rate(y_train, task_type, n_classes=C)
    n = np.asarray(y_eval).shape[0]
    proba = np.tile(p, (n, 1))
    return log_loss_multiclass(y_eval, proba, n_classes=C)


# --------------------------------------------------------------------------- #
# Normalized information gain
# --------------------------------------------------------------------------- #
def information_gain(loss: float, loss_chance: float) -> float:
    """Raw normalized information gain ``G = 1 - L / L_chance`` (in ``(-inf, 1]``).

    Not truncated; ``G`` can be negative under imperfect calibration.
    """
    if loss_chance <= 0:
        # Degenerate task (e.g. a single class / zero-entropy base rate): G is
        # undefined. We fall back to 0 but warn rather than mask it silently.
        warnings.warn(
            "information_gain: L_chance <= 0 (degenerate base rate); "
            "returning G=0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    return 1.0 - loss / loss_chance


def truncate_gain(g: float) -> float:
    """Reporting convention: floor negative gains at 0 ("no useful information")."""
    return float(max(g, 0.0))


def gain_from_predictions(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    proba_eval: np.ndarray,
    task_type: str,
    n_classes: Optional[int] = None,
    truncate: bool = False,
) -> float:
    """End-to-end ``G`` from predicted probabilities on an eval fold.

    Computes ``L`` (model) and ``L_chance`` (train base rate) on the same
    ``y_eval`` and returns ``G``. With ``truncate=True`` applies the floor-at-0
    reporting convention.
    """
    L = task_log_loss(y_eval, proba_eval, task_type, n_classes=n_classes)
    L_chance = chance_log_loss(y_train, y_eval, task_type, n_classes=n_classes)
    g = information_gain(L, L_chance)
    return truncate_gain(g) if truncate else g


# --------------------------------------------------------------------------- #
# Calibration: Expected Calibration Error + reliability curve
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    task_type: str = "multiclass",
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    Multiclass: confidence = max predicted probability; correctness = top-1 hit.
    Multilabel: mean ECE over tags (per-tag binary confidence/accuracy).
    """
    if task_type == "multilabel":
        y_true = np.asarray(y_true)
        proba = np.asarray(proba, dtype=float)
        eces = [
            _binary_ece(y_true[:, t], proba[:, t], n_bins)
            for t in range(proba.shape[1])
        ]
        return float(np.mean(eces))

    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    return _reliability_ece(conf, correct, n_bins)


def _binary_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """ECE for a single binary tag (confidence = predicted prob of positive)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    # For a binary tag, confidence in the predicted label is max(p, 1-p).
    pred = (p >= 0.5).astype(float)
    conf = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y_true).astype(float)
    return _reliability_ece(conf, correct, n_bins)


def _reliability_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    task_type: str = "multiclass",
    n_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """Reliability-diagram data (bin confidence vs. empirical accuracy).

    Returns a dict with arrays ``bin_confidence``, ``bin_accuracy`` and
    ``bin_count`` (length ``n_bins``; empty bins are NaN). For multilabel the
    confidence/accuracy are pooled across tags.
    """
    if task_type == "multilabel":
        proba = np.asarray(proba, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        pred = (proba >= 0.5).astype(float)
        conf = np.where(pred == 1, proba, 1.0 - proba).ravel()
        correct = (pred == y_true).astype(float).ravel()
    else:
        proba = np.asarray(proba, dtype=float)
        y_true = np.asarray(y_true).astype(int)
        conf = proba.max(axis=1)
        correct = (proba.argmax(axis=1) == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_cnt = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        bin_cnt[i] = mask.sum()
        if np.any(mask):
            bin_conf[i] = conf[mask].mean()
            bin_acc[i] = correct[mask].mean()
    return {
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_count": bin_cnt,
        "bin_edges": bins,
    }


def calibration_report(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    outer_folds,
    clf_factory,
    task_type: str,
    n_classes: int,
    scale: str = "medium",
    n_bins: int = 15,
) -> Dict:
    """ECE averaged over outer folds + a pooled reliability curve for one classifier.

    For each outer fold a classifier is trained on the given ``scale`` and scored
    on the held-out fold; the per-fold ECE is averaged and the eval predictions are
    pooled into a single reliability diagram. Returns ``ece_mean``, ``ece_std``,
    ``ece_per_fold`` and ``reliability`` (JSON-friendly lists).
    """
    y = np.asarray(y)
    eces, pooled_proba, pooled_y = [], [], []
    for train_idx, eval_idx in outer_folds:
        train_idx, eval_idx = np.asarray(train_idx), np.asarray(eval_idx)
        clf = clf_factory(input_dim=features[scale].shape[1],
                          n_classes=n_classes, task_type=task_type)
        clf.fit(features[scale][train_idx], y[train_idx])
        proba = clf.predict_proba(features[scale][eval_idx])
        eces.append(expected_calibration_error(y[eval_idx], proba, task_type, n_bins))
        pooled_proba.append(np.asarray(proba))
        pooled_y.append(y[eval_idx])
    proba_all = np.concatenate(pooled_proba, axis=0)
    y_all = np.concatenate(pooled_y, axis=0)
    rc = reliability_curve(y_all, proba_all, task_type, n_bins)
    return {
        "scale": scale,
        "ece_mean": float(np.mean(eces)),
        "ece_std": float(np.std(eces)),
        "ece_per_fold": [float(e) for e in eces],
        "reliability": {
            "bin_confidence": [None if np.isnan(v) else float(v) for v in rc["bin_confidence"]],
            "bin_accuracy": [None if np.isnan(v) else float(v) for v in rc["bin_accuracy"]],
            "bin_count": [int(c) for c in rc["bin_count"]],
        },
    }
