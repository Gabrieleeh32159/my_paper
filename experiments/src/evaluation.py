"""
Evaluation metrics for TSI experiments.

Genre classification: accuracy, F1 macro, F1 per class.
Auto-tagging (MTAT): mAP, ROC-AUC macro, AP per tag.
Instruments (IRMAS): accuracy, F1 macro.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score,
    average_precision_score, roc_auc_score
)


# ---------------------------------------------------------------------------
# Log-loss / information-gain utilities (used by the TSI computation).
#
# The TSI is built on the normalized information gain of a descriptor at a
# given temporal scale:
#
#     G(f, k) = 1 - L_model(f, k) / L_chance
#
# where L_model is the (clipped) log-loss of the calibrated classifier trained
# on descriptor f at scale k, and L_chance is the log-loss of the constant
# class-prior predictor (the best "no-feature" model). G is the fraction of the
# task's label uncertainty that the descriptor resolves; it is 0 at chance and
# 1 for a perfect classifier, and is comparable across tasks with different
# numbers of classes / different class balance.
# ---------------------------------------------------------------------------

def class_prior(y_train: np.ndarray, n_classes: int, eps: float = 1e-12) -> np.ndarray:
    """Empirical class-prior distribution over [0, n_classes) from training labels."""
    counts = np.bincount(np.asarray(y_train).astype(int), minlength=n_classes).astype(float)
    prior = counts / counts.sum()
    prior = np.clip(prior, eps, None)
    return prior / prior.sum()


def tag_prevalence(y_train: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-tag positive prevalence (multilabel chance baseline) from training labels."""
    prev = np.asarray(y_train, dtype=float).mean(axis=0)
    return np.clip(prev, eps, 1.0 - eps)


def get_proba_in_class_order(classifier, X: np.ndarray, n_classes: int,
                             eps: float = 1e-15) -> np.ndarray:
    """Return ``predict_proba`` with columns aligned to class order [0, n_classes).

    Tree/SVM classifiers only emit columns for classes seen during training, so
    a fold missing a class would otherwise produce a misaligned matrix. Columns
    for unseen classes are filled with ~0 and each row is renormalized.
    """
    proba = classifier.predict_proba(X)
    classes_ = None
    if hasattr(classifier, 'model') and hasattr(classifier.model, 'classes_'):
        classes_ = classifier.model.classes_

    if classes_ is None:
        # MLP wrappers already output the full, ordered [0, n_classes) columns.
        return proba

    aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
    for src_col, cls in enumerate(np.asarray(classes_).astype(int).tolist()):
        if 0 <= cls < n_classes:
            aligned[:, cls] = proba[:, src_col]
    aligned = aligned / np.clip(aligned.sum(axis=1, keepdims=True), eps, None)
    return aligned


def multiclass_log_loss(y_true: np.ndarray, proba: np.ndarray,
                        eps: float = 1e-15) -> float:
    """Mean negative log-likelihood for single-label multiclass predictions."""
    p = np.clip(proba, eps, 1.0 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(y_true).astype(int)
    return float(np.mean(-np.log(p[np.arange(len(y)), y])))


def binary_cross_entropy(y_true: np.ndarray, proba: np.ndarray,
                         eps: float = 1e-15) -> float:
    """Mean binary cross-entropy averaged over samples and tags (multilabel)."""
    p = np.clip(proba, eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def normalized_information_gain(y_true: np.ndarray, proba: np.ndarray,
                                prior: np.ndarray, task_type: str = 'multiclass',
                                eps: float = 1e-15) -> tuple:
    """Compute G = 1 - L_model / L_chance for one (descriptor, scale).

    Parameters
    ----------
    y_true : np.ndarray
        True labels: integer class ids (multiclass) or 0/1 matrix (multilabel).
    proba : np.ndarray
        Predicted probabilities, columns aligned to class/tag order.
    prior : np.ndarray
        Chance predictor: class prior (multiclass) or tag prevalence (multilabel).
    task_type : str
        'multiclass' or 'multilabel'.

    Returns
    -------
    (G, L_model, L_chance) : tuple of floats
    """
    n = len(y_true)
    prior = np.asarray(prior, dtype=float).reshape(1, -1)
    proba_chance = np.tile(prior, (n, 1))

    if task_type == 'multilabel':
        ll_model = binary_cross_entropy(y_true, proba, eps)
        ll_chance = binary_cross_entropy(y_true, proba_chance, eps)
    else:
        ll_model = multiclass_log_loss(y_true, proba, eps)
        ll_chance = multiclass_log_loss(y_true, proba_chance, eps)

    g = 1.0 - ll_model / ll_chance if ll_chance > 0 else 0.0
    return float(g), float(ll_model), float(ll_chance)


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                        class_names: Optional[List[str]] = None) -> Dict:
    """
    Evaluate multiclass classification (genre, instruments).

    Returns
    -------
    Dict with keys: accuracy, f1_macro, f1_per_class, classification_report.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    results = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class.tolist(),
    }

    if class_names:
        results['f1_by_name'] = {
            name: f1 for name, f1 in zip(class_names, f1_per_class)
        }

    return results


def evaluate_multilabel(y_true: np.ndarray, y_proba: np.ndarray,
                        tag_names: Optional[List[str]] = None,
                        top_k: int = 10) -> Dict:
    """
    Evaluate multilabel classification (auto-tagging).

    Parameters
    ----------
    y_true : np.ndarray
        Binary label matrix (n_samples, n_tags).
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_tags).
    tag_names : list, optional
        Names of the tags.
    top_k : int
        Number of top tags to report individual AP.

    Returns
    -------
    Dict with keys: mAP, roc_auc_macro, ap_per_tag.
    """
    # Filter out tags with no positive examples in y_true
    valid_tags = y_true.sum(axis=0) > 0
    y_true_valid = y_true[:, valid_tags]
    y_proba_valid = y_proba[:, valid_tags]

    # mAP
    ap_per_tag = average_precision_score(y_true_valid, y_proba_valid, average=None)
    mAP = np.mean(ap_per_tag)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true_valid, y_proba_valid, average='macro')
    except ValueError:
        roc_auc = None

    results = {
        'mAP': mAP,
        'roc_auc_macro': roc_auc,
        'ap_per_tag': ap_per_tag.tolist(),
    }

    # Top-k tag APs
    if tag_names:
        valid_tag_names = [tag_names[i] for i in range(len(tag_names)) if valid_tags[i]]
        tag_aps = list(zip(valid_tag_names, ap_per_tag))
        tag_aps.sort(key=lambda x: -x[1])
        results['top_tags'] = tag_aps[:top_k]

    return results


def run_full_evaluation(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    classifier,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strategy: str,
    task_type: str = 'multiclass',
    class_names: Optional[List[str]] = None,
    fusion_params: Optional[Dict] = None,
) -> Dict:
    """
    Run evaluation for a given fusion strategy.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping scale_name -> feature matrix.
    labels : np.ndarray
        Labels.
    classifier : Classifier instance or factory.
    train_idx, test_idx : np.ndarray
        Split indices.
    strategy : str
        One of 'short', 'medium', 'long', 'early', 'late', 'tsi_weighted'.
    task_type : str
        'multiclass' or 'multilabel'.
    class_names : list, optional
        Names of classes/tags.
    fusion_params : dict, optional
        Required for 'tsi_weighted': {tsi_scores, optimal_scales}.

    Returns
    -------
    Dict with evaluation metrics.
    """
    from .fusion import single_scale, early_fusion, late_fusion, tsi_weighted_fusion

    def _align_proba_columns(
        proba: np.ndarray,
        predicted_classes: Optional[np.ndarray],
        class_order: np.ndarray,
    ) -> np.ndarray:
        """Align probability columns to a shared class order."""
        if predicted_classes is None:
            # MLP wrappers already output fixed class order [0..n_classes-1].
            if proba.shape[1] != len(class_order):
                raise ValueError(
                    f"Probability columns ({proba.shape[1]}) do not match class order size ({len(class_order)})."
                )
            return proba

        aligned = np.zeros((proba.shape[0], len(class_order)), dtype=proba.dtype)
        class_to_pos = {c: i for i, c in enumerate(class_order.tolist())}
        for src_col, cls in enumerate(predicted_classes.tolist()):
            if cls in class_to_pos:
                aligned[:, class_to_pos[cls]] = proba[:, src_col]
        return aligned

    if strategy in ('short', 'medium', 'long'):
        X = single_scale(features, strategy)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if task_type == 'multilabel':
            y_proba = classifier.predict_proba(X_test)
            return evaluate_multilabel(y_test, y_proba, class_names)
        else:
            return evaluate_multiclass(y_test, y_pred, class_names)

    elif strategy == 'early':
        X = early_fusion(features)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if task_type == 'multilabel':
            y_proba = classifier.predict_proba(X_test)
            return evaluate_multilabel(y_test, y_proba, class_names)
        else:
            return evaluate_multiclass(y_test, y_pred, class_names)

    elif strategy == 'late':
        from copy import deepcopy
        from joblib import Parallel, delayed

        y_test = labels[test_idx]
        class_order = np.sort(np.unique(labels))

        def _train_scale(scale_name):
            X_scale = features[scale_name]
            clf_scale = deepcopy(classifier)
            clf_scale.fit(X_scale[train_idx], labels[train_idx])
            proba = clf_scale.predict_proba(X_scale[test_idx])
            classes_ = getattr(getattr(clf_scale, 'model', None), 'classes_', None)
            return scale_name, _align_proba_columns(proba, classes_, class_order)

        scale_results = Parallel(n_jobs=3, prefer='threads')(
            delayed(_train_scale)(s) for s in ['short', 'medium', 'long']
        )
        predictions = dict(scale_results)

        avg_proba = late_fusion(predictions)

        if task_type == 'multilabel':
            return evaluate_multilabel(y_test, avg_proba, class_names)
        else:
            y_pred = class_order[np.argmax(avg_proba, axis=1)]
            return evaluate_multiclass(y_test, y_pred, class_names)

    elif strategy == 'tsi_weighted':
        if fusion_params is None:
            raise ValueError("fusion_params required for tsi_weighted strategy")

        X = tsi_weighted_fusion(
            features,
            tsi_scores=fusion_params['tsi_scores'],
            optimal_scales=fusion_params['optimal_scales']
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if task_type == 'multilabel':
            y_proba = classifier.predict_proba(X_test)
            return evaluate_multilabel(y_test, y_proba, class_names)
        else:
            return evaluate_multiclass(y_test, y_pred, class_names)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
