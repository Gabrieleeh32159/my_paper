"""
Temporal Sensitivity Index (TSI) computation.

The TSI is defined on the *normalized information gain* of each descriptor,
which uses the full predicted probability distribution (log-loss) rather than
the thresholded accuracy:

    G(f, k) = 1 - L(f, k) / L_chance

    TSI(f) = max_k G(f, k) - min_k G(f, k)

where L(f, k) is the (clipped) log-loss of a calibrated classifier trained on
descriptor f alone at scale k, and L_chance is the log-loss of the constant
class-prior predictor. Because G is the fraction of label uncertainty resolved,
the TSI is the share of resolvable label information that is gained or lost by
choosing the best vs. worst temporal scale for that descriptor -- a proper,
imbalance-aware, calibration-aware replacement for the accuracy-range TSI.

The optimal scale is k*(f) = argmax_k G(f, k) = argmin_k L(f, k).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from .features import FEATURE_DIMS, SCALES
from .fusion import extract_descriptor_from_track_vector
from .classifiers import get_classifier
from .evaluation import (
    class_prior, tag_prevalence, get_proba_in_class_order,
    normalized_information_gain,
)


def _descriptor_info_gain(
    X_desc: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    classifier_name: str,
    task_type: str,
    prior: np.ndarray,
    **clf_kwargs,
) -> Tuple[float, float]:
    """Train one classifier on a single descriptor/scale and return (G, log-loss)."""
    X_train, X_test = X_desc[train_idx], X_desc[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    clf = get_classifier(
        classifier_name, input_dim=X_train.shape[1],
        n_classes=n_classes, task_type=task_type, **clf_kwargs
    )
    clf.fit(X_train, y_train)

    if task_type == 'multilabel':
        proba = clf.predict_proba(X_test)
    else:
        proba = get_proba_in_class_order(clf, X_test, n_classes)

    g, ll, _ = normalized_information_gain(y_test, proba, prior, task_type)
    return g, ll


def compute_tsi(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    classifier_name: str = 'rf',
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    task_type: str = 'multiclass',
    **clf_kwargs
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Compute the information-gain TSI for each descriptor on a single split.

    For each descriptor f, trains a classifier using ONLY that descriptor
    at each scale k, and computes:
        G(f, k) = 1 - L(f, k) / L_chance
        TSI(f)  = max_k G(f, k) - min_k G(f, k)

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping scale_name -> feature matrix (n_samples, 192).
    labels : np.ndarray
        Class labels (n_samples,) for multiclass, or (n_samples, n_tags) for
        multilabel.
    n_classes : int
        Number of classes (multiclass) or tags (multilabel).
    classifier_name : str
        Classifier to use ('rf', 'xgb', 'svm', 'mlp'). Multilabel is only
        supported with 'mlp'.
    train_idx, test_idx : np.ndarray, optional
        Split indices. If None, uses a stratified 80/20 split.
    task_type : str
        'multiclass' or 'multilabel'.

    Returns
    -------
    tsi_scores : Dict[str, float]
        Mapping descriptor_name -> TSI value.
    info_gain_matrix : Dict[str, Dict[str, float]]
        Mapping descriptor_name -> {scale_name -> normalized information gain}.
    optimal_scales : Dict[str, str]
        Mapping descriptor_name -> optimal scale name (argmax G).
    """
    from sklearn.model_selection import train_test_split

    n_samples = features['short'].shape[0]

    # Default split if not provided
    if train_idx is None or test_idx is None:
        all_idx = np.arange(n_samples)
        stratify = labels if task_type == 'multiclass' else None
        train_idx, test_idx = train_test_split(
            all_idx, test_size=0.2, stratify=stratify, random_state=42
        )

    y_train = labels[train_idx]
    if task_type == 'multilabel':
        prior = tag_prevalence(y_train)
    else:
        prior = class_prior(y_train, n_classes)

    tsi_scores = {}
    info_gain_matrix = {}
    optimal_scales = {}

    for desc_name in FEATURE_DIMS.keys():
        desc_gains = {}

        for scale_name in SCALES.keys():
            X_desc = extract_descriptor_from_track_vector(
                features[scale_name], desc_name
            )
            g, _ = _descriptor_info_gain(
                X_desc, labels, train_idx, test_idx, n_classes,
                classifier_name, task_type, prior, **clf_kwargs
            )
            desc_gains[scale_name] = g

        info_gain_matrix[desc_name] = desc_gains
        tsi_scores[desc_name] = max(desc_gains.values()) - min(desc_gains.values())
        optimal_scales[desc_name] = max(desc_gains, key=desc_gains.get)

    return tsi_scores, info_gain_matrix, optimal_scales


def _accumulate_cv_info_gain(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    classifier_name: str,
    task_type: str,
    **clf_kwargs,
) -> Dict[str, Dict[str, List[float]]]:
    """Collect per-fold information gain G(f, k) for every descriptor and scale."""
    g_accum = {desc: {scale: [] for scale in SCALES.keys()}
               for desc in FEATURE_DIMS.keys()}

    for train_idx, test_idx in folds:
        y_train = labels[train_idx]
        if task_type == 'multilabel':
            prior = tag_prevalence(y_train)
        else:
            prior = class_prior(y_train, n_classes)

        for desc_name in FEATURE_DIMS.keys():
            for scale_name in SCALES.keys():
                X_desc = extract_descriptor_from_track_vector(
                    features[scale_name], desc_name
                )
                g, _ = _descriptor_info_gain(
                    X_desc, labels, train_idx, test_idx, n_classes,
                    classifier_name, task_type, prior, **clf_kwargs
                )
                g_accum[desc_name][scale_name].append(g)

    return g_accum


def compute_tsi_cv(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    classifier_name: str = 'rf',
    task_type: str = 'multiclass',
    **clf_kwargs
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Compute the information-gain TSI with cross-validation (e.g. GTZAN).

    Averages G(f, k) across folds before computing TSI. For per-descriptor
    uncertainty (std, bootstrap CI) and best-vs-worst-scale significance, use
    :func:`compute_tsi_cv_full`.

    Returns
    -------
    Same shape as :func:`compute_tsi`:
        (tsi_scores, info_gain_matrix, optimal_scales).
    """
    g_accum = _accumulate_cv_info_gain(
        features, labels, n_classes, folds, classifier_name, task_type, **clf_kwargs
    )

    tsi_scores = {}
    info_gain_matrix = {}
    optimal_scales = {}

    for desc_name in FEATURE_DIMS.keys():
        desc_gains = {
            scale: float(np.mean(g_accum[desc_name][scale]))
            for scale in SCALES.keys()
        }
        info_gain_matrix[desc_name] = desc_gains
        tsi_scores[desc_name] = max(desc_gains.values()) - min(desc_gains.values())
        optimal_scales[desc_name] = max(desc_gains, key=desc_gains.get)

    return tsi_scores, info_gain_matrix, optimal_scales


def compute_tsi_cv_full(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    classifier_name: str = 'rf',
    task_type: str = 'multiclass',
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    **clf_kwargs
) -> Dict:
    """
    Cross-validated TSI with full statistics, ready to serialize for reporting.

    For each descriptor it reports the mean TSI, its fold-level std, a bootstrap
    confidence interval (resampling folds), the per-scale information gain
    (mean +/- std), and whether the best vs. worst scale differ significantly
    (paired Wilcoxon across folds). A descriptor only counts as genuinely
    temporally sensitive when that test is significant.

    Returns
    -------
    Dict with keys:
        'tsi_scores', 'tsi_std', 'tsi_ci', 'info_gain_matrix', 'info_gain_std',
        'optimal_scales', 'scale_significant', 'scale_p_value'.
    """
    from .stats import tsi_bootstrap_ci, tsi_scale_significance

    g_accum = _accumulate_cv_info_gain(
        features, labels, n_classes, folds, classifier_name, task_type, **clf_kwargs
    )

    result = {
        'tsi_scores': {}, 'tsi_std': {}, 'tsi_ci': {},
        'info_gain_matrix': {}, 'info_gain_std': {},
        'optimal_scales': {}, 'scale_significant': {}, 'scale_p_value': {},
    }

    for desc_name in FEATURE_DIMS.keys():
        per_fold = {scale: np.asarray(g_accum[desc_name][scale])
                    for scale in SCALES.keys()}
        mean_gains = {scale: float(per_fold[scale].mean()) for scale in SCALES.keys()}
        std_gains = {scale: float(per_fold[scale].std(ddof=1))
                     if len(per_fold[scale]) > 1 else 0.0
                     for scale in SCALES.keys()}

        best = max(mean_gains, key=mean_gains.get)
        worst = min(mean_gains, key=mean_gains.get)

        ci = tsi_bootstrap_ci(per_fold, n_bootstrap=n_bootstrap)
        sig = tsi_scale_significance(per_fold[best], per_fold[worst], alpha=alpha)

        result['tsi_scores'][desc_name] = mean_gains[best] - mean_gains[worst]
        result['tsi_std'][desc_name] = ci['tsi_std']
        result['tsi_ci'][desc_name] = [ci['ci_lower'], ci['ci_upper']]
        result['info_gain_matrix'][desc_name] = mean_gains
        result['info_gain_std'][desc_name] = std_gains
        result['optimal_scales'][desc_name] = best
        result['scale_significant'][desc_name] = sig['significant']
        result['scale_p_value'][desc_name] = sig['p_value']

    return result


def tsi_consistency(
    tsi_results: Dict[str, Dict[str, float]]
) -> Dict[Tuple[str, str], float]:
    """
    Compute Spearman correlation between TSI rankings across tasks/classifiers.

    Parameters
    ----------
    tsi_results : Dict[str, Dict[str, float]]
        Mapping config_name -> {descriptor_name -> TSI value}.

    Returns
    -------
    Dict[Tuple[str, str], float]
        Pairwise Spearman correlations between configurations.
    """
    from scipy.stats import spearmanr

    configs = list(tsi_results.keys())
    descriptors = list(FEATURE_DIMS.keys())
    correlations = {}

    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            rank_i = [tsi_results[configs[i]][d] for d in descriptors]
            rank_j = [tsi_results[configs[j]][d] for d in descriptors]
            rho, pval = spearmanr(rank_i, rank_j)
            correlations[(configs[i], configs[j])] = rho

    return correlations
