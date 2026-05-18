"""
Temporal Sensitivity Index (TSI) computation.

TSI(f) = (max_k Acc(f,k) - min_k Acc(f,k)) / Acc_chance

Quantifies how much a descriptor's discriminability varies across temporal scales.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from .features import FEATURE_DIMS, SCALES
from .fusion import extract_descriptor_from_track_vector
from .classifiers import get_classifier


def compute_tsi(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    classifier_name: str = 'rf',
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    **clf_kwargs
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Compute TSI for each descriptor.

    For each descriptor f, trains a classifier using ONLY that descriptor
    at each scale k, and computes:
    TSI(f) = (max_k Acc(f,k) - min_k Acc(f,k)) / Acc_chance

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping scale_name -> feature matrix (n_samples, 192).
    labels : np.ndarray
        Class labels (n_samples,).
    n_classes : int
        Number of classes.
    classifier_name : str
        Classifier to use ('rf', 'svm', 'mlp').
    train_idx : np.ndarray, optional
        Training indices. If None, uses 80/20 split.
    test_idx : np.ndarray, optional
        Test indices.

    Returns
    -------
    tsi_scores : Dict[str, float]
        Mapping descriptor_name -> TSI value.
    accuracy_matrix : Dict[str, Dict[str, float]]
        Mapping descriptor_name -> {scale_name -> accuracy}.
    optimal_scales : Dict[str, str]
        Mapping descriptor_name -> optimal scale name.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    n_samples = features['short'].shape[0]

    # Default split if not provided
    if train_idx is None or test_idx is None:
        all_idx = np.arange(n_samples)
        train_idx, test_idx = train_test_split(
            all_idx, test_size=0.2, stratify=labels, random_state=42
        )

    acc_chance = 1.0 / n_classes

    tsi_scores = {}
    accuracy_matrix = {}
    optimal_scales = {}

    for desc_name in FEATURE_DIMS.keys():
        desc_accs = {}

        for scale_name in SCALES.keys():
            # Extract this descriptor's features from this scale
            X_desc = extract_descriptor_from_track_vector(
                features[scale_name], desc_name
            )

            X_train = X_desc[train_idx]
            X_test = X_desc[test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            # Train classifier on this descriptor alone
            input_dim = X_train.shape[1]
            clf = get_classifier(
                classifier_name, input_dim=input_dim,
                n_classes=n_classes, **clf_kwargs
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            desc_accs[scale_name] = acc

        accuracy_matrix[desc_name] = desc_accs

        # Compute TSI
        max_acc = max(desc_accs.values())
        min_acc = min(desc_accs.values())
        tsi = (max_acc - min_acc) / acc_chance
        tsi_scores[desc_name] = tsi

        # Optimal scale
        optimal_scales[desc_name] = max(desc_accs, key=desc_accs.get)

    return tsi_scores, accuracy_matrix, optimal_scales


def compute_tsi_cv(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    classifier_name: str = 'rf',
    **clf_kwargs
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Compute TSI with cross-validation (for GTZAN).

    Averages accuracy across folds before computing TSI.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping scale_name -> feature matrix (n_samples, 192).
    labels : np.ndarray
        Class labels.
    n_classes : int
        Number of classes.
    folds : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices).
    classifier_name : str
        Classifier to use.

    Returns
    -------
    Same as compute_tsi.
    """
    from sklearn.metrics import accuracy_score

    acc_chance = 1.0 / n_classes

    # Accumulate accuracies per descriptor per scale across folds
    acc_accum = {desc: {scale: [] for scale in SCALES.keys()}
                 for desc in FEATURE_DIMS.keys()}

    for train_idx, test_idx in folds:
        for desc_name in FEATURE_DIMS.keys():
            for scale_name in SCALES.keys():
                X_desc = extract_descriptor_from_track_vector(
                    features[scale_name], desc_name
                )

                X_train = X_desc[train_idx]
                X_test = X_desc[test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]

                input_dim = X_train.shape[1]
                clf = get_classifier(
                    classifier_name, input_dim=input_dim,
                    n_classes=n_classes, **clf_kwargs
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                acc_accum[desc_name][scale_name].append(acc)

    # Average across folds
    tsi_scores = {}
    accuracy_matrix = {}
    optimal_scales = {}

    for desc_name in FEATURE_DIMS.keys():
        desc_accs = {
            scale: np.mean(acc_accum[desc_name][scale])
            for scale in SCALES.keys()
        }
        accuracy_matrix[desc_name] = desc_accs

        max_acc = max(desc_accs.values())
        min_acc = min(desc_accs.values())
        tsi_scores[desc_name] = (max_acc - min_acc) / acc_chance
        optimal_scales[desc_name] = max(desc_accs, key=desc_accs.get)

    return tsi_scores, accuracy_matrix, optimal_scales


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
