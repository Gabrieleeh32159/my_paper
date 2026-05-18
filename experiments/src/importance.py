"""
Feature importance analysis.

Implements:
1. Permutation Importance (PI) - primary method.
2. Mean Decrease in Impurity (MDI) - secondary (from RF).
3. Scale-conditioned importance aggregation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.inspection import permutation_importance
from .features import FEATURE_DIMS, SCALES


def compute_permutation_importance(
    classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats: int = 30,
    scoring: str = 'f1_macro',
    random_state: int = 42
) -> Dict:
    """
    Compute permutation importance for all features.

    Parameters
    ----------
    classifier : fitted classifier with predict method.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    n_repeats : int
        Number of permutation repetitions.
    scoring : str
        Scoring metric.

    Returns
    -------
    Dict with keys: importances_mean, importances_std, importances (raw).
    """
    result = permutation_importance(
        classifier, X_test, y_test,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1
    )

    return {
        'importances_mean': result.importances_mean,
        'importances_std': result.importances_std,
        'importances': result.importances,
    }


def compute_mdi_importance(rf_classifier) -> np.ndarray:
    """
    Extract Mean Decrease in Impurity from a fitted Random Forest.

    Parameters
    ----------
    rf_classifier : RFClassifier instance (fitted).

    Returns
    -------
    np.ndarray
        Feature importances from MDI.
    """
    return rf_classifier.feature_importances_


def aggregate_importance_by_descriptor(
    importances: np.ndarray,
    vector_dim: int = 192
) -> Dict[str, float]:
    """
    Aggregate feature importances by descriptor name.

    The 192-d vector has 4 blocks of 48 features (mean_mean, mean_std, std_mean, std_std).
    Each 48-d block follows FEATURE_DIMS order.

    Parameters
    ----------
    importances : np.ndarray
        Per-feature importance values (192-d for single-scale, 576-d for early fusion).
    vector_dim : int
        Dimension per scale (192).

    Returns
    -------
    Dict[str, float]
        Aggregated importance per descriptor.
    """
    n_scales = len(importances) // vector_dim
    block_size = sum(FEATURE_DIMS.values())  # 48

    desc_importance = {name: 0.0 for name in FEATURE_DIMS.keys()}

    for scale_idx in range(n_scales):
        base = scale_idx * vector_dim
        for block_offset in [0, block_size, 2 * block_size, 3 * block_size]:
            feat_offset = 0
            for name, dim in FEATURE_DIMS.items():
                start = base + block_offset + feat_offset
                end = start + dim
                desc_importance[name] += np.sum(importances[start:end])
                feat_offset += dim

    return desc_importance


def aggregate_importance_by_scale(
    importances: np.ndarray,
    vector_dim: int = 192
) -> Dict[str, float]:
    """
    Aggregate feature importances by temporal scale (for early fusion 576-d vector).

    Parameters
    ----------
    importances : np.ndarray
        Per-feature importance values (576-d for early fusion).
    vector_dim : int
        Dimension per scale (192).

    Returns
    -------
    Dict[str, float]
        Aggregated importance per scale.
    """
    scale_names = list(SCALES.keys())
    scale_importance = {}

    for i, scale in enumerate(scale_names):
        start = i * vector_dim
        end = start + vector_dim
        scale_importance[scale] = np.sum(importances[start:end])

    return scale_importance


def aggregate_importance_by_descriptor_and_scale(
    importances: np.ndarray,
    vector_dim: int = 192
) -> Dict[str, Dict[str, float]]:
    """
    Build the 7×3 importance matrix (descriptor × scale) for early fusion.

    Parameters
    ----------
    importances : np.ndarray
        Per-feature importance values (576-d for early fusion).

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict: descriptor_name -> scale_name -> importance.
    """
    scale_names = list(SCALES.keys())
    block_size = sum(FEATURE_DIMS.values())  # 48
    matrix = {name: {} for name in FEATURE_DIMS.keys()}

    for scale_idx, scale_name in enumerate(scale_names):
        base = scale_idx * vector_dim
        for block_offset in [0, block_size, 2 * block_size, 3 * block_size]:
            feat_offset = 0
            for name, dim in FEATURE_DIMS.items():
                start = base + block_offset + feat_offset
                end = start + dim
                if scale_name not in matrix[name]:
                    matrix[name][scale_name] = 0.0
                matrix[name][scale_name] += np.sum(importances[start:end])
                feat_offset += dim

    return matrix


def bootstrap_importance_ci(
    importances_raw: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for feature importances.

    Parameters
    ----------
    importances_raw : np.ndarray
        Raw permutation importance matrix (n_features, n_repeats).
    n_bootstrap : int
        Number of bootstrap samples.
    ci : float
        Confidence level.

    Returns
    -------
    Tuple of (lower_bound, upper_bound) arrays.
    """
    rng = np.random.RandomState(random_state)
    n_features, n_repeats = importances_raw.shape

    boot_means = np.zeros((n_bootstrap, n_features))
    for b in range(n_bootstrap):
        idx = rng.choice(n_repeats, size=n_repeats, replace=True)
        boot_means[b] = importances_raw[:, idx].mean(axis=1)

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100, axis=0)
    upper = np.percentile(boot_means, (1 - alpha) * 100, axis=0)

    return lower, upper
