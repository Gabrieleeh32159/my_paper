"""
Feature fusion strategies for multi-scale MIR pipeline.

Implements: single-scale, early fusion, late fusion, TSI-weighted fusion.
"""

import numpy as np
from typing import Dict
from .features import FEATURE_DIMS

# Pre-compute and cache descriptor column indices for the 192-d track vector.
# extract_descriptor_from_track_vector() is called 21×(n_folds) times per TSI
# run; caching avoids recomputing the same index arithmetic on every call.
_DESCRIPTOR_INDICES: Dict[str, list] = {}

def _build_descriptor_index(descriptor_name: str) -> list:
    offset = 0
    for name, dim in FEATURE_DIMS.items():
        if name == descriptor_name:
            break
        offset += dim
    dim = FEATURE_DIMS[descriptor_name]
    block_size = sum(FEATURE_DIMS.values())
    indices = []
    for block_start in [0, block_size, 2 * block_size, 3 * block_size]:
        indices.extend(range(block_start + offset, block_start + offset + dim))
    return indices


def single_scale(features: Dict[str, np.ndarray], scale: str) -> np.ndarray:
    """
    Use features from a single scale only.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping scale_name -> feature matrix (n_samples, 192).
    scale : str
        One of 'short', 'medium', 'long'.

    Returns
    -------
    np.ndarray
        Feature matrix (n_samples, 192).
    """
    return features[scale]


def early_fusion(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate features from all scales.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping scale_name -> feature matrix (n_samples, 192).

    Returns
    -------
    np.ndarray
        Concatenated feature matrix (n_samples, 576).
    """
    return np.hstack([features['short'], features['medium'], features['long']])


def late_fusion(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Average probability predictions from classifiers trained at each scale.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        Mapping scale_name -> probability matrix (n_samples, n_classes).

    Returns
    -------
    np.ndarray
        Averaged probability matrix (n_samples, n_classes).
    """
    pred_list = [predictions['short'], predictions['medium'], predictions['long']]
    return np.mean(pred_list, axis=0)


def tsi_weighted_fusion(
    features: Dict[str, np.ndarray],
    tsi_scores: Dict[str, float],
    optimal_scales: Dict[str, str],
) -> np.ndarray:
    """
    TSI-weighted fusion: for each descriptor, select its optimal scale
    and weight by its TSI score.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping scale_name -> feature matrix (n_samples, 192).
    tsi_scores : Dict[str, float]
        Mapping descriptor_name -> TSI value.
    optimal_scales : Dict[str, str]
        Mapping descriptor_name -> optimal scale name ('short'/'medium'/'long').

    Returns
    -------
    np.ndarray
        TSI-weighted feature matrix (n_samples, variable dim).
    """
    n_samples = features['short'].shape[0]
    parts = []

    for desc_name, dim in FEATURE_DIMS.items():
        scale = optimal_scales[desc_name]
        tsi = tsi_scores[desc_name]

        # Extract descriptor-specific features from the optimal scale
        desc_feats = extract_descriptor_from_track_vector(
            features[scale], desc_name
        )

        # Weight by TSI (higher TSI = more important to use optimal scale)
        # Add small epsilon to avoid zero-weighting robust descriptors
        weight = max(tsi, 0.1)
        parts.append(desc_feats * weight)

    return np.hstack(parts)


def extract_descriptor_from_track_vector(
    X: np.ndarray, descriptor_name: str
) -> np.ndarray:
    """
    Extract columns corresponding to a specific descriptor from the 192-d
    track-level feature vector.

    The 192-d vector structure:
    [mean_of_window_means(48) | mean_of_window_stds(48) |
     std_of_window_means(48) | std_of_window_stds(48)]

    Each 48-d block follows the order in FEATURE_DIMS.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, 192).
    descriptor_name : str
        Name of the descriptor.

    Returns
    -------
    np.ndarray
        Extracted descriptor features (n_samples, 4*dim).
    """
    if descriptor_name not in _DESCRIPTOR_INDICES:
        _DESCRIPTOR_INDICES[descriptor_name] = _build_descriptor_index(descriptor_name)
    indices = _DESCRIPTOR_INDICES[descriptor_name]
    if X.ndim == 1:
        return X[indices]
    return X[:, indices]
