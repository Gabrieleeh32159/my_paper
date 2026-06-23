"""Classifiers must tolerate a training set whose labels are a non-contiguous
subset of the global class space (e.g. IRMAS official split, where the train
partition can lack some of the 11 instruments).

XGBoost's sklearn wrapper (>=1.6) hard-rejects labels that are not exactly
``0..k-1`` ("Invalid classes inferred from unique values of `y`"). The pipeline
passes the GLOBAL ``n_classes`` everywhere and downstream code
(``gain_from_predictions``) expects ``predict_proba`` to have ``n_classes``
columns aligned to the global label ids. These tests pin both invariants.

Run: cd experiments && python -m pytest tests/test_classifier_label_subset.py -q
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifiers import get_classifier


def _subset_data(present_labels, n_per=12, dim=8, seed=0):
    """Synthetic data whose labels are exactly ``present_labels`` (a subset of
    the global class space), with a class-dependent mean so the fit is non-trivial."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c in present_labels:
        X.append(rng.randn(n_per, dim) + c)
        y.append(np.full(n_per, c))
    return np.vstack(X), np.concatenate(y)


def test_xgb_fits_on_noncontiguous_label_subset():
    # Global space is 11 classes; train partition only carries 5..10 (the IRMAS case).
    n_classes = 11
    present = [5, 6, 7, 8, 9, 10]
    X, y = _subset_data(present)

    clf = get_classifier("xgb", input_dim=X.shape[1], n_classes=n_classes,
                         task_type="multiclass")
    clf.fit(X, y)  # must NOT raise "Invalid classes inferred from unique values of `y`"

    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), n_classes)
    # columns for classes absent from training must be exactly zero
    absent = [c for c in range(n_classes) if c not in present]
    assert np.allclose(proba[:, absent], 0.0)
    # rows are still valid distributions over the present classes
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_rf_predict_proba_is_global_width_on_subset():
    n_classes = 11
    present = [5, 6, 7, 8, 9, 10]
    X, y = _subset_data(present)

    clf = get_classifier("rf", input_dim=X.shape[1], n_classes=n_classes,
                         task_type="multiclass")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), n_classes)
    absent = [c for c in range(n_classes) if c not in present]
    assert np.allclose(proba[:, absent], 0.0)


def test_full_class_set_unchanged():
    # When all classes are present and contiguous from 0, behaviour is identical:
    # predict_proba is n_classes wide and a proper distribution.
    n_classes = 4
    X, y = _subset_data([0, 1, 2, 3])
    clf = get_classifier("xgb", input_dim=X.shape[1], n_classes=n_classes,
                         task_type="multiclass")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    test_xgb_fits_on_noncontiguous_label_subset()
    test_rf_predict_proba_is_global_width_on_subset()
    test_full_class_set_unchanged()
    print("ok")
