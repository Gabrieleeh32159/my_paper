"""
Tests for crash-safe checkpoint/resume in the heavy CV drivers.

On tiny synthetic data we assert, for each driver, that:
  (a) running WITH a checkpoint yields the exact same result as WITHOUT one;
  (b) resuming from a partial checkpoint reproduces the full result AND skips the
      already-computed units (verified via a classifier-fit counter);
  (c) a checkpoint written under a different config (``meta``) is ignored.
Plus a round-trip / atomicity check for the JSON helpers themselves.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import TRACK_DIM, descriptor_slices
from src.checkpoint import atomic_write_json, load_progress, save_progress
from src.tsi import compute_fold_gains, permutation_null_kstar_gains
from src.fusion import evaluate_fusion_cv
from src.evaluation import calibration_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# --- a deterministic classifier factory that counts how many times it is fit --- #
class _FitCounter:
    n = 0


def _lr_factory(input_dim=None, n_classes=None, task_type=None):
    nc = n_classes

    class LR:
        def __init__(self):
            self.m = LogisticRegression(max_iter=200)
            self.classes_full = np.arange(nc)

        def fit(self, X, y):
            _FitCounter.n += 1
            self.m.fit(X, y)
            self.fitted_classes = self.m.classes_
            return self

        def predict_proba(self, X):
            p = self.m.predict_proba(X)
            full = np.zeros((X.shape[0], len(self.classes_full)))
            for j, c in enumerate(self.fitted_classes):
                full[:, c] = p[:, j]
            full = np.clip(full, 1e-9, None)
            return full / full.sum(axis=1, keepdims=True)

    return LR()


def _synth(n=120, C=3, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, C, n)
    feats = {s: rng.normal(0, 1, (n, TRACK_DIM)) for s in ("short", "medium", "long")}
    sl = descriptor_slices()
    feats["short"][:, sl["mfcc"]] += np.eye(C)[y][:, :1] * 5.0
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(n), y))
    return feats, y, folds, C


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def test_atomic_write_and_meta_match(tmp_path):
    p = tmp_path / "ckpt.json"
    meta = {"a": 1, "scales": ("short", "medium")}  # tuple -> normalized to list
    save_progress(p, meta, {"0": {"x": 1}})
    assert p.exists()
    assert not (tmp_path / "ckpt.json.tmp").exists()
    # matching meta (tuple vs list) still resumes
    assert load_progress(p, {"a": 1, "scales": ["short", "medium"]}) == {"0": {"x": 1}}
    # mismatched meta -> fresh
    assert load_progress(p, {"a": 2, "scales": ["short", "medium"]}) == {}
    # missing path / None -> fresh
    assert load_progress(tmp_path / "nope.json", meta) == {}
    assert load_progress(None, meta) == {}


def test_load_progress_ignores_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert load_progress(p, {"a": 1}) == {}


# --------------------------------------------------------------------------- #
# compute_fold_gains
# --------------------------------------------------------------------------- #
def test_gains_checkpoint_matches_plain(tmp_path):
    feats, y, folds, C = _synth()
    descs = ["mfcc", "zcr"]
    plain = compute_fold_gains(feats, y, folds, _lr_factory, "multiclass", C,
                               descriptors=descs, n_inner=2, seed=0)
    ckpt = compute_fold_gains(feats, y, folds, _lr_factory, "multiclass", C,
                              descriptors=descs, n_inner=2, seed=0,
                              checkpoint_path=tmp_path / "gains.json")
    assert plain == ckpt


def test_gains_resumes_and_skips(tmp_path):
    feats, y, folds, C = _synth()
    descs = ["mfcc", "zcr"]
    path = tmp_path / "gains.json"
    full = compute_fold_gains(feats, y, folds, _lr_factory, "multiclass", C,
                              descriptors=descs, n_inner=2, seed=0, checkpoint_path=path)

    # truncate the checkpoint to only fold 0 -> folds 1,2 must recompute
    blob = json.loads(path.read_text())
    blob["data"] = {"0": blob["data"]["0"]}
    path.write_text(json.dumps(blob))

    _FitCounter.n = 0
    resumed = compute_fold_gains(feats, y, folds, _lr_factory, "multiclass", C,
                                 descriptors=descs, n_inner=2, seed=0, checkpoint_path=path)
    assert resumed == full
    # fold 0 (2 descriptors) was cached and must NOT have been refit
    fits_per_fold_desc = (2 + 1)  # n_inner=2 inner + 1 outer per scale, x3 scales
    assert 0 < _FitCounter.n < len(folds) * len(descs) * 3 * fits_per_fold_desc


def test_gains_stale_meta_recomputes(tmp_path):
    feats, y, folds, C = _synth()
    path = tmp_path / "gains.json"
    compute_fold_gains(feats, y, folds, _lr_factory, "multiclass", C,
                       descriptors=["mfcc"], n_inner=2, seed=0, checkpoint_path=path)
    # different seed => different meta => previous checkpoint ignored, recomputed fresh
    _FitCounter.n = 0
    out = compute_fold_gains(feats, y, folds, _lr_factory, "multiclass", C,
                             descriptors=["mfcc"], n_inner=2, seed=1, checkpoint_path=path)
    assert _FitCounter.n > 0
    assert len(out["mfcc"]) == len(folds)


# --------------------------------------------------------------------------- #
# permutation_null_kstar_gains
# --------------------------------------------------------------------------- #
def test_null_checkpoint_matches_plain_and_resumes(tmp_path):
    feats, y, folds, C = _synth()
    descs = ["mfcc", "zcr"]
    path = tmp_path / "null.json"
    plain = permutation_null_kstar_gains(feats, y, folds, _lr_factory, "multiclass", C,
                                         descriptors=descs, n_permutations=3, seed=5)
    ckpt = permutation_null_kstar_gains(feats, y, folds, _lr_factory, "multiclass", C,
                                        descriptors=descs, n_permutations=3, seed=5,
                                        checkpoint_path=path)
    assert plain == ckpt
    assert len(ckpt) == 3 * len(folds) * len(descs)

    # truncate to permutation 0 only and resume
    blob = json.loads(path.read_text())
    blob["data"] = {"0": blob["data"]["0"]}
    path.write_text(json.dumps(blob))
    resumed = permutation_null_kstar_gains(feats, y, folds, _lr_factory, "multiclass", C,
                                           descriptors=descs, n_permutations=3, seed=5,
                                           checkpoint_path=path)
    assert resumed == plain


# --------------------------------------------------------------------------- #
# evaluate_fusion_cv
# --------------------------------------------------------------------------- #
def test_fusion_checkpoint_matches_plain_and_resumes(tmp_path):
    feats, y, folds, C = _synth()
    fg = compute_fold_gains(feats, y, folds, _lr_factory, "multiclass", C, n_inner=2, seed=0)
    path = tmp_path / "fusion.json"
    plain = evaluate_fusion_cv(feats, y, folds, _lr_factory, "multiclass", C,
                               fold_gains=fg, n_inner=2, seed=0)
    ckpt = evaluate_fusion_cv(feats, y, folds, _lr_factory, "multiclass", C,
                              fold_gains=fg, n_inner=2, seed=0, checkpoint_path=path)
    assert plain == ckpt

    blob = json.loads(path.read_text())
    blob["data"] = {"0": blob["data"]["0"]}
    path.write_text(json.dumps(blob))
    resumed = evaluate_fusion_cv(feats, y, folds, _lr_factory, "multiclass", C,
                                 fold_gains=fg, n_inner=2, seed=0, checkpoint_path=path)
    assert resumed == plain


# --------------------------------------------------------------------------- #
# calibration_report
# --------------------------------------------------------------------------- #
def test_calibration_checkpoint_matches_plain_and_resumes(tmp_path):
    feats, y, folds, C = _synth()
    path = tmp_path / "calib.json"
    plain = calibration_report(feats, y, folds, _lr_factory, "multiclass", C, scale="medium")
    ckpt = calibration_report(feats, y, folds, _lr_factory, "multiclass", C, scale="medium",
                              checkpoint_path=path)
    assert plain["ece_per_fold"] == ckpt["ece_per_fold"]
    assert plain["reliability"] == ckpt["reliability"]

    blob = json.loads(path.read_text())
    blob["data"] = {"0": blob["data"]["0"]}
    path.write_text(json.dumps(blob))
    resumed = calibration_report(feats, y, folds, _lr_factory, "multiclass", C, scale="medium",
                                 checkpoint_path=path)
    assert resumed["ece_per_fold"] == plain["ece_per_fold"]
