"""
Unit tests for the redesigned TSI method, on small synthetic data only
(no Google Drive, no heavy classifiers required for the core checks).

Run:  cd experiments && python -m pytest tests/ -q
  or: cd experiments && python tests/test_method.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    FEATURE_DIMS, TRACK_DIM, descriptor_slices, extract_descriptor,
)
from src.evaluation import (
    information_gain, truncate_gain, gain_from_predictions, chance_log_loss,
    log_loss_multiclass, base_rate,
)
from src.tsi import (
    select_k_star, tsi_from_fold_gains, gate_threshold, apply_gate, bootstrap_ci,
)
from src.fusion import (
    learn_late_fusion_weights, combine_late, tsi_guided_features, tsi_prior_weights,
)


# --------------------------------------------------------------------------- #
# Descriptor slicing helper
# --------------------------------------------------------------------------- #
def test_descriptor_slices_partition_192():
    sl = descriptor_slices()
    all_cols = np.concatenate([sl[f] for f in FEATURE_DIMS])
    # Every column 0..191 covered exactly once (a clean partition).
    assert sorted(all_cols.tolist()) == list(range(TRACK_DIM))
    assert len(all_cols) == TRACK_DIM == 192


def test_extract_descriptor_dims():
    X = np.arange(5 * TRACK_DIM).reshape(5, TRACK_DIM).astype(float)
    assert extract_descriptor(X, "mfcc").shape == (5, 4 * 20)   # 80
    assert extract_descriptor(X, "zcr").shape == (5, 4 * 1)     # 4
    assert extract_descriptor(X, "tonnetz").shape == (5, 4 * 6)  # 24
    # 1-D input path
    assert extract_descriptor(X[0], "mfcc").shape == (80,)


def test_extract_descriptor_picks_right_columns():
    # ZCR occupies, within each 48-block, the slice after the first 41 dims
    # (mfcc20+chroma12+sc1+scon7+sr1 = 41). Check the actual columns match.
    sl = descriptor_slices()
    X = np.zeros((1, TRACK_DIM))
    X[0, sl["zcr"]] = 7.0
    got = extract_descriptor(X, "zcr")
    assert np.allclose(got, 7.0)
    assert got.shape == (1, 4)


# --------------------------------------------------------------------------- #
# Information gain bounds and truncation
# --------------------------------------------------------------------------- #
def test_information_gain_upper_bound_and_truncation():
    # Perfect predictions -> L ~ 0 -> G ~ 1 (<= 1).
    g_perfect = information_gain(1e-9, 1.0)
    assert g_perfect <= 1.0 + 1e-9
    # Worse-than-chance -> G < 0, truncates to 0.
    g_bad = information_gain(2.0, 1.0)
    assert g_bad < 0
    assert truncate_gain(g_bad) == 0.0
    assert truncate_gain(0.4) == 0.4


def test_gain_from_predictions_bounded_and_chance_is_zero():
    rng = np.random.RandomState(0)
    C, n = 4, 200
    y_train = rng.randint(0, C, 400)
    y_eval = rng.randint(0, C, n)
    # base-rate predictions -> G ~ 0
    p = base_rate(y_train, "multiclass", n_classes=C)
    proba_chance = np.tile(p, (n, 1))
    g0 = gain_from_predictions(y_train, y_eval, proba_chance, "multiclass", n_classes=C)
    assert abs(g0) < 0.05
    # near-perfect predictions -> G close to 1 and <= 1
    proba_perfect = np.full((n, C), 1e-6)
    proba_perfect[np.arange(n), y_eval] = 1.0
    g1 = gain_from_predictions(y_train, y_eval, proba_perfect, "multiclass", n_classes=C)
    assert g1 <= 1.0 + 1e-9 and g1 > 0.9


def test_chance_log_loss_uniform_is_logC():
    y = np.array([0, 1, 2, 3] * 25)
    L = chance_log_loss(y, y, "multiclass", n_classes=4)
    assert abs(L - np.log(4)) < 1e-6


def test_multilabel_base_rate_clips_prevalence_M6():
    # Tag 0 has zero prevalence in train, tag 1 saturated -> both must be clipped
    # so the chance log-loss stays finite when eval contains the opposite labels.
    from src.evaluation import PREVALENCE_EPS, log_loss_multilabel
    y_train = np.zeros((50, 2), dtype=float)
    y_train[:, 1] = 1.0  # tag1 always positive, tag0 always negative
    pi = base_rate(y_train, "multilabel")
    assert pi[0] >= PREVALENCE_EPS and pi[1] <= 1.0 - PREVALENCE_EPS
    # eval flips both tags -> chance log-loss must be large but FINITE (not inf)
    y_eval = np.zeros((10, 2), dtype=float)
    y_eval[:, 0] = 1.0
    L = chance_log_loss(y_train, y_eval, "multilabel")
    assert np.isfinite(L) and L > 0


# --------------------------------------------------------------------------- #
# k* is selected on INNER folds, never on the outer (evaluation) fold
# --------------------------------------------------------------------------- #
def test_nested_selection_uses_inner_not_outer():
    # Inner gains favor 'short'; outer gains favor 'long'. Nested k* must be 'short'.
    fold_gains = [
        {"inner": {"short": 0.5, "medium": 0.2, "long": 0.1},
         "outer": {"short": 0.30, "medium": 0.20, "long": 0.90}},
        {"inner": {"short": 0.6, "medium": 0.2, "long": 0.1},
         "outer": {"short": 0.25, "medium": 0.15, "long": 0.95}},
    ]
    stats = tsi_from_fold_gains(fold_gains, baseline="medium", n_boot=200)
    assert stats["k_star"] == "short"
    assert all(k == "short" for k in stats["k_star_per_fold"])
    # payoff uses outer G at the *inner-selected* scale (short), not outer argmax
    expected = np.mean([0.30 - 0.20, 0.25 - 0.15])
    assert abs(stats["payoff"] - expected) < 1e-9
    # in-sample (outer argmax = long) is more optimistic -> residual_optimism > 0
    assert stats["residual_optimism"] > 0


def test_tsi_rel_bounded_to_unit_interval_under_neg_baseline():
    # Under imperfect calibration G can dip below 0. The absolute TSI (payoff)
    # is reported on RAW G, but TSI_rel is the fraction of the maximum
    # discriminability and must stay in [0,1] -> it is computed on G truncated
    # at the 0 floor and clipped.
    fold_gains = [
        {"inner": {"short": 0.6, "medium": 0.2, "long": 0.1},
         "outer": {"short": 0.50, "medium": -0.30, "long": 0.10}}
        for _ in range(5)
    ]
    stats = tsi_from_fold_gains(fold_gains, baseline="medium", n_boot=200)
    # absolute payoff uses RAW G: 0.50 - (-0.30) = 0.80 (can exceed G_kstar)
    assert abs(stats["payoff"] - 0.80) < 1e-9
    # relative payoff on truncated G: (0.50 - 0) / 0.50 = 1.0 (bounded)
    assert 0.0 <= stats["tsi_rel"] <= 1.0
    assert abs(stats["tsi_rel"] - 1.0) < 1e-9
    lo, hi = stats["tsi_rel_ci"]
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_select_k_star_tiebreak_is_deterministic():
    # All equal -> first in SCALE_ORDER ('short').
    assert select_k_star({"short": 0.3, "medium": 0.3, "long": 0.3}) == "short"
    assert select_k_star({"short": 0.1, "medium": 0.5, "long": 0.5}) == "medium"


# --------------------------------------------------------------------------- #
# Gate tau and payoff CI: consistent k* definition across point / CI / gate
# --------------------------------------------------------------------------- #
def test_gate_threshold_is_upper_ci_bound():
    null = np.linspace(0.0, 0.10, 1001)  # symmetric ramp
    tau = gate_threshold(null, alpha=0.05)
    # upper 97.5 percentile of [0,0.1]
    assert abs(tau - 0.0975) < 1e-3


def test_gate_and_exploitability_consistency():
    # Strong, consistent positive payoff with G_kstar above tau -> exploitable.
    fold_gains = [
        {"inner": {"short": 0.6, "medium": 0.3, "long": 0.2},
         "outer": {"short": 0.55, "medium": 0.30, "long": 0.20}}
        for _ in range(8)
    ]
    stats = tsi_from_fold_gains(fold_gains, baseline="medium", n_boot=500)
    # G(f,k*) computed at nested k* == mean outer short == 0.55
    assert abs(stats["G_kstar"] - 0.55) < 1e-9
    gated = apply_gate(stats, tau=0.10)
    assert gated["gate_pass"] is True
    assert gated["tsi"] == stats["payoff"] > 0
    assert gated["exploitable"] is True

    # Now raise tau above G_kstar -> gate fails, TSI forced to 0, not exploitable.
    gated2 = apply_gate(stats, tau=0.90)
    assert gated2["gate_pass"] is False
    assert gated2["tsi"] == 0.0
    assert gated2["exploitable"] is False


def test_payoff_ci_lower_bound_governs_exploitable():
    # Payoff positive in mean but very noisy -> CI lower bound <= 0 -> not exploitable
    rng = np.random.RandomState(1)
    fold_gains = []
    for _ in range(10):
        noise = rng.normal(0, 0.5)
        fold_gains.append({
            "inner": {"short": 0.5, "medium": 0.3, "long": 0.2},
            "outer": {"short": 0.30 + noise, "medium": 0.30, "long": 0.20},
        })
    stats = tsi_from_fold_gains(fold_gains, baseline="medium", n_boot=1000)
    gated = apply_gate(stats, tau=0.0)
    lo, _ = stats["payoff_ci"]
    assert (gated["exploitable"]) == (lo > 0 and stats["G_kstar"] > 0.0)


# --------------------------------------------------------------------------- #
# Fusion: learned late-fusion weights are on the simplex (>=0, sum 1)
# --------------------------------------------------------------------------- #
def _synth_probas(n=200, C=3, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, C, n)
    onehot = np.eye(C)[y]
    # scale 0 is informative, scales 1,2 are noisy
    def mk(strength):
        logits = strength * onehot + rng.normal(0, 1, (n, C))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return [mk(4.0), mk(0.5), mk(0.2)], y


def test_learned_weights_on_simplex():
    probas, y = _synth_probas()
    w = learn_late_fusion_weights(probas, y, "multiclass", n_classes=3)
    assert w.shape == (3,)
    assert np.all(w >= -1e-9)
    assert abs(w.sum() - 1.0) < 1e-6
    # the informative scale should get the most weight
    assert np.argmax(w) == 0


def test_learned_weights_respect_prior_init_but_stay_simplex():
    probas, y = _synth_probas(seed=2)
    prior = np.array([0.2, 0.2, 0.6])
    w = learn_late_fusion_weights(probas, y, "multiclass", n_classes=3, prior=prior)
    assert np.all(w >= -1e-9) and abs(w.sum() - 1.0) < 1e-6


def test_tsi_prior_weights_normalized():
    tsi = {f: 0.0 for f in FEATURE_DIMS}
    tsi["mfcc"] = 0.4
    tsi["zcr"] = 0.2
    kmap = {f: "medium" for f in FEATURE_DIMS}
    kmap["mfcc"] = "short"
    kmap["zcr"] = "short"
    w = tsi_prior_weights(tsi, kmap, scales=("short", "medium", "long"))
    assert abs(w.sum() - 1.0) < 1e-9
    # short accumulates mfcc+zcr TSI -> largest weight
    assert np.argmax(w) == 0


def test_combine_late_is_weighted_average():
    P = [np.full((4, 3), 0.0), np.full((4, 3), 1.0)]
    P[0][:] = np.array([0.7, 0.2, 0.1])
    P[1][:] = np.array([0.1, 0.3, 0.6])
    mix = combine_late(P, [0.5, 0.5])
    assert np.allclose(mix, np.array([0.4, 0.25, 0.35]))
    assert np.allclose(mix.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# TSI-guided representation places each descriptor at its k*'s columns
# --------------------------------------------------------------------------- #
def test_tsi_guided_features_layout():
    n = 6
    feats = {
        "short": np.full((n, TRACK_DIM), 1.0),
        "medium": np.full((n, TRACK_DIM), 2.0),
        "long": np.full((n, TRACK_DIM), 3.0),
    }
    kmap = {f: "medium" for f in FEATURE_DIMS}
    kmap["mfcc"] = "short"
    kmap["zcr"] = "long"
    X = tsi_guided_features(feats, kmap)
    sl = descriptor_slices()
    assert np.allclose(X[:, sl["mfcc"]], 1.0)   # from short
    assert np.allclose(X[:, sl["zcr"]], 3.0)    # from long
    assert np.allclose(X[:, sl["chroma"]], 2.0)  # from medium (default)


# --------------------------------------------------------------------------- #
# End-to-end driver smoke test with a light sklearn classifier
# --------------------------------------------------------------------------- #
def test_driver_end_to_end_with_logreg():
    from sklearn.linear_model import LogisticRegression

    class LR:
        def __init__(self, input_dim, n_classes, task_type):
            self.m = LogisticRegression(max_iter=200)
            self.classes_full = np.arange(n_classes)
        def fit(self, X, y):
            self.m.fit(X, y); self.fitted_classes = self.m.classes_; return self
        def predict_proba(self, X):
            p = self.m.predict_proba(X)
            # expand to full class set if a fold misses a class
            full = np.zeros((X.shape[0], len(self.classes_full)))
            for j, c in enumerate(self.fitted_classes):
                full[:, c] = p[:, j]
            full = np.clip(full, 1e-9, None)
            return full / full.sum(axis=1, keepdims=True)

    from src.tsi import compute_fold_gains, tsi_from_fold_gains
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.RandomState(0)
    n, C = 120, 3
    y = rng.randint(0, C, n)
    # Build features where MFCC short carries the label signal.
    feats = {s: rng.normal(0, 1, (n, TRACK_DIM)) for s in ("short", "medium", "long")}
    sl = descriptor_slices()
    feats["short"][:, sl["mfcc"]] += np.eye(C)[y][:, :1] * 5.0  # signal in mfcc@short

    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(n), y))
    fg = compute_fold_gains(feats, y, folds, LR, "multiclass", C,
                            descriptors=["mfcc", "zcr"], n_inner=2, seed=0)
    assert set(fg.keys()) == {"mfcc", "zcr"}
    assert len(fg["mfcc"]) == 3
    for rec in fg["mfcc"]:
        assert set(rec["inner"]) == {"short", "medium", "long"}
        # gains are bounded above by 1
        assert all(v <= 1.0 + 1e-9 for v in rec["outer"].values())
    stats = tsi_from_fold_gains(fg["mfcc"], n_boot=200)
    assert "payoff" in stats and "payoff_ci" in stats


def test_mlp_self_provisions_validation_M4():
    # The MLP must derive its own validation split (early stopping + temperature
    # scaling) when fit is called without X_val/y_val, as the drivers do.
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: torch unavailable in this environment)")
        return
    from src.classifiers import MLPClassifier
    rng = np.random.RandomState(0)
    n, C, d = 120, 3, 16
    y = rng.randint(0, C, n)
    X = rng.normal(0, 1, (n, d)) + np.eye(C)[y] @ rng.normal(0, 2, (C, d))
    clf = MLPClassifier(n_classes=C, task_type="multiclass", epochs=20, patience=5)
    clf.fit(X, y)  # NO X_val/y_val passed
    # temperature scaling actually ran (would stay exactly 1.0 if no val set)
    assert clf.temperature != 1.0
    proba = clf.predict_proba(X)
    assert proba.shape == (n, C) and np.allclose(proba.sum(1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Protocol wiring: SVM on MTAT gets the paper's 50% training subsample via the
# factory (and only there). XGBoost/RF/other datasets are unaffected.
# --------------------------------------------------------------------------- #
def test_svm_mtat_factory_passes_subsample():
    from src.classifiers import make_clf_factory, SVMClassifier, MultiLabelWrapper

    # MTAT is multilabel -> the per-tag base SVM must carry subsample=0.5.
    clf = make_clf_factory("svm", dataset="mtat")(576, 50, "multilabel")
    assert isinstance(clf, MultiLabelWrapper)
    base = clf.base_factory()
    assert isinstance(base, SVMClassifier)
    assert base.subsample == 0.5

    # Same classifier on another dataset -> no subsample (full training set).
    other = make_clf_factory("svm", dataset="gtzan")(192, 10, "multiclass")
    assert isinstance(other, SVMClassifier)
    assert other.subsample is None

    # The knob is SVM-specific: XGBoost on MTAT must not receive a `subsample`
    # kwarg (XGBoostClassifier.__init__ would reject it).
    xgb_factory = make_clf_factory("xgb", dataset="mtat")
    try:
        from xgboost import XGBClassifier  # noqa: F401
    except ImportError:
        print("  (xgb subset skipped: xgboost unavailable)")
    else:
        xgb = xgb_factory(576, 50, "multilabel")
        assert isinstance(xgb, MultiLabelWrapper)
        xgb.base_factory()  # constructs fine -> no stray subsample kwarg


# --------------------------------------------------------------------------- #
# Protocol wiring: MLP on MTAT uses weighted BCE with per-tag inverse-frequency
# pos_weight derived from the TRAIN labels; multiclass tasks keep plain CE.
# --------------------------------------------------------------------------- #
def test_inverse_frequency_pos_weight_is_nontrivial_and_clipped():
    from src.classifiers import inverse_frequency_pos_weight

    # tag 0: 10% positives -> pos_weight = 90/10 = 9; tag 1: 50% -> 1; tag 2:
    # all-negative -> clipped to clip_hi (no positives in this train fold).
    y = np.zeros((100, 3), dtype=int)
    y[:10, 0] = 1
    y[:50, 1] = 1
    pw = inverse_frequency_pos_weight(y, clip_lo=1e-3, clip_hi=1e3)
    assert pw.shape == (3,)
    assert abs(pw[0] - 9.0) < 1e-4          # rare tag -> up-weighted
    assert abs(pw[1] - 1.0) < 1e-4          # balanced tag -> ~1
    assert pw[2] == 1e3                       # all-negative tag -> clipped high
    assert np.all(pw >= 1e-3) and np.all(pw <= 1e3)
    # genuinely non-trivial: the per-tag weights are not all equal
    assert len(np.unique(pw)) > 1


def test_svm_subsample_only_early_fusion():
    # Paper protocol: the 50% training subsample applies ONLY to SVM's 576-d
    # early-fusion representation on MTAT. Lower-dimensional fits (single-descriptor
    # TSI 4xdim, single-scale 192-d) must train on the full per-fold train set.
    from src.classifiers import make_clf_factory, SVMClassifier

    base = make_clf_factory("svm", dataset="mtat")(576, 50, "multilabel").base_factory()
    assert isinstance(base, SVMClassifier)
    assert base.subsample == 0.5

    # 576-d early fusion -> subsamples
    assert base._should_subsample(np.zeros((10, 576))) is True
    # every lower-dimensional representation -> full train set, no subsample
    for d in (4, 24, 80, 192, 575, 577):
        assert base._should_subsample(np.zeros((10, d))) is False

    # An SVM without the subsample knob set never subsamples, even at 576-d.
    plain = SVMClassifier()
    assert plain._should_subsample(np.zeros((10, 576))) is False


def test_mlp_mtat_factory_enables_weighted_bce():
    from src.classifiers import make_clf_factory, inverse_frequency_pos_weight

    # The MTAT factory must flip on auto_class_weights for the MLP; a multiclass
    # dataset must NOT (standard cross-entropy). Constructing the MLP needs torch,
    # so the construction/fit checks below are guarded.
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (mlp construction skipped: torch unavailable)")
        return

    from src.classifiers import MLPClassifier

    # MTAT + MLP -> weighted BCE enabled.
    clf = make_clf_factory("mlp", dataset="mtat")(80, 6, "multilabel")
    assert isinstance(clf, MLPClassifier)
    assert clf.auto_class_weights is True
    assert clf.class_weights is None  # not computed until fit sees the train fold

    # Imbalanced multilabel train fold -> after fit the MLP carries a non-trivial
    # per-tag pos_weight derived from THIS fold's training labels.
    rng = np.random.RandomState(0)
    n, T, d = 120, 6, 16
    y = (rng.rand(n, T) < np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])).astype(int)
    X = rng.normal(0, 1, (n, d))
    clf.fit(X, y)
    assert clf.class_weights is not None
    assert np.asarray(clf.class_weights).shape == (T,)
    assert len(np.unique(clf.class_weights)) > 1            # tag-dependent
    assert np.all(clf.class_weights >= 1e-3) and np.all(clf.class_weights <= 1e3)
    # rarer tags get larger pos_weight than the balanced one (monotone-ish)
    assert clf.class_weights[0] > clf.class_weights[-1]

    # Multiclass (e.g. GTZAN) -> no weighting; standard cross-entropy.
    mc = make_clf_factory("mlp", dataset="gtzan")(16, 4, "multiclass")
    assert mc.auto_class_weights is False
    yc = rng.randint(0, 4, n)
    mc.fit(X, yc)
    assert mc.class_weights is None


# --------------------------------------------------------------------------- #
# Optional progress reporting: strict no-op by default, never alters results.
# --------------------------------------------------------------------------- #
def test_progress_iter_noop_and_order_preserving():
    from src.progress import progress_iter
    items = [10, 20, 30, 40]
    # progress=False returns the SAME object untouched (zero overhead, no import).
    assert progress_iter(items, progress=False) is items
    # progress=True still yields the same values in the same order (tqdm-wrapped
    # if available, else a transparent fallback).
    assert list(progress_iter(range(4), progress=True, desc="t", total=4)) == [0, 1, 2, 3]
    assert list(progress_iter(iter(items), progress=True, leave=False)) == items


def test_progress_flag_does_not_change_compute_fold_gains():
    # Turning progress on must change ONLY the display, never the numbers.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from src.tsi import compute_fold_gains

    class LR:
        def __init__(self, input_dim, n_classes, task_type):
            self.m = LogisticRegression(max_iter=200)
            self.k = n_classes
        def fit(self, X, y):
            self.m.fit(X, y); self.fitted = self.m.classes_; return self
        def predict_proba(self, X):
            p = self.m.predict_proba(X)
            full = np.zeros((X.shape[0], self.k))
            for j, c in enumerate(self.fitted):
                full[:, int(c)] = p[:, j]
            full = np.clip(full, 1e-9, None)
            return full / full.sum(axis=1, keepdims=True)

    rng = np.random.RandomState(0)
    n, C = 90, 3
    y = rng.randint(0, C, n)
    feats = {s: rng.normal(0, 1, (n, TRACK_DIM)) for s in ("short", "medium", "long")}
    sl = descriptor_slices()
    feats["short"][:, sl["mfcc"]] += np.eye(C)[y][:, :1] * 4.0
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(n), y))

    common = dict(descriptors=["mfcc", "zcr"], n_inner=2, seed=0)
    fg_off = compute_fold_gains(feats, y, folds, LR, "multiclass", C, progress=False, **common)
    fg_on = compute_fold_gains(feats, y, folds, LR, "multiclass", C, progress=True,
                               desc="unit-test", **common)
    # identical gain records regardless of the progress flag
    for f in ("mfcc", "zcr"):
        for rec_off, rec_on in zip(fg_off[f], fg_on[f]):
            for kind in ("inner", "outer"):
                assert rec_off[kind] == rec_on[kind]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # pragma: no cover
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
