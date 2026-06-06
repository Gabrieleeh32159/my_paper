"""
Generate small EXAMPLE result JSONs from synthetic data (NOT real experiments).

These illustrate the exact schema that ``02_tsi_and_fusion.ipynb`` writes and
that ``show.py`` consumes, so the repo carries runnable, versioned examples
without depending on Google Drive features. Real runs overwrite these from the
notebook. Run:  python results/_gen_example_results.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.features import FEATURE_DIMS, TRACK_DIM, descriptor_slices
from src.tsi import compute_fold_gains, tsi_from_fold_gains, apply_gate, gate_threshold
from src.tsi import permutation_null_kstar_gains
from src.fusion import evaluate_fusion_cv
from src.stats import scale_difference_test, run_pairwise_comparisons
from src.evaluation import calibration_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class LRClf:
    """Calibrated logistic-regression stand-in mirroring the real pipeline.

    Uses ``CalibratedClassifierCV(sigmoid, cv=3)`` -- the same calibration the
    paper mandates for XGBoost/RF -- so example ``G`` values stay bounded in the
    interpretable range rather than blowing up from over-confident log-loss.
    """

    def __init__(self, input_dim, n_classes, task_type):
        base = LogisticRegression(max_iter=300)
        self.m = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self.n_classes = n_classes

    def fit(self, X, y):
        # guard tiny inner folds where a class may have <3 samples
        import numpy as _np
        _, counts = _np.unique(y, return_counts=True)
        if counts.min() < 3:
            self.m = LogisticRegression(max_iter=300)
        self.m.fit(X, y)
        self.fitted = self.m.classes_
        return self

    def predict_proba(self, X):
        p = self.m.predict_proba(X)
        full = np.zeros((X.shape[0], self.n_classes))
        for j, c in enumerate(self.fitted):
            full[:, int(c)] = p[:, j]
        full = np.clip(full, 1e-9, None)
        return full / full.sum(axis=1, keepdims=True)


def synth_dataset(seed=0, n=180, C=3):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, C, n)
    onehot = np.eye(C)[y]
    feats = {s: rng.normal(0, 1, (n, TRACK_DIM)) for s in ("short", "medium", "long")}
    sl = descriptor_slices()
    # MFCC most discriminative at SHORT, chroma at MEDIUM -> nontrivial k* pattern
    feats["short"][:, sl["mfcc"]] += onehot[:, :1] * 4.0
    feats["medium"][:, sl["chroma"]] += onehot[:, :1] * 3.0
    return feats, y, C


def build_tsi(feats, y, C, folds, scales=("short", "medium", "long"),
              n_boot=300, n_perm=20):
    scales = tuple(scales)
    fg = compute_fold_gains(feats, y, folds, LRClf, "multiclass", C,
                            scales=scales, n_inner=2, seed=0)
    null = permutation_null_kstar_gains(feats, y, folds, LRClf, "multiclass", C,
                                        scales=scales, n_permutations=n_perm, seed=7)
    tau = gate_threshold(null)
    descriptors, scale_diff, tsi_by_desc = {}, {}, {}
    for f in FEATURE_DIMS:
        stats = tsi_from_fold_gains(fg[f], scales=scales, n_boot=n_boot, seed=0)
        gated = apply_gate(stats, tau)
        # JSON-safe
        gated["payoff_ci"] = list(gated["payoff_ci"])
        gated["tsi_rel_ci"] = list(gated["tsi_rel_ci"])
        gated.pop("k_star_per_fold", None)
        gated.pop("payoff_per_fold", None)
        for key in ("tsi_rel", "tsi_rel_reported"):
            if not np.isfinite(gated.get(key, np.nan)):
                gated[key] = None
        descriptors[f] = gated
        tsi_by_desc[f] = gated["tsi"]
        per_fold = {k: [rec["outer"][k] for rec in fg[f]] for k in scales}
        scale_diff[f] = scale_difference_test(per_fold, scales=scales)
    return fg, tau, descriptors, scale_diff, tsi_by_desc


def main():
    tsi_results, exp_results = {}, {}
    # Two synthetic "datasets" to show cross-dataset structure. IRMAS is K=2:
    # the long scale collapses, so the example follows the (short, medium) path
    # (gain matrix with no long column, Friedman -> Wilcoxon).
    scales_by_ds = {"gtzan_example": ("short", "medium", "long"),
                    "irmas_example": ("short", "medium")}
    for ds, seed in [("gtzan_example", 0), ("irmas_example", 1)]:
        feats, y, C = synth_dataset(seed=seed)
        scales = scales_by_ds[ds]
        folds = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
        fg, tau, descriptors, scale_diff, tsi_by_desc = build_tsi(
            feats, y, C, folds, scales=scales)

        tsi_results[ds] = {
            "tau": tau, "baseline": "medium",
            "scales": list(scales),
            "descriptors": descriptors, "scale_difference": scale_diff,
        }

        # fusion
        fres = evaluate_fusion_cv(feats, y, folds, LRClf, "multiclass", C,
                                  fold_gains=fg, tsi_by_descriptor=tsi_by_desc,
                                  scales=scales, n_inner=2, seed=0)
        strategies = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                          "per_fold": [float(x) for x in v]}
                      for k, v in fres.items()}
        compare = {k: v for k, v in fres.items() if k != "learned_lf"}
        pairwise = run_pairwise_comparisons(compare)
        # ECE averaged over folds + pooled reliability curve (M-3)
        calib = calibration_report(feats, y, folds, LRClf, "multiclass", C, scale="medium")
        exp_results[ds] = {"logreg_example": {
            "strategies": strategies, "pairwise": pairwise,
            "ece": calib["ece_mean"], "ece_std": calib["ece_std"], "calibration": calib}}

    run_config = {
        "_note": "EXAMPLE/synthetic run — overwritten by 02_tsi_and_fusion.ipynb",
        "seed": 42, "primary_clf": "logreg_example (stand-in)",
        "classifiers": ["logreg_example"], "n_inner": 2, "n_boot": 300, "n_perm": 20,
        "baseline": "medium",
        "scales_by_dataset": {"gtzan_example": ["short", "medium", "long"],
                              "irmas_example": ["short", "medium"]},
        "fold_scheme": {ds: "4-fold stratified CV (synthetic)"
                        for ds in ("gtzan_example", "irmas_example")},
        "classifier_hyperparams": {
            "logreg_example": {"calibration": "sigmoid/cv3",
                               "note": "real runs use XGBoost n_estimators=500, "
                                       "max_depth=6, lr=0.1"}},
    }
    with open(os.path.join(SCRIPT_DIR, "tsi_results.json"), "w") as f:
        json.dump(tsi_results, f, indent=2)
    with open(os.path.join(SCRIPT_DIR, "experiment_results.json"), "w") as f:
        json.dump(exp_results, f, indent=2)
    with open(os.path.join(SCRIPT_DIR, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print("Wrote example tsi_results.json, experiment_results.json, run_config.json")


if __name__ == "__main__":
    main()
