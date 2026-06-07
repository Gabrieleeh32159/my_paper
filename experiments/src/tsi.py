"""
Temporal Sensitivity Index (TSI) -- *redesigned payoff* definition.

This module implements the TSI exactly as specified in ``paper/proposal.tex``
(Sec. "Indice de Sensibilidad Temporal"). It is a *payoff* metric, NOT a
dispersion statistic:

    G(f, k)    = 1 - L(f, k) / L_chance            # normalized information gain
    k*(f)      = argmax_k G(f, k)   via NESTED selection (inner folds)
    TSI(f)     = [ G(f, k*) - G(f, k_bar) ] * 1[ G(f, k*) > tau ]
    TSI_rel(f) = ( G(f, k*) - G(f, k_bar) ) / G(f, k*)   # secondary, gated only

with convention baseline ``k_bar = 'medium'`` (2 s).

Design contract (paper): ``k*`` is defined ONCE, by nested selection, and used
*identically* in the point estimate, the bootstrap CI, the informativeness gate
``tau`` and the fusion strategies. The single source of truth for that choice is
:func:`select_k_star`.

The module is split into:

* **Pure statistics** over per-fold gain records (:func:`tsi_from_fold_gains`,
  :func:`gate_threshold`, :func:`bootstrap_ci`, :func:`select_k_star`) -- these
  are deterministic and unit-tested on synthetic data, with no Drive/classifier
  dependency.
* **A driver** (:func:`compute_fold_gains`, :func:`compute_tsi_cv_full`) that
  trains calibrated classifiers per descriptor / scale / fold to *produce* those
  gain records on real features.

The legacy dispersion metric ``std_k G`` ("TSD") is intentionally absent.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Callable, Sequence

from .features import FEATURE_DIMS, extract_descriptor
from .evaluation import gain_from_predictions, truncate_gain
from .progress import progress_iter
from .checkpoint import load_progress, save_progress

SCALE_ORDER = ["short", "medium", "long"]
BASELINE = "medium"          # k_bar: the 2 s convention the TSI questions


# --------------------------------------------------------------------------- #
# The single definition of k* (used everywhere: point, CI, gate, fusion)
# --------------------------------------------------------------------------- #
def select_k_star(gains_by_scale: Dict[str, float]) -> str:
    """argmax_k G(f, k). Deterministic tie-break following ``SCALE_ORDER``.

    This is the ONLY place k* is chosen, so the point estimate, the CI, the gate
    and the fusion strategies all share an identical notion of the optimal scale.
    """
    scales = [s for s in SCALE_ORDER if s in gains_by_scale]
    scales += [s for s in gains_by_scale if s not in scales]
    best, best_val = scales[0], gains_by_scale[scales[0]]
    for s in scales[1:]:
        if gains_by_scale[s] > best_val:
            best, best_val = s, gains_by_scale[s]
    return best


# --------------------------------------------------------------------------- #
# Bootstrap CI over folds
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
):
    """Percentile bootstrap CI of ``statistic`` (default: the mean) over folds.

    Returns ``(low, high)`` at the ``(alpha/2, 1-alpha/2)`` percentiles.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    n = len(values)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.randint(0, n, size=n)]
        boot[b] = statistic(sample)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Pure TSI statistics from per-fold gain records
# --------------------------------------------------------------------------- #
def tsi_from_fold_gains(
    fold_gains: List[Dict[str, Dict[str, float]]],
    baseline: str = BASELINE,
    scales: Sequence[str] = tuple(SCALE_ORDER),
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict:
    """Compute the (un-gated) TSI statistics for one descriptor.

    Parameters
    ----------
    fold_gains : list of dict
        One entry per outer fold, each ``{'inner': {scale: G}, 'outer': {scale: G}}``.
        ``inner`` gains come from inner folds of the outer-train set (used ONLY to
        pick k*); ``outer`` gains are evaluated on the held-out outer fold.
    baseline : str
        Convention scale ``k_bar`` (default ``'medium'``).

    Returns
    -------
    dict with keys:
        ``k_star``            -- modal nested k* across folds (for reporting)
        ``k_star_per_fold``   -- list of per-fold nested k*
        ``G_kstar``           -- mean outer G at the per-fold nested k* (= G(f,k*))
        ``G_baseline``        -- mean outer G at the baseline scale
        ``payoff``            -- point TSI before gating = mean(G(k*_inner)-G(k_bar))
        ``payoff_per_fold``   -- the per-fold payoffs (for stats / Friedman inputs)
        ``payoff_ci``         -- 95% bootstrap CI of the payoff
        ``tsi_rel``           -- relative payoff in [0,1]: (G+(k*)-G+(k_bar))/G+(k*)
                                 on G truncated at the 0 floor (so it stays bounded
                                 under imperfect calibration), NaN if G+(k*)<=0
        ``tsi_rel_ci``        -- bootstrap CI of the relative payoff
        ``payoff_insample``   -- in-sample payoff (argmax on outer gains: optimistic)
        ``residual_optimism`` -- payoff_insample - payoff (nested)
        ``payoff_loo``        -- robustness baseline: G(k*) - mean_{k!=k*} G(k)
        ``gain_matrix``       -- {scale: {'mean':, 'std':} } truncated for display
    """
    scales = [s for s in scales]
    inner = [fg["inner"] for fg in fold_gains]
    outer = [fg["outer"] for fg in fold_gains]

    # --- nested k* (one definition) ---
    kstar_inner = [select_k_star(g) for g in inner]

    payoffs, g_kstar_vals, g_base_vals, loo_vals = [], [], [], []
    for o, ks in zip(outer, kstar_inner):
        payoffs.append(o[ks] - o[baseline])
        g_kstar_vals.append(o[ks])
        g_base_vals.append(o[baseline])
        others = [o[k] for k in scales if k != ks]
        loo = o[ks] - (np.mean(others) if others else o[ks])
        loo_vals.append(loo)

    payoffs = np.asarray(payoffs, dtype=float)
    g_kstar = float(np.mean(g_kstar_vals))
    g_base = float(np.mean(g_base_vals))
    point_payoff = float(np.mean(payoffs))

    ci = bootstrap_ci(payoffs, n_boot=n_boot, seed=seed)

    # in-sample optimism: select k* directly on the outer (evaluation) gains
    kstar_insample = [select_k_star(o) for o in outer]
    payoff_insample = float(
        np.mean([o[ks] - o[baseline] for o, ks in zip(outer, kstar_insample)])
    )

    # relative payoff (secondary): the fraction of the maximum discriminability,
    # so it must live in [0, 1]. Under imperfect calibration G can dip below 0,
    # which would push the raw ratio payoff/G(k*) above 1 (e.g. a negative
    # baseline). We therefore report TSI_rel on G truncated at the 0 floor
    # (G+ = max(G, 0)) in both numerator and denominator, and clip to [0, 1] for
    # numerical safety. The absolute TSI (``payoff``) is left on RAW G, unchanged.
    g_kstar_pos = float(np.mean([truncate_gain(o[ks]) for o, ks in zip(outer, kstar_inner)]))
    g_base_pos = float(np.mean([truncate_gain(o[baseline]) for o in outer]))
    if g_kstar_pos > 0:
        tsi_rel = float(np.clip((g_kstar_pos - g_base_pos) / g_kstar_pos, 0.0, 1.0))
        rel_per_fold = [
            float(np.clip(
                (truncate_gain(o[ks]) - truncate_gain(o[baseline])) / truncate_gain(o[ks]),
                0.0, 1.0))
            if truncate_gain(o[ks]) > 0 else np.nan
            for o, ks in zip(outer, kstar_inner)
        ]
        rel_clean = np.asarray([r for r in rel_per_fold if np.isfinite(r)], dtype=float)
        tsi_rel_ci = bootstrap_ci(rel_clean, n_boot=n_boot, seed=seed) if len(rel_clean) else (np.nan, np.nan)
    else:
        tsi_rel = np.nan
        tsi_rel_ci = (np.nan, np.nan)

    # gain matrix cell stats (display-truncated to the [0,1] convention)
    gain_matrix = {}
    for k in scales:
        vals = np.asarray([truncate_gain(o[k]) for o in outer], dtype=float)
        gain_matrix[k] = {"mean": float(vals.mean()), "std": float(vals.std())}

    # modal k* for a single reported optimal scale
    vals, counts = np.unique(kstar_inner, return_counts=True)
    k_star_modal = str(vals[int(np.argmax(counts))])

    return {
        "k_star": k_star_modal,
        "k_star_per_fold": kstar_inner,
        "G_kstar": g_kstar,
        "G_baseline": g_base,
        "payoff": point_payoff,
        "payoff_per_fold": payoffs.tolist(),
        "payoff_ci": ci,
        "tsi_rel": tsi_rel,
        "tsi_rel_ci": tsi_rel_ci,
        "payoff_insample": payoff_insample,
        "residual_optimism": payoff_insample - point_payoff,
        "payoff_loo": float(np.mean(loo_vals)),
        "gain_matrix": gain_matrix,
    }


# --------------------------------------------------------------------------- #
# Informativeness gate tau (per task)
# --------------------------------------------------------------------------- #
def gate_threshold(null_kstar_gains: Sequence[float], alpha: float = 0.05) -> float:
    """tau = upper bound of the 95% CI of G(f,k*) under the permutation null.

    The null sample must be generated by permuting labels and *re-selecting the
    argmax within each replicate* (see :func:`permutation_null_kstar_gains`), so
    that it embeds the same maximization bias as the point estimate.
    """
    null = np.asarray(null_kstar_gains, dtype=float)
    if len(null) == 0:
        return 0.0
    return float(np.percentile(null, 100 * (1 - alpha / 2)))


def apply_gate(stats: Dict, tau: float) -> Dict:
    """Apply the informativeness gate and exploitability rule to one descriptor.

    ``TSI = payoff * 1[G(f,k*) > tau]``; *temporally exploitable* iff the lower
    bound of the payoff CI ``> 0`` AND ``G(f,k*) > tau``.
    """
    gated = stats["G_kstar"] > tau
    tsi = stats["payoff"] if gated else 0.0
    ci_lo, _ = stats["payoff_ci"]
    exploitable = bool(gated and ci_lo > 0)
    out = dict(stats)
    out.update(
        {
            "tau": float(tau),
            "gate_pass": bool(gated),
            "tsi": float(tsi),
            "tsi_rel_reported": float(stats["tsi_rel"]) if gated and np.isfinite(stats["tsi_rel"]) else np.nan,
            "exploitable": exploitable,
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Classifier-training driver: produce per-fold gain records on real features
# --------------------------------------------------------------------------- #
def _inner_splits(y: np.ndarray, train_idx: np.ndarray, n_inner: int,
                  task_type: str, seed: int):
    """Inner CV splits over the outer-train indices (Stratified for multiclass)."""
    from sklearn.model_selection import StratifiedKFold, KFold
    train_idx = np.asarray(train_idx)
    if task_type == "multilabel":
        splitter = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
        inner = list(splitter.split(train_idx))
    else:
        splitter = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        inner = list(splitter.split(train_idx, np.asarray(y)[train_idx]))
    # map back to absolute indices
    return [(train_idx[a], train_idx[b]) for a, b in inner]


def _fit_predict_gain(clf_factory, X, y, tr, ev, task_type, n_classes):
    """Train one calibrated classifier on ``tr``, score ``G`` on ``ev``."""
    clf = clf_factory(input_dim=X.shape[1], n_classes=n_classes, task_type=task_type)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[ev])
    return gain_from_predictions(y[tr], y[ev], proba, task_type, n_classes=n_classes)


def compute_fold_gains(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    outer_folds: List,
    clf_factory: Callable,
    task_type: str,
    n_classes: int,
    descriptors: Optional[Sequence[str]] = None,
    scales: Sequence[str] = tuple(SCALE_ORDER),
    n_inner: int = 3,
    seed: int = 42,
    progress: bool = False,
    desc: Optional[str] = None,
    checkpoint_path=None,
) -> Dict[str, List[Dict[str, Dict[str, float]]]]:
    """Train classifiers to produce nested per-fold gain records per descriptor.

    For every descriptor and outer fold this computes, for each scale:
      * ``inner`` gain = mean over inner folds of the outer-train set (selects k*),
      * ``outer`` gain = G on the held-out outer fold (a single classifier per
        scale trained on the full outer-train set).

    Returns ``{descriptor: [ {'inner': {scale:G}, 'outer': {scale:G}}, ... ]}``
    ready for :func:`tsi_from_fold_gains`.

    Set ``progress=True`` for nested tqdm bars (outer over folds, inner over
    descriptors); ``desc`` labels them (e.g. the dataset name). Default is silent.

    Pass ``checkpoint_path`` (a file on persistent storage, e.g. Drive) to make the
    sweep crash-safe: progress is saved after every ``(fold, descriptor)`` and a
    re-run resumes from disk, skipping completed units (each fold seeds its inner
    splits with ``seed + fold_i``, so resumed folds are bit-identical).
    """
    descriptors = list(descriptors) if descriptors is not None else list(FEATURE_DIMS.keys())
    scales = [s for s in scales if s in features]
    y = np.asarray(y)

    # pre-slice each descriptor at each scale once
    desc_X = {
        f: {k: extract_descriptor(features[k], f) for k in scales}
        for f in descriptors
    }

    label = desc or "gains"
    folds = list(enumerate(outer_folds))
    # checkpoint store: {str(fold_i): {descriptor: {"inner":..., "outer":...}}}
    meta = {"kind": "compute_fold_gains", "descriptors": descriptors, "scales": scales,
            "n_inner": n_inner, "seed": seed, "n_folds": len(folds),
            "task_type": task_type, "n_classes": n_classes}
    store = load_progress(checkpoint_path, meta)

    for fold_i, (train_idx, eval_idx) in progress_iter(
            folds, progress, desc=f"{label} | gains [folds]", total=len(folds)):
        fold_done = store.setdefault(str(fold_i), {})
        if all(f in fold_done for f in descriptors):
            continue  # fold fully cached -> skip (no inner-split work needed)
        train_idx = np.asarray(train_idx)
        eval_idx = np.asarray(eval_idx)
        inner = _inner_splits(y, train_idx, n_inner, task_type, seed + fold_i)
        for f in progress_iter(descriptors, progress,
                               desc=f"  fold {fold_i + 1}/{len(folds)} [desc]",
                               leave=False):
            if f in fold_done:
                continue  # already computed on a previous run
            inner_g, outer_g = {}, {}
            for k in scales:
                X = desc_X[f][k]
                # inner gains (nested k* selection)
                ig = [
                    _fit_predict_gain(clf_factory, X, y, itr, iev, task_type, n_classes)
                    for itr, iev in inner
                ]
                inner_g[k] = float(np.mean(ig))
                # outer gain on held-out fold
                outer_g[k] = _fit_predict_gain(
                    clf_factory, X, y, train_idx, eval_idx, task_type, n_classes
                )
            fold_done[f] = {"inner": inner_g, "outer": outer_g}
            save_progress(checkpoint_path, meta, store)

    # assemble the public return shape in fold order, preserving descriptor order
    return {f: [store[str(i)][f] for i in range(len(folds))] for f in descriptors}


def permutation_null_kstar_gains(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    outer_folds: List,
    clf_factory: Callable,
    task_type: str,
    n_classes: int,
    descriptors: Optional[Sequence[str]] = None,
    scales: Sequence[str] = tuple(SCALE_ORDER),
    n_permutations: int = 100,
    seed: int = 1234,
    progress: bool = False,
    desc: Optional[str] = None,
    checkpoint_path=None,
) -> List[float]:
    """Per-task null sample of G(f,k*) by permuting labels and re-selecting argmax.

    For each replicate, labels are permuted, a single classifier per scale is
    trained per outer fold, and ``G(f,k*)`` is taken with ``k* = argmax_k G`` over
    the permuted-label gains -- so the null embeds the maximization bias. Gains are
    pooled across descriptors and folds into one per-task distribution feeding
    :func:`gate_threshold`. (Costly; the notebook scopes ``n_permutations``.)

    Set ``progress=True`` for a tqdm bar over permutation replicates (the slowest,
    most opaque phase); ``desc`` labels it. Default is silent.

    Pass ``checkpoint_path`` to make this (the dominant cost) crash-safe: progress
    is saved after every ``(permutation, fold)`` and a re-run resumes from disk.
    Each replicate is seeded independently (``RandomState(seed + r)``) so a resumed
    replicate reproduces exactly regardless of how many ran before the crash.
    """
    descriptors = list(descriptors) if descriptors is not None else list(FEATURE_DIMS.keys())
    scales = [s for s in scales if s in features]
    y = np.asarray(y)
    folds = list(outer_folds)
    desc_X = {
        f: {k: extract_descriptor(features[k], f) for k in scales}
        for f in descriptors
    }
    label = desc or "gate"
    # checkpoint store: {str(r): {str(fold_i): [G(f,k*) per descriptor, in order]}}
    meta = {"kind": "permutation_null_kstar_gains", "descriptors": descriptors,
            "scales": scales, "n_permutations": n_permutations, "seed": seed,
            "n_folds": len(folds), "task_type": task_type, "n_classes": n_classes}
    store = load_progress(checkpoint_path, meta)

    for r in progress_iter(range(n_permutations), progress,
                           desc=f"{label} | gate τ [perm]", total=n_permutations):
        perm_done = store.setdefault(str(r), {})
        if all(str(fi) in perm_done for fi in range(len(folds))):
            continue  # replicate fully cached -> skip (no permutation/fit work)
        # per-replicate seed: a resumed replicate is bit-identical to a fresh run
        rng = np.random.RandomState(seed + r)
        y_perm = y[rng.permutation(len(y))]
        for fold_i, (train_idx, eval_idx) in enumerate(folds):
            if str(fold_i) in perm_done:
                continue
            train_idx = np.asarray(train_idx)
            eval_idx = np.asarray(eval_idx)
            fold_gains = []
            for f in descriptors:
                g = {}
                for k in scales:
                    X = desc_X[f][k]
                    g[k] = _fit_predict_gain(
                        clf_factory, X, y_perm, train_idx, eval_idx, task_type, n_classes
                    )
                ks = select_k_star(g)
                fold_gains.append(g[ks])
            perm_done[str(fold_i)] = fold_gains
            save_progress(checkpoint_path, meta, store)

    # flatten in (permutation, fold, descriptor) order -> the public return shape
    null_gains: List[float] = []
    for r in range(n_permutations):
        for fold_i in range(len(folds)):
            null_gains.extend(store[str(r)][str(fold_i)])
    return null_gains


# --------------------------------------------------------------------------- #
# High-level orchestration for one dataset
# --------------------------------------------------------------------------- #
def compute_tsi_cv_full(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    outer_folds: List,
    clf_factory: Callable,
    task_type: str,
    n_classes: int,
    descriptors: Optional[Sequence[str]] = None,
    scales: Sequence[str] = tuple(SCALE_ORDER),
    baseline: str = BASELINE,
    n_inner: int = 3,
    n_boot: int = 1000,
    n_permutations: int = 100,
    seed: int = 42,
    progress: bool = False,
    desc: Optional[str] = None,
) -> Dict:
    """Full per-dataset TSI: gain records -> stats -> per-task gate -> exploitability.

    Returns a dict keyed by descriptor with the gated TSI statistics, plus a
    top-level ``'tau'`` and ``'baseline'``. ``scales`` may be length 2 (IRMAS,
    K=2) -- the same code path applies; the baseline must be among ``scales``.

    ``progress``/``desc`` are forwarded to the gain-record and permutation-null
    drivers for tqdm progress bars (silent by default).
    """
    descriptors = list(descriptors) if descriptors is not None else list(FEATURE_DIMS.keys())
    fold_gains = compute_fold_gains(
        features, y, outer_folds, clf_factory, task_type, n_classes,
        descriptors=descriptors, scales=scales, n_inner=n_inner, seed=seed,
        progress=progress, desc=desc,
    )
    null = permutation_null_kstar_gains(
        features, y, outer_folds, clf_factory, task_type, n_classes,
        descriptors=descriptors, scales=scales, n_permutations=n_permutations, seed=seed + 7,
        progress=progress, desc=desc,
    )
    tau = gate_threshold(null)

    out = {"tau": tau, "baseline": baseline, "descriptors": {}}
    for f in descriptors:
        stats = tsi_from_fold_gains(
            fold_gains[f], baseline=baseline, scales=scales, n_boot=n_boot, seed=seed,
        )
        out["descriptors"][f] = apply_gate(stats, tau)
    return out
