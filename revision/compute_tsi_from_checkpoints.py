"""Compute G, k*, tau, TSI from the downloaded XGBoost checkpoints.

Uses the exact paper methodology implemented in experiments/src/tsi.py:
  tsi_from_fold_gains -> gate_threshold -> apply_gate
Writes revision/tsi_computed.json for the HTML poster.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from src.tsi import tsi_from_fold_gains, gate_threshold, apply_gate  # noqa: E402
from src.stats import run_pairwise_comparisons  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints")
DATASETS = {
    "gtzan":     {"task": "Género (multiclase)",       "classes": "10 géneros"},
    "fma_small": {"task": "Género (multiclase)",       "classes": "8 géneros"},
    "mtat":      {"task": "Auto-tagging (multietiqueta)", "classes": "50 tags"},
    "irmas":     {"task": "Instrumento (multiclase, K=2)", "classes": "11 instr."},
}
DESC_ORDER = ["mfcc", "chroma", "spectral_centroid", "spectral_contrast",
              "spectral_rolloff", "zcr", "tonnetz"]


def load_gains(ds):
    g = json.load(open(os.path.join(CKPT, f"{ds}_xgb_gains.json")))
    meta, data = g["meta"], g["data"]
    scales = meta["scales"]
    fold_keys = sorted(data.keys(), key=lambda k: int(k))
    # {descriptor: [ {inner:{scale:G}, outer:{scale:G}}, ... per fold ]}
    fold_gains = {f: [] for f in meta["descriptors"]}
    for fk in fold_keys:
        for f in meta["descriptors"]:
            fold_gains[f].append(data[fk][f])
    return meta, scales, fold_gains


def load_null_flat(ds):
    n = json.load(open(os.path.join(CKPT, f"{ds}_xgb_null.json")))
    meta, d = n["meta"], n["data"]
    flat = []
    full_perms = 0
    for r in sorted(d.keys(), key=lambda k: int(k)):
        if len(d[r]) == meta["n_folds"]:
            full_perms += 1
        for fold in sorted(d[r].keys(), key=lambda k: int(k)):
            flat.extend(d[r][fold])
    meta = dict(meta)
    meta["full_perms"] = full_perms
    return flat, meta


# strategies compared statistically (the 5 paper strategies; learned_lf is a
# reference upper bound shown alongside but not part of the C(5,2) comparison)
FUSION_STRATEGIES = ["single_scale", "early", "late_uniform",
                     "tsi_guided", "tsi_weighted_lf"]
STRAT_LABELS = {
    "single_scale": "Una escala (mejor)",
    "early": "Fusión temprana (576-d)",
    "late_uniform": "Fusión tardía uniforme",
    "tsi_guided": "Selección guiada por TSI",
    "tsi_weighted_lf": "Fusión tardía ponderada (TSI prior)",
    "learned_lf": "Fusión tardía aprendida (ref., sin prior)",
}


def load_fusion(ds):
    """Per-strategy per-fold scores + paired Wilcoxon (Bonferroni) + Cliff's delta."""
    path = os.path.join(CKPT, f"{ds}_xgb_fusion.json")
    if not os.path.exists(path):
        return None
    f = json.load(open(path))
    meta, data = f["meta"], f["data"]
    strat = meta["strategies"]
    fold_keys = sorted(data.keys(), key=lambda k: int(k))
    scores = {s: [data[fk][s] for fk in fold_keys] for s in strat}
    n = len(fold_keys)
    summary = {}
    for s in strat:
        v = scores[s]
        mean = sum(v) / n
        var = sum((x - mean) ** 2 for x in v) / n
        summary[s] = {"mean": mean, "std": var ** 0.5, "label": STRAT_LABELS.get(s, s)}
    # paired comparison over the 5 canonical strategies (need >1 fold)
    comparisons = []
    if n > 1:
        comparisons = run_pairwise_comparisons(
            {s: scores[s] for s in FUSION_STRATEGIES}, alpha=0.05)
    best = max(strat, key=lambda s: summary[s]["mean"])
    return {
        "n_folds": n,
        "metric": "mAP" if meta.get("n_classes", 0) >= 50 else "F1-macro",
        "strategies_order": strat,
        "summary": summary,
        "best": best,
        "comparisons": comparisons,
        "prior_equals_learned": all(
            abs(scores["tsi_weighted_lf"][i] - scores["learned_lf"][i]) < 1e-9
            for i in range(n)) if "learned_lf" in scores else None,
    }


def load_calib(ds):
    """Mean ECE over folds (calibration quality of the medium-scale model)."""
    path = os.path.join(CKPT, f"{ds}_xgb_calib.json")
    if not os.path.exists(path):
        return None
    c = json.load(open(path))
    meta, data = c["meta"], c["data"]
    eces = [data[k]["ece"] for k in data if "ece" in data[k]]
    if not eces:
        return None
    mean = sum(eces) / len(eces)
    var = sum((x - mean) ** 2 for x in eces) / len(eces)
    return {"scale": meta.get("scale"), "n_bins": meta.get("n_bins"),
            "n_folds": len(eces), "ece_mean": mean, "ece_std": var ** 0.5}


def main():
    out = {}
    for ds, info in DATASETS.items():
        meta, scales, fold_gains = load_gains(ds)
        null_flat, nmeta = load_null_flat(ds)
        tau = gate_threshold(null_flat)
        n_folds = meta["n_folds"]

        desc_rows = {}
        for f in meta["descriptors"]:
            stats = tsi_from_fold_gains(
                fold_gains[f], baseline="medium", scales=scales,
                n_boot=1000, seed=42,
            )
            gated = apply_gate(stats, tau)
            desc_rows[f] = {
                "k_star": gated["k_star"],
                "G_kstar": gated["G_kstar"],
                "G_baseline": gated["G_baseline"],
                "gain_matrix": {k: gated["gain_matrix"][k] for k in scales},
                "tsi": gated["tsi"],
                "payoff": gated["payoff"],
                "payoff_ci": gated["payoff_ci"],
                "tsi_rel": gated.get("tsi_rel_reported"),
                "gate_pass": gated["gate_pass"],
                "exploitable": gated["exploitable"],
                "residual_optimism": gated["residual_optimism"],
                "payoff_loo": gated["payoff_loo"],
            }

        out[ds] = {
            "task": info["task"],
            "classes": info["classes"],
            "task_type": meta.get("task_type"),
            "scales": scales,
            "n_folds": n_folds,
            "n_null": len(null_flat),
            "n_perm": nmeta.get("n_permutations"),
            "null_full_perms": nmeta.get("full_perms"),
            "tau_reliable": nmeta.get("full_perms", 0) >= 0.9 * nmeta.get("n_permutations", 1),
            "tau": tau,
            "descriptors": desc_rows,
            "desc_order": [d for d in DESC_ORDER if d in meta["descriptors"]],
            "fusion": load_fusion(ds),
            "calibration": load_calib(ds),
        }
        # console summary
        print(f"\n==== {ds.upper()}  ({info['task']})  tau={tau:.4f}  "
              f"folds={n_folds}  null_n={len(null_flat)} ====")
        print(f"{'descriptor':<18}{'k*':<8}{'G(k*)':>9}{'G(2s)':>9}"
              f"{'TSI':>9}{'CI_lo':>9}{'CI_hi':>9}  expl")
        for f in out[ds]["desc_order"]:
            r = desc_rows[f]
            lo, hi = r["payoff_ci"]
            print(f"{f:<18}{r['k_star']:<8}{r['G_kstar']:>9.4f}{r['G_baseline']:>9.4f}"
                  f"{r['tsi']:>9.4f}{lo:>9.4f}{hi:>9.4f}  {'YES' if r['exploitable'] else '-'}")
        fu = out[ds]["fusion"]
        if fu:
            print(f"  -- fusion ({fu['metric']}, {fu['n_folds']} folds) "
                  f"best={fu['best']} --")
            for s in fu["strategies_order"]:
                m = fu["summary"][s]
                print(f"     {s:18s} {m['mean']:.4f} ± {m['std']:.4f}")
        ca = out[ds]["calibration"]
        if ca:
            print(f"  -- ECE ({ca['scale']}, {ca['n_folds']} folds): "
                  f"{ca['ece_mean']:.4f} ± {ca['ece_std']:.4f} --")

    dst = os.path.join(ROOT, "revision", "tsi_computed.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nWrote {dst}")


if __name__ == "__main__":
    main()
