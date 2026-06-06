"""
Pretty-print the redesigned TSI / fusion results as Markdown tables.

Reads ``tsi_results.json`` and ``experiment_results.json`` from this directory
and prints:
  * the per-dataset TSI table (payoff, 95% CI, k*, gate, exploitable + 7x3 gain),
  * the per-dataset residual-optimism diagnostic (in-sample vs nested TSI),
  * the per-dataset fusion table (5 strategies + references, per classifier),
  * the Bonferroni-corrected pairwise significance between strategies.

Schema is documented in ../README.md. Run:  python results/show.py
"""

import json
import os

try:
    import pandas as pd
    _HAVE_PD = True
except Exception:
    _HAVE_PD = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALES = ("short", "medium", "long")
STRATEGY_LABELS = {
    "single_scale": "Single",
    "early": "Early",
    "late_uniform": "Late(unif)",
    "tsi_guided": "TSI-guided",
    "tsi_weighted_lf": "TSI-LF",
    "learned_lf": "Learned-LF(ref)",
}


def _table(rows):
    if not rows:
        return "(no data)\n"
    if _HAVE_PD:
        return pd.DataFrame(rows).to_markdown(index=False) + "\n"
    # minimal fallback without pandas
    cols = list(rows[0].keys())
    out = "| " + " | ".join(cols) + " |\n"
    out += "| " + " | ".join("---" for _ in cols) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out


def show_tsi(tsi_data):
    out = ""
    for dataset, info in tsi_data.items():
        tau = info.get("tau", float("nan"))
        baseline = info.get("baseline", "medium")
        descs = info.get("descriptors", {})
        rows = []
        for f, d in descs.items():
            lo, hi = d.get("payoff_ci", (float("nan"), float("nan")))
            row = {
                "Descriptor": f,
                "TSI": f"{d.get('tsi', 0.0):.4f}",
                "Payoff": f"{d.get('payoff', 0.0):.4f}",
                "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                "k*": d.get("k_star", "—"),
                "G(k*)": f"{d.get('G_kstar', 0.0):.3f}",
                "G(k̄)": f"{d.get('G_baseline', 0.0):.3f}",
                "Gate": "✓" if d.get("gate_pass") else "—",
                "Exploit": "✓" if d.get("exploitable") else "—",
                "ResOpt": f"{d.get('residual_optimism', 0.0):.3f}",
            }
            gm = d.get("gain_matrix", {})
            for s in SCALES:
                cell = gm.get(s)
                if cell is None:
                    continue
                row[f"G[{s[0]}]"] = f"{cell['mean']:.3f}±{cell['std']:.3f}"
            rows.append(row)
        out += f"### {dataset.upper()} — TSI (baseline k̄={baseline}, τ={tau:.4f})\n\n"
        out += _table(rows) + "\n"

        fried = info.get("scale_difference", {})
        if fried:
            frows = []
            for f, t in fried.items():
                frows.append({
                    "Descriptor": f, "Test": t.get("test", ""),
                    "p": f"{t.get('p_value', float('nan')):.4f}",
                    "Differ?": "✓" if t.get("significant") else "—",
                })
            out += f"#### {dataset.upper()} — scale-difference test (Friedman/Wilcoxon)\n\n"
            out += _table(frows) + "\n"
    return out


def show_residual_optimism(tsi_data):
    """Honest-estimator diagnostic: in-sample vs nested TSI payoff per descriptor.

    The paper requires reporting the **residual optimism** — how much the payoff
    shrinks when the optimal scale ``k*`` is chosen by nested selection (evaluated
    out-of-fold) instead of in-sample (argmax on the same evaluation gains). A
    large gap flags an over-optimistic estimate. One small table per dataset (the
    TSI is computed for the primary classifier only).
    """
    out = ""
    for dataset, info in tsi_data.items():
        descs = info.get("descriptors", {})
        rows = []
        for f, d in descs.items():
            rows.append({
                "Descriptor": f,
                "TSI in-sample": f"{d.get('payoff_insample', 0.0):.4f}",
                "TSI nested": f"{d.get('payoff', 0.0):.4f}",
                "Residual optimism": f"{d.get('residual_optimism', 0.0):.4f}",
            })
        if descs:
            n = len(descs)
            mean_in = sum(d.get("payoff_insample", 0.0) for d in descs.values()) / n
            mean_nested = sum(d.get("payoff", 0.0) for d in descs.values()) / n
            mean_ro = sum(d.get("residual_optimism", 0.0) for d in descs.values()) / n
            rows.append({
                "Descriptor": "**mean**",
                "TSI in-sample": f"{mean_in:.4f}",
                "TSI nested": f"{mean_nested:.4f}",
                "Residual optimism": f"{mean_ro:.4f}",
            })
        out += f"### {dataset.upper()} — residual optimism (in-sample vs nested TSI)\n\n"
        out += _table(rows) + "\n"
    return out


def show_experiments(exp_data):
    out = ""
    for dataset, classifiers in exp_data.items():
        rows = []
        for clf_name, info in classifiers.items():
            strat = info.get("strategies", {})
            row = {"Classifier": clf_name.upper()}
            for key, label in STRATEGY_LABELS.items():
                if key not in strat:
                    continue
                m = strat[key]
                row[label] = f"{m['mean']:.4f}±{m['std']:.4f}"
            if "ece" in info:
                row["ECE"] = f"{info['ece']:.4f}"
            rows.append(row)
        out += f"### {dataset.upper()} — fusion strategies (primary metric ± std over folds)\n\n"
        out += _table(rows) + "\n"

        # pairwise significance (first classifier that has it)
        for clf_name, info in classifiers.items():
            pw = info.get("pairwise")
            if not pw:
                continue
            prows = [{
                "Pair": f"{STRATEGY_LABELS.get(p['a'], p['a'])} vs {STRATEGY_LABELS.get(p['b'], p['b'])}",
                "p": f"{p['p_value']:.4f}",
                "α_corr": f"{p['alpha_corrected']:.4f}",
                "Sig.": "✓" if p["significant"] else "—",
                "δ": f"{p['cliffs_delta']:.3f}",
                "Effect": p["magnitude"],
            } for p in pw]
            out += f"#### {dataset.upper()} / {clf_name.upper()} — pairwise Wilcoxon (Bonferroni)\n\n"
            out += _table(prows) + "\n"
            break
    return out


def main():
    out = ""
    tsi_path = os.path.join(SCRIPT_DIR, "tsi_results.json")
    exp_path = os.path.join(SCRIPT_DIR, "experiment_results.json")
    if os.path.exists(tsi_path):
        with open(tsi_path) as f:
            tsi_data = json.load(f)
        out += show_tsi(tsi_data)
        out += show_residual_optimism(tsi_data)
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            out += show_experiments(json.load(f))
    print(out if out else "No results found. Run 02_tsi_and_fusion.ipynb first.")


if __name__ == "__main__":
    main()
