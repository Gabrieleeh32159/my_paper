import json
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "tsi_results.json")) as f:
    tsi_data = json.load(f)

with open(os.path.join(script_dir, "experiment_results.json")) as f:
    exp_data = json.load(f)

output = ""

# --- TSI Analysis ---
for dataset, dt_info in tsi_data.items():
    rows = []

    # New (information-gain) format with backward-compatible fallback to accuracy.
    if "info_gain_matrix" in dt_info:
        matrix_key, std_key, metric_label = "info_gain_matrix", "info_gain_std", "IG"
    else:
        matrix_key, std_key, metric_label = "accuracy_matrix", "accuracy_std", "Acc"

    has_tsi_std = "tsi_std" in dt_info
    has_cell_std = std_key in dt_info
    has_ci = "tsi_ci" in dt_info
    has_sig = "scale_significant" in dt_info

    for feature in dt_info["tsi_scores"]:
        row = {
            "Feature": feature,
            "TSI Score": f"{dt_info['tsi_scores'][feature]:.4f}",
        }
        if has_tsi_std:
            row["TSI Std"] = f"{dt_info['tsi_std'][feature]:.4f}"
        if has_ci:
            lo, hi = dt_info["tsi_ci"][feature]
            row["95% CI"] = f"[{lo:.4f}, {hi:.4f}]"
        if has_sig:
            row["Sig."] = "✓" if dt_info["scale_significant"][feature] else "—"
        row["Optimal Scale"] = dt_info["optimal_scales"][feature]
        for scale in ("short", "medium", "long"):
            val = dt_info[matrix_key][feature][scale]
            if has_cell_std:
                std = dt_info[std_key][feature][scale]
                row[f"{metric_label} ({scale.capitalize()})"] = f"{val:.4f} ± {std:.4f}"
            else:
                row[f"{metric_label} ({scale.capitalize()})"] = f"{val:.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    output += f"### Dataset: {dataset.upper()} — TSI Analysis\n\n"
    output += df.to_markdown(index=False) + "\n\n"

# --- Classification Results ---
scale_labels = {
    "short": "Short", "medium": "Medium", "long": "Long",
    "early": "Early", "late": "Late", "tsi_weighted": "TSI-Weighted",
}
for dataset, classifiers in exp_data.items():
    rows = []
    for clf_name, scales in classifiers.items():
        row = {"Classifier": clf_name.upper()}
        for scale, label in scale_labels.items():
            if scale not in scales:
                continue
            m = scales[scale]
            if "accuracy_mean" in m:
                acc = f"{m['accuracy_mean']:.4f} ± {m['accuracy_std']:.4f}"
                f1 = f"{m['f1_macro_mean']:.4f} ± {m['f1_macro_std']:.4f}"
            else:
                acc = f"{m['accuracy']:.4f}"
                f1 = f"{m['f1_macro']:.4f}"
            row[f"Acc {label}"] = acc
            row[f"F1 {label}"] = f1
        rows.append(row)
    df = pd.DataFrame(rows)
    output += f"### Dataset: {dataset.upper()} — Classification Results\n\n"
    output += df.to_markdown(index=False) + "\n\n"

print(output)