"""PHASE 4 — A concern threshold sweep, DEVELOPMENT DATA ONLY.

Keeps B, C, and the critical-color override exactly as originally computed
(the override's effect is inferred from the saved risk_level + concern
flags — see metrics_lib.infer_critical_override — since the raw color
fractions aren't stored in results.csv, but its EFFECT is fully
recoverable). Only the A concern threshold is varied.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_lib import (ROOT, load_merged_results, load_split_ids, build_split_rows,
                         compute_metrics, to_float, infer_critical_override,
                         predict_with_A_threshold, CURRENT_A_THRESHOLD)

DEV_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "development_manifest.csv"
OUT_DIR = ROOT / "Evaluation_FinalTargeted" / "ThresholdAnalysis"

THRESHOLDS = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_merged_results()
    dev_ids = load_split_ids(DEV_MANIFEST)
    evaluable_rows, status_counts, n_images = build_split_rows(all_results, dev_ids)
    print(f"Development evaluable rows: {len(evaluable_rows)}")

    for r in evaluable_rows:
        r["_override_fired"] = infer_critical_override(r)
    n_override = sum(1 for r in evaluable_rows if r["_override_fired"])
    print(f"Rows where critical-color override is inferred to have fired: {n_override}")

    # baseline (threshold=0.20) confusion, for "rescued relative to 0.20" comparisons
    baseline_preds = {}
    for r in evaluable_rows:
        baseline_preds[id(r)] = predict_with_A_threshold(r, CURRENT_A_THRESHOLD, r["_override_fired"])

    # melanoma-specific C-only-FN set (the 73.9% failure pattern from earlier analysis)
    def is_c_only_fn(r):
        a = r["A_concern"] == "True"; b = r["B_circularity_concern"] == "True"; c = r["C_concern"] == "True"
        gt_positive = r["ground_truth_binary"] == "1"
        baseline_pred = baseline_preds[id(r)]
        return gt_positive and baseline_pred == 0 and c and not a and not b

    c_only_fn_rows = [r for r in evaluable_rows if is_c_only_fn(r)]
    c_only_fn_ids = {id(r) for r in c_only_fn_rows}
    print(f"C-only melanoma/malignant FN at baseline threshold: {len(c_only_fn_rows)}")

    rows_out = []
    for t in THRESHOLDS:
        tp = tn = fp = fn = 0
        fn_rescued = 0  # was FN at 0.20, now TP at this threshold
        tn_to_fp = 0    # was TN at 0.20, now FP at this threshold
        c_only_rescued = 0
        for r in evaluable_rows:
            gt = int(r["ground_truth_binary"])
            base_pred = baseline_preds[id(r)]
            new_pred = predict_with_A_threshold(r, t, r["_override_fired"])
            if gt == 1 and new_pred == 1:
                tp += 1
            elif gt == 0 and new_pred == 0:
                tn += 1
            elif gt == 0 and new_pred == 1:
                fp += 1
            elif gt == 1 and new_pred == 0:
                fn += 1
            if gt == 1 and base_pred == 0 and new_pred == 1:
                fn_rescued += 1
            if gt == 0 and base_pred == 0 and new_pred == 1:
                tn_to_fp += 1
            if id(r) in c_only_fn_ids and new_pred == 1:
                c_only_rescued += 1

        m = compute_metrics(tp, tn, fp, fn)
        rows_out.append({
            "A_threshold": t, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": round(m["sensitivity"], 4), "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4), "f1": round(m["f1"], 4),
            "accuracy": round(m["accuracy"], 4), "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "FN_rescued_vs_0.20": fn_rescued, "TN_converted_to_FP_vs_0.20": tn_to_fp,
            "C_only_FN_rescued": c_only_rescued, "C_only_FN_rescued_pct":
                round(100 * c_only_rescued / len(c_only_fn_rows), 1) if c_only_fn_rows else None,
        })
        print(f"A>{t}: TP={tp} TN={tn} FP={fp} FN={fn} sens={m['sensitivity']:.3f} "
              f"spec={m['specificity']:.3f} FN_rescued={fn_rescued} TN->FP={tn_to_fp}")

    with open(OUT_DIR / "A_threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

    # plots
    thresholds = [r["A_threshold"] for r in rows_out]
    sens = [r["sensitivity"] for r in rows_out]
    spec = [r["specificity"] for r in rows_out]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#fcfcfb")
    ax.plot(thresholds, sens, marker="o", color="#0ca30c", label="Sensitivity")
    ax.set_xlabel("A concern threshold"); ax.set_ylabel("Sensitivity", color="#0ca30c")
    ax.invert_xaxis()
    ax.set_title("Sensitivity vs A threshold (development)")
    ax.grid(color="#e1e0d9"); fig.tight_layout(); fig.savefig(OUT_DIR / "sensitivity_vs_threshold.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#fcfcfb")
    ax.plot(thresholds, spec, marker="o", color="#2a78d6", label="Specificity")
    ax.set_xlabel("A concern threshold"); ax.set_ylabel("Specificity", color="#2a78d6")
    ax.invert_xaxis()
    ax.set_title("Specificity vs A threshold (development)")
    ax.grid(color="#e1e0d9"); fig.tight_layout(); fig.savefig(OUT_DIR / "specificity_vs_threshold.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor="#fcfcfb")
    ax.plot(thresholds, sens, marker="o", color="#0ca30c", label="Sensitivity")
    ax.plot(thresholds, spec, marker="s", color="#2a78d6", label="Specificity")
    for t, s1, s2 in zip(thresholds, sens, spec):
        ax.annotate(f"{t}", (t, max(s1, s2)), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.invert_xaxis()
    ax.set_xlabel("A concern threshold"); ax.set_ylabel("Rate")
    ax.set_title("Sensitivity/specificity tradeoff vs A threshold (development)")
    ax.legend(); ax.grid(color="#e1e0d9"); fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_specificity_tradeoff.png", dpi=150); plt.close(fig)

    print(f"\nWrote CSV + 3 plots to {OUT_DIR}")
    return rows_out


if __name__ == "__main__":
    main()
