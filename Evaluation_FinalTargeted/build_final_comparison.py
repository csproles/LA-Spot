"""Builds the V2-vs-V3 confusion matrices, side-by-side table, and
dev-vs-locked-test comparison for the final report. Read-only over
already-computed Phase 3/5/7 results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Evaluation_FinalTargeted" / "V2_vs_V3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def draw_confusion(ax, tp, tn, fp, fn, title):
    matrix = np.array([[tn, fp], [fn, tp]])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i,j]}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: LOWER", "Pred: ELEVATED"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: benign", "Actual: melanoma"])
    ax.set_title(title, fontsize=12, fontweight="bold")


def main():
    phase7 = load_json(ROOT / "Evaluation_FinalTargeted" / "LockedTest" / "phase7_locked_test_results.json")
    phase3 = load_json(ROOT / "Evaluation_FinalTargeted" / "Development" / "phase3_v2_baseline_development.json")
    phase5 = load_json(ROOT / "Evaluation_FinalTargeted" / "DecisionModelAnalysis" / "phase5_decision_candidates_development.json")

    v2_test, v3_test = phase7["V2"], phase7["V3"]
    v2_dev = phase3["metrics"]
    v3_dev = phase5["B_binary_revised_A_0.14"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#fcfcfb")
    draw_confusion(axes[0], v2_test["TP"], v2_test["TN"], v2_test["FP"], v2_test["FN"], "V2 — locked test")
    draw_confusion(axes[1], v3_test["TP"], v3_test["TN"], v3_test["FP"], v3_test["FN"], "V3 — locked test")
    fig.suptitle("Confusion matrices, locked test (n=615 evaluable)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices_locked_test.png", dpi=150)
    plt.close(fig)

    def row(label, m):
        return (f"| {label} | {m['TP']} | {m['TN']} | {m['FP']} | {m['FN']} | "
                f"{m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['precision']:.3f} | "
                f"{m['f1']:.3f} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} |")

    table = [
        "| Set / Model | TP | TN | FP | FN | Sens | Spec | Prec | F1 | Acc | BalAcc |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        row("Dev — V2 (A>0.20)", v2_dev),
        row("Dev — V3 (A>0.14)", v3_dev),
        row("Locked test — V2", v2_test),
        row("Locked test — V3", v3_test),
    ]
    table_text = "\n".join(table)
    print(table_text)
    with open(OUT_DIR / "side_by_side_metrics_table.md", "w", encoding="utf-8") as f:
        f.write(table_text + "\n")

    def delta(a, b, key):
        return round(b[key] - a[key], 4)

    generalization = {
        "sensitivity_gain_dev": delta(v2_dev, v3_dev, "sensitivity"),
        "sensitivity_gain_locked_test": delta(v2_test, v3_test, "sensitivity"),
        "specificity_cost_dev": delta(v2_dev, v3_dev, "specificity"),
        "specificity_cost_locked_test": delta(v2_test, v3_test, "specificity"),
        "balanced_accuracy_change_dev": delta(v2_dev, v3_dev, "balanced_accuracy"),
        "balanced_accuracy_change_locked_test": delta(v2_test, v3_test, "balanced_accuracy"),
        "f1_change_dev": delta(v2_dev, v3_dev, "f1"),
        "f1_change_locked_test": delta(v2_test, v3_test, "f1"),
    }
    print(json.dumps(generalization, indent=2))
    with open(OUT_DIR / "development_vs_locked_test_generalization.json", "w", encoding="utf-8") as f:
        json.dump(generalization, f, indent=2)

    return table_text, generalization


if __name__ == "__main__":
    main()
