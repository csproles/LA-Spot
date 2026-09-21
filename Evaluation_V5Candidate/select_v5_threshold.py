"""
V5 operating-threshold selection, using ONLY V5's own development
grouped-OOF probabilities (v5_oof_predictions.json) -- never a model fit on
the same samples being evaluated, never the locked test set.

Exact same threshold grid and selection rule as
Evaluation_V4/select_threshold.py: sweep {0.20, 0.25, 0.30, 0.35, 0.40,
0.45, 0.50}, pick the highest-balanced-accuracy threshold among those where
BOTH sensitivity and specificity are >= 0.55 (fallback: highest balanced
accuracy overall if none qualify). No threshold is hand-picked.
"""

import json
from pathlib import Path

import numpy as np

V5_DIR = Path(__file__).resolve().parent
CANDIDATE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def metrics_at_threshold(y_true, proba, t):
    pred = (proba >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and prec == prec and sens == sens else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    bal_acc = (sens + spec) / 2
    return {"threshold": t, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "precision": round(prec, 4), "f1": round(f1, 4), "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4)}


def main():
    with open(V5_DIR / "v5_oof_predictions.json", encoding="utf-8") as f:
        v5 = json.load(f)
    y_true = np.array(v5["y_true"])
    proba = np.array(v5["oof_probabilities"])
    print(f"V5 OOF: n={len(y_true)}, prevalence={y_true.mean():.3f}")

    sweep = [metrics_at_threshold(y_true, proba, t) for t in CANDIDATE_THRESHOLDS]
    print("\nThreshold | Sensitivity | Specificity | Precision | F1 | Accuracy | Balanced Accuracy")
    for m in sweep:
        print(f"{m['threshold']:.2f}      | {m['sensitivity']:.4f}      | {m['specificity']:.4f}      | "
              f"{m['precision']:.4f}    | {m['f1']:.4f} | {m['accuracy']:.4f}   | {m['balanced_accuracy']:.4f}")

    with open(V5_DIR / "v5_threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"feature_set": v5["feature_set"], "sweep": sweep}, f, indent=2)

    qualifying = [m for m in sweep if m["sensitivity"] >= 0.55 and m["specificity"] >= 0.55]
    pool = qualifying if qualifying else sweep
    best = max(pool, key=lambda m: m["balanced_accuracy"])
    print(f"\nSelected V5 operating threshold: {best['threshold']} "
          f"(qualifying sens>=0.55 AND spec>=0.55 pool used: {bool(qualifying)}, "
          f"{len(qualifying)}/{len(sweep)} thresholds qualified)")
    print(json.dumps(best, indent=2))

    with open(V5_DIR / "v5_selected_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"selected": best,
                  "selection_rule": "max balanced_accuracy among thresholds with sens>=0.55 AND spec>=0.55, "
                                    "else max balanced_accuracy overall -- identical rule to "
                                    "Evaluation_V4/select_threshold.py"}, f, indent=2)
    return best, sweep


if __name__ == "__main__":
    main()
