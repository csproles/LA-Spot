"""STEP 5 — Operating threshold selection, using ONLY the development
cross-validation out-of-fold probabilities from the winning Step 3/4
candidate. The locked test is never touched here.

Winning candidate (from cv_summary.json): full_ABC_D_conf_fraction +
logistic_regression — the only candidate with a meaningful AUC
improvement over core ABC alone (0.779 vs 0.71), i.e. genuine added
discrimination, not just a threshold-driven tradeoff.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CV_DIR = ROOT / "Evaluation_V4" / "CrossValidation"
OUT_DIR = ROOT / "Evaluation_V4" / "ThresholdSelection"

WINNING_KEY = "full_ABC_D_conf_fraction__logistic_regression"
CANDIDATE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def metrics_at_threshold(y_true, proba, t):
    pred = (proba >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and prec == prec and sens == sens else float("nan")
    bal_acc = (sens + spec) / 2
    return {"threshold": t, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "precision": round(prec, 4), "f1": round(f1, 4), "balanced_accuracy": round(bal_acc, 4)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CV_DIR / "oof_predictions_full.json", encoding="utf-8") as f:
        all_cv = json.load(f)

    winner = all_cv[WINNING_KEY]
    y_true = np.array(winner["y_true"])
    proba = np.array(winner["oof_probabilities"])
    print(f"Winning candidate: {WINNING_KEY}  (n={len(y_true)}, prevalence={y_true.mean():.3f})")

    sweep = [metrics_at_threshold(y_true, proba, t) for t in CANDIDATE_THRESHOLDS]
    for m in sweep:
        print(f"t={m['threshold']}: sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
              f"bal_acc={m['balanced_accuracy']:.3f} prec={m['precision']:.3f} f1={m['f1']:.3f}")

    with open(OUT_DIR / "threshold_sweep_cv.json", "w", encoding="utf-8") as f:
        json.dump({"winning_candidate": WINNING_KEY, "sweep": sweep}, f, indent=2)

    # Selection rule (documented, not a search for the single best number):
    # prefer the threshold with the highest balanced accuracy among those
    # where BOTH sensitivity and specificity are at least 0.55 (a floor
    # ruling out "high sensitivity by flagging almost everything" or the
    # reverse) -- if none qualify, fall back to the single highest
    # balanced-accuracy point.
    qualifying = [m for m in sweep if m["sensitivity"] >= 0.55 and m["specificity"] >= 0.55]
    pool = qualifying if qualifying else sweep
    best = max(pool, key=lambda m: m["balanced_accuracy"])
    print(f"\nSelected operating threshold: {best['threshold']}  "
          f"(qualifying-both->=0.55 pool used: {bool(qualifying)})")
    print(json.dumps(best, indent=2))

    with open(OUT_DIR / "selected_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"winning_candidate": WINNING_KEY, "selected": best,
                  "selection_rule": "max balanced_accuracy among thresholds with sens>=0.55 AND spec>=0.55, "
                                    "else max balanced_accuracy overall"}, f, indent=2)
    return best


if __name__ == "__main__":
    main()
