"""
Compare the V5 candidate (at its own selected threshold) against frozen V4
(at its own, unchanged, 0.25 threshold) -- development data only, locked
test never read. V4's cached OOF (Evaluation_V4/CrossValidation/
oof_predictions_full.json) is read-only here; V4 is never refit.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

V5_DIR = Path(__file__).resolve().parent
ROOT = V5_DIR.parent
V4_OOF_JSON = ROOT / "Evaluation_V4" / "CrossValidation" / "oof_predictions_full.json"
V4_FEATURE_TABLE = ROOT / "Evaluation_V4" / "FeatureTable" / "development_feature_table.csv"
V4_CANDIDATE_KEY = "full_ABC_D_conf_fraction__logistic_regression"
V4_THRESHOLD = 0.25


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
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": n,
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "accuracy": acc, "balanced_accuracy": bal_acc}


def load_v4_oof():
    import csv
    with open(V4_FEATURE_TABLE, newline="", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    with open(V4_OOF_JSON, encoding="utf-8") as f:
        candidate = json.load(f)[V4_CANDIDATE_KEY]
    probs = candidate["oof_probabilities"]
    y_true = candidate["y_true"]
    assert len(feat_rows) == len(probs) == len(y_true)
    proba_by_id, y_by_id = {}, {}
    for row, p, y in zip(feat_rows, probs, y_true):
        assert int(row["ground_truth_binary"]) == int(y)
        proba_by_id[row["image_id"]] = p
        y_by_id[row["image_id"]] = int(y)
    return proba_by_id, y_by_id


def main():
    with open(V5_DIR / "v5_oof_predictions.json", encoding="utf-8") as f:
        v5 = json.load(f)
    with open(V5_DIR / "v5_selected_threshold.json", encoding="utf-8") as f:
        v5_thresh = json.load(f)["selected"]["threshold"]

    v5_proba_by_id = dict(zip(v5["image_ids"], v5["oof_probabilities"]))
    v5_y_by_id = dict(zip(v5["image_ids"], v5["y_true"]))
    v4_proba_by_id, v4_y_by_id = load_v4_oof()

    common_ids = sorted(set(v5_proba_by_id.keys()) & set(v4_proba_by_id.keys()))
    print(f"Common OOF population (V5 and V4 both have a prediction): {len(common_ids)} images "
          f"(V5 total: {len(v5_proba_by_id)}, V4 total: {len(v4_proba_by_id)})")
    for i in common_ids:
        assert v5_y_by_id[i] == v4_y_by_id[i], f"ground-truth mismatch for {i}"

    y = np.array([v4_y_by_id[i] for i in common_ids])
    v4_p = np.array([v4_proba_by_id[i] for i in common_ids])
    v5_p = np.array([v5_proba_by_id[i] for i in common_ids])

    v4_auc = roc_auc_score(y, v4_p)
    v5_auc = roc_auc_score(y, v5_p)
    v4_m = metrics_at_threshold(y, v4_p, V4_THRESHOLD)
    v5_m = metrics_at_threshold(y, v5_p, v5_thresh)

    print("\nModel      | Threshold | Sens   | Spec   | Prec   | F1     | Acc    | BalAcc | ROC-AUC")
    print(f"Frozen V4  | {V4_THRESHOLD:.2f}      | {v4_m['sensitivity']:.4f} | {v4_m['specificity']:.4f} | "
          f"{v4_m['precision']:.4f} | {v4_m['f1']:.4f} | {v4_m['accuracy']:.4f} | {v4_m['balanced_accuracy']:.4f} | {v4_auc:.4f}")
    print(f"V5 Candid. | {v5_thresh:.2f}      | {v5_m['sensitivity']:.4f} | {v5_m['specificity']:.4f} | "
          f"{v5_m['precision']:.4f} | {v5_m['f1']:.4f} | {v5_m['accuracy']:.4f} | {v5_m['balanced_accuracy']:.4f} | {v5_auc:.4f}")

    # ---- paired FN/FP shift analysis ----
    fn_rescued = fn_introduced = fp_corrected = fp_introduced = 0
    for i in common_ids:
        yy = v4_y_by_id[i]
        p4 = 1 if v4_proba_by_id[i] >= V4_THRESHOLD else 0
        p5 = 1 if v5_proba_by_id[i] >= v5_thresh else 0
        if yy == 1:
            if p4 == 0 and p5 == 1:
                fn_rescued += 1
            elif p4 == 1 and p5 == 0:
                fn_introduced += 1
        else:
            if p4 == 1 and p5 == 0:
                fp_corrected += 1
            elif p4 == 0 and p5 == 1:
                fp_introduced += 1

    net_tp = v5_m["TP"] - v4_m["TP"]
    net_tn = v5_m["TN"] - v4_m["TN"]
    net_fp = v5_m["FP"] - v4_m["FP"]
    net_fn = v5_m["FN"] - v4_m["FN"]

    print(f"\nMelanoma false negatives rescued by V5:  {fn_rescued}")
    print(f"New melanoma false negatives introduced: {fn_introduced}")
    print(f"V4 false positives corrected by V5:       {fp_corrected}")
    print(f"New false positives introduced by V5:     {fp_introduced}")
    print(f"Net change: TP {net_tp:+d}  TN {net_tn:+d}  FP {net_fp:+d}  FN {net_fn:+d}")

    # ---- fold-by-fold for V5 at its selected threshold ----
    fold_rows = []
    for fr in v5["fold_records"]:
        fold_rows.append({
            "fold": fr["fold"], "n_val": fr["n_val"], "auc": fr["auc"],
            "note": "fold metrics below recomputed at V5's selected threshold, not the CV-monitoring 0.25 used inside run_group_cv",
        })
    print("\nV5 fold-by-fold ROC-AUC (from the CV run itself, threshold-independent):")
    for fr in v5["fold_records"]:
        print(f"  fold {fr['fold']}: n_val={fr['n_val']}  auc={fr['auc']:.4f}  "
              f"sens@0.25={fr['sensitivity']:.4f}  spec@0.25={fr['specificity']:.4f}  "
              f"balacc@0.25={fr['balanced_accuracy']:.4f}")

    summary = {
        "common_n": len(common_ids),
        "v4": {"threshold": V4_THRESHOLD, "roc_auc": round(float(v4_auc), 4),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in v4_m.items()}},
        "v5": {"threshold": v5_thresh, "roc_auc": round(float(v5_auc), 4),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in v5_m.items()}},
        "paired_shift": {"fn_rescued": fn_rescued, "fn_introduced": fn_introduced,
                          "fp_corrected": fp_corrected, "fp_introduced": fp_introduced,
                          "net_tp": net_tp, "net_tn": net_tn, "net_fp": net_fp, "net_fn": net_fn},
        "v5_fold_by_fold": v5["fold_records"],
    }
    with open(V5_DIR / "v5_vs_v4_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote v5_vs_v4_comparison.json")
    return summary


if __name__ == "__main__":
    main()
