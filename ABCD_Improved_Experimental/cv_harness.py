"""
Generalized CV harness for the ABCD_Improved_Experimental modeling phase.

Reuses, unmodified in spirit, the exact CV recipe from
ABCD_Audit_V5/_source_from_research_branch/Evaluation_FeatureEngineering/run_ablation_cv.py
(GroupKFold(n_splits=5) grouped by group_id, fresh
Pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=20260918))
fit inside each fold, OOF probabilities collected) and
.../Evaluation_V5Candidate/select_v5_threshold.py (threshold sweep rule),
but generalized to accept an ARBITRARY full feature list per call (not just
"base + extra"), since several experiments here substitute/replace base
features rather than only adding to them.

Reads ONLY ABCD_Improved_Experimental/dev_feature_table_v2.csv (the 2090-row
development population). Never reads or references the locked test set.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

N_FOLDS = 5
SEED = 20260918
CANDIDATE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Frozen V5's exact 17-feature set (EXP0 baseline), reproduced from
# Evaluation_V5Candidate/build_v5_candidate.py's V5_ALL_FEATURES.
V5_BASE_FEATURES = ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"]
V5_NEW_FEATURES = ["color_entropy", "lab_a_std", "lab_b_std", "red_fraction", "bluegray_fraction",
                    "dark_fraction", "skin_contrast", "solidity", "turning_angle_std",
                    "eccentricity", "D_px_normalized"]
V5_ALL_FEATURES = V5_BASE_FEATURES + V5_NEW_FEATURES
EXPECTED_V5_REPRO = {"roc_auc": 0.8008, "sensitivity": 0.7712, "specificity": 0.6733, "balanced_accuracy": 0.7222}


def load_table(path="dev_feature_table_v2.csv"):
    df = pd.read_csv(path)
    return df


def metrics_at_threshold(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
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
    bal_acc = (sens + spec) / 2 if sens == sens and spec == spec else float("nan")
    return {"threshold": threshold, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": n,
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "accuracy": acc, "balanced_accuracy": bal_acc}


def select_threshold(y_true, proba):
    """Exact same sweep + selection rule as select_v5_threshold.py."""
    sweep = [metrics_at_threshold(y_true, proba, t) for t in CANDIDATE_THRESHOLDS]
    qualifying = [m for m in sweep if m["sensitivity"] >= 0.55 and m["specificity"] >= 0.55]
    pool = qualifying if qualifying else sweep
    best = max(pool, key=lambda m: m["balanced_accuracy"])
    return best, sweep, bool(qualifying)


def evaluate_experiment(df, fields, name):
    """Full pipeline for one experiment: run CV, select this experiment's OWN
    threshold via the sweep rule on its own OOF, compute overall + per-fold
    metrics AT THAT THRESHOLD, and return a single results dict."""
    y_all = df["ground_truth_binary"].to_numpy(dtype=int)
    groups_all = df["group_id"].to_numpy()
    ids_all = df["image_id"].to_numpy()
    X_all = df[fields].to_numpy(dtype=float)

    valid_mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[valid_mask]
    y = y_all[valid_mask]
    groups = groups_all[valid_mask]
    ids = ids_all[valid_mask]
    n_dropped = int((~valid_mask).sum())

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.full(len(y), np.nan)
    fold_splits = []
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = Pipeline([("scale", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[val_idx])[:, 1]
        oof_proba[val_idx] = proba
        fold_splits.append(val_idx)

    overall_auc = float(roc_auc_score(y, oof_proba))
    best_thr, sweep, qualifying = select_threshold(y, oof_proba)
    threshold = best_thr["threshold"]
    overall_m = metrics_at_threshold(y, oof_proba, threshold)

    fold_bal_accs = []
    fold_aucs = []
    for fold_i, val_idx in enumerate(fold_splits):
        fy = y[val_idx]
        fp_ = oof_proba[val_idx]
        try:
            fauc = roc_auc_score(fy, fp_) if len(set(fy)) > 1 else float("nan")
        except ValueError:
            fauc = float("nan")
        fm = metrics_at_threshold(fy, fp_, threshold)
        fold_bal_accs.append(fm["balanced_accuracy"])
        fold_aucs.append(fauc)

    cv_mean_bal_acc = float(np.nanmean(fold_bal_accs))
    cv_std_bal_acc = float(np.nanstd(fold_bal_accs, ddof=1))

    return {
        "name": name, "fields": fields, "n_features": len(fields),
        "n_used": len(y), "n_dropped_missing": n_dropped,
        "threshold": threshold, "threshold_qualifying_pool_used": qualifying,
        "threshold_sweep": sweep,
        "overall_auc": overall_auc, "overall_metrics": overall_m,
        "cv_mean_balanced_accuracy": cv_mean_bal_acc, "cv_std_balanced_accuracy": cv_std_bal_acc,
        "fold_balanced_accuracies": fold_bal_accs, "fold_aucs": fold_aucs,
        "oof_by_id": {iid: p for iid, p in zip(ids, oof_proba)},
        "y_by_id": {iid: int(yy) for iid, yy in zip(ids, y)},
    }
