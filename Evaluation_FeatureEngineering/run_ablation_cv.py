"""
Controlled ablation study: frozen V4's 6-feature baseline vs. V4 + each new
candidate feature group, using the EXACT SAME grouped cross-validation
methodology as Evaluation_V4/build_feature_table_and_cv.py (same patient/
lesion union-find grouping, GroupKFold n_splits=5, StandardScaler ->
LogisticRegression(max_iter=1000, random_state=20260918) pipeline fit fresh
inside each fold). V4 itself is never refit or modified; its own cached OOF
(Evaluation_V4/CrossValidation/oof_predictions_full.json) is read read-only
as the baseline for paired comparison.

All reported Sensitivity/Specificity/Precision/F1/Accuracy/Balanced-Accuracy
use the SAME fixed 0.25 decision threshold already selected for frozen V4
(Evaluation_V4/ThresholdSelection/selected_threshold.json) -- no threshold
is re-tuned for any candidate here, so the comparison isn't won by
threshold-shopping.

Locked test set is never read.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
EXTENDED_TABLE = BENCH_DIR / "extended_feature_table.csv"
V4_OOF_JSON = ROOT / "Evaluation_V4" / "CrossValidation" / "oof_predictions_full.json"
V4_FEATURE_TABLE = ROOT / "Evaluation_V4" / "FeatureTable" / "development_feature_table.csv"
V4_CANDIDATE_KEY = "full_ABC_D_conf_fraction__logistic_regression"

N_FOLDS = 5
SEED = 20260918
THRESHOLD = 0.25  # frozen V4's own selected operating threshold, reused unchanged

BASE_FEATURES = ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"]

# Candidate feature-group additions, PRUNED per feature_analysis_table.csv /
# correlation_matrix.csv / redundancy_report.md (see experiment_log.md for
# the full reasoning):
#   - isoperimetric_ratio dropped: rho=1.00 with existing B_circularity
#     (mathematically 1/(1-B_circularity)) -- adds zero new information.
#   - convexity_deficit dropped: rho=-1.00 with solidity (exact complement);
#     solidity kept as the more standard metric.
#   - aspect_ratio dropped: rho=1.00 with eccentricity; eccentricity kept
#     (bounded [0,1), standard in shape-analysis literature).
#   - major_axis_px, minor_axis_px, lesion_area_px dropped: rho=0.92-1.00
#     with the EXISTING D_px feature -- no new information over baseline.
#   - color_cluster_count, white_fraction dropped: univariate AUC 0.46/0.51,
#     i.e. no better than chance -- not "available" enough to justify.
#   - entropy_intensity dropped: rho=0.945 with lab_L_std, and both are weak
#     (AUC ~0.51-0.53) -- kept lab_L_std as the simpler of the pair.
#   - glcm_contrast, glcm_homogeneity dropped: rho=0.91/-0.95 with
#     local_contrast -- local_contrast kept as the simplest representative.
#   - texture_asymmetry dropped: univariate AUC 0.501 -- literally chance
#     level, no measurable signal.
# NOTE flagged for interpretation, not pruned here (kept in for the ablation
# to test empirically): D_px_normalized has rho=0.989 with the EXISTING
# lesion_fraction feature already in V4's baseline -- its strong standalone
# univariate AUC (0.758) is expected to mostly already be captured by V4,
# so a large ablation gain from this group would be a surprise worth
# double-checking, not simply accepted.
FEATURE_GROUPS = {
    "V4_baseline": [],
    "plus_improved_asymmetry": ["color_asymmetry"],
    "plus_border": ["solidity", "radial_cv", "turning_angle_std"],
    "plus_color": ["color_entropy", "lab_a_std", "lab_b_std", "red_fraction", "bluegray_fraction",
                   "dark_fraction", "skin_contrast"],
    "plus_geometry": ["eccentricity", "D_px_normalized"],
    "plus_texture": ["local_contrast", "glcm_energy"],
}
# Filled in after single-group results are known (section 4: "combinations
# of feature groups that showed useful independent signal"). Single-group
# ablation results (paired dAUC vs V4): plus_color +0.0149, plus_border
# +0.0091, plus_geometry +0.0062, plus_improved_asymmetry +0.0012,
# plus_texture +0.0011 -- the last two show no measurable independent
# signal and are excluded from the combined candidate. color+border+geometry
# are combined below, EXCEPT radial_cv is dropped from this combination
# specifically (not from the standalone plus_border ablation above): it is
# cross-group redundant with geometry's eccentricity (Spearman rho=0.913,
# just over the 0.90 redundancy threshold), so keeping both here would
# double-count essentially the same shape-elongation signal.
BEST_COMBO_FEATURES = (FEATURE_GROUPS["plus_color"]
                        + [f for f in FEATURE_GROUPS["plus_border"] if f != "radial_cv"]
                        + FEATURE_GROUPS["plus_geometry"])


def load_table():
    with open(EXTENDED_TABLE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ground_truth_binary"] = int(r["ground_truth_binary"])
        for k, v in list(r.items()):
            if k in ("image_id", "group_id", "ground_truth_binary", "_errors"):
                continue
            r[k] = float(v) if v not in ("", "nan") else np.nan
    return rows


def load_v4_baseline_oof():
    with open(V4_FEATURE_TABLE, newline="", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    with open(V4_OOF_JSON, encoding="utf-8") as f:
        candidate = json.load(f)[V4_CANDIDATE_KEY]
    probs = candidate["oof_probabilities"]
    y_true = candidate["y_true"]
    assert len(feat_rows) == len(probs) == len(y_true)
    out = {}
    for row, p, y in zip(feat_rows, probs, y_true):
        assert int(row["ground_truth_binary"]) == int(y)
        out[row["image_id"]] = p
    return out


def metrics_at_threshold(y_true, proba, threshold):
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
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": n, "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "accuracy": acc, "balanced_accuracy": bal_acc}


def run_group_cv(rows, extra_features):
    """Same recipe as Evaluation_V4/build_feature_table_and_cv.py::run_cv,
    generalized to an arbitrary feature list. Returns per-image_id OOF
    probability dict, fold-level metrics, and the row image_ids actually
    used (after dropping rows missing any of these features)."""
    fields = BASE_FEATURES + extra_features
    y_all = np.array([r["ground_truth_binary"] for r in rows])
    groups_all = np.array([r["group_id"] for r in rows])
    ids_all = np.array([r["image_id"] for r in rows])
    X_all = np.array([[r[f] for f in fields] for r in rows], dtype=float)

    valid_mask = ~np.isnan(X_all).any(axis=1)
    X = X_all[valid_mask]
    y = y_all[valid_mask]
    groups = groups_all[valid_mask]
    ids = ids_all[valid_mask]
    n_dropped = int((~valid_mask).sum())

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.full(len(y), np.nan)
    fold_records = []
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[val_idx])[:, 1]
        oof_proba[val_idx] = proba

        fold_y = y[val_idx]
        try:
            fold_auc = roc_auc_score(fold_y, proba) if len(set(fold_y)) > 1 else float("nan")
        except ValueError:
            fold_auc = float("nan")
        fold_m = metrics_at_threshold(fold_y, proba, THRESHOLD)
        fold_records.append({"fold": fold_i, "n_val": len(val_idx), "auc": fold_auc, **fold_m})

    overall_auc = roc_auc_score(y, oof_proba)
    overall_m = metrics_at_threshold(y, oof_proba, THRESHOLD)

    oof_by_id = {iid: p for iid, p in zip(ids, oof_proba)}
    y_by_id = {iid: int(yy) for iid, yy in zip(ids, y)}
    return {
        "fields": fields, "n_used": len(y), "n_dropped_missing": n_dropped,
        "overall_auc": round(float(overall_auc), 4), "overall_metrics": overall_m,
        "fold_records": fold_records, "oof_by_id": oof_by_id, "y_by_id": y_by_id,
    }


def paired_bootstrap_ci(ids_common, y_by_id, oof_a, oof_b, n_boot=2000, seed=SEED):
    """Paired bootstrap (image-level resampling) for the difference in
    ROC-AUC and balanced accuracy between candidate A and V4 baseline B,
    both restricted to the same common image set. This treats images as
    exchangeable units; it does not additionally account for the
    patient/lesion grouping used at training time, which is a known
    simplification, noted in the report."""
    rng = np.random.RandomState(seed)
    ids_arr = np.array(ids_common)
    y = np.array([y_by_id[i] for i in ids_arr])
    a = np.array([oof_a[i] for i in ids_arr])
    b = np.array([oof_b[i] for i in ids_arr])
    n = len(ids_arr)

    auc_diffs = []
    balacc_diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        auc_a = roc_auc_score(yb, a[idx])
        auc_b = roc_auc_score(yb, b[idx])
        auc_diffs.append(auc_a - auc_b)
        ma = metrics_at_threshold(yb, a[idx], THRESHOLD)
        mb = metrics_at_threshold(yb, b[idx], THRESHOLD)
        if ma["balanced_accuracy"] == ma["balanced_accuracy"] and mb["balanced_accuracy"] == mb["balanced_accuracy"]:
            balacc_diffs.append(ma["balanced_accuracy"] - mb["balanced_accuracy"])

    def ci(vals):
        vals = np.array(vals)
        return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    auc_mean, auc_lo, auc_hi = ci(auc_diffs)
    bal_mean, bal_lo, bal_hi = ci(balacc_diffs)
    return {
        "n_bootstrap": n_boot,
        "auc_diff_mean": round(auc_mean, 4), "auc_diff_ci95": [round(auc_lo, 4), round(auc_hi, 4)],
        "balacc_diff_mean": round(bal_mean, 4), "balacc_diff_ci95": [round(bal_lo, 4), round(bal_hi, 4)],
    }


def paired_error_analysis(ids_common, y_by_id, oof_a, oof_b):
    """Confusion-shift analysis between candidate A and V4 baseline B at the
    shared 0.25 threshold, on the common image set."""
    fn_rescued = fn_introduced = fp_removed = fp_introduced = 0
    for iid in ids_common:
        y = y_by_id[iid]
        pa = 1 if oof_a[iid] >= THRESHOLD else 0
        pb = 1 if oof_b[iid] >= THRESHOLD else 0
        if y == 1:
            if pb == 0 and pa == 1:
                fn_rescued += 1
            elif pb == 1 and pa == 0:
                fn_introduced += 1
        else:
            if pb == 1 and pa == 0:
                fp_removed += 1
            elif pb == 0 and pa == 1:
                fp_introduced += 1
    return {"fn_rescued": fn_rescued, "fn_introduced": fn_introduced,
            "fp_removed": fp_removed, "fp_introduced": fp_introduced}


def main():
    rows = load_table()
    v4_oof_full = load_v4_baseline_oof()  # V4's own cached OOF, read-only, never refit here

    all_candidates = {}
    for name, extra in FEATURE_GROUPS.items():
        result = run_group_cv(rows, extra)
        all_candidates[name] = result
        print(f"{name}: fields={result['fields']} n_used={result['n_used']} "
              f"dropped={result['n_dropped_missing']} AUC={result['overall_auc']} "
              f"balacc={result['overall_metrics']['balanced_accuracy']:.4f}")

    if BEST_COMBO_FEATURES:
        combo_extra = list(dict.fromkeys(BEST_COMBO_FEATURES))  # dedupe, keep order
        result = run_group_cv(rows, combo_extra)
        all_candidates["best_combined"] = result
        print(f"best_combined: fields={result['fields']} n_used={result['n_used']} "
              f"AUC={result['overall_auc']} balacc={result['overall_metrics']['balanced_accuracy']:.4f}")

    # ---- comparison table ----
    table_fields = ["candidate", "n_used", "n_dropped_missing", "sensitivity", "specificity",
                     "precision", "f1", "accuracy", "balanced_accuracy", "roc_auc"]
    with open(BENCH_DIR / "ablation_comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=table_fields)
        w.writeheader()
        for name, r in all_candidates.items():
            m = r["overall_metrics"]
            w.writerow({"candidate": name, "n_used": r["n_used"], "n_dropped_missing": r["n_dropped_missing"],
                        "sensitivity": round(m["sensitivity"], 4), "specificity": round(m["specificity"], 4),
                        "precision": round(m["precision"], 4), "f1": round(m["f1"], 4),
                        "accuracy": round(m["accuracy"], 4), "balanced_accuracy": round(m["balanced_accuracy"], 4),
                        "roc_auc": r["overall_auc"]})

    # ---- fold-by-fold table ----
    fold_fields = ["candidate", "fold", "n_val", "auc", "sensitivity", "specificity", "balanced_accuracy"]
    with open(BENCH_DIR / "fold_by_fold_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fold_fields)
        w.writeheader()
        for name, r in all_candidates.items():
            for fr in r["fold_records"]:
                w.writerow({"candidate": name, "fold": fr["fold"], "n_val": fr["n_val"],
                            "auc": round(fr["auc"], 4) if fr["auc"] == fr["auc"] else "",
                            "sensitivity": round(fr["sensitivity"], 4) if fr["sensitivity"] == fr["sensitivity"] else "",
                            "specificity": round(fr["specificity"], 4) if fr["specificity"] == fr["specificity"] else "",
                            "balanced_accuracy": round(fr["balanced_accuracy"], 4) if fr["balanced_accuracy"] == fr["balanced_accuracy"] else ""})

    # ---- paired comparison vs V4's own cached OOF, error analysis + bootstrap CI ----
    paired_results = {}
    for name, r in all_candidates.items():
        if name == "V4_baseline":
            continue
        common_ids = sorted(set(r["oof_by_id"].keys()) & set(v4_oof_full.keys()))
        y_by_id = {i: r["y_by_id"][i] for i in common_ids}
        err_analysis = paired_error_analysis(common_ids, y_by_id, r["oof_by_id"], v4_oof_full)
        boot = paired_bootstrap_ci(common_ids, y_by_id, r["oof_by_id"], v4_oof_full)

        cand_m = metrics_at_threshold(np.array([y_by_id[i] for i in common_ids]),
                                       np.array([r["oof_by_id"][i] for i in common_ids]), THRESHOLD)
        v4_m = metrics_at_threshold(np.array([y_by_id[i] for i in common_ids]),
                                     np.array([v4_oof_full[i] for i in common_ids]), THRESHOLD)
        cand_auc = roc_auc_score([y_by_id[i] for i in common_ids], [r["oof_by_id"][i] for i in common_ids])
        v4_auc = roc_auc_score([y_by_id[i] for i in common_ids], [v4_oof_full[i] for i in common_ids])

        paired_results[name] = {
            "n_common": len(common_ids),
            "delta_sensitivity": round(cand_m["sensitivity"] - v4_m["sensitivity"], 4),
            "delta_specificity": round(cand_m["specificity"] - v4_m["specificity"], 4),
            "delta_balanced_accuracy": round(cand_m["balanced_accuracy"] - v4_m["balanced_accuracy"], 4),
            "delta_roc_auc": round(float(cand_auc - v4_auc), 4),
            "delta_f1": round(cand_m["f1"] - v4_m["f1"], 4),
            **err_analysis,
            "bootstrap": boot,
        }
        print(f"[paired vs V4] {name}: n_common={len(common_ids)} "
              f"dAUC={paired_results[name]['delta_roc_auc']} dBalAcc={paired_results[name]['delta_balanced_accuracy']} "
              f"FN_rescued={err_analysis['fn_rescued']} FN_introduced={err_analysis['fn_introduced']} "
              f"FP_removed={err_analysis['fp_removed']} FP_introduced={err_analysis['fp_introduced']}")

    with open(BENCH_DIR / "paired_error_analysis_vs_v4.json", "w", encoding="utf-8") as f:
        json.dump(paired_results, f, indent=2)

    print(f"\nWrote ablation_comparison_table.csv, fold_by_fold_results.csv, paired_error_analysis_vs_v4.json")
    return all_candidates, paired_results


if __name__ == "__main__":
    main()
