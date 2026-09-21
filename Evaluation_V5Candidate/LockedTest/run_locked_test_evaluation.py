"""
ONE-TIME locked-test evaluation of frozen V4 vs. the V5 candidate.

Uses:
  - V4: the exact, already-frozen pipeline_v4/frozen_model.pkl (vendored
    read-only copy here, never refit), 6 features, threshold 0.25 --
    identical to Evaluation_V4/freeze_and_locked_test.py's own scoring.
  - V5: the exact, already-fitted Evaluation_V5Candidate/v5_reference_model.pkl
    (fit on all development rows, never refit here), 17 features
    (unchanged from the ablation study), development-selected threshold
    0.25 (unchanged).

The ONLY new computation here is extracting V5's 11 new features for the
locked-test images -- using the exact same, unmodified feature-extraction
functions from Evaluation_FeatureEngineering/extract_features.py that were
already finalized before any locked-test result existed. No feature, no
coefficient, no threshold, no preprocessing step is changed here or as a
result of anything seen in this run.

Locked-test population: identical to Evaluation_V4/freeze_and_locked_test.py
-- the 615 SINGLE_LESION_EVALUABLE images among the 786-image locked test
set (Evaluation_FinalTargeted/Cohort/locked_test_manifest.csv). No case is
excluded based on any result.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

LOCKED_DIR = Path(__file__).resolve().parent
V5_DIR = LOCKED_DIR.parent
ROOT = V5_DIR.parent
FEATENG_DIR = ROOT / "Evaluation_FeatureEngineering"
sys.path.insert(0, str(FEATENG_DIR))
sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))

from extract_features import process_one, ALL_NEW_KEYS  # noqa: E402 -- unmodified, reused as-is
import metrics_lib as ml  # noqa: E402

COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
LOCKED_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "locked_test_manifest.csv"
V4_MODEL_PATH = LOCKED_DIR / "_vendored_v4_frozen_model.pkl"
V5_MODEL_PATH = V5_DIR / "v5_reference_model.pkl"

V4_FEATURES = ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"]
V5_NEW_FEATURES = ["color_entropy", "lab_a_std", "lab_b_std", "red_fraction", "bluegray_fraction",
                    "dark_fraction", "skin_contrast", "solidity", "turning_angle_std", "eccentricity",
                    "D_px_normalized"]
V5_FEATURES = V4_FEATURES + V5_NEW_FEATURES
THRESHOLD = 0.25

EXPECTED_V4_LOCKED = {"TP": 95, "TN": 336, "FP": 141, "FN": 43, "n": 615,
                      "sensitivity": 0.6884057971014492, "specificity": 0.7044025157232704,
                      "roc_auc": 0.7484}


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_locked_evaluable_rows():
    with open(COHORT_RESULTS, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    with open(LOCKED_MANIFEST, newline="", encoding="utf-8") as f:
        locked_rows = list(csv.DictReader(f))
    locked_ids = {r["image_id"] for r in locked_rows}
    path_by_id = {r["image_id"]: r["image_path"] for r in locked_rows}

    by_image = {}
    for r in results:
        image_id = ml.image_id_from_name(r["image_name"])
        if image_id not in locked_ids:
            continue
        by_image.setdefault(image_id, []).append(r)

    status_counts = {"SINGLE_LESION_EVALUABLE": 0, "MULTI_LESION_AMBIGUOUS": 0,
                     "NO_DETECTION": 0, "FAILED": 0}
    evaluable = []
    for image_id, rows in by_image.items():
        status = rows[0]["evaluation_status"] or rows[0]["status"]
        if status in status_counts:
            status_counts[status] += 1
        if status == "SINGLE_LESION_EVALUABLE":
            evaluable.append(rows[0])
    return evaluable, status_counts, len(by_image), path_by_id


def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def compute_metrics(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and prec == prec and sens == sens else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    bal_acc = (sens + spec) / 2
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": n, "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": f1, "accuracy": acc, "balanced_accuracy": bal_acc}


def paired_bootstrap_ci(y, a, b, n_boot=2000, seed=20260918):
    rng = np.random.RandomState(seed)
    n = len(y)
    auc_diffs, bal_diffs = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        auc_a = roc_auc_score(yb, a[idx])
        auc_b = roc_auc_score(yb, b[idx])
        auc_diffs.append(auc_a - auc_b)
        ma = compute_metrics(*confusion(yb, (a[idx] >= THRESHOLD).astype(int)))
        mb = compute_metrics(*confusion(yb, (b[idx] >= THRESHOLD).astype(int)))
        if ma["balanced_accuracy"] == ma["balanced_accuracy"] and mb["balanced_accuracy"] == mb["balanced_accuracy"]:
            bal_diffs.append(ma["balanced_accuracy"] - mb["balanced_accuracy"])

    def ci(vals):
        v = np.array(vals)
        return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    am, alo, ahi = ci(auc_diffs)
    bm, blo, bhi = ci(bal_diffs)
    return {"n_bootstrap": n_boot, "auc_diff_mean": round(am, 4), "auc_diff_ci95": [round(alo, 4), round(ahi, 4)],
            "balacc_diff_mean": round(bm, 4), "balacc_diff_ci95": [round(blo, 4), round(bhi, 4)]}


def main():
    evaluable, status_counts, n_images, path_by_id = build_locked_evaluable_rows()
    print(f"Locked test: {n_images} images total, coverage: {status_counts}")
    print(f"Evaluable (SINGLE_LESION_EVALUABLE): {len(evaluable)}")

    image_ids = [ml.image_id_from_name(r["image_name"]) for r in evaluable]
    y_true = np.array([int(r["ground_truth_binary"]) for r in evaluable])

    # ---- V4: exact frozen model, exact existing 6 features, threshold 0.25 ----
    with open(V4_MODEL_PATH, "rb") as f:
        v4_pipe = pickle.load(f)
    X_v4 = np.array([[to_float(r[f]) for f in V4_FEATURES] for r in evaluable])
    proba_v4 = v4_pipe.predict_proba(X_v4)[:, 1]
    v4_pred = (proba_v4 >= THRESHOLD).astype(int)
    v4_metrics = compute_metrics(*confusion(y_true, v4_pred))
    v4_auc = float(roc_auc_score(y_true, proba_v4))

    print("\n--- V4 reproducibility check against the existing published locked-test result ---")
    for k, expected in EXPECTED_V4_LOCKED.items():
        actual = v4_auc if k == "roc_auc" else v4_metrics[k]
        # roc_auc in the existing published file was rounded to 4 decimals;
        # compare at that same precision rather than full float precision
        tol = 5e-5 if k == "roc_auc" else 1e-6
        status = "OK" if abs(actual - expected) < tol else "MISMATCH"
        print(f"  [{k}] expected={expected} actual={actual} -> {status}")
        if status == "MISMATCH":
            raise RuntimeError(f"V4 locked-test reproduction does not match the existing published result for {k}: "
                                f"expected {expected}, got {actual}. Stopping -- investigate before proceeding.")

    # ---- V5: NEW feature extraction on locked-test images (first time), exact frozen model, threshold 0.25 ----
    print(f"\nExtracting V5's 11 new features for {len(evaluable)} locked-test images "
          f"(unmodified extract_features.process_one, first application to locked test)...")
    new_feature_rows = []
    for i, r in enumerate(evaluable):
        iid = image_ids[i]
        rec = process_one(r, path_by_id[iid])
        new_feature_rows.append(rec)
        if i % 100 == 0 or i == len(evaluable) - 1:
            print(f"  [{i+1}/{len(evaluable)}] {iid}")
    n_feature_errors = sum(1 for r in new_feature_rows if r["_errors"])
    print(f"Feature-extraction errors on locked test: {n_feature_errors}/{len(new_feature_rows)}")

    with open(V5_MODEL_PATH, "rb") as f:
        v5_pipe = pickle.load(f)

    X_v5 = []
    for r_base, r_new in zip(evaluable, new_feature_rows):
        row = {f: to_float(r_base[f]) for f in V4_FEATURES}
        for f in V5_NEW_FEATURES:
            row[f] = r_new[f] if r_new[f] == r_new[f] else np.nan  # nan stays nan
        X_v5.append([row[f] for f in V5_FEATURES])
    X_v5 = np.array(X_v5, dtype=float)

    valid_mask = ~np.isnan(X_v5).any(axis=1)
    n_dropped_v5 = int((~valid_mask).sum())
    print(f"V5 rows with missing feature values (excluded from V5 scoring only): {n_dropped_v5}")

    proba_v5_full = np.full(len(evaluable), np.nan)
    proba_v5_full[valid_mask] = v5_pipe.predict_proba(X_v5[valid_mask])[:, 1]

    y_v5 = y_true[valid_mask]
    proba_v5 = proba_v5_full[valid_mask]
    v5_pred = (proba_v5 >= THRESHOLD).astype(int)
    v5_metrics = compute_metrics(*confusion(y_v5, v5_pred))
    v5_auc = float(roc_auc_score(y_v5, proba_v5))

    print(f"\nV5 locked-test result (n={v5_metrics['n']}): "
          f"sens={v5_metrics['sensitivity']:.4f} spec={v5_metrics['specificity']:.4f} "
          f"bal_acc={v5_metrics['balanced_accuracy']:.4f} auc={v5_auc:.4f}")

    # ---- confirm identical evaluable population for the paired comparison ----
    common_mask = valid_mask  # V4 has no missing features on this population; V5's mask defines the common set
    common_ids = [image_ids[i] for i in range(len(evaluable)) if common_mask[i]]
    print(f"\nCommon evaluable population used for BOTH V4 and V5 paired comparison: {len(common_ids)} "
          f"of {len(evaluable)} locked-test SINGLE_LESION_EVALUABLE images "
          f"({'identical to full evaluable set' if n_dropped_v5 == 0 else f'{n_dropped_v5} excluded due to missing V5 feature values only'})")

    y_common = y_true[common_mask]
    v4_p_common = proba_v4[common_mask]
    v5_p_common = proba_v5_full[common_mask]
    v4_pred_common = (v4_p_common >= THRESHOLD).astype(int)
    v5_pred_common = (v5_p_common >= THRESHOLD).astype(int)
    v4_metrics_common = compute_metrics(*confusion(y_common, v4_pred_common))
    v4_auc_common = float(roc_auc_score(y_common, v4_p_common))

    # ---- paired FN/FP shift ----
    fn_rescued = fn_introduced = fp_corrected = fp_introduced = 0
    for i in range(len(common_ids)):
        yy = y_common[i]
        p4 = v4_pred_common[i]
        p5 = v5_pred_common[i]
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

    net_tp = v5_metrics["TP"] - v4_metrics_common["TP"]
    net_tn = v5_metrics["TN"] - v4_metrics_common["TN"]
    net_fp = v5_metrics["FP"] - v4_metrics_common["FP"]
    net_fn = v5_metrics["FN"] - v4_metrics_common["FN"]

    boot = paired_bootstrap_ci(y_common, v5_p_common, v4_p_common)

    print(f"\nMelanoma false negatives rescued by V5:  {fn_rescued}")
    print(f"New melanoma false negatives introduced: {fn_introduced}")
    print(f"V4 false positives corrected by V5:       {fp_corrected}")
    print(f"New false positives introduced by V5:     {fp_introduced}")
    print(f"Net change: TP {net_tp:+d}  TN {net_tn:+d}  FP {net_fp:+d}  FN {net_fn:+d}")
    print(f"Paired ROC-AUC difference (V5-V4): {v5_auc - v4_auc_common:+.4f}")
    print(f"Paired balanced-accuracy difference (V5-V4): {v5_metrics['balanced_accuracy'] - v4_metrics_common['balanced_accuracy']:+.4f}")
    print(f"Bootstrap (n={boot['n_bootstrap']}): AUC diff = {boot['auc_diff_mean']} "
          f"95% CI {boot['auc_diff_ci95']}; BalAcc diff = {boot['balacc_diff_mean']} "
          f"95% CI {boot['balacc_diff_ci95']}")

    report = {
        "locked_test_total_images": n_images,
        "coverage": status_counts,
        "evaluable_total": len(evaluable),
        "common_paired_population": len(common_ids),
        "v5_rows_dropped_missing_features": n_dropped_v5,
        "V4": {"threshold": THRESHOLD, "roc_auc": round(v4_auc, 4),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in v4_metrics.items()}},
        "V4_on_common_population": {"threshold": THRESHOLD, "roc_auc": round(v4_auc_common, 4),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in v4_metrics_common.items()}},
        "V5": {"threshold": THRESHOLD, "roc_auc": round(v5_auc, 4),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in v5_metrics.items()}},
        "paired_shift": {"fn_rescued": fn_rescued, "fn_introduced": fn_introduced,
                          "fp_corrected": fp_corrected, "fp_introduced": fp_introduced,
                          "net_tp": net_tp, "net_tn": net_tn, "net_fp": net_fp, "net_fn": net_fn},
        "paired_roc_auc_diff": round(v5_auc - v4_auc_common, 4),
        "paired_balanced_accuracy_diff": round(v5_metrics["balanced_accuracy"] - v4_metrics_common["balanced_accuracy"], 4),
        "bootstrap": boot,
    }
    with open(LOCKED_DIR / "locked_test_v5_vs_v4.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # per-image record for audit
    fieldnames = ["image_id", "ground_truth_binary", "v4_proba", "v4_pred", "v5_proba", "v5_pred",
                  "v5_features_valid"]
    with open(LOCKED_DIR / "locked_test_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, iid in enumerate(image_ids):
            w.writerow({
                "image_id": iid, "ground_truth_binary": int(y_true[i]),
                "v4_proba": round(float(proba_v4[i]), 6), "v4_pred": int(v4_pred[i]),
                "v5_proba": round(float(proba_v5_full[i]), 6) if valid_mask[i] else "",
                "v5_pred": int(1 if valid_mask[i] and proba_v5_full[i] >= THRESHOLD else 0) if valid_mask[i] else "",
                "v5_features_valid": bool(valid_mask[i]),
            })

    print(f"\nWrote locked_test_v5_vs_v4.json and locked_test_per_image.csv")
    return report


if __name__ == "__main__":
    main()
