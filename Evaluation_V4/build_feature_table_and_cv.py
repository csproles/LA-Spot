"""STEPS 1-4 — V4 feature table + patient-aware cross-validation.

READ-ONLY with respect to V2, V3, Evaluation_3500, Evaluation_FinalTargeted's
existing results, YOLO, segmentation, and the locked-test split. Reuses the
EXACT same development_manifest.csv / locked_test_manifest.csv and the same
cohort_v2_results.csv already produced for V3 — nothing is regenerated.

Locked test is never touched by this script.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
DEV_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "development_manifest.csv"
OUT_DIR = ROOT / "Evaluation_V4" / "FeatureTable"
CV_OUT_DIR = ROOT / "Evaluation_V4" / "CrossValidation"

N_FOLDS = 5
SEED = 20260918


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def image_id_from_name(name):
    return Path(name).stem


class UnionFind:
    """Identical grouping logic to Evaluation_FinalTargeted/split_dev_test.py,
    reapplied here (not imported, to keep V4 self-contained) restricted to
    development images only, so patient-aware CV respects the same
    patient/lesion non-crossing rule the original split used."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_dev_groups(dev_rows):
    uf = UnionFind()
    for r in dev_rows:
        img_key = ("img", r["image_id"])
        uf.find(img_key)
        if r["patient_id"].strip():
            uf.union(img_key, ("pat", r["patient_id"]))
        if r["lesion_id"].strip():
            uf.union(img_key, ("les", r["lesion_id"]))
    # stringify the union-find root (a tuple like ("img", "ISIC_...")) so it's
    # a plain hashable scalar GroupKFold can bincount/encode directly
    return {r["image_id"]: "|".join(map(str, uf.find(("img", r["image_id"])))) for r in dev_rows}


def build_feature_table():
    with open(COHORT_RESULTS, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    with open(DEV_MANIFEST, newline="", encoding="utf-8") as f:
        dev_manifest_rows = list(csv.DictReader(f))
    dev_ids = {r["image_id"] for r in dev_manifest_rows}
    group_of = build_dev_groups(dev_manifest_rows)

    by_image = {}
    for r in results:
        image_id = image_id_from_name(r["image_name"])
        if image_id not in dev_ids:
            continue
        by_image.setdefault(image_id, []).append(r)

    table = []
    status_counts = {"SINGLE_LESION_EVALUABLE": 0, "MULTI_LESION_AMBIGUOUS": 0,
                     "NO_DETECTION": 0, "FAILED": 0}
    for image_id, rows in by_image.items():
        status = rows[0]["evaluation_status"] or rows[0]["status"]
        if status in status_counts:
            status_counts[status] += 1
        if status != "SINGLE_LESION_EVALUABLE":
            continue
        r = rows[0]
        table.append({
            "image_id": image_id,
            "group_id": group_of[image_id],
            "ground_truth_binary": int(r["ground_truth_binary"]),
            "A_value": to_float(r["A_value"]),
            "B_circularity": to_float(r["B_circularity"]),
            "B_experimental": to_float(r["B_experimental"]),
            "C_value": to_float(r["C_value"]),
            "D_px": to_float(r["D_px"]),
            "lesion_fraction": to_float(r["lesion_fraction"]),
            "confidence": to_float(r["confidence"]),
        })

    with open(OUT_DIR / "development_feature_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)

    print(f"Development coverage: {status_counts}")
    print(f"Feature table rows (SINGLE_LESION_EVALUABLE): {len(table)}")
    n_groups = len(set(r["group_id"] for r in table))
    print(f"Distinct patient/lesion groups in development: {n_groups}")
    return table, status_counts


FEATURE_SETS = {
    "core_ABC": ["A_value", "B_circularity", "C_value"],
    "core_ABC_plus_D": ["A_value", "B_circularity", "C_value", "D_px"],
    "core_ABC_plus_confidence": ["A_value", "B_circularity", "C_value", "confidence"],
    "full_ABC_D_conf_fraction": ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"],
}

MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "shallow_decision_tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
}


def run_cv(table):
    y = np.array([r["ground_truth_binary"] for r in table])
    groups = np.array([r["group_id"] for r in table])
    gkf = GroupKFold(n_splits=N_FOLDS)

    all_results = {}
    for fs_name, fs_fields in FEATURE_SETS.items():
        X_full = np.array([[r[f] if r[f] is not None else np.nan for f in fs_fields] for r in table])
        # drop rows with any missing feature value for this feature set (D_px/confidence
        # should be fully populated for SINGLE_LESION_EVALUABLE rows, but guard anyway)
        valid_mask = ~np.isnan(X_full).any(axis=1)
        X = X_full[valid_mask]
        y_fs = y[valid_mask]
        groups_fs = groups[valid_mask]
        n_dropped = (~valid_mask).sum()

        for model_name, model_fn in MODELS.items():
            oof_proba = np.full(len(y_fs), np.nan)
            fold_metrics = []

            for train_idx, val_idx in gkf.split(X, y_fs, groups=groups_fs):
                pipe = Pipeline([("scale", StandardScaler()), ("clf", model_fn())])
                pipe.fit(X[train_idx], y_fs[train_idx])
                proba = pipe.predict_proba(X[val_idx])[:, 1]
                oof_proba[val_idx] = proba

                pred_05 = (proba >= 0.5).astype(int)
                tp = int(((pred_05 == 1) & (y_fs[val_idx] == 1)).sum())
                tn = int(((pred_05 == 0) & (y_fs[val_idx] == 0)).sum())
                fp = int(((pred_05 == 1) & (y_fs[val_idx] == 0)).sum())
                fn = int(((pred_05 == 0) & (y_fs[val_idx] == 1)).sum())
                sens = tp / (tp + fn) if (tp + fn) else float("nan")
                spec = tn / (tn + fp) if (tn + fp) else float("nan")
                bal_acc = (sens + spec) / 2
                fold_metrics.append({"sensitivity": sens, "specificity": spec, "balanced_accuracy": bal_acc})

            auc = roc_auc_score(y_fs, oof_proba)
            sens_arr = [m["sensitivity"] for m in fold_metrics]
            spec_arr = [m["specificity"] for m in fold_metrics]
            bal_arr = [m["balanced_accuracy"] for m in fold_metrics]

            key = f"{fs_name}__{model_name}"
            all_results[key] = {
                "feature_set": fs_name, "features": fs_fields, "model": model_name,
                "n_rows_used": int(valid_mask.sum()), "n_dropped_missing": int(n_dropped),
                "oof_roc_auc": round(float(auc), 4),
                "cv_sensitivity_mean_at_0.5": round(float(np.mean(sens_arr)), 4),
                "cv_sensitivity_std_at_0.5": round(float(np.std(sens_arr)), 4),
                "cv_specificity_mean_at_0.5": round(float(np.mean(spec_arr)), 4),
                "cv_specificity_std_at_0.5": round(float(np.std(spec_arr)), 4),
                "cv_balanced_accuracy_mean_at_0.5": round(float(np.mean(bal_arr)), 4),
                "cv_balanced_accuracy_std_at_0.5": round(float(np.std(bal_arr)), 4),
                "oof_probabilities": oof_proba.tolist(),
                "y_true": y_fs.tolist(),
            }
            print(f"{key}: AUC={auc:.4f}  bal_acc@0.5={np.mean(bal_arr):.4f}+-{np.std(bal_arr):.4f}  "
                  f"sens@0.5={np.mean(sens_arr):.4f}  spec@0.5={np.mean(spec_arr):.4f}")

    return all_results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, status_counts = build_feature_table()
    cv_results = run_cv(table)

    # save without the huge OOF arrays in the human-readable summary
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("oof_probabilities", "y_true")}
              for k, v in cv_results.items()}
    with open(CV_OUT_DIR / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"development_coverage": status_counts, "candidates": summary}, f, indent=2)

    # save full OOF predictions separately (needed for Step 5 threshold search)
    with open(CV_OUT_DIR / "oof_predictions_full.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f)

    print(f"\nWrote feature table and CV results to {OUT_DIR} / {CV_OUT_DIR}")
    return cv_results


if __name__ == "__main__":
    main()
