"""STEP 7-8 — Fit the final frozen V4 model on ALL of development (no CV
splitting now — CV was only for model/feature/threshold selection), save
it, then run the locked test exactly once. Locked-test labels/results are
never used for any fitting or selection above this point.
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
DEV_FEATURE_TABLE = ROOT / "Evaluation_V4" / "FeatureTable" / "development_feature_table.csv"
TEST_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "locked_test_manifest.csv"
MODEL_OUT = ROOT / "pipeline_v4" / "frozen_model.pkl"
OUT_DIR = ROOT / "Evaluation_V4" / "LockedTest"

sys.path.insert(0, str(ROOT))
from pipeline_v4.decision_model import V4_CONFIG, featurize  # noqa: E402

sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))
from metrics_lib import v2_predict, infer_critical_override, predict_with_A_threshold, CURRENT_A_THRESHOLD  # noqa: E402
sys.path.insert(0, str(ROOT))
from pipeline_v3.decision import v3_predict_from_row  # noqa: E402


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def image_id_from_name(name):
    return Path(name).stem


def fit_final_model():
    with open(DEV_FEATURE_TABLE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[feat]) for feat in V4_CONFIG["features"]] for r in rows])
    y = np.array([int(r["ground_truth_binary"]) for r in rows])
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=20260918))])
    pipe.fit(X, y)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Fitted final V4 model on {len(rows)} development rows, saved to {MODEL_OUT}")
    return pipe


def build_test_rows():
    with open(COHORT_RESULTS, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    with open(TEST_MANIFEST, newline="", encoding="utf-8") as f:
        test_ids = {r["image_id"] for r in csv.DictReader(f)}

    by_image = {}
    for r in results:
        image_id = image_id_from_name(r["image_name"])
        if image_id not in test_ids:
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
    return evaluable, status_counts, len(by_image)


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


def main():
    pipe = fit_final_model()

    evaluable, status_counts, n_images = build_test_rows()
    print(f"Locked test: {n_images} images total, coverage: {status_counts}")

    y_true = np.array([int(r["ground_truth_binary"]) for r in evaluable])

    # V2 (unchanged)
    v2_pred = np.array([v2_predict(r) for r in evaluable])
    v2_metrics = compute_metrics(*confusion(y_true, v2_pred))

    # V3 (frozen, A>0.14)
    for r in evaluable:
        r["_override_fired"] = infer_critical_override(r)
    v3_pred = np.array([v3_predict_from_row(r, r["_override_fired"]) for r in evaluable])
    v3_metrics = compute_metrics(*confusion(y_true, v3_pred))

    # V4 (frozen, logistic regression + threshold 0.25)
    X_test = np.array([[to_float(r["A_value"]), to_float(r["B_circularity"]), to_float(r["C_value"]),
                        to_float(r["D_px"]), to_float(r["confidence"]), to_float(r["lesion_fraction"])]
                       for r in evaluable])
    proba_v4 = pipe.predict_proba(X_test)[:, 1]
    v4_pred = (proba_v4 >= V4_CONFIG["operating_threshold"]).astype(int)
    v4_metrics = compute_metrics(*confusion(y_true, v4_pred))

    from sklearn.metrics import roc_auc_score
    v4_auc = float(roc_auc_score(y_true, proba_v4))

    report = {
        "phase": "V4 locked test (run once)",
        "total_locked_test_images": n_images,
        "coverage": status_counts,
        "V2": v2_metrics, "V3": v3_metrics, "V4": v4_metrics,
        "V4_locked_test_roc_auc": round(v4_auc, 4),
    }
    print(json.dumps(report, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "locked_test_v2_v3_v4.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
