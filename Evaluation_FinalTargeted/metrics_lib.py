"""Shared, read-only metrics helpers for Phases 3-7. No pipeline code here —
only loading already-computed results and scoring them under different
decision rules.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MERGED_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"


def image_id_from_name(name):
    return Path(name).stem


def load_merged_results():
    with open(MERGED_RESULTS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split_ids(manifest_path):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {r["image_id"] for r in csv.DictReader(f)}


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def build_split_rows(all_results, split_ids):
    """One row per image (the single instance for SINGLE_LESION_EVALUABLE
    images), restricted to the given split's image IDs. Also returns full
    per-image status counts (including NO_DETECTION/MULTI/FAILED) for
    coverage reporting."""
    by_image = {}
    for r in all_results:
        image_id = image_id_from_name(r["image_name"])
        if image_id not in split_ids:
            continue
        by_image.setdefault(image_id, []).append(r)

    evaluable_rows = []
    status_counts = {"SINGLE_LESION_EVALUABLE": 0, "MULTI_LESION_AMBIGUOUS": 0,
                     "NO_DETECTION": 0, "FAILED": 0}
    for image_id, rows in by_image.items():
        status = rows[0]["evaluation_status"] or rows[0]["status"]
        if status in status_counts:
            status_counts[status] += 1
        if status == "SINGLE_LESION_EVALUABLE":
            evaluable_rows.append(rows[0])

    return evaluable_rows, status_counts, len(by_image)


def confusion_from_predictions(rows, predict_fn):
    """predict_fn(row) -> 1 (positive/HIGH-equivalent) or 0 (negative/LOW-equivalent)."""
    tp = tn = fp = fn = 0
    for r in rows:
        gt = int(r["ground_truth_binary"])
        pred = predict_fn(r)
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
    return tp, tn, fp, fn


def compute_metrics(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and prec == prec and sens == sens and (prec + sens) > 0 else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    bal_acc = (sens + spec) / 2 if sens == sens and spec == spec else float("nan")
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": n,
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": f1, "accuracy": acc, "balanced_accuracy": bal_acc,
    }


def v2_predict(row):
    """Existing, unchanged V2 decision: risk_level HIGH -> positive."""
    return 1 if row["provisional_prediction"] == "positive" else 0


CURRENT_A_THRESHOLD = 0.20  # matches revised_abcd/revised_asymmetry.py, unchanged


def infer_critical_override(row):
    """results.csv does not store the raw dangerous-color-fraction values
    used by score.py's critical-color override, only the final risk_level
    and the three concern booleans. But the override's EFFECT is fully
    recoverable: if risk_level is HIGH while fewer than 2 of A/B/C concern
    flags are true (at the threshold actually used when this row was
    scored, i.e. 0.20 for A), the only way score.py's combine logic could
    have produced HIGH is the critical-color override firing. This lets a
    different A threshold be tested later while faithfully preserving
    whatever the override actually did for this row, without needing the
    underlying color fractions."""
    a = row["A_concern"] == "True"
    b = row["B_circularity_concern"] == "True"
    c = row["C_concern"] == "True"
    concerns_at_saved_threshold = sum([a, b, c])
    is_high = row["provisional_prediction"] == "positive"
    return is_high and concerns_at_saved_threshold < 2


def predict_with_A_threshold(row, a_threshold, override_fired):
    a_value = to_float(row["A_value"])
    if a_value is None:
        a_concern = row["A_concern"] == "True"  # fall back to saved flag if value missing
    else:
        a_concern = a_value > a_threshold
    b_concern = row["B_circularity_concern"] == "True"
    c_concern = row["C_concern"] == "True"
    concerns = sum([a_concern, b_concern, c_concern])
    if override_fired:
        return 1
    return 1 if concerns >= 2 else 0
