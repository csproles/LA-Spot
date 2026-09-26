"""
Evaluate ai-abcd's clinical_abcd.py judgment layer on the SAME development
cohort (n=2090, GroupKFold(5) by group_id) used for frozen V5 and the
ABCD_Improved_Experimental work -- WITHOUT rerunning YOLO/preprocessing,
since every underlying measurement clinical_abcd.py consumes (A_value,
border_irreg/B_circularity, n_significant_defects, color_cv/C_value, the 4
color-fraction values, D_px) was confirmed byte-identical in formula/code to
what dev_feature_table_v2.csv already contains (see
ai_abcd_feature_inventory.md). judge_instance() itself is IMPORTED, unmodified,
from a detached read-only git worktree of origin/ai-abcd -- never
reimplemented/re-thresholded by hand, so there is zero transcription risk.

clinical_abcd.py produces per-criterion booleans (A/B/C concern; D is always
"N/A"), never a combined verdict or continuous score. It has NO trainable
component (fixed a-priori thresholds, not fit on any data) -- so there is no
leakage risk and no fold-wise refitting is needed; the SAME fixed rule is
just evaluated on each of the same 5 GroupKFold folds used elsewhere, purely
to report fold-to-fold balanced-accuracy variability in a comparable format.

Any aggregation of A/B/C into a single verdict is NOT part of ai-abcd's own
design -- clearly labeled as an evaluation-only addition, and more than one
reasonable aggregation is reported so no single arbitrary choice is
presented as "the" ai-abcd result.

Never touches pipeline_v5/, revised_abcd/, Code/, any Evaluation_* directory,
ABCD_Improved_Experimental/'s existing files, or the locked test set. Never
modifies the ai-abcd branch or worktree.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKTREE = REPO_ROOT.parent / "ai-abcd-worktree"
IMPROVED_DIR = REPO_ROOT / "ABCD_Improved_Experimental"

assert WORKTREE.exists(), f"ai-abcd worktree not found at {WORKTREE} -- create it first (read-only, detached)"
sys.path.insert(0, str(WORKTREE / "bulk_analysis"))
from clinical_abcd import judge_instance  # noqa: E402  -- imported UNMODIFIED from ai-abcd, never reimplemented

T0 = time.perf_counter()

# ---------------------------------------------------------------- load data --
dev = pd.read_csv(IMPROVED_DIR / "dev_feature_table_v2.csv")
print(f"Loaded dev_feature_table_v2.csv: {len(dev)} rows")

# Join B_experimental_n_defects (clinical_abcd.py's border "notch count" input)
# from cohort_v2_results.csv -- present there but not in dev_feature_table_v2.csv's
# 17-model-feature columns. Same population, same cached values, no recomputation.
cohort = pd.read_csv(REPO_ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv")
cohort["image_id"] = cohort["image_name"].str.split(".").str[0]
cohort_single = cohort[cohort["evaluation_status"] == "SINGLE_LESION_EVALUABLE"][
    ["image_id", "B_experimental_n_defects"]
].drop_duplicates(subset="image_id", keep="first")

df = dev.merge(cohort_single, on="image_id", how="left", validate="one_to_one")
n_missing_defects = df["B_experimental_n_defects"].isna().sum()
print(f"Joined B_experimental_n_defects: {n_missing_defects} rows missing (should be 0)")
assert n_missing_defects == 0, "every dev row should have a matched cohort_v2_results.csv row -- investigate before proceeding"

# ------------------------------------------------------- apply judge_instance --
records = []
for _, row in df.iterrows():
    j = judge_instance(
        a_value=row["A_value"],
        border_irreg=row["B_circularity"],
        n_significant_defects=int(row["B_experimental_n_defects"]),
        color_cv=row["C_value"],
        pink_red_px=row["red_fraction"], blue_gray_px=row["bluegray_fraction"],
        white_px=row["white_fraction"], black_px=row["dark_fraction"],
        diameter_px=row["D_px"], relative_size_pct=None,  # never affects `concern` (D is always "N/A")
    )
    records.append(j)

judged = pd.DataFrame(records)
df = pd.concat([df.reset_index(drop=True), judged.reset_index(drop=True)], axis=1)

for c in ["A_concern", "B_concern", "C_concern"]:
    assert df[c].isin([True, False]).all(), f"{c} has unexpected non-boolean values -- inspect before proceeding"
assert (df["D_concern"] == "N/A").all(), "D_concern should always be the literal string 'N/A' per clinical_abcd.py"

df["n_concerns_ABC"] = df["A_concern"].astype(int) + df["B_concern"].astype(int) + df["C_concern"].astype(int)

# ---------------------------------------- EVALUATION-ONLY aggregation rules --
# NOT part of ai-abcd's own design (it has no combined verdict at all) --
# added here only to produce a single predicted label for comparison.
df["pred_ANY"] = (df["n_concerns_ABC"] >= 1).astype(int)       # any of A/B/C flags -> concern
df["pred_MAJORITY"] = (df["n_concerns_ABC"] >= 2).astype(int)  # 2-of-3 flags -> concern

y = df["ground_truth_binary"].to_numpy(dtype=int)
groups = df["group_id"].to_numpy()
ids = df["image_id"].to_numpy()
n_total, n_benign, n_malignant = len(y), int((y == 0).sum()), int((y == 1).sum())


def metrics_from_pred(y_true, pred):
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


# ---------------------------------------------- fold-wise stability (no fitting) --
# clinical_abcd.py has no trainable component -- the SAME fixed rule is applied
# to every fold; this only reports how much the FIXED rule's balanced accuracy
# varies across the same 5 group-based folds V5/Improved were evaluated on,
# for a directly comparable CV_Mean/CV_STD figure. No leakage is possible
# since nothing is fit on any fold's data.
gkf = GroupKFold(n_splits=5)
fold_bal_acc = {"pred_ANY": [], "pred_MAJORITY": []}
for fold_i, (_, val_idx) in enumerate(gkf.split(np.zeros(len(y)), y, groups=groups)):
    fy = y[val_idx]
    for rule in ("pred_ANY", "pred_MAJORITY"):
        fp_ = df[rule].to_numpy()[val_idx]
        m = metrics_from_pred(fy, fp_)
        fold_bal_acc[rule].append(m["balanced_accuracy"])

cv_summary = {}
for rule in ("pred_ANY", "pred_MAJORITY"):
    vals = np.array(fold_bal_acc[rule])
    cv_summary[rule] = {"cv_mean_balanced_accuracy": float(np.nanmean(vals)),
                         "cv_std_balanced_accuracy": float(np.nanstd(vals, ddof=1))}

# ---------------------------------------------------------- pseudo-AUC note --
# NOT a true continuous-probability AUC -- clinical_abcd.py never produces a
# probability. n_concerns_ABC (0-3, ordinal count of fired criteria) is the
# closest thing to a "score" available, reported separately and labeled
# explicitly as non-standard; NOT compared on equal footing with V5/
# Improved's calibrated logistic-regression probabilities.
from sklearn.metrics import roc_auc_score
try:
    pseudo_auc = float(roc_auc_score(y, df["n_concerns_ABC"].to_numpy()))
except ValueError:
    pseudo_auc = float("nan")

TOTAL_RUNTIME = time.perf_counter() - T0

# --------------------------------------------------------------- write CSV --
rows_out = []
for rule, label, threshold_desc in [
    ("pred_ANY", "ai-abcd clinical_abcd.py, evaluation-only aggregation: ANY of A/B/C concern", "n_concerns_ABC >= 1"),
    ("pred_MAJORITY", "ai-abcd clinical_abcd.py, evaluation-only aggregation: MAJORITY (2-of-3) of A/B/C concern", "n_concerns_ABC >= 2"),
]:
    m = metrics_from_pred(y, df[rule].to_numpy())
    rows_out.append({
        "Pipeline": label, "N_Total": n_total, "N_Benign": n_benign, "N_Malignant": n_malignant,
        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
        "Accuracy": round(m["accuracy"], 4), "Balanced_Accuracy": round(m["balanced_accuracy"], 4),
        "Sensitivity": round(m["sensitivity"], 4), "Specificity": round(m["specificity"], 4),
        "Precision": round(m["precision"], 4), "F1": round(m["f1"], 4),
        "ROC_AUC": "N/A (no continuous score -- see pseudo-AUC note)",
        "Pseudo_AUC_from_ordinal_concern_count_0to3": round(pseudo_auc, 4),
        "Threshold": threshold_desc,
        "CV_Mean_Balanced_Accuracy": round(cv_summary[rule]["cv_mean_balanced_accuracy"], 4),
        "CV_STD_Balanced_Accuracy": round(cv_summary[rule]["cv_std_balanced_accuracy"], 4),
        "Runtime_sec": round(TOTAL_RUNTIME, 3),
        "Notes": "Aggregation rule NOT part of ai-abcd's own design (it has no combined verdict); "
                 "added for evaluation purposes only. No training/fitting occurred (fixed a-priori "
                 "thresholds) so GroupKFold folds here measure rule stability, not model variance.",
    })

with open(HERE / "ai_abcd_performance.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

# per-image judgments, for transparency / spot-checking
df[["image_id", "group_id", "ground_truth_binary", "A_value", "A_concern", "A_label",
    "B_circularity", "B_experimental_n_defects", "B_concern", "B_label",
    "C_value", "red_fraction", "bluegray_fraction", "white_fraction", "dark_fraction", "C_concern", "C_label",
    "D_px", "D_concern", "n_concerns_ABC", "pred_ANY", "pred_MAJORITY"]].to_csv(
    HERE / "ai_abcd_per_image_judgments.csv", index=False)

print(json.dumps(rows_out, indent=2))
print(f"\nPseudo-AUC (ordinal 0-3 concern count vs ground truth, NOT a true probability AUC): {pseudo_auc:.4f}")
print(f"Total runtime: {TOTAL_RUNTIME:.3f}s")
print("Wrote ai_abcd_performance.csv, ai_abcd_per_image_judgments.csv")
