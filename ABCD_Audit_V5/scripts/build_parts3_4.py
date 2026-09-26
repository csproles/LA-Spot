"""Part 3 (TP/TN/FP/FN splits) and Part 4 (statistical feature-failure
analysis) for the V5 ABCD audit. Reads ONLY the reconstructed locked-test
master CSV (ABCD_Audit_V5/V5_locked_test_master.csv, itself built from
unmodified frozen-pipeline code and verified to reproduce the published
TP/TN/FP/FN exactly). Does not touch the model, does not rerun the
classifier, does not tune anything -- this is pure post-hoc analysis of an
already-existing, already-locked result.
"""
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"C:\Users\sirjanaa\Documents\huggingface_projects\Advanced-Melanoma-Detection\ABCD_Audit_V5"
df = pd.read_csv(f"{BASE}\\V5_locked_test_master.csv")
assert len(df) == 615, f"expected 615 rows, got {len(df)}"

FEATURES_17 = ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction",
               "color_entropy", "lab_a_std", "lab_b_std", "red_fraction", "bluegray_fraction",
               "dark_fraction", "skin_contrast", "solidity", "turning_angle_std", "eccentricity",
               "D_px_normalized"]

ABCD_GROUP = {
    "A_value": "A", "B_circularity": "B", "C_value": "C", "D_px": "D",
    "confidence": "Other", "lesion_fraction": "Other",
    "color_entropy": "C", "lab_a_std": "C", "lab_b_std": "C", "red_fraction": "C",
    "bluegray_fraction": "C", "dark_fraction": "C", "skin_contrast": "C",
    "solidity": "B", "turning_angle_std": "B", "eccentricity": "Other",
    "D_px_normalized": "D",
}

# expected clinical direction: does a HIGHER value intuitively suggest melanoma?
# derived from Part 1/2 reading of the code + ABCD clinical convention
EXPECTED_DIRECTION = {
    "A_value": "higher", "B_circularity": "higher", "C_value": "higher", "D_px": "higher",
    "confidence": "not_abcd", "lesion_fraction": "not_abcd",
    "color_entropy": "higher", "lab_a_std": "higher", "lab_b_std": "higher", "red_fraction": "higher",
    "bluegray_fraction": "higher", "dark_fraction": "higher", "skin_contrast": "higher (naive prior)",
    "solidity": "lower", "turning_angle_std": "higher", "eccentricity": "unclear",
    "D_px_normalized": "higher",
}

df["pred_class"] = np.select(
    [(df.v5_pred == 1) & (df.ground_truth_binary == 1),
     (df.v5_pred == 0) & (df.ground_truth_binary == 0),
     (df.v5_pred == 1) & (df.ground_truth_binary == 0),
     (df.v5_pred == 0) & (df.ground_truth_binary == 1)],
    ["TP", "TN", "FP", "FN"], default="?"
)

counts = df["pred_class"].value_counts().to_dict()
print("Confusion counts:", counts)
expected = {"TP": 98, "TN": 340, "FP": 137, "FN": 40}
assert counts == expected, f"MISMATCH: {counts} vs expected {expected}"
print("Confusion counts MATCH published locked-test result exactly.")

out_cols = ["image_id", "ground_truth_binary", "ground_truth_label", "v5_proba", "v5_pred",
            "v4_proba", "v4_pred", "v5_features_valid", "pred_class"] + FEATURES_17

for cls in ["TP", "TN", "FP", "FN"]:
    sub = df[df.pred_class == cls][out_cols].sort_values("v5_proba", ascending=(cls in ("TN", "FN")))
    sub.to_csv(f"{BASE}\\V5_{cls}_cases.csv", index=False)
    print(f"Wrote V5_{cls}_cases.csv: {len(sub)} rows")

# ---------------- Part 4: statistical feature-failure analysis ----------------

def desc(series):
    s = series.dropna()
    if len(s) == 0:
        return dict(mean=np.nan, median=np.nan, std=np.nan, iqr=np.nan, min=np.nan, max=np.nan, n=0)
    q1, q3 = np.percentile(s, [25, 75])
    return dict(mean=s.mean(), median=s.median(), std=s.std(), iqr=q3 - q1, min=s.min(), max=s.max(), n=len(s))

rows = []
tp = df[df.pred_class == "TP"]
tn = df[df.pred_class == "TN"]
fp = df[df.pred_class == "FP"]
fn = df[df.pred_class == "FN"]
benign = df[df.ground_truth_binary == 0]
malignant = df[df.ground_truth_binary == 1]

for feat in FEATURES_17:
    d_tp, d_fn, d_tn, d_fp = desc(tp[feat]), desc(fn[feat]), desc(tn[feat]), desc(fp[feat])
    d_ben, d_mal = desc(benign[feat]), desc(malignant[feat])

    # Mann-Whitney U test, benign vs malignant (two-sided) -- only test actually run;
    # only claim significance where this test says so.
    b_vals, m_vals = benign[feat].dropna(), malignant[feat].dropna()
    if len(b_vals) > 0 and len(m_vals) > 0:
        try:
            # mannwhitneyu(x, y) returns U for x; U/(n_x*n_y) = P(x > y) (+0.5*P(tie)).
            # We want P(malignant_value > benign_value), so malignant must be the FIRST argument.
            u_stat, p_value = stats.mannwhitneyu(m_vals, b_vals, alternative="two-sided")
            auc_locked = u_stat / (len(m_vals) * len(b_vals))  # = P(malignant_value > benign_value)
        except ValueError:
            p_value, auc_locked = np.nan, np.nan
    else:
        p_value, auc_locked = np.nan, np.nan

    # overlap: IQR overlap between benign and malignant
    overlap = "N/A"
    if not any(np.isnan([d_ben["iqr"], d_mal["iqr"]])):
        b_lo, b_hi = d_ben["median"] - d_ben["iqr"] / 2, d_ben["median"] + d_ben["iqr"] / 2
        m_lo, m_hi = d_mal["median"] - d_mal["iqr"] / 2, d_mal["median"] + d_mal["iqr"] / 2
        inter = max(0, min(b_hi, m_hi) - max(b_lo, m_lo))
        union_span = max(b_hi, m_hi) - min(b_lo, m_lo)
        overlap_frac = inter / union_span if union_span > 0 else np.nan
        overlap = f"{overlap_frac:.2f}" if overlap_frac == overlap_frac else "N/A"

    direction_expected = EXPECTED_DIRECTION[feat]
    if auc_locked == auc_locked:  # not nan
        direction_observed = "higher_in_malignant" if auc_locked > 0.5 else (
            "lower_in_malignant" if auc_locked < 0.5 else "no_difference")
    else:
        direction_observed = "N/A"

    contradicts = "unclear" not in direction_expected and "not_abcd" not in direction_expected and (
        ("higher" in direction_expected and direction_observed == "lower_in_malignant") or
        ("lower" in direction_expected and direction_observed == "higher_in_malignant")
    )

    problems = []
    if p_value == p_value and p_value >= 0.05:
        problems.append("no_significant_benign_vs_malignant_difference(Mann-Whitney p>=0.05)")
    if overlap != "N/A" and float(overlap) > 0.5:
        problems.append(f"large_IQR_overlap({overlap})")
    if contradicts:
        problems.append("direction_contradicts_expected_clinical_interpretation")
    if abs(auc_locked - 0.5) < 0.03 if auc_locked == auc_locked else False:
        problems.append("near_chance_locked_test_AUC")
    if direction_expected == "not_abcd":
        problems.append("not_a_clinical_ABCD_signal_by_design")
    potential_problem = "; ".join(problems) if problems else "none_observed"

    rows.append({
        "Feature": feat, "ABCD_Group": ABCD_GROUP[feat],
        "TP_Median": round(d_tp["median"], 4) if d_tp["median"] == d_tp["median"] else "",
        "FN_Median": round(d_fn["median"], 4) if d_fn["median"] == d_fn["median"] else "",
        "TN_Median": round(d_tn["median"], 4) if d_tn["median"] == d_tn["median"] else "",
        "FP_Median": round(d_fp["median"], 4) if d_fp["median"] == d_fp["median"] else "",
        "Malignant_Median": round(d_mal["median"], 4) if d_mal["median"] == d_mal["median"] else "",
        "Benign_Median": round(d_ben["median"], 4) if d_ben["median"] == d_ben["median"] else "",
        "Malignant_Mean": round(d_mal["mean"], 4), "Benign_Mean": round(d_ben["mean"], 4),
        "Malignant_Std": round(d_mal["std"], 4), "Benign_Std": round(d_ben["std"], 4),
        "Malignant_IQR": round(d_mal["iqr"], 4), "Benign_IQR": round(d_ben["iqr"], 4),
        "Malignant_Min": round(d_mal["min"], 4), "Malignant_Max": round(d_mal["max"], 4),
        "Benign_Min": round(d_ben["min"], 4), "Benign_Max": round(d_ben["max"], 4),
        "Benign_vs_Malignant_IQR_Overlap_Fraction": overlap,
        "MannWhitney_p_value": round(p_value, 5) if p_value == p_value else "",
        "Locked_Test_AUC_equivalent": round(auc_locked, 4) if auc_locked == auc_locked else "",
        "Direction_Expected": direction_expected,
        "Direction_Observed": direction_observed,
        "Potential_Problem": potential_problem,
        "Notes": "",
    })

out = pd.DataFrame(rows)
out.to_csv(f"{BASE}\\V5_Feature_Failure_Analysis.csv", index=False)
print(f"Wrote V5_Feature_Failure_Analysis.csv: {len(out)} feature rows")
print(out[["Feature", "Direction_Expected", "Direction_Observed", "Potential_Problem"]].to_string(index=False))
