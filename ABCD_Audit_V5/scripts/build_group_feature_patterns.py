"""V5 ABCD audit -- follow-up analysis: group-level and case-level feature
patterns across TP/TN/FP/FN on the locked test set (n=615). Read-only
analysis of already-reconstructed, already-verified data
(V5_locked_test_master.csv, VisualReview/*/feature_values.json). Does not
touch the model, formulas, thresholds, or locked-test results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"C:\Users\sirjanaa\Documents\huggingface_projects\Advanced-Melanoma-Detection\ABCD_Audit_V5")

df = pd.read_csv(BASE / "V5_locked_test_master.csv")
assert len(df) == 615

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

# known audit findings (Parts 1/2/4/6-10 of the prior audit) -- used only to
# ANNOTATE observed patterns, never to override what the data actually shows.
KNOWN_CONCERNS = {
    "B_circularity": "Univariate direction is correct, but the multivariate model coefficient is sign-flipped (multicollinearity with excluded isoperimetric_ratio/solidity). Do not read a raw group difference here as evidence the model 'uses' this feature correctly.",
    "D_px_normalized": "Near-duplicate of lesion_fraction (rho=0.989, prior audit). Any pattern seen here should be cross-checked against lesion_fraction before treating it as independent evidence.",
    "lesion_fraction": "Near-duplicate of D_px_normalized. Possible photo-framing/zoom confound rather than true lesion size -- a strong-looking pattern may reflect photography, not biology.",
    "eccentricity": "Chance-level univariate signal on both dev and locked cohorts in the prior audit (AUC ~0.50). Any apparent group pattern here is a strong candidate for POSSIBLE_OUTLIER_DRIVEN_PATTERN rather than real signal.",
    "dark_fraction": "~90%+ of values are exactly 0 in the prior audit; step-function internal floor (0.12). Any nonzero-driven pattern is likely a handful of cases, not a population-level shift.",
    "bluegray_fraction": "Step-function internal floor (0.15); most values are exactly 0. Same outlier-driven risk as dark_fraction.",
    "red_fraction": "Both class medians are ~0 in the prior audit; any AUC/percentile signal is concentrated in a sparse minority of cases.",
    "skin_contrast": "Prior audit found a reproducible but clinically counter-intuitive direction (lower in malignant) across two independent cohorts -- treat direction findings here as confirmatory of a real (if unexplained) pattern, not a new anomaly.",
    "A_value": "Prior audit found segmentation-sensitivity risks (edge-clamped crop, small-mask PCA instability) that a population-level pattern cannot distinguish from genuine asymmetry -- check case-level flags against known small/edge-adjacent lesions.",
    "D_px": "Never a physical measurement (raw pixels, framing/zoom-sensitive) -- an apparent group pattern may reflect image resolution/framing rather than diameter.",
    "confidence": "Not a clinical ABCD signal by design (YOLO detection confidence) -- expected to be near-chance; a strong pattern here would itself be noteworthy (e.g. errors concentrated in low-confidence detections).",
}

df["pred_class"] = np.select(
    [(df.v5_pred == 1) & (df.ground_truth_binary == 1),
     (df.v5_pred == 0) & (df.ground_truth_binary == 0),
     (df.v5_pred == 1) & (df.ground_truth_binary == 0),
     (df.v5_pred == 0) & (df.ground_truth_binary == 1)],
    ["TP", "TN", "FP", "FN"], default="?"
)
assert (df.pred_class == "?").sum() == 0

TP = df[df.pred_class == "TP"]
TN = df[df.pred_class == "TN"]
FP = df[df.pred_class == "FP"]
FN = df[df.pred_class == "FN"]
MAL = df[df.ground_truth_binary == 1]   # malignant ground truth = TP + FN
BEN = df[df.ground_truth_binary == 0]   # benign ground truth   = TN + FP
assert len(TP) == 98 and len(TN) == 340 and len(FP) == 137 and len(FN) == 40
assert len(MAL) == 138 and len(BEN) == 477

GROUPS = {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "Malignant": MAL, "Benign": BEN, "All": df}

# ---------------------------------------------------------------------------
# Part 1: full descriptive stats per feature per group
# ---------------------------------------------------------------------------

def describe(series):
    s = series.dropna()
    if len(s) == 0:
        return dict(count=0, mean=np.nan, median=np.nan, std=np.nan, q1=np.nan, q3=np.nan, min=np.nan, max=np.nan)
    q1, q3 = np.percentile(s, [25, 75])
    return dict(count=len(s), mean=s.mean(), median=s.median(), std=s.std(), q1=q1, q3=q3, min=s.min(), max=s.max())

desc_rows = []
for feat in FEATURES_17:
    for gname, gdf in GROUPS.items():
        d = describe(gdf[feat])
        d.update(Feature=feat, ABCD_Group=ABCD_GROUP[feat], Outcome_Group=gname)
        desc_rows.append(d)
desc_df = pd.DataFrame(desc_rows)[["Feature", "ABCD_Group", "Outcome_Group", "count", "mean", "median",
                                    "std", "q1", "q3", "min", "max"]]
desc_df.to_csv(BASE / "V5_Group_Level_Descriptive_Stats.csv", index=False)
print(f"Wrote V5_Group_Level_Descriptive_Stats.csv: {len(desc_df)} rows "
      f"({len(FEATURES_17)} features x {len(GROUPS)} groups)")

# ---------------------------------------------------------------------------
# Percentile machinery: overall (all 615) and within-true-class (benign/malignant)
# ---------------------------------------------------------------------------

def percentile_rank(value, population):
    """% of population <= value (0-100). NaN-safe."""
    pop = population.dropna().values
    if len(pop) == 0 or pd.isna(value):
        return np.nan
    return float((pop <= value).sum()) / len(pop) * 100.0

def tukey_fences(population):
    pop = population.dropna()
    if len(pop) == 0:
        return np.nan, np.nan
    q1, q3 = np.percentile(pop, [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

# precompute overall + within-class percentile lookups are done per-value via percentile_rank
CLASS_POP = {0: BEN, 1: MAL}  # ground_truth_binary -> the population to compare a case's TRUE class against

def case_unusual_flags(row, basis="class"):
    """Return dict {feature: (percentile, flag)} where flag in
    {UNUSUALLY_HIGH, UNUSUALLY_LOW, OUTLIER_HIGH, OUTLIER_LOW, ''}.
    basis='class': percentile within the case's own ground-truth class population
    (this is the 'more useful' framing per the audit spec's own example).
    """
    pop_df = CLASS_POP[row.ground_truth_binary] if basis == "class" else df
    out = {}
    for feat in FEATURES_17:
        val = row[feat]
        pct = percentile_rank(val, pop_df[feat])
        lo_fence, hi_fence = tukey_fences(pop_df[feat])
        flag = ""
        if pd.notna(val) and pd.notna(hi_fence) and val > hi_fence:
            flag = "OUTLIER_HIGH"
        elif pd.notna(val) and pd.notna(lo_fence) and val < lo_fence:
            flag = "OUTLIER_LOW"
        elif pd.notna(pct) and pct >= 90:
            flag = "UNUSUALLY_HIGH"
        elif pd.notna(pct) and pct <= 10:
            flag = "UNUSUALLY_LOW"
        out[feat] = (pct, flag)
    return out

# ---------------------------------------------------------------------------
# Part 4/5/6/10: per-feature, per-outcome-group "unusual" percentages
# (within TRUE CLASS population, per the audit spec's preferred framing)
# ---------------------------------------------------------------------------

def group_unusual_percentages(group_df, feature, true_class_binary):
    pop = CLASS_POP[true_class_binary][feature]
    vals = group_df[feature]
    pcts = vals.apply(lambda v: percentile_rank(v, pop))
    n = pcts.notna().sum()
    if n == 0:
        return np.nan, np.nan
    hi = (pcts >= 90).sum() / n * 100.0
    lo = (pcts <= 10).sum() / n * 100.0
    return hi, lo

# ---------------------------------------------------------------------------
# Direction / effect-size machinery (Mann-Whitney, ORIENTATION VERIFIED:
# mannwhitneyu(x, y) returns U for x; U/(n_x*n_y) = P(x_value > y_value).
# See prior audit note -- this exact bug was caught and fixed there.
# ---------------------------------------------------------------------------

def compare(group_a, group_b, feature, label_a, label_b):
    """Returns dict: auc (=P(a>b)), p_value, cohens_d, direction string, median_a, median_b."""
    a = group_a[feature].dropna()
    b = group_b[feature].dropna()
    if len(a) < 2 or len(b) < 2:
        return dict(auc=np.nan, p=np.nan, cohens_d=np.nan, direction="insufficient_data",
                    median_a=a.median() if len(a) else np.nan, median_b=b.median() if len(b) else np.nan)
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))  # P(a_value > b_value)
    except ValueError:
        auc, p = np.nan, np.nan
    pooled_std = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) / (len(a) + len(b) - 2))
    cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan
    if auc == auc:
        if p == p and p < 0.05 and abs(auc - 0.5) >= 0.05:
            direction = f"higher_in_{label_a}" if auc > 0.5 else f"higher_in_{label_b}"
        else:
            direction = "no_clear_difference"
    else:
        direction = "insufficient_data"
    return dict(auc=auc, p=p, cohens_d=cohens_d, direction=direction,
                median_a=a.median(), median_b=b.median())

# ---------------------------------------------------------------------------
# Build the main per-feature pattern table
# ---------------------------------------------------------------------------

pattern_rows = []
for feat in FEATURES_17:
    tp_vs_fn = compare(TP, FN, feat, "TP", "FN")     # within malignant: what makes TP different from FN
    tn_vs_fp = compare(TN, FP, feat, "TN", "FP")     # within benign: what makes TN different from FP
    tp_vs_fp = compare(TP, FP, feat, "TP", "FP")     # both "predicted positive"
    tn_vs_fn = compare(TN, FN, feat, "TN", "FN")     # both "predicted negative"
    mal_vs_ben = compare(MAL, BEN, feat, "malignant", "benign")

    fp_hi, fp_lo = group_unusual_percentages(FP, feat, 0)   # FP's true class = benign
    fn_hi, fn_lo = group_unusual_percentages(FN, feat, 1)   # FN's true class = malignant
    tp_hi, tp_lo = group_unusual_percentages(TP, feat, 1)   # TP's true class = malignant
    tn_hi, tn_lo = group_unusual_percentages(TN, feat, 0)   # TN's true class = benign

    # variability / outlier-driven checks
    def cv(s):
        s = s.dropna()
        return (s.std() / abs(s.mean())) if len(s) and s.mean() != 0 else np.nan
    zero_frac_all = (df[feat] == 0).mean()
    high_variability = any(cv(g[feat]) is not None and cv(g[feat]) == cv(g[feat]) and cv(g[feat]) > 1.0
                            for g in (TP, TN, FP, FN))
    sparse_driven = zero_frac_all > 0.5  # majority-zero features: any signal is minority-driven

    # -------- label assignment (only when supported by data) --------
    labels = []
    STRONG = 0.15
    if mal_vs_ben["p"] == mal_vs_ben["p"] and mal_vs_ben["p"] < 0.05 and abs(mal_vs_ben["auc"] - 0.5) >= STRONG:
        labels.append("STRONG_CLASS_SEPARATION")
    if tn_vs_fp["p"] == tn_vs_fp["p"] and tn_vs_fp["p"] < 0.05 and abs(tn_vs_fp["auc"] - 0.5) >= STRONG:
        labels.append("STRONG_ERROR_SEPARATION")
    if tp_vs_fn["p"] == tp_vs_fn["p"] and tp_vs_fn["p"] < 0.05 and abs(tp_vs_fn["auc"] - 0.5) >= STRONG:
        if "STRONG_ERROR_SEPARATION" not in labels:
            labels.append("STRONG_ERROR_SEPARATION")

    if tn_vs_fp["direction"] == "higher_in_FP":
        labels.append("HIGH_IN_FP")
    elif tn_vs_fp["direction"] == "higher_in_TN":
        labels.append("LOW_IN_FP")
        labels.append("HIGH_IN_TN")
    if tp_vs_fn["direction"] == "higher_in_FN":
        labels.append("HIGH_IN_FN")
    elif tp_vs_fn["direction"] == "higher_in_TP":
        labels.append("LOW_IN_FN")
        labels.append("HIGH_IN_TP")

    if sparse_driven or (fp_hi == fp_hi and fp_hi > 0 and zero_frac_all > 0.3) or \
       (fn_hi == fn_hi and fn_hi > 0 and zero_frac_all > 0.3):
        labels.append("POSSIBLE_OUTLIER_DRIVEN_PATTERN")
    if high_variability:
        labels.append("HIGH_VARIABILITY")
    if not labels:
        labels.append("NO_CLEAR_PATTERN")

    labels = list(dict.fromkeys(labels))  # de-dup, preserve order

    concern = KNOWN_CONCERNS.get(feat, "")

    pattern_rows.append({
        "Feature": feat, "ABCD_Group": ABCD_GROUP[feat],
        "TP_Median": round(tp_vs_fn["median_a"], 4) if tp_vs_fn["median_a"] == tp_vs_fn["median_a"] else "",
        "TN_Median": round(tn_vs_fp["median_a"], 4) if tn_vs_fp["median_a"] == tn_vs_fp["median_a"] else "",
        "FP_Median": round(tn_vs_fp["median_b"], 4) if tn_vs_fp["median_b"] == tn_vs_fp["median_b"] else "",
        "FN_Median": round(tp_vs_fn["median_b"], 4) if tp_vs_fn["median_b"] == tp_vs_fn["median_b"] else "",
        "TP_vs_FN_Direction": tp_vs_fn["direction"],
        "TP_vs_FN_AUC_equiv": round(tp_vs_fn["auc"], 4) if tp_vs_fn["auc"] == tp_vs_fn["auc"] else "",
        "TP_vs_FN_p": round(tp_vs_fn["p"], 5) if tp_vs_fn["p"] == tp_vs_fn["p"] else "",
        "TN_vs_FP_Direction": tn_vs_fp["direction"],
        "TN_vs_FP_AUC_equiv": round(tn_vs_fp["auc"], 4) if tn_vs_fp["auc"] == tn_vs_fp["auc"] else "",
        "TN_vs_FP_p": round(tn_vs_fp["p"], 5) if tn_vs_fp["p"] == tn_vs_fp["p"] else "",
        "TP_vs_FP_Direction": tp_vs_fp["direction"],
        "TN_vs_FN_Direction": tn_vs_fn["direction"],
        "Malignant_vs_Benign_Direction": mal_vs_ben["direction"],
        "Malignant_vs_Benign_AUC_equiv": round(mal_vs_ben["auc"], 4) if mal_vs_ben["auc"] == mal_vs_ben["auc"] else "",
        "FP_Unusually_High_Percent": round(fp_hi, 1) if fp_hi == fp_hi else "",
        "FP_Unusually_Low_Percent": round(fp_lo, 1) if fp_lo == fp_lo else "",
        "FN_Unusually_High_Percent": round(fn_hi, 1) if fn_hi == fn_hi else "",
        "FN_Unusually_Low_Percent": round(fn_lo, 1) if fn_lo == fn_lo else "",
        "TP_Unusually_High_Percent": round(tp_hi, 1) if tp_hi == tp_hi else "",
        "TP_Unusually_Low_Percent": round(tp_lo, 1) if tp_lo == tp_lo else "",
        "TN_Unusually_High_Percent": round(tn_hi, 1) if tn_hi == tn_hi else "",
        "TN_Unusually_Low_Percent": round(tn_lo, 1) if tn_lo == tn_lo else "",
        "Zero_Value_Fraction_All": round(zero_frac_all, 3),
        "Labels": ";".join(labels),
        "Main_Pattern": labels[0] if labels else "NO_CLEAR_PATTERN",
        "Concern": concern,
    })

pattern_df = pd.DataFrame(pattern_rows)
pattern_df.to_csv(BASE / "V5_Group_Feature_Patterns.csv", index=False)
print(f"Wrote V5_Group_Feature_Patterns.csv: {len(pattern_df)} rows")
print(pattern_df[["Feature", "Labels"]].to_string(index=False))

# ---------------------------------------------------------------------------
# FP ranked table (vs TN) and FN ranked table (vs TP) -- for the .md
# ---------------------------------------------------------------------------

def ranked_table(pattern_df, unusual_hi_col, unusual_lo_col, auc_col, median_self_col, median_other_col):
    t = pattern_df.copy()
    t["_effect"] = (t[auc_col].apply(lambda x: abs(x - 0.5) if x == x else 0))
    return t.sort_values("_effect", ascending=False)[
        ["Feature", "ABCD_Group", median_self_col, median_other_col, auc_col, unusual_hi_col, unusual_lo_col, "Concern"]
    ]

fp_ranked = ranked_table(pattern_df, "FP_Unusually_High_Percent", "FP_Unusually_Low_Percent",
                          "TN_vs_FP_AUC_equiv", "FP_Median", "TN_Median")
fn_ranked = ranked_table(pattern_df, "FN_Unusually_High_Percent", "FN_Unusually_Low_Percent",
                          "TP_vs_FN_AUC_equiv", "FN_Median", "TP_Median")

fp_ranked.to_csv(BASE / "_fp_ranked_supporting.csv", index=False)
fn_ranked.to_csv(BASE / "_fn_ranked_supporting.csv", index=False)

# ---------------------------------------------------------------------------
# Part 8/11: case-level flags for all 42 VisualReview cases
# ---------------------------------------------------------------------------

vr_dir = BASE / "VisualReview"
case_folders = sorted([p for p in vr_dir.iterdir() if p.is_dir()])
print(f"\nFound {len(case_folders)} VisualReview case folders")

# preserve any existing human notes from V5_Manual_Visual_Review.csv
mvr_path = BASE / "V5_Manual_Visual_Review.csv"
existing_notes = {}
if mvr_path.exists():
    mvr = pd.read_csv(mvr_path)
    for _, r in mvr.iterrows():
        note = str(r.get("Notes_Human", "")).strip()
        if note and note.lower() != "nan":
            existing_notes[r["Image_ID"]] = note

case_report_rows = []
case_text_blocks = []

for folder in case_folders:
    jpath = folder / "feature_values.json"
    with open(jpath, encoding="utf-8") as f:
        j = json.load(f)
    image_id = j["image_id"]
    row = df[df.image_id == image_id]
    if len(row) == 0:
        print(f"  WARNING: {image_id} not found in master CSV, skipping")
        continue
    row = row.iloc[0]
    pred_class = row.pred_class

    flags = case_unusual_flags(row, basis="class")
    # rank by distance from 50th percentile, most extreme first
    ranked = sorted(flags.items(), key=lambda kv: (abs(kv[1][0] - 50) if kv[1][0] == kv[1][0] else -1), reverse=True)
    top = [f for f, (pct, flg) in ranked if flg][:5]
    if len(top) < 3:
        # fall back to top-5 by extremity even without a formal flag, so every case gets >=3 entries
        top = [f for f, _ in ranked[:5]]

    hi_feats = [f for f in top if flags[f][1] in ("UNUSUALLY_HIGH", "OUTLIER_HIGH")]
    lo_feats = [f for f in top if flags[f][1] in ("UNUSUALLY_LOW", "OUTLIER_LOW")]

    class_name = "benign" if row.ground_truth_binary == 0 else "malignant"

    def fmt(feats):
        return "; ".join(f"{f}={flags[f][0]:.0f}th pct among {class_name}" for f in feats) if feats else "(none)"

    # coarse pattern description
    abcd_involved = sorted(set(ABCD_GROUP[f] for f in top))
    likely_seg = ""
    la = row.get("lesion_area_px", np.nan)
    area_pop = BEN["lesion_area_px"] if row.ground_truth_binary == 0 else MAL["lesion_area_px"]
    if pd.notna(la) and la < area_pop.quantile(0.05):
        likely_seg = "very small lesion area (bottom 5% of true class) -- segmentation-quality risk per Part 6/7 audit"

    case_report_rows.append({
        "Image_ID": image_id, "Case_Type": pred_class, "Ground_Truth": row.ground_truth_label,
        "Prediction": "ELEVATED" if row.v5_pred == 1 else "LOWER", "Prediction_Score": row.v5_proba,
        "Top_Unusually_High_Features": fmt(hi_feats),
        "Top_Unusually_Low_Features": fmt(lo_feats),
        "Closest_Group_Pattern": pred_class,
        "Possible_ABCD_Group_Involved": ",".join(abcd_involved),
        "Possible_Segmentation_Influence": likely_seg,
        "Human_Note": existing_notes.get(image_id, ""),
        "Human_Final_Judgment": "",
    })

    lines = [f"Image: {image_id}", f"Type: {pred_class}", "", "Unusual features:"]
    for f in top:
        pct, flg = flags[f]
        lines.append(f"  - {f}: {pct:.0f}th percentile among {class_name} ({flg or 'notable'})")
    lines.append("")
    case_text_blocks.append("\n".join(lines))

case_df = pd.DataFrame(case_report_rows)
case_df.to_csv(BASE / "V5_VisualReview_Unusual_Features.csv", index=False)
print(f"Wrote V5_VisualReview_Unusual_Features.csv: {len(case_df)} rows")

with open(BASE / "_case_level_text_blocks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(case_text_blocks))

print("\nDone.")
