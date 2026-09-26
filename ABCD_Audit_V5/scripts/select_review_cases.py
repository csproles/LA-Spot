"""Deterministic selection of representative cases for Part 5 (manual visual
review) and Parts 6-9's debug visualizations. Pure selection logic -- no
image processing here. Reads only V5_locked_test_master.csv.
"""
import numpy as np
import pandas as pd

BASE = r"C:\Users\sirjanaa\Documents\huggingface_projects\Advanced-Melanoma-Detection\ABCD_Audit_V5"
df = pd.read_csv(f"{BASE}\\V5_locked_test_master.csv")

df["pred_class"] = np.select(
    [(df.v5_pred == 1) & (df.ground_truth_binary == 1),
     (df.v5_pred == 0) & (df.ground_truth_binary == 0),
     (df.v5_pred == 1) & (df.ground_truth_binary == 0),
     (df.v5_pred == 0) & (df.ground_truth_binary == 1)],
    ["TP", "TN", "FP", "FN"], default="?"
)

fn = df[df.pred_class == "FN"].copy()
fp = df[df.pred_class == "FP"].copy()
tp = df[df.pred_class == "TP"].copy()
tn = df[df.pred_class == "TN"].copy()

selected = {}

# FN: 10 most confidently wrong (lowest v5_proba = most confidently called benign, but actually malignant)
fn_sorted = fn.sort_values("v5_proba")
selected["FN_high_confidence_wrong"] = fn_sorted.head(6)["image_id"].tolist()
# plus a couple near-threshold FN (proba just under 0.25) -- borderline misses
fn_near = fn[(fn.v5_proba >= 0.15) & (fn.v5_proba < 0.25)].sort_values("v5_proba", ascending=False)
selected["FN_near_threshold"] = fn_near.head(4)["image_id"].tolist()

# FP: 10 most confidently wrong (highest v5_proba = most confidently called malignant, but actually benign)
fp_sorted = fp.sort_values("v5_proba", ascending=False)
selected["FP_high_confidence_wrong"] = fp_sorted.head(6)["image_id"].tolist()
fp_near = fp[(fp.v5_proba >= 0.25) & (fp.v5_proba < 0.35)].sort_values("v5_proba")
selected["FP_near_threshold"] = fp_near.head(4)["image_id"].tolist()

# TP / TN: representative -- a spread across probability, not just extremes
tp_sorted = tp.sort_values("v5_proba", ascending=False)
selected["TP_representative"] = (
    tp_sorted.head(3)["image_id"].tolist() +
    tp_sorted.iloc[len(tp_sorted) // 2 - 1: len(tp_sorted) // 2 + 2]["image_id"].tolist()
)
tn_sorted = tn.sort_values("v5_proba")
selected["TN_representative"] = (
    tn_sorted.head(3)["image_id"].tolist() +
    tn_sorted.iloc[len(tn_sorted) // 2 - 1: len(tn_sorted) // 2 + 2]["image_id"].tolist()
)

# Unusual feature values: smallest lesion_area_px (segmentation-risk candidates,
# per Parts 6/7's "very small lesion" concern), and largest D_px_normalized/
# lesion_fraction gap-defying cases (the redundancy/confound flagged in Parts 1/2/9)
small_lesion = df.nsmallest(4, "lesion_area_px")["image_id"].tolist()
selected["Unusual_small_lesion_area"] = small_lesion

# cases where D_px_normalized and lesion_fraction disagree unusually (tests the
# "near-duplicate" claim -- if they never meaningfully diverge, that's evidence
# FOR the redundancy claim; if a few do diverge, those are the interesting ones)
df["_lf_rank"] = df["lesion_fraction"].rank(pct=True)
df["_dpxn_rank"] = df["D_px_normalized"].rank(pct=True)
df["_rank_gap"] = (df["_lf_rank"] - df["_dpxn_rank"]).abs()
selected["Unusual_lesionfraction_vs_Dpxnorm_divergence"] = df.nlargest(4, "_rank_gap")["image_id"].tolist()

# high B_circularity but predicted benign (tests the sign-flip concern directly)
selected["Unusual_high_Bcircularity_predicted_benign"] = (
    df[(df.v5_pred == 0)].nlargest(3, "B_circularity")["image_id"].tolist()
)

all_ids = sorted(set(x for lst in selected.values() for x in lst))
print(f"Total unique selected cases: {len(all_ids)}")
for k, v in selected.items():
    print(f"  {k} ({len(v)}): {v}")

# write manifest with pred_class, ground truth, proba for easy lookup downstream
sel_rows = []
for group, ids in selected.items():
    for iid in ids:
        row = df[df.image_id == iid].iloc[0]
        sel_rows.append({
            "selection_group": group, "image_id": iid, "pred_class": row.pred_class,
            "ground_truth_label": row.ground_truth_label, "v5_proba": row.v5_proba,
            "v5_pred": row.v5_pred, "lesion_area_px": row.lesion_area_px,
            "B_circularity": row.B_circularity, "lesion_fraction": row.lesion_fraction,
            "D_px_normalized": row.D_px_normalized,
        })
sel_df = pd.DataFrame(sel_rows).drop_duplicates(subset=["image_id"], keep="first")
sel_df.to_csv(f"{BASE}\\_case_selection_manifest.csv", index=False)
print(f"\nWrote _case_selection_manifest.csv: {len(sel_df)} unique cases")
