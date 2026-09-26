"""
Stability check: rerun the EXACT SAME experiment matrix and decision logic as
../run_all_experiments.py, but with (1) a shuffled row order fed into
GroupKFold -- sklearn's GroupKFold has no shuffle parameter and assigns folds
deterministically based on the order groups first appear, so shuffling row
order is what actually produces a DIFFERENT split, not just a different
random_state -- and (2) a different LogisticRegression random_state. This
answers: are the conclusions (which features help/hurt, which config wins)
stable across a different CV partition, or an artifact of one particular
split?

Every decision-logic line below is intentionally IDENTICAL to
../run_all_experiments.py (same feature-set construction, same "better()"
priority rule, same cascade) -- the only differences are marked SPLIT2 CHANGE.
This guarantees the comparison is apples-to-apples in structure; any
divergence in outcomes is attributable to the split/seed, not to a different
procedure.

Reads ONLY ../dev_feature_table_v2.csv. Never touches pipeline_v5/,
revised_abcd/, Code/, any Evaluation_* directory, or the locked test set.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

import cv_harness  # noqa: E402
from cv_harness import evaluate_experiment, V5_ALL_FEATURES, EXPECTED_V5_REPRO  # noqa: E402

T0 = time.perf_counter()

# ---- SPLIT2 CHANGE #1: shuffle row order (this is what changes the GroupKFold split) ----
SHUFFLE_SEED = 777
df = cv_harness.load_table(str(PARENT / "dev_feature_table_v2.csv"))
df = df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

# ---- SPLIT2 CHANGE #2: different LogisticRegression seed ----
ORIGINAL_SEED = cv_harness.SEED
NEW_SEED = 999983
cv_harness.SEED = NEW_SEED
print(f"SPLIT2: shuffled row order (pandas sample random_state={SHUFFLE_SEED}), "
      f"LogisticRegression SEED {ORIGINAL_SEED} -> {NEW_SEED}\n")

records = []
ablation_records = []
timings = {}


def run(exp_id, pipeline_name, fields, notes=""):
    t0 = time.perf_counter()
    r = evaluate_experiment(df, fields, exp_id)
    dt = time.perf_counter() - t0
    timings[exp_id] = dt
    y_used = np.array(list(r["y_by_id"].values()))
    m = r["overall_metrics"]
    rec = {
        "Experiment_ID": exp_id,
        "Pipeline_Name": pipeline_name,
        "Dataset": "development (GroupKFold(5) OOF), n=2090, SPLIT2 (shuffled order + new LR seed)",
        "N_Total": r["n_used"],
        "N_Benign": int((y_used == 0).sum()),
        "N_Malignant": int((y_used == 1).sum()),
        "Feature_Count": r["n_features"],
        "Threshold": r["threshold"],
        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
        "Accuracy": round(m["accuracy"], 4),
        "Balanced_Accuracy": round(m["balanced_accuracy"], 4),
        "Sensitivity": round(m["sensitivity"], 4),
        "Specificity": round(m["specificity"], 4),
        "Precision": round(m["precision"], 4),
        "F1": round(m["f1"], 4),
        "ROC_AUC": round(r["overall_auc"], 4),
        "CV_Mean_Balanced_Accuracy": round(r["cv_mean_balanced_accuracy"], 4),
        "CV_STD_Balanced_Accuracy": round(r["cv_std_balanced_accuracy"], 4),
        "Runtime": round(dt, 3),
        "Notes": notes,
    }
    records.append(rec)
    print(f"{exp_id:32s} n_feat={r['n_features']:2d} n_used={r['n_used']:4d} thr={r['threshold']:.2f} "
          f"AUC={r['overall_auc']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
          f"CVmean={r['cv_mean_balanced_accuracy']:.4f} CVstd={r['cv_std_balanced_accuracy']:.4f}")
    return r, rec


def better(a, b):
    ba, bb = a["overall_metrics"]["balanced_accuracy"], b["overall_metrics"]["balanced_accuracy"]
    if abs(ba - bb) > 1e-9:
        return ba > bb
    return a["overall_auc"] > b["overall_auc"]


def add_ablation(change_desc, feature_added, feature_removed, baseline_rec, variant_rec, baseline_result, variant_result):
    d_bal = variant_rec["Balanced_Accuracy"] - baseline_rec["Balanced_Accuracy"]
    d_auc = variant_rec["ROC_AUC"] - baseline_rec["ROC_AUC"]
    cv_std = baseline_result["cv_std_balanced_accuracy"]
    if abs(d_bal) <= cv_std:
        verdict = "no meaningful change (within CV std)"
    elif d_bal > 0:
        verdict = "helped"
    else:
        verdict = "hurt"
    ablation_records.append({
        "Change": change_desc, "Feature_Added": feature_added, "Feature_Removed": feature_removed,
        "Baseline_Experiment_ID": baseline_rec["Experiment_ID"], "Variant_Experiment_ID": variant_rec["Experiment_ID"],
        "Baseline_Balanced_Accuracy": baseline_rec["Balanced_Accuracy"], "Variant_Balanced_Accuracy": variant_rec["Balanced_Accuracy"],
        "Delta_Balanced_Accuracy": round(d_bal, 4),
        "Baseline_ROC_AUC": baseline_rec["ROC_AUC"], "Variant_ROC_AUC": variant_rec["ROC_AUC"], "Delta_ROC_AUC": round(d_auc, 4),
        "Baseline_CV_STD_Balanced_Accuracy": round(cv_std, 4), "Verdict": verdict,
    })


# ============================================================ EXP0 ==========
exp0_result, exp0_rec = run("EXP0", "Frozen V5 baseline (17 original features)", V5_ALL_FEATURES,
                             notes="Sanity-gate reproduction check (SPLIT2 still uses the frozen, un-refit V5 "
                                   "feature DEFINITIONS -- reproducing V5's own historical OOF numbers exactly is "
                                   "NOT expected here since the CV partition itself has changed; this row instead "
                                   "shows how much EXP0's OWN performance naturally varies across splits, which is "
                                   "the baseline noise floor for judging every other delta below.")
for k, expected in EXPECTED_V5_REPRO.items():
    key_map = {"roc_auc": "ROC_AUC", "sensitivity": "Sensitivity", "specificity": "Specificity", "balanced_accuracy": "Balanced_Accuracy"}
    actual = exp0_rec[key_map[k]]
    print(f"  vs original-split V5 number [{k}]: original={expected} split2={actual} "
          f"delta={round(actual-expected,4)}")
print()

# ============================================================ EXP1 ==========
EXP1_FEATURES = ["A_value_pc", "B_circularity", "C_value_pc", "D_px", "confidence", "lesion_fraction",
                  "color_entropy_pc", "lab_a_std_pc", "lab_b_std_pc", "red_fraction_pc", "bluegray_fraction_pc",
                  "dark_fraction_pc", "skin_contrast_pc", "solidity", "turning_angle_std", "eccentricity",
                  "D_px_normalized"]
exp1_result, exp1_rec = run("EXP1", "Primary-component-consistent features (A_value_pc + color group _pc)", EXP1_FEATURES)

EXP1a_FEATURES = list(V5_ALL_FEATURES)
EXP1a_FEATURES[EXP1a_FEATURES.index("A_value")] = "A_value_pc"
exp1a_result, exp1a_rec = run("EXP1a", "EXP0 + A_value_pc only (isolate asymmetry fix)", EXP1a_FEATURES)
add_ablation("A_value -> A_value_pc (primary-component fix), holding color features at original",
             "A_value_pc", "A_value", exp0_rec, exp1a_rec, exp0_result, exp1a_result)

EXP1b_FEATURES = list(V5_ALL_FEATURES)
_color_swap = {"C_value": "C_value_pc", "color_entropy": "color_entropy_pc", "lab_a_std": "lab_a_std_pc",
               "lab_b_std": "lab_b_std_pc", "red_fraction": "red_fraction_pc",
               "bluegray_fraction": "bluegray_fraction_pc", "dark_fraction": "dark_fraction_pc",
               "skin_contrast": "skin_contrast_pc"}
EXP1b_FEATURES = [_color_swap.get(f, f) for f in EXP1b_FEATURES]
exp1b_result, exp1b_rec = run("EXP1b", "EXP0 + color-group _pc only (isolate color fix)", EXP1b_FEATURES)
add_ablation("8 color features -> their _pc versions (primary-component fix), holding A_value at original",
             "C_value_pc,color_entropy_pc,lab_a_std_pc,lab_b_std_pc,red_fraction_pc,bluegray_fraction_pc,dark_fraction_pc,skin_contrast_pc",
             "C_value,color_entropy,lab_a_std,lab_b_std,red_fraction,bluegray_fraction,dark_fraction,skin_contrast",
             exp0_rec, exp1b_rec, exp0_result, exp1b_result)
add_ablation("EXP1 combined (A_value_pc + color-group _pc together) vs EXP0", "all 9 _pc features",
             "all 9 original counterparts", exp0_rec, exp1_rec, exp0_result, exp1_result)

# ============================================================ EXP2 ==========
EXP2_FEATURES = [f for f in EXP1_FEATURES if f not in ("eccentricity", "dark_fraction_pc", "bluegray_fraction_pc", "D_px")]
exp2_result, exp2_rec = run("EXP2", "EXP1 minus 4 weak/near-chance features", EXP2_FEATURES)

for feat_to_drop, label in [("eccentricity", "eccentricity"), ("dark_fraction_pc", "dark_fraction_pc"),
                             ("bluegray_fraction_pc", "bluegray_fraction_pc"), ("D_px", "D_px (raw pixels)")]:
    fset = [f for f in EXP1_FEATURES if f != feat_to_drop]
    r_, rec_ = run(f"EXP2_minus_{feat_to_drop}_only", f"EXP1 minus {label} only", fset)
    add_ablation(f"Remove {label} alone from EXP1", None, feat_to_drop, exp1_rec, rec_, exp1_result, r_)
add_ablation("Remove all 4 weak features together (EXP2 vs EXP1)", None,
             "eccentricity,dark_fraction_pc,bluegray_fraction_pc,D_px", exp1_rec, exp2_rec, exp1_result, exp2_result)

# ============================================================ EXP3 ==========
exp3A_features = list(EXP2_FEATURES)
exp3B_features = [f for f in EXP2_FEATURES if f != "D_px_normalized"]
exp3C_features = [f for f in EXP2_FEATURES if f != "lesion_fraction"]
exp3D_features = [f for f in EXP2_FEATURES if f not in ("lesion_fraction", "D_px_normalized")]

r3A, rec3A = run("EXP3A", "EXP2 + both lesion_fraction & D_px_normalized (baseline)", exp3A_features)
r3B, rec3B = run("EXP3B", "EXP2 + lesion_fraction only (drop D_px_normalized)", exp3B_features)
r3C, rec3C = run("EXP3C", "EXP2 + D_px_normalized only (drop lesion_fraction)", exp3C_features)
r3D, rec3D = run("EXP3D", "EXP2 + neither lesion_fraction nor D_px_normalized", exp3D_features)
add_ablation("Drop D_px_normalized only (keep lesion_fraction)", None, "D_px_normalized", rec3A, rec3B, r3A, r3B)
add_ablation("Drop lesion_fraction only (keep D_px_normalized)", None, "lesion_fraction", rec3A, rec3C, r3A, r3C)
add_ablation("Drop both lesion_fraction and D_px_normalized", None, "lesion_fraction,D_px_normalized", rec3A, rec3D, r3A, r3D)

exp3_candidates = [("EXP3A", r3A, rec3A, exp3A_features), ("EXP3B", r3B, rec3B, exp3B_features),
                    ("EXP3C", r3C, rec3C, exp3C_features), ("EXP3D", r3D, rec3D, exp3D_features)]
best_exp3_name, best_exp3_result, best_exp3_rec, best_exp3_features = exp3_candidates[0]
for name, r_, rec_, feats_ in exp3_candidates[1:]:
    if better(r_, best_exp3_result):
        best_exp3_name, best_exp3_result, best_exp3_rec, best_exp3_features = name, r_, rec_, feats_
print(f"\n>>> SPLIT2 best EXP3 variant: {best_exp3_name}\n")

# ============================================================ EXP4 ==========
base4_a = list(best_exp3_features)
base4_b = [f for f in best_exp3_features if f != "color_entropy_pc"] + ["entropy_L", "entropy_a", "entropy_b"] \
    if "color_entropy_pc" in best_exp3_features else list(best_exp3_features) + ["entropy_L", "entropy_a", "entropy_b"]
base4_c = list(best_exp3_features)
if "color_entropy_pc" not in base4_c:
    base4_c = base4_c + ["color_entropy_pc"]
base4_c = base4_c + ["entropy_L", "entropy_a", "entropy_b"]

r4a, rec4a = run("EXP4a", f"{best_exp3_name} + color_entropy_pc only (no channel entropy)", base4_a)
r4b, rec4b = run("EXP4b", f"{best_exp3_name} + entropy_L/a/b replacing color_entropy_pc", base4_b)
r4c, rec4c = run("EXP4c", f"{best_exp3_name} + color_entropy_pc AND entropy_L/a/b together", base4_c)
add_ablation("Replace color_entropy_pc with entropy_L/a/b (channel-specific only)", "entropy_L,entropy_a,entropy_b",
             "color_entropy_pc", rec4a, rec4b, r4a, r4b)
add_ablation("Add entropy_L/a/b alongside existing color_entropy_pc", "entropy_L,entropy_a,entropy_b", None,
             rec4a, rec4c, r4a, r4c)

entropy_candidates = [("EXP4a", r4a, rec4a, base4_a), ("EXP4b", r4b, rec4b, base4_b), ("EXP4c", r4c, rec4c, base4_c)]
best_entropy_name, best_entropy_result, best_entropy_rec, best_entropy_features = entropy_candidates[0]
for name, r_, rec_, feats_ in entropy_candidates[1:]:
    if better(r_, best_entropy_result):
        best_entropy_name, best_entropy_result, best_entropy_rec, best_entropy_features = name, r_, rec_, feats_
print(f">>> SPLIT2 best EXP4 entropy variant: {best_entropy_name}\n")

FRACTION_FEATS = ["red_fraction_pc", "bluegray_fraction_pc", "dark_fraction_pc"]
frac_base = [f for f in best_entropy_features if f not in FRACTION_FEATS]
from itertools import combinations
frac_results = {}
for r_size in range(0, 4):
    for combo in combinations(FRACTION_FEATS, r_size):
        label = "none" if not combo else "+".join(c.replace("_fraction_pc", "") for c in combo)
        exp_id = f"EXP4_frac_{label}"
        feats = frac_base + list(combo)
        r_, rec_ = run(exp_id, f"{best_entropy_name} fraction-group test: {label}", feats)
        frac_results[label] = (r_, rec_, feats)

baseline_frac_rec = frac_results["none"][1]
baseline_frac_result = frac_results["none"][0]
for single in ["red", "bluegray", "dark"]:
    r_, rec_, feats_ = frac_results[single]
    add_ablation(f"Add {single}_fraction_pc alone on top of best entropy config (no other fraction feats)",
                 f"{single}_fraction_pc", None, baseline_frac_rec, rec_, baseline_frac_result, r_)
r_all3, rec_all3, feats_all3 = frac_results["red+bluegray+dark"]
add_ablation("Add all 3 fraction features (red+bluegray+dark) together on top of best entropy config",
             "red_fraction_pc,bluegray_fraction_pc,dark_fraction_pc", None, baseline_frac_rec, rec_all3, baseline_frac_result, r_all3)
for single in ["red", "bluegray", "dark"]:
    remaining = [s for s in ["red", "bluegray", "dark"] if s != single]
    remaining_label = "+".join(remaining)
    r_, rec_, feats_ = frac_results[remaining_label]
    add_ablation(f"Remove {single}_fraction_pc alone from the all-3-present state", None, f"{single}_fraction_pc",
                 rec_all3, rec_, r_all3, r_)

best_frac_label = max(frac_results.keys(), key=lambda k: (frac_results[k][0]["overall_metrics"]["balanced_accuracy"], frac_results[k][0]["overall_auc"]))
best_frac_result, best_frac_rec, best_exp4_features = frac_results[best_frac_label]
print(f">>> SPLIT2 best fraction-group config: '{best_frac_label}'\n")

exp4_final_result, exp4_final_rec = run("EXP4", f"Best combined EXP4 config ({best_entropy_name} entropy + '{best_frac_label}' fractions)", best_exp4_features)

# ============================================================ EXP5 ==========
if "A_value_pc" in best_exp4_features:
    exp5_single_features = list(best_exp4_features)
    exp5_split_features = [f for f in best_exp4_features if f != "A_value_pc"] + ["asymmetry_major_axis", "asymmetry_minor_axis"]
else:
    exp5_single_features = list(best_exp4_features)
    exp5_split_features = list(best_exp4_features) + ["asymmetry_major_axis", "asymmetry_minor_axis"]

r5_single, rec5_single = run("EXP5_single_A_value_pc", "Best EXP4 config with single A_value_pc (baseline)", exp5_single_features)
r5_split, rec5_split = run("EXP5_split_major_minor", "Best EXP4 config with separate major/minor asymmetry", exp5_split_features)
add_ablation("Replace A_value_pc (single) with asymmetry_major_axis + asymmetry_minor_axis (two features)",
             "asymmetry_major_axis,asymmetry_minor_axis", "A_value_pc", rec5_single, rec5_split, r5_single, r5_split)

if better(r5_split, r5_single):
    best_exp5_features, exp5_winner_rec, exp5_winner_result, exp5_winner_id = exp5_split_features, rec5_split, r5_split, "EXP5_split_major_minor"
else:
    best_exp5_features, exp5_winner_rec, exp5_winner_result, exp5_winner_id = exp5_single_features, rec5_single, r5_single, "EXP5_single_A_value_pc"
print(f">>> SPLIT2 best EXP5 variant: {exp5_winner_id}\n")

exp5_final_result, exp5_final_rec = run("EXP5", f"Best EXP5 config ({exp5_winner_id})", best_exp5_features)

# ============================================================ EXP6 ==========
exp6_without_features = list(best_exp5_features)
exp6_with_features = list(best_exp5_features) + ["border_fractal_dimension"]
r6_without, rec6_without = run("EXP6_without_fractal", "Best EXP5 config, no border_fractal_dimension", exp6_without_features)
r6_with, rec6_with = run("EXP6_with_fractal", "Best EXP5 config + border_fractal_dimension", exp6_with_features)
add_ablation("Add border_fractal_dimension to best EXP5 config", "border_fractal_dimension", None,
             rec6_without, rec6_with, r6_without, r6_with)

if better(r6_with, r6_without):
    best_exp6_features, exp6_winner_rec, exp6_winner_result, exp6_winner_id = exp6_with_features, rec6_with, r6_with, "EXP6_with_fractal"
else:
    best_exp6_features, exp6_winner_rec, exp6_winner_result, exp6_winner_id = exp6_without_features, rec6_without, r6_without, "EXP6_without_fractal"
print(f">>> SPLIT2 best EXP6 variant: {exp6_winner_id}\n")

exp6_final_result, exp6_final_rec = run("EXP6", f"Best EXP6 config ({exp6_winner_id})", best_exp6_features)

# ============================================================ FINAL =========
final_features = list(best_exp6_features)
final_result, final_rec = run("FINAL", "SPLIT2 final candidate (cascade of best sub-variants)", final_features,
                               notes="NOT run on locked test set. Development-only, SPLIT2 stability check.")

TOTAL_RUNTIME = time.perf_counter() - T0
print(f"\nSPLIT2 total experiments: {len(records)}  ablation rows: {len(ablation_records)}  runtime: {TOTAL_RUNTIME:.2f}s")

# ============================================================ WRITE CSVs ====
COLS = ["Experiment_ID", "Pipeline_Name", "Dataset", "N_Total", "N_Benign", "N_Malignant", "Feature_Count",
        "Threshold", "TP", "TN", "FP", "FN", "Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity",
        "Precision", "F1", "ROC_AUC", "CV_Mean_Balanced_Accuracy", "CV_STD_Balanced_Accuracy", "Runtime", "Notes"]
with open(HERE / "pipeline_performance_comparison_split2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for rec in records:
        w.writerow(rec)

ABL_COLS = ["Change", "Feature_Added", "Feature_Removed", "Baseline_Experiment_ID", "Variant_Experiment_ID",
            "Baseline_Balanced_Accuracy", "Variant_Balanced_Accuracy", "Delta_Balanced_Accuracy",
            "Baseline_ROC_AUC", "Variant_ROC_AUC", "Delta_ROC_AUC", "Baseline_CV_STD_Balanced_Accuracy", "Verdict"]
with open(HERE / "feature_ablation_results_split2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=ABL_COLS)
    w.writeheader()
    for rec in ablation_records:
        w.writerow(rec)

state = {
    "shuffle_seed": SHUFFLE_SEED, "lr_seed": NEW_SEED,
    "final_features": final_features, "final_rec": final_rec,
    "best_exp3_name": best_exp3_name, "best_entropy_name": best_entropy_name, "best_frac_label": best_frac_label,
    "exp5_winner_id": exp5_winner_id, "exp6_winner_id": exp6_winner_id,
    "exp0_rec": exp0_rec, "exp1_rec": exp1_rec, "exp2_rec": exp2_rec,
    "exp3A": rec3A, "exp3B": rec3B, "exp3C": rec3C, "exp3D": rec3D,
    "exp4a": rec4a, "exp4b": rec4b, "exp4c": rec4c, "exp4_final": exp4_final_rec,
    "exp5_single": rec5_single, "exp5_split": rec5_split, "exp5_final": exp5_final_rec,
    "exp6_without": rec6_without, "exp6_with": rec6_with, "exp6_final": exp6_final_rec,
    "total_runtime_sec": TOTAL_RUNTIME, "n_experiments": len(records), "n_ablation_rows": len(ablation_records),
}
with open(HERE / "_experiment_state_split2.json", "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2, default=str)

cv_harness.SEED = ORIGINAL_SEED  # restore, in case this module is imported elsewhere later
print("Wrote pipeline_performance_comparison_split2.csv, feature_ablation_results_split2.csv, _experiment_state_split2.json")
