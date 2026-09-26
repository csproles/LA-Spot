"""Post-run error analysis of the completed Evaluation_3500 experiment.

READ-ONLY analysis of already-saved results. Does not rerun YOLO, ABCD,
thresholds, scoring, or masks, and does not touch the evaluation policy.
The only "new" computation is reading already-saved per-instance mask PNGs
to count lesion pixel area (an absolute-pixel-count feature the CSV does not
store directly, only lesion_fraction) — this reads existing saved artifacts,
it does not regenerate or alter them.

Source: Evaluation_3500/full_run/results.csv (the user's message referenced
"Evaluation_3500/results.csv", which does not exist; the actual results file
is at full_run/results.csv — used here, and noted in the summary output).

Outputs -> Evaluation_3500/ErrorAnalysis/
"""

import csv
import statistics as st
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "Evaluation_3500" / "full_run" / "results.csv"
OUT_DIR = ROOT / "Evaluation_3500" / "ErrorAnalysis"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLOR_TP_TN = "#0ca30c"   # status "good" — correct prediction
COLOR_FN_FP = "#d03b3b"   # status "critical" — classification error


def load_rows():
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def lesion_pixel_area(mask_path_str):
    """Reads an already-saved instance mask PNG and counts foreground
    pixels. Does not modify or regenerate the mask."""
    if not mask_path_str:
        return None
    p = ROOT / mask_path_str
    if not p.exists():
        return None
    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return int(np.sum(mask > 127))


def classify_group(row):
    """Returns 'TP','TN','FP','FN', or None (not evaluable / not this analysis's concern)."""
    if row["evaluation_status"] != "SINGLE_LESION_EVALUABLE":
        return None
    gt = row["ground_truth_binary"]
    pred = row["provisional_prediction"]
    if gt == "1" and pred == "positive":
        return "TP"
    if gt == "1" and pred == "negative":
        return "FN"
    if gt == "0" and pred == "negative":
        return "TN"
    if gt == "0" and pred == "positive":
        return "FP"
    return None


def build_feature_row(r):
    """Extracts every requested feature for one instance row, deriving
    lesion_pixel_area from the saved mask file and flag booleans from the
    saved quality_flags string. Never invents a value: missing/unavailable
    fields stay None and are reported as such."""
    quality_flags = r.get("quality_flags", "") or ""
    return {
        "image_name": r["image_name"],
        "A_value": to_float(r.get("A_value")),
        "A_concern": r.get("A_concern") == "True",
        "B_circularity": to_float(r.get("B_circularity")),
        "B_circularity_concern": r.get("B_circularity_concern") == "True",
        "B_experimental": to_float(r.get("B_experimental")),
        "B_experimental_n_defects": to_float(r.get("B_experimental_n_defects")),
        "C_value": to_float(r.get("C_value")),
        "C_concern": r.get("C_concern") == "True",
        "D_px": to_float(r.get("D_px")),
        "confidence": to_float(r.get("confidence")),
        "lesion_fraction": to_float(r.get("lesion_fraction")),
        "lesion_pixel_area": lesion_pixel_area(r.get("mask_path")),
        "num_total_components": to_float(r.get("num_total_components")),
        "num_meaningful_components": to_float(r.get("num_meaningful_components")),
        "num_tiny_artifacts_ignored": to_float(r.get("num_tiny_artifacts_ignored")),
        "flag_fragmented": "fragmented" in quality_flags,
        "flag_large_area": "large_area_fraction" in quality_flags,
        "flag_low_confidence": "low_confidence" in quality_flags,
        "flag_tiny_artifact_ignored": "ignored_" in quality_flags,
        "flag_any_quality_issue": quality_flags != "ok",
        "quality_flags_raw": quality_flags,
    }


CONTINUOUS_FIELDS = [
    "A_value", "B_circularity", "B_experimental", "B_experimental_n_defects",
    "C_value", "D_px", "confidence", "lesion_fraction", "lesion_pixel_area",
    "num_total_components", "num_meaningful_components", "num_tiny_artifacts_ignored",
]

FLAG_FIELDS = [
    "A_concern", "B_circularity_concern", "C_concern",
    "flag_fragmented", "flag_large_area", "flag_low_confidence",
    "flag_tiny_artifact_ignored", "flag_any_quality_issue",
]


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "q1": None, "q3": None, "min": None, "max": None, "n_missing_noted": True}
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_missing_noted": False,
    }


def feature_summary_csv(group_a_rows, group_b_rows, name_a, name_b, out_path):
    fieldnames = ["field", f"{name_a}_count", f"{name_a}_mean", f"{name_a}_median", f"{name_a}_std",
                  f"{name_a}_q1", f"{name_a}_q3", f"{name_a}_min", f"{name_a}_max",
                  f"{name_b}_count", f"{name_b}_mean", f"{name_b}_median", f"{name_b}_std",
                  f"{name_b}_q1", f"{name_b}_q3", f"{name_b}_min", f"{name_b}_max",
                  "mannwhitney_p_value", "field_saved"]
    rows_out = []
    for field in CONTINUOUS_FIELDS:
        vals_a = [r[field] for r in group_a_rows]
        vals_b = [r[field] for r in group_b_rows]
        da = describe(vals_a)
        db = describe(vals_b)
        field_saved = any(v is not None for v in vals_a + vals_b)

        clean_a = [v for v in vals_a if v is not None]
        clean_b = [v for v in vals_b if v is not None]
        p_value = ""
        if len(clean_a) >= 2 and len(clean_b) >= 2 and field_saved:
            try:
                _, p_value = mannwhitneyu(clean_a, clean_b, alternative="two-sided")
                p_value = round(float(p_value), 6)
            except ValueError:
                p_value = ""

        rows_out.append({
            "field": field,
            f"{name_a}_count": da["count"], f"{name_a}_mean": da["mean"], f"{name_a}_median": da["median"],
            f"{name_a}_std": da["std"], f"{name_a}_q1": da["q1"], f"{name_a}_q3": da["q3"],
            f"{name_a}_min": da["min"], f"{name_a}_max": da["max"],
            f"{name_b}_count": db["count"], f"{name_b}_mean": db["mean"], f"{name_b}_median": db["median"],
            f"{name_b}_std": db["std"], f"{name_b}_q1": db["q1"], f"{name_b}_q3": db["q3"],
            f"{name_b}_min": db["min"], f"{name_b}_max": db["max"],
            "mannwhitney_p_value": p_value,
            "field_saved": "yes" if field_saved else "NOT SAVED IN RESULTS.CSV",
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def flag_summary_csv(group_a_rows, group_b_rows, name_a, name_b, out_path):
    fieldnames = ["flag", f"{name_a}_count", f"{name_a}_pct", f"{name_b}_count", f"{name_b}_pct",
                  "fisher_exact_p_value"]
    rows_out = []
    n_a = len(group_a_rows)
    n_b = len(group_b_rows)
    for flag in FLAG_FIELDS:
        a_true = sum(1 for r in group_a_rows if r[flag])
        b_true = sum(1 for r in group_b_rows if r[flag])
        a_false = n_a - a_true
        b_false = n_b - b_true
        try:
            _, p_value = fisher_exact([[a_true, a_false], [b_true, b_false]])
            p_value = round(float(p_value), 6)
        except ValueError:
            p_value = ""
        rows_out.append({
            "flag": flag,
            f"{name_a}_count": a_true, f"{name_a}_pct": round(100 * a_true / n_a, 1) if n_a else "",
            f"{name_b}_count": b_true, f"{name_b}_pct": round(100 * b_true / n_b, 1) if n_b else "",
            "fisher_exact_p_value": p_value,
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def concern_combinations_csv(group_a_rows, group_b_rows, name_a, name_b, out_path):
    def combo_label(a, b, c):
        letters = "".join(x for x, flag in (("A", a), ("B", b), ("C", c)) if flag)
        return letters if letters else "none"

    combos_order = ["A", "B", "C", "AB", "AC", "BC", "ABC", "none"]
    labels_map = {"A": "A only", "B": "B only", "C": "C only", "AB": "A+B", "AC": "A+C",
                  "BC": "B+C", "ABC": "A+B+C", "none": "no A/B/C concerns"}

    def count_combos(rows):
        counts = {c: 0 for c in combos_order}
        for r in rows:
            key = combo_label(r["A_concern"], r["B_circularity_concern"], r["C_concern"])
            counts[key] += 1
        return counts

    counts_a = count_combos(group_a_rows)
    counts_b = count_combos(group_b_rows)
    n_a = len(group_a_rows)
    n_b = len(group_b_rows)

    rows_out = []
    for c in combos_order:
        rows_out.append({
            "combination": labels_map[c],
            f"{name_a}_count": counts_a[c],
            f"{name_a}_pct": round(100 * counts_a[c] / n_a, 1) if n_a else "",
            f"{name_b}_count": counts_b[c],
            f"{name_b}_pct": round(100 * counts_b[c] / n_b, 1) if n_b else "",
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["combination", f"{name_a}_count", f"{name_a}_pct",
                                                f"{name_b}_count", f"{name_b}_pct"])
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def make_comparison_plot(group_a_rows, group_b_rows, field, name_a, name_b, title, ylabel, save_path):
    vals_a = [r[field] for r in group_a_rows if r[field] is not None]
    vals_b = [r[field] for r in group_b_rows if r[field] is not None]

    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    data = [vals_a, vals_b]
    positions = [1, 2]
    bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                     showfliers=False, medianprops={"color": "#0b0b0b", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], [COLOR_TP_TN, COLOR_FN_FP]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, [COLOR_TP_TN, COLOR_FN_FP]):
        if not vals:
            continue
        sample = vals if len(vals) <= 400 else list(rng.choice(vals, 400, replace=False))
        jitter = rng.uniform(-0.12, 0.12, size=len(sample))
        ax.scatter(np.full(len(sample), pos) + jitter, sample, color=color,
                   alpha=0.5, s=14, edgecolors="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{name_a} (n={len(vals_a)})", f"{name_b} (n={len(vals_b)})"],
                        color="#0b0b0b", fontsize=10)
    ax.set_ylabel(ylabel, color="#0b0b0b", fontsize=10)
    ax.set_title(title, color="#0b0b0b", fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for pos, vals in zip(positions, data):
        if vals:
            ax.annotate(f"median={np.median(vals):.3g}", (pos, np.percentile(vals, 75)),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    all_rows = load_rows()
    print(f"Loaded {len(all_rows)} rows from {RESULTS_CSV}")

    groups_raw = {"TP": [], "TN": [], "FP": [], "FN": []}
    for r in all_rows:
        g = classify_group(r)
        if g:
            groups_raw[g].append(r)

    print(f"Verification against reported counts: TP={len(groups_raw['TP'])} (expected 440), "
          f"FN={len(groups_raw['FN'])} (expected 891), "
          f"TN={len(groups_raw['TN'])} (expected 997), FP={len(groups_raw['FP'])} (expected 308)")

    tp_matches = len(groups_raw["TP"]) == 440
    fn_matches = len(groups_raw["FN"]) == 891
    tn_matches = len(groups_raw["TN"]) == 997
    fp_matches = len(groups_raw["FP"]) == 308
    verification_ok = tp_matches and fn_matches and tn_matches and fp_matches

    print(f"TP matches: {tp_matches}  FN matches: {fn_matches}  TN matches: {tn_matches}  FP matches: {fp_matches}")

    print("Building per-instance feature rows (reading saved mask files for pixel area)...")
    features = {k: [build_feature_row(r) for r in v] for k, v in groups_raw.items()}

    # --- 1. TP vs FN feature summary
    feature_summary_csv(features["TP"], features["FN"], "TP", "FN",
                         OUT_DIR / "TP_vs_FN_feature_summary.csv")

    # --- 2. TP vs FN flag summary
    flag_summary_csv(features["TP"], features["FN"], "TP", "FN",
                      OUT_DIR / "TP_vs_FN_flag_summary.csv")

    # --- 3. TP vs FN concern combinations
    concern_combinations_csv(features["TP"], features["FN"], "TP", "FN",
                              OUT_DIR / "TP_vs_FN_concern_combinations.csv")

    # --- plots
    plot_specs = [
        ("A_value", "Asymmetry (A) — TP vs FN (malignant)", "A score", "A_TP_vs_FN.png"),
        ("B_circularity", "Border (B, circularity) — TP vs FN (malignant)", "B score", "B_TP_vs_FN.png"),
        ("C_value", "Color (C) — TP vs FN (malignant)", "C score", "C_TP_vs_FN.png"),
        ("confidence", "YOLO confidence — TP vs FN (malignant)", "confidence", "confidence_TP_vs_FN.png"),
        ("lesion_fraction", "Mask area fraction — TP vs FN (malignant)", "lesion_fraction", "mask_area_fraction_TP_vs_FN.png"),
        ("D_px", "Diameter in pixels (D_px) — TP vs FN (malignant)", "D_px", "D_px_TP_vs_FN.png"),
    ]
    for field, title, ylabel, filename in plot_specs:
        make_comparison_plot(features["TP"], features["FN"], field, "TP", "FN", title, ylabel,
                              PLOTS_DIR / filename)

    # --- smaller TN vs FP comparison
    feature_summary_csv(features["TN"], features["FP"], "TN", "FP",
                         OUT_DIR / "TN_vs_FP_feature_summary.csv")
    flag_summary_csv(features["TN"], features["FP"], "TN", "FP",
                      OUT_DIR / "TN_vs_FP_flag_summary.csv")

    print("\nDone. Outputs written to", OUT_DIR)
    return features, verification_ok


if __name__ == "__main__":
    main()
