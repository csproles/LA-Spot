"""Melanoma-specific TP vs FN failure analysis for Evaluation_3500.

READ-ONLY: joins the existing manifest, Evaluation_3500/full_run/results.csv,
and ISIC source metadata.csv by isic_id. Does not rerun or modify YOLO, ABCD,
masks, thresholds, scoring, or the evaluation policy. The only new
computation is reading already-saved instance mask PNGs to count lesion
pixel area (not stored directly in results.csv).

Restricted throughout to the 333 SINGLE_LESION_EVALUABLE melanoma cases
(134 TP + 199 FN), per instruction.
"""

import csv
import statistics as st
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "Evaluation_3500" / "full_run" / "results.csv"
MANIFEST_CSV = ROOT / "Evaluation_3500" / "evaluation_manifest_3500.csv"
SOURCE_METADATA = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images\metadata.csv")
OUT_DIR = ROOT / "Evaluation_3500" / "MelanomaFailureAnalysis"
PLOTS_DIR = OUT_DIR / "plots"

COLOR_TP = "#0ca30c"
COLOR_FN = "#d03b3b"

MELANOMA_DIAG3 = {"Melanoma Invasive", "Melanoma in situ", "Melanoma, NOS", "Melanoma metastasis"}


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def lesion_pixel_area(mask_path_str):
    if not mask_path_str:
        return None
    p = ROOT / mask_path_str
    if not p.exists():
        return None
    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return int(np.sum(mask > 127))


def load_all():
    manifest = list(csv.DictReader(open(MANIFEST_CSV, newline="", encoding="utf-8")))
    results = list(csv.DictReader(open(RESULTS_CSV, newline="", encoding="utf-8")))
    meta = {r["isic_id"]: r for r in csv.DictReader(open(SOURCE_METADATA, newline="", encoding="utf-8"))}
    return manifest, results, meta


def image_id_from_name(name):
    return Path(name).stem


def build_feature_row(r, meta_row):
    qf = r.get("quality_flags", "") or ""
    return {
        "image_id": image_id_from_name(r["image_name"]),
        "A_value": to_float(r.get("A_value")), "A_concern": r.get("A_concern") == "True",
        "B_circularity": to_float(r.get("B_circularity")), "B_circularity_concern": r.get("B_circularity_concern") == "True",
        "C_value": to_float(r.get("C_value")), "C_concern": r.get("C_concern") == "True",
        "confidence": to_float(r.get("confidence")),
        "lesion_fraction": to_float(r.get("lesion_fraction")),
        "lesion_pixel_area": lesion_pixel_area(r.get("mask_path")),
        "D_px": to_float(r.get("D_px")),
        "num_total_components": to_float(r.get("num_total_components")),
        "num_meaningful_components": to_float(r.get("num_meaningful_components")),
        "flag_fragmented": "fragmented" in qf,
        "flag_large_area": "large_area_fraction" in qf,
        "flag_low_confidence": "low_confidence" in qf,
        "flag_tiny_artifact_ignored": "ignored_" in qf,
        "flag_any_quality_issue": qf != "ok",
        "quality_flags_raw": qf,
        "diagnosis_3": meta_row.get("diagnosis_3", ""),
        "attribution": meta_row.get("attribution", ""),
        "mask_path": r.get("mask_path", ""),
        "image_name": r["image_name"],
    }


CONTINUOUS_FIELDS = ["A_value", "B_circularity", "C_value", "confidence", "lesion_fraction",
                     "lesion_pixel_area", "D_px", "num_total_components", "num_meaningful_components"]
FLAG_FIELDS = ["A_concern", "B_circularity_concern", "C_concern", "flag_fragmented",
              "flag_large_area", "flag_low_confidence", "flag_tiny_artifact_ignored", "flag_any_quality_issue"]


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "q1": None, "q3": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {"count": len(arr), "mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "q1": float(np.percentile(arr, 25)), "q3": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)), "max": float(np.max(arr))}


def rank_biserial(u_stat, n1, n2):
    """Effect size for Mann-Whitney U: r = 1 - 2U/(n1*n2). Ranges -1..1."""
    return 1 - (2 * u_stat) / (n1 * n2)


def feature_summary(tp_rows, fn_rows, out_path):
    fieldnames = ["field", "TP_count", "TP_mean", "TP_median", "TP_std", "TP_q1", "TP_q3", "TP_min", "TP_max",
                  "FN_count", "FN_mean", "FN_median", "FN_std", "FN_q1", "FN_q3", "FN_min", "FN_max",
                  "mannwhitney_p", "rank_biserial_effect_size"]
    out_rows = []
    for field in CONTINUOUS_FIELDS:
        va = [r[field] for r in tp_rows]
        vb = [r[field] for r in fn_rows]
        da, db = describe(va), describe(vb)
        clean_a = [v for v in va if v is not None]
        clean_b = [v for v in vb if v is not None]
        p, eff = "", ""
        if len(clean_a) >= 2 and len(clean_b) >= 2:
            try:
                u, p = mannwhitneyu(clean_a, clean_b, alternative="two-sided")
                p = round(float(p), 6)
                eff = round(rank_biserial(u, len(clean_a), len(clean_b)), 4)
            except ValueError:
                pass
        out_rows.append({
            "field": field, "TP_count": da["count"], "TP_mean": da["mean"], "TP_median": da["median"],
            "TP_std": da["std"], "TP_q1": da["q1"], "TP_q3": da["q3"], "TP_min": da["min"], "TP_max": da["max"],
            "FN_count": db["count"], "FN_mean": db["mean"], "FN_median": db["median"],
            "FN_std": db["std"], "FN_q1": db["q1"], "FN_q3": db["q3"], "FN_min": db["min"], "FN_max": db["max"],
            "mannwhitney_p": p, "rank_biserial_effect_size": eff,
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def flag_summary(tp_rows, fn_rows, out_path):
    fieldnames = ["flag", "TP_count", "TP_pct", "FN_count", "FN_pct", "fisher_p", "odds_ratio"]
    n_tp, n_fn = len(tp_rows), len(fn_rows)
    out_rows = []
    for flag in FLAG_FIELDS:
        a = sum(1 for r in tp_rows if r[flag]); a0 = n_tp - a
        b = sum(1 for r in fn_rows if r[flag]); b0 = n_fn - b
        try:
            odds_ratio, p = fisher_exact([[a, a0], [b, b0]])
            p = round(float(p), 6)
            odds_ratio = round(float(odds_ratio), 4) if odds_ratio not in (float("inf"),) else "inf"
        except ValueError:
            p = odds_ratio = ""
        out_rows.append({"flag": flag, "TP_count": a, "TP_pct": round(100 * a / n_tp, 1) if n_tp else "",
                         "FN_count": b, "FN_pct": round(100 * b / n_fn, 1) if n_fn else "",
                         "fisher_p": p, "odds_ratio": odds_ratio})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def concern_combinations(tp_rows, fn_rows, out_path):
    def label(a, b, c):
        s = "".join(x for x, f in (("A", a), ("B", b), ("C", c)) if f)
        return s if s else "none"
    order = ["A", "B", "C", "AB", "AC", "BC", "ABC", "none"]
    names = {"A": "A only", "B": "B only", "C": "C only", "AB": "A+B", "AC": "A+C", "BC": "B+C",
            "ABC": "A+B+C", "none": "no A/B/C concerns"}

    def counts(rows):
        c = {k: 0 for k in order}
        for r in rows:
            c[label(r["A_concern"], r["B_circularity_concern"], r["C_concern"])] += 1
        return c
    ctp, cfn = counts(tp_rows), counts(fn_rows)
    n_tp, n_fn = len(tp_rows), len(fn_rows)
    out_rows = []
    for k in order:
        out_rows.append({"combination": names[k], "TP_count": ctp[k],
                         "TP_pct": round(100 * ctp[k] / n_tp, 1) if n_tp else "",
                         "FN_count": cfn[k], "FN_pct": round(100 * cfn[k] / n_fn, 1) if n_fn else ""})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["combination", "TP_count", "TP_pct", "FN_count", "FN_pct"])
        w.writeheader()
        w.writerows(out_rows)
    return out_rows, cfn, n_fn


def make_plot(tp_rows, fn_rows, field, title, ylabel, save_path):
    va = [r[field] for r in tp_rows if r[field] is not None]
    vb = [r[field] for r in fn_rows if r[field] is not None]
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bp = ax.boxplot([va, vb], positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False,
                    medianprops={"color": "#0b0b0b", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], [COLOR_TP, COLOR_FN]):
        patch.set_facecolor(color); patch.set_alpha(0.25); patch.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for pos, vals, color in zip([1, 2], [va, vb], [COLOR_TP, COLOR_FN]):
        if not vals:
            continue
        sample = vals if len(vals) <= 300 else list(rng.choice(vals, 300, replace=False))
        jitter = rng.uniform(-0.12, 0.12, size=len(sample))
        ax.scatter(np.full(len(sample), pos) + jitter, sample, color=color, alpha=0.55, s=16, edgecolors="none", zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"TP (n={len(va)})", f"FN (n={len(vb)})"], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def main():
    manifest, results, meta = load_all()
    rows_by_image = defaultdict(list)
    for r in results:
        rows_by_image[image_id_from_name(r["image_name"])].append(r)

    melanoma_ids = [r["image_id"] for r in manifest if r["ground_truth"] == "malignant"
                    and (meta[r["image_id"]].get("diagnosis_3") or "").strip() in MELANOMA_DIAG3]
    print(f"Melanoma images in manifest: {len(melanoma_ids)}")

    tp_rows, fn_rows = [], []
    for image_id in melanoma_ids:
        image_rows = rows_by_image.get(image_id, [])
        if not image_rows or image_rows[0]["evaluation_status"] != "SINGLE_LESION_EVALUABLE":
            continue
        row = image_rows[0]
        feat = build_feature_row(row, meta[image_id])
        if row["provisional_prediction"] == "positive":
            tp_rows.append(feat)
        elif row["provisional_prediction"] == "negative":
            fn_rows.append(feat)

    print(f"TP={len(tp_rows)} (expected 134)  FN={len(fn_rows)} (expected 199)")
    assert len(tp_rows) == 134 and len(fn_rows) == 199, "COUNT MISMATCH — stopping, do not proceed on wrong data"

    feature_summary(tp_rows, fn_rows, OUT_DIR / "melanoma_TP_vs_FN_feature_summary.csv")
    flag_summary(tp_rows, fn_rows, OUT_DIR / "melanoma_TP_vs_FN_flag_summary.csv")
    combo_rows, cfn_counts, n_fn = concern_combinations(tp_rows, fn_rows, OUT_DIR / "melanoma_TP_vs_FN_concern_combinations.csv")

    for field, title, ylabel, fname in [
        ("A_value", "Melanoma: Asymmetry (A) — TP vs FN", "A score", "A_melanoma_TP_vs_FN.png"),
        ("B_circularity", "Melanoma: Border (B) — TP vs FN", "B score", "B_melanoma_TP_vs_FN.png"),
        ("C_value", "Melanoma: Color (C) — TP vs FN", "C score", "C_melanoma_TP_vs_FN.png"),
        ("confidence", "Melanoma: YOLO confidence — TP vs FN", "confidence", "confidence_melanoma_TP_vs_FN.png"),
        ("lesion_fraction", "Melanoma: mask area fraction — TP vs FN", "lesion_fraction", "mask_fraction_melanoma_TP_vs_FN.png"),
        ("D_px", "Melanoma: D_px — TP vs FN", "D_px", "D_px_melanoma_TP_vs_FN.png"),
    ]:
        make_plot(tp_rows, fn_rows, field, title, ylabel, PLOTS_DIR / fname)

    # --- subtype breakdown within melanoma ---
    subtype_stats = []
    for subtype in sorted(MELANOMA_DIAG3):
        sub_tp = [r for r in tp_rows if r["diagnosis_3"] == subtype]
        sub_fn = [r for r in fn_rows if r["diagnosis_3"] == subtype]
        n = len(sub_tp) + len(sub_fn)
        if n == 0:
            continue
        sens = len(sub_tp) / n if n else None
        a_tp = [r["A_value"] for r in sub_tp if r["A_value"] is not None]
        a_fn = [r["A_value"] for r in sub_fn if r["A_value"] is not None]
        large_tp = sum(1 for r in sub_tp if r["flag_large_area"])
        large_fn = sum(1 for r in sub_fn if r["flag_large_area"])
        subtype_stats.append({
            "subtype": subtype, "TP": len(sub_tp), "FN": len(sub_fn), "sensitivity": round(sens, 4) if sens else None,
            "A_mean_TP": round(st.mean(a_tp), 4) if a_tp else None,
            "A_mean_FN": round(st.mean(a_fn), 4) if a_fn else None,
            "large_area_rate_TP_pct": round(100 * large_tp / len(sub_tp), 1) if sub_tp else None,
            "large_area_rate_FN_pct": round(100 * large_fn / len(sub_fn), 1) if sub_fn else None,
        })
    with open(OUT_DIR / "melanoma_subtype_TP_FN_pattern.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(subtype_stats[0].keys()))
        w.writeheader(); w.writerows(subtype_stats)

    # --- institution/source association ---
    attrs = sorted(set(r["attribution"] for r in tp_rows + fn_rows))
    table = []
    contingency = []
    for a in attrs:
        tp_n = sum(1 for r in tp_rows if r["attribution"] == a)
        fn_n = sum(1 for r in fn_rows if r["attribution"] == a)
        table.append({"attribution": a, "TP": tp_n, "FN": fn_n,
                      "sensitivity": round(tp_n / (tp_n + fn_n), 4) if (tp_n + fn_n) else None})
        contingency.append([tp_n, fn_n])
    chi2_p = ""
    try:
        chi2, chi2_p, dof, expected = chi2_contingency(contingency)
        chi2_p = round(float(chi2_p), 6)
    except ValueError:
        pass
    with open(OUT_DIR / "melanoma_institution_TP_FN.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["attribution", "TP", "FN", "sensitivity"])
        w.writeheader(); w.writerows(table)
        f.write(f"\nchi2_contingency_p_value,{chi2_p}\n")

    print("\n=== Institution breakdown ===")
    for row in table:
        print(f"  {row['attribution']}: TP={row['TP']} FN={row['FN']} sens={row['sensitivity']}")
    print(f"  chi-square p-value: {chi2_p}")

    # --- console summary for the key questions ---
    print("\n=== Concern combinations, FN only ===")
    for k, v in cfn_counts.items():
        print(f"  {k}: {v} ({100*v/n_fn:.1f}%)")

    c_only_no_ab = cfn_counts.get("C", 0)
    none_at_all = cfn_counts.get("none", 0)
    print(f"\nFN with C concern but NO A/B concern (i.e. 'C only'): {c_only_no_ab} / {n_fn} "
          f"({100*c_only_no_ab/n_fn:.1f}%)")
    print(f"FN with NO A/B/C concern at all: {none_at_all} / {n_fn} ({100*none_at_all/n_fn:.1f}%)")

    print(f"\nWrote outputs to {OUT_DIR}")
    return tp_rows, fn_rows


if __name__ == "__main__":
    main()
