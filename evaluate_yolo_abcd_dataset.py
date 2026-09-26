"""Baseline / exploratory evaluation of the EXISTING, UNCHANGED YOLO-mask ->
ABCD pipeline against ground-truth benign/malignant labels.

Does NOT modify:
  - any ABCD formula or threshold (Code/MelanomaDeterminingStuff/*)
  - the YOLO model (inference only, same best.pt, same conf=0.25, no retraining)
  - the original 11-image results (YoloMaskABCDTest/results.csv is read, not
    regenerated or overwritten)

Ground truth comes ONLY from the dataset folder (Images/Benign, Images/Malignant),
which matches each image's own `diagnosis_1` field in metadata.csv. Images
outside those folders (the user's own IMG_*.jpg/png photos) have no known
diagnosis and are excluded from every ground-truth-based metric, but still
appear in the master CSV with ground_truth=unknown.

Provisional (EXPLORATORY ONLY) mapping, kept in separate columns so nothing
about the existing heuristic is renamed or reinterpreted:
    LOW  -> provisional_prediction = negative -> provisional_prediction_binary = 0
    HIGH -> provisional_prediction = positive -> provisional_prediction_binary = 1

IMPORTANT NOTE ON "risk_score": the existing score.py does NOT define a
continuous 0-100 risk score. It only returns `concerns` (an integer count,
0-4) and `risk_level` (LOW/HIGH). Rather than inventing a new formula, this
script uses concerns_count * 25 as `risk_score` (0/25/50/75/100) purely as a
monotonic, already-existing-derived stand-in so ROC-AUC has more than two
distinct values to work with. This is called out explicitly in the final
report and is NOT a claim that the pipeline has a calibrated score.
"""

import contextlib
import csv
import io
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import test_yolo_abcd_pipeline as pipeline  # our own harness, not Code/
from ultralytics import YOLO

ROOT = pipeline.ROOT
IMAGES_ROOT = ROOT / "Images"
MASK_DIR = pipeline.MASK_DIR
OUTPUT_DIR = pipeline.OUTPUT_DIR
OVERLAY_DIR = pipeline.OVERLAY_DIR
DASHBOARD_DIR = pipeline.DASHBOARD_DIR
ORIGINAL_CSV = pipeline.CSV_PATH
MASTER_CSV = OUTPUT_DIR / "master_results.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.json"
REPORT_PATH = OUTPUT_DIR / "exploratory_evaluation_report.txt"

LARGE_DIAMETER_MM = 20.0

# Validated 2-color categorical palette (dataviz skill reference palette,
# slots 1 and 8 of the fixed 8-hue order): blue = benign, red = malignant.
COLOR_BENIGN = "#2a78d6"
COLOR_MALIGNANT = "#e34948"


# ---------------------------------------------------------------------------
# 1. Ground-truth manifest
# ---------------------------------------------------------------------------

def build_manifest():
    """Return {stem: {"path": Path, "ground_truth": "benign"/"malignant"}}."""
    manifest = {}
    for label, folder in (("benign", "Benign"), ("malignant", "Malignant")):
        for p in sorted((IMAGES_ROOT / folder).glob("*.jpg")):
            manifest[p.stem] = {"path": p, "ground_truth": label}
    return manifest


def load_existing_results():
    rows = []
    if ORIGINAL_CSV.exists():
        with open(ORIGINAL_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return rows


# ---------------------------------------------------------------------------
# 2. YOLO inference for the remaining images (no retraining, same weights)
# ---------------------------------------------------------------------------

def run_yolo_for_remaining(remaining_paths):
    """Run inference for images that don't yet have a mask in raster_masks/,
    saving new masks alongside the existing ones. Returns {stem: mask_path}."""
    mask_paths = {}
    to_predict = []
    for stem, path in remaining_paths.items():
        mask_path = MASK_DIR / f"{stem}_mask.png"
        if mask_path.exists():
            mask_paths[stem] = mask_path
        else:
            to_predict.append((stem, path))

    if not to_predict:
        return mask_paths

    print(f"Running YOLO inference (conf=0.25, retina_masks=True) on {len(to_predict)} new image(s)...")
    model = YOLO(pipeline_default_model())
    file_list = [str(p) for _, p in to_predict]
    results = model.predict(source=file_list, retina_masks=True, conf=0.25, verbose=False)

    for (stem, image_path), result in zip(to_predict, results):
        source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        src_shape = source_image.shape
        raster_mask = raster_mask_from_result(result)
        if raster_mask is None:
            binary_mask = np.zeros(src_shape[:2], dtype=np.uint8)
        else:
            binary_mask = to_binary_mask(raster_mask, src_shape)
        mask_path = MASK_DIR / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), binary_mask)
        mask_paths[stem] = mask_path

    return mask_paths


def pipeline_default_model():
    import raster_mask_inference as rmi
    return rmi.DEFAULT_MODEL


def raster_mask_from_result(result):
    import raster_mask_inference as rmi
    return rmi.raster_mask_from_result(result, debug=False)


def to_binary_mask(raster_mask, target_shape):
    import raster_mask_inference as rmi
    return rmi.to_binary_mask(raster_mask, target_shape)


# ---------------------------------------------------------------------------
# 3. Row post-processing: status, ground truth, provisional prediction
# ---------------------------------------------------------------------------

def to_float_or_none(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def to_bool_or_none(x):
    if isinstance(x, bool):
        return x
    if x in ("True", "true", "1"):
        return True
    if x in ("False", "false", "0"):
        return False
    return None


def derive_status(mask_fraction, errors_text, a_value):
    if mask_fraction is not None and mask_fraction == 0.0:
        return "NO_DETECTION"
    if errors_text and (a_value in (None, "")):
        return "FAILED"
    return "PROCESSED"


def finalize_row(raw, manifest, batch_label):
    """Normalize one row (from either the original CSV or a freshly computed
    dict) into the master-CSV schema, attaching ground truth + provisional
    prediction. Never invents A/B/C/D values."""
    image_name = raw.get("image_filename") or raw.get("image_name")
    mask_name = raw.get("mask_filename") or raw.get("mask_name")
    stem = Path(image_name).stem

    mask_fraction = to_float_or_none(raw.get("mask_lesion_fraction"))
    errors_text = raw.get("errors", "") or ""
    warnings_text = raw.get("warnings", "") or ""
    a_value = raw.get("A_value")
    status = derive_status(mask_fraction, errors_text, a_value)

    gt_info = manifest.get(stem)
    if gt_info is not None:
        ground_truth = gt_info["ground_truth"]
        ground_truth_binary = 1 if ground_truth == "malignant" else 0
    else:
        ground_truth = "unknown"
        ground_truth_binary = ""

    risk_level = raw.get("risk_level") or ""
    concerns_count = to_float_or_none(raw.get("concerns_count"))
    risk_score = concerns_count * 25 if concerns_count is not None else ""

    if status == "PROCESSED" and risk_level in ("LOW", "HIGH"):
        provisional_prediction = "negative" if risk_level == "LOW" else "positive"
        provisional_prediction_binary = 0 if risk_level == "LOW" else 1
    else:
        provisional_prediction = ""
        provisional_prediction_binary = ""

    if (
        status == "PROCESSED"
        and ground_truth_binary != ""
        and provisional_prediction_binary != ""
    ):
        correct = int(provisional_prediction_binary == ground_truth_binary)
    else:
        correct = ""

    d_mm = to_float_or_none(raw.get("D_value_mm"))
    hair_width_px = to_float_or_none(raw.get("hair_width_px"))
    mm_per_px = to_float_or_none(raw.get("mm_per_px"))
    d_diameter_px = (d_mm / mm_per_px) if (d_mm is not None and mm_per_px) else None

    row = {
        "batch": batch_label,
        "image_name": image_name,
        "mask_name": mask_name,
        "status": status,
        "ground_truth": ground_truth,
        "ground_truth_binary": ground_truth_binary,
        "mask_lesion_fraction": mask_fraction,
        "circle_detected": raw.get("circle_detected", ""),
        "hair_width_px": hair_width_px if hair_width_px is not None else "",
        "mm_per_px": mm_per_px if mm_per_px is not None else "",
        "A_raw": raw.get("A_value", ""),
        "A_concern": raw.get("A_concern", ""),
        "B_raw": raw.get("B_value", ""),
        "B_concern": raw.get("B_concern", ""),
        "C_raw": raw.get("C_value", ""),
        "C_concern": raw.get("C_concern", ""),
        "C_label": raw.get("C_label", ""),
        "D_raw": raw.get("D_value_mm", ""),
        "D_mm": raw.get("D_value_mm", ""),
        "D_diameter_px": round(d_diameter_px, 1) if d_diameter_px is not None else "",
        "D_concern": raw.get("D_concern", ""),
        "concerns_count": raw.get("concerns_count", ""),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "abcd_summary_label": raw.get("abcd_summary_label", ""),
        "provisional_prediction": provisional_prediction,
        "provisional_prediction_binary": provisional_prediction_binary,
        "correct": correct,
        "overlay_path": raw.get("overlay_path", ""),
        "dashboard_path": raw.get("dashboard_path", ""),
        "error_or_warning": " | ".join(x for x in (warnings_text, errors_text) if x),
    }
    return row


MASTER_FIELDNAMES = [
    "batch", "image_name", "mask_name", "status",
    "ground_truth", "ground_truth_binary",
    "mask_lesion_fraction", "circle_detected", "hair_width_px", "mm_per_px",
    "A_raw", "A_concern", "B_raw", "B_concern",
    "C_raw", "C_concern", "C_label",
    "D_raw", "D_mm", "D_diameter_px", "D_concern",
    "concerns_count", "risk_score", "risk_level", "abcd_summary_label",
    "provisional_prediction", "provisional_prediction_binary", "correct",
    "overlay_path", "dashboard_path", "error_or_warning",
]


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def confusion_counts(rows):
    tp = tn = fp = fn = 0
    for r in rows:
        if r["status"] != "PROCESSED" or r["ground_truth_binary"] == "":
            continue
        gt = r["ground_truth_binary"]
        pred = r["provisional_prediction_binary"]
        if pred == "":
            continue
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
    return tp, tn, fp, fn


def safe_div(a, b):
    return a / b if b else float("nan")


def roc_auc(rows):
    """Rank-based (Mann-Whitney U) AUC of risk_score vs ground_truth_binary."""
    scores, labels = [], []
    for r in rows:
        if r["status"] != "PROCESSED" or r["ground_truth_binary"] == "" or r["risk_score"] == "":
            continue
        scores.append(float(r["risk_score"]))
        labels.append(int(r["ground_truth_binary"]))
    scores = np.array(scores)
    labels = np.array(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return None, n_pos, n_neg
    ranks = rankdata(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def group_stats(rows, field, ground_truth_filter):
    vals = []
    missing = 0
    for r in rows:
        if r["status"] != "PROCESSED" or r["ground_truth"] != ground_truth_filter:
            continue
        v = r[field]
        if v == "" or v is None:
            missing += 1
            continue
        vals.append(float(v))
    if not vals:
        return {"count": 0, "missing": missing, "mean": None, "median": None,
                "std": None, "min": None, "max": None}
    arr = np.array(vals)
    return {
        "count": len(vals), "missing": missing,
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def concern_rate(rows, concern_field, ground_truth_filter):
    total = 0
    flagged = 0
    for r in rows:
        if r["status"] != "PROCESSED" or r["ground_truth"] != ground_truth_filter:
            continue
        v = r[concern_field]
        if v == "":
            continue
        total += 1
        if str(v) in ("True", "1", "1.0"):
            flagged += 1
    return flagged, total


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def make_comparison_plot(rows, field, title, ylabel, save_path):
    benign_vals = [float(r[field]) for r in rows
                   if r["status"] == "PROCESSED" and r["ground_truth"] == "benign" and r[field] not in ("", None)]
    malignant_vals = [float(r[field]) for r in rows
                      if r["status"] == "PROCESSED" and r["ground_truth"] == "malignant" and r[field] not in ("", None)]

    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    data = [benign_vals, malignant_vals]
    positions = [1, 2]
    bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True,
                     showfliers=False, medianprops={"color": "#0b0b0b", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], [COLOR_BENIGN, COLOR_MALIGNANT]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, [COLOR_BENIGN, COLOR_MALIGNANT]):
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, color=color,
                   alpha=0.85, s=28, edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Benign (n={len(benign_vals)})", f"Malignant (n={len(malignant_vals)})"],
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
            ax.annotate(f"mean={np.mean(vals):.2f}", (pos, max(vals)),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color="#52514e")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    existing_rows_raw = load_existing_results()
    existing_stems = {Path(r["image_filename"]).stem for r in existing_rows_raw}

    remaining = {stem: info["path"] for stem, info in manifest.items() if stem not in existing_stems}
    print(f"Dataset total (Benign+Malignant): {len(manifest)}")
    print(f"Already processed (kept as-is): {len(existing_rows_raw)} rows")
    print(f"Remaining to process now: {len(remaining)}")

    mask_paths = run_yolo_for_remaining(remaining)

    n_new_processed = 0
    n_new_no_detection = 0
    n_new_failed = 0
    new_raw_rows = []

    t_process_start = time.time()
    for stem, image_path in sorted(remaining.items()):
        mask_path = mask_paths[stem]
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        lesion_fraction = float(np.mean(raw_mask > 127)) if raw_mask is not None else 0.0

        if lesion_fraction == 0.0:
            print(f"--- {stem}: NO_DETECTION (empty YOLO mask) — ABCD not run ---")
            new_raw_rows.append({
                "image_filename": image_path.name,
                "mask_filename": mask_path.name,
                "mask_lesion_fraction": 0.0,
                "warnings": "EMPTY MASK: YOLO detected no lesion; ABCD deliberately skipped per no-detection policy",
                "errors": "",
            })
            n_new_no_detection += 1
            continue

        row = pipeline.run_one(stem, image_path, mask_path, make_overlay=False, make_dashboard=False)
        new_raw_rows.append(row)
        if row["errors"]:
            n_new_failed += 1
            print(f"--- {stem}: FAILED ({row['errors']}) ---")
        else:
            n_new_processed += 1
            print(f"--- {stem}: A={row.get('A_value')} B={row.get('B_value')} "
                  f"C={row.get('C_value')} D={row.get('D_value_mm')} risk={row.get('risk_level')} ---")

    t_process_end = time.time()

    # --- finalize all rows (existing + new) into the master schema
    master_rows = []
    for r in existing_rows_raw:
        master_rows.append(finalize_row(r, manifest, "initial_11"))
    for r in new_raw_rows:
        master_rows.append(finalize_row(r, manifest, "expanded_dataset"))

    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(master_rows)
    print(f"\nMaster CSV written: {MASTER_CSV} ({len(master_rows)} rows)")

    # --- select images needing visualization: representative sample + unusual/failure cases
    labeled_processed = [r for r in master_rows if r["status"] == "PROCESSED" and r["ground_truth"] != "unknown"]
    fp_rows = [r for r in labeled_processed if r["ground_truth_binary"] == 0 and r["provisional_prediction_binary"] == 1]
    fn_rows = [r for r in labeled_processed if r["ground_truth_binary"] == 1 and r["provisional_prediction_binary"] == 0]
    tp_rows = [r for r in labeled_processed if r["ground_truth_binary"] == 1 and r["provisional_prediction_binary"] == 1]
    tn_rows = [r for r in labeled_processed if r["ground_truth_binary"] == 0 and r["provisional_prediction_binary"] == 0]
    no_detection_rows = [r for r in master_rows if r["status"] == "NO_DETECTION"]
    failed_rows = [r for r in master_rows if r["status"] == "FAILED"]
    large_diam_rows = [r for r in master_rows if r["D_mm"] not in ("", None) and float(r["D_mm"]) > LARGE_DIAMETER_MM]
    suspicious_mask_rows = [r for r in master_rows if r["mask_lesion_fraction"] not in ("", None) and float(r["mask_lesion_fraction"]) > 0.5]

    new_stems_processed = [Path(r["image_name"]).stem for r in master_rows
                            if r["batch"] == "expanded_dataset" and r["status"] == "PROCESSED"]
    representative = new_stems_processed[:1] + (new_stems_processed[len(new_stems_processed)//2:len(new_stems_processed)//2+1] if new_stems_processed else []) + new_stems_processed[-1:]

    selected_stems = set(representative)
    for group in (fp_rows, fn_rows, large_diam_rows, suspicious_mask_rows):
        selected_stems.update(Path(r["image_name"]).stem for r in group if r["batch"] == "expanded_dataset")
    selected_stems.update(Path(r["image_name"]).stem for r in no_detection_rows if r["batch"] == "expanded_dataset")

    print(f"\nGenerating overlay/dashboard visuals for {len(selected_stems)} selected image(s): {sorted(selected_stems)}")
    for stem in selected_stems:
        image_path = remaining.get(stem)
        if image_path is None:
            continue
        mask_path = mask_paths[stem]
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        lesion_fraction = float(np.mean(raw_mask > 127)) if raw_mask is not None else 0.0
        if lesion_fraction == 0.0:
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            binary_mask = (raw_mask > 127).astype(np.uint8) * 255 if raw_mask is not None else np.zeros(original.shape[:2], np.uint8)
            overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
            pipeline.save_overlay(original, binary_mask, overlay_path)
            for r in master_rows:
                if r["image_name"] == image_path.name:
                    r["overlay_path"] = str(overlay_path.relative_to(ROOT))
        else:
            row = pipeline.run_one(stem, image_path, mask_path, make_overlay=True, make_dashboard=True)
            for r in master_rows:
                if r["image_name"] == image_path.name:
                    r["overlay_path"] = row.get("overlay_path", "")
                    r["dashboard_path"] = row.get("dashboard_path", "")

    # rewrite master CSV with updated visualization paths
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(master_rows)

    # --- metrics
    tp, tn, fp, fn = confusion_counts(master_rows)
    n_eval = tp + tn + fp + fn
    accuracy = safe_div(tp + tn, n_eval)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * sensitivity, precision + sensitivity) if not (np.isnan(precision) or np.isnan(sensitivity)) else float("nan")
    auc, n_pos, n_neg = roc_auc(master_rows)

    # --- distribution stats + concern rates
    stats = {}
    for field, label in (("A_raw", "A"), ("B_raw", "B"), ("C_raw", "C"), ("D_mm", "D"), ("risk_score", "risk_score")):
        stats[label] = {
            "benign": group_stats(master_rows, field, "benign"),
            "malignant": group_stats(master_rows, field, "malignant"),
        }

    concern_rates = {}
    for field, label in (("A_concern", "A"), ("B_concern", "B"), ("C_concern", "C"), ("D_concern", "D")):
        b_flag, b_total = concern_rate(master_rows, field, "benign")
        m_flag, m_total = concern_rate(master_rows, field, "malignant")
        concern_rates[label] = {
            "benign": f"{b_flag}/{b_total} ({safe_div(b_flag, b_total)*100:.0f}%)" if b_total else "n/a",
            "malignant": f"{m_flag}/{m_total} ({safe_div(m_flag, m_total)*100:.0f}%)" if m_total else "n/a",
        }

    # --- plots
    make_comparison_plot(master_rows, "A_raw", "Asymmetry (A) — benign vs malignant", "asymmetry score", PLOTS_DIR / "A_asymmetry.png")
    make_comparison_plot(master_rows, "B_raw", "Border irregularity (B) — benign vs malignant", "border irregularity", PLOTS_DIR / "B_border.png")
    make_comparison_plot(master_rows, "C_raw", "Color variation (C) — benign vs malignant", "color CV", PLOTS_DIR / "C_color.png")
    make_comparison_plot(master_rows, "D_mm", "Diameter (D) — benign vs malignant", "diameter (mm)", PLOTS_DIR / "D_diameter.png")
    make_comparison_plot(master_rows, "risk_score", "Risk score proxy — benign vs malignant", "risk_score (0-100 proxy)", PLOTS_DIR / "risk_score.png")

    # --- diameter investigation
    diameter_flags = []
    for r in master_rows:
        if r["D_mm"] not in ("", None) and float(r["D_mm"]) > LARGE_DIAMETER_MM:
            diameter_flags.append({
                "image_name": r["image_name"],
                "D_mm": r["D_mm"],
                "D_diameter_px": r["D_diameter_px"],
                "hair_width_px": r["hair_width_px"],
                "mm_per_px": r["mm_per_px"],
                "ground_truth": r["ground_truth"],
            })

    # --- error analysis payload
    def compact(rows_subset):
        return [{
            "image_name": r["image_name"], "ground_truth": r["ground_truth"],
            "A_raw": r["A_raw"], "A_concern": r["A_concern"],
            "B_raw": r["B_raw"], "B_concern": r["B_concern"],
            "C_raw": r["C_raw"], "C_concern": r["C_concern"],
            "D_mm": r["D_mm"], "D_concern": r["D_concern"],
            "risk_score": r["risk_score"], "risk_level": r["risk_level"],
        } for r in rows_subset]

    error_analysis = {
        "true_positives": compact(tp_rows),
        "true_negatives": compact(tn_rows),
        "false_positives": compact(fp_rows),
        "false_negatives": compact(fn_rows),
        "no_detection": [{"image_name": r["image_name"], "ground_truth": r["ground_truth"]} for r in no_detection_rows],
        "processing_failures": [{"image_name": r["image_name"], "error": r["error_or_warning"]} for r in failed_rows],
        "large_diameter_flags": diameter_flags,
        "suspicious_mask_alignment": [{"image_name": r["image_name"], "mask_lesion_fraction": r["mask_lesion_fraction"]} for r in suspicious_mask_rows],
    }
    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, default=str)

    t_end = time.time()
    runtime_report = {
        "already_processed_before_this_run": len(existing_rows_raw),
        "newly_processed_this_run": len(remaining),
        "new_processed_ok": n_new_processed,
        "new_no_detection": n_new_no_detection,
        "new_failed": n_new_failed,
        "total_rows_in_master_csv": len(master_rows),
        "total_runtime_sec": round(t_end - t_start, 1),
        "processing_only_sec": round(t_process_end - t_process_start, 1),
        "avg_sec_per_new_image": round((t_process_end - t_process_start) / len(remaining), 2) if remaining else None,
    }

    # --- write text report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("EXPLORATORY PERFORMANCE OF THE EXISTING HEURISTIC ABCD RISK SYSTEM\n")
        f.write("using LOW->negative and HIGH->positive. NOT a validated diagnostic accuracy.\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Evaluated (labeled, processed): {n_eval}\n")
        f.write(f"TP={tp} TN={tn} FP={fp} FN={fn}\n")
        f.write(f"Accuracy={accuracy:.3f} Sensitivity={sensitivity:.3f} Specificity={specificity:.3f} "
                f"Precision={precision:.3f} F1={f1:.3f}\n")
        if auc is not None:
            f.write(f"ROC-AUC (risk_score proxy vs ground truth, n_pos={n_pos}, n_neg={n_neg}) = {auc:.3f}\n")
        else:
            f.write("ROC-AUC: not computable (one class empty)\n")
        f.write("\nDistribution stats (benign vs malignant):\n")
        f.write(json.dumps(stats, indent=2))
        f.write("\n\nConcern rates (benign vs malignant):\n")
        f.write(json.dumps(concern_rates, indent=2))
        f.write("\n\nRuntime:\n")
        f.write(json.dumps(runtime_report, indent=2))

    print("\n" + "=" * 60)
    print("DONE")
    print(json.dumps({
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1, "auc": auc,
        "runtime": runtime_report,
    }, indent=2, default=str))

    return {
        "master_rows": master_rows,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1, "auc": auc, "n_pos": n_pos, "n_neg": n_neg,
        "stats": stats, "concern_rates": concern_rates,
        "diameter_flags": diameter_flags, "runtime": runtime_report,
        "fp_rows": fp_rows, "fn_rows": fn_rows,
        "no_detection_rows": no_detection_rows, "failed_rows": failed_rows,
    }


if __name__ == "__main__":
    main()
