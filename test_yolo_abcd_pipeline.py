"""Test harness: run the EXISTING ABCD pipeline using externally generated
YOLO raster masks instead of the built-in segment_lesion() stage.

This script does NOT modify any file under Code/. It only:
  - reuses the existing preprocessing chain (vignette removal, denoise,
    bilateral filter, hair-width calibration, hair removal) unchanged
  - BYPASSES Code/ComputerVisionStuff/segment_lesion.py
  - loads a pre-computed YOLO raster mask (from raster_masks/) in its place
  - calls the existing, unmodified analyze_abcde() / score_* functions
  - reuses the existing visualize_pipeline()/build_abcd_visuals() for the
    dashboard image, and adds a lightweight mask/contour overlay image

Inputs:
    test_images/    original images (used earlier for YOLO mask inference)
    raster_masks/   YOLO binary masks, named "<stem>_mask.png"

Outputs (NEW directory, existing Results/ and LocalResults/ are untouched):
    YoloMaskABCDTest/
        overlays/<stem>_overlay.png     original + YOLO mask contour/fill
        dashboards/<stem>_dashboard.png full existing ABCD dashboard, reusing
                                         visualize_pipeline() unmodified
        results.csv                     one row per image/mask pair
"""

import contextlib
import csv
import io
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless — never block on plt.show()

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# --- Existing, UNMODIFIED pipeline code -----------------------------------
from HandlingStuff import load_image  # noqa: E402
from ComputerVisionStuff import (  # noqa: E402
    remove_vignette,
    remove_salt_pepper_noise,
    apply_bilateral_filter,
    measure_hair_width_px,
    remove_hair,
    detect_edges,
    # segment_lesion is deliberately NOT imported/used — bypassed for this test
)
from MelanomaDeterminingStuff import analyze_abcde  # noqa: E402
from visualization import visualize_pipeline  # noqa: E402

VELLUS_HAIR_UM = 70.0

IMAGE_DIR = ROOT / "test_images"
MASK_DIR = ROOT / "raster_masks"
OUTPUT_DIR = ROOT / "YoloMaskABCDTest"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
DASHBOARD_DIR = OUTPUT_DIR / "dashboards"
CSV_PATH = OUTPUT_DIR / "results.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

INFO_KEYWORDS = (
    "no circle found",
    "circle too small",
    "no hair detected",
    "too few points",
    "few clean pixels",
    "baseline too dark",
    "no lesion contour",
    "skipped",
)


def find_pairs(image_dir: Path, mask_dir: Path):
    """Match each mask '<stem>_mask.png' to its original image by stem."""
    pairs = []
    unmatched_masks = []
    mask_paths = sorted(mask_dir.glob("*_mask.png"))
    for mask_path in mask_paths:
        stem = mask_path.name[: -len("_mask.png")]
        candidates = [
            image_dir / f"{stem}{ext}"
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
        ]
        image_path = next((c for c in candidates if c.exists()), None)
        if image_path is None:
            unmatched_masks.append(mask_path.name)
            continue
        pairs.append((stem, image_path, mask_path))
    return pairs, unmatched_masks


def load_binary_mask(mask_path: Path, target_shape):
    """Load a mask as strict binary {0,255}, matching target (H, W).

    Also returns diagnostic notes: dimension mismatches, possible inversion,
    or an empty (no-lesion) mask.
    """
    notes = []
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    target_h, target_w = target_shape[:2]
    if raw.shape[0] != target_h or raw.shape[1] != target_w:
        notes.append(
            f"mask size {raw.shape[1]}x{raw.shape[0]} did not match image "
            f"{target_w}x{target_h}; resized with nearest-neighbor"
        )
        raw = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    binary = (raw > 127).astype(np.uint8) * 255

    lesion_fraction = float(np.mean(binary > 0))
    if lesion_fraction == 0.0:
        notes.append("EMPTY MASK: YOLO detected no lesion in this image")
    elif lesion_fraction > 0.5:
        notes.append(
            f"POSSIBLE INVERSION: mask foreground covers {lesion_fraction:.0%} "
            f"of the image (unusually large for a lesion) — verify manually"
        )

    return binary, lesion_fraction, notes


def save_overlay(original_bgr, mask, save_path: Path):
    """Simple visualization: original image + translucent YOLO mask fill + contour."""
    overlay = original_bgr.copy()
    fill = overlay.copy()
    fill[mask > 0] = (0, 0, 255)  # red fill (BGR)
    blended = cv2.addWeighted(overlay, 0.7, fill, 0.3, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = max(2, original_bgr.shape[1] // 300)
    cv2.drawContours(blended, contours, -1, (0, 255, 255), thickness)  # yellow contour

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), blended)


def run_one(stem, image_path: Path, mask_path: Path, make_overlay=True, make_dashboard=True):
    """Run the existing preprocessing + ABCD pipeline for one image/mask pair,
    with segment_lesion() bypassed in favor of the YOLO mask.

    Returns a dict of results plus a list of warning strings and (if raised)
    an error string; never raises for expected per-image conditions.
    """
    row = {
        "image_filename": image_path.name,
        "mask_filename": mask_path.name,
    }
    warnings_list = []
    errors_list = []

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")

            original = load_image(str(image_path))

            yolo_mask, lesion_fraction, mask_notes = load_binary_mask(
                mask_path, original.shape
            )
            warnings_list.extend(mask_notes)
            row["mask_lesion_fraction"] = round(lesion_fraction, 4)

            # Save the overlay immediately, based only on original+mask, so it
            # always exists even if downstream ABCD scoring later raises.
            overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
            if make_overlay:
                save_overlay(original, yolo_mask, overlay_path)
                row["overlay_path"] = str(overlay_path.relative_to(ROOT))
            else:
                row["overlay_path"] = ""

            # --- existing preprocessing chain, UNCHANGED, independent of
            # --- which segmentation method is used downstream
            no_vignette, circle_info = remove_vignette(original)
            row["circle_detected"] = circle_info is not None
            if circle_info is None:
                warnings_list.append("no vignette circle detected; color.py falls back to image-corner skin sampling")

            denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
            bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)

            hair_width_px = measure_hair_width_px(bilateral)
            mm_per_px = (VELLUS_HAIR_UM / hair_width_px) / 1000.0 if hair_width_px else None
            row["hair_width_px"] = hair_width_px
            row["mm_per_px"] = mm_per_px
            if mm_per_px is None:
                warnings_list.append("no hair detected for calibration; diameter (D) will be N/A")

            no_hair = remove_hair(bilateral, kernel_size=17, threshold=10)

            # --- segment_lesion() BYPASSED here — using the YOLO mask instead
            masked = cv2.bitwise_and(no_hair, no_hair, mask=yolo_mask)
            edges = detect_edges(masked, low_threshold=50, high_threshold=150)

            # --- existing, UNMODIFIED ABCD scoring
            abcde = analyze_abcde(
                yolo_mask, no_hair, circle_info=circle_info, mm_per_px=mm_per_px
            )

            # --- dashboard visualization (existing dashboard code, reused unmodified)
            dashboard_path = DASHBOARD_DIR / f"{stem}_dashboard.png"
            if make_dashboard:
                edges_display = cv2.dilate(edges, np.ones((3, 3), np.uint8))
                visualize_pipeline(
                    original, denoised, bilateral, no_hair, yolo_mask,
                    edges_display, abcde, save_path=str(dashboard_path),
                    analysis_mask=yolo_mask,
                )

        # numpy / library RuntimeWarnings captured during the block above
        for w in wlist:
            warnings_list.append(f"{w.category.__name__}: {w.message}")

        # informational prints from the existing pipeline (e.g. "no circle
        # found", "no hair detected", "few clean pixels", ...)
        for line in stdout_buf.getvalue().splitlines():
            lower = line.lower()
            if any(kw in lower for kw in INFO_KEYWORDS):
                warnings_list.append(line.strip())

        if lesion_fraction == 0.0:
            warnings_list.append(
                "NOTE: with an empty mask, asymmetry.py's crop-overlap math degenerates "
                "(0/0-style ratio) and can spuriously report 'Irregular'; color.py's "
                "stats over zero lesion pixels are NaN and always read as 'no concern'. "
                "Treat this row's A/C results as not meaningful, not as a real finding."
            )

        row["A_value"] = abcde["A_asymmetry"]["value"]
        row["A_concern"] = abcde["A_asymmetry"]["concern"]
        row["B_value"] = abcde["B_border"]["value"]
        row["B_concern"] = abcde["B_border"]["concern"]
        row["C_value"] = abcde["C_color"]["value"]
        row["C_concern"] = abcde["C_color"]["concern"]
        row["C_label"] = abcde["C_color"]["label"]
        row["D_value_mm"] = abcde["D_diameter"]["value"]
        row["D_concern"] = abcde["D_diameter"]["concern"]
        row["concerns_count"] = abcde["_summary"]["concerns"]
        row["risk_level"] = abcde["_summary"]["risk_level"]
        row["abcd_summary_label"] = abcde["_summary"]["label"]
        row["benign_malignant_prediction"] = (
            "N/A - existing pipeline only outputs LOW/HIGH risk_level, "
            "it does not classify benign/malignant"
        )
        row["dashboard_path"] = str(dashboard_path.relative_to(ROOT)) if make_dashboard else ""

    except Exception as exc:  # never crash the whole batch
        errors_list.append(f"{type(exc).__name__}: {exc}")
        for key in (
            "mask_lesion_fraction", "circle_detected", "hair_width_px", "mm_per_px",
            "A_value", "A_concern", "B_value", "B_concern", "C_value",
            "C_concern", "C_label", "D_value_mm", "D_concern",
            "concerns_count", "risk_level", "abcd_summary_label",
            "benign_malignant_prediction", "overlay_path", "dashboard_path",
        ):
            row.setdefault(key, "")

    row["warnings"] = " | ".join(warnings_list) if warnings_list else ""
    row["errors"] = " | ".join(errors_list) if errors_list else ""
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    pairs, unmatched_masks = find_pairs(IMAGE_DIR, MASK_DIR)
    if unmatched_masks:
        print(f"[WARN] {len(unmatched_masks)} mask(s) had no matching image: {unmatched_masks}")

    print(f"Found {len(pairs)} image/mask pairs.\n")

    fieldnames = [
        "image_filename", "mask_filename", "mask_lesion_fraction",
        "circle_detected", "hair_width_px", "mm_per_px",
        "A_value", "A_concern", "B_value", "B_concern",
        "C_value", "C_concern", "C_label",
        "D_value_mm", "D_concern",
        "concerns_count", "risk_level", "abcd_summary_label",
        "benign_malignant_prediction",
        "overlay_path", "dashboard_path",
        "warnings", "errors",
    ]

    rows = []
    for stem, image_path, mask_path in pairs:
        print(f"--- {stem} ---")
        row = run_one(stem, image_path, mask_path)
        rows.append(row)
        print(f"    image: {row['image_filename']}  mask: {row['mask_filename']}")
        print(f"    A={row.get('A_value')} B={row.get('B_value')} "
              f"C={row.get('C_value')} D={row.get('D_value_mm')}")
        print(f"    risk_level={row.get('risk_level')} "
              f"concerns={row.get('concerns_count')}")
        if row["warnings"]:
            print(f"    warnings: {row['warnings']}")
        if row["errors"]:
            print(f"    ERRORS: {row['errors']}")
        print()

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} rows written to {CSV_PATH}")


if __name__ == "__main__":
    main()
