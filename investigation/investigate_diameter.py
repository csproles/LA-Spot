"""Diagnostic-only investigation of the D (diameter) criterion and its
hair-width-based physical calibration.

Does NOT modify Code/ComputerVisionStuff/hair.py or Code/main.py. This is a
line-for-line reproduction of measure_hair_width_px, instrumented to record
every intermediate quantity (verified to reproduce the same hair_width_px
values already in the master CSV), plus a component-level elongation check
that is NOT part of the current implementation, added purely to show what a
"is this actually hair-shaped?" filter would do differently.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ComputerVisionStuff import remove_vignette, remove_salt_pepper_noise, apply_bilateral_filter  # noqa: E402
from HandlingStuff import load_image  # noqa: E402

IMAGES_ROOT = ROOT / "Images"
OUT_DIR = Path(__file__).resolve().parent / "diameter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VELLUS_HAIR_UM_CURRENT = 70.0  # the constant actually used in main.py today


def build_manifest():
    manifest = {}
    for label, folder in (("benign", "Benign"), ("malignant", "Malignant")):
        for p in sorted((IMAGES_ROOT / folder).glob("*.jpg")):
            manifest[p.stem] = {"image_path": p, "ground_truth": label}
    return manifest


def instrumented_measure_hair_width_px(img, kernel_size=17, threshold=10):
    """Exact reproduction of Code/ComputerVisionStuff/hair.py::measure_hair_width_px,
    with every intermediate value captured."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    hair_mask_area = int(np.sum(hair_mask > 0))
    if hair_mask_area < 50:
        return {"hair_width_px": None, "reason": "hair_mask_area<50", "hair_mask": hair_mask,
                "blackhat": blackhat, "hair_mask_area": hair_mask_area, "n_components": 0,
                "half_widths_n": 0}

    dist = cv2.distanceTransform(hair_mask, cv2.DIST_L2, 5)
    kernel_sk = np.ones((3, 3), np.uint8)
    dist_dilated = cv2.dilate(dist, kernel_sk)
    skeleton = (dist == dist_dilated) & (dist > 0)

    half_widths = dist[skeleton]
    half_widths = half_widths[(half_widths >= 0.5) & (half_widths <= 8.0)]

    if len(half_widths) < 10:
        return {"hair_width_px": None, "reason": f"too_few_points({len(half_widths)})",
                "hair_mask": hair_mask, "blackhat": blackhat, "hair_mask_area": hair_mask_area,
                "n_components": 0, "half_widths_n": len(half_widths)}

    hair_width_px = float(np.median(half_widths) * 2)

    # --- component-level elongation analysis (NOT part of current code) ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask)
    elongations = []
    for i in range(1, n_labels):
        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = contours[0]
        if len(c) < 5:
            elongations.append(1.0)  # too small to fit an ellipse; treat as blob-like
            continue
        (_, _), (minor, major), _ = cv2.fitEllipse(c)
        if minor < 1e-6:
            continue
        elongations.append(major / minor)

    hair_like = sum(1 for e in elongations if e >= 4.0)  # aspect ratio >=4:1 = hair-shaped
    blob_like = sum(1 for e in elongations if e < 4.0)

    return {
        "hair_width_px": hair_width_px, "reason": "ok",
        "hair_mask": hair_mask, "blackhat": blackhat, "hair_mask_area": hair_mask_area,
        "n_components": n_labels - 1, "half_widths_n": len(half_widths),
        "mean_elongation": float(np.mean(elongations)) if elongations else None,
        "median_elongation": float(np.median(elongations)) if elongations else None,
        "n_hair_like_components": hair_like, "n_blob_like_components": blob_like,
    }


def preprocess_to_bilateral(image_path):
    """Reproduce main.py's preprocessing chain up to the bilateral-filtered
    image, exactly as measure_hair_width_px receives it in production."""
    original = load_image(str(image_path))
    no_vignette, _ = remove_vignette(original)
    denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
    bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
    return original, bilateral


def main():
    manifest = build_manifest()
    rows = []
    for stem, info in sorted(manifest.items()):
        original, bilateral = preprocess_to_bilateral(info["image_path"])
        result = instrumented_measure_hair_width_px(bilateral)

        hair_width_px = result["hair_width_px"]
        mm_per_px_current = (VELLUS_HAIR_UM_CURRENT / hair_width_px) / 1000.0 if hair_width_px else None

        rows.append({
            "stem": stem, "ground_truth": info["ground_truth"],
            "hair_width_px": round(hair_width_px, 3) if hair_width_px else "",
            "reason": result["reason"],
            "hair_mask_area_px": result["hair_mask_area"],
            "n_components": result["n_components"],
            "half_widths_n": result["half_widths_n"],
            "mean_elongation": round(result.get("mean_elongation") or 0, 2) if result.get("mean_elongation") else "",
            "n_hair_like_components": result.get("n_hair_like_components", ""),
            "n_blob_like_components": result.get("n_blob_like_components", ""),
            "mm_per_px_current": mm_per_px_current,
            "hit_2px_floor": hair_width_px == 2.0,
        })
        print(f"{stem}: hair_width_px={hair_width_px} reason={result['reason']} "
              f"hair_like={result.get('n_hair_like_components')} blob_like={result.get('n_blob_like_components')} "
              f"mean_elong={result.get('mean_elongation')}")

    csv_path = OUT_DIR / "diameter_hairwidth_diagnostics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    n_total = len(rows)
    n_none = sum(1 for r in rows if r["hair_width_px"] == "")
    n_2px = sum(1 for r in rows if r["hit_2px_floor"])
    n_hair_like_dominant = sum(1 for r in rows if r["n_hair_like_components"] != "" and
                               r["n_hair_like_components"] >= r["n_blob_like_components"])
    print(f"\nOf {n_total} images:")
    print(f"  {n_none} returned None ('no reliable hair')")
    print(f"  {n_2px} hit exactly hair_width_px=2.0 (the effective floor)")
    print(f"  {n_total - n_none} returned SOME numeric value (i.e. calibration was attempted)")
    print(f"  {n_hair_like_dominant} of those had hair-LIKE (elongated, aspect>=4:1) components "
          f"as the majority of detected blobs")

    # --- visualize: a 2px-floor case and a "normal-looking" hair case, side by side
    floor_cases = [r for r in rows if r["hit_2px_floor"]]
    normal_cases = [r for r in rows if r["hair_width_px"] != "" and not r["hit_2px_floor"]
                    and r["n_hair_like_components"] != "" and r["n_hair_like_components"] > r["n_blob_like_components"]]
    picks = (floor_cases[:2] if floor_cases else []) + (normal_cases[:1] if normal_cases else [])

    for r in picks:
        stem = r["stem"]
        info = manifest[stem]
        original, bilateral = preprocess_to_bilateral(info["image_path"])
        result = instrumented_measure_hair_width_px(bilateral)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#fcfcfb")
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"{stem} ({r['ground_truth']})\nhair_width_px={r['hair_width_px']}")
        axes[0].axis("off")
        axes[1].imshow(result["blackhat"], cmap="gray")
        axes[1].set_title("blackhat response\n(dark-on-light detector, any scale)")
        axes[1].axis("off")
        axes[2].imshow(result["hair_mask"], cmap="gray")
        axes[2].set_title(f"hair_mask (threshold>10)\ncomponents={r['n_components']} "
                          f"hair-like={r['n_hair_like_components']} blob-like={r['n_blob_like_components']}")
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{stem}_hairwidth_diagnostic.png", dpi=130)
        plt.close(fig)

    print(f"\nSaved diagnostic visualizations for: {[r['stem'] for r in picks]}")


if __name__ == "__main__":
    main()
