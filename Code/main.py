import sys
import cv2
import numpy as np

from HandlingStuff       import load_image
from ComputerVisionStuff import (remove_vignette, remove_salt_pepper_noise,
                                  apply_bilateral_filter, measure_hair_width_px,
                                  remove_hair, segment_lesion, detect_edges)
from MelanomaDeterminingStuff import analyze_abcde
from visualization            import visualize_pipeline

VELLUS_HAIR_UM = 70.0

def run_pipeline(image_path, output_path=None):

    print("\n" + "═" * 50)
    print("  MELANOMA DETECTION PIPELINE")
    print("═" * 50)

    original                   = load_image(image_path)
    no_vignette, circle_info   = remove_vignette(original)
    denoised                   = remove_salt_pepper_noise(no_vignette, kernel_size=3)
    bilateral                  = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
    hair_width_px              = measure_hair_width_px(bilateral)
    mm_per_px                  = (VELLUS_HAIR_UM / hair_width_px) / 1000.0 if hair_width_px else None
    no_hair                    = remove_hair(bilateral, kernel_size=17, threshold=10)
    mask, masked               = segment_lesion(no_hair)
    edges                      = detect_edges(masked, low_threshold=50, high_threshold=150)
    abcde                      = analyze_abcde(mask, no_hair, circle_info=circle_info, mm_per_px=mm_per_px)

    print("\n" + "─" * 50)
    print("  ABCD RESULTS")
    print("─" * 50)
    for val in abcde.values():
        print(f"  {val['label']}")
    print("─" * 50 + "\n")

    print("\n" + "═" * 50)
    print("  MELANOMA DETECTION PIPELINE")
    print("═" * 50)

    original                   = load_image(image_path)
    no_vignette, circle_info   = remove_vignette(original)
    denoised                   = remove_salt_pepper_noise(no_vignette, kernel_size=3)
    bilateral                  = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
    hair_width_px              = measure_hair_width_px(bilateral)
    mm_per_px                  = (VELLUS_HAIR_UM / hair_width_px) / 1000.0 if hair_width_px else None
    no_hair                    = remove_hair(bilateral, kernel_size=17, threshold=10)
    mask, masked               = segment_lesion(no_hair)
    edges                      = detect_edges(masked, low_threshold=50, high_threshold=150)
    abcde                      = analyze_abcde(mask, no_hair, circle_info=circle_info, mm_per_px=mm_per_px)

    print("\n" + "─" * 50)
    print("  ABCD RESULTS")
    print("─" * 50)
    for val in abcde.values():
        print(f"  {val['label']}")
    print("─" * 50 + "\n")

    if output_path:
        base = output_path.rsplit(".", 1)[0]

        # Pipeline stage images
        for img_stage, label in [
            (original,  "1_original"),
            (denoised,  "2_median_filter"),
            (bilateral, "3_bilateral_filter"),
            (no_hair,   "4_hair_removal"),
            (mask,      "5_segmentation"),
            (edges,     "6_canny_edges"),
        ]:
            cv2.imwrite(f"{base}_{label}.png", img_stage)
            print(f"[✔] Stage image saved: {base}_{label}.png")

        # ABCD criterion images
        from visualization import build_abcd_visuals
        abcd_visuals = build_abcd_visuals(original, mask, abcde)
        for key, label in [
            ("A", "7_asymmetry"),
            ("B", "8_border"),
            ("C", "9_color"),
            ("D", "10_diameter"),
        ]:
            # ABCD visuals are RGB — convert back to BGR for cv2.imwrite
            img_bgr = cv2.cvtColor(abcd_visuals[key], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{base}_{label}.png", img_bgr)
            print(f"[✔] ABCD image saved: {base}_{label}.png")

    edges_display = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    visualize_pipeline(original, denoised, bilateral, no_hair, mask,
                       edges_display, abcde, save_path=output_path)
    return abcde


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python main.py <image_path> [output_path]")
        print("Example: python main.py Images/Benign/ISIC_0000005.jpg Results/result.png")
        sys.exit(1)
    run_pipeline(sys.argv[1], output_path=sys.argv[2] if len(sys.argv) > 2 else None)