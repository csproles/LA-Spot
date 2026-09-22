import numpy as np

from MelanomaDeterminingStuff.asymmetry import score_asymmetry
from MelanomaDeterminingStuff.border    import score_border
from MelanomaDeterminingStuff.color     import score_color
from MelanomaDeterminingStuff.diameter  import score_diameter


def analyze_abcde(
    mask,
    original,
    circle_info=None,
    mm_per_px=None,
):
    results = {}

    results["A_asymmetry"] = score_asymmetry(mask)
    results["B_border"]    = score_border(mask)

    color_result, pink_red_px, blue_gray_px, white_px, black_px = score_color(
        mask, original, circle_info
    )
    results["C_color"]    = color_result
    results["D_diameter"] = score_diameter(mask, mm_per_px)

    concerns = sum(1 for v in results.values() if v["concern"])

    critical_color    = max(pink_red_px, blue_gray_px, white_px, black_px)
    critical_override = critical_color > 0.50

    if critical_override:
        risk_level = "HIGH"
        print(f"Critical color override triggered ({critical_color:.0%} dominant dangerous color)")
    else:
        risk_level = "LOW" if concerns <= 1 else "HIGH"

    n_scored = "4" if results["D_diameter"]["value"] is not None else "3"
    results["_summary"] = {
        "concerns":   concerns,
        "risk_level": risk_level,
        "label":      f"Overall Risk: {risk_level}  ({concerns}/{n_scored} criteria flagged)",
    }

    return results