"""PHASE 10 — Evolution / change-over-time interface.

Not a validated longitudinal or physical-growth model. This module only
packages a single analysis into a stable, comparable record, and computes
simple relative differences between two records of presumably the same
spot. It does not claim, and must not be described as, a measurement of
true physical growth: camera distance, crop, resolution, and angle can all
change a lesion's apparent pixel size and apparent shape between photos
without the real lesion changing at all.
"""

from datetime import datetime, timezone


def make_evolution_record(structured_result, captured_at=None, spot_label=None):
    """Turns one analyze_image() result (see structured_output.py) into a
    stable record suitable for storing and later comparing. Only the
    primary instance is recorded for single-lesion images; multi-lesion
    images are recorded as-is (comparison across sessions is left to the
    caller / user to match up instances, since instance IDs are not stable
    identities across separate photos)."""
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    instances = structured_result.get("lesion_instances", [])
    return {
        "spot_label": spot_label,
        "captured_at": captured_at,
        "segmentation_status": structured_result.get("segmentation_status"),
        "num_lesion_instances": structured_result.get("num_lesion_instances", 0),
        "instances": [
            {
                "asymmetry_value": inst["asymmetry"]["value"],
                "border_value": inst["border"]["value"],
                "color_value": inst["color"]["value"],
                "diameter_px": inst["diameter_px"],
                "mask_area_fraction": inst["mask_area_fraction"],
                "overall_visual_concern": inst["overall_visual_concern"],
            }
            for inst in instances
        ],
        "config_version": structured_result.get("config_version"),
    }


def compare_records(earlier, later):
    """Simple, cautious relative-change summary between two records of
    (presumably) the same spot, using their PRIMARY instance only. Never
    claims physical growth — reports pixel-based and score-based deltas
    only, with an explicit caveat."""
    if not earlier.get("instances") or not later.get("instances"):
        return {"comparable": False, "reason": "one or both records have no detected lesion instance"}

    e, l = earlier["instances"][0], later["instances"][0]

    def delta(key):
        a, b = e.get(key), l.get(key)
        if a is None or b is None:
            return None
        return round(b - a, 4)

    return {
        "comparable": True,
        "caveat": "Pixel-based deltas only. Camera distance, crop, resolution, and angle can all "
                  "change apparent size/shape between photos without true physical change. This "
                  "is visual change tracking, not a validated measurement of physical growth.",
        "asymmetry_delta": delta("asymmetry_value"),
        "border_delta": delta("border_value"),
        "color_delta": delta("color_value"),
        "diameter_px_delta": delta("diameter_px"),
        "mask_area_fraction_delta": delta("mask_area_fraction"),
        "concern_changed": e.get("overall_visual_concern") != l.get("overall_visual_concern"),
        "earlier_concern": e.get("overall_visual_concern"),
        "later_concern": l.get("overall_visual_concern"),
    }
