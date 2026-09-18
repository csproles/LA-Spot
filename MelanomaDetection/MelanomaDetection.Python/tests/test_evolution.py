"""Change scoring between two checks of the same spot.

The failure mode that matters here is a false positive: telling someone their
mole is changing when all that changed is how they held the phone. The first
three tests exist specifically to pin that down.
"""

import cv2
import numpy as np
import pytest

import evolution


def disc(radius, shape=(200, 200), center=None, squash=1.0):
    mask = np.zeros(shape, np.uint8)
    center_y, center_x = center or (shape[0] // 2, shape[1] // 2)
    cv2.ellipse(
        mask, (center_x, center_y), (int(radius), int(radius * squash)), 0, 0, 360, 255, -1
    )
    return mask


def check(mask, lab=(135.0, 153.0, 132.0), mm_per_px=None, risk_score=50.0):
    return {
        "mask": mask,
        "area_px": evolution.lesion_area_px(mask),
        "lab": lab,
        "mm_per_px": mm_per_px,
        "risk_score": risk_score,
    }


class TestNoFalsePositives:
    def test_identical_masks_score_zero(self):
        score, _ = evolution.score_change(check(disc(30)), check(disc(30)))
        assert score == 0.0

    def test_translation_alone_is_not_change(self):
        """Centroid alignment: the same mole photographed off-center."""
        score, _ = evolution.score_change(
            check(disc(30, center=(60, 140))), check(disc(30))
        )
        assert score == 0.0

    def test_rotation_alone_is_not_change(self):
        """A lopsided shape photographed at 90 degrees should still match itself."""
        upright = disc(30, squash=0.5)
        rotated = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
        score, details = evolution.score_change(check(rotated), check(upright))
        assert details["shape_iou"] > 0.95
        assert score < 1.0


class TestGrowthRequiresCalibration:
    def test_growth_is_not_scored_without_mm_calibration(self):
        """Apparent size is dominated by camera distance, so it must be withheld."""
        score, details = evolution.score_change(
            check(disc(45)), check(disc(30))
        )
        assert details["size_calibrated"] is False
        assert "growth" not in details["signals"]
        assert "area_growth_ratio" not in details
        assert "size_note" in details

    def test_growth_is_scored_when_both_checks_are_calibrated(self):
        score, details = evolution.score_change(
            check(disc(45), mm_per_px=0.05), check(disc(30), mm_per_px=0.05)
        )
        assert details["size_calibrated"] is True
        assert "growth" in details["signals"]
        assert details["area_growth_ratio"] > 0.3
        assert score >= 5.0  # past the concern threshold

    def test_one_sided_calibration_is_still_uncalibrated(self):
        _, details = evolution.score_change(
            check(disc(45), mm_per_px=0.05), check(disc(30), mm_per_px=None)
        )
        assert details["size_calibrated"] is False

    def test_shrinking_does_not_raise_the_score(self):
        """A lesion getting smaller is not the evolution being screened for."""
        _, details = evolution.score_change(
            check(disc(20), mm_per_px=0.05), check(disc(30), mm_per_px=0.05)
        )
        assert details["area_growth_ratio"] < 0


class TestChangeIsDetected:
    def test_shape_distortion_crosses_the_concern_threshold(self):
        score, details = evolution.score_change(
            check(disc(30, squash=0.55)), check(disc(30))
        )
        assert details["shape_iou"] < 0.8
        assert score >= 5.0

    def test_color_shift_is_detected(self):
        score, details = evolution.score_change(
            check(disc(30), lab=(90.0, 170.0, 150.0)), check(disc(30))
        )
        assert details["color_distance"] > 12.0
        assert score >= 5.0

    def test_risk_delta_is_always_reported(self):
        _, details = evolution.score_change(
            check(disc(30), risk_score=62.0), check(disc(30), risk_score=48.0)
        )
        assert details["risk_score_delta"] == pytest.approx(14.0)


class TestDegenerateInputs:
    def test_empty_current_mask_yields_no_score(self):
        score, details = evolution.score_change(
            check(np.zeros((200, 200), np.uint8), lab=None), check(disc(30))
        )
        assert score is None
        assert "reason" in details

    def test_missing_color_still_scores_on_shape(self):
        score, details = evolution.score_change(
            check(disc(30, squash=0.55), lab=None), check(disc(30), lab=None)
        )
        assert details["signals"] == ["shape"]
        assert score is not None


class TestHelpers:
    def test_lesion_area_counts_only_mask_pixels(self):
        assert evolution.lesion_area_px(disc(10)) > 0
        assert evolution.lesion_area_px(np.zeros((50, 50), np.uint8)) == 0
        assert evolution.lesion_area_px(None) == 0

    def test_lab_mean_is_none_for_an_empty_mask(self):
        image = np.full((50, 50, 3), (120, 110, 170), np.uint8)
        assert evolution.lesion_lab_mean(image, np.zeros((50, 50), np.uint8)) is None
        assert evolution.lesion_lab_mean(image, disc(10, shape=(50, 50))) is not None
