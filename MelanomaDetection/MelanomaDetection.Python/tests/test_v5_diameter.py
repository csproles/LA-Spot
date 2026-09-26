"""The diameter's details with and without a coin scale (v5_detector.diameter_details)."""

import cv2
import numpy as np
import pytest

pytest.importorskip("ultralytics")  # the detector needs YOLO; it runs in the Docker image

from v5_detector import diameter_details  # noqa: E402


def disc_mask(radius):
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), radius, 255, -1)
    return mask


def test_without_a_coin_there_are_no_millimetres():
    details = diameter_details(disc_mask(50), 100.0, 25.0, None)
    assert "diameter_mm" not in details
    assert details["concern"] is False
    assert "no coin" in details["reason"]
    assert details["lesion_size_px"] == 100.0


def test_a_coin_scale_gives_millimetres():
    details = diameter_details(disc_mask(30), 60.0, 15.0, 0.1)  # ~60 px across at 0.1 mm/px
    assert details["diameter_mm"] == pytest.approx(6.0, abs=0.3)
    assert details["scale_source"] == "coin"
    assert details["concern"] is False
    assert "reason" not in details


def test_the_unchanged_10mm_threshold_flags_a_large_spot():
    details = diameter_details(disc_mask(80), 160.0, 40.0, 0.1)  # ~16 mm
    assert details["diameter_mm"] > 10
    assert details["concern"] is True
