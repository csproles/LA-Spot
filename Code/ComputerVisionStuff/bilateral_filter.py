import cv2
import numpy as np


def apply_bilateral_filter(
    img,
    diameter=9,
    sigma_color=75,
    sigma_space=75,
):
    filtered = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    return filtered
#