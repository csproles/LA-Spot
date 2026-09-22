import cv2
import numpy as np


def detect_edges(
    img,
    low_threshold=50,
    high_threshold=150,
):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges
#