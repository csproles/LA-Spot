import cv2
import numpy as np
import os


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    print(f"[✔] Loaded image: {image_path}  |  Size: {img.shape[1]}x{img.shape[0]}")
    return img