import cv2
import numpy as np


def remove_salt_pepper_noise(img, kernel_size=3):
    denoised = cv2.medianBlur(img, kernel_size)
    return denoised
    return denoised
#