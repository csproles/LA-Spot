import cv2


def remove_salt_pepper_noise(img, kernel_size=3):
    denoised = cv2.medianBlur(img, kernel_size)
    return denoised
