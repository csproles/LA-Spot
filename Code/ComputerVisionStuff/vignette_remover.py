import cv2
import numpy as np


def remove_vignette(img, shrink=0.95):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("no circle found")
        return img, None

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)

    circle_area = cv2.contourArea(largest)
    img_area = img.shape[0] * img.shape[1]

    if circle_area / img_area < 0.3:
        print("circle too small")
        return img, None

    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (int(cx), int(cy)), int(radius * shrink), 255, -1)

    skin_color = cv2.mean(img, mask=circle_mask)[:3]
    result     = img.copy()
    result[circle_mask == 0] = [int(c) for c in skin_color]

    print(f"Vignette removed center=({cx:.0f},{cy:.0f}) radius={radius:.0f}px")
    return result, (int(cx), int(cy), int(radius))