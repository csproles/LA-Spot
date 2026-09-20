import cv2
import numpy as np


def segment_lesion(img):
    # get image size
    h, w = img.shape[:2]

    # make a mask for the border
    border_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(border_mask, (int(w*0.05), int(h*0.05)), (int(w*0.95), int(h*0.95)), 255, int(min(h, w) * 0.12))

    # convert to lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_pixels = lab[border_mask > 0]
    skin_color = np.median(border_pixels, axis=0)

    # calculate distance
    dist = np.sqrt(np.sum((lab - skin_color) ** 2, axis=2))
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # keep largest blob
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest) * 255

    # check if mask is too small
    lesion_area = np.sum(mask > 0)
    total_area = h * w
    if lesion_area / total_area < 0.005:
        print("LAB didn't work, using grayscale")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.uint8(labels == largest) * 255

    masked = cv2.bitwise_and(img, img, mask=mask)
    return mask, masked
#