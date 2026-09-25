import cv2


def remove_hair(
    img,
    kernel_size=17,
    threshold=10,
):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(img, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result
#