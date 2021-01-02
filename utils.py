import cv2
import numpy as np

def improve_contrast_image_using_clahe(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def image_size_limit(image, size):
    h, w, _ = image.shape
    if w < h:
        return image_resize(image, height=size)
    else:
        return image_resize(image, width=size)

def generate_weights(size):
    ret = []
    remaining = 1
    for i in range(0, size-1):
        ret.append(remaining/2)
        remaining /= 2
    ret.append(remaining)
    return ret

if __name__ == '__main__':
    print(generate_weights(10))
    print(sum(generate_weights(10)))