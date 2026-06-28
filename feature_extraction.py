import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_features(image_path, fruit_type="banana"):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image   = cv2.resize(image, (224, 224))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    ranges = {
        "banana":     ([15, 30, 30],  [40, 255, 255]),
        "mango":      ([5,  30, 30],  [35, 255, 255]),
        "apple":      ([0,  30, 30],  [179, 255, 255]),
        "orange":     ([5,  50, 50],  [25, 255, 255]),
        "strawberry": ([0,  50, 50],  [15, 255, 255]),  # red
    }
    lo, hi = ranges.get(fruit_type, ([0, 30, 30], [179, 255, 255]))
    mask   = cv2.inRange(hsv, np.array(lo), np.array(hi))

    # Colour histogram (512)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask,
                        [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # LBP texture (59)
    try:
        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp      = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    except:
        lbp_hist = np.zeros(59)

    # Shape (4)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c           = max(contours, key=cv2.contourArea)
        area        = cv2.contourArea(c)
        perimeter   = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        x, y, w, h  = cv2.boundingRect(c)
        extent      = area / (w * h + 1e-6)
        shape_feats = np.array([area, perimeter, circularity, extent])
    else:
        shape_feats = np.zeros(4)

    return np.concatenate([hist, lbp_hist, shape_feats])
