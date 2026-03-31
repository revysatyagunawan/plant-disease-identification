import cv2
import numpy as np

TARGET_SIZE = (256, 256)


def preprocess(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Step 1 — Light Gaussian blur
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0)

    # Step 2 — CLAHE on L channel
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)

    enhanced_lab = cv2.merge([l_enhanced, a_ch, b_ch])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return result