import cv2
import numpy as np

def segmentation(preprocessed_img, k=3):
    # ubah citra menjadi 2d dan konversi ke float32 untuk k-means
    pixel_vals = preprocessed_img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # tentukan kriteria henti k-means, lalu lakukan clustering-nya
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # kembalikan array 1D hasil label ke dimensi citra 2D
    labels_2d = labels.reshape(preprocessed_img.shape[:2])

    # asumsikan pixel di (0,0) sebagai latar belakang
    bg_label = labels_2d[0, 0]

    # membuat mask 
    binary_mask = np.ones(labels_2d.shape, dtype=np.uint8) * 255
    binary_mask[labels_2d == bg_label] = 0

    # morphological processing 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return clean_mask
