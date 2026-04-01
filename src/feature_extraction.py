"""
Feature Extraction Pipeline - PlantVillage Dataset
====================================================
Techniques : Edge (Canny) + GLCM + LBP + Color Statistics
Input      : img (H x W x 3, BGR/RGB uint8) + clean_mask (H x W, binary uint8)
Output     : feature_vector (1D np.ndarray)

Feature breakdown
-----------------
Canny Edge  :  4 features
GLCM        : 24 features   (4 props × 4 angles × 1 distance)
LBP         : 26 features   (26-bin normalised histogram, P=8, R=1)
Color Stats : 18 features   (mean, std, skewness per channel × 3 channels × 2 spaces)
─────────────────────────────────────────────────────
Total       : 72 features
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew


# ─────────────────────────────────────────────────────────────────────────────
# 1. CANNY EDGE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_canny_features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract edge-density statistics from a Canny edge map.

    Parameters
    ----------
    gray : np.ndarray  – Grayscale image  (H × W, uint8)
    mask : np.ndarray  – Binary mask      (H × W, uint8, 0/255)

    Returns
    -------
    features : np.ndarray shape (4,)
        [edge_density, mean_edge_intensity, std_edge_intensity, edge_pixel_ratio]
    """
    # Adaptive thresholds based on Otsu for robustness across lighting conditions
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low  = max(0.5 * otsu_thresh, 10)
    high = min(1.5 * otsu_thresh, 250)

    edges = cv2.Canny(gray, low, high)

    # Restrict to masked (lesion) region only
    mask_bool   = mask > 0
    roi_pixels  = mask_bool.sum()

    if roi_pixels == 0:
        return np.zeros(4, dtype=np.float32)

    edges_in_roi = edges[mask_bool]

    edge_density         = edges_in_roi.mean() / 255.0          # [0, 1]
    mean_edge_intensity  = edges_in_roi.mean()                   # raw mean
    std_edge_intensity   = edges_in_roi.std()                    # raw std
    edge_pixel_ratio     = (edges_in_roi > 0).sum() / roi_pixels # fraction of edge pixels

    return np.array(
        [edge_density, mean_edge_intensity, std_edge_intensity, edge_pixel_ratio],
        dtype=np.float32,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GLCM FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_glcm_features(
    gray    : np.ndarray,
    mask    : np.ndarray,
    levels  : int = 256,
    distances: list[int] = [1],
    angles  : list[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
) -> np.ndarray:
    """
    Compute GLCM-based texture descriptors inside the masked region.

    Properties extracted: contrast, dissimilarity, homogeneity,
                          energy, correlation, ASM
    Each property is computed for every (distance × angle) pair, then
    averaged across angles → distance × n_props scalar values.

    Parameters
    ----------
    gray      : np.ndarray  – Grayscale image (H × W, uint8)
    mask      : np.ndarray  – Binary mask     (H × W, uint8, 0/255)
    levels    : int         – Grey levels to quantise to (default 256 → rescaled to 64)
    distances : list[int]   – Pixel-pair distances
    angles    : list[float] – Directions in radians

    Returns
    -------
    features : np.ndarray shape (len(distances) × 4,) = (4,) for default config
        props averaged over angles: [contrast, dissimilarity, homogeneity, energy,
                                     correlation, ASM]  × distances
    """
    # Quantise to 64 levels to reduce GLCM noise and computation cost
    n_levels = 64
    gray_q   = (gray / 255.0 * (n_levels - 1)).astype(np.uint8)

    # Crop to bounding box of mask to speed up computation
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros(len(distances) * 4, dtype=np.float32)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    roi      = gray_q[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]

    # Zero out non-mask pixels so they don't contribute to co-occurrences
    roi_masked = roi.copy()
    roi_masked[roi_mask == 0] = 0

    glcm = graycomatrix(
        roi_masked,
        distances=distances,
        angles=angles,
        levels=n_levels,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy"]
    feature_list = []
    for prop in props:
        values = graycoprops(glcm, prop)   # shape: (n_distances, n_angles)
        feature_list.append(values.mean(axis=1))  # average over angles

    # Flatten: (n_props × n_distances,) → 1D
    return np.concatenate(feature_list).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LBP FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_lbp_features(
    gray    : np.ndarray,
    mask    : np.ndarray,
    P       : int = 8,
    R       : float = 1.0,
    method  : str = "uniform",
) -> np.ndarray:
    """
    Compute normalised LBP histogram restricted to the masked region.

    Parameters
    ----------
    gray   : np.ndarray  – Grayscale image (H × W, uint8)
    mask   : np.ndarray  – Binary mask     (H × W, uint8, 0/255)
    P      : int         – Number of circularly symmetric neighbour points
    R      : float       – Radius of circle
    method : str         – LBP method ('uniform' → P+2 bins, 'default' → 2^P bins)

    Returns
    -------
    features : np.ndarray shape (P+2,) = (10,) for P=8 uniform
        Normalised histogram of LBP codes
    """
    lbp = local_binary_pattern(gray, P, R, method=method)

    # Uniform LBP produces P+2 distinct patterns
    n_bins = P + 2 if method == "uniform" else 2 ** P

    # Only histogram pixels inside the mask
    mask_bool  = mask > 0
    lbp_masked = lbp[mask_bool]

    if lbp_masked.size == 0:
        return np.zeros(n_bins, dtype=np.float32)

    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=False)
    hist     = hist.astype(np.float32)

    # Normalise to sum = 1
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


# ─────────────────────────────────────────────────────────────────────────────
# 4. COLOR STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def extract_color_stats(
    img  : np.ndarray,
    mask : np.ndarray,
) -> np.ndarray:
    """
    Compute per-channel colour statistics inside the masked region.

    Colour spaces: RGB + HSV
    Statistics    : mean, std, skewness  →  3 stats × 3 channels × 2 spaces = 18 features

    Parameters
    ----------
    img  : np.ndarray  – BGR image (H × W × 3, uint8) — OpenCV default
    mask : np.ndarray  – Binary mask (H × W, uint8, 0/255)

    Returns
    -------
    features : np.ndarray shape (18,)
    """
    mask_bool = mask > 0

    if mask_bool.sum() == 0:
        return np.zeros(18, dtype=np.float32)

    def channel_stats(channel: np.ndarray) -> list[float]:
        pixels = channel[mask_bool].astype(np.float32)
        return [pixels.mean(), pixels.std(), float(skew(pixels))]

    features = []

    # ── RGB ──────────────────────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for c in range(3):
        features.extend(channel_stats(img_rgb[:, :, c]))

    # ── HSV ──────────────────────────────────────────────────────────────────
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for c in range(3):
        features.extend(channel_stats(img_hsv[:, :, c]))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(img: np.ndarray, clean_mask: np.ndarray) -> np.ndarray:
    """
    Full feature-extraction pipeline.

    Parameters
    ----------
    img        : np.ndarray  – BGR image (H × W × 3, uint8)
    clean_mask : np.ndarray  – Binary mask after morphological cleaning
                               (H × W, uint8).  Non-zero pixels = ROI.

    Returns
    -------
    feature_vector : np.ndarray  – 1D array of dtype float32
        Concatenation of:
          [canny(4) | glcm(4) | lbp(10) | color_stats(18)]
          Total: 36 features
          
          ▸ Change P in LBP → P+2 bins
          ▸ Add distances / angles in GLCM → more features
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    assert img.ndim == 3 and img.shape[2] == 3, \
        f"img must be (H, W, 3); got {img.shape}"
    assert clean_mask.ndim == 2, \
        f"clean_mask must be (H, W); got {clean_mask.shape}"
    assert img.shape[:2] == clean_mask.shape, \
        "img and clean_mask must have the same spatial dimensions"

    # ── Normalise mask to 0/255 uint8 ─────────────────────────────────────────
    mask = (clean_mask > 0).astype(np.uint8) * 255

    # ── Preprocessing shared across extractors ────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Extract individual feature groups ─────────────────────────────────────
    f_canny  = extract_canny_features(gray, mask)        # (4,)
    f_glcm   = extract_glcm_features(gray, mask)         # (4,)
    f_lbp    = extract_lbp_features(gray, mask)          # (10,)
    f_color  = extract_color_stats(img, mask)            # (18,)

    # ── Concatenate into single 1D vector ─────────────────────────────────────
    feature_vector = np.concatenate([f_canny, f_glcm, f_lbp, f_color])

    return feature_vector


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NAMES (useful for DataFrame columns / model interpretation)
# ─────────────────────────────────────────────────────────────────────────────
def get_feature_names(P: int = 8) -> list[str]:
    """Return ordered feature names matching the vector from extract_features()."""
    names = []

    # Canny
    names += ["canny_edge_density", "canny_mean", "canny_std", "canny_pixel_ratio"]

    # GLCM
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy"]:
        names.append(f"glcm_{prop}_d1_avg")

    # LBP
    for i in range(P + 2):
        names.append(f"lbp_bin_{i}")

    # Color stats
    for space in ["rgb", "hsv"]:
        ch_names = (
            ["r", "g", "b"] if space == "rgb" else ["h", "s", "v"]
        )
        for ch in ch_names:
            for stat in ["mean", "std", "skew"]:
                names.append(f"{space}_{ch}_{stat}")

    return names


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def batch_extract(
    image_mask_pairs: list[tuple[np.ndarray, np.ndarray]],
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract features for a list of (image, mask) pairs.

    Parameters
    ----------
    image_mask_pairs : list of (img, mask) tuples
    verbose          : print progress every 100 samples

    Returns
    -------
    X : np.ndarray, shape (N, n_features)
    """
    results = []
    for i, (img, mask) in enumerate(image_mask_pairs):
        try:
            fv = extract_features(img, mask)
            results.append(fv)
        except Exception as e:
            print(f"[WARNING] Sample {i} failed: {e}. Using zero vector.")
            results.append(np.zeros(36, dtype=np.float32))

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} / {len(image_mask_pairs)}")

    return np.stack(results, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic test with a circular mask in the center
    H, W = 256, 256
    dummy_img  = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    dummy_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(dummy_mask, (W // 2, H // 2), 80, 255, -1)

    fv = extract_features(dummy_img, dummy_mask)
    names = get_feature_names()

    print(f"\n{'='*55}")
    print(f"  Feature vector length : {len(fv)}")
    print(f"  dtype                 : {fv.dtype}")
    print(f"  any NaN / Inf         : {np.any(~np.isfinite(fv))}")
    print(f"{'='*55}")
    print(f"\n{'Feature':<30} {'Value':>12}")
    print("-" * 44)
    for name, val in zip(names, fv):
        print(f"  {name:<28} {val:>12.4f}")
    print(f"{'='*55}\n")