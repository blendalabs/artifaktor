import cv2
import numpy as np


def cluster_heatmap(
    heatmap: np.ndarray,
    global_threshold: float = 0.35,
    min_region_size: int = 500,
    patch_size: int = 128,
) -> list[tuple[int, int, int, int]]:
    """Convert a combined heatmap into bounding box regions.

    Steps:
    1. Gaussian blur to smooth the heatmap
    2. Threshold to binary mask
    3. Find connected components
    4. Filter by minimum region size
    5. Compute bounding boxes, padded to minimum patch_size

    Returns list of (x, y, w, h) bounding boxes.
    """
    h, w = heatmap.shape[:2]

    # Smooth
    smoothed = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Threshold to binary
    binary = (smoothed > global_threshold).astype(np.uint8) * 255

    # Morphological close to merge nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, n_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_region_size:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Pad to minimum patch_size
        if bw < patch_size:
            cx = x + bw // 2
            x = max(0, cx - patch_size // 2)
            bw = min(patch_size, w - x)
        if bh < patch_size:
            cy = y + bh // 2
            y = max(0, cy - patch_size // 2)
            bh = min(patch_size, h - y)

        boxes.append((x, y, bw, bh))

    return boxes
