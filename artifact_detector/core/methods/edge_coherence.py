import cv2
import numpy as np


def detect(frame: np.ndarray, min_length: int = 30, smoothness_threshold: float = 0.6) -> np.ndarray:
    """Detect broken/degraded outlines in a flat-color animation frame.

    Two complementary signals:

    1. **Broken lines** — the primary signal. Where there's a visible gradient
       (Sobel) but NO corresponding Canny edge, the outline is broken or
       blurred. Crucially, the gradient must be **spatially broad** (not a thin
       seam line). Thin features like deck plank lines are eroded away; only
       broad degraded zones (melted body parts, smudges) survive.

    2. **Fragment density** — secondary. Zones with many short/jagged edge
       fragments. Long connected edges (intentional outlines) are excluded.

    Parameters:
        frame: BGR input frame
        min_length: edge fragments shorter than this are suspicious
        smoothness_threshold: 0-1, lower = more sensitive

    Returns a heatmap (float32, 0-1).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray_f = gray.astype(np.float32)

    edges = cv2.Canny(gray, 50, 150)

    # ── Signal 1: Broken lines (broad gradient without sharp edge) ────
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # KEY: Filter by gradient BREADTH. Thin seam lines (3-5px wide gradient)
    # are intentional. Only spatially broad gradient zones (20px+) indicate
    # real artifacts like melted/smudged regions.
    # Erode the gradient mask: features narrower than the kernel vanish.
    grad_binary = (grad_mag > 10).astype(np.uint8)
    erode_k = 9
    grad_broad = cv2.erode(
        grad_binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)),
    )
    # Re-dilate to recover original extent of surviving broad regions
    grad_broad = cv2.dilate(
        grad_broad,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)),
    )
    broad_mask = cv2.GaussianBlur(grad_broad.astype(np.float32), (15, 15), 0)

    # Smooth gradient and Canny for spatial comparison.
    # Use wider kernel (15px) so Canny edges fully "cover" their gradient zone.
    k = 15
    grad_smooth = cv2.GaussianBlur(grad_mag, (k, k), 0)
    canny_smooth = cv2.GaussianBlur(edges.astype(np.float32), (k, k), 0)

    grad_norm = np.clip(grad_smooth / 12.0, 0, 1)
    canny_max = max(canny_smooth.max(), 1.0)
    canny_norm = np.clip(canny_smooth / canny_max, 0, 1)

    explain_factor = 3.0 + (1.0 - smoothness_threshold) * 3.0
    broken = grad_norm * np.clip(1.0 - canny_norm * explain_factor, 0, 1)

    # Apply breadth mask: kill thin-line signals, keep broad degradation
    broken *= broad_mask

    # Spatial smoothing
    broken = cv2.GaussianBlur(broken, (31, 31), 0)

    # Median filter for remaining isolated features
    br_max = broken.max() or 1.0
    br_u8 = (np.clip(broken / br_max, 0, 1) * 255).astype(np.uint8)
    broken_filtered = cv2.medianBlur(br_u8, 25).astype(np.float32) / 255.0

    # ── Signal 2: Fragment density ────────────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)

    smoothness_cutoff = 1.0 - smoothness_threshold
    max_area = min_length * 15

    artifact_weights = np.zeros((h, w), dtype=np.float32)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        segment_mask = labels == i

        if area < min_length:
            artifact_weights[segment_mask] = 1.0
            continue
        if area > max_area:
            continue

        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        diagonal = np.sqrt(bw ** 2 + bh ** 2)
        if diagonal < 1:
            continue

        quality = min(diagonal / area, 1.0)
        if quality < smoothness_cutoff:
            weight = (smoothness_cutoff - quality) / max(smoothness_cutoff, 0.01)
            artifact_weights[segment_mask] = min(weight, 1.0)

    density = cv2.GaussianBlur(artifact_weights, (51, 51), 0)
    d_max = np.percentile(density, 99.5) if density.max() > 0 else 1.0
    density_hm = np.clip(density / max(d_max, 1e-6), 0, 1)
    density_hm[density_hm < 0.1] = 0

    # ── Combine ───────────────────────────────────────────────────────
    heatmap = np.maximum(broken_filtered, density_hm)

    return heatmap.astype(np.float32)
