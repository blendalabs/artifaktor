import cv2
import numpy as np


def detect(frame: np.ndarray, threshold: float = 0.20, n_clusters: int = 16) -> np.ndarray:
    """Detect blurred/smudged areas in flat-color animation frames.

    In clean flat-color animation, every pixel is either:
    - Perfectly flat (uniform color → no gradient at all)
    - Perfectly sharp (crisp outline → strong first AND second derivative)

    AI artifacts introduce BLUR: areas with visible color transitions (gradient)
    but lacking sharpness (weak second derivative). The ratio of the second
    derivative (Laplacian) to the first derivative (Sobel) measures local
    sharpness. Sharp edges have ratio ≈ 1; blurred zones have ratio → 0.

    Parameters:
        frame: BGR input frame
        threshold: 0-1, higher = less sensitive
        n_clusters: unused, kept for API compatibility

    Returns a heatmap (float32, 0-1).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── First derivative (gradient): where color transitions exist ────
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx ** 2 + gy ** 2)

    # ── Second derivative (Laplacian): where transitions are SHARP ────
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    # ── Local averages (smooth out pixel noise) ───────────────────────
    k = 25
    grad_avg = cv2.GaussianBlur(gradient, (k, k), 0)
    lap_avg = cv2.GaussianBlur(laplacian, (k, k), 0)

    # ── Sharpness ratio: Laplacian / Gradient ─────────────────────────
    # Sharp edge → ratio ≈ 1.0 (both derivatives present)
    # Blurred zone → ratio ≈ 0 (gradient present, Laplacian weak)
    # Flat area → 0/0, handled by gradient threshold below
    eps = 1e-6
    sharpness = np.clip(lap_avg / (grad_avg + eps), 0, 2.0)

    # ── Blur signal ───────────────────────────────────────────────────
    # Gradient must be present (not a flat color region).
    # ~8 intensity units of gradient = a visible color transition.
    grad_present = np.clip(grad_avg / 8.0, 0, 1)

    # Invert sharpness: 1 = perfectly blurry, 0 = perfectly sharp
    blur = grad_present * np.clip(1.0 - sharpness, 0, 1)

    # ── Spatial smoothing + median filter ─────────────────────────────
    blur = cv2.GaussianBlur(blur, (35, 35), 0)

    b_max = blur.max() or 1.0
    b_u8 = (np.clip(blur / b_max, 0, 1) * 255).astype(np.uint8)
    blur_filtered = cv2.medianBlur(b_u8, 31).astype(np.float32) / 255.0

    # ── Threshold ─────────────────────────────────────────────────────
    heatmap = np.clip(
        (blur_filtered - threshold) / max(1.0 - threshold, 0.01), 0, 1
    )

    return heatmap.astype(np.float32)
