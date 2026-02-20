import cv2
import numpy as np


def detect(
    get_frame,
    current_idx: int,
    total_frames: int,
    threshold: float = 0.3,
    window: int = 2,
) -> np.ndarray:
    """Detect temporal anomalies between animation frames.

    Artifacts often appear suddenly and don't track with motion. A region that
    changes dramatically between frames without corresponding optical flow is
    suspicious.

    Parameters:
        get_frame: callable(idx) -> BGR numpy array
        current_idx: index of the frame to analyze
        total_frames: total number of frames
        threshold: 0-1, lower = more sensitive to unexplained changes
        window: number of neighboring frames to compare (1-5)

    Returns a heatmap (float32, 0-1) where high values indicate temporal anomalies.
    """
    current = get_frame(current_idx)
    h, w = current.shape[:2]
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    max_anomaly = np.zeros((h, w), dtype=np.float32)

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        neighbor_idx = current_idx + offset
        if neighbor_idx < 0 or neighbor_idx >= total_frames:
            continue

        neighbor = get_frame(neighbor_idx)
        neigh_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)

        # Backward flow: for each pixel in current, where it came from in neighbor
        flow_back = cv2.calcOpticalFlowFarneback(
            curr_gray, neigh_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Warp neighbor to align with current
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = x_coords + flow_back[..., 0]
        map_y = y_coords + flow_back[..., 1]
        warped = cv2.remap(neighbor, map_x, map_y, cv2.INTER_LINEAR)

        # Per-pixel difference between current and warped neighbor
        diff = cv2.absdiff(current, warped)
        diff_score = np.mean(diff.astype(np.float32), axis=2) / 255.0

        # Flow magnitude — high flow areas may have warping residual errors
        # so we reduce anomaly weight there
        flow_fwd = cv2.calcOpticalFlowFarneback(
            neigh_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_mag = np.sqrt(flow_fwd[..., 0] ** 2 + flow_fwd[..., 1] ** 2)
        flow_weight = np.clip(flow_mag / 15.0, 0, 1)

        # Anomaly: high diff in low-motion areas
        adjusted = diff_score * (1.0 - 0.5 * flow_weight)
        max_anomaly = np.maximum(max_anomaly, adjusted)

    # Apply threshold
    heatmap = np.clip(
        (max_anomaly - threshold * 0.3) / max(1.0 - threshold * 0.3, 0.01), 0, 1
    )

    return heatmap.astype(np.float32)
