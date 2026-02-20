from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import cluster_heatmap
from .loader import FrameLoader
from .methods import color_uniformity, edge_coherence, temporal_anomaly


@dataclass
class DetectionParams:
    # Color uniformity — detects soft gradients where flat color is expected
    color_uniformity_enabled: bool = True
    color_uniformity_threshold: float = 0.20
    color_clusters: int = 16

    # Edge coherence — detects broken/missing outlines
    edge_coherence_enabled: bool = False
    edge_min_length: int = 30
    edge_smoothness_threshold: float = 0.55

    # Temporal anomaly
    temporal_enabled: bool = True
    temporal_diff_threshold: float = 0.35
    temporal_window: int = 2

    # Clustering — region filtering
    min_region_size: int = 500
    patch_size: int = 128


@dataclass
class FrameResult:
    heatmap: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


class Detector:
    """Orchestrates detection methods and caches results per frame."""

    def __init__(self, loader: FrameLoader):
        self.loader = loader
        self.results: dict[int, FrameResult] = {}

    def clear_results(self) -> None:
        self.results.clear()

    def process_frame(self, idx: int, params: DetectionParams) -> FrameResult | None:
        frame = self.loader.get_frame(idx)
        if frame is None:
            return None

        h, w = frame.shape[:2]
        heatmaps: list[np.ndarray] = []

        # Color uniformity
        if params.color_uniformity_enabled:
            hm = color_uniformity.detect(
                frame,
                threshold=params.color_uniformity_threshold,
                n_clusters=params.color_clusters,
            )
            heatmaps.append(hm)

        # Edge coherence
        if params.edge_coherence_enabled:
            hm = edge_coherence.detect(
                frame,
                min_length=params.edge_min_length,
                smoothness_threshold=params.edge_smoothness_threshold,
            )
            heatmaps.append(hm)

        # Temporal anomaly (only if multiple frames available)
        if params.temporal_enabled and self.loader.frame_count > 1:
            hm = temporal_anomaly.detect(
                get_frame=self.loader.get_frame,
                current_idx=idx,
                total_frames=self.loader.frame_count,
                threshold=params.temporal_diff_threshold,
                window=params.temporal_window,
            )
            heatmaps.append(hm)

        # Combine: per-pixel maximum
        if heatmaps:
            combined = heatmaps[0].copy()
            for hm in heatmaps[1:]:
                combined = np.maximum(combined, hm)
        else:
            combined = np.zeros((h, w), dtype=np.float32)

        # Cluster into bounding boxes
        boxes = cluster_heatmap(
            combined,
            global_threshold=0.3,
            min_region_size=params.min_region_size,
            patch_size=params.patch_size,
        )

        result = FrameResult(heatmap=combined, boxes=boxes)
        self.results[idx] = result
        return result

    def process_all(
        self,
        params: DetectionParams,
        progress_callback=None,
        cancel_check=None,
    ) -> None:
        """Process all frames. Calls progress_callback(current, total) after each frame."""
        self.clear_results()
        total = self.loader.frame_count

        for i in range(total):
            if cancel_check and cancel_check():
                break
            self.process_frame(i, params)
            if progress_callback:
                progress_callback(i + 1, total)

    def get_result(self, idx: int) -> FrameResult | None:
        return self.results.get(idx)
