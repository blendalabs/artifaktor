from __future__ import annotations

import glob
import os
from collections import OrderedDict

import cv2
import numpy as np


class FrameLoader:
    """Loads frames from a video file or image sequence folder.

    Uses lazy loading with an LRU cache to avoid holding all frames in memory.
    """

    def __init__(self, cache_size: int = 150):
        self._source_type: str | None = None  # "video" or "folder"
        self._video_path: str | None = None
        self._image_paths: list[str] = []
        self._frame_count: int = 0
        self._fps: float = 24.0
        self._frame_size: tuple[int, int] = (0, 0)  # (width, height)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def load_video(self, path: str) -> None:
        """Load an mp4 video file."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        self._source_type = "video"
        self._video_path = path
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        self._frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        self._cache.clear()

    def load_folder(self, path: str) -> None:
        """Load an image sequence from a folder of PNGs/JPGs."""
        exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        files: list[str] = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
        files.sort()

        if not files:
            raise ValueError(f"No PNG/JPG images found in: {path}")

        self._source_type = "folder"
        self._image_paths = files
        self._frame_count = len(files)
        self._fps = 24.0

        # Read first image to get dimensions
        first = cv2.imread(files[0])
        if first is not None:
            self._frame_size = (first.shape[1], first.shape[0])

        self._cache.clear()

    def get_frame(self, idx: int) -> np.ndarray | None:
        """Get frame by index. Returns BGR numpy array or None."""
        if idx < 0 or idx >= self._frame_count:
            return None

        # Check cache
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        # Load frame
        frame = self._load_frame(idx)
        if frame is None:
            return None

        # Add to cache
        self._cache[idx] = frame
        self._cache.move_to_end(idx)

        # Evict oldest if over capacity
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return frame

    def _load_frame(self, idx: int) -> np.ndarray | None:
        if self._source_type == "video":
            cap = cv2.VideoCapture(self._video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        elif self._source_type == "folder":
            if 0 <= idx < len(self._image_paths):
                return cv2.imread(self._image_paths[idx])
        return None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    @property
    def is_loaded(self) -> bool:
        return self._source_type is not None

    @property
    def source_name(self) -> str:
        if self._source_type == "video":
            return os.path.basename(self._video_path or "")
        elif self._source_type == "folder":
            if self._image_paths:
                return os.path.basename(os.path.dirname(self._image_paths[0]))
        return ""
