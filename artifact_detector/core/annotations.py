"""Annotation persistence layer with crash-safe atomic writes.

Format of ``annotations.json`` (written next to the image folder):

.. code-block:: json

    {
      "frame_000001.jpg": [
        {"x": 120, "y": 80, "w": 64, "h": 92}
      ],
      "frame_000002.jpg": []
    }

All coordinates are in **image-pixel** space.
"""
from __future__ import annotations

import json
from pathlib import Path


class AnnotationStore:
    """In-memory store backed by an ``annotations.json`` file.

    All mutating operations immediately persist via an atomic rename so that
    a crash at any point cannot corrupt the file.
    """

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, int]]] = {}
        self._path: Path | None = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, folder: str) -> None:
        """Load ``annotations.json`` from *folder*, or start empty."""
        self._path = Path(folder) / "annotations.json"
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except Exception:
                # Corrupt or empty file — start fresh rather than crashing.
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        """Atomic write: write to ``.tmp`` then rename (POSIX-safe)."""
        if self._path is None:
            return
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        # os.rename is atomic on the same filesystem (POSIX).
        tmp.rename(self._path)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_boxes(self, frame_id: str) -> list[dict[str, int]]:
        """Return the list of ``{x, y, w, h}`` dicts for *frame_id*."""
        return list(self._data.get(frame_id, []))

    def get_boxes_as_tuples(self, frame_id: str) -> list[tuple[int, int, int, int]]:
        """Return boxes as ``(x, y, w, h)`` tuples for FrameViewer."""
        return [(b["x"], b["y"], b["w"], b["h"]) for b in self._data.get(frame_id, [])]

    # ------------------------------------------------------------------
    # Mutations (each auto-saves)
    # ------------------------------------------------------------------

    def add_box(self, frame_id: str, x: int, y: int, w: int, h: int) -> None:
        """Append a box and immediately persist."""
        self._data.setdefault(frame_id, []).append({"x": x, "y": y, "w": w, "h": h})
        self.save()

    def delete_box(self, frame_id: str, idx: int) -> None:
        """Delete the box at *idx* and immediately persist."""
        boxes = self._data.get(frame_id, [])
        if 0 <= idx < len(boxes):
            boxes.pop(idx)
            self.save()

    def clear_frame(self, frame_id: str) -> None:
        """Remove all boxes for *frame_id* and immediately persist."""
        if frame_id in self._data:
            self._data[frame_id] = []
            self.save()

    def set_boxes(self, frame_id: str, boxes: list[dict[str, int]]) -> None:
        """Bulk-replace boxes for *frame_id* (used by Predict) and persist."""
        self._data[frame_id] = list(boxes)
        self.save()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_data(self) -> dict[str, list[dict[str, int]]]:
        """Read-only view of all annotation data."""
        return self._data

    @property
    def is_loaded(self) -> bool:
        return self._path is not None

    @property
    def path(self) -> Path | None:
        return self._path
