"""HTTP client + QThread workers for the Label Studio ML backend.

The existing backend (``ml_backend/grounding_dino_backend``) uses the
``label_studio_ml`` SDK wire format:

* ``GET  /health``       — liveness probe
* ``POST /predict``      — ``{"tasks": [{"data": {"image": "<path>"}}]}``
* ``POST /train``        — ``{"annotations": [...]}``

Predictions are returned in Label Studio percentage-coordinate format and
converted here to image-pixel ``{x, y, w, h}`` dicts.
"""
from __future__ import annotations

import logging
import os

import cv2
import requests

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP client  (no Qt dependency — importable without a display)
# ---------------------------------------------------------------------------


class BackendClient:
    """Thin wrapper around the Label Studio ML backend HTTP API."""

    def __init__(self, base_url: str = "http://localhost:9090") -> None:
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the backend responds within 3 s."""
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=3)
            return resp.ok
        except Exception:
            return False

    def predict(self, image_path: str, _frame_id: str) -> list[dict[str, int]]:
        """Return a list of ``{x, y, w, h}`` pixel-coord dicts.

        Sends the image *path* to ``/predict`` and converts the Label Studio
        percentage-coordinate response back to pixel coordinates.
        """
        task = {"data": {"image": image_path}}
        resp = requests.post(
            f"{self._base_url}/predict",
            json={"tasks": [task]},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()
        LOGGER.debug("Predict raw response: %s", raw)

        # Determine image dimensions for coordinate conversion.
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image for dimension lookup: {image_path}")
        img_h, img_w = img.shape[:2]

        return self._parse_predictions(raw, img_w, img_h)

    def train(self, annotations: dict[str, list[dict]], image_dir: str) -> dict:
        """Trigger training and return the backend's status response.

        *annotations* is our ``{filename: [{x,y,w,h}, ...]}`` dict.
        """
        tasks = self._annotations_to_tasks(annotations, image_dir)
        resp = requests.post(
            f"{self._base_url}/train",
            json={"annotations": tasks},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Format conversion helpers (pure Python, no Qt)
    # ------------------------------------------------------------------

    def _parse_predictions(
        self,
        response: dict | list,
        img_w: int,
        img_h: int,
    ) -> list[dict[str, int]]:
        """Convert Label Studio prediction format to pixel ``{x, y, w, h}``."""
        boxes: list[dict[str, int]] = []

        # The ML SDK returns ``{"results": [{"result": [...]}]}`` or
        # ``[{"result": [...]}]`` depending on SDK version.
        if isinstance(response, dict):
            results_list = response.get("results", [])
        elif isinstance(response, list):
            results_list = response
        else:
            return boxes

        for prediction in results_list:
            if not isinstance(prediction, dict):
                continue
            for item in prediction.get("result", []):
                if not isinstance(item, dict):
                    continue

                item_type = str(item.get("type") or "").lower()
                if item_type != "rectanglelabels":
                    # Ignore non-region results (e.g. per-region choices tags).
                    continue

                value = item.get("value", {})
                if not isinstance(value, dict):
                    continue

                try:
                    # Label Studio stores x/y/width/height as percentages (0–100).
                    x_pct = float(value["x"])
                    y_pct = float(value["y"])
                    w_pct = float(value["width"])
                    h_pct = float(value["height"])
                except (KeyError, TypeError, ValueError):
                    continue

                boxes.append(
                    {
                        "x": int(x_pct / 100.0 * img_w),
                        "y": int(y_pct / 100.0 * img_h),
                        "w": int(w_pct / 100.0 * img_w),
                        "h": int(h_pct / 100.0 * img_h),
                    }
                )

        return boxes

    def _annotations_to_tasks(
        self,
        annotations: dict[str, list[dict]],
        image_dir: str,
    ) -> list[dict]:
        """Convert ``{filename: [{x,y,w,h}]}`` to Label Studio task list."""
        tasks: list[dict] = []
        for filename, boxes in annotations.items():
            image_path = os.path.join(image_dir, filename)

            # Read dimensions for percentage conversion.
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            result = []
            for box in boxes:
                result.append(
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": box["x"] / img_w * 100.0,
                            "y": box["y"] / img_h * 100.0,
                            "width": box["w"] / img_w * 100.0,
                            "height": box["h"] / img_h * 100.0,
                            "rectanglelabels": ["body_distortion"],
                        },
                    }
                )

            tasks.append(
                {
                    "data": {"image": image_path},
                    "annotations": [{"result": result}],
                }
            )

        return tasks


# ---------------------------------------------------------------------------
# QThread workers (keep UI responsive)
# Qt is imported lazily here so that BackendClient remains importable
# without a running Qt application (useful in headless test environments).
# ---------------------------------------------------------------------------

try:
    from PySide6.QtCore import QThread, Signal as _Signal  # type: ignore[assignment]

    class PredictWorker(QThread):  # type: ignore[misc]
        """Run ``BackendClient.predict()`` off the UI thread."""

        finished = _Signal(list)   # list of {x, y, w, h} dicts
        error = _Signal(str)

        def __init__(
            self,
            client: BackendClient,
            image_path: str,
            frame_id: str,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self._client = client
            self._image_path = image_path
            self._frame_id = frame_id

        def run(self) -> None:
            try:
                boxes = self._client.predict(self._image_path, self._frame_id)
                self.finished.emit(boxes)
            except Exception as exc:
                LOGGER.exception("Predict failed")
                self.error.emit(str(exc))

    class TrainWorker(QThread):  # type: ignore[misc]
        """Run ``BackendClient.train()`` off the UI thread."""

        finished = _Signal(dict)   # raw response from backend
        error = _Signal(str)

        def __init__(
            self,
            client: BackendClient,
            annotations: dict,
            image_dir: str,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self._client = client
            self._annotations = annotations
            self._image_dir = image_dir

        def run(self) -> None:
            try:
                result = self._client.train(self._annotations, self._image_dir)
                self.finished.emit(result)
            except Exception as exc:
                LOGGER.exception("Train failed")
                self.error.emit(str(exc))

except ImportError:
    # Qt not available (e.g. headless test environment).
    # The worker classes won't be defined; the annotation-only workflow
    # remains fully functional via BackendClient alone.
    PredictWorker = None  # type: ignore[assignment,misc]
    TrainWorker = None    # type: ignore[assignment,misc]
