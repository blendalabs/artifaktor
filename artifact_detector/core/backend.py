"""Local YOLO detector backend for Artifaktor.

Trains a single-class detector ("artifact") from annotations.json and
predicts bounding boxes on frames.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

LOGGER = logging.getLogger(__name__)


@dataclass
class _DatasetPaths:
    root: Path
    train_images: Path
    val_images: Path
    train_labels: Path
    val_labels: Path
    yaml_path: Path


class BackendClient:
    """Local quality backend using Ultralytics YOLO."""

    def __init__(self, base_url: str = "") -> None:
        self._root = Path(os.getenv("ARTIFAKTOR_MODEL_DIR", "models/quality")).resolve()
        self._run_name = os.getenv("ARTIFAKTOR_RUN_NAME", "artifact")
        self._weights = os.getenv("ARTIFAKTOR_PRETRAINED", "yolov8n.pt")
        self._predict_conf = float(os.getenv("ARTIFAKTOR_PREDICT_CONF", "0.25"))
        self._predict_iou = float(os.getenv("ARTIFAKTOR_PREDICT_IOU", "0.45"))

    # -----------------------
    # public API used by UI
    # -----------------------

    def health_check(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False

    def predict(self, image_path: str, _frame_id: str) -> list[dict[str, int]]:
        model_path = self._latest_model_path()
        if model_path is None:
            raise RuntimeError("No trained model found. Click Train first.")

        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.predict(
            source=image_path,
            conf=self._predict_conf,
            iou=self._predict_iou,
            verbose=False,
        )
        if not results:
            return []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        out: list[dict[str, int]] = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            out.append({
                "x": int(round(max(0, x1))),
                "y": int(round(max(0, y1))),
                "w": int(round(max(1, x2 - x1))),
                "h": int(round(max(1, y2 - y1))),
            })
        return out

    def train(self, annotations: dict[str, list[dict]], image_dir: str) -> dict[str, Any]:
        if not annotations:
            return {"status": "no-data", "message": "No annotations found."}

        from ultralytics import YOLO

        image_dir_path = Path(image_dir).resolve()
        dataset = self._build_yolo_dataset(annotations, image_dir_path)

        device = self._resolve_device()
        epochs = int(os.getenv("ARTIFAKTOR_TRAIN_EPOCHS", "100"))
        imgsz = int(os.getenv("ARTIFAKTOR_TRAIN_IMGSZ", "640"))
        batch = int(os.getenv("ARTIFAKTOR_TRAIN_BATCH", "-1"))  # -1 = auto

        self._root.mkdir(parents=True, exist_ok=True)

        # Delete old run so we get a clean training
        old_run = self._root / self._run_name
        if old_run.exists():
            shutil.rmtree(old_run)

        model = YOLO(self._weights)
        t0 = time.time()
        train_res = model.train(
            data=str(dataset.yaml_path),
            project=str(self._root),
            name=self._run_name,
            exist_ok=True,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=20,
            verbose=True,
        )
        elapsed = round(time.time() - t0, 2)

        best = Path(train_res.save_dir) / "weights" / "best.pt"
        if not best.exists():
            return {
                "status": "error",
                "message": "Training finished but best.pt not found.",
                "save_dir": str(train_res.save_dir),
            }

        # Quick validation: predict on a few training images to sanity check
        val_model = YOLO(str(best))
        val_imgs = list(dataset.val_images.glob("*.jpg"))[:5]
        val_hits = 0
        for vi in val_imgs:
            vr = val_model.predict(source=str(vi), conf=0.25, verbose=False)
            if vr and vr[0].boxes is not None and len(vr[0].boxes) > 0:
                val_hits += 1

        return {
            "status": "ok",
            "message": f"Trained {epochs} epochs in {elapsed}s on {device}. Val sanity: {val_hits}/{len(val_imgs)} images had detections.",
            "elapsed_s": elapsed,
            "sample_count": len(annotations),
            "nonempty_count": sum(1 for v in annotations.values() if v),
            "save_dir": str(train_res.save_dir),
            "best_model": str(best),
            "epochs": epochs,
            "device": device,
        }

    # -----------------------
    # internals
    # -----------------------

    def _resolve_device(self) -> str:
        preferred = os.getenv("ARTIFAKTOR_TRAIN_DEVICE", "auto")
        if preferred != "auto":
            return preferred
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _latest_model_path(self) -> Path | None:
        candidate = self._root / self._run_name / "weights" / "best.pt"
        if candidate.exists():
            return candidate
        pts = sorted(self._root.glob("**/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return pts[0] if pts else None

    def _build_yolo_dataset(self, annotations: dict[str, list[dict]], image_dir: Path) -> _DatasetPaths:
        tmp_root = image_dir / ".artifaktor_yolo"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

        train_images = tmp_root / "images" / "train"
        val_images = tmp_root / "images" / "val"
        train_labels = tmp_root / "labels" / "train"
        val_labels = tmp_root / "labels" / "val"

        for p in [train_images, val_images, train_labels, val_labels]:
            p.mkdir(parents=True, exist_ok=True)

        # Only include frames that have at least one box
        labeled_items = [(fn, boxes) for fn, boxes in annotations.items() if boxes]
        if not labeled_items:
            raise ValueError("No frames with annotations found.")

        random.Random(42).shuffle(labeled_items)
        val_count = max(1, int(len(labeled_items) * 0.15))
        val_set = set(fn for fn, _ in labeled_items[:val_count])

        skipped = 0
        for filename, boxes in labeled_items:
            src = image_dir / filename
            if not src.exists():
                skipped += 1
                continue

            img = cv2.imread(str(src))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]

            is_val = filename in val_set
            img_dst = (val_images if is_val else train_images) / filename
            lbl_dst = (val_labels if is_val else train_labels) / f"{Path(filename).stem}.txt"
            shutil.copy2(src, img_dst)

            lines: list[str] = []
            for b in boxes:
                bx = float(b.get("x", 0))
                by = float(b.get("y", 0))
                bw = float(b.get("w", 0))
                bh = float(b.get("h", 0))
                if bw <= 0 or bh <= 0:
                    continue
                xc = (bx + bw / 2.0) / w
                yc = (by + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                # Clamp to [0, 1]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))
                lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            with open(lbl_dst, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))

        LOGGER.info(
            "Dataset built: %d train, %d val, %d skipped",
            len(labeled_items) - val_count - skipped,
            val_count,
            skipped,
        )

        yaml_path = tmp_root / "dataset.yaml"
        yaml_path.write_text(
            "\n".join([
                f"path: {tmp_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: artifact",
            ]),
            encoding="utf-8",
        )

        return _DatasetPaths(
            root=tmp_root,
            train_images=train_images,
            val_images=val_images,
            train_labels=train_labels,
            val_labels=val_labels,
            yaml_path=yaml_path,
        )


# ---------------------------------------------------------------------------
# QThread workers (keep UI responsive)
# ---------------------------------------------------------------------------

try:
    from PySide6.QtCore import QThread, Signal as _Signal

    class PredictWorker(QThread):
        finished = _Signal(list)
        error = _Signal(str)

        def __init__(self, client: BackendClient, image_path: str, frame_id: str, parent=None) -> None:
            super().__init__(parent)
            self._client = client
            self._image_path = image_path
            self._frame_id = frame_id

        def run(self) -> None:
            try:
                self.finished.emit(self._client.predict(self._image_path, self._frame_id))
            except Exception as exc:
                LOGGER.exception("Predict failed")
                self.error.emit(str(exc))

    class TrainWorker(QThread):
        finished = _Signal(dict)
        error = _Signal(str)

        def __init__(self, client: BackendClient, annotations: dict, image_dir: str, parent=None) -> None:
            super().__init__(parent)
            self._client = client
            self._annotations = annotations
            self._image_dir = image_dir

        def run(self) -> None:
            try:
                self.finished.emit(self._client.train(self._annotations, self._image_dir))
            except Exception as exc:
                LOGGER.exception("Train failed")
                self.error.emit(str(exc))

except ImportError:  # pragma: no cover
    PredictWorker = None  # type: ignore[assignment,misc]
    TrainWorker = None  # type: ignore[assignment,misc]
