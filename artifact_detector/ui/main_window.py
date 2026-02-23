"""Main annotation window.

Layout:

    ┌──────────────────────────────────────────────────────────────┐
    │ [Load Folder]  [Train]  [Predict]        backend status…     │ ← toolbar
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │                  FrameViewer (canvas)                        │ ← main
    │                                                              │
    ├──────────────────────────────────────────────────────────────┤
    │  |< < Play > >|  F: 1/500  ══════════════ slider ══════════ │ ← transport
    ├──────────────────────────────────────────────────────────────┤
    │  ←/→ navigate  |  Drag: add box  |  Click box: delete  |... │ ← hints
    └──────────────────────────────────────────────────────────────┘

Keyboard shortcuts
------------------
* ``Left`` / ``Right``     — previous / next frame
* ``Space``                — play / pause
* ``Ctrl+Backspace``       — clear all boxes on current frame
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core.annotations import AnnotationStore
from ..core.backend import BackendClient, PredictWorker, TrainWorker
from ..core.loader import FrameLoader
from .frame_viewer import FrameViewer
from .transport import TransportControls


class MainWindow(QMainWindow):
    """Lightweight frame annotation window."""

    backend_status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Artifaktor — Frame Annotator")
        self.setMinimumSize(900, 600)
        self.resize(1280, 780)

        self._loader = FrameLoader()
        self._annotations = AnnotationStore()
        self._backend = BackendClient(
            os.environ.get("ML_BACKEND_URL", "http://localhost:9090")
        )
        self._predict_worker: PredictWorker | None = None
        self._train_worker: TrainWorker | None = None

        self._setup_ui()
        self.backend_status_changed.connect(self._backend_label.setText)
        self._setup_shortcuts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Top toolbar ────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._btn_load = QPushButton("📂  Load Folder")
        self._btn_load.clicked.connect(self._load_folder)
        toolbar.addWidget(self._btn_load)

        toolbar.addSpacing(12)

        self._btn_train = QPushButton("🧠  Train")
        self._btn_train.setEnabled(False)
        self._btn_train.setToolTip("Send current annotations to the ML backend for training")
        self._btn_train.clicked.connect(self._start_training)
        toolbar.addWidget(self._btn_train)

        self._btn_predict = QPushButton("🔮  Predict")
        self._btn_predict.setEnabled(False)
        self._btn_predict.setToolTip("Ask the ML backend to predict boxes on the current frame")
        self._btn_predict.clicked.connect(self._start_predict)
        toolbar.addWidget(self._btn_predict)

        toolbar.addStretch()

        self._backend_label = QLabel("Backend: —")
        self._backend_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._backend_label)

        root.addLayout(toolbar)

        # ── FrameViewer ────────────────────────────────────────────
        self._viewer = FrameViewer()
        self._viewer.rectangle_created.connect(self._on_rect_created)
        self._viewer.rectangle_deleted.connect(self._on_rect_deleted)
        root.addWidget(self._viewer, 1)

        # ── Transport ──────────────────────────────────────────────
        self._transport = TransportControls()
        self._transport.frame_changed.connect(self._on_frame_changed)
        root.addWidget(self._transport)

        # ── Shortcut hints ─────────────────────────────────────────
        hints = QLabel(
            "←/→ or Q/W  navigate  │  Drag: add box  │  Click box: delete  │"
            "  Ctrl+Backspace: clear frame"
        )
        hints.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hints.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        root.addWidget(hints)

        # ── Status bar ─────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — load an image folder to begin")

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._transport.step_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._transport.step_next)
        QShortcut(QKeySequence(Qt.Key.Key_Q), self, self._transport.step_prev)
        QShortcut(QKeySequence(Qt.Key.Key_W), self, self._transport.step_next)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._transport.toggle_play)
        QShortcut(
            QKeySequence(Qt.Key.Key_Home),
            self,
            lambda: self._transport.set_current_frame(0),
        )
        QShortcut(
            QKeySequence("Ctrl+Backspace"),
            self,
            self._on_clear_frame,
        )

    # ------------------------------------------------------------------
    # Folder loading
    # ------------------------------------------------------------------

    def _load_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if not path:
            return
        try:
            self._loader.load_folder(path)
            self._on_loaded()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load folder:\n{exc}")

    def _on_loaded(self) -> None:
        # Load (or start fresh) annotation store
        src = self._loader.source_dir
        if src:
            self._annotations.load(src)

        self._transport.set_frame_count(self._loader.frame_count, self._loader.fps)
        self._btn_train.setEnabled(True)
        self._btn_predict.setEnabled(True)

        self._status.showMessage(
            f"Loaded: {self._loader.source_name}  —  "
            f"{self._loader.frame_count} frames  "
            f"{self._loader.frame_size[0]}×{self._loader.frame_size[1]}"
        )

        # Show first frame
        self._show_frame(0)

        # Async backend health check
        self._check_backend_async()

    def _check_backend_async(self) -> None:
        """Update backend status without blocking the UI thread."""
        import threading

        def _ping() -> None:
            ok = self._backend.health_check()
            self.backend_status_changed.emit(
                "Backend: ✅ online" if ok else "Backend: ⚠️ offline"
            )

        threading.Thread(target=_ping, daemon=True).start()

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _on_frame_changed(self, idx: int) -> None:
        self._show_frame(idx)

    def _show_frame(self, idx: int) -> None:
        frame = self._loader.get_frame(idx)
        if frame is None:
            return

        frame_id = self._loader.filename_at(idx) or ""
        boxes = self._annotations.get_boxes_as_tuples(frame_id)
        self._viewer.set_frame(frame, boxes=boxes)
        self._update_status(idx, frame_id)

    def _update_status(self, idx: int, frame_id: str) -> None:
        total = self._loader.frame_count
        box_count = len(self._annotations.get_boxes(frame_id))
        self._status.showMessage(
            f"Frame {idx + 1}/{total}  │  {frame_id}  │  {box_count} box{'es' if box_count != 1 else ''}"
        )

    # ------------------------------------------------------------------
    # Rectangle interactions
    # ------------------------------------------------------------------

    def _current_frame_id(self) -> str:
        idx = self._transport.current_frame()
        return self._loader.filename_at(idx) or ""

    def _on_rect_created(self, x: int, y: int, w: int, h: int) -> None:
        frame_id = self._current_frame_id()
        if not frame_id:
            return
        self._annotations.add_box(frame_id, x, y, w, h)
        self._refresh_boxes()

    def _on_rect_deleted(self, idx: int) -> None:
        frame_id = self._current_frame_id()
        if not frame_id:
            return
        self._annotations.delete_box(frame_id, idx)
        self._refresh_boxes()

    def _on_clear_frame(self) -> None:
        frame_id = self._current_frame_id()
        if not frame_id:
            return
        self._annotations.clear_frame(frame_id)
        self._refresh_boxes()

    def _refresh_boxes(self) -> None:
        """Redraw boxes and update status without reloading the frame."""
        frame_id = self._current_frame_id()
        idx = self._transport.current_frame()
        boxes = self._annotations.get_boxes_as_tuples(frame_id)
        self._viewer.set_boxes(boxes)
        self._update_status(idx, frame_id)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        if TrainWorker is None:
            self._status.showMessage("Training unavailable: Qt worker support not loaded")
            return

        if self._train_worker and self._train_worker.isRunning():
            return

        src = self._loader.source_dir
        if not src:
            return

        self._btn_train.setEnabled(False)
        self._backend_label.setText("Backend: 🔄 training…")
        self._status.showMessage("Sending training request to backend…")

        self._train_worker = TrainWorker(
            self._backend,
            dict(self._annotations.all_data),
            src,
        )
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.start()

    def _on_train_done(self, result: dict) -> None:
        self._btn_train.setEnabled(True)
        self._backend_label.setText("Backend: ✅ online")

        status = str(result.get("status", "ok"))
        sample_count = result.get("sample_count")
        if status == "ok":
            if sample_count is not None:
                self._status.showMessage(f"Training done: {sample_count} samples")
            else:
                self._status.showMessage("Training done")
            return

        msg = result.get("message") or result.get("reason") or status
        self._status.showMessage(f"Training finished: {msg}")

    def _on_train_error(self, msg: str) -> None:
        self._btn_train.setEnabled(True)
        self._backend_label.setText("Backend: ⚠️ offline")
        self._status.showMessage(f"Training failed: {msg}")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _start_predict(self) -> None:
        if PredictWorker is None:
            self._status.showMessage("Prediction unavailable: Qt worker support not loaded")
            return

        if self._predict_worker and self._predict_worker.isRunning():
            return

        idx = self._transport.current_frame()
        image_path = self._loader.image_path_at(idx)
        frame_id = self._loader.filename_at(idx)
        if not image_path or not frame_id:
            return

        self._btn_predict.setEnabled(False)
        self._backend_label.setText("Backend: 🔄 predicting…")
        self._status.showMessage("Requesting predictions from backend…")

        self._predict_worker = PredictWorker(self._backend, image_path, frame_id)
        self._predict_worker.finished.connect(lambda boxes: self._on_predict_done(boxes, frame_id))
        self._predict_worker.error.connect(self._on_predict_error)
        self._predict_worker.start()

    def _on_predict_done(self, boxes: list[dict], frame_id: str) -> None:
        self._btn_predict.setEnabled(True)
        self._backend_label.setText("Backend: ✅ online")

        # Replace current frame's boxes with predictions
        self._annotations.set_boxes(frame_id, boxes)
        self._refresh_boxes()
        self._status.showMessage(
            f"Predicted {len(boxes)} box{'es' if len(boxes) != 1 else ''} on {frame_id}"
        )

    def _on_predict_error(self, msg: str) -> None:
        self._btn_predict.setEnabled(True)
        self._backend_label.setText("Backend: ⚠️ offline")
        self._status.showMessage(f"Prediction failed: {msg}")

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Final safety save
        if self._annotations.is_loaded:
            self._annotations.save()

        # Stop any running workers cleanly
        for worker in (self._train_worker, self._predict_worker):
            if worker and worker.isRunning():
                worker.wait(2000)

        super().closeEvent(event)
