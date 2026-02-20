from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core.detector import DetectionParams, Detector
from ..core.loader import FrameLoader
from .frame_viewer import FrameViewer
from .param_panel import ParamPanel
from .transport import TransportControls


class ProcessingWorker(QThread):
    """Runs detection on all frames in a background thread."""

    progress = Signal(int, int)  # current, total
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, detector: Detector, params: DetectionParams):
        super().__init__()
        self._detector = detector
        self._params = params
        self._cancelled = False

    def run(self) -> None:
        try:
            self._detector.process_all(
                self._params,
                progress_callback=lambda cur, total: self.progress.emit(cur, total),
                cancel_check=lambda: self._cancelled,
            )
            if not self._cancelled:
                self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self) -> None:
        self._cancelled = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artifaktor — Artifact Detector")
        self.setMinimumSize(900, 600)
        self.resize(1200, 750)

        self._loader = FrameLoader()
        self._detector: Detector | None = None
        self._worker: ProcessingWorker | None = None

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # --- Top toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._btn_load_video = QPushButton("Load Video")
        self._btn_load_video.clicked.connect(self._load_video)
        toolbar.addWidget(self._btn_load_video)

        self._btn_load_folder = QPushButton("Load Folder")
        self._btn_load_folder.clicked.connect(self._load_folder)
        toolbar.addWidget(self._btn_load_folder)

        self._btn_process = QPushButton("Process All")
        self._btn_process.setEnabled(False)
        self._btn_process.clicked.connect(self._start_processing)
        self._btn_process.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self._btn_process)

        self._btn_process_frame = QPushButton("Process Frame")
        self._btn_process_frame.setEnabled(False)
        self._btn_process_frame.clicked.connect(self._process_current_frame)
        toolbar.addWidget(self._btn_process_frame)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(22)
        toolbar.addWidget(self._progress, 1)

        toolbar.addStretch()
        root.addLayout(toolbar)

        # --- Main content: viewer + param panel ---
        content = QHBoxLayout()
        content.setSpacing(6)

        # Left: viewer + transport
        left = QVBoxLayout()
        left.setSpacing(4)

        self._viewer = FrameViewer()
        left.addWidget(self._viewer, 1)

        self._transport = TransportControls()
        self._transport.frame_changed.connect(self._on_frame_changed)
        left.addWidget(self._transport)

        content.addLayout(left, 1)

        # Right: param panel
        self._param_panel = ParamPanel()
        self._param_panel.params_changed.connect(self._on_display_changed)
        content.addWidget(self._param_panel)

        root.addLayout(content, 1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — load a video or image folder")

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._transport.step_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._transport.step_next)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._transport.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, lambda: self._transport.set_current_frame(0))

    # --- Loading ---

    def _load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return
        try:
            self._loader.load_video(path)
            self._on_loaded()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video:\n{e}")

    def _load_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if not path:
            return
        try:
            self._loader.load_folder(path)
            self._on_loaded()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load folder:\n{e}")

    def _on_loaded(self) -> None:
        self._detector = Detector(self._loader)
        self._transport.set_frame_count(self._loader.frame_count, self._loader.fps)
        self._btn_process.setEnabled(True)
        self._btn_process_frame.setEnabled(True)
        self._status.showMessage(
            f"Loaded: {self._loader.source_name} — "
            f"{self._loader.frame_count} frames, "
            f"{self._loader.frame_size[0]}x{self._loader.frame_size[1]}"
        )
        self._param_panel.set_region_count(0)
        # Show first frame
        self._show_frame(0)

    # --- Processing ---

    def _start_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()

        if not self._detector:
            return

        params = self._param_panel.get_params()
        self._worker = ProcessingWorker(self._detector, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_processing_done)
        self._worker.error.connect(self._on_processing_error)

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_process.setEnabled(False)
        self._btn_process_frame.setEnabled(False)
        self._btn_load_video.setEnabled(False)
        self._btn_load_folder.setEnabled(False)
        self._status.showMessage("Processing...")

        self._worker.start()

    def _on_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._progress.setFormat(f"Processing frame {current}/{total}")

    def _on_processing_done(self) -> None:
        self._progress.setVisible(False)
        self._btn_process.setEnabled(True)
        self._btn_process_frame.setEnabled(True)
        self._btn_load_video.setEnabled(True)
        self._btn_load_folder.setEnabled(True)
        self._status.showMessage("Processing complete")
        # Refresh current frame with results
        self._show_frame(self._transport.current_frame())

    def _process_current_frame(self) -> None:
        if not self._detector:
            return
        idx = self._transport.current_frame()
        params = self._param_panel.get_params()
        self._status.showMessage(f"Processing frame {idx + 1}...")
        self._btn_process_frame.setEnabled(False)
        try:
            self._detector.process_frame(idx, params)
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))
        self._btn_process_frame.setEnabled(True)
        self._show_frame(idx)
        self._status.showMessage(f"Frame {idx + 1} processed")

    def _on_processing_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._btn_process.setEnabled(True)
        self._btn_process_frame.setEnabled(True)
        self._btn_load_video.setEnabled(True)
        self._btn_load_folder.setEnabled(True)
        QMessageBox.critical(self, "Processing Error", msg)
        self._status.showMessage("Processing failed")

    # --- Frame display ---

    def _on_frame_changed(self, idx: int) -> None:
        self._show_frame(idx)

    def _on_display_changed(self) -> None:
        self._show_frame(self._transport.current_frame())

    def _show_frame(self, idx: int) -> None:
        frame = self._loader.get_frame(idx)
        if frame is None:
            return

        heatmap = None
        boxes = []

        if self._detector:
            result = self._detector.get_result(idx)
            if result:
                heatmap = result.heatmap
                boxes = result.boxes

        self._viewer.set_frame(frame, heatmap=heatmap, boxes=boxes)
        self._viewer.set_display_options(
            show_heatmap=self._param_panel.show_heatmap.isChecked(),
            show_boxes=self._param_panel.show_boxes.isChecked(),
        )
        self._param_panel.set_region_count(len(boxes))
