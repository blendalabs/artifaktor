from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class TransportControls(QWidget):
    """Playback and frame scrub controls."""

    frame_changed = Signal(int)  # emitted when user requests a frame change

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_count = 0
        self._current_frame = 0
        self._playing = False
        self._fps = 24.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Buttons
        self._btn_start = QPushButton("|<")
        self._btn_start.setFixedWidth(32)
        self._btn_start.clicked.connect(self._go_start)

        self._btn_prev = QPushButton("<")
        self._btn_prev.setFixedWidth(32)
        self._btn_prev.clicked.connect(self._go_prev)

        self._btn_play = QPushButton("Play")
        self._btn_play.setFixedWidth(50)
        self._btn_play.clicked.connect(self._toggle_play)

        self._btn_next = QPushButton(">")
        self._btn_next.setFixedWidth(32)
        self._btn_next.clicked.connect(self._go_next)

        self._btn_end = QPushButton(">|")
        self._btn_end.setFixedWidth(32)
        self._btn_end.clicked.connect(self._go_end)

        layout.addWidget(self._btn_start)
        layout.addWidget(self._btn_prev)
        layout.addWidget(self._btn_play)
        layout.addWidget(self._btn_next)
        layout.addWidget(self._btn_end)

        # Frame counter
        self._frame_label = QLabel("F: 0/0")
        self._frame_label.setFixedWidth(80)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._frame_label)

        # Scrub slider
        self._scrub = QSlider(Qt.Orientation.Horizontal)
        self._scrub.setRange(0, 0)
        self._scrub.valueChanged.connect(self._on_scrub)
        layout.addWidget(self._scrub, 1)

        # Playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_frame_count(self, count: int, fps: float = 24.0) -> None:
        self._frame_count = count
        self._fps = max(fps, 1.0)
        self._scrub.setRange(0, max(count - 1, 0))
        self._current_frame = 0
        self._scrub.setValue(0)
        self._update_label()
        self._stop_playback()

    def set_current_frame(self, idx: int) -> None:
        """Programmatically set the current frame without emitting signal."""
        self._current_frame = idx
        self._scrub.blockSignals(True)
        self._scrub.setValue(idx)
        self._scrub.blockSignals(False)
        self._update_label()

    def current_frame(self) -> int:
        return self._current_frame

    def _update_label(self) -> None:
        total = self._frame_count
        cur = self._current_frame + 1 if total > 0 else 0
        self._frame_label.setText(f"F: {cur}/{total}")

    def _emit_frame(self, idx: int) -> None:
        idx = max(0, min(idx, self._frame_count - 1))
        self._current_frame = idx
        self._update_label()
        self.frame_changed.emit(idx)

    def _on_scrub(self, val: int) -> None:
        self._emit_frame(val)

    def _go_start(self) -> None:
        self._stop_playback()
        self._scrub.setValue(0)

    def _go_end(self) -> None:
        self._stop_playback()
        self._scrub.setValue(max(self._frame_count - 1, 0))

    def _go_prev(self) -> None:
        self._stop_playback()
        self._scrub.setValue(max(self._current_frame - 1, 0))

    def _go_next(self) -> None:
        self._stop_playback()
        self._scrub.setValue(min(self._current_frame + 1, self._frame_count - 1))

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._frame_count <= 1:
            return
        self._playing = True
        self._btn_play.setText("Pause")
        interval = max(int(1000.0 / self._fps), 1)
        self._timer.start(interval)

    def _stop_playback(self) -> None:
        self._playing = False
        self._btn_play.setText("Play")
        self._timer.stop()

    def _tick(self) -> None:
        next_frame = self._current_frame + 1
        if next_frame >= self._frame_count:
            next_frame = 0  # loop
        self._scrub.setValue(next_frame)

    def step_prev(self) -> None:
        self._go_prev()

    def step_next(self) -> None:
        self._go_next()

    def toggle_play(self) -> None:
        self._toggle_play()
