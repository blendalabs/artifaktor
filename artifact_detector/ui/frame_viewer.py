from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class FrameViewer(QWidget):
    """Displays a frame with optional heatmap and bounding box overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._heatmap: np.ndarray | None = None
        self._boxes: list[tuple[int, int, int, int]] = []
        self._show_heatmap = True
        self._show_boxes = True
        self._pixmap: QPixmap | None = None
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1a1a1a;")

    def set_frame(
        self,
        frame: np.ndarray | None,
        heatmap: np.ndarray | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self._frame = frame
        self._heatmap = heatmap
        self._boxes = boxes or []
        self._rebuild_pixmap()
        self.update()

    def set_display_options(self, show_heatmap: bool, show_boxes: bool) -> None:
        self._show_heatmap = show_heatmap
        self._show_boxes = show_boxes
        self._rebuild_pixmap()
        self.update()

    def _rebuild_pixmap(self) -> None:
        if self._frame is None:
            self._pixmap = None
            return

        # Start with the original frame (BGR)
        display = self._frame.copy()

        # Composite heatmap overlay
        if self._show_heatmap and self._heatmap is not None:
            overlay = self._make_heatmap_overlay(self._heatmap)
            display = self._blend_overlay(display, overlay)

        # Convert BGR → RGB for Qt
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)

    def _make_heatmap_overlay(self, heatmap: np.ndarray) -> np.ndarray:
        """Create a colored BGRA overlay from a 0-1 heatmap."""
        # Colormap: blue (low) → red (high)
        heatmap_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        # Create BGRA with alpha proportional to heatmap intensity
        alpha = (heatmap * 180).astype(np.uint8)  # max alpha ~0.7
        bgra = np.zeros((*heatmap.shape, 4), dtype=np.uint8)
        bgra[..., :3] = colored
        bgra[..., 3] = alpha

        return bgra

    @staticmethod
    def _blend_overlay(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
        """Alpha-blend BGRA overlay onto BGR base."""
        alpha = overlay_bgra[..., 3:4].astype(np.float32) / 255.0
        base_f = base_bgr.astype(np.float32)
        over_f = overlay_bgra[..., :3].astype(np.float32)
        blended = base_f * (1.0 - alpha) + over_f * alpha
        return blended.astype(np.uint8)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._pixmap is None:
            painter.fillRect(self.rect(), QColor(26, 26, 26))
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a video or image folder to begin")
            painter.end()
            return

        # Scale pixmap to fit widget while preserving aspect ratio
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        sw, sh = int(pw * scale), int(ph * scale)
        ox = (ww - sw) // 2
        oy = (wh - sh) // 2

        painter.fillRect(self.rect(), QColor(26, 26, 26))
        painter.drawPixmap(QRect(ox, oy, sw, sh), self._pixmap)

        # Draw bounding boxes
        if self._show_boxes and self._boxes:
            self._draw_boxes(painter, ox, oy, scale)

        painter.end()

    def _draw_boxes(self, painter: QPainter, ox: int, oy: int, scale: float) -> None:
        colors = [
            QColor(0, 255, 100, 180),
            QColor(255, 200, 0, 180),
            QColor(255, 80, 80, 180),
            QColor(80, 180, 255, 180),
            QColor(200, 100, 255, 180),
        ]

        for i, (x, y, bw, bh) in enumerate(self._boxes):
            color = colors[i % len(colors)]

            # Scale coordinates to widget space
            rx = int(x * scale) + ox
            ry = int(y * scale) + oy
            rw = int(bw * scale)
            rh = int(bh * scale)
            rect = QRect(rx, ry, rw, rh)

            # Semi-transparent fill
            fill_color = QColor(color)
            fill_color.setAlpha(35)
            painter.fillRect(rect, fill_color)

            # Solid outline
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(rect)
