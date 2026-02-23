"""Interactive frame viewer with rectangle draw/delete capabilities.

Mouse interactions:
* **Left-drag** on empty canvas → draw a new box (dashed cyan preview).
* **Left-click** inside an existing box → delete it (topmost box wins when
  boxes overlap).

Signals:
* ``rectangle_created(x, y, w, h)`` — emitted when a new rectangle is
  finalised; coordinates are in *image-pixel* space.
* ``rectangle_deleted(idx)`` — emitted when the user clicks inside an
  existing box; *idx* is the position in ``self._boxes``.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class FrameViewer(QWidget):
    """Displays a frame with optional bounding-box overlays.

    Supports interactive rectangle drawing (drag) and deletion (click).
    Heatmap overlay parameters are retained for backward compatibility but
    are no longer composited by default.
    """

    # Emitted with image-pixel coordinates when the user draws a rectangle.
    rectangle_created = Signal(int, int, int, int)  # x, y, w, h
    # Emitted with the *index* of the deleted box.
    rectangle_deleted = Signal(int)

    # Minimum drag size (image pixels) before a release is treated as a new
    # rectangle rather than an accidental click.
    _MIN_DRAG_PX = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._heatmap: np.ndarray | None = None
        self._boxes: list[tuple[int, int, int, int]] = []
        self._show_heatmap = False
        self._show_boxes = True
        self._pixmap: QPixmap | None = None

        # Drag state (widget-pixel coordinates)
        self._dragging = False
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None

        # Cached layout geometry (updated in paintEvent)
        self._img_ox: int = 0
        self._img_oy: int = 0
        self._img_sw: int = 0
        self._img_sh: int = 0
        self._img_scale: float = 1.0

        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame(
        self,
        frame: np.ndarray | None,
        heatmap: np.ndarray | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self._frame = frame
        self._heatmap = heatmap
        self._boxes = list(boxes) if boxes else []
        self._rebuild_pixmap()
        self.update()

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        """Update only the box list (cheap, no full rebuild)."""
        self._boxes = list(boxes)
        self.update()

    def set_display_options(self, show_heatmap: bool, show_boxes: bool) -> None:
        """Retained for backward compatibility."""
        self._show_heatmap = show_heatmap
        self._show_boxes = show_boxes
        self._rebuild_pixmap()
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_pixmap(self) -> None:
        if self._frame is None:
            self._pixmap = None
            return

        display = self._frame.copy()

        if self._show_heatmap and self._heatmap is not None:
            overlay = self._make_heatmap_overlay(self._heatmap)
            display = self._blend_overlay(display, overlay)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(qimg)

    def _compute_layout(self) -> None:
        """Compute and cache the scale/offset for the current pixmap/widget."""
        if self._pixmap is None:
            self._img_scale = 1.0
            self._img_ox = self._img_oy = self._img_sw = self._img_sh = 0
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        self._img_scale = scale
        self._img_sw = int(pw * scale)
        self._img_sh = int(ph * scale)
        self._img_ox = (ww - self._img_sw) // 2
        self._img_oy = (wh - self._img_sh) // 2

    def _widget_to_image(self, pt: QPoint) -> tuple[int, int] | None:
        """Convert a widget-pixel point to image-pixel coords.

        Returns ``None`` if the point is outside the displayed image area.
        """
        if self._pixmap is None:
            return None

        # Recompute here too so mouse interactions stay correct even if the
        # widget geometry changed since the last paintEvent.
        self._compute_layout()

        if self._img_scale == 0.0:
            return None

        ix = (pt.x() - self._img_ox) / self._img_scale
        iy = (pt.y() - self._img_oy) / self._img_scale
        if 0 <= ix < self._pixmap.width() and 0 <= iy < self._pixmap.height():
            return int(ix), int(iy)
        return None

    def _rect_from_drag(self) -> tuple[int, int, int, int] | None:
        """Return (x, y, w, h) in image-pixel space from current drag state."""
        if self._drag_start is None or self._drag_current is None:
            return None
        p1 = self._widget_to_image(self._drag_start)
        p2 = self._widget_to_image(self._drag_current)
        if p1 is None or p2 is None:
            return None
        x = min(p1[0], p2[0])
        y = min(p1[1], p2[1])
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        return x, y, w, h

    def _point_in_box(self, ix: int, iy: int, box: tuple[int, int, int, int]) -> bool:
        bx, by, bw, bh = box
        return bx <= ix < bx + bw and by <= iy < by + bh

    def _drag_preview_widget_rect(self) -> QRect | None:
        """Return the preview rectangle in *widget* coords for painting."""
        if self._drag_start is None or self._drag_current is None:
            return None
        x1, y1 = self._drag_start.x(), self._drag_start.y()
        x2, y2 = self._drag_current.x(), self._drag_current.y()
        return QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.position().toPoint()
        coord = self._widget_to_image(pos)
        if coord is None:
            return

        ix, iy = coord

        # Check topmost-first (reverse iteration) — delete on click inside box
        for i in range(len(self._boxes) - 1, -1, -1):
            if self._point_in_box(ix, iy, self._boxes[i]):
                self.rectangle_deleted.emit(i)
                return

        # Start a new drag
        self._dragging = True
        self._drag_start = pos
        self._drag_current = pos
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._drag_current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._dragging:
            return

        self._drag_current = event.position().toPoint()
        rect = self._rect_from_drag()
        if rect is not None:
            x, y, w, h = rect
            if w >= self._MIN_DRAG_PX and h >= self._MIN_DRAG_PX:
                self.rectangle_created.emit(x, y, w, h)

        # Reset drag state
        self._dragging = False
        self._drag_start = None
        self._drag_current = None
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._pixmap is None:
            painter.fillRect(self.rect(), QColor(26, 26, 26))
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load an image folder to begin",
            )
            painter.end()
            return

        self._compute_layout()
        ox, oy = self._img_ox, self._img_oy
        sw, sh = self._img_sw, self._img_sh
        scale = self._img_scale

        painter.fillRect(self.rect(), QColor(26, 26, 26))
        painter.drawPixmap(QRect(ox, oy, sw, sh), self._pixmap)

        # Draw stored boxes
        if self._show_boxes and self._boxes:
            self._draw_boxes(painter, ox, oy, scale)

        # Draw drag preview
        if self._dragging:
            preview = self._drag_preview_widget_rect()
            if preview and preview.width() > 0 and preview.height() > 0:
                pen = QPen(QColor(255, 220, 0), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                fill = QColor(255, 220, 0, 30)
                painter.fillRect(preview, fill)
                painter.drawRect(preview)

        painter.end()

    def _draw_boxes(self, painter: QPainter, ox: int, oy: int, scale: float) -> None:
        box_color = QColor(0, 200, 255, 220)   # cyan for all annotation boxes
        fill_color = QColor(0, 200, 255, 40)

        for x, y, bw, bh in self._boxes:
            rx = int(x * scale) + ox
            ry = int(y * scale) + oy
            rw = int(bw * scale)
            rh = int(bh * scale)
            rect = QRect(rx, ry, rw, rh)

            painter.fillRect(rect, fill_color)
            pen = QPen(box_color, 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    # ------------------------------------------------------------------
    # Heatmap helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _make_heatmap_overlay(self, heatmap: np.ndarray) -> np.ndarray:
        heatmap_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        alpha = (heatmap * 180).astype(np.uint8)
        bgra = np.zeros((*heatmap.shape, 4), dtype=np.uint8)
        bgra[..., :3] = colored
        bgra[..., 3] = alpha
        return bgra

    @staticmethod
    def _blend_overlay(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
        alpha = overlay_bgra[..., 3:4].astype(np.float32) / 255.0
        base_f = base_bgr.astype(np.float32)
        over_f = overlay_bgra[..., :3].astype(np.float32)
        blended = base_f * (1.0 - alpha) + over_f * alpha
        return blended.astype(np.uint8)
