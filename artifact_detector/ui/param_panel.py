from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ParamSlider(QWidget):
    """A labeled slider that shows its current numeric value."""

    value_changed = Signal()

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float = 0.01,
        is_int: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._step = step
        self._is_int = is_int

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self._label = QLabel(label)
        self._label.setFixedWidth(120)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        steps = int((max_val - min_val) / step)
        self._slider.setRange(0, steps)
        self._slider.setValue(int((default - min_val) / step))
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, 1)

        self._value_label = QLabel()
        self._value_label.setFixedWidth(45)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._value_label)

        self._update_value_label()

    def _on_slider_changed(self) -> None:
        self._update_value_label()
        self.value_changed.emit()

    def _update_value_label(self) -> None:
        val = self.value()
        if self._is_int:
            self._value_label.setText(str(int(val)))
        else:
            self._value_label.setText(f"{val:.2f}")

    def value(self) -> float:
        raw = self._slider.value() * self._step + self._min
        return int(round(raw)) if self._is_int else round(raw, 4)


class ParamPanel(QWidget):
    """Side panel with grouped detection parameters."""

    params_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(310)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # --- Blur Detection ---
        cu_group = QGroupBox("Blur Detection")
        cu_layout = QVBoxLayout(cu_group)
        cu_layout.setSpacing(4)

        self.cu_enabled = QCheckBox("Enable")
        self.cu_enabled.setChecked(True)
        cu_layout.addWidget(self.cu_enabled)

        self.cu_threshold = ParamSlider("Threshold", 0.0, 1.0, 0.20)
        cu_layout.addWidget(self.cu_threshold)

        main_layout.addWidget(cu_group)

        # --- Edge Coherence ---
        ec_group = QGroupBox("Edge Coherence")
        ec_layout = QVBoxLayout(ec_group)
        ec_layout.setSpacing(4)

        self.ec_enabled = QCheckBox("Enable")
        self.ec_enabled.setChecked(False)
        ec_layout.addWidget(self.ec_enabled)

        self.ec_min_length = ParamSlider("Min Length", 5, 100, 30, step=1, is_int=True)
        ec_layout.addWidget(self.ec_min_length)

        self.ec_smoothness = ParamSlider("Smoothness", 0.0, 1.0, 0.55)
        ec_layout.addWidget(self.ec_smoothness)

        main_layout.addWidget(ec_group)

        # --- Temporal Anomaly ---
        ta_group = QGroupBox("Temporal Anomaly")
        ta_layout = QVBoxLayout(ta_group)
        ta_layout.setSpacing(4)

        self.ta_enabled = QCheckBox("Enable")
        self.ta_enabled.setChecked(True)
        ta_layout.addWidget(self.ta_enabled)

        self.ta_threshold = ParamSlider("Diff Threshold", 0.0, 1.0, 0.35)
        ta_layout.addWidget(self.ta_threshold)

        self.ta_window = ParamSlider("Window Size", 1, 5, 2, step=1, is_int=True)
        ta_layout.addWidget(self.ta_window)

        main_layout.addWidget(ta_group)

        # --- Region Settings ---
        region_group = QGroupBox("Region Settings")
        region_layout = QVBoxLayout(region_group)
        region_layout.setSpacing(4)

        self.min_region_size = ParamSlider("Min Region", 100, 5000, 500, step=100, is_int=True)
        region_layout.addWidget(self.min_region_size)

        self.patch_size = ParamSlider("Patch Size", 32, 512, 128, step=16, is_int=True)
        region_layout.addWidget(self.patch_size)

        main_layout.addWidget(region_group)

        # --- Display ---
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.show_heatmap = QCheckBox("Show heatmap")
        self.show_heatmap.setChecked(True)
        self.show_heatmap.stateChanged.connect(self.params_changed)
        display_layout.addWidget(self.show_heatmap)

        self.show_boxes = QCheckBox("Show boxes")
        self.show_boxes.setChecked(True)
        self.show_boxes.stateChanged.connect(self.params_changed)
        display_layout.addWidget(self.show_boxes)

        main_layout.addWidget(display_group)

        # --- Info ---
        self._region_count_label = QLabel("Detected: 0 regions")
        self._region_count_label.setStyleSheet("font-weight: bold; padding: 4px;")
        main_layout.addWidget(self._region_count_label)

        main_layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_region_count(self, count: int) -> None:
        self._region_count_label.setText(f"Detected: {count} regions")

    def get_params(self):
        from ..core.detector import DetectionParams

        return DetectionParams(
            color_uniformity_enabled=self.cu_enabled.isChecked(),
            color_uniformity_threshold=self.cu_threshold.value(),
            edge_coherence_enabled=self.ec_enabled.isChecked(),
            edge_min_length=int(self.ec_min_length.value()),
            edge_smoothness_threshold=self.ec_smoothness.value(),
            temporal_enabled=self.ta_enabled.isChecked(),
            temporal_diff_threshold=self.ta_threshold.value(),
            temporal_window=int(self.ta_window.value()),
            min_region_size=int(self.min_region_size.value()),
            patch_size=int(self.patch_size.value()),
        )
