"""Video player widget with playback controls, timeline slider, frame skip and detection rendering."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence
import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.schemas.entities import DetectedWeed
from geo_weed_station.ui.widgets.detection_filter_panel import DetectionFilterState
from geo_weed_station.ui.widgets.image_viewer import CLASS_COLORS

logger = logging.getLogger(__name__)


class VideoFrameCanvas(QWidget):
    """High-performance canvas rendering video frames and detection boxes at >= 30 FPS."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: Optional[QPixmap] = None
        self._detections: List[DetectedWeed] = []
        self._filter_state = DetectionFilterState()

    def set_frame(self, frame: np.ndarray, detections: Sequence[DetectedWeed]) -> None:
        """Update active frame and detection overlay."""
        if frame is None or frame.size == 0:
            self._pixmap = None
            self._detections = []
            self.update()
            return

        h, w = frame.shape[:2]
        arr = np.ascontiguousarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        elif arr.shape[2] == 3:
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        elif arr.shape[2] == 4:
            qimg = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        else:
            return

        self._pixmap = QPixmap.fromImage(qimg)
        self._detections = list(detections)
        self.update()

    def set_filter(self, state: DetectionFilterState) -> None:
        """Update detection filter state."""
        self._filter_state = state
        self.update()

    def clear(self) -> None:
        """Clear canvas."""
        self._pixmap = None
        self._detections = []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Видео не загружено")
            return

        # Scale pixmap to fit widget preserving aspect ratio
        widget_size = self.size()
        scaled = self._pixmap.scaled(
            widget_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,  # Fast transform for high FPS
        )

        offset_x = (widget_size.width() - scaled.width()) // 2
        offset_y = (widget_size.height() - scaled.height()) // 2
        painter.drawPixmap(offset_x, offset_y, scaled)

        # Draw detections if enabled
        if not self._filter_state.show_boxes or not self._detections:
            return

        scale_x = float(scaled.width())
        scale_y = float(scaled.height())

        font = QFont("sans-serif", 8, QFont.Weight.Bold)
        painter.setFont(font)

        for weed in self._detections:
            if weed.confidence < self._filter_state.min_confidence:
                continue
            if weed.class_name in self._filter_state.hidden_classes:
                continue

            color = CLASS_COLORS[weed.class_id % len(CLASS_COLORS)]
            pen = QPen(color, 2.0)
            painter.setPen(pen)
            brush_color = QColor(color)
            brush_color.setAlpha(30)
            painter.setBrush(QBrush(brush_color))

            rx = offset_x + weed.bbox.x * scale_x
            ry = offset_y + weed.bbox.y * scale_y
            rw = weed.bbox.w * scale_x
            rh = weed.bbox.h * scale_y
            rect = QRectF(rx, ry, rw, rh)
            painter.drawRect(rect)

            # Draw label
            label_parts: List[str] = []
            if self._filter_state.show_labels:
                label_parts.append(weed.class_name)
            if self._filter_state.show_confidence:
                label_parts.append(f"{weed.confidence:.2f}")

            if label_parts:
                text = " ".join(label_parts)
                painter.setPen(Qt.GlobalColor.white)
                painter.drawText(QPointF(rx + 2.0, max(12.0, ry - 3.0)), text)


class VideoPlayerWidget(QWidget):
    """Video player control widget with playback controls, timeline and frame skip selector."""

    play_requested = Signal()
    pause_requested = Signal()
    seek_requested = Signal(int)
    frame_skip_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._is_playing = False
        self._current_frame = 0
        self._total_frames = 0
        self._is_user_dragging = False

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # 1. Canvas
        self.canvas = VideoFrameCanvas(self)
        main_layout.addWidget(self.canvas, stretch=1)

        # 2. Timeline slider + position label
        timeline_layout = QHBoxLayout()
        timeline_layout.setSpacing(8)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        timeline_layout.addWidget(self.slider, stretch=1)

        self.label_position = QLabel("0 / 0", self)
        self.label_position.setMinimumWidth(80)
        timeline_layout.addWidget(self.label_position)
        main_layout.addLayout(timeline_layout)

        # 3. Playback controls toolbar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.btn_play_pause = QPushButton("▶ Воспроизведение", self)
        self.btn_play_pause.clicked.connect(self._on_play_pause_clicked)
        controls_layout.addWidget(self.btn_play_pause)

        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Пропуск кадров:", self))

        self.combo_skip = QComboBox(self)
        self.combo_skip.addItem("Каждый кадр (1x)", 1)
        self.combo_skip.addItem("Каждый 2-й кадр", 2)
        self.combo_skip.addItem("Каждый 5-й кадр", 5)
        self.combo_skip.addItem("Каждый 10-й кадр", 10)
        self.combo_skip.currentIndexChanged.connect(self._on_skip_changed)
        controls_layout.addWidget(self.combo_skip)

        controls_layout.addStretch(1)
        main_layout.addLayout(controls_layout)

    # -------------------------------------------------------------------------
    # Public slots & methods
    # -------------------------------------------------------------------------
    @Slot(object, list)
    def show_frame(self, frame: np.ndarray, detections: list[DetectedWeed]) -> None:
        """Render a single video frame with detections."""
        self.canvas.set_frame(frame, detections)

    @Slot(int, int)
    def set_position(self, current: int, total: int) -> None:
        """Update playback position without triggering seek signal."""
        self._current_frame = current
        self._total_frames = total

        self.label_position.setText(f"{current} / {total}")

        if not self._is_user_dragging:
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(0, total - 1))
            self.slider.setValue(current)
            self.slider.blockSignals(False)

    @Slot(bool)
    def set_playing(self, is_playing: bool) -> None:
        """Update play/pause button state."""
        self._is_playing = is_playing
        if is_playing:
            self.btn_play_pause.setText("⏸ Пауза")
        else:
            self.btn_play_pause.setText("▶ Воспроизведение")

    def apply_filter(self, state: DetectionFilterState) -> None:
        """Pass filter state down to canvas."""
        self.canvas.set_filter(state)

    def clear(self) -> None:
        """Reset player state."""
        self._is_playing = False
        self._current_frame = 0
        self._total_frames = 0
        self.set_playing(False)
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.label_position.setText("0 / 0")
        self.canvas.clear()

    # -------------------------------------------------------------------------
    # Internal handlers
    # -------------------------------------------------------------------------
    def _on_play_pause_clicked(self) -> None:
        if self._is_playing:
            self._is_playing = False
            self.btn_play_pause.setText("▶ Воспроизведение")
            self.pause_requested.emit()
        else:
            self._is_playing = True
            self.btn_play_pause.setText("⏸ Пауза")
            self.play_requested.emit()

    def _on_slider_pressed(self) -> None:
        self._is_user_dragging = True

    def _on_slider_released(self) -> None:
        self._is_user_dragging = False
        self.seek_requested.emit(self.slider.value())

    def _on_slider_value_changed(self, value: int) -> None:
        if self._is_user_dragging:
            self.label_position.setText(f"{value} / {self._total_frames}")

    def _on_skip_changed(self, index: int) -> None:
        val = int(self.combo_skip.currentData())
        self.frame_skip_changed.emit(val)
