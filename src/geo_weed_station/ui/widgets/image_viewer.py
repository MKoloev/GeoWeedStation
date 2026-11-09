"""Interactive QGraphicsView-based image viewer with pan, zoom, and detection overlay."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

from geo_weed_station.schemas.entities import DetectedWeed
from geo_weed_station.ui.widgets.detection_filter_panel import DetectionFilterState

logger = logging.getLogger(__name__)

# Palette of distinct colors for detection classes
CLASS_COLORS = [
    QColor(230, 57, 70),    # Red
    QColor(244, 162, 97),   # Orange
    QColor(42, 157, 143),   # Teal
    QColor(38, 70, 83),     # Dark Blue
    QColor(155, 93, 229),   # Purple
    QColor(241, 91, 181),   # Pink
    QColor(0, 187, 249),    # Cyan
    QColor(0, 245, 212),    # Bright mint
    QColor(254, 217, 102),  # Yellow
]


class DetectionGraphicsItem(QGraphicsRectItem):
    """Clickable graphics item for a single weed detection box and text label."""

    def __init__(
        self,
        weed: DetectedWeed,
        rect: QRectF,
        color: QColor,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(rect, parent)
        self.weed = weed
        self.color = color

        # Configure appearance
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)  # Width stays 2px regardless of zoom
        self.setPen(pen)
        brush_color = QColor(color)
        brush_color.setAlpha(30)
        self.setBrush(QBrush(brush_color))
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Label item
        self.label_item = QGraphicsSimpleTextItem(self)
        self.label_item.setBrush(QBrush(Qt.GlobalColor.white))
        font = QFont("sans-serif", 9, QFont.Weight.Bold)
        self.label_item.setFont(font)
        self.label_item.setPos(rect.left(), max(0.0, rect.top() - 16.0))

    def update_visibility(self, state: DetectionFilterState) -> None:
        """Update visibility based on filter state."""
        # 1. Check confidence and hidden classes
        if self.weed.confidence < state.min_confidence:
            self.setVisible(False)
            return

        if self.weed.class_name in state.hidden_classes:
            self.setVisible(False)
            return

        # 2. Check box visibility
        self.setVisible(state.show_boxes)

        # 3. Text label visibility and content
        label_parts: list[str] = []
        if state.show_labels:
            label_parts.append(self.weed.class_name)
        if state.show_confidence:
            label_parts.append(f"{self.weed.confidence:.2f}")

        if label_parts and state.show_boxes:
            self.label_item.setText(" ".join(label_parts))
            self.label_item.setVisible(True)
        else:
            self.label_item.setVisible(False)

    def set_highlighted(self, highlight: bool) -> None:
        """Apply high-contrast highlight styling when selected or focused."""
        if highlight:
            pen = QPen(QColor(255, 235, 20), 3.5)
            pen.setCosmetic(True)
            self.setPen(pen)
            brush_color = QColor(255, 235, 20)
            brush_color.setAlpha(70)
            self.setBrush(QBrush(brush_color))
            self.setZValue(100)
        else:
            pen = QPen(self.color, 2.0)
            pen.setCosmetic(True)
            self.setPen(pen)
            brush_color = QColor(self.color)
            brush_color.setAlpha(30)
            self.setBrush(QBrush(brush_color))
            self.setZValue(1)


class ImageViewer(QGraphicsView):
    """High-performance image viewer supporting smooth zoom, pan, and detection overlays."""

    zoom_changed = Signal(float)
    weed_clicked = Signal(str)  # weed_id

    MAX_PIXMAP_DIM = 4096  # Downscale threshold to prevent OOM on 8000x6000 images
    ZOOM_STEP = 1.15
    MIN_ZOOM = 0.05
    MAX_ZOOM = 50.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._detection_items: list[DetectionGraphicsItem] = []
        self._filter_state = DetectionFilterState()

        # Resolution scaling tracking
        self._orig_width: int = 0
        self._orig_height: int = 0
        self._scale_to_scene_x: float = 1.0
        self._scale_to_scene_y: float = 1.0
        self._current_zoom: float = 1.0

        self._setup_view()

    def _setup_view(self) -> None:
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        """Display numpy image array (H, W, 3) or (H, W). Scales if exceeding MAX_PIXMAP_DIM."""
        self.clear()
        if image is None or image.size == 0:
            return

        h, w = image.shape[:2]
        self._orig_width = w
        self._orig_height = h

        # Handle downscaling for extreme dimensions to avoid memory issues
        scale = 1.0
        max_dim = max(h, w)
        if max_dim > self.MAX_PIXMAP_DIM:
            scale = self.MAX_PIXMAP_DIM / float(max_dim)
            scaled_w = max(1, int(round(w * scale)))
            scaled_h = max(1, int(round(h * scale)))
            # Downscale image by strided slicing / box decimation for fast safe display
            step_y = max(1, int(round(1.0 / scale)))
            step_x = max(1, int(round(1.0 / scale)))
            display_img = np.ascontiguousarray(image[::step_y, ::step_x])
            disp_h, disp_w = display_img.shape[:2]
        else:
            display_img = np.ascontiguousarray(image)
            disp_h, disp_w = h, w

        # Map normalized coordinates directly to display image scene coords
        self._scale_to_scene_x = float(disp_w)
        self._scale_to_scene_y = float(disp_h)

        qimage = self._numpy_to_qimage(display_img)
        pixmap = QPixmap.fromImage(qimage)

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(0, 0, disp_w, disp_h)
        self.fit_to_window()

    def set_detections(self, detections: Sequence[DetectedWeed]) -> None:
        """Add bounding box overlays for detected weeds."""
        # Clear existing detection items
        for item in self._detection_items:
            self._scene.removeItem(item)
        self._detection_items.clear()

        if not detections or self._scale_to_scene_x <= 0:
            return

        for weed in detections:
            color = CLASS_COLORS[weed.class_id % len(CLASS_COLORS)]
            # BBox coords are normalized [0..1]
            bx = weed.bbox.x * self._scale_to_scene_x
            by = weed.bbox.y * self._scale_to_scene_y
            bw = weed.bbox.w * self._scale_to_scene_x
            bh = weed.bbox.h * self._scale_to_scene_y

            rect = QRectF(bx, by, bw, bh)
            item = DetectionGraphicsItem(weed=weed, rect=rect, color=color)
            item.update_visibility(self._filter_state)
            self._scene.addItem(item)
            self._detection_items.append(item)

    def focus_on_weed(self, weed_id: str, bbox: Any = None) -> None:
        """Focus and highlight a weed detection box on the image."""
        target_item: DetectionGraphicsItem | None = None
        for item in self._detection_items:
            if item.weed.weed_id == weed_id:
                item.set_highlighted(True)
                target_item = item
            else:
                item.set_highlighted(False)

        if target_item is not None:
            self.centerOn(target_item)
            rect = target_item.sceneBoundingRect().adjusted(-60, -60, 60, 60)
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._current_zoom = self.transform().m11()
            self.zoom_changed.emit(self._current_zoom)

    @Slot(DetectionFilterState)
    def apply_filter(self, state: DetectionFilterState) -> None:
        """Apply detection filter state dynamically to all active items."""
        self._filter_state = state
        for item in self._detection_items:
            item.update_visibility(state)

    def fit_to_window(self) -> None:
        """Scale image to fit within the viewport while preserving aspect ratio."""
        rect = self._scene.sceneRect()
        if rect.isValid() and not rect.isEmpty():
            self.resetTransform()
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._current_zoom = self.transform().m11()
            self.zoom_changed.emit(self._current_zoom)

    def clear(self) -> None:
        """Clear scene, image, and detections."""
        self._scene.clear()
        self._pixmap_item = None
        self._detection_items.clear()
        self._orig_width = 0
        self._orig_height = 0
        self._scale_to_scene_x = 1.0
        self._scale_to_scene_y = 1.0
        self.resetTransform()

    # -------------------------------------------------------------------------
    # Qt Event Handlers
    # -------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Smooth zoom anchored at cursor position."""
        if self._pixmap_item is None:
            super().wheelEvent(event)
            return

        angle = event.angleDelta().y()
        if angle == 0:
            return

        factor = self.ZOOM_STEP if angle > 0 else (1.0 / self.ZOOM_STEP)
        new_zoom = self._current_zoom * factor

        if self.MIN_ZOOM <= new_zoom <= self.MAX_ZOOM:
            self.scale(factor, factor)
            self._current_zoom = new_zoom
            self.zoom_changed.emit(self._current_zoom)

        event.accept()

    def mousePressEvent(self, event) -> None:
        """Handle clicks on detections to emit weed_clicked signal."""
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, DetectionGraphicsItem):
            self.weed_clicked.emit(item.weed.weed_id)
        elif isinstance(item, QGraphicsSimpleTextItem) and isinstance(item.parentItem(), DetectionGraphicsItem):
            self.weed_clicked.emit(item.parentItem().weed.weed_id)
        super().mousePressEvent(event)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _numpy_to_qimage(arr: np.ndarray) -> QImage:
        """Convert numpy uint8 array to QImage."""
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            h, w = arr.shape
            return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

        h, w, c = arr.shape
        if c == 1:
            return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        if c == 3:
            bytes_per_line = 3 * w
            return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        if c == 4:
            bytes_per_line = 4 * w
            return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()

        raise ValueError(f"Unsupported image shape for display: {arr.shape}")
