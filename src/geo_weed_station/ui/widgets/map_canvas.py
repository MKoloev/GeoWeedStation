"""Interactive 2D digital mapping canvas (QGraphicsView) for weed points, heatmap, route and telemetry."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import HeatmapGrid, WeedPoint
from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.telemetry import DronePose
from geo_weed_station.ui.widgets.image_viewer import CLASS_COLORS

logger = logging.getLogger(__name__)

LOCKED_DEFAULT_MESSAGE = "Геопривязка отсутствует. Доступен только просмотр исходных изображений"


class WeedGraphicsPointItem(QGraphicsEllipseItem):
    """Clickable point marker on the map representing a localized weed."""

    def __init__(
        self,
        point: WeedPoint,
        scene_x: float,
        scene_y: float,
        color: QColor,
        radius: float = 3.5,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(scene_x - radius, scene_y - radius, radius * 2.0, radius * 2.0)
        self.point = point
        self.color = color

        pen = QPen(Qt.GlobalColor.white, 1.0)
        pen.setCosmetic(True)  # Keeps constant pixel size when zooming
        self.setPen(pen)
        self.setBrush(QBrush(color))
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if point.coord_system == CoordinateSystem.WGS84 and point.lat is not None and point.lon is not None:
            self.setToolTip(
                f"{point.class_name} ({point.confidence:.2f})\n"
                f"GPS: {point.lat:.7f}°, {point.lon:.7f}°\n"
                f"Local: X={point.x:.1f} m, Y={point.y:.1f} m"
            )
        else:
            self.setToolTip(f"{point.class_name} ({point.confidence:.2f})\nX: {point.x:.2f} m, Y: {point.y:.2f} m")


class MapCanvas(QGraphicsView):
    """Interactive 2D digital map canvas supporting weed point layers, heatmaps, route and drone telemetry."""

    weed_selected = Signal(str)  # weed_id
    zoom_changed = Signal(float)

    ZOOM_STEP = 1.15
    MIN_ZOOM = 0.001
    MAX_ZOOM = 1000.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._coord_system = CoordinateSystem.LOCAL_ARUCO
        self._locked = False
        self._heatmap_opacity_pct = 70
        self._current_zoom = 1.0

        # Layer containers
        self._heatmap_item: QGraphicsPixmapItem | None = None
        self._route_group: QGraphicsItemGroup | None = None
        self._points_group: QGraphicsItemGroup | None = None
        self._drone_group: QGraphicsItemGroup | None = None
        self._locked_overlay: QGraphicsItemGroup | None = None

        self._points_list: list[WeedPoint] = []
        self._point_items: dict[str, WeedGraphicsPointItem] = {}

        self._setup_view()

    def _setup_view(self) -> None:
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(24, 28, 24)))

        # Create persistent groups for layered rendering
        self._heatmap_group = self._scene.createItemGroup([])
        self._route_group = self._scene.createItemGroup([])
        self._points_group = self._scene.createItemGroup([])
        self._drone_group = self._scene.createItemGroup([])

        # Ensure correct z-stacking
        self._heatmap_group.setZValue(10)
        self._route_group.setZValue(20)
        self._points_group.setZValue(30)
        self._drone_group.setZValue(40)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def set_coordinate_system(self, cs: CoordinateSystem) -> None:
        """Switch coordinate system (local_aruco or wgs84)."""
        self._coord_system = cs

    def set_locked(self, locked: bool, message: str | None = None) -> None:
        """Lock or unlock map canvas. When locked, layers are hidden and warning is shown."""
        self._locked = locked
        text = message or LOCKED_DEFAULT_MESSAGE

        if self._locked_overlay is not None:
            self._scene.removeItem(self._locked_overlay)
            self._locked_overlay = None

        if locked:
            # Hide mapping layers
            self._heatmap_group.setVisible(False)
            self._route_group.setVisible(False)
            self._points_group.setVisible(False)
            self._drone_group.setVisible(False)

            # Create prominent warning banner in center
            self._locked_overlay = QGraphicsItemGroup()
            self._locked_overlay.setZValue(100)

            rect_item = QGraphicsRectItem(-250, -50, 500, 100)
            rect_item.setBrush(QBrush(QColor(40, 20, 20, 220)))
            rect_item.setPen(QPen(QColor(220, 50, 50), 2.0))

            text_item = QGraphicsSimpleTextItem(text, rect_item)
            text_item.setBrush(QBrush(Qt.GlobalColor.white))
            font = QFont("sans-serif", 10, QFont.Weight.Bold)
            text_item.setFont(font)
            text_rect = text_item.boundingRect()
            text_item.setPos(-text_rect.width() / 2.0, -text_rect.height() / 2.0)

            self._locked_overlay.addToGroup(rect_item)
            self._scene.addItem(self._locked_overlay)
        else:
            self._heatmap_group.setVisible(True)
            self._route_group.setVisible(True)
            self._points_group.setVisible(True)
            self._drone_group.setVisible(True)

    def is_locked(self) -> bool:
        """Return True if map canvas is locked."""
        return self._locked

    def set_points(self, points: Sequence[WeedPoint]) -> None:
        """Replace all weed points on the map."""
        self._clear_points()
        self.append_points(points)
        self._fit_scene_bounds()

    def append_points(self, points: Sequence[WeedPoint]) -> None:
        """Append points to the existing map layer."""
        if not points:
            return

        for p in points:
            self._points_list.append(p)
            # In cartesian mapping: Y points UP, in Qt graphics scene Y points DOWN
            sx = p.x
            sy = -p.y

            color_idx = hash(p.class_name) % len(CLASS_COLORS)
            color = CLASS_COLORS[color_idx]

            item = WeedGraphicsPointItem(point=p, scene_x=sx, scene_y=sy, color=color)
            self._points_group.addToGroup(item)
            self._point_items[p.weed_id] = item

    def set_heatmap(self, grid: HeatmapGrid | None) -> None:
        """Render density heatmap grid as semi-transparent raster overlay."""
        # Remove old heatmap
        if self._heatmap_item is not None:
            self._scene.removeItem(self._heatmap_item)
            self._heatmap_item = None

        if grid is None or grid.matrix is None:
            return

        matrix = np.asarray(grid.matrix, dtype=np.float32)
        if matrix.size == 0 or matrix.ndim != 2:
            return

        rows, cols = matrix.shape
        max_val = float(np.max(matrix))
        if max_val <= 0:
            norm = np.zeros_like(matrix)
        else:
            norm = matrix / max_val

        # Create RGBA colormap: Dark Green (low) -> Yellow (mid) -> Bright Red (high)
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        # Vectorized color gradient calculation
        r = np.clip(np.where(norm < 0.5, 2.0 * norm * 255.0, 255.0), 0, 255).astype(np.uint8)
        g = np.clip(np.where(norm < 0.5, 200.0, 2.0 * (1.0 - norm) * 200.0), 0, 255).astype(np.uint8)
        b = np.full((rows, cols), 20, dtype=np.uint8)
        alpha = np.where(norm > 0.01, 220, 0).astype(np.uint8)

        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = alpha

        # Flip vertically so row 0 is top
        rgba = np.ascontiguousarray(rgba)
        qimg = QImage(rgba.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(qimg)

        # Scale and position pixmap on map
        bounds = grid.bounds
        self._heatmap_item = QGraphicsPixmapItem(pixmap)
        self._heatmap_item.setOpacity(self._heatmap_opacity_pct / 100.0)

        # Bounds: min_x, min_y, max_x, max_y.
        # In scene: x = min_x, y = -max_y, width = w, height = h
        bw = max(0.1, bounds.width)
        bh = max(0.1, bounds.height)
        sx = bw / float(cols)
        sy = bh / float(rows)

        self._heatmap_item.setPos(bounds.min_x, -bounds.max_y)
        self._heatmap_item.setTransform(self._heatmap_item.transform().scale(sx, sy))
        self._heatmap_group.addToGroup(self._heatmap_item)

    @Slot(int)
    def set_heatmap_opacity(self, percent: int) -> None:
        """Update opacity of the heatmap overlay (0 to 100%)."""
        self._heatmap_opacity_pct = max(0, min(100, int(percent)))
        if self._heatmap_item is not None:
            self._heatmap_item.setOpacity(self._heatmap_opacity_pct / 100.0)

    def set_route(self, waypoints: Sequence[Waypoint]) -> None:
        """Render serpentine flight route waypoints and connecting trajectory."""
        # Clear existing route items
        for it in list(self._route_group.childItems()):
            self._scene.removeItem(it)

        if not waypoints:
            return

        path = QPainterPath()
        p0 = waypoints[0]
        path.moveTo(p0.x, -p0.y)

        for wp in waypoints[1:]:
            path.lineTo(wp.x, -wp.y)

        # Trajectory path item
        path_item = QGraphicsPathItem(path)
        pen = QPen(QColor(80, 160, 255), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        path_item.setPen(pen)
        self._route_group.addToGroup(path_item)

        # Waypoint nodes with numbers
        font = QFont("sans-serif", 8)
        for idx, wp in enumerate(waypoints):
            node = QGraphicsEllipseItem(wp.x - 2.5, -wp.y - 2.5, 5.0, 5.0)
            n_pen = QPen(QColor(255, 255, 255), 1.0)
            n_pen.setCosmetic(True)
            node.setPen(n_pen)
            node.setBrush(QBrush(QColor(30, 144, 255)))
            self._route_group.addToGroup(node)

            if idx % max(1, len(waypoints) // 10) == 0 or idx == len(waypoints) - 1:
                label = QGraphicsSimpleTextItem(f"{idx}", node)
                label.setBrush(QBrush(Qt.GlobalColor.white))
                label.setFont(font)
                label.setPos(wp.x + 3.0, -wp.y - 10.0)

    def set_drone_pose(self, pose: DronePose | None) -> None:
        """Render drone current position marker and optical heading vector (<= 100 ms latency)."""
        # Clear drone group
        for it in list(self._drone_group.childItems()):
            self._scene.removeItem(it)

        if pose is None:
            return

        cx = pose.x
        cy = -pose.y

        # Drone triangle marker pointing towards yaw
        yaw_rad = math.radians(pose.yaw)
        size = 8.0

        # Create triangle pointing along yaw
        tip = QPointF(cx + size * math.cos(yaw_rad), cy - size * math.sin(yaw_rad))
        left = QPointF(
            cx + (size / 1.5) * math.cos(yaw_rad + 2.5),
            cy - (size / 1.5) * math.sin(yaw_rad + 2.5),
        )
        right = QPointF(
            cx + (size / 1.5) * math.cos(yaw_rad - 2.5),
            cy - (size / 1.5) * math.sin(yaw_rad - 2.5),
        )

        poly = QPolygonF([tip, left, right])
        poly_item = QGraphicsPolygonItem(poly)
        d_pen = QPen(Qt.GlobalColor.black, 1.0)
        d_pen.setCosmetic(True)
        poly_item.setPen(d_pen)
        poly_item.setBrush(QBrush(QColor(255, 215, 0)))  # Gold / Yellow
        self._drone_group.addToGroup(poly_item)

        # Optical axis line vector (forward direction)
        vec_len = max(size * 2.0, pose.z if pose.z > 0 else 15.0)
        end_pt = QPointF(cx + vec_len * math.cos(yaw_rad), cy - vec_len * math.sin(yaw_rad))
        line_item = QGraphicsLineItem(cx, cy, end_pt.x(), end_pt.y())
        l_pen = QPen(QColor(255, 230, 100), 1.5, Qt.PenStyle.DotLine)
        l_pen.setCosmetic(True)
        line_item.setPen(l_pen)
        self._drone_group.addToGroup(line_item)

    def render_to_image(self, scale: float = 1.0) -> np.ndarray:
        """Render current scene rect to an RGBA numpy array (H, W, 4)."""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            rect = QRectF(-50, -50, 100, 100)

        w = max(1, int(round(rect.width() * scale)))
        h = max(1, int(round(rect.height() * scale)))

        image = QImage(w, h, QImage.Format.Format_RGBA8888)
        image.fill(QColor(24, 28, 24).rgba())

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._scene.render(painter, QRectF(0, 0, w, h), rect)
        painter.end()

        # Convert QImage to numpy array
        ptr = image.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        return arr.copy()

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
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
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, WeedGraphicsPointItem):
            self.weed_selected.emit(item.point.weed_id)
        super().mousePressEvent(event)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    def _clear_points(self) -> None:
        self._points_list.clear()
        self._point_items.clear()
        for it in list(self._points_group.childItems()):
            self._scene.removeItem(it)

    def _fit_scene_bounds(self) -> None:
        rect = self._scene.itemsBoundingRect()
        if not rect.isEmpty():
            # Add padding
            padded = rect.adjusted(-10, -10, 10, 10)
            self._scene.setSceneRect(padded)
            self.fitInView(padded, Qt.AspectRatioMode.KeepAspectRatio)
            self._current_zoom = self.transform().m11()
            self.zoom_changed.emit(self._current_zoom)
