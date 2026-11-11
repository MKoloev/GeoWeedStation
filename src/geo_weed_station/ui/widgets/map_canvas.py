"""Interactive 2D digital mapping canvas (QGraphicsView) for weed points, heatmap, route and telemetry."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import cv2
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
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import (
    FrameFootprint,
    HeatmapGrid,
    PrescriptionZone,
    WeedPoint,
)
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
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0, parent)
        self.point = point
        self.color = color

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setPos(QPointF(scene_x, scene_y))

        pen = QPen(Qt.GlobalColor.white, 1.0)
        pen.setCosmetic(True)  # Keeps constant pixel size when zooming
        self.setPen(pen)
        self.setBrush(QBrush(color))
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        frame_lbl = point.best_frame_id or point.source_frame
        dup_str = f" (дублей: {point.duplicate_count})" if point.duplicate_count > 1 else ""
        if point.coord_system == CoordinateSystem.WGS84 and point.lat is not None and point.lon is not None:
            self.setToolTip(
                f"{point.class_name} ({point.confidence:.2f}){dup_str}\n"
                f"GPS: {point.lat:.7f}°, {point.lon:.7f}°\n"
                f"Local: X={point.x:.1f} m, Y={point.y:.1f} m\n"
                f"Кадр: {frame_lbl}"
            )
        else:
            self.setToolTip(
                f"{point.class_name} ({point.confidence:.2f}){dup_str}\n"
                f"X: {point.x:.2f} m, Y: {point.y:.2f} m\n"
                f"Кадр: {frame_lbl}"
            )


class MosaicFootprintItem(QGraphicsPolygonItem):
    """Interactive footprint polygon for an aerial frame with on-demand thumbnail pixmap."""

    def __init__(
        self,
        footprint: FrameFootprint,
        pixmap: QPixmap | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        poly = QPolygonF([QPointF(x, -y) for x, y in footprint.polygon])
        super().__init__(poly, parent)
        self.footprint = footprint

        pen = QPen(QColor(80, 160, 240, 160), 1.0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(40, 70, 100, 35)))
        self.setAcceptHoverEvents(True)
        self.setToolTip(
            f"Кадр: {footprint.frame_id}\n"
            f"Высота H: {footprint.altitude_m:.1f} м, Yaw: {footprint.yaw_deg:.1f}°"
        )

        pm = pixmap
        if pm is None:
            if footprint.thumbnail is not None:
                th = footprint.thumbnail
                th_h, th_w = th.shape[:2]
                qimg = QImage(th.data, th_w, th_h, th_w * 3, QImage.Format.Format_RGB888).copy()
                pm = QPixmap.fromImage(qimg)
            elif footprint.source_path and Path(footprint.source_path).is_file():
                try:
                    img = cv2.imread(str(footprint.source_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        scale = 400.0 / float(max(h, w))
                        thumb = cv2.resize(
                            img,
                            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                        th_h, th_w = thumb_rgb.shape[:2]
                        qimg = QImage(thumb_rgb.data, th_w, th_h, th_w * 3, QImage.Format.Format_RGB888).copy()
                        pm = QPixmap.fromImage(qimg)
                except Exception:
                    pass

        if pm is not None and not pm.isNull():
            pix_item = QGraphicsPixmapItem(pm, self)
            pix_item.setOpacity(0.85)
            pw = pm.width()
            ph = pm.height()
            if pw > 0 and ph > 0 and len(footprint.polygon) >= 4:
                src_quad = QPolygonF([
                    QPointF(0, 0),
                    QPointF(pw, 0),
                    QPointF(pw, ph),
                    QPointF(0, ph),
                ])
                dst_quad = QPolygonF([QPointF(x, -y) for x, y in footprint.polygon[:4]])
                try:
                    trans = QTransform.quadToQuad(src_quad, dst_quad)
                    pix_item.setTransform(trans)
                except Exception:
                    pass


class PrescriptionZoneGraphicsItem(QGraphicsPolygonItem):
    """Polygon graphics item representing a high-density prescription treatment zone."""

    def __init__(
        self,
        zone: PrescriptionZone,
        parent: QGraphicsItem | None = None,
    ) -> None:
        poly = QPolygonF([QPointF(x, -y) for x, y in zone.polygon])
        super().__init__(poly, parent)
        self.zone = zone

        pen = QPen(QColor(230, 40, 40), 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(230, 40, 40, 75)))
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            f"Зона: {zone.zone_id}\n"
            f"Площадь: {zone.area_m2:.1f} м²\n"
            f"Сорняков: {zone.weed_count} шт\n"
            f"Плотность: {zone.mean_density_per_m2:.2f} шт/м²\n"
            f"Рекомендация: {zone.treatment_recommendation}"
        )


class LayerControlPanel(QWidget):
    """Panel containing layer toggles for interactive map control."""

    layer_toggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._canvas: MapCanvas | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("Слои карты", self)
        title.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        layout.addWidget(title)

        self.cb_mosaic = QCheckBox("Подложка / Мозаика", self)
        self.cb_grid = QCheckBox("Координатная сетка", self)
        self.cb_heatmap = QCheckBox("Тепловая сетка", self)
        self.cb_prescription = QCheckBox("Очаги (Prescription)", self)
        self.cb_points = QCheckBox("Точки сорняков", self)
        self.cb_route = QCheckBox("Маршрут полёта", self)

        self.cb_mosaic.setChecked(True)
        self.cb_grid.setChecked(True)
        self.cb_heatmap.setChecked(True)
        self.cb_prescription.setChecked(True)
        self.cb_points.setChecked(True)
        self.cb_route.setChecked(True)

        for cb in (
            self.cb_mosaic,
            self.cb_grid,
            self.cb_heatmap,
            self.cb_prescription,
            self.cb_points,
            self.cb_route,
        ):
            layout.addWidget(cb)

        self.cb_mosaic.toggled.connect(lambda val: self._on_toggled("mosaic", val))
        self.cb_grid.toggled.connect(lambda val: self._on_toggled("grid", val))
        self.cb_heatmap.toggled.connect(lambda val: self._on_toggled("heatmap", val))
        self.cb_prescription.toggled.connect(lambda val: self._on_toggled("prescription", val))
        self.cb_points.toggled.connect(lambda val: self._on_toggled("points", val))
        self.cb_route.toggled.connect(lambda val: self._on_toggled("route", val))

    def _on_toggled(self, name: str, state: bool) -> None:
        if self._canvas is not None:
            self._canvas.set_layer_visible(name, state)
        self.layer_toggled.emit(name, state)

    def connect_canvas(self, canvas: MapCanvas) -> None:
        """Connect toggle checkboxes directly to target MapCanvas layer visibility."""
        self._canvas = canvas
        self._canvas.set_layer_visible("mosaic", self.cb_mosaic.isChecked())
        self._canvas.set_layer_visible("grid", self.cb_grid.isChecked())
        self._canvas.set_layer_visible("heatmap", self.cb_heatmap.isChecked())
        self._canvas.set_layer_visible("prescription", self.cb_prescription.isChecked())
        self._canvas.set_layer_visible("points", self.cb_points.isChecked())
        self._canvas.set_layer_visible("route", self.cb_route.isChecked())


class MapCanvas(QGraphicsView):
    """Interactive 2D digital map canvas supporting weed point layers, heatmaps, route and drone telemetry."""

    weed_selected = Signal(str)  # weed_id
    zone_selected = Signal(str)  # zone_id
    frame_selected = Signal(str)  # frame_id
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
        self._selected_weed_point: WeedPoint | None = None
        self._class_colors: dict[str, QColor] = {}

        # Layer containers
        self._heatmap_item: QGraphicsPixmapItem | None = None
        self._mosaic_group: QGraphicsItemGroup | None = None
        self._grid_lines_group: QGraphicsItemGroup | None = None
        self._heatmap_group: QGraphicsItemGroup | None = None
        self._prescription_group: QGraphicsItemGroup | None = None
        self._route_group: QGraphicsItemGroup | None = None
        self._points_group: QGraphicsItemGroup | None = None
        self._drone_group: QGraphicsItemGroup | None = None
        self._locked_overlay: QGraphicsItemGroup | None = None

        self._points_list: list[WeedPoint] = []
        self._point_items: dict[str, WeedGraphicsPointItem] = {}
        self._footprint_items: list[MosaicFootprintItem] = []
        self._prescription_items: list[PrescriptionZoneGraphicsItem] = []

        self._setup_view()

    def _setup_view(self) -> None:
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(24, 28, 24)))

        # Create persistent groups for layered rendering
        self._mosaic_group = self._scene.createItemGroup([])
        self._grid_lines_group = self._scene.createItemGroup([])
        self._heatmap_group = self._scene.createItemGroup([])
        self._prescription_group = self._scene.createItemGroup([])
        self._route_group = self._scene.createItemGroup([])
        self._points_group = self._scene.createItemGroup([])
        self._drone_group = self._scene.createItemGroup([])

        # Ensure correct z-stacking
        self._mosaic_group.setZValue(5)
        self._heatmap_group.setZValue(10)
        self._grid_lines_group.setZValue(12)
        self._prescription_group.setZValue(15)
        self._route_group.setZValue(20)
        self._points_group.setZValue(30)
        self._drone_group.setZValue(40)

    # -------------------------------------------------------------------------
    # Public Layer Visibility API
    # -------------------------------------------------------------------------
    def set_layer_visible(self, layer_name: str, visible: bool) -> None:
        """Toggle visibility of a specific map layer."""
        name = layer_name.lower()
        if name in ("mosaic", "footprints"):
            if self._mosaic_group is not None:
                self._mosaic_group.setVisible(visible)
        elif name in ("grid", "grid_lines", "contour"):
            if self._grid_lines_group is not None:
                self._grid_lines_group.setVisible(visible)
        elif name in ("heatmap", "density"):
            if self._heatmap_group is not None:
                self._heatmap_group.setVisible(visible)
        elif name in ("prescription", "zones", "prescription_zones"):
            if self._prescription_group is not None:
                self._prescription_group.setVisible(visible)
        elif name in ("points", "weeds"):
            if self._points_group is not None:
                self._points_group.setVisible(visible)
        elif name in ("route", "trajectory"):
            if self._route_group is not None:
                self._route_group.setVisible(visible)
        elif name in ("drone", "telemetry"):
            if self._drone_group is not None:
                self._drone_group.setVisible(visible)

    def is_layer_visible(self, layer_name: str) -> bool:
        """Return True if specified layer is currently visible."""
        name = layer_name.lower()
        if name in ("mosaic", "footprints"):
            return self._mosaic_group.isVisible() if self._mosaic_group else True
        if name in ("grid", "grid_lines", "contour"):
            return self._grid_lines_group.isVisible() if self._grid_lines_group else True
        if name in ("heatmap", "density"):
            return self._heatmap_group.isVisible() if self._heatmap_group else True
        if name in ("prescription", "zones", "prescription_zones"):
            return self._prescription_group.isVisible() if self._prescription_group else True
        if name in ("points", "weeds"):
            return self._points_group.isVisible() if self._points_group else True
        if name in ("route", "trajectory"):
            return self._route_group.isVisible() if self._route_group else True
        if name in ("drone", "telemetry"):
            return self._drone_group.isVisible() if self._drone_group else True
        return True

    # -------------------------------------------------------------------------
    # Data Layer Management
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
            self._mosaic_group.setVisible(False)
            self._grid_lines_group.setVisible(False)
            self._heatmap_group.setVisible(False)
            self._prescription_group.setVisible(False)
            self._route_group.setVisible(False)
            self._points_group.setVisible(False)
            self._drone_group.setVisible(False)

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
            self._mosaic_group.setVisible(True)
            self._grid_lines_group.setVisible(True)
            self._heatmap_group.setVisible(True)
            self._prescription_group.setVisible(True)
            self._route_group.setVisible(True)
            self._points_group.setVisible(True)
            self._drone_group.setVisible(True)

    def is_locked(self) -> bool:
        """Return True if map canvas is locked."""
        return self._locked

    def set_mosaic_footprints(self, footprints: Sequence[FrameFootprint]) -> None:
        """Display quick-stitch aerial photo footprints and on-demand thumbnails."""
        for it in list(self._mosaic_group.childItems()):
            self._scene.removeItem(it)
        self._footprint_items.clear()

        if not footprints:
            return

        for fp in footprints:
            item = MosaicFootprintItem(fp)
            self._mosaic_group.addToGroup(item)
            self._footprint_items.append(item)

    def set_prescription_zones(self, zones: Sequence[PrescriptionZone]) -> None:
        """Display vector prescription treatment polygons."""
        for it in list(self._prescription_group.childItems()):
            self._scene.removeItem(it)
        self._prescription_items.clear()

        if not zones:
            return

        for zone in zones:
            item = PrescriptionZoneGraphicsItem(zone)
            self._prescription_group.addToGroup(item)
            self._prescription_items.append(item)

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
            sx = p.x
            sy = -p.y

            if p.class_name not in self._class_colors:
                c_idx = len(self._class_colors) % len(CLASS_COLORS)
                self._class_colors[p.class_name] = CLASS_COLORS[c_idx]
            color = self._class_colors[p.class_name]

            item = WeedGraphicsPointItem(point=p, scene_x=sx, scene_y=sy, color=color)
            self._points_group.addToGroup(item)
            self._point_items[p.weed_id] = item

    def set_heatmap(self, grid: HeatmapGrid | None) -> None:
        """Render density heatmap grid as semi-transparent raster overlay."""
        if self._heatmap_item is not None:
            self._scene.removeItem(self._heatmap_item)
            self._heatmap_item = None

        for it in list(self._grid_lines_group.childItems()):
            self._scene.removeItem(it)

        if grid is None or grid.matrix is None:
            return

        matrix = np.asarray(grid.matrix, dtype=np.float32)
        if matrix.size == 0 or matrix.ndim != 2:
            return

        # Flip vertically so that matrix row 0 (southernmost / min_y) maps to the bottom
        # and row (rows - 1) (northernmost / max_y) maps to -bounds.max_y at the top
        matrix = np.flipud(matrix)

        rows, cols = matrix.shape
        max_val = float(np.max(matrix))
        if max_val <= 0:
            norm = np.zeros_like(matrix)
        else:
            norm = matrix / max_val

        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        r = np.clip(np.where(norm < 0.5, 2.0 * norm * 255.0, 255.0), 0, 255).astype(np.uint8)
        g = np.clip(np.where(norm < 0.5, 200.0, 2.0 * (1.0 - norm) * 200.0), 0, 255).astype(np.uint8)
        b = np.full((rows, cols), 20, dtype=np.uint8)
        alpha = np.where(norm > 0.01, 220, 0).astype(np.uint8)

        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = alpha

        rgba = np.ascontiguousarray(rgba)
        qimg = QImage(rgba.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(qimg)

        self._heatmap_item = QGraphicsPixmapItem(pixmap)
        self._heatmap_item.setOpacity(self._heatmap_opacity_pct / 100.0)

        sx = grid.cell_size_m
        sy = grid.cell_size_m
        origin_x, origin_y = grid.origin

        # Top of the raster in world coords is (origin_y + rows * cell_size_m).
        # In Qt scene coordinates (y_scene = -y_world), this maps to:
        top_scene_y = -(origin_y + rows * sy)

        self._heatmap_item.setPos(origin_x, top_scene_y)
        self._heatmap_item.setTransform(self._heatmap_item.transform().scale(sx, sy))
        self._heatmap_group.addToGroup(self._heatmap_item)

        # Draw full coordinate/cell grid in grid layer
        self._draw_grid_overlay(origin_x, origin_y, rows, cols, sx, sy, grid.cell_size_m)

    def _draw_grid_overlay(
        self,
        origin_x: float,
        origin_y: float,
        rows: int,
        cols: int,
        sx: float,
        sy: float,
        cell_size_m: float,
    ) -> None:
        """Render coordinate lines, prominent origin axes, and meter labels onto grid layer."""
        for it in list(self._grid_lines_group.childItems()):
            self._scene.removeItem(it)

        bw = cols * sx
        bh = rows * sy
        top_scene_y = -(origin_y + rows * sy)
        bottom_scene_y = top_scene_y + bh

        # 1. Base cell lines
        grid_path = QPainterPath()
        for c in range(cols + 1):
            gx = origin_x + c * sx
            grid_path.moveTo(gx, top_scene_y)
            grid_path.lineTo(gx, bottom_scene_y)

        for r in range(rows + 1):
            gy = top_scene_y + r * sy
            grid_path.moveTo(origin_x, gy)
            grid_path.lineTo(origin_x + bw, gy)

        grid_item = QGraphicsPathItem(grid_path)
        grid_pen = QPen(QColor(180, 200, 220, 90), 1.0, Qt.PenStyle.SolidLine)
        grid_pen.setCosmetic(True)  # Keeps constant thin 1px size regardless of zoom
        grid_item.setPen(grid_pen)
        self._grid_lines_group.addToGroup(grid_item)

        # 2. Outer survey boundary accent
        border_rect = QGraphicsRectItem(origin_x, top_scene_y, bw, bh)
        b_pen = QPen(QColor(80, 160, 240, 180), 1.0, Qt.PenStyle.DashLine)
        b_pen.setCosmetic(True)
        border_rect.setPen(b_pen)
        self._grid_lines_group.addToGroup(border_rect)

        # 3. Metric axis labels with constant screen font
        max_span = max(bw, bh)
        if max_span <= 6.0:
            step_m = 0.5
        elif max_span <= 20.0:
            step_m = 1.0
        elif max_span <= 50.0:
            step_m = 2.0
        else:
            step_m = 5.0

        label_font = QFont("sans-serif", 8)
        text_brush = QBrush(QColor(220, 235, 255, 220))

        # X-axis meter marks along the bottom
        for c in range(cols + 1):
            gx = origin_x + c * sx
            rem = abs(gx) % step_m
            is_step = (rem < 1e-3) or (abs(rem - step_m) < 1e-3) or (c == 0) or (c == cols)
            if is_step:
                lbl_text = f"{gx:+.1f} м" if abs(gx) > 0.05 else "0 м"
                txt_item = QGraphicsSimpleTextItem(lbl_text)
                txt_item.setFont(label_font)
                txt_item.setBrush(text_brush)
                txt_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                txt_item.setPos(gx, bottom_scene_y + 0.05)
                self._grid_lines_group.addToGroup(txt_item)

        # Y-axis meter marks along the left
        for r in range(rows + 1):
            gy = top_scene_y + r * sy
            world_y = -gy
            rem = abs(world_y) % step_m
            is_step = (rem < 1e-3) or (abs(rem - step_m) < 1e-3) or (r == 0) or (r == rows)
            if is_step:
                lbl_text = f"{world_y:+.1f} м" if abs(world_y) > 0.05 else "0 м"
                txt_item = QGraphicsSimpleTextItem(lbl_text)
                txt_item.setFont(label_font)
                txt_item.setBrush(text_brush)
                txt_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                txt_item.setPos(origin_x - (sx * 0.3 if sx > 1 else 0.25), gy - 0.05)
                self._grid_lines_group.addToGroup(txt_item)

        # 5. Header / North compass indicator at top-left
        header_text = f"↑ Север (Y+)  |  Шаг сетки: {cell_size_m:.1f} м"
        header_item = QGraphicsSimpleTextItem(header_text)
        header_item.setFont(QFont("sans-serif", 8, QFont.Weight.Bold))
        header_item.setBrush(QBrush(QColor(90, 210, 255, 230)))
        header_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        header_item.setPos(origin_x, top_scene_y - 0.2)
        self._grid_lines_group.addToGroup(header_item)

        # 6. Origin badge
        if origin_x <= 0.0 <= origin_x + bw and origin_y <= 0.0 <= (origin_y + bh):
            origin_badge = QGraphicsSimpleTextItem("⦿ (0, 0)")
            origin_badge.setFont(QFont("sans-serif", 8, QFont.Weight.Bold))
            origin_badge.setBrush(QBrush(QColor(70, 230, 130, 240)))
            origin_badge.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            origin_badge.setPos(0.04, -0.04)
            self._grid_lines_group.addToGroup(origin_badge)

    @Slot(int)
    def set_heatmap_opacity(self, percent: int) -> None:
        """Update opacity of the heatmap overlay (0 to 100%)."""
        self._heatmap_opacity_pct = max(0, min(100, int(percent)))
        if self._heatmap_item is not None:
            self._heatmap_item.setOpacity(self._heatmap_opacity_pct / 100.0)

    def set_route(self, waypoints: Sequence[Waypoint]) -> None:
        """Render serpentine flight route waypoints and connecting trajectory."""
        self.clear_route()
        if not waypoints:
            return

        path = QPainterPath()
        p0 = waypoints[0]
        path.moveTo(p0.x, -p0.y)

        for wp in waypoints[1:]:
            path.lineTo(wp.x, -wp.y)

        path_item = QGraphicsPathItem(path)
        pen = QPen(QColor(80, 160, 255), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        path_item.setPen(pen)
        self._route_group.addToGroup(path_item)

        diamond = QPolygonF([QPointF(0, -4), QPointF(4, 0), QPointF(0, 4), QPointF(-4, 0)])
        font = QFont("sans-serif", 8)

        for idx, wp in enumerate(waypoints):
            station = QGraphicsPolygonItem(diamond)
            station.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            station.setPos(wp.x, -wp.y)
            n_pen = QPen(QColor(255, 255, 255), 1.0)
            n_pen.setCosmetic(True)
            station.setPen(n_pen)
            station.setBrush(QBrush(QColor(30, 144, 255)))
            self._route_group.addToGroup(station)

            if idx % max(1, len(waypoints) // 10) == 0 or idx == len(waypoints) - 1:
                label = QGraphicsSimpleTextItem(f"{idx}", station)
                label.setBrush(QBrush(Qt.GlobalColor.white))
                label.setFont(font)
                label.setPos(6.0, -6.0)

    def clear_route(self) -> None:
        """Clear all route trajectories and waypoint stations."""
        for it in list(self._route_group.childItems()):
            self._scene.removeItem(it)

    def set_drone_pose(self, pose: DronePose | None) -> None:
        """Render drone current position marker and optical heading vector."""
        for it in list(self._drone_group.childItems()):
            self._scene.removeItem(it)

        if pose is None:
            return

        cx = pose.x
        cy = -pose.y

        yaw_rad = math.radians(pose.yaw)
        size = 8.0

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
        poly_item.setBrush(QBrush(QColor(255, 215, 0)))
        self._drone_group.addToGroup(poly_item)

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

    def selected_weed_point(self) -> WeedPoint | None:
        """Return the most recently selected WeedPoint instance."""
        return self._selected_weed_point

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, WeedGraphicsPointItem):
            self._selected_weed_point = item.point
            self.weed_selected.emit(item.point.weed_id)
        elif isinstance(item, PrescriptionZoneGraphicsItem):
            self.zone_selected.emit(item.zone.zone_id)
        elif isinstance(item, QGraphicsSimpleTextItem) and isinstance(item.parentItem(), PrescriptionZoneGraphicsItem):
            self.zone_selected.emit(item.parentItem().zone.zone_id)
        elif isinstance(item, MosaicFootprintItem):
            self.frame_selected.emit(item.footprint.frame_id)
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
            padded = rect.adjusted(-10, -10, 10, 10)
            self._scene.setSceneRect(padded)
            self.fitInView(padded, Qt.AspectRatioMode.KeepAspectRatio)
            self._current_zoom = self.transform().m11()
            self.zoom_changed.emit(self._current_zoom)
