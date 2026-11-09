"""Visual tests verifying map canvas rendering fixes, route cleaning, and transformations."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
)

from geo_weed_station.controllers.app_controller import AppController
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import FrameFootprint, WeedPoint
from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.ui.widgets.map_canvas import MapCanvas, MosaicFootprintItem


def test_weed_point_ignores_transformations(qtbot):
    canvas = MapCanvas()
    qtbot.addWidget(canvas)

    pt = WeedPoint(
        weed_id="w1",
        class_name="Ambrosia",
        confidence=0.9,
        x=10.0,
        y=20.0,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="valid_000.jpg",
    )
    canvas.set_points([pt])
    item = canvas._point_items["w1"]
    assert item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
    assert item.pos() == QPointF(10.0, -20.0)


def test_route_waypoints_fixed_screen_size_no_duplicates(qtbot):
    canvas = MapCanvas()
    qtbot.addWidget(canvas)

    waypoints = [
        Waypoint(x=0.0, y=0.0, z=15.0, yaw=0.0),
        Waypoint(x=10.0, y=5.0, z=15.0, yaw=0.0),
    ]
    canvas.set_route(waypoints)

    station_items = [
        it for it in canvas._route_group.childItems() if isinstance(it, QGraphicsPolygonItem)
    ]
    assert len(station_items) == 2


def test_mosaic_footprint_on_demand_thumbnail_and_no_center_dot():
    sample_img = Path("GeoWeedStation/workdir/sample_data/images/valid_000.jpg")
    fp = FrameFootprint(
        frame_id="valid_000",
        source_path=str(sample_img),
        center_x=0.0,
        center_y=0.0,
        width_m=20.0,
        height_m=15.0,
        altitude_m=20.0,
        yaw_deg=0.0,
        polygon=[(-10.0, 7.5), (10.0, 7.5), (10.0, -7.5), (-10.0, -7.5)],
    )
    item = MosaicFootprintItem(fp, pixmap=None)
    pix_items = [it for it in item.childItems() if isinstance(it, QGraphicsPixmapItem)]
    assert len(pix_items) == 1
    assert not pix_items[0].pixmap().isNull()
    dot_items = [it for it in item.childItems() if isinstance(it, QGraphicsEllipseItem)]
    assert len(dot_items) == 0


def test_clear_all_and_reload_folder_clears_previous_route(qtbot):
    app_ctrl = AppController()
    offline_ctrl = app_ctrl.offline_controller
    qtbot.addWidget(app_ctrl.window)

    wps = [
        Waypoint(x=0.0, y=0.0, z=10.0),
        Waypoint(x=10.0, y=10.0, z=10.0),
    ]
    offline_ctrl.map_canvas.set_route(wps)
    assert len(offline_ctrl.map_canvas._route_group.childItems()) > 0

    offline_ctrl._on_file_list_cleared()
    assert len(offline_ctrl.map_canvas._route_group.childItems()) == 0
    assert len(offline_ctrl.map_canvas._points_group.childItems()) == 0


def test_mosaic_footprint_quad_alignment_with_yaw():
    from geo_weed_station.core.map_engine.mosaic import compute_frame_footprint
    from geo_weed_station.core.map_engine.projector import NadirProjector
    from geo_weed_station.core.source_factory import load_image_payload
    from geo_weed_station.schemas.entities import BBox, DetectedWeed

    img_p = Path("GeoWeedStation/workdir/sample_data/images/valid_004.jpg")
    payload = load_image_payload(img_p)
    det = DetectedWeed(
        weed_id="w_test",
        class_id=0,
        class_name="Ambrosia",
        confidence=0.95,
        bbox=BBox(x=0.1875, y=0.208333, w=0.1, h=0.1),
    )
    payload = payload.model_copy(update={"detections": [det]})
    fp = compute_frame_footprint(payload)
    assert fp is not None

    item = MosaicFootprintItem(fp)
    pix_items = [it for it in item.childItems() if isinstance(it, QGraphicsPixmapItem)]
    assert len(pix_items) == 1
    pix = pix_items[0]

    pw = pix.pixmap().width()
    ph = pix.pixmap().height()
    norm_cx = det.bbox.x + det.bbox.w / 2.0
    norm_cy = det.bbox.y + det.bbox.h / 2.0

    mapped_pos = pix.transform().map(QPointF(norm_cx * pw, norm_cy * ph))

    projector = NadirProjector()
    proj = projector.project(payload)[0]
    expected_scene_x = proj.world_x
    expected_scene_y = -proj.world_y

    dist = ((mapped_pos.x() - expected_scene_x)**2 + (mapped_pos.y() - expected_scene_y)**2)**0.5
    assert dist < 0.005, f"Discrepancy {dist}m is too large! Pixmap is misaligned with footprint."


def test_heatmap_vertical_orientation_correctness():
    import numpy as np

    from geo_weed_station.schemas.map_models import Bounds, HeatmapGrid

    bounds = Bounds(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0)
    matrix = np.zeros((10, 10), dtype=np.float32)
    # Put maximum density at row 9 (which corresponds to north / max_y in HeatmapGrid)
    matrix[9, 5] = 100.0

    grid = HeatmapGrid(origin=(0.0, 0.0), cell_size_m=1.0, bounds=bounds, matrix=matrix)
    canvas = MapCanvas()
    canvas.set_heatmap(grid)

    assert canvas._heatmap_item is not None
    pm = canvas._heatmap_item.pixmap()
    qimg = pm.toImage()

    # In Qt QImage, top row (y=0) corresponds to -max_y (-10.0).
    # Since matrix[9, 5] is northernmost, the high red value should be at y=0, NOT y=9!
    top_pixel = qimg.pixelColor(5, 0)
    bottom_pixel = qimg.pixelColor(5, 9)

    assert top_pixel.red() > 200, "Northern high-density spot was not placed at top of QImage!"
    assert bottom_pixel.alpha() == 0, "Southern zero-density spot was erroneously given alpha!"


def test_heatmap_cell_encloses_weed_point_with_zero_shift():
    from geo_weed_station.core.map_engine.heatmap import CountHeatmap
    from geo_weed_station.schemas.entities import CoordinateSystem
    from geo_weed_station.schemas.map_models import Bounds, WeedPoint

    # Fractional boundary similar to actual flight survey
    bounds = Bounds(min_x=-9.533, min_y=-7.400, max_x=31.651, max_y=23.388)
    cell_m = 0.5
    points = [
        WeedPoint(
            weed_id="p1", class_name="w", confidence=0.9, x=0.0, y=0.0,
            coord_system=CoordinateSystem.LOCAL_ARUCO, source_frame="f1"
        ),
        WeedPoint(
            weed_id="p2", class_name="w", confidence=0.9, x=-5.333, y=-3.733,
            coord_system=CoordinateSystem.LOCAL_ARUCO, source_frame="f1"
        ),
        WeedPoint(
            weed_id="p3", class_name="w", confidence=0.9, x=5.0, y=3.0,
            coord_system=CoordinateSystem.LOCAL_ARUCO, source_frame="f2"
        ),
        WeedPoint(
            weed_id="p4", class_name="w", confidence=0.9, x=0.396, y=7.603,
            coord_system=CoordinateSystem.LOCAL_ARUCO, source_frame="f2"
        ),
    ]

    grid = CountHeatmap().compute(points, cell_m=cell_m, bounds=bounds)
    rows, _cols = grid.matrix.shape
    origin_x, origin_y = grid.origin

    top_scene_y = -(origin_y + rows * cell_m)

    for pt in points:
        col = int((pt.x - origin_x) / cell_m)
        row = int((pt.y - origin_y) / cell_m)
        qimg_x = col
        qimg_y = rows - 1 - row

        cell_scene_x = origin_x + qimg_x * cell_m
        cell_scene_y = top_scene_y + qimg_y * cell_m

        pt_scene_x = pt.x
        pt_scene_y = -pt.y

        assert cell_scene_x <= pt_scene_x <= (cell_scene_x + cell_m + 1e-5), (
            f"Point X {pt_scene_x} is outside cell [{cell_scene_x} .. {cell_scene_x + cell_m}]!"
        )
        assert cell_scene_y <= pt_scene_y <= (cell_scene_y + cell_m + 1e-5), (
            f"Point Y {pt_scene_y} is outside cell [{cell_scene_y} .. {cell_scene_y + cell_m}]!"
        )


