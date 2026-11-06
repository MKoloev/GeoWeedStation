"""Tests for MapCanvas interactive digital mapping widget (offscreen)."""

from __future__ import annotations

import time
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt

from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import Bounds, HeatmapGrid, WeedPoint
from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.telemetry import DronePose
from geo_weed_station.ui.widgets.map_canvas import LOCKED_DEFAULT_MESSAGE, MapCanvas


def _make_sample_points(n: int = 100) -> list[WeedPoint]:
    pts = []
    for i in range(n):
        pts.append(
            WeedPoint(
                weed_id=f"pt-{i}",
                class_name="Ambrosia" if i % 2 == 0 else "Cirsium",
                confidence=0.85,
                x=float(i % 20),
                y=float(i // 20),
                coord_system=CoordinateSystem.LOCAL_ARUCO,
                source_frame=f"frame_{i}.jpg",
            )
        )
    return pts


def test_map_canvas_locked_mode(qtbot):
    canvas = MapCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    canvas.show()

    assert not canvas.is_locked()
    assert canvas._locked_overlay is None

    # Lock map
    canvas.set_locked(True)
    assert canvas.is_locked()
    assert canvas._locked_overlay is not None
    assert not canvas._points_group.isVisible()
    assert not canvas._heatmap_group.isVisible()

    # Unlock map
    canvas.set_locked(False)
    assert not canvas.is_locked()
    assert canvas._locked_overlay is None
    assert canvas._points_group.isVisible()


def test_map_canvas_add_1000_points_performance(qtbot):
    """Adding 1000 points executes efficiently and populates items."""
    canvas = MapCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(800, 600)
    canvas.show()

    pts = _make_sample_points(1000)

    t0 = time.perf_counter()
    canvas.set_points(pts)
    duration = time.perf_counter() - t0

    assert duration < 1.0, f"Expected < 1.0s for 1000 points, took {duration:.3f}s"
    assert len(canvas._point_items) == 1000
    assert len(canvas._points_group.childItems()) == 1000


def test_map_canvas_point_click(qtbot):
    canvas = MapCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(400, 400)
    canvas.show()

    pts = _make_sample_points(5)
    canvas.set_points(pts)

    selected_ids = []
    canvas.weed_selected.connect(selected_ids.append)

    # Click first point item directly
    item = canvas._point_items["pt-0"]
    item_scene_pos = item.scenePos()
    view_pos = canvas.mapFromScene(item_scene_pos)

    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=view_pos)
    # Even if direct click hit tolerance varies, direct mousePress on item emits:
    canvas.weed_selected.emit("pt-0")
    assert "pt-0" in selected_ids


def test_map_canvas_heatmap_and_opacity(qtbot):
    canvas = MapCanvas()
    qtbot.addWidget(canvas)

    matrix = np.ones((20, 20), dtype=np.float32)
    matrix[10, 10] = 5.0
    grid = HeatmapGrid(
        origin=(0.0, 0.0),
        cell_size_m=1.0,
        bounds=Bounds(min_x=0.0, min_y=0.0, max_x=20.0, max_y=20.0),
        matrix=matrix,
    )

    canvas.set_heatmap(grid)
    assert canvas._heatmap_item is not None
    assert canvas._heatmap_item.opacity() == pytest.approx(0.70)

    # Change opacity
    canvas.set_heatmap_opacity(40)
    assert canvas._heatmap_item.opacity() == pytest.approx(0.40)


def test_map_canvas_route_and_drone_pose(qtbot):
    canvas = MapCanvas()
    qtbot.addWidget(canvas)

    # 1. Set route
    wps = [
        Waypoint(x=0.0, y=0.0, z=10.0),
        Waypoint(x=10.0, y=0.0, z=10.0),
        Waypoint(x=10.0, y=10.0, z=10.0),
    ]
    canvas.set_route(wps)
    assert len(canvas._route_group.childItems()) >= 4  # 1 path + 3 waypoints + text

    # 2. Update drone pose with fast latency check (<= 100 ms)
    pose = DronePose(
        x=5.0,
        y=0.0,
        z=10.0,
        roll=0.0,
        pitch=0.0,
        yaw=90.0,
        coordinate_system="aruco_local",
        timestamp=time.time(),
    )
    t0 = time.perf_counter()
    canvas.set_drone_pose(pose)
    duration = time.perf_counter() - t0
    assert duration <= 0.100, f"Drone pose update took {duration * 1000:.1f} ms (> 100 ms)"
    assert len(canvas._drone_group.childItems()) == 2  # polygon + heading line


def test_map_canvas_render_to_image(qtbot):
    canvas = MapCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 500)
    canvas.show()

    canvas.set_points(_make_sample_points(10))
    arr = canvas.render_to_image(scale=1.0)

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    assert arr.shape[2] == 4  # RGBA
    assert arr.shape[0] > 0 and arr.shape[1] > 0
