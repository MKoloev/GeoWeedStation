"""Non-Functional Requirements (NFR) benchmark tests: latencies, throughput, and memory bounds."""

from __future__ import annotations

import time

import numpy as np
import pytest

from geo_weed_station.core.xmp_parser import parse_xmp
from geo_weed_station.schemas.telemetry import DronePose
from geo_weed_station.ui.widgets.image_viewer import ImageViewer
from geo_weed_station.ui.widgets.map_canvas import MapCanvas
from tests.helpers.dataset_factory import make_geotagged_jpeg


@pytest.mark.nfr
def test_nfr_xmp_parse_latency(tmp_path):
    """NFR: XMP metadata parsing from JPEG must take <= 50 ms per image."""
    img_path = tmp_path / "nfr_test.jpg"
    make_geotagged_jpeg(img_path)

    t0 = time.perf_counter()
    res = parse_xmp(img_path)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert res.has_spatial_data is True
    assert duration_ms <= 50.0, f"XMP parse took {duration_ms:.2f} ms (expected <= 50.0 ms)"


@pytest.mark.nfr
def test_nfr_drone_pose_render_latency(qtbot):
    """NFR: MapCanvas live drone marker update must complete in <= 100 ms."""
    canvas = MapCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    canvas.show()

    pose = DronePose(
        x=25.0,
        y=30.0,
        z=15.0,
        roll=0.0,
        pitch=0.0,
        yaw=60.0,
        coordinate_system="aruco_local",
        timestamp=time.time(),
    )

    t0 = time.perf_counter()
    canvas.set_drone_pose(pose)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert duration_ms <= 100.0, f"Drone pose render took {duration_ms:.2f} ms (expected <= 100.0 ms)"


@pytest.mark.nfr
def test_nfr_large_image_8000x6000_memory_bounds(qtbot):
    """NFR: Ultra-high resolution 8000x6000 aerial images must be safely displayed without OOM."""
    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(800, 600)
    viewer.show()

    large_img = np.zeros((6000, 8000, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    viewer.set_image(large_img)
    duration_s = time.perf_counter() - t0

    assert duration_s <= 2.0, f"Image downscale took {duration_s:.2f} s"
    rect = viewer.scene().sceneRect()
    assert rect.width() <= viewer.MAX_PIXMAP_DIM
    assert rect.height() <= viewer.MAX_PIXMAP_DIM
