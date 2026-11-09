"""Unit tests for interactive image viewer focus and map canvas layer toggles."""

from __future__ import annotations

import numpy as np

from geo_weed_station.schemas.entities import BBox, CoordinateSystem, DetectedWeed
from geo_weed_station.schemas.map_models import PrescriptionZone, WeedPoint
from geo_weed_station.ui.widgets.image_viewer import ImageViewer
from geo_weed_station.ui.widgets.map_canvas import LayerControlPanel, MapCanvas


def test_image_viewer_focus_on_weed(qtbot):
    viewer = ImageViewer()
    qtbot.addWidget(viewer)

    img = np.ones((800, 1000, 3), dtype=np.uint8) * 100
    viewer.set_image(img)

    det = DetectedWeed(
        weed_id="weed_target",
        class_id=0,
        class_name="Ambrosia",
        confidence=0.95,
        bbox=BBox(x=0.3, y=0.4, w=0.1, h=0.1),
    )
    viewer.set_detections([det])
    assert len(viewer._detection_items) == 1
    item = viewer._detection_items[0]

    viewer.focus_on_weed(weed_id="weed_target", bbox=det.bbox)
    assert item.zValue() == 100


def test_map_canvas_layer_controls_and_clicks(qtbot):
    canvas = MapCanvas()
    panel = LayerControlPanel()
    qtbot.addWidget(canvas)
    qtbot.addWidget(panel)
    panel.connect_canvas(canvas)

    assert canvas.is_layer_visible("mosaic") is True
    panel.cb_mosaic.setChecked(False)
    assert canvas.is_layer_visible("mosaic") is False

    point = WeedPoint(
        weed_id="weed_click_1",
        class_name="weed_dicot",
        confidence=0.9,
        x=5.0,
        y=5.0,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="frame_1",
    )
    canvas.set_points([point])
    selected_weeds = []
    canvas.weed_selected.connect(selected_weeds.append)

    pt_item = canvas._point_items["weed_click_1"]
    canvas.weed_selected.emit(pt_item.point.weed_id)
    assert selected_weeds == ["weed_click_1"]

    zone = PrescriptionZone(
        zone_id="zone_001",
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        area_m2=100.0,
        mean_density_per_m2=3.5,
        weed_count=350,
        dominant_class="Ambrosia",
        treatment_recommendation="Full Spray",
    )
    canvas.set_prescription_zones([zone])
    selected_zones = []
    canvas.zone_selected.connect(selected_zones.append)
    canvas.zone_selected.emit("zone_001")
    assert selected_zones == ["zone_001"]
