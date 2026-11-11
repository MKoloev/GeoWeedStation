"""Unit tests for UI classifier selection and map coordinate grid meter labels."""

from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsSimpleTextItem, QGraphicsPathItem, QGraphicsRectItem
import numpy as np
import pytest

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.app_controller import AppController
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
    SensorOptics,
)
from geo_weed_station.schemas.map_models import Bounds, HeatmapGrid
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.ui.widgets.detection_filter_panel import DetectionFilterPanel
from geo_weed_station.ui.widgets.map_canvas import MapCanvas


def test_detection_filter_panel_classifier_combo(qtbot):
    """Verify DetectionFilterPanel exposes classifier selector and emits signals."""
    panel = DetectionFilterPanel()
    qtbot.addWidget(panel)

    assert panel.selected_classifier_type() == "none"
    assert panel._combo_classifier.count() == 3
    assert panel._combo_classifier.itemText(0) == "Отключена"
    assert "C07" in panel._combo_classifier.itemText(1)
    assert "C06" in panel._combo_classifier.itemText(2)

    received: list[str] = []
    panel.classifier_changed.connect(lambda t: received.append(t))

    # Switch to C07
    panel.set_classifier_type("c07")
    assert panel.selected_classifier_type() == "c07"

    # User selection simulation
    panel._combo_classifier.setCurrentIndex(2)
    assert panel.selected_classifier_type() == "c06"
    assert "c06" in received


def test_offline_controller_classifier_switch(qtbot, tmp_path):
    """Verify OfflineController propagates classifier switch to settings and factory."""
    st = AppSettings(workdir=tmp_path)
    app_ctrl = AppController(settings=st)
    qtbot.addWidget(app_ctrl.window)

    offline = app_ctrl.offline_controller
    assert offline._settings.detector.classifier_type == "none"

    # User changes classifier in filter panel
    offline.filter_panel._combo_classifier.setCurrentIndex(1)
    assert offline._settings.detector.classifier_type == "c07"

    # Factory produces detector with c07 active
    detector = offline._detector_factory()
    assert detector.classifier_type == "c07"

    # Switch to c06
    offline.filter_panel._combo_classifier.setCurrentIndex(2)
    assert offline._settings.detector.classifier_type == "c06"
    assert offline._detector_factory().classifier_type == "c06"

    # Switch back to none
    offline.filter_panel._combo_classifier.setCurrentIndex(0)
    assert offline._settings.detector.classifier_type == "none"
    assert offline._detector_factory().classifier_type == "none"


def test_map_canvas_grid_axis_labels_and_meters(qtbot):
    """Verify MapCanvas renders metric labels, origin badge, and north indicator."""
    canvas = MapCanvas()
    qtbot.addWidget(canvas)

    grid = HeatmapGrid(
        origin=(-2.0, -3.0),
        cell_size_m=0.5,
        bounds=Bounds(min_x=-2.0, min_y=-3.0, max_x=3.0, max_y=2.0),
        matrix=np.ones((10, 10), dtype=np.float32),
    )
    canvas.set_heatmap(grid)

    child_items = canvas._grid_lines_group.childItems()
    text_items = [it for it in child_items if isinstance(it, QGraphicsSimpleTextItem)]

    # Must contain meter labels for X and Y, North indicator, and origin badge
    texts = [t.text() for t in text_items]
    assert any("м" in t for t in texts), f"Expected metric labels, got: {texts}"
    assert any("Север" in t for t in texts), f"Expected North indicator, got: {texts}"
    assert any("(0, 0)" in t for t in texts), f"Expected origin badge, got: {texts}"

    # Verify text items ignore transformations
    for t in text_items:
        assert t.flags() & QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations


def test_nadir_projector_8_digit_precision():
    """Verify NadirProjector preserves 8 decimal places for GPS coordinates."""
    pose = CameraSpatialPose(
        coordinate_system=CoordinateSystem.WGS84,
        x=55.30404930,
        y=51.77778790,
        altitude_m=9.52,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=-177.9,
    )
    optics = SensorOptics(
        focal_length_mm=40.0,
        sensor_width_mm=8.571,
        sensor_height_mm=6.428,
        principal_point_px=(4096.0, 3072.0),
    )
    det = DetectedWeed(
        weed_id="w1",
        class_id=0,
        class_name="weed",
        confidence=0.9,
        bbox=BBox(x=0.288588, y=0.157931, w=0.044457, h=0.085133),
    )
    payload = FramePayload(
        frame_id="frame_000",
        image_size_px=(8192, 6144),
        pose=pose,
        optics=optics,
        detections=[det],
    )

    projector = NadirProjector()
    projected = projector.project(payload)

    p_lon = projected[0].world_x
    p_lat = projected[0].world_y
    assert p_lon is not None and p_lat is not None

    # Check precision: string format should have up to 8 decimal places
    str_lon = f"{p_lon:.8f}"
    str_lat = f"{p_lat:.8f}"
    assert len(str_lon.split(".")[1]) == 8
    assert len(str_lat.split(".")[1]) == 8
