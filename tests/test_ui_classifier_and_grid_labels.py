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
    assert len(offline.filter_panel._class_names) == 23
    assert all("S000" not in c and "Фон" not in c for c in offline.filter_panel._class_names)

    # Switch to c07
    offline.filter_panel._combo_classifier.setCurrentIndex(1)
    assert offline._settings.detector.classifier_type == "c07"
    from geo_weed_station.core.onnx_detector import C07_CATEGORIES

    assert offline.filter_panel._class_names == list(C07_CATEGORIES[1:])
    assert "background" not in offline.filter_panel._class_names
    assert "plant" not in offline.filter_panel._class_names

    # Switch back to none
    offline.filter_panel._combo_classifier.setCurrentIndex(0)
    assert offline._settings.detector.classifier_type == "none"
    assert offline._detector_factory().classifier_type == "none"


def test_filter_panel_extra_classes_and_map_report_filtering(qtbot, tmp_path):
    """Verify extra classes appending and class filter propagation to MapCanvas, statistics, and reports."""
    import cv2
    from PySide6.QtCore import QSettings
    from geo_weed_station.controllers.map_controller import MapController
    from geo_weed_station.controllers.offline_controller import OfflineController
    from geo_weed_station.core.map_engine.weed_map import MapEngine
    from geo_weed_station.core.onnx_detector import C07_CATEGORIES
    from geo_weed_station.schemas.session import FileEntry, ProcessingStatus
    from geo_weed_station.ui.main_window import MainWindow

    st = AppSettings(workdir=tmp_path)
    engine = MapEngine(projector=NadirProjector(), settings=st.map)
    map_ctrl = MapController(engine=engine)
    qset = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    window = MainWindow(qsettings=qset)
    qtbot.addWidget(window)
    offline = OfflineController(window=window, map_ctrl=map_ctrl, settings=st)

    # 1. Switch to C07 -> 10 base categories
    offline.filter_panel._combo_classifier.setCurrentIndex(1)
    assert offline.filter_panel._class_names == list(C07_CATEGORIES[1:])

    # 2. Append extra classes from legacy annotations (background and plant must be ignored in C07 mode)
    offline.filter_panel.append_extra_classes(["background", "plant", "Legacy_Amaranthus"])
    assert offline.filter_panel._class_names == list(C07_CATEGORIES[1:]) + ["Legacy_Amaranthus"]

    # 3. Ingest two georeferenced detections of different classes into MapEngine and OfflineController
    img_path = tmp_path / "frame_001.jpg"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
    entry = FileEntry(
        path=str(img_path),
        size_bytes=img_path.stat().st_size,
        status=ProcessingStatus.DONE,
        has_xmp=True,
        detections_count=2,
    )
    offline.file_list_panel.set_entries([entry])

    det_vyunok = DetectedWeed(
        weed_id="w_vyunok",
        class_id=2,
        class_name="Вьюнок",
        confidence=0.85,
        bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1),
        stage_name="всходы",
        class_prob=0.9,
    )
    det_zlak = DetectedWeed(
        weed_id="w_zlak",
        class_id=8,
        class_name="Злаковые",
        confidence=0.80,
        bbox=BBox(x=0.5, y=0.5, w=0.1, h=0.1),
        stage_name="2-4 листа",
        class_prob=0.88,
    )
    det_bg = DetectedWeed(
        weed_id="w_bg",
        class_id=0,
        class_name="background",
        confidence=0.90,
        bbox=BBox(x=0.7, y=0.7, w=0.1, h=0.1),
    )

    pose = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=0.0,
        y=0.0,
        altitude_m=2.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    optics = SensorOptics(
        focal_length_mm=16.0,
        sensor_width_mm=13.2,
        sensor_height_mm=8.8,
        principal_point_px=(500.0, 500.0),
    )
    payload = FramePayload(
        frame_id="frame_001",
        source_path=str(img_path),
        image_size_px=(1000, 1000),
        pose=pose,
        optics=optics,
        detections=[det_vyunok, det_zlak, det_bg],
    )
    offline._on_batch_frame_processed(payload)
    offline._on_map_updated()

    # Background is automatically excluded; 2 real weeds are active
    assert len(offline._map_ctrl.engine.filtered_points()) == 2
    assert offline._map_ctrl.engine.statistics().total_weeds == 2
    assert offline.map_info_panel.lbl_total_weeds.text() == "2"

    # 4. Uncheck "Вьюнок" in filter_panel -> MapCanvas, statistics, and report must exclude "Вьюнок"
    for i in range(offline.filter_panel._list_classes.count()):
        it = offline.filter_panel._list_classes.item(i)
        if it.text() == "Вьюнок":
            it.setCheckState(Qt.CheckState.Unchecked)
            break

    assert "Вьюнок" in offline.filter_panel.current_state().hidden_classes
    assert [p.class_name for p in offline._map_ctrl.engine.filtered_points()] == ["Злаковые"]
    assert offline._map_ctrl.engine.statistics().total_weeds == 1
    assert offline._map_ctrl.engine.statistics().by_class == {"Злаковые": 1}
    assert offline.map_info_panel.lbl_total_weeds.text() == "1"

    report = offline.build_folder_report()
    assert report.total_detections == 1
    assert report.unique_weeds == 1
    assert report.by_class == {"Злаковые": 1}
    assert [d.class_name for d in report.detections] == ["Злаковые"]

    # 5. Clearing file list resets extra classes back to base C07 categories
    offline._on_file_list_cleared()
    assert offline.filter_panel._class_names == list(C07_CATEGORIES[1:])


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
