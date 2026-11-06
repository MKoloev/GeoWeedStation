"""Tests for ExportController map layer export workflow (offscreen)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog

from geo_weed_station.config.settings import AppSettings, MapSettings
from geo_weed_station.controllers.export_controller import ExportController
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
    SensorOptics,
)
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.ui.dialogs.export_dialog import ExportDialog, ExportFormat, ExportRequest
from geo_weed_station.ui.main_window import MainWindow


@pytest.fixture
def test_setup(tmp_path):
    settings = AppSettings(
        map=MapSettings(heatmap_cell_m=1.0),
        workdir=tmp_path / "workdir",
    )
    engine = MapEngine(projector=NadirProjector(), settings=settings.map)
    map_ctrl = MapController(engine=engine)
    qset = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    window = MainWindow(qsettings=qset)
    export_ctrl = ExportController(window=window, map_ctrl=map_ctrl, settings=settings)
    return export_ctrl, map_ctrl, window, tmp_path


def test_export_blocked_when_map_locked(test_setup, qtbot):
    export_ctrl, map_ctrl, window, _tmp_path = test_setup

    # Lock map
    map_ctrl.on_frame(
        FramePayload(
            frame_id="unref",
            image_size_px=(100, 100),
            pose=None,
            detections=[],
        )
    )
    assert map_ctrl.is_locked

    failed_reasons = []
    export_ctrl.export_failed.connect(failed_reasons.append)

    export_ctrl.export()
    assert len(failed_reasons) == 1
    assert "Геопривязка отсутствует" in failed_reasons[0]


def test_export_successful_execution(test_setup, monkeypatch, qtbot):
    export_ctrl, map_ctrl, window, tmp_path = test_setup

    # Add a valid point
    pose = CameraSpatialPose(
        x=5.0,
        y=5.0,
        altitude_m=10.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
    )
    optics = SensorOptics(focal_length_mm=4.5, sensor_width_mm=6.4, sensor_height_mm=4.8)
    det = DetectedWeed(
        weed_id="w-exp",
        class_id=0,
        class_name="Ambrosia",
        confidence=0.88,
        bbox=BBox(x=0.5, y=0.5, w=0.1, h=0.1),
    )
    payload = FramePayload(
        frame_id="frame_exp",
        image_size_px=(4000, 3000),
        pose=pose,
        optics=optics,
        detections=[det],
    )
    map_ctrl.on_frame(payload)

    export_target = tmp_path / "exported_results"

    # Mock dialog execution to return Accepted with CSV, GeoJSON, PNG
    mock_request = ExportRequest(
        formats={ExportFormat.CSV, ExportFormat.GEOJSON, ExportFormat.PNG},
        target_dir=export_target,
        base_name="my_field",
    )
    monkeypatch.setattr(ExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(ExportDialog, "result_request", lambda self: mock_request)

    completed_dirs = []
    export_ctrl.export_completed.connect(completed_dirs.append)

    export_ctrl.export()

    assert completed_dirs == [str(export_target)]
    assert (export_target / "my_field.csv").exists()
    assert (export_target / "my_field.geojson").exists()
    assert (export_target / "my_field.png").exists()
