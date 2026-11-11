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


def test_export_report_succeeds_even_when_map_locked(tmp_path, monkeypatch, qtbot):
    from geo_weed_station.controllers.offline_controller import OfflineController
    from geo_weed_station.schemas.session import FileEntry, ProcessingStatus

    settings = AppSettings(
        map=MapSettings(heatmap_cell_m=1.0),
        workdir=tmp_path / "workdir",
    )
    engine = MapEngine(projector=NadirProjector(), settings=settings.map)
    map_ctrl = MapController(engine=engine)
    qset = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    window = MainWindow(qsettings=qset)
    offline_ctrl = OfflineController(window=window, map_ctrl=map_ctrl, settings=settings)
    export_ctrl = ExportController(
        window=window,
        map_ctrl=map_ctrl,
        settings=settings,
        offline_ctrl=offline_ctrl,
    )

    # Simulate locked map (no XMP georef), but offline controller has a processed image
    map_ctrl.on_frame(
        FramePayload(frame_id="no_xmp", image_size_px=(100, 100), pose=None, detections=[])
    )
    assert map_ctrl.is_locked

    img_path = tmp_path / "img_01.jpg"
    img_path.write_bytes(b"img")
    entry = FileEntry(
        path=str(img_path),
        size_bytes=123,
        status=ProcessingStatus.NO_METADATA,
        has_xmp=False,
        detections_count=1,
    )
    offline_ctrl.file_list_panel.set_entries([entry])
    offline_ctrl._detections_by_path[str(img_path)] = [
        DetectedWeed(
            weed_id="w1",
            class_id=1,
            class_name="Вьюнок",
            confidence=0.9,
            bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
            stage_name="всходы",
            class_prob=0.88,
        )
    ]

    report_dir = tmp_path / "report_out"
    mock_request = ExportRequest(
        formats={
            ExportFormat.HTML_REPORT,
            ExportFormat.SUMMARY_JSON,
            ExportFormat.FRAMES_CSV,
            ExportFormat.CSV,
            ExportFormat.ZIP_BUNDLE,
        },
        target_dir=report_dir,
        base_name="folder_rep",
    )
    monkeypatch.setattr(ExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(ExportDialog, "result_request", lambda self: mock_request)

    completed_dirs = []
    export_ctrl.export_completed.connect(completed_dirs.append)

    # Trigger via button click on FileListPanel
    offline_ctrl.file_list_panel.btn_export_report.click()

    assert completed_dirs == [str(report_dir)]
    assert (report_dir / "folder_rep.html").is_file()
    assert (report_dir / "folder_rep_report.json").is_file()
    assert (report_dir / "folder_rep_frames.csv").is_file()
    assert (report_dir / "folder_rep.csv").is_file()
    assert (report_dir / "folder_rep_report.zip").is_file()


def test_export_report_creates_separate_report_subfolder_inside_image_folder(tmp_path, monkeypatch, qtbot):
    from geo_weed_station.controllers.offline_controller import OfflineController
    from geo_weed_station.schemas.session import FileEntry, ProcessingStatus

    images_dir = tmp_path / "field_images"
    images_dir.mkdir()
    img_path = images_dir / "img_01.jpg"
    img_path.write_bytes(b"img")

    settings = AppSettings(
        map=MapSettings(heatmap_cell_m=1.0),
        workdir=tmp_path / "workdir",
    )
    engine = MapEngine(projector=NadirProjector(), settings=settings.map)
    map_ctrl = MapController(engine=engine)
    qset = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    window = MainWindow(qsettings=qset)
    offline_ctrl = OfflineController(window=window, map_ctrl=map_ctrl, settings=settings)
    offline_ctrl._current_source_folder = images_dir
    offline_ctrl.file_list_panel.set_entries(
        [
            FileEntry(
                path=str(img_path),
                size_bytes=123,
                status=ProcessingStatus.DONE,
                has_xmp=False,
                detections_count=0,
            )
        ]
    )
    export_ctrl = ExportController(
        window=window,
        map_ctrl=map_ctrl,
        settings=settings,
        offline_ctrl=offline_ctrl,
    )

    assert export_ctrl._default_report_dir() == images_dir / "report"

    # Even if user selects images_dir itself, controller redirects output to images_dir / "report"
    mock_request = ExportRequest(
        formats={ExportFormat.HTML_REPORT, ExportFormat.SUMMARY_JSON},
        target_dir=images_dir,
        base_name="detection_report",
    )
    monkeypatch.setattr(ExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(ExportDialog, "result_request", lambda self: mock_request)

    completed_dirs = []
    export_ctrl.export_completed.connect(completed_dirs.append)
    export_ctrl.export_report()

    expected_subdir = images_dir / "report"
    assert completed_dirs == [str(expected_subdir)]
    assert (expected_subdir / "detection_report.html").is_file()
    assert (expected_subdir / "detection_report_report.json").is_file()

