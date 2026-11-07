"""Tests for OfflineController orchestrating file viewers, batch workers, and map engine."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from geo_weed_station.config.settings import AppSettings, MapSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.controllers.offline_controller import OfflineController
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.schemas.entities import BBox, DetectedWeed
from geo_weed_station.ui.main_window import MainWindow
from geo_weed_station.ui.widgets.detection_filter_panel import DetectionFilterState


class DummyTestDetector(BaseDetector):
    def __init__(self):
        self._loaded = False
        self.closed = False

    def load(self) -> None:
        self._loaded = True

    def detect(self, image: np.ndarray) -> list[DetectedWeed]:
        return [
            DetectedWeed(
                weed_id="dummy-1",
                class_id=0,
                class_name="Ambrosia",
                confidence=0.92,
                bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
            )
        ]

    def close(self) -> None:
        self.closed = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded


@pytest.fixture
def offline_setup(tmp_path):
    settings = AppSettings(
        map=MapSettings(heatmap_cell_m=1.0),
        workdir=tmp_path / "workdir",
    )
    engine = MapEngine(projector=NadirProjector(), settings=settings.map)
    map_ctrl = MapController(engine=engine)
    qset = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    window = MainWindow(qsettings=qset)

    ctrl = OfflineController(
        window=window,
        map_ctrl=map_ctrl,
        settings=settings,
        detector_factory=lambda: DummyTestDetector(),
    )
    return ctrl, map_ctrl, window, tmp_path


def test_offline_controller_initialization(offline_setup, qtbot):
    ctrl, map_ctrl, window, _ = offline_setup
    qtbot.addWidget(window)

    assert ctrl.image_viewer is not None
    assert ctrl.video_player is not None
    assert ctrl.map_canvas is not None
    assert ctrl.file_list_panel is not None
    assert ctrl.filter_panel is not None


def test_offline_controller_filter_propagation(offline_setup, qtbot):
    ctrl, _map_ctrl, _window, _ = offline_setup

    state = DetectionFilterState(show_boxes=False, min_confidence=0.75)
    ctrl.on_filter_changed(state)

    assert ctrl.image_viewer._filter_state.show_boxes is False
    assert ctrl.image_viewer._filter_state.min_confidence == 0.75
    assert ctrl.video_player.canvas._filter_state.show_boxes is False


def test_offline_controller_cancel(offline_setup, qtbot):
    ctrl, _map_ctrl, window, _ = offline_setup
    ctrl.cancel()
    # Does not crash when no worker is running
    assert window.status_bar_panel.cancel_button.isEnabled() is False


def test_offline_controller_model_switching(offline_setup, qtbot):
    ctrl, _map_ctrl, window, tmp_path = offline_setup

    m1 = tmp_path / "custom_weed.onnx"
    m1.touch()
    c1 = tmp_path / "custom_weed.json"
    c1.write_text('{"0": "Corn", "1": "Sunflower"}', encoding="utf-8")

    ctrl._on_model_changed(m1)

    assert ctrl._settings.detector.model_path == m1
    assert ctrl._settings.detector.classes_path == c1
    assert "Corn" in ctrl.filter_panel._class_names
    assert "Sunflower" in ctrl.filter_panel._class_names
    assert "custom_weed.onnx" in window.status_bar_panel.message_text


def test_offline_controller_batch_and_row_selection_displays_detections(offline_setup, qtbot):
    ctrl, _map_ctrl, window, tmp_path = offline_setup
    qtbot.addWidget(window)

    from geo_weed_station.core.source_factory import FolderSource
    from tests.helpers.dataset_factory import make_dataset

    data_dir = tmp_path / "survey_batch"
    make_dataset(data_dir, n_valid=2, n_no_xmp=0, n_broken=0)

    src = FolderSource(data_dir)
    ctrl._current_source_folder = data_dir
    ctrl.file_list_panel.set_entries(src.entries())

    assert ctrl.file_list_panel.model.rowCount() == 2

    # Start batch processing
    ctrl.start_batch()
    assert ctrl._batch_worker is not None

    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=10000)

    # 1. Check detections cache is populated
    assert len(ctrl._detections_by_path) >= 2

    # 2. Check table column detections count
    det_val_0 = ctrl.file_list_panel.model.index(0, 4).data()
    det_val_1 = ctrl.file_list_panel.model.index(1, 4).data()
    assert det_val_0 == "1"
    assert det_val_1 == "1"

    # 3. Switching between images in table displays detections in viewer
    ctrl.file_list_panel.table_view.selectRow(0)
    assert len(ctrl.image_viewer._detection_items) == 1
    assert ctrl.image_viewer._detection_items[0].weed.class_name == "Ambrosia"

    ctrl.file_list_panel.table_view.selectRow(1)
    assert len(ctrl.image_viewer._detection_items) == 1
    assert ctrl.image_viewer._detection_items[0].weed.class_name == "Ambrosia"

    # 4. Clear resets viewer and cache
    ctrl.file_list_panel.clear()
    assert len(ctrl._detections_by_path) == 0
    assert len(ctrl.image_viewer._detection_items) == 0

