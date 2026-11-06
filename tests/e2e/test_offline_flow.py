"""End-to-End tests for complete offline data processing workflows."""

from __future__ import annotations

from pathlib import Path
import pytest
from PySide6.QtCore import QSettings

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.app_controller import AppController
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.source_factory import FolderSource
from geo_weed_station.schemas.entities import BBox, DetectedWeed
from tests.helpers.dataset_factory import make_dataset, make_geotagged_jpeg


class MockOfflineDetector(BaseDetector):
    def __init__(self) -> None:
        self._loaded = False
        self.closed = False
        self._counter = 0

    def load(self) -> None:
        self._loaded = True

    def detect(self, image):
        self._counter += 1
        return [
            DetectedWeed(
                weed_id=f"e2e-{self._counter}",
                class_id=0,
                class_name="Ambrosia",
                confidence=0.89,
                bbox=BBox(x=0.2, y=0.2, w=0.1, h=0.1),
            )
        ]

    def close(self) -> None:
        self.closed = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded


@pytest.fixture
def app_ctrl(tmp_path):
    settings = AppSettings(workdir=tmp_path / "workdir")
    ctrl = AppController(
        settings=settings,
        detector_factory=lambda: MockOfflineDetector(),
    )
    return ctrl, tmp_path


@pytest.mark.e2e
def test_e2e_offline_batch_and_map_generation(app_ctrl, qtbot):
    ctrl, tmp_path = app_ctrl
    data_dir = tmp_path / "survey_dataset"
    make_dataset(data_dir, n_valid=3, n_no_xmp=0, n_broken=0)

    # 1. Populate folder in offline controller
    src = FolderSource(data_dir)
    ctrl.offline_controller._current_source_folder = data_dir
    ctrl.offline_controller.file_list_panel.set_entries(src.entries())

    assert ctrl.offline_controller.file_list_panel.model.rowCount() == 3

    # 2. Run batch processing
    ctrl.offline_controller.start_batch()
    worker = ctrl.offline_controller._batch_worker
    assert worker is not None

    # Wait for completion
    qtbot.waitUntil(lambda: ctrl.offline_controller._batch_worker is None, timeout=5000)

    # 3. Verify points and heatmap populated
    engine = ctrl.map_engine
    assert len(engine.weed_map.points) == 3
    assert engine.statistics().total_weeds == 3
    assert not engine.is_locked()

    # 4. Verify map canvas items
    assert len(ctrl.offline_controller.map_canvas._point_items) == 3
    assert ctrl.offline_controller.map_canvas._heatmap_item is not None


@pytest.mark.e2e
def test_e2e_offline_no_xmp_locks_map(app_ctrl, qtbot):
    ctrl, tmp_path = app_ctrl
    no_xmp_img = tmp_path / "no_xmp.jpg"
    import cv2
    import numpy as np

    cv2.imwrite(str(no_xmp_img), np.full((100, 100, 3), 50, dtype=np.uint8))

    # Ingest unreferenced payload
    from geo_weed_station.schemas.payload import FramePayload

    unref_payload = FramePayload(
        frame_id="no_xmp",
        image_size_px=(100, 100),
        pose=None,
        detections=[],
    )
    ctrl.map_controller.on_frame(unref_payload)

    # Map tab must be locked
    assert ctrl.map_controller.is_locked
    assert ctrl.window.is_map_tab_locked
    assert "Геопривязка отсутствует" in ctrl.window.map_locked_text()
