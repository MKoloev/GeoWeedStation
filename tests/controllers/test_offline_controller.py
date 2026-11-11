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


def test_offline_controller_new_folder_resets_map_and_statistics(offline_setup, qtbot):
    ctrl, map_ctrl, window, tmp_path = offline_setup
    qtbot.addWidget(window)

    from geo_weed_station.core.source_factory import FolderSource
    from tests.helpers.dataset_factory import make_dataset

    # Batch 1: 2 valid images
    batch1_dir = tmp_path / "batch1"
    make_dataset(batch1_dir, n_valid=2, n_no_xmp=0, n_broken=0)

    src1 = FolderSource(batch1_dir)
    ctrl._current_source_folder = batch1_dir
    ctrl.file_list_panel.set_entries(src1.entries())
    ctrl.start_batch()
    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=10000)

    # Statistics after batch 1: total_weeds == 2
    assert len(map_ctrl.engine.weed_map.points) == 2
    assert map_ctrl.engine.statistics().total_weeds == 2
    assert ctrl.map_info_panel.lbl_total_weeds.text() == "2"

    # Batch 2: 1 valid image in a new folder
    batch2_dir = tmp_path / "batch2"
    make_dataset(batch2_dir, n_valid=1, n_no_xmp=0, n_broken=0)

    src2 = FolderSource(batch2_dir)
    # Simulate loading new folder
    ctrl._current_source_folder = batch2_dir
    ctrl._detections_by_path.clear()
    ctrl._map_ctrl.reset()
    ctrl.map_info_panel.clear_selection()
    ctrl.file_list_panel.set_entries(src2.entries())

    # Map and statistics must be reset to 0
    assert len(map_ctrl.engine.weed_map.points) == 0
    assert map_ctrl.engine.statistics().total_weeds == 0
    assert ctrl.map_info_panel.lbl_total_weeds.text() == "0"

    # Process new batch
    ctrl.start_batch()
    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=10000)

    # Statistics after batch 2 must reflect ONLY batch 2 (total_weeds == 1, NOT 3)
    assert len(map_ctrl.engine.weed_map.points) == 1
    assert map_ctrl.engine.statistics().total_weeds == 1
    assert ctrl.map_info_panel.lbl_total_weeds.text() == "1"


def test_offline_controller_single_image_pipeline_processes_on_button(offline_setup, qtbot):
    ctrl, map_ctrl, window, tmp_path = offline_setup
    qtbot.addWidget(window)

    from tests.helpers.make_media import make_real_jpeg
    from tests.helpers.make_xmp_image import SAMPLE_XMP_LOCAL_ARUCO
    from geo_weed_station.schemas.session import FileEntry, ProcessingStatus

    single_img = tmp_path / "single_test.jpg"
    make_real_jpeg(single_img, SAMPLE_XMP_LOCAL_ARUCO)

    entry = FileEntry(path=str(single_img), size_bytes=single_img.stat().st_size, has_xmp=True)
    ctrl.file_list_panel.set_entries([entry])
    ctrl._load_entry_to_viewer(entry)

    # 1. Before clicking process: not detected yet, status is PENDING
    assert ctrl.file_list_panel.model.rowCount() == 1
    assert ctrl.file_list_panel.model.index(0, 2).data() == "Ожидание"
    assert ctrl.file_list_panel.model.index(0, 4).data() == "-"
    assert len(ctrl.image_viewer._detection_items) == 0

    # 2. Click process
    ctrl.start_batch()
    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=10000)

    # 3. After process: status is DONE, weeds count is 1, detections overlay visible
    assert ctrl.file_list_panel.model.index(0, 2).data() == "Готово"
    assert ctrl.file_list_panel.model.index(0, 4).data() == "1"
    assert len(ctrl.image_viewer._detection_items) == 1
    assert map_ctrl.engine.statistics().total_weeds == 1


def test_offline_controller_open_image_filter_and_load(offline_setup, qtbot, monkeypatch):
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QFileDialog

    ctrl, map_ctrl, win, tmp_path = offline_setup
    qtbot.addWidget(win)

    img_jpg = tmp_path / "test_frame.jpg"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_jpg), arr)

    captured_filters = []

    def mock_get_open_file(parent, caption, start_dir, filter_str):
        captured_filters.append(filter_str)
        return str(img_jpg), filter_str

    monkeypatch.setattr(QFileDialog, "getOpenFileName", mock_get_open_file)

    ctrl.open_image()

    assert len(captured_filters) == 1
    f_str = captured_filters[0]
    assert "*.png" in f_str
    assert "*.jpg" in f_str
    assert "*.JPG" in f_str
    assert "*.PNG" in f_str
    assert "PNG" in f_str
    assert "JPEG" in f_str or "JPG" in f_str
    assert len(ctrl.file_list_panel.model.entries) == 1
    assert ctrl.file_list_panel.model.entries[0].path == str(img_jpg)


def test_offline_controller_open_video_and_reopen(offline_setup, qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from tests.helpers.make_media import make_mp4

    ctrl, _map_ctrl, win, tmp_path = offline_setup
    qtbot.addWidget(win)

    vid1 = make_mp4(tmp_path / "v1.mp4", n_frames=5, fps=25)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(vid1), "Videos (*.mp4)"))

    ctrl.open_video()
    assert ctrl._video_worker is not None
    assert "v1.mp4" in win.status_bar_panel.label_message.text()

    # Re-open the same or another video
    ctrl.open_video()
    assert ctrl._video_worker is not None
    assert "v1.mp4" in win.status_bar_panel.label_message.text()

    if ctrl._video_worker:
        ctrl._video_worker.request_stop()


def test_offline_controller_save_annotations_when_empty(offline_setup, qtbot):
    ctrl, _map_ctrl, win, tmp_path = offline_setup
    qtbot.addWidget(win)

    ctrl.save_annotations()
    assert "Нет разметки для сохранения" in win.status_bar_panel.label_message.text()
    assert not (tmp_path / "annotations.json").exists()


def test_offline_controller_save_and_preload_annotations(offline_setup, qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from tests.helpers.make_media import make_real_jpeg
    from tests.helpers.make_xmp_image import SAMPLE_XMP_LOCAL_ARUCO

    ctrl, map_ctrl, win, tmp_path = offline_setup
    qtbot.addWidget(win)

    folder = tmp_path / "test_survey"
    folder.mkdir()
    img_path = folder / "frame_001.jpg"
    make_real_jpeg(img_path, SAMPLE_XMP_LOCAL_ARUCO)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(folder))

    # 1. Open folder
    ctrl.open_folder()
    assert len(ctrl.file_list_panel.model.entries) == 1
    # Initially no annotations file, no detections
    assert (folder / "annotations.json").exists() is False
    assert len(ctrl._detections_by_path) == 0

    # 2. Run batch processing
    ctrl.start_batch()
    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=10000)

    # After batch: detections exist in memory, BUT annotations.json is NOT saved yet
    assert len(ctrl._detections_by_path) > 0
    assert (folder / "annotations.json").exists() is False  # Saving ONLY on button click!

    # 3. User clicks save annotations
    ctrl.save_annotations()
    annot_file = folder / "annotations.json"
    assert annot_file.is_file()
    assert "Разметка успешно сохранена" in win.status_bar_panel.label_message.text()

    # 4. Clear and re-open folder: should automatically preload the annotations!
    ctrl.file_list_panel.clear()
    assert len(ctrl._detections_by_path) == 0

    ctrl.open_folder()
    # Verify preloaded state
    assert len(ctrl._detections_by_path) > 0
    assert ctrl.file_list_panel.model.index(0, 2).data() == "Готово"
    assert ctrl.file_list_panel.model.index(0, 4).data() == "1"
    assert "Подгружена разметка" in win.status_bar_panel.label_message.text()
    assert len(ctrl.image_viewer._detection_items) == 1

    # 5. User can still re-run batch processing!
    ctrl.start_batch()
    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=10000)
    assert ctrl.file_list_panel.model.index(0, 2).data() == "Готово"


def test_offline_controller_status_fidelity_on_reload(offline_setup, qtbot, monkeypatch):
    """Verify that when saving and reloading, images with NO_METADATA and 0 detections keep their exact statuses."""
    from PySide6.QtWidgets import QFileDialog
    from tests.helpers.make_media import make_real_jpeg
    from tests.helpers.make_xmp_image import SAMPLE_XMP_LOCAL_ARUCO

    ctrl, _map_ctrl, win, tmp_path = offline_setup
    qtbot.addWidget(win)

    folder = tmp_path / "mixed_survey"
    folder.mkdir()

    # Image 1: with XMP
    img1 = folder / "frame_001.jpg"
    make_real_jpeg(img1, SAMPLE_XMP_LOCAL_ARUCO)

    # Image 2: plain JPEG without XMP
    img2 = folder / "frame_002.jpg"
    make_real_jpeg(img2, None)

    # Image 3: with XMP but 0 weeds
    img3 = folder / "frame_003.jpg"
    make_real_jpeg(img3, SAMPLE_XMP_LOCAL_ARUCO)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(folder))

    ctrl.open_folder()
    assert len(ctrl.file_list_panel.model.entries) == 3

    # Provide custom detections:
    # frame_001 has 1 weed, frame_002 has 0 weeds, frame_003 has 0 weeds
    from geo_weed_station.schemas.entities import BBox, DetectedWeed
    from geo_weed_station.schemas.session import ProcessingStatus

    weed = DetectedWeed(
        weed_id="w1",
        class_id=0,
        class_name="weed",
        confidence=0.9,
        bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
        world_x=1.0,
        world_y=2.0,
    )
    ctrl._detections_by_path[str(img1)] = [weed]
    ctrl._detections_by_path[str(img2)] = []
    ctrl._detections_by_path[str(img3)] = []

    # Update statuses as if worker processed them:
    # frame_001: DONE (has XMP, 1 weed)
    # frame_002: NO_METADATA (no XMP, 0 weeds)
    # frame_003: DONE (has XMP, 0 weeds)
    ctrl.file_list_panel.update_status(0, ProcessingStatus.DONE)
    ctrl.file_list_panel.update_entry_metadata(0, has_xmp=True, detections_count=1)

    ctrl.file_list_panel.update_status(1, ProcessingStatus.NO_METADATA)
    ctrl.file_list_panel.update_entry_metadata(1, has_xmp=False, detections_count=0)

    ctrl.file_list_panel.update_status(2, ProcessingStatus.DONE)
    ctrl.file_list_panel.update_entry_metadata(2, has_xmp=True, detections_count=0)

    assert ctrl.file_list_panel.model.index(0, 2).data() == "Готово"
    assert ctrl.file_list_panel.model.index(0, 4).data() == "1"
    assert ctrl.file_list_panel.model.index(1, 2).data() == "Без XMP"
    assert ctrl.file_list_panel.model.index(1, 4).data() == "0"
    assert ctrl.file_list_panel.model.index(2, 2).data() == "Готово"
    assert ctrl.file_list_panel.model.index(2, 4).data() == "0"

    # Save annotations
    ctrl.save_annotations()
    assert (folder / "annotations.json").is_file()

    # Clear and re-open
    ctrl.file_list_panel.clear()
    ctrl._detections_by_path.clear()
    assert len(ctrl.file_list_panel.model.entries) == 0

    ctrl.open_folder()
    assert len(ctrl.file_list_panel.model.entries) == 3

    # Status of frame_001 MUST still be "Готово" and count "1"
    assert ctrl.file_list_panel.model.index(0, 2).data() == "Готово"
    assert ctrl.file_list_panel.model.index(0, 4).data() == "1"

    # Status of frame_002 MUST still be "Без XMP" and count "0"
    assert ctrl.file_list_panel.model.index(1, 2).data() == "Без XMP"
    assert ctrl.file_list_panel.model.index(1, 4).data() == "0"

    # Status of frame_003 MUST still be "Готово" and count "0"
    assert ctrl.file_list_panel.model.index(2, 2).data() == "Готово"
    assert ctrl.file_list_panel.model.index(2, 4).data() == "0"

    # Crucial check: on the map, BOTH frame_001 (with weed) and frame_003 (without weeds) MUST be rendered!
    # Frame footprints (footprint_items) for georeferenced images must exist on map
    assert len(ctrl.map_canvas._footprint_items) == 2
    # Only frame_001 had weed points
    assert len(ctrl.map_canvas._points_list) == 1





