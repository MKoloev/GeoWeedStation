"""Offline mode controller coordinating file viewers, offline workers, and map engine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.image_io import read_image
from geo_weed_station.core.onnx_detector import (
    OnnxDetector,
    get_models_dir,
    resolve_classes_path,
    scan_available_models,
)
from geo_weed_station.core.source_factory import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    FolderSource,
    load_image_payload,
)
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.schemas.session import FileEntry
from geo_weed_station.ui.app_mode import AppMode
from geo_weed_station.ui.main_window import (
    MAP_LOCKED_TEXT,
    TAB_IMAGE,
    TAB_MAP,
    TAB_VIDEO,
    MainWindow,
)
from geo_weed_station.ui.widgets.detection_filter_panel import (
    DetectionFilterPanel,
    DetectionFilterState,
)
from geo_weed_station.ui.widgets.file_list_panel import FileListPanel
from geo_weed_station.ui.widgets.image_viewer import ImageViewer
from geo_weed_station.ui.widgets.map_canvas import MapCanvas
from geo_weed_station.ui.widgets.map_info_panel import MapInfoPanel
from geo_weed_station.ui.widgets.video_player_widget import VideoPlayerWidget
from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.batch_image_worker import BatchImageWorker
from geo_weed_station.workers.video_worker import VideoWorker

logger = logging.getLogger(__name__)


class OfflineController(QObject):
    """Coordinates offline analysis workflow: single images, image batch directories, and video playback."""

    batch_start_requested = Signal(list, bool)

    def __init__(
        self,
        window: MainWindow,
        map_ctrl: MapController,
        settings: AppSettings,
        detector_factory: Callable[[], BaseDetector] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._map_ctrl = map_ctrl
        self._settings = settings
        self._detector_factory = detector_factory or (lambda: OnnxDetector(settings.detector))

        # Instantiate UI widgets
        self.image_viewer = ImageViewer()
        self.video_player = VideoPlayerWidget()
        self.map_canvas = MapCanvas()
        self.map_info_panel = MapInfoPanel()
        self.file_list_panel = FileListPanel()
        self.filter_panel = DetectionFilterPanel()

        # Active worker references
        self._batch_worker: BatchImageWorker | None = None
        self._video_worker: VideoWorker | None = None
        self._current_source_folder: Path | None = None

        self._embed_widgets_in_ui()
        self._wire_signals()
        self._init_models_in_ui()

    # -------------------------------------------------------------------------
    # Setup & Wiring
    # -------------------------------------------------------------------------
    def _init_models_in_ui(self) -> None:
        """Scan models directory, populate filter panel selector and pre-load classes."""
        active_model = Path(self._settings.detector.model_path)
        models_dir = get_models_dir(active_model)
        available = scan_available_models(models_dir)
        self.filter_panel.set_available_models(available, active=active_model)

        classes_p = resolve_classes_path(active_model, self._settings.detector.classes_path)
        if classes_p.is_file():
            self.filter_panel.load_classes_from_file(classes_p)

    def apply_settings(self, settings: AppSettings) -> None:
        """Dynamically apply updated application settings and refresh model configuration."""
        self._settings = settings
        self._detector_factory = lambda: OnnxDetector(self._settings.detector)
        self._init_models_in_ui()

    def _embed_widgets_in_ui(self) -> None:
        # Install content widgets into central tabs
        self._window.set_tab_widget(TAB_IMAGE, self.image_viewer)
        self._window.set_tab_widget(TAB_VIDEO, self.video_player)
        self._window.set_tab_widget(TAB_MAP, self.map_canvas)

        # Install side panels for offline mode
        self._window.add_mode_panel(AppMode.OFFLINE, self.file_list_panel, "Пакетные файлы")
        self._window.add_mode_panel(AppMode.OFFLINE, self.filter_panel, "Фильтры детекций")
        self._window.add_mode_panel(AppMode.OFFLINE, self.map_info_panel, "Статистика карты")

    def _wire_signals(self) -> None:
        # Window actions
        self._window.open_image_requested.connect(self.open_image)
        self._window.open_folder_requested.connect(self.open_folder)
        self._window.open_video_requested.connect(self.open_video)
        self._window.status_bar_panel.cancel_requested.connect(self.cancel)

        # File list panel
        self.file_list_panel.file_selected.connect(self._on_file_selected)
        self.file_list_panel.process_requested.connect(self.start_batch)

        # Filters & Model selection
        self.filter_panel.filter_changed.connect(self.on_filter_changed)
        self.filter_panel.model_changed.connect(self._on_model_changed)

        # Map synchronization
        self._map_ctrl.map_updated.connect(self._on_map_updated)
        self._map_ctrl.statistics_changed.connect(self.map_info_panel.set_statistics)
        self._map_ctrl.map_locked_changed.connect(self._on_map_locked_changed)
        self.map_canvas.weed_selected.connect(self._on_weed_selected)

    def _on_model_changed(self, model_path: Path) -> None:
        """Handle model switch: resolve classes, reconfigure factory and update UI filter checkboxes."""
        classes_path = resolve_classes_path(model_path, self._settings.detector.classes_path)
        logger.info("Switching detection model to %s with classes %s", model_path.name, classes_path.name)

        updated_det = self._settings.detector.model_copy(
            update={"model_path": model_path, "classes_path": classes_path}
        )
        self._settings = self._settings.model_copy(update={"detector": updated_det})
        self._detector_factory = lambda: OnnxDetector(self._settings.detector)

        if classes_path.is_file():
            self.filter_panel.load_classes_from_file(classes_path)

        self._window.status_bar_panel.set_message(
            f"Модель детекции переключена: {model_path.name} (классы: {classes_path.name})"
        )

    # -------------------------------------------------------------------------
    # Actions: Open single image, folder, video
    # -------------------------------------------------------------------------
    @Slot()
    def open_image(self) -> None:
        """Open a single image file, display in viewer, parse XMP, and run detection."""
        exts = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Открыть изображение", str(self._settings.workdir), f"Images ({exts})"
        )
        if not path:
            return

        file_path = Path(path)
        self._window.status_bar_panel.set_message(f"Чтение {file_path.name}...")
        self._window.central_tabs().setCurrentIndex(self._window.tab_index(TAB_IMAGE))

        try:
            # 1. Load image and XMP metadata
            payload = load_image_payload(file_path)
            if payload.image is not None:
                self.image_viewer.set_image(payload.image)

            # 2. Run inference if detector available
            try:
                detector = self._detector_factory()
                detector.load()
                detections = detector.detect(payload.image) if payload.image is not None else []
                payload = payload.model_copy(update={"detections": detections})
                self.image_viewer.set_detections(detections)

                # Collect class names for filter panel
                class_names = sorted({d.class_name for d in detections})
                if class_names:
                    self.filter_panel.set_classes(class_names)
            except Exception as det_err:
                logger.warning("Local inference failed: %s", det_err)

            # 3. Ingest into MapEngine
            self._map_ctrl.on_frame(payload)
            self._window.status_bar_panel.set_message(f"Файл {file_path.name} успешно загружен")
        except Exception as exc:
            logger.exception("Failed to open image: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка открытия файла: {exc}")

    @Slot()
    def open_folder(self) -> None:
        """Open directory of images and populate file list panel."""
        folder = QFileDialog.getExistingDirectory(
            self._window, "Выбрать каталог снимков", str(self._settings.workdir)
        )
        if not folder:
            return

        folder_path = Path(folder)
        self._current_source_folder = folder_path
        self._window.status_bar_panel.set_message(f"Сканирование каталога {folder_path.name}...")

        try:
            src = FolderSource(folder_path)
            entries = src.entries()
            self.file_list_panel.set_entries(entries)
            self._window.status_bar_panel.set_message(f"Найдено {len(entries)} снимков в каталоге")

            if entries:
                # Preview first file
                self._load_entry_to_viewer(entries[0])
        except Exception as exc:
            logger.exception("Failed to open folder: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка открытия каталога: {exc}")

    @Slot()
    def open_video(self) -> None:
        """Open video file, initialize background VideoWorker and wire playback controls."""
        exts = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Открыть видеофайл", str(self._settings.workdir), f"Videos ({exts})"
        )
        if not path:
            return

        # Switch to video tab
        self._window.central_tabs().setCurrentIndex(self._window.tab_index(TAB_VIDEO))
        self._window.status_bar_panel.set_message(f"Открытие видео {Path(path).name}...")

        # Stop previous video worker if active
        if self._video_worker is not None:
            self._video_worker.request_stop()
            self._video_worker = None

        try:
            worker = VideoWorker(
                detector_factory=self._detector_factory,
                max_inflight=3,
                parent=None,
            )
            self._video_worker = worker

            # Connect signals
            worker.frame_ready.connect(self.video_player.show_frame)
            self.video_player.frame_consumed.connect(worker.frame_consumed)
            worker.position_changed.connect(self.video_player.set_position)
            worker.playback_ended.connect(lambda: self.video_player.set_playing(False))

            self.video_player.play_requested.connect(worker.play)
            self.video_player.pause_requested.connect(worker.pause)
            self.video_player.seek_requested.connect(worker.seek)
            self.video_player.frame_skip_changed.connect(worker.set_frame_skip)

            # Move to worker thread
            thread = start_in_thread(worker, "video_worker_thread")
            self._window.register_thread(thread, worker)

            worker.open(path)
            self._window.status_bar_panel.set_message(f"Видео {Path(path).name} готово к просмотру")
        except Exception as exc:
            logger.exception("Failed to initialize video worker: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка открытия видео: {exc}")

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------
    @Slot()
    def start_batch(self) -> None:
        """Launch background BatchImageWorker processing across loaded folder entries."""
        entries = self.file_list_panel.model.entries
        if not entries:
            self._window.status_bar_panel.set_message("Пакетный список файлов пуст")
            return

        source_folder = self._current_source_folder or Path(entries[0].path).parent
        source = FolderSource(source_folder)

        worker = BatchImageWorker(
            source=source,
            detector_factory=self._detector_factory,
        )
        self._batch_worker = worker

        # Connect lifecycle signals
        worker.progress.connect(self._on_batch_progress)
        worker.frame_processed.connect(self._on_batch_frame_processed)
        worker.entry_status_changed.connect(self.file_list_panel.update_status)
        worker.batch_finished.connect(self._on_batch_finished)

        # Enable cancel button in status bar
        self._window.status_bar_panel.set_cancel_enabled(True)
        self._window.status_bar_panel.set_progress(0)
        self._window.status_bar_panel.set_message("Запуск пакетной обработки снимков...")

        # Run in thread
        thread = start_in_thread(worker, "batch_image_worker_thread")
        self._window.register_thread(thread, worker)

        self.batch_start_requested.connect(worker.process)
        self.batch_start_requested.emit(entries, True)

    @Slot()
    def cancel(self) -> None:
        """Request cooperative stop on active background workers."""
        stopped = False
        if self._batch_worker is not None:
            self._batch_worker.request_stop()
            stopped = True
        if self._video_worker is not None:
            self._video_worker.request_stop()
            stopped = True

        if stopped:
            self._window.status_bar_panel.set_message("Остановка фоновых операций...")
            self._window.status_bar_panel.set_cancel_enabled(False)

    @Slot(DetectionFilterState)
    def on_filter_changed(self, state: DetectionFilterState) -> None:
        """Propagate detection display filter state to active viewers."""
        self.image_viewer.apply_filter(state)
        self.video_player.apply_filter(state)

    # -------------------------------------------------------------------------
    # Internal Handlers
    # -------------------------------------------------------------------------
    def _on_file_selected(self, row: int) -> None:
        entry = self.file_list_panel.model.entry_at(row)
        if entry is not None:
            self._load_entry_to_viewer(entry)

    def _load_entry_to_viewer(self, entry: FileEntry) -> None:
        try:
            path = Path(entry.path)
            if path.is_file():
                arr = read_image(path)
                self.image_viewer.set_image(arr)
        except Exception as exc:
            logger.warning("Could not preview file %s: %s", entry.path, exc)

    def _on_batch_progress(self, index: int, total: int, status_text: str) -> None:
        pct = int(round((index / float(max(1, total))) * 100))
        self._window.status_bar_panel.set_progress(pct)
        self._window.status_bar_panel.set_message(f"[{index}/{total}] {status_text}")

    def _on_batch_frame_processed(self, payload: FramePayload) -> None:
        # Ingest into map engine
        self._map_ctrl.on_frame(payload)

    def _on_batch_finished(self, succeeded: int, failed: int) -> None:
        try:
            self.batch_start_requested.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._window.status_bar_panel.set_cancel_enabled(False)
        self._window.status_bar_panel.set_progress(100)
        self._window.status_bar_panel.set_message(
            f"Обработка завершена: успешно {succeeded}, ошибок {failed}"
        )
        self._batch_worker = None

    def _on_map_updated(self) -> None:
        points = self._map_ctrl.engine.weed_map.points
        self.map_canvas.set_points(points)
        grid = self._map_ctrl.engine.build_heatmap()
        self.map_canvas.set_heatmap(grid)

    def _on_map_locked_changed(self, locked: bool, reason: str) -> None:
        self.map_canvas.set_locked(locked, reason or MAP_LOCKED_TEXT)
        self._window.set_map_tab_locked(locked, reason or MAP_LOCKED_TEXT)

    def _on_weed_selected(self, weed_id: str) -> None:
        for p in self._map_ctrl.engine.weed_map.points:
            if p.weed_id == weed_id:
                self.map_info_panel.show_point(p, None)
                break
