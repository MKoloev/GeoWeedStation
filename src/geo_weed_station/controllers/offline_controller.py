"""Offline mode controller coordinating file viewers, offline workers, and map engine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QApplication, QFileDialog, QProgressDialog

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.core.annotation_io import (
    AnnotationsMap,
    find_annotation_file,
    load_annotations_file,
    save_annotations_to_folder,
)
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.image_io import read_image
from geo_weed_station.core.onnx_detector import (
    OnnxDetector,
    get_models_dir,
    resolve_classes_path,
    scan_available_models,
)
from geo_weed_station.core.report_generator import ReportGenerator
from geo_weed_station.core.source_factory import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    FolderSource,
    ImageSource,
)
from geo_weed_station.core.xmp_parser import has_xmp
from geo_weed_station.schemas.entities import CoordinateSystem, DetectedWeed
from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.schemas.report import FolderDetectionReport
from geo_weed_station.schemas.session import FileEntry, ProcessingStatus
from geo_weed_station.ui.app_mode import AppMode
from geo_weed_station.ui.dialogs.export_dialog import ExportDialog, ExportFormat
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
from geo_weed_station.ui.widgets.map_canvas import LayerControlPanel, MapCanvas
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
        self._has_custom_detector_factory = detector_factory is not None
        self._detector_factory = detector_factory or (lambda: OnnxDetector(settings.detector))

        # Instantiate UI widgets
        self.image_viewer = ImageViewer()
        self.video_player = VideoPlayerWidget()
        self.map_canvas = MapCanvas()
        self.layer_control_panel = LayerControlPanel()
        self.layer_control_panel.connect_canvas(self.map_canvas)
        self.map_info_panel = MapInfoPanel()
        self.file_list_panel = FileListPanel()
        self.filter_panel = DetectionFilterPanel()

        # Active worker references
        self._batch_worker: BatchImageWorker | None = None
        self._video_worker: VideoWorker | None = None
        self._current_source_folder: Path | None = None
        self._detections_by_path: dict[str, list[DetectedWeed]] = {}

        self._embed_widgets_in_ui()
        self._wire_signals()
        self._init_models_in_ui()
        self.on_filter_changed(self.filter_panel.current_state())

    # -------------------------------------------------------------------------
    # Setup & Wiring
    # -------------------------------------------------------------------------
    def _init_models_in_ui(self) -> None:
        """Scan models directory, populate filter panel selector and pre-load classes."""
        active_model = Path(self._settings.detector.model_path)
        models_dir = get_models_dir(active_model)
        available = scan_available_models(models_dir)

        # Resolve active_model to an absolute path if possible
        resolved_model = active_model
        if not resolved_model.is_file():
            candidate = models_dir / active_model.name
            if candidate.is_file():
                resolved_model = candidate
            elif available:
                resolved_model = available[0]

        self.filter_panel.set_available_models(available, active=resolved_model)
        self.filter_panel.set_classifier_type(self._settings.detector.classifier_type)

        classes_p = resolve_classes_path(resolved_model, self._settings.detector.classes_path)
        if classes_p.is_file():
            self.filter_panel.load_classes_from_file(classes_p)

        if resolved_model.is_file():
            updated_det = self._settings.detector.model_copy(
                update={"model_path": resolved_model, "classes_path": classes_p}
            )
            self._settings = self._settings.model_copy(update={"detector": updated_det})
            if not self._has_custom_detector_factory:
                self._detector_factory = lambda: OnnxDetector(self._settings.detector)

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
        self._window.clear_default_panels(AppMode.OFFLINE)
        self._window.add_mode_panel(AppMode.OFFLINE, self.file_list_panel, "Пакетные файлы")
        self._window.add_mode_panel(AppMode.OFFLINE, self.layer_control_panel, "Слои карты")
        self._window.add_mode_panel(AppMode.OFFLINE, self.filter_panel, "Фильтры детекций")
        self._window.add_mode_panel(AppMode.OFFLINE, self.map_info_panel, "Статистика карты")

    def _wire_signals(self) -> None:
        # Window actions
        self._window.open_image_requested.connect(self.open_image)
        self._window.open_folder_requested.connect(self.open_folder)
        self._window.open_video_requested.connect(self.open_video)
        self._window.save_annotations_requested.connect(self.save_annotations)
        self._window.status_bar_panel.cancel_requested.connect(self.cancel)

        # File list panel
        self.file_list_panel.file_selected.connect(self._on_file_selected)
        self.file_list_panel.process_requested.connect(self.start_batch)
        self.file_list_panel.save_annotations_requested.connect(self.save_annotations)
        self.file_list_panel.export_report_requested.connect(self.export_report)
        self.file_list_panel.cleared.connect(self._on_file_list_cleared)

        # Filters & Model selection
        self.filter_panel.filter_changed.connect(self.on_filter_changed)
        self.filter_panel.model_changed.connect(self._on_model_changed)
        self.filter_panel.classifier_changed.connect(self._on_classifier_changed)

        # Map synchronization
        self._map_ctrl.map_updated.connect(self._on_map_updated)
        self._map_ctrl.statistics_changed.connect(self.map_info_panel.set_statistics)
        self._map_ctrl.map_locked_changed.connect(self._on_map_locked_changed)
        self.map_canvas.weed_selected.connect(self._on_weed_selected)
        self.map_canvas.zone_selected.connect(self._on_zone_selected)
        self.map_canvas.frame_selected.connect(self._on_frame_selected)

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

        if self._video_worker is not None:
            self._video_worker.set_detector_factory(self._detector_factory)

        self._window.status_bar_panel.set_message(
            f"Модель детекции переключена: {model_path.name} (классы: {classes_path.name})"
        )

    def _on_classifier_changed(self, cls_type: str) -> None:
        """Handle crop classifier switch: update settings and detector factory."""
        logger.info("Switching crop classifier to %s", cls_type)
        updated_det = self._settings.detector.model_copy(
            update={"classifier_type": cls_type}
        )
        self._settings = self._settings.model_copy(update={"detector": updated_det})
        self._detector_factory = lambda: OnnxDetector(self._settings.detector)

        if self._video_worker is not None:
            self._video_worker.set_detector_factory(self._detector_factory)

        labels = {
            "none": "Отключена",
            "c07": "C07 (7 видов + фазы)",
            "c06": "C06 (6 видов)",
        }
        lbl = labels.get(cls_type.lower(), cls_type)
        self._window.status_bar_panel.set_message(f"Классификация видов: {lbl}")

    # -------------------------------------------------------------------------
    # Actions: Open single image, folder, video
    # -------------------------------------------------------------------------
    @Slot()
    def open_image(self) -> None:
        """Open a single image file, populate file list panel and preview without auto-inference."""
        exts = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Открыть изображение", str(self._settings.workdir), f"Images ({exts})"
        )
        if not path:
            return

        file_path = Path(path)
        self._current_source_folder = file_path.parent
        self._detections_by_path.clear()
        self._map_ctrl.reset()
        self.map_info_panel.clear_selection()
        self._window.central_tabs().setCurrentIndex(self._window.tab_index(TAB_IMAGE))

        progress = QProgressDialog(self._window)
        progress.setWindowTitle("Загрузка данных")
        progress.setLabelText(f"Чтение {file_path.name}...")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setRange(0, 100)
        progress.setValue(30)
        progress.resize(360, 100)
        progress.show()
        QApplication.processEvents()

        try:
            size = file_path.stat().st_size if file_path.is_file() else 0
            has_xmp_flag = has_xmp(file_path) if file_path.is_file() else False
            entry = FileEntry(path=str(file_path), size_bytes=size, has_xmp=has_xmp_flag)

            # Check for existing annotations in the same folder
            annot_file = find_annotation_file(file_path.parent)
            if annot_file is not None:
                progress.setLabelText("Проверка существующей разметки...")
                progress.setValue(60)
                QApplication.processEvents()

                dets_map = load_annotations_file(annot_file, [file_path])
                norm_p = str(file_path)
                dets = dets_map.get(norm_p)
                if dets is None:
                    try:
                        dets = dets_map.get(str(file_path.resolve()))
                    except Exception:
                        pass
                if dets is not None:
                    self._detections_by_path[norm_p] = dets
                    self._detections_by_path[str(file_path)] = dets
                    entry.detections_count = len(dets)
                    if isinstance(dets_map, AnnotationsMap):
                        saved_st = dets_map.get_status(file_path)
                        saved_cnt = dets_map.get_detections_count(file_path)
                        if saved_st is not None:
                            entry.status = saved_st
                        else:
                            entry.status = ProcessingStatus.DONE if entry.has_xmp else ProcessingStatus.NO_METADATA
                        if saved_cnt is not None:
                            entry.detections_count = saved_cnt
                    else:
                        entry.status = ProcessingStatus.DONE if entry.has_xmp else ProcessingStatus.NO_METADATA

                    new_classes = {d.class_name for d in dets if d.class_name}
                    if new_classes:
                        existing = set(self.filter_panel._class_names)
                        self.filter_panel.set_classes(sorted(existing | new_classes))

            progress.setLabelText("Подготовка предпросмотра...")
            progress.setValue(85)
            QApplication.processEvents()

            self.file_list_panel.set_entries([entry])
            self.file_list_panel.update_status(0, entry.status)
            if entry.detections_count is not None:
                self.file_list_panel.update_entry_metadata(
                    0, has_xmp=entry.has_xmp, detections_count=entry.detections_count
                )

            progress.setValue(100)
            QApplication.processEvents()

            if entry.detections_count is not None and entry.detections_count > 0:
                self._window.status_bar_panel.set_message(
                    f"Изображение {file_path.name} загружено. Подгружена разметка ({entry.detections_count} детекций)."
                )
            elif entry.status != ProcessingStatus.PENDING:
                self._window.status_bar_panel.set_message(
                    f"Изображение {file_path.name} загружено. Подгружен статус ({entry.status.value})."
                )
            else:
                self._window.status_bar_panel.set_message(
                    f"Изображение {file_path.name} загружено. Нажмите «⚡ Запустить обработку» для анализа."
                )
        except Exception as exc:
            logger.exception("Failed to open image: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка открытия файла: {exc}")
        finally:
            progress.close()
            QApplication.processEvents()

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
        self._detections_by_path.clear()
        self._map_ctrl.reset()
        self.map_info_panel.clear_selection()

        # Loading progress dialog giving immediate visual feedback to the user
        progress = QProgressDialog(self._window)
        progress.setWindowTitle("Загрузка данных")
        progress.setLabelText(f"Сканирование каталога {folder_path.name}...")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setRange(0, 100)
        progress.setValue(5)
        progress.resize(420, 110)
        progress.show()
        self._window.status_bar_panel.set_message(f"Сканирование каталога {folder_path.name}...")
        QApplication.processEvents()

        try:
            def on_scan_progress(current: int, total: int) -> None:
                pct = 5 + int(35 * current / max(1, total))
                progress.setValue(pct)
                progress.setLabelText(f"Сканирование снимков: {current} из {total}...")
                QApplication.processEvents()

            progress.setLabelText("Поиск файлов и аннотаций...")
            progress.setValue(10)
            QApplication.processEvents()

            src = FolderSource(folder_path)
            entries = src.entries(progress_cb=on_scan_progress)

            # Auto-load annotations if an annotation file was discovered
            if src.annotation_file is not None and src.annotations_map:
                progress.setLabelText("Загрузка аннотаций сорняков...")
                progress.setValue(45)
                QApplication.processEvents()

                for p_str, dets in src.annotations_map.items():
                    norm_p = str(Path(p_str))
                    self._detections_by_path[norm_p] = dets
                    self._detections_by_path[p_str] = dets

                # Sync classes with filter panel
                all_classes = {d.class_name for dets in src.annotations_map.values() for d in dets if d.class_name}
                if all_classes:
                    existing = set(self.filter_panel._class_names)
                    combined = sorted(existing | all_classes)
                    if combined != self.filter_panel._class_names:
                        self.filter_panel.set_classes(combined)

                # Ingest georeferenced frames into MapEngine (without decoding full pixel buffers)
                georef_entries = [e for e in entries if e.has_xmp]
                total_geo_frames = len(georef_entries)
                all_new_pts = []
                has_ingested_frames = False

                for i, entry in enumerate(georef_entries):
                    try:
                        payload = src.load(entry, load_pixels=False)
                        if payload.has_georef():
                            pts = self._map_ctrl.engine.ingest_frame(payload)
                            all_new_pts.extend(pts)
                            has_ingested_frames = True
                    except Exception as err:
                        logger.debug("Could not ingest preloaded frame: %s", err)

                    if i % 3 == 0 or i == total_geo_frames - 1:
                        pct = 45 + int(35 * (i + 1) / max(1, total_geo_frames))
                        progress.setValue(pct)
                        progress.setLabelText(f"Синхронизация карты: кадр {i + 1} из {total_geo_frames}...")
                        QApplication.processEvents()

                if has_ingested_frames:
                    progress.setLabelText("Построение слоев карты...")
                    progress.setValue(85)
                    QApplication.processEvents()

                    self.map_canvas.append_points(all_new_pts)
                    self._map_ctrl.statistics_changed.emit(self._map_ctrl.engine.statistics())
                    self._map_ctrl.map_locked_changed.emit(False, "")
                    self._on_map_updated()

            progress.setLabelText("Формирование списка файлов...")
            progress.setValue(90)
            QApplication.processEvents()

            self.file_list_panel.set_entries(entries)

            if src.annotation_file is not None and src.annotations_map:
                total_dets = sum(len(self._detections_by_path.get(str(Path(e.path)), [])) for e in entries)
                count_frames = sum(1 for e in entries if len(self._detections_by_path.get(str(Path(e.path)), [])) > 0)
                if total_dets > 0:
                    self._window.status_bar_panel.set_message(
                        f"Каталог {folder_path.name} загружен ({len(entries)} снимков). Подгружена разметка из {src.annotation_file.name} ({total_dets} детекций в {count_frames} кадрах)."
                    )
                else:
                    self._window.status_bar_panel.set_message(
                        f"Каталог {folder_path.name} загружен ({len(entries)} снимков). Подгружена разметка из {src.annotation_file.name} (детекций не обнаружено)."
                    )
            else:
                self._window.status_bar_panel.set_message(f"Найдено {len(entries)} снимков в каталоге")

            progress.setValue(100)
            QApplication.processEvents()

        except Exception as exc:
            logger.exception("Failed to open folder: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка открытия каталога: {exc}")
        finally:
            progress.close()
            QApplication.processEvents()

    @Slot()
    def save_annotations(self) -> None:
        """Save current in-memory detections into annotations.json in the image directory."""
        entries = self.file_list_panel.model.entries
        has_processed = any(
            e.status in (ProcessingStatus.DONE, ProcessingStatus.NO_METADATA, ProcessingStatus.ERROR)
            for e in entries
        ) or bool(self._detections_by_path)

        if not has_processed:
            msg = "Нет разметки для сохранения. Сначала запустите обработку."
            logger.info(msg)
            self._window.status_bar_panel.set_message(msg)
            return

        folder = self._current_source_folder
        if folder is None or not folder.is_dir():
            if entries and Path(entries[0].path).is_file():
                folder = Path(entries[0].path).parent
            else:
                msg = "Каталог снимков не определен. Сохранение отменено."
                self._window.status_bar_panel.set_message(msg)
                return

        unique_paths = {str(Path(e.path)) for e in entries} if entries else set(self._detections_by_path.keys())
        total_unique_dets = sum(len(self._detections_by_path.get(p, [])) for p in unique_paths)

        try:
            saved_path = save_annotations_to_folder(
                folder=folder,
                detections_by_path=self._detections_by_path,
                entries=entries,
            )
            frames_with_dets = sum(1 for p in unique_paths if self._detections_by_path.get(p))
            processed_count = sum(
                1 for e in entries if e.status in (ProcessingStatus.DONE, ProcessingStatus.NO_METADATA)
            )
            if total_unique_dets > 0:
                success_msg = f"Разметка успешно сохранена в {saved_path.name} ({total_unique_dets} детекций в {frames_with_dets} кадрах)"
            else:
                success_msg = f"Результаты обработки успешно сохранены в {saved_path.name} (обработано {processed_count} кадров, сорняков не обнаружено)"
            logger.info(success_msg)
            self._window.status_bar_panel.set_message(success_msg)
        except Exception as exc:
            logger.exception("Failed to save annotations: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка сохранения разметки: {exc}")

    def build_folder_report(self) -> FolderDetectionReport:
        """Build a structured FolderDetectionReport for the currently loaded folder/files."""
        gen = ReportGenerator()
        return gen.build_report(
            entries=self.file_list_panel.model.entries,
            detections_by_path=self._detections_by_path,
            map_engine=self._map_ctrl.engine,
            detector_settings=self._settings.detector,
            source_folder=self._current_source_folder,
        )

    @Slot()
    def export_report(self) -> None:
        """Open report export dialog and save analytical report files to disk."""
        entries = self.file_list_panel.model.entries
        points = self._map_ctrl.engine.weed_map.points
        if not entries and not points:
            msg = "Нет данных для выгрузки отчета. Сначала откройте и обработайте папку со снимками."
            logger.info(msg)
            self._window.status_bar_panel.set_message(msg)
            return

        has_georef = bool(not self._map_ctrl.is_locked and points)
        coord_system = self._map_ctrl.engine.weed_map.coord_system or CoordinateSystem.LOCAL_ARUCO
        if self._current_source_folder is not None:
            default_dir = (
                self._current_source_folder
                if self._current_source_folder.name.lower() in ("report", "reports")
                else self._current_source_folder / "report"
            )
        else:
            default_dir = self._settings.workdir / "report"

        dialog = ExportDialog(
            coord_system=coord_system,
            has_orthophoto=False,
            default_dir=default_dir,
            parent=self._window,
            report_mode=True,
            has_georef=has_georef,
        )
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return

        try:
            req = dialog.result_request()
            target_dir = req.target_dir
            if self._current_source_folder is not None:
                try:
                    if target_dir.resolve() == self._current_source_folder.resolve():
                        target_dir = self._current_source_folder / "report"
                except OSError:
                    pass
            target_dir.mkdir(parents=True, exist_ok=True)
            report = self.build_folder_report()
            gen = ReportGenerator()

            if ExportFormat.CSV in req.formats:
                csv_path = target_dir / f"{req.base_name}.csv"
                if points:
                    gen._exporter.export_csv(points, csv_path)
                else:
                    gen.export_detections_csv(report, csv_path)

            if ExportFormat.GEOJSON in req.formats and has_georef:
                gen._exporter.export_geojson(
                    self._map_ctrl.engine.weed_map,
                    target_dir / f"{req.base_name}.geojson",
                    include_zones=True,
                )

            if ExportFormat.PNG in req.formats and has_georef:
                grid = self._map_ctrl.engine.build_heatmap()
                gen._exporter.export_png(
                    points=self._map_ctrl.engine.weed_map,
                    output_path=target_dir / f"{req.base_name}.png",
                    grid=grid,
                    options=req.png_options,
                )

            if ExportFormat.HTML_REPORT in req.formats:
                map_bytes = (
                    gen.render_map_png_bytes(self._map_ctrl.engine, options=req.png_options)
                    if has_georef
                    else None
                )
                gen.export_html(report, target_dir / f"{req.base_name}.html", map_png_bytes=map_bytes)

            if ExportFormat.SUMMARY_JSON in req.formats:
                gen.export_json(report, target_dir / f"{req.base_name}_report.json")

            if ExportFormat.FRAMES_CSV in req.formats:
                gen.export_frames_csv(report, target_dir / f"{req.base_name}_frames.csv")

            if ExportFormat.ZIP_BUNDLE in req.formats:
                gen.export_zip_bundle(
                    report=report,
                    zip_path=target_dir / f"{req.base_name}_report.zip",
                    map_engine=self._map_ctrl.engine if has_georef else None,
                    png_options=req.png_options,
                )

            success_msg = f"Отчет успешно выгружен в каталог: {target_dir}"
            logger.info(success_msg)
            self._window.status_bar_panel.set_message(success_msg)
        except Exception as exc:
            logger.exception("Failed to export report: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка выгрузки отчета: {exc}")

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

        # Stop previous video worker if active and disconnect prior signals
        if self._video_worker is not None:
            old_worker = self._video_worker
            for sig, slot in (
                (self.video_player.play_requested, old_worker.play),
                (self.video_player.pause_requested, old_worker.pause),
                (self.video_player.seek_requested, old_worker.seek),
                (self.video_player.frame_skip_changed, old_worker.set_frame_skip),
                (self.video_player.frame_consumed, old_worker.frame_consumed),
            ):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            old_worker.request_stop()
            self._video_worker = None

        self.video_player.clear()
        self._map_ctrl.reset()
        self.map_info_panel.clear_selection()
        self._detections_by_path.clear()

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
            worker.playback_ended.connect(self._on_video_playback_ended)
            worker.failed.connect(lambda msg: self._window.status_bar_panel.set_message(f"Ошибка видео: {msg}"))

            self.video_player.play_requested.connect(worker.play)
            self.video_player.pause_requested.connect(worker.pause)
            self.video_player.seek_requested.connect(worker.seek)
            self.video_player.frame_skip_changed.connect(worker.set_frame_skip)

            # Move to worker thread
            thread = start_in_thread(worker, "video_worker_thread")
            self._window.register_thread(thread, worker)

            worker.open(path)
            self._window.status_bar_panel.set_message(
                f"Видео {Path(path).name} загружено. Нажмите «▶ Воспроизведение и обработка» для старта."
            )
        except Exception as exc:
            logger.exception("Failed to initialize video worker: %s", exc)
            self._window.status_bar_panel.set_message(f"Ошибка открытия видео: {exc}")

    def _on_video_playback_ended(self) -> None:
        self.video_player.set_playing(False)
        self._window.status_bar_panel.set_message("Воспроизведение видео завершено.")

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

        self._detections_by_path.clear()
        self._map_ctrl.reset()
        self.map_info_panel.clear_selection()

        if len(entries) == 1 and Path(entries[0].path).is_file():
            source = ImageSource(Path(entries[0].path))
        else:
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
        """Propagate detection display filter state to active viewers and update detector settings."""
        self.image_viewer.apply_filter(state)
        self.video_player.apply_filter(state)

        updated_det = self._settings.detector.model_copy(
            update={
                "confidence_threshold": state.min_confidence,
                "nms_iou_threshold": state.nms_iou_threshold,
            }
        )
        self._settings = self._settings.model_copy(update={"detector": updated_det})
        if not self._has_custom_detector_factory:
            self._detector_factory = lambda: OnnxDetector(self._settings.detector)
            if self._video_worker is not None:
                self._video_worker.set_detector_factory(self._detector_factory)

    # -------------------------------------------------------------------------
    # Internal Handlers
    # -------------------------------------------------------------------------
    def _on_file_selected(self, row: int) -> None:
        entry = self.file_list_panel.model.entry_at(row)
        if entry is not None:
            self._window.central_tabs().setCurrentIndex(self._window.tab_index(TAB_IMAGE))
            self._load_entry_to_viewer(entry)

    def _load_entry_to_viewer(self, entry: FileEntry) -> None:
        try:
            path = Path(entry.path)
            if path.is_file():
                arr = read_image(path)
                self.image_viewer.set_image(arr)
                detections = self._detections_by_path.get(
                    str(path), self._detections_by_path.get(entry.path, [])
                )
                self.image_viewer.set_detections(detections)
        except Exception as exc:
            logger.warning("Could not preview file %s: %s", entry.path, exc)

    def _on_file_list_cleared(self) -> None:
        self._detections_by_path.clear()
        self.image_viewer.clear()
        self._map_ctrl.reset()
        self.map_info_panel.clear_selection()
        self.map_canvas.clear_route()
        self.map_canvas.set_mosaic_footprints([])
        self.map_canvas.set_prescription_zones([])

    def _on_batch_progress(self, index: int, total: int, status_text: str) -> None:
        pct = int(round((index / float(max(1, total))) * 100))
        self._window.status_bar_panel.set_progress(pct)
        self._window.status_bar_panel.set_message(f"[{index}/{total}] {status_text}")

    def _on_batch_frame_processed(self, payload: FramePayload) -> None:
        # Cache detections for viewer
        if payload.source_path:
            norm_path = str(Path(payload.source_path))
            detections = list(payload.detections)
            self._detections_by_path[norm_path] = detections
            self._detections_by_path[payload.source_path] = detections

            # Update entry in table model
            for row, entry in enumerate(self.file_list_panel.model.entries):
                if entry.path in (payload.source_path, norm_path):
                    has_xmp = payload.has_georef() or (payload.pose is not None)
                    self.file_list_panel.update_entry_metadata(
                        row, has_xmp=has_xmp, detections_count=len(detections)
                    )
                    break

            # If currently selected item is this file, update detection overlay immediately
            selected = self.file_list_panel.selected_entry()
            if selected is not None and selected.path in (payload.source_path, norm_path):
                self.image_viewer.set_detections(detections)

        # Merge classes into filter panel
        if payload.detections:
            new_classes = {d.class_name for d in payload.detections if d.class_name}
            if new_classes:
                existing = set(self.filter_panel._class_names)
                combined = sorted(existing | new_classes)
                if combined != self.filter_panel._class_names:
                    self.filter_panel.set_classes(combined)

        # Stream-ingest into map engine during batch without blocking GUI thread
        if payload.has_georef():
            try:
                new_pts = self._map_ctrl.engine.ingest_frame(payload)
                self.map_canvas.append_points(new_pts)
                self._map_ctrl.statistics_changed.emit(self._map_ctrl.engine.statistics())
                self._map_ctrl.map_locked_changed.emit(False, "")
            except Exception as exc:
                logger.warning("Frame spatial ingestion error: %s", exc)
        else:
            self._map_ctrl.map_locked_changed.emit(True, MAP_LOCKED_TEXT)

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

        # Build final aggregated layers: deduplicated points, KDE heatmap, prescription zones, mosaic, route
        self._on_map_updated()

        zones = self._map_ctrl.engine.build_prescription_zones()
        self.map_canvas.set_prescription_zones(zones)
        footprints = self._map_ctrl.engine.weed_map.footprints
        if footprints:
            wps = [
                Waypoint(x=fp.center_x, y=fp.center_y, z=fp.altitude_m, yaw=fp.yaw_deg)
                for fp in footprints
            ]
            self.map_canvas.set_route(wps)

    def _on_map_updated(self) -> None:
        points = self._map_ctrl.engine.weed_map.points
        dedup = self._map_ctrl.engine.weed_map.deduplicated_points()
        self.map_canvas.set_points(dedup if dedup else points)
        grid = self._map_ctrl.engine.build_heatmap()
        self.map_canvas.set_heatmap(grid)
        self.map_canvas.set_mosaic_footprints(self._map_ctrl.engine.weed_map.footprints)

    def _on_map_locked_changed(self, locked: bool, reason: str) -> None:
        self.map_canvas.set_locked(locked, reason or MAP_LOCKED_TEXT)
        self._window.set_map_tab_locked(locked, reason or MAP_LOCKED_TEXT)

    def _on_weed_selected(self, weed_id: str) -> None:
        target_pt = self.map_canvas.selected_weed_point()
        if target_pt is None or target_pt.weed_id != weed_id:
            for p in self._map_ctrl.engine.weed_map.points:
                if p.weed_id == weed_id:
                    target_pt = p
                    break

        if target_pt is None:
            return

        self.map_info_panel.show_point(target_pt, None)

        frame_name = target_pt.best_frame_id or target_pt.source_frame
        matching_entry = None
        for entry in self.file_list_panel.model.entries:
            p = Path(entry.path)
            if p.name == frame_name or entry.path == frame_name or p.stem == frame_name:
                matching_entry = entry
                break

        if matching_entry is not None:
            self._window.central_tabs().setCurrentIndex(self._window.tab_index(TAB_IMAGE))
            self._load_entry_to_viewer(matching_entry)
            self.image_viewer.focus_on_weed(target_pt.weed_id, target_pt.bbox)
            self._window.status_bar_panel.set_message(
                f"Сорняк {target_pt.weed_id} ({target_pt.class_name}) выделен на снимке {Path(matching_entry.path).name}"
            )

    def _on_frame_selected(self, frame_id: str) -> None:
        matching_entry = None
        for entry in self.file_list_panel.model.entries:
            p = Path(entry.path)
            if p.name == frame_id or entry.path == frame_id or p.stem == frame_id:
                matching_entry = entry
                break

        if matching_entry is not None:
            self._window.central_tabs().setCurrentIndex(self._window.tab_index(TAB_IMAGE))
            self._load_entry_to_viewer(matching_entry)
            self._window.status_bar_panel.set_message(f"Кадр {Path(matching_entry.path).name} открыт в инспекторе")

    def _on_zone_selected(self, zone_id: str) -> None:
        for z in self._map_ctrl.engine.weed_map.prescription_zones:
            if z.zone_id == zone_id:
                self._window.status_bar_panel.set_message(
                    f"Очаг {z.zone_id}: площадь {z.area_m2:.1f} м², сорняков {z.weed_count}, норма: {z.treatment_recommendation}"
                )
                break
