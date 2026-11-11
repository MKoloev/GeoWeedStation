"""Export controller coordinating ExportDialog, MapExporter, and ReportGenerator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QDialog

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.core.map_engine.exporter import MapExporter
from geo_weed_station.core.report_generator import ReportGenerator
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.report import FolderDetectionReport
from geo_weed_station.ui.dialogs.export_dialog import (
    ExportDialog,
    ExportFormat,
    ExportRequest,
)
from geo_weed_station.ui.main_window import MainWindow

if TYPE_CHECKING:
    from geo_weed_station.controllers.offline_controller import OfflineController

logger = logging.getLogger(__name__)


class ExportController(QObject):
    """Controller triggering and executing agronomic map exports and folder detection reports."""

    export_completed = Signal(str)  # Target export folder path
    export_failed = Signal(str)  # Error description

    def __init__(
        self,
        window: MainWindow,
        map_ctrl: MapController,
        settings: AppSettings,
        offline_ctrl: OfflineController | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._map_ctrl = map_ctrl
        self._settings = settings
        self._offline_ctrl = offline_ctrl
        self._exporter = MapExporter()
        self._report_generator = ReportGenerator(exporter=self._exporter)

        # Wire window export actions to controller slots
        self._window.request_export.connect(self.export)
        self._window.export_report_requested.connect(self.export_report)

        if self._offline_ctrl is not None:
            self.set_offline_controller(self._offline_ctrl)

    def set_offline_controller(self, offline_ctrl: OfflineController) -> None:
        """Bind OfflineController to access batch entries, detections, and button signals."""
        self._offline_ctrl = offline_ctrl
        try:
            self._offline_ctrl.file_list_panel.export_report_requested.disconnect(
                self._offline_ctrl.export_report
            )
        except (RuntimeError, TypeError):
            pass
        self._offline_ctrl.file_list_panel.export_report_requested.connect(self.export_report)

    def build_current_report(self) -> FolderDetectionReport:
        """Build a structured FolderDetectionReport from the current offline/map session."""
        entries = self._offline_ctrl.file_list_panel.model.entries if self._offline_ctrl else []
        dets_by_path = self._offline_ctrl._detections_by_path if self._offline_ctrl else {}
        source_folder = self._offline_ctrl._current_source_folder if self._offline_ctrl else None
        det_settings = (
            self._offline_ctrl._settings.detector
            if self._offline_ctrl is not None
            else self._settings.detector
        )
        filter_state = (
            self._offline_ctrl.filter_panel.current_state()
            if self._offline_ctrl is not None
            else None
        )
        return self._report_generator.build_report(
            entries=entries,
            detections_by_path=dets_by_path,
            map_engine=self._map_ctrl.engine,
            detector_settings=det_settings,
            source_folder=source_folder,
            hidden_classes=filter_state.hidden_classes if filter_state is not None else None,
            min_confidence=filter_state.min_confidence if filter_state is not None else None,
        )

    @Slot()
    def export(self) -> None:
        """Open export configuration dialog and perform export if map is valid and unlocked."""
        if self._map_ctrl.is_locked:
            msg = "Геопривязка отсутствует. Экспорт слоев карты недоступен."
            logger.warning(msg)
            self._window.status_bar_panel.set_message(msg)
            self.export_failed.emit(msg)
            return

        points = self._map_ctrl.engine.weed_map.points
        if not points:
            msg = "Экспорт отменен: на карте отсутствуют точки сорняков."
            logger.info(msg)
            self._window.status_bar_panel.set_message(msg)
            return

        coord_system = self._map_ctrl.engine.weed_map.coord_system or CoordinateSystem.LOCAL_ARUCO
        dialog = ExportDialog(
            coord_system=coord_system,
            has_orthophoto=False,
            default_dir=self._default_report_dir(),
            parent=self._window,
            report_mode=False,
            has_georef=True,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._execute_export_request(dialog.result_request())

    def _default_report_dir(self) -> Path:
        """Return the dedicated 'report' subfolder inside the current source folder or workdir."""
        source_folder = (
            self._offline_ctrl._current_source_folder
            if (self._offline_ctrl and self._offline_ctrl._current_source_folder)
            else None
        )
        if source_folder is not None:
            return (
                source_folder
                if source_folder.name.lower() in ("report", "reports")
                else source_folder / "report"
            )
        return self._settings.workdir / "report"

    @Slot()
    def export_report(self) -> None:
        """Open report export dialog and generate analytical report (works with or without XMP georeferencing)."""
        entries = self._offline_ctrl.file_list_panel.model.entries if self._offline_ctrl else []
        points = self._map_ctrl.engine.weed_map.points

        if not entries and not points:
            msg = "Нет данных для выгрузки отчета. Сначала откройте и обработайте папку со снимками."
            logger.info(msg)
            self._window.status_bar_panel.set_message(msg)
            self.export_failed.emit(msg)
            return

        has_georef = bool(not self._map_ctrl.is_locked and points)
        coord_system = self._map_ctrl.engine.weed_map.coord_system or CoordinateSystem.LOCAL_ARUCO
        default_dir = self._default_report_dir()

        dialog = ExportDialog(
            coord_system=coord_system,
            has_orthophoto=False,
            default_dir=default_dir,
            parent=self._window,
            report_mode=True,
            has_georef=has_georef,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._execute_export_request(dialog.result_request())

    def _execute_export_request(self, req: ExportRequest) -> None:
        """Execute all requested export and report formats and emit completion signal."""
        try:
            target_dir = req.target_dir
            source_folder = (
                self._offline_ctrl._current_source_folder
                if (self._offline_ctrl and self._offline_ctrl._current_source_folder)
                else None
            )
            if source_folder is not None:
                try:
                    if target_dir.resolve() == source_folder.resolve():
                        target_dir = source_folder / "report"
                except OSError:
                    pass

            target_dir.mkdir(parents=True, exist_ok=True)
            all_points = self._map_ctrl.engine.weed_map.points
            points = self._map_ctrl.engine.filtered_points()
            has_georef = bool(not self._map_ctrl.is_locked and all_points)
            filtered_wm = self._map_ctrl.engine.filtered_weed_map()

            report: FolderDetectionReport | None = None
            needs_report = any(
                fmt in req.formats
                for fmt in (
                    ExportFormat.HTML_REPORT,
                    ExportFormat.SUMMARY_JSON,
                    ExportFormat.FRAMES_CSV,
                    ExportFormat.ZIP_BUNDLE,
                )
            ) or (ExportFormat.CSV in req.formats and not all_points)

            if needs_report:
                report = self.build_current_report()

            if ExportFormat.CSV in req.formats:
                csv_path = target_dir / f"{req.base_name}.csv"
                if all_points:
                    self._exporter.export_csv(points, csv_path)
                elif report is not None:
                    self._report_generator.export_detections_csv(report, csv_path)

            if ExportFormat.GEOJSON in req.formats and has_georef:
                geojson_path = target_dir / f"{req.base_name}.geojson"
                self._exporter.export_geojson(
                    filtered_wm, geojson_path, include_zones=True
                )

            if ExportFormat.PNG in req.formats and has_georef:
                png_path = target_dir / f"{req.base_name}.png"
                grid = self._map_ctrl.engine.build_heatmap()
                self._exporter.export_png(
                    points=filtered_wm,
                    output_path=png_path,
                    grid=grid,
                    options=req.png_options,
                )

            if ExportFormat.HTML_REPORT in req.formats and report is not None:
                html_path = target_dir / f"{req.base_name}.html"
                map_bytes = (
                    self._report_generator.render_map_png_bytes(
                        self._map_ctrl.engine, options=req.png_options
                    )
                    if has_georef
                    else None
                )
                self._report_generator.export_html(report, html_path, map_png_bytes=map_bytes)

            if ExportFormat.SUMMARY_JSON in req.formats and report is not None:
                json_path = target_dir / f"{req.base_name}_report.json"
                self._report_generator.export_json(report, json_path)

            if ExportFormat.FRAMES_CSV in req.formats and report is not None:
                frames_csv_path = target_dir / f"{req.base_name}_frames.csv"
                self._report_generator.export_frames_csv(report, frames_csv_path)

            if ExportFormat.ZIP_BUNDLE in req.formats and report is not None:
                zip_path = target_dir / f"{req.base_name}_report.zip"
                self._report_generator.export_zip_bundle(
                    report=report,
                    zip_path=zip_path,
                    map_engine=self._map_ctrl.engine if has_georef else None,
                    png_options=req.png_options,
                )

            success_msg = f"Экспорт успешно сохранен в каталог: {target_dir}"
            logger.info(success_msg)
            self._window.status_bar_panel.set_message(success_msg)
            self.export_completed.emit(str(target_dir))
        except Exception as exc:
            err_msg = f"Ошибка экспорта: {exc}"
            logger.exception("Export failed: %s", exc)
            self._window.status_bar_panel.set_message(err_msg)
            self.export_failed.emit(err_msg)
