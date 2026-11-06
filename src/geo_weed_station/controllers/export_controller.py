"""Export controller coordinating ExportDialog and MapExporter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QDialog

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.core.map_engine.exporter import MapExporter
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.ui.dialogs.export_dialog import (
    ExportDialog,
    ExportFormat,
    ExportRequest,
)
from geo_weed_station.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class ExportController(QObject):
    """Controller triggering and executing agronomic map exports across CSV, GeoJSON, and PNG formats."""

    export_completed = Signal(str)  # Target export folder path
    export_failed = Signal(str)  # Error description

    def __init__(
        self,
        window: MainWindow,
        map_ctrl: MapController,
        settings: AppSettings,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._map_ctrl = map_ctrl
        self._settings = settings
        self._exporter = MapExporter()

        # Wire window export action to controller slot
        self._window.request_export.connect(self.export)

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
            default_dir=self._settings.workdir,
            parent=self._window,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            req = dialog.result_request()
            req.target_dir.mkdir(parents=True, exist_ok=True)

            if ExportFormat.CSV in req.formats:
                csv_path = req.target_dir / f"{req.base_name}.csv"
                self._exporter.export_csv(points, csv_path)

            if ExportFormat.GEOJSON in req.formats:
                geojson_path = req.target_dir / f"{req.base_name}.geojson"
                self._exporter.export_geojson(points, geojson_path)

            if ExportFormat.PNG in req.formats:
                png_path = req.target_dir / f"{req.base_name}.png"
                grid = self._map_ctrl.engine.build_heatmap()
                self._exporter.export_png(
                    points=points,
                    output_path=png_path,
                    grid=grid,
                    options=req.png_options,
                )

            success_msg = f"Экспорт успешно сохранен в каталог: {req.target_dir}"
            logger.info(success_msg)
            self._window.status_bar_panel.set_message(success_msg)
            self.export_completed.emit(str(req.target_dir))
        except Exception as exc:
            err_msg = f"Ошибка экспорта: {exc}"
            logger.exception("Export failed: %s", exc)
            self._window.status_bar_panel.set_message(err_msg)
            self.export_failed.emit(err_msg)
