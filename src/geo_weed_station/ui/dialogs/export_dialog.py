"""Export dialog allowing selection of output formats, directory, and PNG map rendering options."""

from __future__ import annotations

from enum import Enum
import logging
from pathlib import Path
from typing import Optional, Set
from pydantic import BaseModel, ConfigDict, Field
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import PngExportOptions

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export data formats."""

    CSV = "csv"
    GEOJSON = "geojson"
    PNG = "png"


class ExportRequest(BaseModel):
    """Complete specification of an agronomic map export request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    formats: Set[ExportFormat] = Field(..., min_length=1)
    target_dir: Path
    base_name: str
    png_options: Optional[PngExportOptions] = None


class ExportDialog(QDialog):
    """Dialog for configuring export formats (CSV, GeoJSON, PNG) and styling options."""

    def __init__(
        self,
        coord_system: CoordinateSystem = CoordinateSystem.LOCAL_ARUCO,
        has_orthophoto: bool = False,
        default_dir: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Экспорт результатов картографирования")
        self.resize(520, 440)

        self._coord_system = coord_system
        self._has_orthophoto = has_orthophoto
        self._target_dir = default_dir or Path.cwd()
        self._result_request: Optional[ExportRequest] = None

        self._init_ui()
        self._validate_inputs()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        # 1. Output directory and filename prefix
        target_group = QGroupBox("Параметры сохранения", self)
        target_layout = QFormLayout(target_group)

        row_dir = QHBoxLayout()
        self.edit_dir = QLineEdit(str(self._target_dir), target_group)
        btn_browse_dir = QPushButton("Обзор...", target_group)
        btn_browse_dir.clicked.connect(self._browse_dir)
        row_dir.addWidget(self.edit_dir)
        row_dir.addWidget(btn_browse_dir)
        target_layout.addRow("Каталог экспорта:", row_dir)

        self.edit_name = QLineEdit("weed_map_export", target_group)
        target_layout.addRow("Имя файлов (префикс):", self.edit_name)

        # Coordinate system notice
        if self._coord_system == CoordinateSystem.LOCAL_ARUCO:
            cs_text = "Локальная метрическая СК полигона (метры, EPSG:LOCAL)"
        else:
            cs_text = "Глобальная географическая система координат WGS84 (EPSG:4326)"
        lbl_cs = QLabel(cs_text, target_group)
        lbl_cs.setStyleSheet("color: #4dabf7; font-weight: bold;")
        target_layout.addRow("Система координат:", lbl_cs)

        main_layout.addWidget(target_group)

        # 2. Formats checkboxes
        formats_group = QGroupBox("Выбор форматов данных", self)
        formats_layout = QVBoxLayout(formats_group)

        self.cb_csv = QCheckBox("Таблица сорняков CSV (X, Y, класс, confidence)", formats_group)
        self.cb_csv.setChecked(True)
        self.cb_csv.toggled.connect(self._validate_inputs)
        formats_layout.addWidget(self.cb_csv)

        self.cb_geojson = QCheckBox("Векторный геопространственный слой GeoJSON", formats_group)
        self.cb_geojson.setChecked(True)
        self.cb_geojson.toggled.connect(self._validate_inputs)
        formats_layout.addWidget(self.cb_geojson)

        self.cb_png = QCheckBox("Растровая тепловая карта высокого разрешения (PNG)", formats_group)
        self.cb_png.setChecked(True)
        self.cb_png.toggled.connect(self._on_png_toggled)
        formats_layout.addWidget(self.cb_png)

        main_layout.addWidget(formats_group)

        # 3. PNG Options Group (active only when PNG format checked)
        self.png_group = QGroupBox("Параметры графического экспорта (PNG)", self)
        png_layout = QFormLayout(self.png_group)

        self.combo_bg = QComboBox(self.png_group)
        self.combo_bg.addItem("Белый фон", "white")
        self.combo_bg.addItem("Прозрачный фон", "transparent")
        if self._has_orthophoto:
            self.combo_bg.addItem("Подложка ортофотоплана", "orthophoto")
        png_layout.addRow("Фон изображения:", self.combo_bg)

        self.cb_grid = QCheckBox("Координатная сетка", self.png_group)
        self.cb_grid.setChecked(True)
        png_layout.addRow("Сетка:", self.cb_grid)

        self.cb_legend = QCheckBox("Легенда плотности и классов", self.png_group)
        self.cb_legend.setChecked(True)
        png_layout.addRow("Легенда:", self.cb_legend)

        self.cb_scalebar = QCheckBox("Масштабная линейка", self.png_group)
        self.cb_scalebar.setChecked(True)
        png_layout.addRow("Масштаб:", self.cb_scalebar)

        self.spin_dpi = QSpinBox(self.png_group)
        self.spin_dpi.setRange(72, 1200)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSuffix(" DPI")
        png_layout.addRow("Разрешение печати:", self.spin_dpi)

        main_layout.addWidget(self.png_group)

        # 4. Error message banner
        self.lbl_error = QLabel("", self)
        self.lbl_error.setStyleSheet("color: #fa5252; font-weight: bold;")
        self.lbl_error.setVisible(False)
        main_layout.addWidget(self.lbl_error)

        # 5. Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        main_layout.addWidget(self.buttons)

        self.edit_dir.textChanged.connect(self._validate_inputs)
        self.edit_name.textChanged.connect(self._validate_inputs)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def result_request(self) -> ExportRequest:
        """Return the validated ExportRequest object."""
        if self._result_request is None:
            raise RuntimeError("Dialog was not accepted or request is invalid")
        return self._result_request

    # -------------------------------------------------------------------------
    # Internal handlers
    # -------------------------------------------------------------------------
    def _on_png_toggled(self, checked: bool) -> None:
        self.png_group.setEnabled(checked)
        self._validate_inputs()

    def _validate_inputs(self) -> bool:
        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)

        # 1. At least one format must be selected
        has_fmt = self.cb_csv.isChecked() or self.cb_geojson.isChecked() or self.cb_png.isChecked()
        if not has_fmt:
            self._show_error("Выберите хотя бы один формат для экспорта.")
            if ok_btn:
                ok_btn.setEnabled(False)
            return False

        # 2. Output directory must not be empty
        dir_text = self.edit_dir.text().strip()
        if not dir_text:
            self._show_error("Укажите каталог для сохранения файлов.")
            if ok_btn:
                ok_btn.setEnabled(False)
            return False

        # 3. Base name must not be empty
        base_name = self.edit_name.text().strip()
        if not base_name:
            self._show_error("Укажите базовое имя файлов.")
            if ok_btn:
                ok_btn.setEnabled(False)
            return False

        self.lbl_error.setVisible(False)
        if ok_btn:
            ok_btn.setEnabled(True)
        return True

    def _show_error(self, msg: str) -> None:
        self.lbl_error.setText(f"• {msg}")
        self.lbl_error.setVisible(True)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выбрать каталог для экспорта")
        if path:
            self.edit_dir.setText(path)

    def _on_accept(self) -> None:
        if not self._validate_inputs():
            return

        formats: Set[ExportFormat] = set()
        if self.cb_csv.isChecked():
            formats.add(ExportFormat.CSV)
        if self.cb_geojson.isChecked():
            formats.add(ExportFormat.GEOJSON)
        if self.cb_png.isChecked():
            formats.add(ExportFormat.PNG)

        png_opts: Optional[PngExportOptions] = None
        if ExportFormat.PNG in formats:
            png_opts = PngExportOptions(
                background=self.combo_bg.currentData(),
                show_grid=self.cb_grid.isChecked(),
                show_legend=self.cb_legend.isChecked(),
                show_scalebar=self.cb_scalebar.isChecked(),
                dpi=self.spin_dpi.value(),
            )

        self._result_request = ExportRequest(
            formats=formats,
            target_dir=Path(self.edit_dir.text().strip()),
            base_name=self.edit_name.text().strip(),
            png_options=png_opts,
        )
        self.accept()
