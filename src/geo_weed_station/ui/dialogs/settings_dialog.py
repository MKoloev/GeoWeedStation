"""Settings dialog with tabbed configuration for network, detector, map and workdir."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.config.settings import (
    AppSettings,
    DetectorSettings,
    MapSettings,
    NetworkSettings,
)

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Settings dialog allowing safe, validated modification of application configuration."""

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки GeoWeedStation")
        self.resize(550, 480)

        self._initial_settings = settings
        self._result_settings = settings

        self._init_ui()
        self._load_from_settings(settings)
        self._validate_all()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        self.tabs = QTabWidget(self)

        # Tab 1: Network
        self.tab_network = QWidget()
        net_layout = QFormLayout(self.tab_network)
        self.edit_host = QLineEdit(self.tab_network)
        self.spin_port = QSpinBox(self.tab_network)
        self.spin_port.setRange(1, 65535)

        self.spin_timeout = QDoubleSpinBox(self.tab_network)
        self.spin_timeout.setRange(0.0, 10.0)  # Allows entering invalid value to trigger validation error
        self.spin_timeout.setSingleStep(0.1)
        self.spin_timeout.setSuffix(" с")

        self.spin_poll_hz = QDoubleSpinBox(self.tab_network)
        self.spin_poll_hz.setRange(0.0, 10.0)  # Allows entering invalid value to trigger validation error
        self.spin_poll_hz.setSingleStep(0.1)
        self.spin_poll_hz.setSuffix(" Гц")

        self.spin_backoff_init = QDoubleSpinBox(self.tab_network)
        self.spin_backoff_init.setRange(0.1, 60.0)
        self.spin_backoff_init.setSuffix(" с")

        self.spin_backoff_max = QDoubleSpinBox(self.tab_network)
        self.spin_backoff_max.setRange(0.1, 300.0)
        self.spin_backoff_max.setSuffix(" с")

        net_layout.addRow("Хост борта (IP/DNS):", self.edit_host)
        net_layout.addRow("Порт REST API:", self.spin_port)
        net_layout.addRow("Таймаут запросов (≤ 2.0 с):", self.spin_timeout)
        net_layout.addRow("Частота телеметрии (0.5..2.0 Гц):", self.spin_poll_hz)
        net_layout.addRow("Начальная задержка переподключения:", self.spin_backoff_init)
        net_layout.addRow("Макс. задержка переподключения:", self.spin_backoff_max)
        self.tabs.addTab(self.tab_network, "Сеть и борт")

        # Tab 2: Detector
        self.tab_detector = QWidget()
        det_layout = QFormLayout(self.tab_detector)

        row_model = QHBoxLayout()
        self.edit_model_path = QLineEdit(self.tab_detector)
        btn_browse_model = QPushButton("...", self.tab_detector)
        btn_browse_model.clicked.connect(self._browse_model)
        row_model.addWidget(self.edit_model_path)
        row_model.addWidget(btn_browse_model)
        det_layout.addRow("Файл ONNX-модели:", row_model)

        row_classes = QHBoxLayout()
        self.edit_classes_path = QLineEdit(self.tab_detector)
        btn_browse_classes = QPushButton("...", self.tab_detector)
        btn_browse_classes.clicked.connect(self._browse_classes)
        row_classes.addWidget(self.edit_classes_path)
        row_classes.addWidget(btn_browse_classes)
        det_layout.addRow("Файл классов (JSON):", row_classes)

        self.spin_conf_thresh = QDoubleSpinBox(self.tab_detector)
        self.spin_conf_thresh.setRange(0.01, 1.0)
        self.spin_conf_thresh.setSingleStep(0.05)
        det_layout.addRow("Порог уверенности (Confidence):", self.spin_conf_thresh)

        self.spin_iou_thresh = QDoubleSpinBox(self.tab_detector)
        self.spin_iou_thresh.setRange(0.01, 1.0)
        self.spin_iou_thresh.setSingleStep(0.05)
        det_layout.addRow("Порог NMS IoU:", self.spin_iou_thresh)

        self.spin_frame_skip = QSpinBox(self.tab_detector)
        self.spin_frame_skip.setRange(1, 60)
        det_layout.addRow("Пропуск кадров видео:", self.spin_frame_skip)
        self.tabs.addTab(self.tab_detector, "Нейросеть (ONNX)")

        # Tab 3: Map
        self.tab_map = QWidget()
        map_layout = QFormLayout(self.tab_map)

        self.spin_cell_m = QDoubleSpinBox(self.tab_map)
        self.spin_cell_m.setRange(0.1, 10.0)
        self.spin_cell_m.setSingleStep(0.1)
        self.spin_cell_m.setSuffix(" м")
        map_layout.addRow("Разрешение ячейки Heatmap:", self.spin_cell_m)

        self.spin_crit_density = QDoubleSpinBox(self.tab_map)
        self.spin_crit_density.setRange(0.1, 100.0)
        self.spin_crit_density.setSuffix(" шт/м²")
        map_layout.addRow("Критический порог засоренности:", self.spin_crit_density)

        self.spin_max_extent = QDoubleSpinBox(self.tab_map)
        self.spin_max_extent.setRange(0.5, 100.0)
        self.spin_max_extent.setSuffix(" м")
        map_layout.addRow("Макс. размер полигона ArUco:", self.spin_max_extent)

        # WGS84 origin reference point
        origin_group = QGroupBox("Опорная точка WGS84 (Origin)", self.tab_map)
        orig_layout = QFormLayout(origin_group)
        self.cb_has_origin = QCheckBox("Задать опорную геопривязку для ArUco", origin_group)
        self.spin_lat = QDoubleSpinBox(origin_group)
        self.spin_lat.setRange(-90.0, 90.0)
        self.spin_lat.setDecimals(6)
        self.spin_lon = QDoubleSpinBox(origin_group)
        self.spin_lon.setRange(-180.0, 180.0)
        self.spin_lon.setDecimals(6)
        self.cb_has_origin.toggled.connect(self._toggle_origin)
        orig_layout.addRow(self.cb_has_origin)
        orig_layout.addRow("Широта (Lat):", self.spin_lat)
        orig_layout.addRow("Долгота (Lon):", self.spin_lon)
        map_layout.addRow(origin_group)

        self.tabs.addTab(self.tab_map, "Карта и Heatmap")

        # Tab 4: Workdir
        self.tab_workdir = QWidget()
        work_layout = QFormLayout(self.tab_workdir)
        row_work = QHBoxLayout()
        self.edit_workdir = QLineEdit(self.tab_workdir)
        btn_browse_work = QPushButton("...", self.tab_workdir)
        btn_browse_work.clicked.connect(self._browse_workdir)
        row_work.addWidget(self.edit_workdir)
        row_work.addWidget(btn_browse_work)
        work_layout.addRow("Рабочий каталог:", row_work)
        self.tabs.addTab(self.tab_workdir, "Хранилище")

        main_layout.addWidget(self.tabs)

        # Validation error banner
        self.lbl_error = QLabel("", self)
        self.lbl_error.setStyleSheet("color: #fa5252; font-weight: bold;")
        self.lbl_error.setVisible(False)
        main_layout.addWidget(self.lbl_error)

        # Buttons Box
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        main_layout.addWidget(self.buttons)

        # Connect change signals for live validation
        self.edit_host.textChanged.connect(self._validate_all)
        self.spin_port.valueChanged.connect(self._validate_all)
        self.spin_timeout.valueChanged.connect(self._validate_all)
        self.spin_poll_hz.valueChanged.connect(self._validate_all)
        self.spin_backoff_init.valueChanged.connect(self._validate_all)
        self.spin_backoff_max.valueChanged.connect(self._validate_all)
        self.spin_conf_thresh.valueChanged.connect(self._validate_all)
        self.spin_iou_thresh.valueChanged.connect(self._validate_all)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def result_settings(self) -> AppSettings:
        """Return the validated AppSettings configuration object."""
        return self._result_settings

    # -------------------------------------------------------------------------
    # Data loading & validation
    # -------------------------------------------------------------------------
    def _load_from_settings(self, s: AppSettings) -> None:
        self.edit_host.setText(s.network.host)
        self.spin_port.setValue(s.network.port)
        self.spin_timeout.setValue(s.network.request_timeout_s)
        self.spin_poll_hz.setValue(s.network.poll_rate_hz)
        self.spin_backoff_init.setValue(s.network.reconnect_backoff_initial_s)
        self.spin_backoff_max.setValue(s.network.reconnect_backoff_max_s)

        self.edit_model_path.setText(str(s.detector.model_path))
        self.edit_classes_path.setText(str(s.detector.classes_path))
        self.spin_conf_thresh.setValue(s.detector.confidence_threshold)
        self.spin_iou_thresh.setValue(s.detector.nms_iou_threshold)
        self.spin_frame_skip.setValue(s.detector.video_frame_skip)

        self.spin_cell_m.setValue(s.map.heatmap_cell_m)
        self.spin_crit_density.setValue(s.map.critical_density_threshold)
        self.spin_max_extent.setValue(s.map.max_local_extent_m)

        if s.map.wgs84_origin is not None:
            self.cb_has_origin.setChecked(True)
            self.spin_lat.setValue(s.map.wgs84_origin[0])
            self.spin_lon.setValue(s.map.wgs84_origin[1])
        else:
            self.cb_has_origin.setChecked(False)
            self.spin_lat.setValue(0.0)
            self.spin_lon.setValue(0.0)
        self._toggle_origin(self.cb_has_origin.isChecked())

        self.edit_workdir.setText(str(s.workdir))

    def _toggle_origin(self, checked: bool) -> None:
        self.spin_lat.setEnabled(checked)
        self.spin_lon.setEnabled(checked)

    def _validate_all(self) -> bool:
        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)

        # Check timeout <= 2.0s
        if self.spin_timeout.value() > 2.0 or self.spin_timeout.value() <= 0:
            self._show_error("Таймаут запроса не должен превышать 2.0 с.")
            if ok_btn:
                ok_btn.setEnabled(False)
            return False

        # Check poll rate 0.5 .. 2.0 Hz
        if not (0.5 <= self.spin_poll_hz.value() <= 2.0):
            self._show_error("Частота опроса телеметрии должна быть в диапазоне 0.5 .. 2.0 Гц.")
            if ok_btn:
                ok_btn.setEnabled(False)
            return False

        # Check backoff max >= initial
        if self.spin_backoff_max.value() < self.spin_backoff_init.value():
            self._show_error("Макс. задержка переподключения не может быть меньше начальной.")
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

    def _on_accept(self) -> None:
        if not self._validate_all():
            return

        try:
            origin: Optional[Tuple[float, float]] = None
            if self.cb_has_origin.isChecked():
                origin = (self.spin_lat.value(), self.spin_lon.value())

            net = NetworkSettings(
                host=self.edit_host.text().strip() or "192.168.4.1",
                port=self.spin_port.value(),
                request_timeout_s=self.spin_timeout.value(),
                poll_rate_hz=self.spin_poll_hz.value(),
                reconnect_backoff_initial_s=self.spin_backoff_init.value(),
                reconnect_backoff_max_s=self.spin_backoff_max.value(),
            )
            det = DetectorSettings(
                model_path=Path(self.edit_model_path.text()),
                classes_path=Path(self.edit_classes_path.text()),
                confidence_threshold=self.spin_conf_thresh.value(),
                nms_iou_threshold=self.spin_iou_thresh.value(),
                video_frame_skip=self.spin_frame_skip.value(),
            )
            map_s = MapSettings(
                heatmap_cell_m=self.spin_cell_m.value(),
                critical_density_threshold=self.spin_crit_density.value(),
                max_local_extent_m=self.spin_max_extent.value(),
                wgs84_origin=origin,
            )
            self._result_settings = AppSettings(
                network=net,
                detector=det,
                map=map_s,
                workdir=Path(self.edit_workdir.text()),
            )
            self.accept()
        except Exception as exc:
            self._show_error(str(exc))

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать ONNX-модель", "", "ONNX models (*.onnx);;All (*)")
        if path:
            self.edit_model_path.setText(path)

    def _browse_classes(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать файл классов", "", "JSON (*.json);;All (*)")
        if path:
            self.edit_classes_path.setText(path)

    def _browse_workdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выбрать рабочий каталог")
        if path:
            self.edit_workdir.setText(path)
