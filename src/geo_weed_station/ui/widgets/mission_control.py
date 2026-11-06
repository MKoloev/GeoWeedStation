"""Flight mission planning and execution control widget."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.core.mission_planner import PlannerParams
from geo_weed_station.schemas.entities import CoordinateSystem, SensorOptics
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy, Waypoint
from geo_weed_station.schemas.onboard_api import StopAction

logger = logging.getLogger(__name__)


class MissionControl(QWidget):
    """Mission planner UI: polygon input, mode switch, flight command buttons and validation."""

    plan_requested = Signal(PlannerParams)
    send_mission_requested = Signal(MissionTask)
    stop_requested = Signal(StopAction)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._planned_waypoints: list[Waypoint] = []
        self._is_flight_locked = False

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 1. Survey Area & Coordinate System Group
        area_group = QGroupBox("Полетная зона полигона", container)
        area_layout = QFormLayout(area_group)
        area_layout.setContentsMargins(6, 6, 6, 6)

        self.combo_cs = QComboBox(area_group)
        self.combo_cs.addItem("local_aruco (метры)", CoordinateSystem.LOCAL_ARUCO)
        self.combo_cs.addItem("wgs84 (широта/долгота)", CoordinateSystem.WGS84)
        area_layout.addRow("Система координат:", self.combo_cs)

        # Corner 1 (X1, Y1)
        row_c1 = QHBoxLayout()
        self.spin_x1 = QDoubleSpinBox(area_group)
        self.spin_x1.setRange(-100000.0, 100000.0)
        self.spin_x1.setValue(0.0)
        self.spin_x1.setDecimals(2)
        self.spin_y1 = QDoubleSpinBox(area_group)
        self.spin_y1.setRange(-100000.0, 100000.0)
        self.spin_y1.setValue(0.0)
        self.spin_y1.setDecimals(2)
        row_c1.addWidget(QLabel("X1:"))
        row_c1.addWidget(self.spin_x1)
        row_c1.addWidget(QLabel("Y1:"))
        row_c1.addWidget(self.spin_y1)
        area_layout.addRow("Угол 1 (TL):", row_c1)

        # Corner 2 (X2, Y2)
        row_c2 = QHBoxLayout()
        self.spin_x2 = QDoubleSpinBox(area_group)
        self.spin_x2.setRange(-100000.0, 100000.0)
        self.spin_x2.setValue(20.0)
        self.spin_x2.setDecimals(2)
        self.spin_y2 = QDoubleSpinBox(area_group)
        self.spin_y2.setRange(-100000.0, 100000.0)
        self.spin_y2.setValue(20.0)
        self.spin_y2.setDecimals(2)
        row_c2.addWidget(QLabel("X2:"))
        row_c2.addWidget(self.spin_x2)
        row_c2.addWidget(QLabel("Y2:"))
        row_c2.addWidget(self.spin_y2)
        area_layout.addRow("Угол 2 (BR):", row_c2)

        # Altitude
        self.spin_alt = QDoubleSpinBox(area_group)
        self.spin_alt.setRange(1.0, 150.0)
        self.spin_alt.setValue(10.0)
        self.spin_alt.setSingleStep(1.0)
        self.spin_alt.setSuffix(" м")
        area_layout.addRow("Высота съемки H:", self.spin_alt)

        # Overlap
        self.spin_overlap = QSpinBox(area_group)
        self.spin_overlap.setRange(0, 90)
        self.spin_overlap.setValue(20)
        self.spin_overlap.setSuffix(" %")
        area_layout.addRow("Перекрытие кадров:", self.spin_overlap)

        layout.addWidget(area_group)

        # 2. Mission Mode & Storage Group
        mode_group = QGroupBox("Режим работы борта", container)
        mode_layout = QFormLayout(mode_group)
        mode_layout.setContentsMargins(6, 6, 6, 6)

        self.combo_mode = QComboBox(mode_group)
        self.combo_mode.addItem("Сбор данных (без ИИ)", MissionMode.COLLECTION_ONLY)
        self.combo_mode.addItem("Последовательный бортовой ИИ", MissionMode.INFERENCE_SEQUENTIAL)
        mode_layout.addRow("Режим миссии:", self.combo_mode)

        # Storage policies checkboxes
        self.cb_save_images = QCheckBox("Сохранять снимки на борт", mode_group)
        self.cb_save_images.setChecked(True)
        mode_layout.addRow("Снимки:", self.cb_save_images)

        self.cb_save_metadata = QCheckBox("Сохранять метаданные и детекции (JSON)", mode_group)
        self.cb_save_metadata.setChecked(True)
        mode_layout.addRow("Метаданные:", self.cb_save_metadata)

        layout.addWidget(mode_group)

        # 3. Validation errors message box
        self.lbl_errors = QLabel("", container)
        self.lbl_errors.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.lbl_errors.setWordWrap(True)
        self.lbl_errors.setVisible(False)
        layout.addWidget(self.lbl_errors)

        # 4. Mission Planning & Start Actions
        actions_group = QGroupBox("Управление полетным заданием", container)
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_plan = QPushButton("📐 Рассчитать маршрут", actions_group)
        self.btn_plan.clicked.connect(self._on_plan_clicked)
        actions_layout.addWidget(self.btn_plan)

        self.lbl_route_summary = QLabel("Маршрут не рассчитан", actions_group)
        self.lbl_route_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(self.lbl_route_summary)

        self.btn_send = QPushButton("🚀 Отправить задание и Старт", actions_group)
        self.btn_send.setStyleSheet("background-color: #2b8a3e; font-weight: bold;")
        self.btn_send.clicked.connect(self._on_send_clicked)
        actions_layout.addWidget(self.btn_send)

        layout.addWidget(actions_group)

        # 5. Emergency Stop Actions Group
        stop_group = QGroupBox("Аварийное управление", container)
        stop_layout = QHBoxLayout(stop_group)
        stop_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_land = QPushButton("⬇ Посадка (Land)", stop_group)
        self.btn_land.setStyleSheet("background-color: #c92a2a; color: white;")
        self.btn_land.clicked.connect(lambda: self.stop_requested.emit(StopAction.LAND))
        stop_layout.addWidget(self.btn_land)

        self.btn_rth = QPushButton("🏠 Возврат домой (RTH)", stop_group)
        self.btn_rth.setStyleSheet("background-color: #d9480f; color: white;")
        self.btn_rth.clicked.connect(lambda: self.stop_requested.emit(StopAction.RTH))
        stop_layout.addWidget(self.btn_rth)

        layout.addWidget(stop_group)

        layout.addStretch(1)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def set_flight_locked(self, locked: bool) -> None:
        """Lock or unlock parameter editing while drone is in active flight."""
        self._is_flight_locked = locked

        # Parameter fields and launch buttons are disabled during flight
        self.combo_cs.setEnabled(not locked)
        self.spin_x1.setEnabled(not locked)
        self.spin_y1.setEnabled(not locked)
        self.spin_x2.setEnabled(not locked)
        self.spin_y2.setEnabled(not locked)
        self.spin_alt.setEnabled(not locked)
        self.spin_overlap.setEnabled(not locked)
        self.combo_mode.setEnabled(not locked)
        self.cb_save_images.setEnabled(not locked)
        self.cb_save_metadata.setEnabled(not locked)
        self.btn_plan.setEnabled(not locked)
        self.btn_send.setEnabled(not locked)

        # Stop and abort buttons ALWAYS remain enabled!
        self.btn_land.setEnabled(True)
        self.btn_rth.setEnabled(True)

    def set_validation_errors(self, errors: Sequence[str]) -> None:
        """Display validation errors banner or hide if empty."""
        if errors:
            self.lbl_errors.setText("\n".join(f"• {e}" for e in errors))
            self.lbl_errors.setVisible(True)
        else:
            self.lbl_errors.clear()
            self.lbl_errors.setVisible(False)

    def set_calculated_waypoints(self, waypoints: Sequence[Waypoint]) -> None:
        """Store planned waypoints and update UI summary."""
        self._planned_waypoints = list(waypoints)
        count = len(self._planned_waypoints)
        if count > 0:
            self.lbl_route_summary.setText(f"Маршрут готов: {count} точек съемки")
            self.btn_send.setEnabled(not self._is_flight_locked)
        else:
            self.lbl_route_summary.setText("Маршрут пуст")
            self.btn_send.setEnabled(False)

    # -------------------------------------------------------------------------
    # Internal actions & validation
    # -------------------------------------------------------------------------
    def _collect_save_policy(self) -> SavePolicy | None:
        save_img = self.cb_save_images.isChecked()
        save_meta = self.cb_save_metadata.isChecked()
        if save_img and save_meta:
            return SavePolicy.ALL
        if save_img:
            return SavePolicy.IMAGES_ONLY
        if save_meta:
            return SavePolicy.METADATA_ONLY
        return None

    def _validate_inputs(self) -> list[str]:
        errors: list[str] = []
        x1, y1 = self.spin_x1.value(), self.spin_y1.value()
        x2, y2 = self.spin_x2.value(), self.spin_y2.value()

        if abs(x2 - x1) < 1e-4 or abs(y2 - y1) < 1e-4:
            errors.append("Полетная зона имеет нулевую ширину или высоту.")

        if self.spin_alt.value() <= 0:
            errors.append("Высота съемки должна быть больше нуля.")

        policy = self._collect_save_policy()
        if policy is None:
            errors.append("Необходимо выбрать сохранение снимков или метаданных.")

        mode = self.combo_mode.currentData()
        if mode == MissionMode.COLLECTION_ONLY and policy == SavePolicy.METADATA_ONLY:
            errors.append("В режиме 'Только сбор' сохранение снимков обязательно (SavePolicy.METADATA_ONLY недопустимо).")

        return errors

    def _on_plan_clicked(self) -> None:
        errors = self._validate_inputs()
        if errors:
            self.set_validation_errors(errors)
            return

        self.set_validation_errors([])
        policy = self._collect_save_policy() or SavePolicy.ALL
        mode = self.combo_mode.currentData()
        cs = self.combo_cs.currentData()

        # Build PlannerParams
        optics = SensorOptics(focal_length_mm=4.5, sensor_width_mm=6.4, sensor_height_mm=4.8)
        params = PlannerParams(
            corner_tl=(self.spin_x1.value(), self.spin_y1.value()),
            corner_br=(self.spin_x2.value(), self.spin_y2.value()),
            coordinate_system=cs,
            altitude_m=self.spin_alt.value(),
            optics=optics,
            overlap=self.spin_overlap.value() / 100.0,
            mode=mode,
            save_policy=policy,
        )
        self.plan_requested.emit(params)

    def _on_send_clicked(self) -> None:
        errors = self._validate_inputs()
        if not self._planned_waypoints:
            errors.append("Сначала необходимо рассчитать маршрут (список точек пуст).")

        if errors:
            self.set_validation_errors(errors)
            return

        self.set_validation_errors([])
        policy = self._collect_save_policy() or SavePolicy.ALL
        mode = self.combo_mode.currentData()
        cs = self.combo_cs.currentData()

        task = MissionTask(
            coordinate_system=cs,
            mode=mode,
            save_policy=policy,
            waypoints=self._planned_waypoints,
        )
        self.send_mission_requested.emit(task)
