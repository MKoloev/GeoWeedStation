"""Telemetry monitoring panel displaying real-time drone flight state, coordinates, and captures."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.schemas.telemetry import DroneStatus, FlightState

logger = logging.getLogger(__name__)

STATE_COLORS = {
    FlightState.IDLE: "#868e96",
    FlightState.TAKEOFF: "#339af0",
    FlightState.FLYING_TO_WP: "#228be6",
    FlightState.PROCESSING: "#fab005",
    FlightState.LANDING: "#fd7e14",
    FlightState.COMPLETED: "#40c057",
    FlightState.ABORTED: "#fa5252",
    FlightState.ERROR: "#e03131",
}

STATE_RUSSIAN_LABELS = {
    FlightState.IDLE: "Ожидание (Idle)",
    FlightState.TAKEOFF: "Взлет",
    FlightState.FLYING_TO_WP: "Полет к точке",
    FlightState.PROCESSING: "Съемка / Инференс",
    FlightState.LANDING: "Посадка",
    FlightState.COMPLETED: "Миссия завершена",
    FlightState.ABORTED: "Прервано (Abort)",
    FlightState.ERROR: "Ошибка борта",
}


class TelemetryPanel(QWidget):
    """Panel showing live drone flight status, progress, 6-DOF coordinates and capture counters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        group = QGroupBox("Телеметрия БПЛА (Бортовой комплекс)", self)
        layout = QFormLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)

        # 1. State badge
        self.lbl_state = QLabel("Отключен", group)
        self.lbl_state.setStyleSheet(
            "background-color: #495057; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addRow("Статус борта:", self.lbl_state)

        # 2. Waypoint progress
        self.lbl_progress = QLabel("0 / 0", group)
        self.bar_progress = QProgressBar(group)
        self.bar_progress.setRange(0, 100)
        self.bar_progress.setValue(0)
        self.bar_progress.setTextVisible(False)
        self.bar_progress.setFixedHeight(8)

        prog_layout = QVBoxLayout()
        prog_layout.addWidget(self.lbl_progress)
        prog_layout.addWidget(self.bar_progress)
        layout.addRow("Точки маршрута:", prog_layout)

        # 3. Position & Attitude
        self.lbl_pos_xy = QLabel("X: 0.00 м, Y: 0.00 м", group)
        self.lbl_alt_z = QLabel("0.00 м", group)
        self.lbl_yaw = QLabel("0.0°", group)
        layout.addRow("Координаты:", self.lbl_pos_xy)
        layout.addRow("Высота Z:", self.lbl_alt_z)
        layout.addRow("Курс (Yaw):", self.lbl_yaw)

        # 4. Captures & Active Run
        self.lbl_captures = QLabel("0", group)
        self.lbl_run_id = QLabel("-", group)
        self.lbl_preset = QLabel("-", group)
        layout.addRow("Снимков получено:", self.lbl_captures)
        layout.addRow("Текущий запуск ID:", self.lbl_run_id)
        layout.addRow("Пресет БПЛА:", self.lbl_preset)

        main_layout.addWidget(group)
        main_layout.addStretch(1)

    # -------------------------------------------------------------------------
    # Public slot
    # -------------------------------------------------------------------------
    @Slot(DroneStatus)
    def update_status(self, status: DroneStatus) -> None:
        """Update telemetry indicators from drone status payload."""
        state = status.flight_state
        state_text = STATE_RUSSIAN_LABELS.get(state, state.value)
        color = STATE_COLORS.get(state, "#495057")

        self.lbl_state.setText(state_text)
        self.lbl_state.setStyleSheet(
            f"background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )

        total_wp = max(0, status.total_waypoints)
        cur_wp = max(0, status.current_wp_index)
        self.lbl_progress.setText(f"Точка {cur_wp} из {total_wp}")
        if total_wp > 0:
            pct = int(min(100, round((cur_wp / float(total_wp)) * 100)))
            self.bar_progress.setValue(pct)
        else:
            self.bar_progress.setValue(0)

        if status.current_pose is not None:
            p = status.current_pose
            self.lbl_pos_xy.setText(f"X: {p.x:.2f} м, Y: {p.y:.2f} м")
            self.lbl_alt_z.setText(f"{p.z:.2f} м")
            self.lbl_yaw.setText(f"{p.yaw:.1f}°")
        else:
            self.lbl_pos_xy.setText("X: — , Y: —")
            self.lbl_alt_z.setText("—")
            self.lbl_yaw.setText("—")

        self.lbl_captures.setText(str(status.captures_count))
        self.lbl_run_id.setText(status.active_run_id or "—")
        self.lbl_preset.setText(status.preset)
