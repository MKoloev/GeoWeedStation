"""Flight mission planning, SSH execution control, and real-time detection stream widget."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.core.mission_planner import PlannerParams
from geo_weed_station.schemas.entities import CoordinateSystem, SensorOptics
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy, Waypoint
from geo_weed_station.schemas.onboard_api import StopAction
from geo_weed_station.ui.widgets.live_stream_widget import LiveStreamWidget
from geo_weed_station.workers.ssh_mission_worker import (
    DEFAULT_SSH_COMMAND,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_PASSWORD,
    DEFAULT_SSH_PORT,
    DEFAULT_SSH_USER,
)

logger = logging.getLogger(__name__)


class MissionControl(QWidget):
    """Mission control UI: tabbed layout for Skyris live detection stream, onboard launcher and logs."""

    plan_requested = Signal(PlannerParams)
    send_mission_requested = Signal(MissionTask)
    stop_requested = Signal(StopAction)

    start_drone_mission_requested = Signal(dict)
    emergency_land_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._planned_waypoints: list[Waypoint] = []
        self._is_flight_locked = False

        self._init_compat_attributes()
        self._init_ui()

    def _init_compat_attributes(self) -> None:
        """Initialize invisible headless attributes for backward compatibility with legacy tests."""
        self.spin_x1 = QDoubleSpinBox()
        self.spin_y1 = QDoubleSpinBox()
        self.spin_x2 = QDoubleSpinBox()
        self.spin_x2.setValue(20.0)
        self.spin_y2 = QDoubleSpinBox()
        self.spin_y2.setValue(20.0)
        self.spin_alt = QDoubleSpinBox()
        self.spin_alt.setValue(10.0)
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setValue(20)

        self.combo_cs = QComboBox()
        self.combo_cs.addItem("local_aruco", CoordinateSystem.LOCAL_ARUCO)
        self.combo_cs.addItem("wgs84", CoordinateSystem.WGS84)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("collection", MissionMode.COLLECTION_ONLY)
        self.combo_mode.addItem("inference", MissionMode.INFERENCE_SEQUENTIAL)

        self.cb_save_images = QCheckBox()
        self.cb_save_images.setChecked(True)
        self.cb_save_metadata = QCheckBox()
        self.cb_save_metadata.setChecked(True)

        self.btn_plan = QPushButton("Рассчитать")
        self.btn_plan.clicked.connect(self._on_plan_clicked)

        self.lbl_route_summary = QLabel("Маршрут не рассчитан")

        self.btn_send = QPushButton("Старт REST")
        self.btn_send.clicked.connect(self._on_send_clicked)

        self.btn_rth = QPushButton("RTH")
        self.btn_rth.clicked.connect(lambda: self.stop_requested.emit(StopAction.RTH))

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # =====================================================================
        # 1. TOP PERSISTENT FLIGHT ACTION BAR (Always visible across all tabs)
        # =====================================================================
        top_bar = QFrame(self)
        top_bar.setFrameShape(QFrame.Shape.StyledPanel)
        top_bar.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #333333; border-radius: 6px; padding: 4px;"
        )
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 6, 6, 6)
        top_layout.setSpacing(12)

        # Launch Button
        self.btn_start_drone_mission = QPushButton("🚀 Запустить миссию на дроне", top_bar)
        self.btn_start_drone_mission.setStyleSheet(
            "background-color: #2b8a3e; color: #ffffff; font-weight: bold; "
            "font-size: 13px; padding: 10px 18px; border-radius: 4px;"
        )
        self.btn_start_drone_mission.clicked.connect(self._on_start_drone_mission_clicked)
        top_layout.addWidget(self.btn_start_drone_mission)

        # Emergency Land Button
        self.btn_emergency_land = QPushButton("⬇ Экстренная посадка (Land)", top_bar)
        self.btn_emergency_land.setStyleSheet(
            "background-color: #c92a2a; color: #ffffff; font-weight: bold; "
            "font-size: 13px; padding: 10px 18px; border-radius: 4px;"
        )
        self.btn_emergency_land.clicked.connect(self._on_emergency_land_clicked)
        top_layout.addWidget(self.btn_emergency_land)

        self.btn_land = self.btn_emergency_land

        # Separator line
        sep_top = QFrame(top_bar)
        sep_top.setFrameShape(QFrame.Shape.VLine)
        sep_top.setFrameShadow(QFrame.Shadow.Sunken)
        top_layout.addWidget(sep_top)

        # Status and Progress Display
        self.lbl_status_badge = QLabel("Готов к запуску", top_bar)
        self.lbl_status_badge.setStyleSheet(
            "background-color: #212529; color: #a9e34b; font-weight: bold; "
            "padding: 5px 12px; border-radius: 4px; border: 1px solid #343a40;"
        )
        top_layout.addWidget(self.lbl_status_badge)

        self.lbl_flight_progress = QLabel("Контрольная точка: —", top_bar)
        self.lbl_flight_progress.setStyleSheet("font-weight: bold; color: #ced4da; font-size: 12px;")
        top_layout.addWidget(self.lbl_flight_progress)

        top_layout.addStretch(1)

        # Validation errors banner inside top area
        self.lbl_errors = QLabel("", top_bar)
        self.lbl_errors.setStyleSheet("color: #ff6b6b; font-weight: bold; padding: 2px 6px;")
        self.lbl_errors.setWordWrap(True)
        self.lbl_errors.setVisible(False)
        top_layout.addWidget(self.lbl_errors)

        main_layout.addWidget(top_bar)

        # =====================================================================
        # 2. MAIN TABBED SUB-WINDOWS ("по окнам"): Video Stream vs Onboard Logs
        # =====================================================================
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #333333; background: #181818; border-radius: 4px; } "
            "QTabBar::tab { font-weight: bold; font-size: 12px; padding: 8px 16px; margin-right: 4px; } "
            "QTabBar::tab:selected { background: #2d3748; color: #63b3ed; border-bottom: 2px solid #3182ce; }"
        )

        # ---------------------------------------------------------------------
        # Tab 1: Video Stream with Weed Detections & Single Snapshot Studio
        # ---------------------------------------------------------------------
        self.tab_stream = QWidget(self.tabs)
        tab_stream_layout = QVBoxLayout(self.tab_stream)
        tab_stream_layout.setContentsMargins(4, 4, 4, 4)

        self.live_stream_widget = LiveStreamWidget(
            host=DEFAULT_SSH_HOST, port=8080, parent=self.tab_stream
        )
        self.cb_auto_snapshot = self.live_stream_widget.btn_auto_snapshot
        tab_stream_layout.addWidget(self.live_stream_widget)

        self.tabs.addTab(self.tab_stream, "🎥 Видеотрансляция и детекция сорняков")

        # ---------------------------------------------------------------------
        # Tab 2: Drone Onboard Settings & Real-Time SSH Console Logs
        # ---------------------------------------------------------------------
        self.tab_onboard = QWidget(self.tabs)
        tab_onboard_layout = QVBoxLayout(self.tab_onboard)
        tab_onboard_layout.setContentsMargins(8, 8, 8, 8)
        tab_onboard_layout.setSpacing(8)

        # Parameters Group (Compact)
        params_group = QGroupBox("Параметры связи с бортовым компьютером Skyris (Orange Pi 5 Pro)", self.tab_onboard)
        params_layout = QVBoxLayout(params_group)
        params_layout.setContentsMargins(8, 8, 8, 8)
        params_layout.setSpacing(6)

        conn_form = QFormLayout()
        conn_form.setContentsMargins(2, 2, 2, 2)

        row_conn = QHBoxLayout()
        self.edit_ssh_host = QLineEdit(DEFAULT_SSH_HOST, params_group)
        self.spin_ssh_port = QSpinBox(params_group)
        self.spin_ssh_port.setRange(1, 65535)
        self.spin_ssh_port.setValue(DEFAULT_SSH_PORT)
        row_conn.addWidget(QLabel("IP дрона:"))
        row_conn.addWidget(self.edit_ssh_host)
        row_conn.addWidget(QLabel("Порт SSH:"))
        row_conn.addWidget(self.spin_ssh_port)

        self.edit_ssh_user = QLineEdit(DEFAULT_SSH_USER, params_group)
        self.edit_ssh_pass = QLineEdit(DEFAULT_SSH_PASSWORD, params_group)
        self.edit_ssh_pass.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        row_conn.addWidget(QLabel("Логин:"))
        row_conn.addWidget(self.edit_ssh_user)
        row_conn.addWidget(QLabel("Пароль:"))
        row_conn.addWidget(self.edit_ssh_pass)
        conn_form.addRow("Подключение:", row_conn)

        self.edit_ssh_cmd = QPlainTextEdit(DEFAULT_SSH_COMMAND, params_group)
        self.edit_ssh_cmd.setMaximumHeight(70)
        self.edit_ssh_cmd.setStyleSheet("font-family: monospace; font-size: 10px;")
        conn_form.addRow("Команда запуска:", self.edit_ssh_cmd)

        self.btn_reset_cmd = QPushButton("Восстановить проверенную команду", params_group)
        self.btn_reset_cmd.clicked.connect(
            lambda: self.edit_ssh_cmd.setPlainText(DEFAULT_SSH_COMMAND)
        )
        conn_form.addRow("", self.btn_reset_cmd)

        params_layout.addLayout(conn_form)
        tab_onboard_layout.addWidget(params_group)

        # Console Logs Group (Spacious, full height)
        log_group = QGroupBox("Бортовые логи миссии в реальном времени (SSH Console)", self.tab_onboard)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(4)

        log_tools = QHBoxLayout()
        self.lbl_mission_progress = QLabel("Прогресс: Ожидание запуска", log_group)
        self.lbl_mission_progress.setStyleSheet("font-weight: bold; color: #ced4da;")
        log_tools.addWidget(self.lbl_mission_progress)

        log_tools.addStretch(1)

        self.btn_clear_log = QPushButton("Очистить", log_group)
        self.btn_clear_log.clicked.connect(self.clear_log)
        log_tools.addWidget(self.btn_clear_log)

        self.btn_copy_log = QPushButton("Копировать", log_group)
        self.btn_copy_log.clicked.connect(self._copy_log_to_clipboard)
        log_tools.addWidget(self.btn_copy_log)

        log_layout.addLayout(log_tools)

        self.log_console = QPlainTextEdit(log_group)
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(10000)
        self.log_console.setStyleSheet(
            "background-color: #121212; color: #dcdcdc; font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 11px; border: 1px solid #333333; border-radius: 4px;"
        )
        log_layout.addWidget(self.log_console, 1)

        tab_onboard_layout.addWidget(log_group, 1)

        self.tabs.addTab(self.tab_onboard, "📟 Управление бортом и логи SSH")

        main_layout.addWidget(self.tabs, 1)

    # -------------------------------------------------------------------------
    # Public methods & Slots
    # -------------------------------------------------------------------------
    @Slot(str)
    def append_log(self, text: str) -> None:
        """Append log line to mission console and scroll to bottom."""
        self.log_console.appendPlainText(text)
        self.log_console.moveCursor(QTextCursor.MoveOperation.End)

    @Slot()
    def clear_log(self) -> None:
        """Clear mission console."""
        self.log_console.clear()
        self.lbl_mission_progress.setText("Прогресс: Очищено")

    @Slot(int, int)
    def set_mission_progress(self, current_wp: int, total_wps: int) -> None:
        """Update waypoint progress indicator across both tabs."""
        msg = f"Точка {current_wp} из {total_wps}"
        self.lbl_mission_progress.setText(f"Прогресс: {msg}")
        self.lbl_flight_progress.setText(f"Контрольная точка: {msg}")

    def get_ssh_launch_params(self) -> dict:
        """Collect current SSH connection parameters and launch command."""
        return {
            "host": self.edit_ssh_host.text().strip(),
            "port": self.spin_ssh_port.value(),
            "username": self.edit_ssh_user.text().strip(),
            "password": self.edit_ssh_pass.text(),
            "command": self.edit_ssh_cmd.toPlainText().strip(),
        }

    def set_flight_locked(self, locked: bool) -> None:
        """Lock or unlock parameter editing while drone is in active flight."""
        self._is_flight_locked = locked

        self.btn_start_drone_mission.setEnabled(not locked)
        self.edit_ssh_host.setEnabled(not locked)
        self.spin_ssh_port.setEnabled(not locked)
        self.edit_ssh_user.setEnabled(not locked)
        self.edit_ssh_pass.setEnabled(not locked)
        self.edit_ssh_cmd.setEnabled(not locked)
        self.btn_reset_cmd.setEnabled(not locked)

        # Legacy compat widgets
        self.btn_plan.setEnabled(not locked)
        self.btn_send.setEnabled(not locked)
        self.spin_x1.setEnabled(not locked)
        self.spin_y1.setEnabled(not locked)
        self.spin_x2.setEnabled(not locked)
        self.spin_y2.setEnabled(not locked)
        self.spin_alt.setEnabled(not locked)
        self.spin_overlap.setEnabled(not locked)
        self.combo_mode.setEnabled(not locked)
        self.cb_save_images.setEnabled(not locked)
        self.cb_save_metadata.setEnabled(not locked)

        if locked:
            self.btn_start_drone_mission.setText("⏳ Миссия выполняется на дроне...")
            self.lbl_status_badge.setText("⏳ Полет выполняется")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #2b8a3e; color: #ffffff; font-weight: bold; "
                "padding: 5px 12px; border-radius: 4px;"
            )
        else:
            self.btn_start_drone_mission.setText("🚀 Запустить миссию на дроне")
            self.lbl_status_badge.setText("Готов к запуску")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #212529; color: #a9e34b; font-weight: bold; "
                "padding: 5px 12px; border-radius: 4px; border: 1px solid #343a40;"
            )

        # Emergency land ALWAYS remains enabled!
        self.btn_emergency_land.setEnabled(True)
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
    # Internal actions & handlers
    # -------------------------------------------------------------------------
    def _on_start_drone_mission_clicked(self) -> None:
        params = self.get_ssh_launch_params()
        if not params["host"]:
            self.set_validation_errors(["Укажите IP-адрес дрона для подключения SSH."])
            return
        if not params["command"]:
            self.set_validation_errors(["Команда запуска миссии не может быть пустой."])
            return

        self.set_validation_errors([])

        # Switch to video stream tab so operator immediately monitors detections
        self.tabs.setCurrentIndex(0)

        # Auto-snapshot activates if checked in stream studio
        if self.live_stream_widget.btn_auto_snapshot.isChecked():
            self.live_stream_widget.start_auto_snapshot(1.0)

        self.start_drone_mission_requested.emit(params)

    def _on_emergency_land_clicked(self) -> None:
        # Stop auto-snapshot immediately upon emergency landing
        self.live_stream_widget.stop_auto_snapshot()
        self.lbl_status_badge.setText("Посадка (Land)...")
        self.lbl_status_badge.setStyleSheet(
            "background-color: #c92a2a; color: #ffffff; font-weight: bold; "
            "padding: 5px 12px; border-radius: 4px;"
        )
        self.emergency_land_requested.emit()
        self.stop_requested.emit(StopAction.LAND)

    def _copy_log_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.log_console.toPlainText())
            self.lbl_mission_progress.setText("Прогресс: Логи скопированы в буфер")

    # -------------------------------------------------------------------------
    # Legacy Planner Handlers (for backward compatibility with tests)
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
            errors.append(
                "В режиме 'Только сбор' сохранение снимков обязательно (SavePolicy.METADATA_ONLY недопустимо)."
            )

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
