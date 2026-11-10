"""Flight mission planning, SSH execution control, and real-time detection stream widget."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal, Slot
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
    QScrollArea,
    QSpinBox,
    QSplitter,
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
    """Mission control UI: flight planning, live detection video stream, and SSH onboard launcher."""

    plan_requested = Signal(PlannerParams)
    send_mission_requested = Signal(MissionTask)
    stop_requested = Signal(StopAction)

    start_drone_mission_requested = Signal(dict)
    emergency_land_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._planned_waypoints: list[Waypoint] = []
        self._is_flight_locked = False

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # Main horizontal splitter: Left panel (Settings & Controls) | Right panel (Video & Logs)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # =====================================================================
        # LEFT PANE: Mission Settings, Planning & Launch Controls (Scrollable)
        # =====================================================================
        scroll = QScrollArea(self.main_splitter)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)

        # 1. Skyris Drone SBC Launch Group (SSH)
        ssh_group = QGroupBox("🚀 Бортовой комплекс Skyris (Orange Pi 5 Pro)", left_container)
        ssh_layout = QVBoxLayout(ssh_group)
        ssh_layout.setContentsMargins(8, 8, 8, 8)
        ssh_layout.setSpacing(8)

        # Action Buttons Row: Big Start & Emergency Land
        btn_row = QHBoxLayout()
        self.btn_start_drone_mission = QPushButton("🚀 Запустить миссию на дроне", ssh_group)
        self.btn_start_drone_mission.setStyleSheet(
            "background-color: #2b8a3e; color: #ffffff; font-weight: bold; "
            "font-size: 13px; padding: 10px 14px; border-radius: 4px;"
        )
        self.btn_start_drone_mission.clicked.connect(self._on_start_drone_mission_clicked)
        btn_row.addWidget(self.btn_start_drone_mission, 2)

        self.btn_emergency_land = QPushButton("⬇ Экстренная посадка (Land)", ssh_group)
        self.btn_emergency_land.setStyleSheet(
            "background-color: #c92a2a; color: #ffffff; font-weight: bold; "
            "font-size: 13px; padding: 10px 14px; border-radius: 4px;"
        )
        self.btn_emergency_land.clicked.connect(self._on_emergency_land_clicked)
        btn_row.addWidget(self.btn_emergency_land, 1)

        ssh_layout.addLayout(btn_row)

        # Expandable/Configurable SSH parameters
        ssh_form = QFormLayout()
        ssh_form.setContentsMargins(2, 4, 2, 4)

        row_conn = QHBoxLayout()
        self.edit_ssh_host = QLineEdit(DEFAULT_SSH_HOST, ssh_group)
        self.spin_ssh_port = QSpinBox(ssh_group)
        self.spin_ssh_port.setRange(1, 65535)
        self.spin_ssh_port.setValue(DEFAULT_SSH_PORT)
        row_conn.addWidget(QLabel("IP:"))
        row_conn.addWidget(self.edit_ssh_host)
        row_conn.addWidget(QLabel("Порт:"))
        row_conn.addWidget(self.spin_ssh_port)
        ssh_form.addRow("Подключение SSH:", row_conn)

        row_auth = QHBoxLayout()
        self.edit_ssh_user = QLineEdit(DEFAULT_SSH_USER, ssh_group)
        self.edit_ssh_pass = QLineEdit(DEFAULT_SSH_PASSWORD, ssh_group)
        self.edit_ssh_pass.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        row_auth.addWidget(QLabel("Логин:"))
        row_auth.addWidget(self.edit_ssh_user)
        row_auth.addWidget(QLabel("Пароль:"))
        row_auth.addWidget(self.edit_ssh_pass)
        ssh_form.addRow("Авторизация:", row_auth)

        self.edit_ssh_cmd = QPlainTextEdit(DEFAULT_SSH_COMMAND, ssh_group)
        self.edit_ssh_cmd.setMaximumHeight(85)
        self.edit_ssh_cmd.setStyleSheet("font-family: monospace; font-size: 10px;")
        ssh_form.addRow("Команда запуска:", self.edit_ssh_cmd)

        self.btn_reset_cmd = QPushButton("Восстановить проверенную команду", ssh_group)
        self.btn_reset_cmd.clicked.connect(
            lambda: self.edit_ssh_cmd.setPlainText(DEFAULT_SSH_COMMAND)
        )
        ssh_form.addRow("", self.btn_reset_cmd)

        ssh_layout.addLayout(ssh_form)
        left_layout.addWidget(ssh_group)

        # 2. Survey Area & Coordinate System Group
        area_group = QGroupBox("Полетная зона полигона", left_container)
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

        left_layout.addWidget(area_group)

        # 3. Mission Mode & Storage Group
        mode_group = QGroupBox("Режим работы борта", left_container)
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

        left_layout.addWidget(mode_group)

        # 4. Validation errors message box
        self.lbl_errors = QLabel("", left_container)
        self.lbl_errors.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.lbl_errors.setWordWrap(True)
        self.lbl_errors.setVisible(False)
        left_layout.addWidget(self.lbl_errors)

        # 5. Mission Planning & Start Actions (REST API)
        actions_group = QGroupBox("Управление полетным заданием (REST)", left_container)
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_plan = QPushButton("📐 Рассчитать маршрут", actions_group)
        self.btn_plan.clicked.connect(self._on_plan_clicked)
        actions_layout.addWidget(self.btn_plan)

        self.lbl_route_summary = QLabel("Маршрут не рассчитан", actions_group)
        self.lbl_route_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(self.lbl_route_summary)

        self.btn_send = QPushButton("🚀 Отправить задание и Старт (REST)", actions_group)
        self.btn_send.setStyleSheet("background-color: #2b8a3e; font-weight: bold;")
        self.btn_send.clicked.connect(self._on_send_clicked)
        actions_layout.addWidget(self.btn_send)

        left_layout.addWidget(actions_group)

        # 6. Emergency Stop Actions Group (REST fallback)
        stop_group = QGroupBox("Аварийное управление", left_container)
        stop_layout = QHBoxLayout(stop_group)
        stop_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_land = QPushButton("⬇ Посадка (Land)", stop_group)
        self.btn_land.setStyleSheet("background-color: #c92a2a; color: white;")
        self.btn_land.clicked.connect(self._on_emergency_land_clicked)
        stop_layout.addWidget(self.btn_land)

        self.btn_rth = QPushButton("🏠 Возврат домой (RTH)", stop_group)
        self.btn_rth.setStyleSheet("background-color: #d9480f; color: white;")
        self.btn_rth.clicked.connect(lambda: self.stop_requested.emit(StopAction.RTH))
        stop_layout.addWidget(self.btn_rth)

        left_layout.addWidget(stop_group)

        left_layout.addStretch(1)
        scroll.setWidget(left_container)
        self.main_splitter.addWidget(scroll)

        # =====================================================================
        # RIGHT PANE: Live Video Stream Viewport + Console Log Monitor
        # =====================================================================
        self.right_splitter = QSplitter(Qt.Orientation.Vertical, self.main_splitter)

        # 1. Live video stream widget
        self.live_stream_widget = LiveStreamWidget(
            host=DEFAULT_SSH_HOST, port=8080, parent=self.right_splitter
        )
        self.right_splitter.addWidget(self.live_stream_widget)

        # 2. Remote console output group
        log_group = QGroupBox("Бортовые логи миссии (SSH Console)", self.right_splitter)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(4)

        # Log toolbar
        log_tools = QHBoxLayout()
        self.lbl_mission_progress = QLabel("Прогресс: Ожидание запуска", log_group)
        self.lbl_mission_progress.setStyleSheet("font-weight: bold; color: #aaaaaa;")
        log_tools.addWidget(self.lbl_mission_progress)

        log_tools.addStretch(1)

        self.btn_clear_log = QPushButton("Очистить", log_group)
        self.btn_clear_log.clicked.connect(self.clear_log)
        log_tools.addWidget(self.btn_clear_log)

        self.btn_copy_log = QPushButton("Копировать", log_group)
        self.btn_copy_log.clicked.connect(self._copy_log_to_clipboard)
        log_tools.addWidget(self.btn_copy_log)

        log_layout.addLayout(log_tools)

        # Console text area
        self.log_console = QPlainTextEdit(log_group)
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(8000)
        self.log_console.setStyleSheet(
            "background-color: #1a1a1a; color: #dcdcdc; font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 11px; border: 1px solid #333333; border-radius: 4px;"
        )
        log_layout.addWidget(self.log_console, 1)

        self.right_splitter.addWidget(log_group)

        # Set right splitter sizes (video stream larger, console log lower)
        self.right_splitter.setSizes([380, 240])

        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setSizes([430, 670])

        main_layout.addWidget(self.main_splitter)

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
        """Update waypoint progress indicator."""
        self.lbl_mission_progress.setText(f"Прогресс: Точка {current_wp} из {total_wps}")

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

        self.btn_start_drone_mission.setEnabled(not locked)
        self.edit_ssh_host.setEnabled(not locked)
        self.spin_ssh_port.setEnabled(not locked)
        self.edit_ssh_user.setEnabled(not locked)
        self.edit_ssh_pass.setEnabled(not locked)
        self.edit_ssh_cmd.setEnabled(not locked)
        self.btn_reset_cmd.setEnabled(not locked)

        if locked:
            self.btn_start_drone_mission.setText("⏳ Миссия выполняется на дроне...")
        else:
            self.btn_start_drone_mission.setText("🚀 Запустить миссию на дроне")

        # Stop and abort buttons ALWAYS remain enabled!
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

    def _on_start_drone_mission_clicked(self) -> None:
        params = self.get_ssh_launch_params()
        if not params["host"]:
            self.set_validation_errors(["Укажите IP-адрес дрона для подключения SSH."])
            return
        if not params["command"]:
            self.set_validation_errors(["Команда запуска миссии не может быть пустой."])
            return

        self.set_validation_errors([])
        self.start_drone_mission_requested.emit(params)

    def _on_emergency_land_clicked(self) -> None:
        self.emergency_land_requested.emit()
        self.stop_requested.emit(StopAction.LAND)

    def _copy_log_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.log_console.toPlainText())
            self.lbl_mission_progress.setText("Прогресс: Логи скопированы в буфер")
