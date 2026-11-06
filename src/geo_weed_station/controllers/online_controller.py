"""Online mode controller managing drone telemetry, mission dispatch, live tracking and aborts."""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QThread, Signal, Slot

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.core.mission_planner import generate_waypoints
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.schemas.mission import MissionTask
from geo_weed_station.schemas.onboard_api import (
    StartResponse,
    StopAction,
    StopResponse,
    SummaryDetections,
)
from geo_weed_station.schemas.telemetry import (
    ConnectionState,
    DroneStatus,
    FlightState,
)
from geo_weed_station.ui.app_mode import AppMode
from geo_weed_station.ui.dialogs.confirm_dialog import ConfirmDialog
from geo_weed_station.ui.main_window import (
    TAB_MISSION,
    MainWindow,
)
from geo_weed_station.ui.widgets.map_canvas import MapCanvas
from geo_weed_station.ui.widgets.mission_control import MissionControl
from geo_weed_station.ui.widgets.telemetry_panel import TelemetryPanel
from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.command_worker import CommandWorker
from geo_weed_station.workers.telemetry_worker import TelemetryWorker

logger = logging.getLogger(__name__)


class OnlineController(QObject):
    """Orchestrates online drone communication, mission upload, live telemetry, and emergency stops."""

    connection_state_changed = Signal(ConnectionState)
    status_updated = Signal(DroneStatus)

    def __init__(
        self,
        window: MainWindow,
        map_ctrl: MapController,
        settings: AppSettings,
        map_canvas: MapCanvas,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._map_ctrl = map_ctrl
        self._settings = settings
        self.map_canvas = map_canvas

        # Online UI widgets
        self.telemetry_panel = TelemetryPanel()
        self.mission_control = MissionControl()

        # Network client and workers
        self._client: DroneClient | None = None
        self._telemetry_worker: TelemetryWorker | None = None
        self._command_worker: CommandWorker | None = None
        self._telemetry_thread: QThread | None = None
        self._command_thread: QThread | None = None

        self._last_status: DroneStatus | None = None
        self._flight_start_time: float | None = None
        self._active_run_id: str | None = None
        self._summary_requested: bool = False

        self._embed_widgets_in_ui()
        self._wire_signals()

    # -------------------------------------------------------------------------
    # Setup & UI embedding
    # -------------------------------------------------------------------------
    def _embed_widgets_in_ui(self) -> None:
        self._window.set_tab_widget(TAB_MISSION, self.mission_control)
        self._window.add_mode_panel(AppMode.ONLINE, self.telemetry_panel, "Телеметрия БПЛА")

    def _wire_signals(self) -> None:
        self.mission_control.plan_requested.connect(self.plan)
        self.mission_control.send_mission_requested.connect(self.send_mission)
        self.mission_control.stop_requested.connect(self.stop)

    # -------------------------------------------------------------------------
    # Network Connection Lifecycle
    # -------------------------------------------------------------------------
    @Slot(str, int)
    def connect_to(self, host: str, port: int) -> None:
        """Connect to onboard REST API and initiate background telemetry polling."""
        self.disconnect()

        base_url = f"http://{host}:{port}"
        logger.info("Connecting to drone onboard API at %s", base_url)
        self._window.status_bar_panel.set_connection_state(ConnectionState.CONNECTING)
        self._window.status_bar_panel.set_message(f"Подключение к БПЛА ({host}:{port})...")

        self._client = DroneClient(base_url=base_url, timeout_s=self._settings.network.request_timeout_s)

        # 1. Telemetry Worker
        self._telemetry_worker = TelemetryWorker(client=self._client, settings=self._settings.network)
        self._telemetry_worker.connection_changed.connect(self._on_connection_changed)
        self._telemetry_worker.status_received.connect(self.on_status)

        self._telemetry_thread = start_in_thread(self._telemetry_worker, "telemetry_worker_thread")
        self._window.register_thread(self._telemetry_thread, self._telemetry_worker)

        # 2. Command Worker
        self._command_worker = CommandWorker(client=self._client)
        self._command_worker.mission_started.connect(self._on_mission_started)
        self._command_worker.mission_stopped.connect(self._on_mission_stopped)
        self._command_worker.summary_loaded.connect(self._on_summary_loaded)
        self._command_worker.command_failed.connect(self._on_command_failed)

        self._command_thread = start_in_thread(self._command_worker, "command_worker_thread")
        self._window.register_thread(self._command_thread, self._command_worker)

    @Slot()
    def disconnect(self) -> None:
        """Stop background telemetry and command workers."""
        if self._telemetry_worker is not None:
            self._telemetry_worker.request_stop()
            self._telemetry_worker = None
        if self._command_worker is not None:
            self._command_worker.request_stop()
            self._command_worker = None

        self._window.status_bar_panel.set_connection_state(ConnectionState.DISCONNECTED)
        self._window.status_bar_panel.set_message("Соединение с БПЛА разорвано")

    # -------------------------------------------------------------------------
    # Mission Planning & Dispatch
    # -------------------------------------------------------------------------
    @Slot(object)
    def plan(self, params) -> None:
        """Calculate serpentine grid waypoints and render on map."""
        try:
            waypoints = generate_waypoints(params)
            self.mission_control.set_calculated_waypoints(waypoints)
            self.map_canvas.set_route(waypoints)
            self._window.status_bar_panel.set_message(
                f"Маршрут рассчитан: {len(waypoints)} точек съемки"
            )
        except Exception as exc:
            logger.exception("Failed to plan mission: %s", exc)
            self.mission_control.set_validation_errors([f"Ошибка расчета маршрута: {exc}"])

    @Slot(MissionTask)
    def send_mission(self, task: MissionTask) -> None:
        """Send mission to drone onboard computer (requires flight_state == IDLE)."""
        # Check flight state
        if self._last_status is not None and self._last_status.flight_state != FlightState.IDLE:
            err = f"Запуск отклонен: борт находится в состоянии '{self._last_status.flight_state.value}' (требуется IDLE)."
            self.mission_control.set_validation_errors([err])
            return

        if self._command_worker is None:
            self.mission_control.set_validation_errors(["Нет подключения к бортовому комплексу БПЛА!"])
            return

        self._window.status_bar_panel.set_message("Отправка полетного задания на борт...")
        self.mission_control.set_flight_locked(True)
        self._flight_start_time = time.time()
        self._command_worker.start_mission(task)

    @Slot(StopAction)
    def stop(self, action: StopAction) -> None:
        """Issue confirmed emergency stop or return-to-home command."""
        if self._command_worker is None:
            return

        if not ConfirmDialog.confirm_abort(action, parent=self._window):
            return

        self._window.status_bar_panel.set_message(f"Отправка команды остановки ({action.value})...")
        self._command_worker.stop_mission(action)

    # -------------------------------------------------------------------------
    # Live Status & Telemetry Handling
    # -------------------------------------------------------------------------
    @Slot(DroneStatus)
    def on_status(self, status: DroneStatus) -> None:
        """Handle incoming telemetry packet: update UI, marker, and slot duration warning."""
        self._last_status = status
        self.status_updated.emit(status)

        # 1. Update telemetry indicators
        self.telemetry_panel.update_status(status)
        self._window.status_bar_panel.set_counters(captures=status.captures_count, weeds=None)

        # 2. Update live drone marker on map
        if status.current_pose is not None:
            self.map_canvas.set_drone_pose(status.current_pose)

        # 3. Check 10-minute flight slot limit
        if self._flight_start_time is not None:
            elapsed = time.time() - self._flight_start_time
            if elapsed > 600.0:  # 10 minutes exceeded
                self._window.status_bar_panel.set_message(
                    f"Внимание: время полетного слота превышено ({int(elapsed // 60)} мин)!"
                )

        # 4. Check flight completion / abort
        if status.flight_state in (FlightState.COMPLETED, FlightState.ABORTED, FlightState.IDLE):
            if self._flight_start_time is not None:
                self.mission_control.set_flight_locked(False)
                self._flight_start_time = None

            # Fetch mission summary if data ready
            run_id = status.active_run_id or self._active_run_id
            if status.is_data_ready and run_id and self._command_worker is not None and not self._summary_requested:
                self._summary_requested = True
                logger.info("Requesting summary for run %s", run_id)
                self._command_worker.request_summary(run_id)

    # -------------------------------------------------------------------------
    # Command Callbacks
    # -------------------------------------------------------------------------
    def _on_connection_changed(self, state: ConnectionState) -> None:
        self.connection_state_changed.emit(state)
        self._window.status_bar_panel.set_connection_state(state)

    def _on_mission_started(self, resp: StartResponse) -> None:
        self._active_run_id = resp.run_id
        self._summary_requested = False
        self._window.status_bar_panel.set_message(
            f"Миссия запущена (Run ID: {resp.run_id}, точек: {resp.total_waypoints})"
        )

    def _on_mission_stopped(self, resp: StopResponse) -> None:
        self._window.status_bar_panel.set_message(f"Команда остановки выполнена: {resp.status}")

    def _on_summary_loaded(self, summary: SummaryDetections) -> None:
        logger.info("Received mission detections summary from onboard: %s", summary.run_id)
        self._map_ctrl.on_summary(summary)
        self._window.status_bar_panel.set_message(
            f"Данные детекций для запуска {summary.run_id} успешно нанесены на карту"
        )

    def _on_command_failed(self, cmd: str, err: str) -> None:
        logger.error("Command '%s' failed: %s", cmd, err)
        self.mission_control.set_flight_locked(False)
        self._window.status_bar_panel.set_message(f"Ошибка команды {cmd}: {err}")
        self.mission_control.set_validation_errors([f"Ошибка борта ({cmd}): {err}"])
