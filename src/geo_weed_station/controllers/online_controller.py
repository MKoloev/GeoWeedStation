"""Online mode controller managing drone telemetry, mission dispatch, live tracking, and SSH/Stream workers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

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
from geo_weed_station.workers.live_stream_worker import LiveStreamWorker
from geo_weed_station.workers.ssh_mission_worker import SshMissionWorker
from geo_weed_station.workers.telemetry_worker import TelemetryWorker

logger = logging.getLogger(__name__)


class OnlineController(QObject):
    """Orchestrates online drone communication, mission upload, live telemetry, video streaming and SSH runs."""

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

        # SSH Mission and Live Stream Workers
        self._ssh_worker: SshMissionWorker | None = None
        self._ssh_thread: QThread | None = None
        self._stream_worker: LiveStreamWorker | None = None
        self._stream_thread: QThread | None = None

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
        self._window.clear_default_panels(AppMode.ONLINE)
        self._window.add_mode_panel(AppMode.ONLINE, self.telemetry_panel, "Телеметрия БПЛА")

    def _wire_signals(self) -> None:
        # Existing mission planner & REST commands
        self.mission_control.plan_requested.connect(self.plan)
        self.mission_control.send_mission_requested.connect(self.send_mission)
        self.mission_control.stop_requested.connect(self.stop)

        # SSH Mission launch and abort
        self.mission_control.start_drone_mission_requested.connect(self.start_ssh_mission)
        self.mission_control.emergency_land_requested.connect(self.emergency_land)

        # Live Stream Widget events
        self.mission_control.live_stream_widget.topic_changed.connect(self._on_stream_topic_changed)

    # -------------------------------------------------------------------------
    # Network Connection Lifecycle
    # -------------------------------------------------------------------------
    @Slot(str, int)
    def connect_to(self, host: str, port: int) -> None:
        """Connect to onboard REST API and initiate background telemetry and stream polling."""
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

        # 3. Live Video Stream Worker (default ROS web_video_server port 8080)
        stream_port = getattr(self._settings.stream, "port", 8080) if hasattr(self._settings, "stream") else 8080
        self.start_stream(host=host, port=stream_port)

    @Slot()
    def disconnect(self) -> None:
        """Stop background telemetry, command, stream, and SSH workers."""
        if self._telemetry_worker is not None:
            self._telemetry_worker.request_stop()
            self._telemetry_worker = None
        if self._command_worker is not None:
            self._command_worker.request_stop()
            self._command_worker = None

        self.stop_stream()
        self.stop_ssh_mission()

        self._window.status_bar_panel.set_connection_state(ConnectionState.DISCONNECTED)
        self._window.status_bar_panel.set_message("Соединение с БПЛА разорвано")

    # -------------------------------------------------------------------------
    # Live Video Stream Management
    # -------------------------------------------------------------------------
    @Slot(str, int)
    def start_stream(self, host: str | None = None, port: int | None = None) -> None:
        """Start or restart the background live stream polling worker."""
        self.stop_stream()

        stream_host = host or getattr(self._settings.stream, "host", "10.42.0.1") if hasattr(self._settings, "stream") else "10.42.0.1"
        stream_port = port or getattr(self._settings.stream, "port", 8080) if hasattr(self._settings, "stream") else 8080
        topic = self.mission_control.live_stream_widget.current_topic()
        poll_rate = getattr(self._settings.stream, "poll_rate_hz", 3.0) if hasattr(self._settings, "stream") else 3.0

        self.mission_control.live_stream_widget.set_endpoint(stream_host, stream_port)

        self._stream_worker = LiveStreamWorker(
            host=stream_host,
            port=stream_port,
            topic=topic,
            poll_rate_hz=poll_rate,
        )
        self._stream_worker.frame_ready.connect(self.mission_control.live_stream_widget.set_frame)
        self._stream_worker.status_changed.connect(self.mission_control.live_stream_widget.set_status)

        self._stream_thread = start_in_thread(self._stream_worker, "live_stream_worker_thread")
        self._window.register_thread(self._stream_thread, self._stream_worker)
        logger.info("LiveStreamWorker initiated for %s:%d (topic=%s)", stream_host, stream_port, topic)

    @Slot()
    def stop_stream(self) -> None:
        """Stop background live stream worker."""
        if self._stream_worker is not None:
            self._stream_worker.request_stop()
            self._stream_worker = None
        self._stream_thread = None

    def _on_stream_topic_changed(self, new_topic: str) -> None:
        """Forward UI topic selection to active stream worker."""
        if self._stream_worker is not None:
            self._stream_worker.set_topic(new_topic)

    # -------------------------------------------------------------------------
    # SSH Mission Execution & Emergency Abort (Skyris Orange Pi)
    # -------------------------------------------------------------------------
    @Slot(dict)
    def start_ssh_mission(self, params: dict | None = None) -> None:
        """Launch autonomous collection mission on Orange Pi 5 Pro over SSH."""
        if self._ssh_worker is not None and not self._ssh_worker.is_stop_requested:
            self.mission_control.append_log("⚠️ Миссия уже выполняется на дроне!")
            return

        cfg_params = params or self.mission_control.get_ssh_launch_params()
        host = cfg_params.get("host", "10.42.0.1")
        port = int(cfg_params.get("port", 22))
        username = cfg_params.get("username", "orangepi")
        password = cfg_params.get("password", "orangepi")
        command = cfg_params.get("command", "")

        local_runs_dir = self._settings.workdir / "runs"
        local_runs_dir.mkdir(parents=True, exist_ok=True)

        # Ensure video stream worker is active on the drone host
        if self._stream_worker is None:
            stream_port = getattr(self._settings.stream, "port", 8080) if hasattr(self._settings, "stream") else 8080
            self.start_stream(host=host, port=stream_port)

        self._ssh_worker = SshMissionWorker(
            host=host,
            port=port,
            username=username,
            password=password,
            command=command,
            local_runs_dir=local_runs_dir,
        )

        self._ssh_worker.log_received.connect(self.mission_control.append_log)
        self._ssh_worker.progress_updated.connect(self.mission_control.set_mission_progress)
        self._ssh_worker.mission_started.connect(self._on_ssh_mission_started)
        self._ssh_worker.mission_finished.connect(self._on_ssh_mission_finished)
        self._ssh_worker.summary_downloaded.connect(self._on_ssh_summary_downloaded)
        self._ssh_worker.failed.connect(self._on_ssh_mission_failed)

        self.mission_control.set_flight_locked(True)
        self._flight_start_time = time.time()
        self._window.status_bar_panel.set_message("Запуск автономной миссии на дроне Skyris (SSH)...")

        self._ssh_thread = start_in_thread(self._ssh_worker, "ssh_mission_worker_thread")
        self._window.register_thread(self._ssh_thread, self._ssh_worker)

    @Slot()
    def emergency_land(self) -> None:
        """Send SIGINT (Ctrl+C) to running remote mission and issue REST land fallback."""
        logger.warning("Emergency land triggered from MissionControl")
        self._window.status_bar_panel.set_message("Экстренная посадка: отправка SIGINT (Ctrl+C) на дрон...")

        if self._ssh_worker is not None:
            self._ssh_worker.abort_mission()

        if self._command_worker is not None:
            self._command_worker.stop_mission(StopAction.LAND)

    @Slot()
    def stop_ssh_mission(self) -> None:
        """Stop SSH mission worker if running."""
        if self._ssh_worker is not None:
            self._ssh_worker.abort_mission()
            self._ssh_worker = None
        self._ssh_thread = None

    def _on_ssh_mission_started(self) -> None:
        self._window.status_bar_panel.set_message("Миссия Skyris выполняется на дроне (SSH)...")

    def _on_ssh_mission_finished(self, success: bool, message: str) -> None:
        self.mission_control.set_flight_locked(False)
        self._flight_start_time = None
        self._ssh_worker = None
        self._ssh_thread = None
        self._window.status_bar_panel.set_message(message)

    def _on_ssh_mission_failed(self, error: str) -> None:
        self.mission_control.set_flight_locked(False)
        self._flight_start_time = None
        self._ssh_worker = None
        self._ssh_thread = None
        self._window.status_bar_panel.set_message(f"Ошибка миссии: {error}")

    def _on_ssh_summary_downloaded(self, summary_path_str: str) -> None:
        """Parse downloaded summary.json from drone and plot detections on map."""
        summary_path = Path(summary_path_str)
        if not summary_path.exists():
            return

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary = SummaryDetections.model_validate(data)
            self._map_ctrl.on_summary(summary)
            self._window.status_bar_panel.set_message(
                f"Данные детекций миссии {summary.run_id} успешно нанесены на карту"
            )
            self.mission_control.append_log(
                f"\n🗺️ Детекции сорняков ({summary.detections_count} шт.) нанесены на карту станции."
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Failed to parse downloaded mission summary %s: %s", summary_path, exc)
            self.mission_control.append_log(f"⚠️ Ошибка парсинга артефакта summary.json: {exc}")

    # -------------------------------------------------------------------------
    # Mission Planning & Dispatch (REST API)
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
            logger.exception("Failed to plan mission")
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
        # Also abort active SSH mission if landing
        if action == StopAction.LAND and self._ssh_worker is not None:
            self._ssh_worker.abort_mission()

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
