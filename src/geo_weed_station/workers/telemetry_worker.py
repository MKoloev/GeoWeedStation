"""Telemetry worker polling onboard drone status in a dedicated background QThread."""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer, Signal, Slot

from geo_weed_station.config.settings import NetworkSettings
from geo_weed_station.network.backoff import ExponentialBackoff
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.network.exceptions import (
    DroneApiError,
    DroneConnectionError,
    DroneTimeoutError,
)
from geo_weed_station.schemas.telemetry import ConnectionState
from geo_weed_station.workers.base_worker import ManagedWorker


class TelemetryWorker(ManagedWorker):
    """Background worker continuously querying drone telemetry at 1-2 Hz with backoff reconnect."""

    status_received = Signal(object)
    connection_changed = Signal(object)
    latency_measured = Signal(float)

    def __init__(
        self,
        client: DroneClient,
        settings: NetworkSettings,
    ) -> None:
        super().__init__()
        self.client = client
        self.settings = settings
        self._poll_interval_ms = int(1000.0 / max(0.5, min(2.0, float(settings.poll_rate_hz))))
        self._backoff = ExponentialBackoff(
            initial_s=settings.reconnect_backoff_initial_s,
            max_s=settings.reconnect_backoff_max_s,
        )
        self._connection_state = ConnectionState.DISCONNECTED
        self._timer: QTimer | None = None
        self._is_querying = False

    @Slot(str, int)
    def set_endpoint(self, host: str, port: int) -> None:
        """Update drone target host and port."""
        self.client.configure(host, port)

    @Slot(float)
    def set_poll_rate(self, hz: float) -> None:
        """Update polling rate frequency in Hz (0.5 to 2.0)."""
        clamped_hz = max(0.5, min(2.0, float(hz)))
        self._poll_interval_ms = int(1000.0 / clamped_hz)
        if self._timer and self._timer.isActive():
            self._timer.setInterval(self._poll_interval_ms)

    @Slot()
    def start_work(self) -> None:
        """Start non-blocking event-driven polling timer in worker thread."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_tick)
        self._timer.start(0)  # Trigger immediate first poll
        self.started_ok.emit()

    @Slot()
    def request_stop(self) -> None:
        """Stop polling and trigger thread completion."""
        super().request_stop()
        self.finished.emit()

    def _poll_tick(self) -> None:
        """Single polling tick handler executed by QTimer."""
        if self.is_stop_requested:
            if self._timer is not None:
                self._timer.stop()
            return

        if self.is_paused:
            return

        if self._is_querying:
            return

        self._is_querying = True
        try:
            t0 = time.time()
            status = self.client.get_status()
            latency = time.time() - t0

            self.status_received.emit(status)
            self.latency_measured.emit(latency)

            if self._connection_state != ConnectionState.CONNECTED:
                self._connection_state = ConnectionState.CONNECTED
                self.connection_changed.emit(ConnectionState.CONNECTED)
                self._backoff.reset()

            if self._timer and not self.is_stop_requested:
                self._timer.setInterval(self._poll_interval_ms)

        except (DroneConnectionError, DroneTimeoutError, DroneApiError, Exception):
            new_state = (
                ConnectionState.RECONNECTING
                if self._connection_state == ConnectionState.CONNECTED
                else ConnectionState.TIMEOUT
            )
            if self._connection_state != new_state:
                self._connection_state = new_state
                self.connection_changed.emit(new_state)

            backoff_delay_ms = int(self._backoff.next_delay() * 1000.0)
            if self._timer and not self.is_stop_requested:
                self._timer.setInterval(max(10, backoff_delay_ms))
        finally:
            self._is_querying = False
