"""Unit and integration tests for TelemetryWorker using pytest-qt and mock server."""

from __future__ import annotations

import time
import pytest
from PySide6.QtCore import QThread

from geo_weed_station.config.settings import NetworkSettings
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.schemas.telemetry import ConnectionState, DroneStatus, FlightState
from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.telemetry_worker import TelemetryWorker
from tests.mocks.mock_onboard_server import MockConfig, run_in_thread


def test_telemetry_worker_polling_and_stop(mock_server, qtbot):
    """Test TelemetryWorker polling cycle, signals emission, and quick thread termination."""
    client = DroneClient(base_url=mock_server.url, timeout_s=1.0)
    settings = NetworkSettings(
        poll_rate_hz=2.0,
        reconnect_backoff_initial_s=0.1,
        reconnect_backoff_max_s=0.5,
    )
    worker = TelemetryWorker(client=client, settings=settings)

    received_statuses = []
    connection_states = []
    latencies = []

    worker.status_received.connect(received_statuses.append)
    worker.connection_changed.connect(connection_states.append)
    worker.latency_measured.connect(latencies.append)

    thread = start_in_thread(worker, "test_telemetry_thread")

    # Wait until status is received
    qtbot.waitUntil(lambda: len(received_statuses) >= 2, timeout=3000)

    assert ConnectionState.CONNECTED in connection_states
    assert isinstance(received_statuses[0], DroneStatus)
    assert len(latencies) >= 2

    # Request stop and verify thread terminates <= 3.0 s
    t0 = time.time()
    worker.request_stop()
    thread.wait(3000)
    duration = time.time() - t0

    assert thread.isFinished()
    assert duration <= 3.0


def test_telemetry_worker_reconnection(qtbot):
    """Test connection state changes to RECONNECTING on drop and returns to CONNECTED when restored."""
    server_handle = run_in_thread(MockConfig(tick_s=0.1, fail_mode="none", initial_runs=0))
    try:
        client = DroneClient(base_url=server_handle.url, timeout_s=0.5)
        settings = NetworkSettings(
            poll_rate_hz=2.0,
            reconnect_backoff_initial_s=0.05,
            reconnect_backoff_max_s=0.2,
        )
        worker = TelemetryWorker(client=client, settings=settings)

        states = []
        worker.connection_changed.connect(states.append)

        thread = start_in_thread(worker, "test_reconnect_thread")

        # Initial connection
        qtbot.waitUntil(lambda: ConnectionState.CONNECTED in states, timeout=3000)

        # Trigger failure on server
        server_handle.state.config.fail_mode = "drop"
        qtbot.waitUntil(lambda: ConnectionState.RECONNECTING in states, timeout=3000)

        # Restore server
        server_handle.state.config.fail_mode = "none"
        # Wait until it reconnects
        qtbot.waitUntil(lambda: states.count(ConnectionState.CONNECTED) >= 2, timeout=4000)

        worker.request_stop()
        thread.wait(3000)
        assert thread.isFinished()
    finally:
        server_handle.stop()
