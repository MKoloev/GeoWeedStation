"""Resilience tests: network disconnection, reconnect backoff and server failure recovery."""

from __future__ import annotations

import time

import pytest

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.app_controller import AppController
from geo_weed_station.schemas.telemetry import ConnectionState
from tests.mocks.mock_onboard_server import MockConfig, run_in_thread


@pytest.mark.e2e
def test_network_reconnect_resilience(tmp_path, qtbot):
    # 1. Start mock server with fail_mode="timeout"
    cfg = MockConfig(
        tick_s=0.1,
        fail_mode="timeout",
    )
    handle = run_in_thread(cfg)
    time.sleep(0.1)

    try:
        settings = AppSettings(workdir=tmp_path / "workdir")
        settings.network.request_timeout_s = 0.5
        settings.network.poll_rate_hz = 2.0
        settings.network.reconnect_backoff_initial_s = 0.1
        settings.network.reconnect_backoff_max_s = 0.3

        app_ctrl = AppController(settings=settings)
        qtbot.addWidget(app_ctrl.window)

        conn_states = []
        app_ctrl.online_controller.connection_state_changed.connect(conn_states.append)

        # Connect to failing server
        app_ctrl.online_controller.connect_to(handle.host, handle.port)

        qtbot.waitUntil(
            lambda: ConnectionState.RECONNECTING in conn_states or ConnectionState.TIMEOUT in conn_states,
            timeout=3000,
        )

        # 2. Recover server to healthy
        handle.config.fail_mode = "none"

        # Telemetry worker should auto-reconnect
        qtbot.waitUntil(
            lambda: app_ctrl.online_controller._last_status is not None,
            timeout=4000,
        )
        assert app_ctrl.online_controller._last_status is not None

        app_ctrl.shutdown()
    finally:
        handle.stop()
