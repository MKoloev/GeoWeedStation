"""End-to-End tests for complete online drone interaction with local mock server."""

from __future__ import annotations

import time
import pytest
from PySide6.QtCore import QSettings

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.app_controller import AppController
from geo_weed_station.core.mission_planner import PlannerParams
from geo_weed_station.schemas.entities import CoordinateSystem, SensorOptics
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy, Waypoint
from geo_weed_station.schemas.telemetry import ConnectionState, FlightState
from tests.mocks.mock_onboard_server import MockConfig, run_in_thread


@pytest.fixture
def mock_server():
    cfg = MockConfig(
        tick_s=0.1,
        fail_mode="none",
    )
    handle = run_in_thread(cfg)
    time.sleep(0.1)
    yield handle
    handle.stop()


@pytest.mark.e2e
def test_e2e_online_full_mission_cycle(mock_server, tmp_path, qtbot):
    settings = AppSettings(
        workdir=tmp_path / "workdir",
    )
    settings.network.request_timeout_s = 1.0
    settings.network.poll_rate_hz = 2.0

    app_ctrl = AppController(settings=settings)
    qtbot.addWidget(app_ctrl.window)

    online_ctrl = app_ctrl.online_controller

    # 1. Connect
    online_ctrl.connect_to(mock_server.host, mock_server.port)
    qtbot.waitUntil(
        lambda: online_ctrl._last_status is not None and online_ctrl._last_status.flight_state == FlightState.IDLE,
        timeout=3000,
    )
    assert online_ctrl._last_status.flight_state == FlightState.IDLE

    # 2. Plan mission
    optics = SensorOptics(focal_length_mm=4.5, sensor_width_mm=6.4, sensor_height_mm=4.8)
    params = PlannerParams(
        corner_tl=(0.0, 0.0),
        corner_br=(6.0, 6.0),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=10.0,
        optics=optics,
        overlap=0.1,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
    )
    online_ctrl.plan(params)
    assert len(online_ctrl.mission_control._planned_waypoints) > 0

    # 3. Start Mission
    task = MissionTask(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        waypoints=online_ctrl.mission_control._planned_waypoints,
    )
    online_ctrl.send_mission(task)

    # 4. Wait for completion and summary loading
    qtbot.waitUntil(
        lambda: online_ctrl._last_status is not None
        and online_ctrl._last_status.flight_state in (FlightState.COMPLETED, FlightState.IDLE)
        and len(app_ctrl.map_engine.weed_map.points) > 0,
        timeout=8000,
    )

    # Verify points on map
    assert len(app_ctrl.map_engine.weed_map.points) > 0
    assert not app_ctrl.map_engine.is_locked()

    app_ctrl.shutdown()
