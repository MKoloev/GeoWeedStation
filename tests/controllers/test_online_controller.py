"""Tests for OnlineController drone communication and mission execution (offscreen)."""

from __future__ import annotations

import time

import pytest
from PySide6.QtCore import QSettings

from geo_weed_station.config.settings import AppSettings, MapSettings, NetworkSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.controllers.online_controller import OnlineController
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.core.mission_planner import PlannerParams
from geo_weed_station.schemas.entities import CoordinateSystem, SensorOptics
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy, Waypoint
from geo_weed_station.schemas.telemetry import (
    DronePose,
    DroneStatus,
    FlightState,
)
from geo_weed_station.ui.main_window import MainWindow
from geo_weed_station.ui.widgets.map_canvas import MapCanvas


@pytest.fixture
def online_setup(tmp_path):
    settings = AppSettings(
        network=NetworkSettings(
            host="127.0.0.1",
            port=8000,
            request_timeout_s=1.0,
            poll_rate_hz=2.0,
        ),
        map=MapSettings(heatmap_cell_m=1.0),
        workdir=tmp_path / "workdir",
    )
    engine = MapEngine(projector=NadirProjector(), settings=settings.map)
    map_ctrl = MapController(engine=engine)
    qset = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    window = MainWindow(qsettings=qset)
    canvas = MapCanvas()

    ctrl = OnlineController(
        window=window,
        map_ctrl=map_ctrl,
        settings=settings,
        map_canvas=canvas,
    )
    return ctrl, map_ctrl, window, canvas, settings


def test_online_controller_plan_mission(online_setup, qtbot):
    ctrl, map_ctrl, window, canvas, _ = online_setup

    optics = SensorOptics(focal_length_mm=4.5, sensor_width_mm=6.4, sensor_height_mm=4.8)
    params = PlannerParams(
        corner_tl=(0.0, 0.0),
        corner_br=(20.0, 20.0),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=10.0,
        optics=optics,
        overlap=0.2,
        mode=MissionMode.COLLECTION_ONLY,
        save_policy=SavePolicy.ALL,
    )

    ctrl.plan(params)
    assert len(ctrl.mission_control._planned_waypoints) > 0
    assert len(canvas._route_group.childItems()) > 0


def test_online_controller_blocks_start_when_not_idle(online_setup, qtbot):
    ctrl, _map_ctrl, _window, _canvas, _ = online_setup
    _window.show()
    tabs = _window.central_tabs()
    tabs.setCurrentIndex(3)
    ctrl.mission_control.show()

    # Simulate drone reporting FLYING_TO_WP
    busy_status = DroneStatus(
        preset="test",
        flight_state=FlightState.FLYING_TO_WP,
        current_wp_index=2,
        total_waypoints=10,
        captures_count=5,
        is_data_ready=False,
    )
    ctrl.on_status(busy_status)

    task = MissionTask(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        mode=MissionMode.COLLECTION_ONLY,
        save_policy=SavePolicy.ALL,
        waypoints=[Waypoint(x=0.0, y=0.0, z=10.0)],
    )

    ctrl.send_mission(task)
    assert not ctrl.mission_control.lbl_errors.isHidden()
    assert "требуется IDLE" in ctrl.mission_control.lbl_errors.text()


def test_online_controller_status_updates_marker_and_telemetry(online_setup, qtbot):
    ctrl, _map_ctrl, _window, canvas, _ = online_setup

    status = DroneStatus(
        preset="test",
        flight_state=FlightState.IDLE,
        current_wp_index=0,
        total_waypoints=5,
        current_pose=DronePose(
            x=10.0,
            y=12.0,
            z=15.0,
            roll=0.0,
            pitch=0.0,
            yaw=45.0,
            coordinate_system="aruco_local",
            timestamp=time.time(),
        ),
        captures_count=0,
        is_data_ready=False,
    )

    ctrl.on_status(status)
    assert len(canvas._drone_group.childItems()) == 2
    assert "10.00" in ctrl.telemetry_panel.lbl_pos_xy.text()


def test_online_controller_start_ssh_mission_and_emergency_land(online_setup, qtbot):
    """Test start_ssh_mission launches worker and emergency_land aborts it."""
    from unittest.mock import MagicMock, patch

    ctrl, _map_ctrl, _window, _canvas, _ = online_setup

    mock_client = MagicMock()
    mock_transport = MagicMock()
    mock_channel = MagicMock()
    mock_channel.recv_ready.return_value = False
    mock_channel.exit_status_ready.return_value = False
    mock_transport.open_session.return_value = mock_channel
    mock_client.get_transport.return_value = mock_transport

    with patch("geo_weed_station.workers.ssh_mission_worker.paramiko.SSHClient", return_value=mock_client), \
         patch("geo_weed_station.controllers.online_controller.start_in_thread") as mock_start:
        mock_thread = MagicMock()
        mock_start.return_value = mock_thread

        ctrl.start_ssh_mission({
            "host": "10.42.0.1",
            "port": 22,
            "username": "orangepi",
            "password": "orangepi",
            "command": "python3 main.py",
        })

        assert ctrl._ssh_worker is not None
        assert ctrl._ssh_worker.host == "10.42.0.1"
        assert ctrl.mission_control._is_flight_locked is True

        # Test emergency land triggers abort
        with patch.object(ctrl._ssh_worker, "abort_mission") as mock_abort:
            ctrl.emergency_land()
            mock_abort.assert_called_once()


def test_online_controller_ssh_summary_loaded_to_map(online_setup, tmp_path):
    """Test downloading and plotting summary.json onto map engine."""
    import json
    from geo_weed_station.schemas.onboard_api import SummaryDetections

    ctrl, map_ctrl, _window, canvas, _ = online_setup

    summary_file = tmp_path / "summary.json"
    summary_data = {
        "run_id": "run_test_001",
        "coordinate_system": "aruco_local",
        "detections_count": 1,
        "items": [
            {
                "wp_index": 0,
                "timestamp": 1234567.89,
                "camera_x": 5.0,
                "camera_y": 6.0,
                "altitude_m": 2.0,
                "detections": [
                    {
                        "class_name": "weed",
                        "confidence": 0.95,
                        "bbox_px": [100, 100, 200, 200],
                        "target_local_coords": {"x": 5.2, "y": 6.3},
                    }
                ],
            }
        ],
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f)

    ctrl._on_ssh_summary_downloaded(str(summary_file))
    assert len(map_ctrl.engine.weed_map.points) > 0


def test_online_controller_ssh_summary_zero_altitude_locks_map_and_warns(online_setup, tmp_path):
    """When SSH summary.json has altitude_m=0, map locks and warning is logged."""
    import json
    from unittest.mock import patch

    ctrl, map_ctrl, _window, _canvas, _ = online_setup

    summary_file = tmp_path / "summary.json"
    summary_data = {
        "run_id": "run_zero_alt",
        "coordinate_system": "aruco_local",
        "detections_count": 1,
        "items": [
            {
                "wp_index": 0,
                "timestamp": 1234567.89,
                "camera_x": 0.0,
                "camera_y": 0.0,
                "altitude_m": 0.0,
                "detections": [
                    {
                        "class_name": "weed",
                        "confidence": 0.95,
                        "bbox_px": [100, 100, 200, 200],
                    }
                ],
            }
        ],
    }
    summary_file.write_text(json.dumps(summary_data), encoding="utf-8")

    with patch.object(ctrl.mission_control, "append_log", wraps=ctrl.mission_control.append_log) as mock_log:
        ctrl._on_ssh_summary_downloaded(str(summary_file))

    assert map_ctrl.is_locked is True
    assert len(map_ctrl.engine.weed_map.points) == 0
    logged = "\n".join(str(call.args[0]) for call in mock_log.call_args_list)
    assert "заблокирована" in logged.lower()
    assert "--fallback-telemetry-altitude" in logged

