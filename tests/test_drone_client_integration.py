"""Integration tests for DroneClient and DownloadManager against mock onboard server."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from geo_weed_station.core.xmp_parser import extract_spatial_data
from geo_weed_station.network.backoff import ExponentialBackoff
from geo_weed_station.network.download_manager import DownloadManager
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.network.exceptions import (
    DroneApiError,
    DroneConflictError,
    DroneConnectionError,
    DroneTimeoutError,
)
from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.onboard_api import StartRequest, StopAction
from geo_weed_station.schemas.telemetry import FlightState
from tests.mocks.mock_onboard_server import MockConfig, run_in_thread


def test_full_mission_lifecycle(mock_server, tmp_path: Path):
    """Scenario: status -> start -> polling to COMPLETED -> summary -> archive -> delete."""
    client = DroneClient(base_url=mock_server.url, timeout_s=2.0)
    dest_dir = tmp_path / "downloads"
    dl_manager = DownloadManager(client=client, target_dir=dest_dir)

    # 1. Initial status
    status = client.get_status()
    assert status.flight_state == FlightState.IDLE
    assert status.is_data_ready is False

    # 2. Start mission with 2 waypoints
    req = StartRequest(
        coordinate_system="aruco_local",
        use_nn=True,
        save_policy="all",
        waypoints=[
            Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0),
            Waypoint(x=1.0, y=0.0, z=2.0, yaw=0.0),
        ],
    )
    start_resp = client.start_mission(req)
    run_id = start_resp.run_id
    assert start_resp.total_waypoints == 2

    # 3. Poll status until completed (ticker runs at 0.1s tick)
    deadline = time.time() + 10.0
    completed = False
    st = client.get_status()
    while time.time() < deadline:
        st = client.get_status()
        if st.flight_state == FlightState.COMPLETED and st.is_data_ready:
            completed = True
            break
        time.sleep(0.05)

    assert completed, f"Mission did not complete in time, state={st.flight_state}"

    # 4. Fetch summary
    summary = client.get_summary(run_id)
    assert summary.run_id == run_id
    assert len(summary.items) == 2

    # 5. Download archive using DownloadManager
    progress_updates = []
    zip_path = dl_manager.download_run(
        run_id=run_id,
        progress_cb=lambda curr, tot: progress_updates.append((curr, tot)),
    )
    assert zip_path.exists()
    assert len(progress_updates) > 0

    # Extract and check files
    extracted_dir = dl_manager.extract_run(zip_path)
    wp0_img = extracted_dir / "wp_0000.jpg"
    assert wp0_img.exists()
    meta = extract_spatial_data(wp0_img)
    assert meta.pose is not None
    assert meta.pose.altitude_m == pytest.approx(2.0)

    # 6. Delete run
    del_resp = client.delete_run(run_id)
    assert del_resp.deleted_run_id == run_id

    # 7. Confirm run is gone
    runs = client.list_runs()
    assert all(r.run_id != run_id for r in runs)


def test_mission_stop_land(mock_server):
    """Scenario: start mission and immediately stop with LAND."""
    client = DroneClient(base_url=mock_server.url, timeout_s=2.0)

    req = StartRequest(
        coordinate_system="aruco_local",
        use_nn=False,
        save_policy="all",
        waypoints=[
            Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0),
            Waypoint(x=10.0, y=10.0, z=2.0, yaw=0.0),
            Waypoint(x=20.0, y=20.0, z=2.0, yaw=0.0),
        ],
    )
    client.start_mission(req)

    # Issue STOP command
    stop_resp = client.stop_mission(action=StopAction.LAND)
    assert stop_resp.status == "ok"
    assert stop_resp.action_initiated == "land"

    # Poll until aborted
    deadline = time.time() + 5.0
    aborted = False
    while time.time() < deadline:
        st = client.get_status()
        if st.flight_state == FlightState.ABORTED and st.is_data_ready:
            aborted = True
            break
        time.sleep(0.05)

    assert aborted


def test_mission_stop_rth(mock_server):
    """Scenario: start mission and stop with RTH."""
    client = DroneClient(base_url=mock_server.url, timeout_s=2.0)

    req = StartRequest(
        coordinate_system="aruco_local",
        use_nn=False,
        save_policy="all",
        waypoints=[
            Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0),
            Waypoint(x=5.0, y=5.0, z=2.0, yaw=0.0),
        ],
    )
    client.start_mission(req)
    stop_resp = client.stop_mission(action=StopAction.RTH)
    assert stop_resp.status == "ok"
    assert stop_resp.action_initiated == "rth"


def test_start_conflict_409(mock_server):
    """Scenario: starting when flight already active raises DroneConflictError."""
    client = DroneClient(base_url=mock_server.url, timeout_s=2.0)

    req = StartRequest(
        coordinate_system="aruco_local",
        use_nn=True,
        save_policy="all",
        waypoints=[
            Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0),
            Waypoint(x=1.0, y=0.0, z=2.0, yaw=0.0),
            Waypoint(x=2.0, y=0.0, z=2.0, yaw=0.0),
            Waypoint(x=3.0, y=0.0, z=2.0, yaw=0.0),
        ],
    )
    client.start_mission(req)

    with pytest.raises(DroneConflictError):
        client.start_mission(req)


def test_reconnect_and_backoff_recovery():
    """Scenario: mock server experiences failure, backoff reconnects when restored."""
    server_handle = run_in_thread(MockConfig(tick_s=0.1, fail_mode="drop", initial_runs=0))
    try:
        client = DroneClient(base_url=server_handle.url, timeout_s=0.5)

        # In drop mode, requests fail
        with pytest.raises((DroneConnectionError, DroneTimeoutError, DroneApiError)):
            client.get_status()

        # Disable fail mode on server
        server_handle.state.config.fail_mode = "none"

        # Now reconnect with backoff
        backoff = ExponentialBackoff(initial_s=0.05, max_s=0.2, factor=2.0)
        recovered = False
        retries = 0
        while retries < 5:
            try:
                st = client.get_status()
                recovered = True
                assert st.flight_state == FlightState.IDLE
                break
            except Exception:
                delay = backoff.next_delay()
                time.sleep(delay)
                retries += 1

        assert recovered
    finally:
        server_handle.stop()
