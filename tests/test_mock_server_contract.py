"""Contract tests for Agrovision Onboard mock server endpoints."""

from __future__ import annotations

import io
import time
import zipfile
import pytest
import requests

from geo_weed_station.core.xmp_parser import extract_spatial_data
from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.onboard_api import (
    DeleteRunResponse,
    DroneRun,
    StartRequest,
    StartResponse,
    StopAction,
    StopRequest,
    StopResponse,
    SummaryDetections,
)
from geo_weed_station.schemas.telemetry import DroneStatus, FlightState
from tests.mocks.mock_onboard_server import MockConfig, run_in_thread


def test_status_endpoint(mock_server):
    resp = requests.get(f"{mock_server.url}/status", timeout=2.0)
    assert resp.status_code == 200
    status = DroneStatus.model_validate(resp.json())
    assert status.preset == "clever4_csi_onnx"
    assert status.flight_state == FlightState.IDLE
    assert status.is_data_ready is False


def test_runs_list_endpoint(mock_server):
    resp = requests.get(f"{mock_server.url}/runs", timeout=2.0)
    assert resp.status_code == 200
    runs = [DroneRun.model_validate(r) for r in resp.json()]
    assert len(runs) >= 1
    assert runs[0].run_id.startswith("run_historical_")
    assert runs[0].status == "completed"


def test_start_and_conflict(mock_server):
    req = StartRequest(
        coordinate_system="aruco_local",
        use_nn=True,
        save_policy="all",
        waypoints=[
            Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0),
            Waypoint(x=1.0, y=1.0, z=2.0, yaw=0.0),
        ],
    )
    resp = requests.post(f"{mock_server.url}/start", json=req.model_dump(), timeout=2.0)
    assert resp.status_code == 202
    start_resp = StartResponse.model_validate(resp.json())
    assert start_resp.status == "accepted"
    assert start_resp.total_waypoints == 2

    # Immediate second start while in flight should produce 409
    resp2 = requests.post(f"{mock_server.url}/start", json=req.model_dump(), timeout=2.0)
    assert resp2.status_code == 409


def test_stop_endpoint(mock_server):
    # Stop while idle
    resp = requests.post(f"{mock_server.url}/stop", json={"action": "land"}, timeout=2.0)
    assert resp.status_code == 200
    stop_resp = StopResponse.model_validate(resp.json())
    assert stop_resp.status == "ok"


def test_run_data_archive_contains_valid_xmp(mock_server, tmp_path):
    runs = requests.get(f"{mock_server.url}/runs", timeout=2.0).json()
    run_id = runs[0]["run_id"]

    resp = requests.get(f"{mock_server.url}/runs/{run_id}/data?type=archive", timeout=2.0)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    # Verify ZIP archive contents
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert "wp_0000.jpg" in names
    assert "summary.json" in names

    # Verify XMP parsing on downloaded image
    img_bytes = zf.read("wp_0000.jpg")
    img_path = tmp_path / "wp_0000.jpg"
    img_path.write_bytes(img_bytes)

    meta = extract_spatial_data(img_path)
    assert meta.pose is not None
    assert meta.pose.altitude_m == pytest.approx(2.0)
    assert meta.pose.pitch_deg == pytest.approx(-90.0)


def test_run_data_summary(mock_server):
    runs = requests.get(f"{mock_server.url}/runs", timeout=2.0).json()
    run_id = runs[0]["run_id"]

    resp = requests.get(f"{mock_server.url}/runs/{run_id}/data?type=summary", timeout=2.0)
    assert resp.status_code == 200
    summary = SummaryDetections.model_validate(resp.json())
    assert summary.run_id == run_id
    assert len(summary.items) > 0


def test_run_data_image_and_404(mock_server):
    runs = requests.get(f"{mock_server.url}/runs", timeout=2.0).json()
    run_id = runs[0]["run_id"]

    resp = requests.get(f"{mock_server.url}/runs/{run_id}/data?type=image&wp_index=0", timeout=2.0)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 100

    # Non-existing waypoint
    resp_404 = requests.get(f"{mock_server.url}/runs/{run_id}/data?type=image&wp_index=999", timeout=2.0)
    assert resp_404.status_code == 404

    # Non-existing run
    resp_bad_run = requests.get(f"{mock_server.url}/runs/non_existent_run/data?type=summary", timeout=2.0)
    assert resp_bad_run.status_code == 404


def test_delete_run(mock_server):
    runs = requests.get(f"{mock_server.url}/runs", timeout=2.0).json()
    run_id = runs[0]["run_id"]

    del_resp = requests.delete(f"{mock_server.url}/runs/{run_id}", timeout=2.0)
    assert del_resp.status_code == 200
    res = DeleteRunResponse.model_validate(del_resp.json())
    assert res.deleted_run_id == run_id

    # Deleting again should return 404
    del_again = requests.delete(f"{mock_server.url}/runs/{run_id}", timeout=2.0)
    assert del_again.status_code == 404


def test_fault_injection():
    # Test http500 mode
    handle_500 = run_in_thread(MockConfig(tick_s=0.1, fail_mode="http500", initial_runs=0))
    try:
        resp = requests.get(f"{handle_500.url}/status", timeout=2.0)
        assert resp.status_code == 500
    finally:
        handle_500.stop()

    # Test drop mode
    handle_drop = run_in_thread(MockConfig(tick_s=0.1, fail_mode="drop", initial_runs=0))
    try:
        resp = requests.get(f"{handle_drop.url}/status", timeout=2.0)
        assert resp.status_code == 502
    finally:
        handle_drop.stop()
