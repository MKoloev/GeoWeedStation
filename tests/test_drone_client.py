"""Unit tests for DroneClient REST API operations and error handling."""

from __future__ import annotations

import io
import json
from pathlib import Path
import pytest
import requests
from requests.adapters import BaseAdapter
from requests.models import Response

from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.network.exceptions import (
    DroneApiError,
    DroneConflictError,
    DroneConnectionError,
    DroneNotFoundError,
    DroneTimeoutError,
)
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy, Waypoint
from geo_weed_station.schemas.onboard_api import StopAction
from geo_weed_station.schemas.telemetry import FlightState

SAMPLES_DIR = Path(__file__).parent / "data" / "api_samples"


class DictMockAdapter(BaseAdapter):
    """Custom requests transport adapter returning configured responses or raising errors."""

    def __init__(self) -> None:
        super().__init__()
        self.routes = {}

    def add_route(
        self,
        method: str,
        path: str,
        status_code: int = 200,
        data: bytes | str | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.routes[(method.upper(), path)] = (status_code, data, exception)

    def send(self, request, **kwargs):
        # Extract relative path from URL
        from urllib.parse import urlparse
        parsed = urlparse(request.url)
        path = parsed.path
        if parsed.query:
            path += f"?{parsed.query}"

        # Match path with or without query
        match_key = (request.method.upper(), path)
        if match_key not in self.routes:
            match_key = (request.method.upper(), parsed.path)

        if match_key not in self.routes:
            raise requests.exceptions.ConnectionError(f"No mock route for {request.method} {path}")

        status_code, body_data, exc = self.routes[match_key]
        if exc is not None:
            raise exc

        resp = Response()
        resp.status_code = status_code
        resp.url = request.url
        resp.request = request
        if body_data is not None:
            if isinstance(body_data, str):
                resp.raw = io.BytesIO(body_data.encode("utf-8"))
            else:
                resp.raw = io.BytesIO(body_data)
        else:
            resp.raw = io.BytesIO(b"")
        return resp

    def close(self):
        pass


@pytest.fixture
def mock_client() -> tuple[DroneClient, DictMockAdapter]:
    client = DroneClient(host="192.168.4.1", port=8000, timeout_s=2.0)
    adapter = DictMockAdapter()
    client._session.mount("http://", adapter)
    return client, adapter


def test_drone_client_timeout_capped():
    """Verify DroneClient caps timeout at 2.0s per Non-Functional Requirements."""
    c = DroneClient(host="10.0.0.1", port=8000, timeout_s=5.0)
    assert c.timeout_s == 2.0


def test_get_status_success(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify GET /status parses into valid DroneStatus model."""
    client, adapter = mock_client
    status_sample = (SAMPLES_DIR / "status_response.json").read_text(encoding="utf-8")
    adapter.add_route("GET", "/status", status_code=200, data=status_sample)

    status = client.get_status()
    assert status.flight_state == FlightState.FLYING_TO_WP
    assert status.current_wp_index == 2
    assert status.total_waypoints == 3


def test_start_mission_success(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify POST /start sends serialized task and returns StartResponse."""
    client, adapter = mock_client
    start_sample = (SAMPLES_DIR / "start_response.json").read_text(encoding="utf-8")
    adapter.add_route("POST", "/start", status_code=202, data=start_sample)

    task = MissionTask(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        waypoints=[Waypoint(x=1.0, y=1.0, z=1.5)],
    )

    resp = client.start_mission(task)
    assert resp.status == "started"
    assert resp.run_id == "run_20261105_153022"


def test_stop_mission_success(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify POST /stop initiates landing and returns StopResponse."""
    client, adapter = mock_client
    stop_sample = (SAMPLES_DIR / "stop_response.json").read_text(encoding="utf-8")
    adapter.add_route("POST", "/stop", status_code=200, data=stop_sample)

    resp = client.stop_mission(action=StopAction.LAND)
    assert resp.action_initiated == "land"
    assert "aborted by operator" in resp.message.lower()


def test_list_runs_success(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify GET /runs returns list of DroneRun objects."""
    client, adapter = mock_client
    runs_sample = (SAMPLES_DIR / "runs_response.json").read_text(encoding="utf-8")
    adapter.add_route("GET", "/runs", status_code=200, data=runs_sample)

    runs = client.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == "run_20261105_153022"
    assert runs[0].status == "completed"


def test_get_summary_success(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify GET /runs/{id}/data?type=summary parses SummaryDetections."""
    client, adapter = mock_client
    summary_sample = (SAMPLES_DIR / "summary_response.json").read_text(encoding="utf-8")
    adapter.add_route("GET", "/runs/run_1/data?type=summary", status_code=200, data=summary_sample)

    summary = client.get_summary("run_1")
    assert summary.run_id == "run_20261105_153022"
    assert summary.detections_count == 4


def test_get_image_success(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify GET /runs/{id}/data?type=image&wp_index=1 returns image bytes."""
    client, adapter = mock_client
    dummy_bytes = b"\xff\xd8\xff\xe0dummy_jpeg_bytes\xff\xd9"
    adapter.add_route("GET", "/runs/run_1/data?type=image&wp_index=1", status_code=200, data=dummy_bytes)

    data = client.get_image("run_1", wp_index=1)
    assert data == dummy_bytes


def test_delete_run_success(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify DELETE /runs/{id} returns DeleteRunResponse."""
    client, adapter = mock_client
    del_sample = (SAMPLES_DIR / "delete_run_response.json").read_text(encoding="utf-8")
    adapter.add_route("DELETE", "/runs/run_1", status_code=200, data=del_sample)

    resp = client.delete_run("run_1")
    assert resp.status == "success"
    assert resp.deleted_run_id == "run_20261105_153022"


def test_error_mapping_404_and_409(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify HTTP 404 and 409 map to DroneNotFoundError and DroneConflictError."""
    client, adapter = mock_client
    adapter.add_route("GET", "/status", status_code=404, data=json.dumps({"detail": "Not found"}))
    with pytest.raises(DroneNotFoundError):
        client.get_status()

    adapter.add_route("POST", "/start", status_code=409, data=json.dumps({"detail": "Mission active"}))
    task = MissionTask(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        waypoints=[Waypoint(x=0.0, y=0.0, z=1.0)],
    )
    with pytest.raises(DroneConflictError):
        client.start_mission(task)


def test_timeout_and_connection_error_mapping(mock_client: tuple[DroneClient, DictMockAdapter]):
    """Verify requests Timeout and ConnectionError map to DroneTimeoutError and DroneConnectionError."""
    client, adapter = mock_client
    adapter.add_route("GET", "/status", exception=requests.exceptions.Timeout("timed out"))
    with pytest.raises(DroneTimeoutError):
        client.get_status()

    adapter.add_route("GET", "/runs", exception=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(DroneConnectionError):
        client.list_runs()
