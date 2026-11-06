"""Synchronous REST API client for the agrovision_onboard drone system."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional
import requests

from geo_weed_station.network.exceptions import (
    DroneApiError,
    DroneConflictError,
    DroneConnectionError,
    DroneNotFoundError,
    DroneTimeoutError,
)
from geo_weed_station.schemas.mission import MissionTask
from geo_weed_station.schemas.onboard_api import (
    DeleteRunResponse,
    DroneRun,
    StartRequest,
    StartResponse,
    StopAction,
    StopRequest,
    StopResponse,
    SummaryDetections,
    mission_to_start_request,
)
from geo_weed_station.schemas.telemetry import DroneStatus


class DroneClient:
    """REST API HTTP client communicating with agrovision_onboard drone server.
    
    This class is synchronous and intentionally UI-agnostic; it must be invoked
    from background worker threads (QThread) to ensure non-blocking UI operation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout_s: float = 2.0,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize client. timeout_s must be <= 2.0s per Non-Functional Requirements."""
        if base_url:
            self._base_url = base_url.rstrip("/")
            # extract host and port if possible
            parts = self._base_url.replace("http://", "").replace("https://", "").split(":")
            self.host = parts[0]
            self.port = int(parts[1]) if len(parts) > 1 else 80
        else:
            self.host = host
            self.port = port
            self._base_url = f"http://{self.host}:{self.port}"
        self.timeout_s = min(2.0, max(0.1, float(timeout_s)))
        self._session = requests.Session()

    def configure(self, host: str, port: int) -> None:
        """Update host and port connection settings."""
        self.host = host
        self.port = port
        self._base_url = f"http://{self.host}:{self.port}"

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Perform HTTP request with error handling and status code mapping."""
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                timeout=self.timeout_s,
                stream=stream,
            )
        except requests.exceptions.Timeout as e:
            raise DroneTimeoutError(f"Request {method} {path} timed out after {self.timeout_s}s: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise DroneConnectionError(f"Failed to connect to drone at {url}: {e}") from e
        except requests.exceptions.RequestException as e:
            raise DroneConnectionError(f"Network error on {method} {url}: {e}") from e

        if resp.status_code >= 400:
            detail = resp.text
            try:
                err_json = resp.json()
                if isinstance(err_json, dict) and "detail" in err_json:
                    detail = str(err_json["detail"])
            except Exception:
                pass

            if resp.status_code == 404:
                raise DroneNotFoundError(404, detail)
            elif resp.status_code == 409:
                raise DroneConflictError(409, detail)
            else:
                raise DroneApiError(resp.status_code, detail)

        return resp

    def get_status(self) -> DroneStatus:
        """Fetch current drone telemetry status (GET /status)."""
        resp = self._request("GET", "/status")
        return DroneStatus.model_validate(resp.json())

    def start_mission(self, task: MissionTask | StartRequest) -> StartResponse:
        """Send flight task to drone and start autonomous mission (POST /start)."""
        if isinstance(task, StartRequest):
            req_dto = task
        else:
            req_dto = mission_to_start_request(task)
        resp = self._request("POST", "/start", json_data=req_dto.model_dump())
        return StartResponse.model_validate(resp.json())

    def stop_mission(self, action: StopAction = StopAction.LAND) -> StopResponse:
        """Execute emergency or regular mission abort (POST /stop)."""
        req_dto = StopRequest(action=action)
        resp = self._request("POST", "/stop", json_data=req_dto.model_dump())
        return StopResponse.model_validate(resp.json())

    def list_runs(self) -> List[DroneRun]:
        """List all completed and active mission runs stored on drone (GET /runs)."""
        resp = self._request("GET", "/runs")
        raw_items = resp.json()
        if not isinstance(raw_items, list):
            return []
        return [DroneRun.model_validate(r) for r in raw_items]

    def get_summary(self, run_id: str) -> SummaryDetections:
        """Fetch aggregated detections JSON summary for run (GET /runs/{id}/data?type=summary)."""
        resp = self._request("GET", f"/runs/{run_id}/data", params={"type": "summary"})
        return SummaryDetections.model_validate(resp.json())

    def get_image(self, run_id: str, wp_index: int) -> bytes:
        """Fetch individual capture JPEG image for waypoint (GET /runs/{id}/data?type=image&wp_index=N)."""
        resp = self._request(
            "GET",
            f"/runs/{run_id}/data",
            params={"type": "image", "wp_index": wp_index},
        )
        return resp.content

    def delete_run(self, run_id: str) -> DeleteRunResponse:
        """Delete mission run directory from onboard storage (DELETE /runs/{id})."""
        resp = self._request("DELETE", f"/runs/{run_id}")
        return DeleteRunResponse.model_validate(resp.json())

    def stream_archive(self, run_id: str, chunk_bytes: int = 65536) -> Iterator[bytes]:
        """Stream mission ZIP archive in binary chunks without loading entire file to RAM."""
        resp = self._request(
            "GET",
            f"/runs/{run_id}/data",
            params={"type": "archive"},
            stream=True,
        )
        return resp.iter_content(chunk_size=chunk_bytes)

    def close(self) -> None:
        """Close underlying HTTP session adapter."""
        self._session.close()
