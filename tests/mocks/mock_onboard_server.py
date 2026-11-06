"""FastAPI mock server emulating agrovision_onboard drone REST API.

Provides full state machine emulation, telemetry, mission execution, image/summary/archive
generation with XMP, and configurable fault injection for testing.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.responses import Response as RawResponse
from pydantic import BaseModel

from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.onboard_api import (
    DeleteRunResponse,
    DroneRun,
    StartRequest,
    StartResponse,
    StopRequest,
    StopResponse,
    SummaryDetections,
)
from geo_weed_station.schemas.telemetry import (
    DronePose,
    DroneStatus,
    FlightState,
)
from tests.mocks.fixtures_generator import (
    make_jpeg_bytes,
    make_run_zip_bytes,
    make_summary_detections,
    make_xmp_xml,
)


class MockConfig(BaseModel):
    """Configuration for mock onboard server."""

    tick_s: float = 0.2
    fail_mode: Literal["none", "timeout", "drop", "http500"] = "none"
    latency_s: float = 0.0
    preset: str = "clever4_csi_onnx"
    initial_runs: int = 1


class MockRunData:
    """In-memory storage for a simulated mission run."""

    def __init__(
        self,
        run_id: str,
        coordinate_system: str,
        waypoints: list[Waypoint],
        start_time_iso: str,
    ) -> None:
        self.run_id = run_id
        self.coordinate_system = coordinate_system
        self.waypoints = waypoints
        self.start_time_iso = start_time_iso
        self.status = "in_progress"
        self.images: dict[int, bytes] = {}
        self.summary: SummaryDetections | None = None
        self.zip_bytes: bytes | None = None

    def finalize(self, aborted: bool = False) -> None:
        self.status = "aborted" if aborted else "completed"
        # Generate final summary & zip
        self.summary = make_summary_detections(
            self.run_id,
            self.waypoints[: len(self.images)],
            coordinate_system=self.coordinate_system,
        )
        self.zip_bytes = make_run_zip_bytes(
            self.run_id,
            self.waypoints[: len(self.images)],
            coordinate_system=self.coordinate_system,
        )

    def to_drone_run(self) -> DroneRun:
        contents = [f"wp_{idx:04d}.jpg" for idx in self.images]
        if self.summary is not None:
            contents.append("summary.json")
        size_bytes = (
            len(self.zip_bytes)
            if self.zip_bytes is not None
            else sum(len(b) for b in self.images.values())
        )
        return DroneRun(
            run_id=self.run_id,
            start_time=self.start_time_iso,
            status=self.status,
            directory_path=f"/data/runs/{self.run_id}",
            size_bytes=size_bytes,
            contents=contents,
        )


class MockDroneState:
    """Thread-safe state machine emulating onboard drone control logic."""

    def __init__(self, config: MockConfig | None = None) -> None:
        self.lock = threading.Lock()
        self.config = config or MockConfig()
        self.flight_state: FlightState = FlightState.IDLE
        self.current_wp_index: int = 0
        self.total_waypoints: int = 0
        self.current_pose: DronePose | None = DronePose(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            coordinate_system="aruco_local",
            timestamp=time.time(),
        )
        self.captures_count: int = 0
        self.is_data_ready: bool = False
        self.active_run_id: str | None = None
        self._abort_requested: bool = False
        self._active_waypoints: list[Waypoint] = []
        self._active_cs: str = "aruco_local"

        self.runs: dict[str, MockRunData] = {}

        if self.config.initial_runs > 0:
            self._create_initial_runs(self.config.initial_runs)

    def _create_initial_runs(self, count: int) -> None:
        for i in range(1, count + 1):
            run_id = f"run_historical_{i:03d}"
            wps = [
                Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0),
                Waypoint(x=1.0, y=0.0, z=2.0, yaw=0.0),
            ]
            run_data = MockRunData(
                run_id=run_id,
                coordinate_system="aruco_local",
                waypoints=wps,
                start_time_iso="2026-11-01T10:00:00Z",
            )
            for idx, wp in enumerate(wps):
                xmp = make_xmp_xml(cs="aruco_local", x=wp.x, y=wp.y, z=wp.z)
                run_data.images[idx] = make_jpeg_bytes(xmp)
            run_data.finalize()
            self.runs[run_id] = run_data

    def start(self, req: StartRequest) -> StartResponse:
        """Handle mission start request."""
        with self.lock:
            # If flight is currently active, return 409
            active_states = {
                FlightState.TAKEOFF,
                FlightState.FLYING_TO_WP,
                FlightState.PROCESSING,
                FlightState.LANDING,
            }
            if self.flight_state in active_states:
                raise HTTPException(status_code=409, detail="Mission already in progress")

            run_id = f"run_{int(time.time() * 1000)}"
            self.active_run_id = run_id
            self._active_waypoints = list(req.waypoints)
            self._active_cs = req.coordinate_system
            self.total_waypoints = len(self._active_waypoints)
            self.current_wp_index = 0
            self.captures_count = 0
            self.is_data_ready = False
            self._abort_requested = False
            self.flight_state = FlightState.TAKEOFF

            run_data = MockRunData(
                run_id=run_id,
                coordinate_system=req.coordinate_system,
                waypoints=self._active_waypoints,
                start_time_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.runs[run_id] = run_data

            return StartResponse(
                status="accepted",
                run_id=run_id,
                run_path=f"/data/runs/{run_id}",
                total_waypoints=self.total_waypoints,
            )

    def stop(self, req: StopRequest) -> StopResponse:
        """Handle mission stop/abort request."""
        with self.lock:
            active_states = {
                FlightState.TAKEOFF,
                FlightState.FLYING_TO_WP,
                FlightState.PROCESSING,
            }
            if self.flight_state not in active_states:
                return StopResponse(
                    status="ok",
                    action_initiated=req.action.value,
                    active_run_id=self.active_run_id,
                    message="Drone is already idle or landing",
                )

            self._abort_requested = True
            self.flight_state = FlightState.LANDING
            return StopResponse(
                status="ok",
                action_initiated=req.action.value,
                active_run_id=self.active_run_id,
                message=f"Action '{req.action.value}' initiated: entering landing sequence",
            )

    def snapshot(self) -> DroneStatus:
        """Return current drone telemetry status snapshot."""
        with self.lock:
            return DroneStatus(
                preset=self.config.preset,
                flight_state=self.flight_state,
                current_wp_index=self.current_wp_index,
                total_waypoints=self.total_waypoints,
                current_pose=self.current_pose,
                captures_count=self.captures_count,
                is_data_ready=self.is_data_ready,
                active_run_id=self.active_run_id,
            )

    def advance(self) -> None:
        """Advance drone state machine by one discrete step."""
        with self.lock:
            if self.flight_state == FlightState.TAKEOFF:
                self.flight_state = FlightState.FLYING_TO_WP
                if self._active_waypoints:
                    wp = self._active_waypoints[0]
                    self.current_pose = DronePose(
                        x=wp.x,
                        y=wp.y,
                        z=wp.z,
                        roll=0.0,
                        pitch=-90.0,
                        yaw=getattr(wp, "yaw", 0.0),
                        coordinate_system=self._active_cs,
                        timestamp=time.time(),
                    )
            elif self.flight_state == FlightState.FLYING_TO_WP:
                self.flight_state = FlightState.PROCESSING
                if 0 <= self.current_wp_index < len(self._active_waypoints):
                    wp = self._active_waypoints[self.current_wp_index]
                    self.current_pose = DronePose(
                        x=wp.x,
                        y=wp.y,
                        z=wp.z,
                        roll=0.0,
                        pitch=-90.0,
                        yaw=getattr(wp, "yaw", 0.0),
                        coordinate_system=self._active_cs,
                        timestamp=time.time(),
                    )
            elif self.flight_state == FlightState.PROCESSING:
                # Capture frame at current waypoint
                if 0 <= self.current_wp_index < len(self._active_waypoints):
                    wp = self._active_waypoints[self.current_wp_index]
                    xmp = make_xmp_xml(
                        cs=self._active_cs,
                        x=wp.x,
                        y=wp.y,
                        z=wp.z,
                        yaw=getattr(wp, "yaw", 0.0),
                        pitch=-90.0,
                    )
                    img_bytes = make_jpeg_bytes(xmp)
                    if self.active_run_id and self.active_run_id in self.runs:
                        self.runs[self.active_run_id].images[self.current_wp_index] = img_bytes
                    self.captures_count += 1

                if self._abort_requested:
                    self.flight_state = FlightState.LANDING
                elif self.current_wp_index + 1 < self.total_waypoints:
                    self.current_wp_index += 1
                    self.flight_state = FlightState.FLYING_TO_WP
                else:
                    self.flight_state = FlightState.LANDING
            elif self.flight_state == FlightState.LANDING:
                if self._abort_requested:
                    self.flight_state = FlightState.ABORTED
                else:
                    self.flight_state = FlightState.COMPLETED
                self.is_data_ready = True
                if self.active_run_id and self.active_run_id in self.runs:
                    self.runs[self.active_run_id].finalize(aborted=self._abort_requested)
                self.active_run_id = None
                self.current_pose = DronePose(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                    coordinate_system=self._active_cs,
                    timestamp=time.time(),
                )


def create_app(
    config: MockConfig | None = None,
    state: MockDroneState | None = None,
) -> FastAPI:
    """Create FastAPI application with onboard drone endpoints and fault injection."""
    cfg = config or MockConfig()
    drone_state = state or MockDroneState(cfg)

    app = FastAPI(title="Agrovision Onboard Mock Server")

    @app.middleware("http")
    async def fault_injection_middleware(request: Request, call_next):
        # Fault injection
        fail_mode = drone_state.config.fail_mode
        if fail_mode == "timeout":
            await asyncio.sleep(5.0)
            return Response("Gateway Timeout", status_code=504)
        if fail_mode == "drop":
            # Force abrupt 502/broken connection response
            return Response(b"", status_code=502)
        if fail_mode == "http500":
            return JSONResponse({"detail": "Simulated onboard server failure"}, status_code=500)

        if drone_state.config.latency_s > 0:
            await asyncio.sleep(drone_state.config.latency_s)

        response = await call_next(request)
        return response

    @app.get("/status", response_model=DroneStatus)
    def get_status() -> DroneStatus:
        return drone_state.snapshot()

    @app.post("/start", response_model=StartResponse, status_code=202)
    def post_start(req: StartRequest) -> StartResponse:
        return drone_state.start(req)

    @app.post("/stop", response_model=StopResponse)
    def post_stop(req: StopRequest) -> StopResponse:
        return drone_state.stop(req)

    @app.get("/runs", response_model=list[DroneRun])
    def get_runs() -> list[DroneRun]:
        with drone_state.lock:
            return [run.to_drone_run() for run in drone_state.runs.values()]

    @app.get("/runs/{run_id}/data")
    def get_run_data(
        run_id: str,
        type: str = Query(..., pattern="^(summary|archive|image)$"),
        wp_index: int | None = Query(None),
    ):
        with drone_state.lock:
            run = drone_state.runs.get(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

            if type == "summary":
                if run.summary is not None:
                    return JSONResponse(run.summary.model_dump())
                # Return current progress summary
                cur_summary = make_summary_detections(
                    run.run_id,
                    run.waypoints[: len(run.images)],
                    coordinate_system=run.coordinate_system,
                )
                return JSONResponse(cur_summary.model_dump())

            if type == "archive":
                if run.zip_bytes is None:
                    # Generate on-the-fly if needed
                    zip_bytes = make_run_zip_bytes(
                        run.run_id,
                        run.waypoints[: len(run.images)],
                        coordinate_system=run.coordinate_system,
                    )
                else:
                    zip_bytes = run.zip_bytes

                return StreamingResponse(
                    io.BytesIO(zip_bytes),
                    media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={run_id}.zip"},
                )

            if type == "image":
                idx = 0 if wp_index is None else wp_index
                img = run.images.get(idx)
                if img is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Image for waypoint {idx} not found in run '{run_id}'",
                    )
                return RawResponse(img, media_type="image/jpeg")

        raise HTTPException(status_code=400, detail="Invalid data type requested")

    @app.delete("/runs/{run_id}", response_model=DeleteRunResponse)
    def delete_run(run_id: str) -> DeleteRunResponse:
        with drone_state.lock:
            if run_id not in drone_state.runs:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

            active_states = {
                FlightState.TAKEOFF,
                FlightState.FLYING_TO_WP,
                FlightState.PROCESSING,
                FlightState.LANDING,
            }
            if drone_state.active_run_id == run_id and drone_state.flight_state in active_states:
                raise HTTPException(
                    status_code=409, detail="Cannot delete active mission run in progress"
                )

            run = drone_state.runs.pop(run_id)
            freed = (
                len(run.zip_bytes)
                if run.zip_bytes
                else sum(len(b) for b in run.images.values()) + 1024
            )
            return DeleteRunResponse(
                deleted_run_id=run_id,
                freed_bytes=freed,
                status="deleted",
            )

    return app


class MockServerHandle:
    """Handle to a running in-thread mock server."""

    def __init__(
        self,
        server: uvicorn.Server,
        server_thread: threading.Thread,
        ticker_thread: threading.Thread | None,
        stop_event: threading.Event,
        host: str,
        port: int,
        state: MockDroneState,
        config: MockConfig,
    ) -> None:
        self._server = server
        self._server_thread = server_thread
        self._ticker_thread = ticker_thread
        self._stop_event = stop_event
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.state = state
        self.config = config

    def stop(self) -> None:
        """Stop server and background ticker thread."""
        self._stop_event.set()
        self._server.should_exit = True
        self._server_thread.join(timeout=3.0)
        if self._ticker_thread is not None:
            self._ticker_thread.join(timeout=3.0)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_in_thread(
    config: MockConfig | None = None,
    port: int = 0,
    state: MockDroneState | None = None,
) -> MockServerHandle:
    """Launch mock onboard server in a background thread."""
    cfg = config or MockConfig()
    drone_state = state or MockDroneState(cfg)
    app = create_app(config=cfg, state=drone_state)

    if port == 0:
        port = _find_free_port()

    host = "127.0.0.1"
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(uvicorn_config)
    stop_event = threading.Event()

    def run_server() -> None:
        server.run()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to be up
    timeout = 5.0
    start_time = time.time()
    while not server.started:
        if time.time() - start_time > timeout:
            raise RuntimeError("Mock server failed to start within timeout")
        time.sleep(0.02)

    ticker_thread: threading.Thread | None = None
    if cfg.tick_s > 0:

        def run_ticker() -> None:
            while not stop_event.is_set():
                time.sleep(cfg.tick_s)
                if not stop_event.is_set():
                    drone_state.advance()

        ticker_thread = threading.Thread(target=run_ticker, daemon=True)
        ticker_thread.start()

    return MockServerHandle(
        server=server,
        server_thread=server_thread,
        ticker_thread=ticker_thread,
        stop_event=stop_event,
        host=host,
        port=port,
        state=drone_state,
        config=cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrovision Onboard Mock Server")
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    parser.add_argument("--tick", type=float, default=0.2, help="State tick interval in seconds")
    parser.add_argument(
        "--fail-mode",
        type=str,
        default="none",
        choices=["none", "timeout", "drop", "http500"],
        help="Simulated failure mode",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="clever4_csi_onnx",
        help="Drone hardware/sensor preset name",
    )
    args = parser.parse_args()

    mock_cfg = MockConfig(
        tick_s=args.tick,
        fail_mode=args.fail_mode,  # type: ignore
        preset=args.preset,
    )
    mock_app = create_app(mock_cfg)
    uvicorn.run(mock_app, host="0.0.0.0", port=args.port)
