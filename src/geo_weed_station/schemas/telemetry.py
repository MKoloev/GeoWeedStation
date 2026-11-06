"""Telemetry schemas for drone status monitoring and connection state."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class FlightState(str, Enum):
    """Drone state machine status values from onboard system."""

    IDLE = "idle"
    TAKEOFF = "takeoff"
    FLYING_TO_WP = "flying_to_wp"
    PROCESSING = "processing"
    LANDING = "landing"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


class DronePose(BaseModel):
    """Real-time 6-DOF spatial pose and coordinate system of the drone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    coordinate_system: str
    timestamp: float


class DroneStatus(BaseModel):
    """Response model for GET /status onboard endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preset: str
    flight_state: FlightState
    current_wp_index: int
    total_waypoints: int
    current_pose: DronePose | None = None
    captures_count: int
    is_data_ready: bool
    active_run_id: str | None = None


class ConnectionState(str, Enum):
    """Connection state of the GCS telemetry client."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    TIMEOUT = "timeout"
    ERROR = "error"
