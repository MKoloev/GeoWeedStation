"""DTO models for onboard REST API (agrovision_onboard) communication and format conversions."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
)
from geo_weed_station.schemas.mission import MissionMode, MissionTask, Waypoint
from geo_weed_station.schemas.payload import FramePayload


class StopAction(str, Enum):
    """Action for drone emergency or regular stop."""

    LAND = "land"
    RTH = "rth"


class StartRequest(BaseModel):
    """Request payload for POST /start endpoint on drone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coordinate_system: str = Field(..., description="'aruco_local' or 'wgs84'")
    use_nn: bool
    save_policy: str = Field(..., description="'all', 'images_only', or 'metadata_only'")
    waypoints: List[Waypoint]


class StartResponse(BaseModel):
    """Response payload from POST /start endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    run_id: str
    run_path: str
    total_waypoints: int


class StopRequest(BaseModel):
    """Request payload for POST /stop endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: StopAction = StopAction.LAND


class StopResponse(BaseModel):
    """Response payload from POST /stop endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    action_initiated: str
    active_run_id: Optional[str] = None
    message: str


class DroneRun(BaseModel):
    """Information about a mission run directory on the drone (GET /runs)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    start_time: str
    status: str
    directory_path: str
    size_bytes: int
    contents: List[str]


class SummaryTargetCoords(BaseModel):
    """Local ground metric coordinates of detected weed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float


class SummaryDetection(BaseModel):
    """Single detection entry in mission summary JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_name: str
    confidence: float
    bbox_px: List[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixels")
    target_local_coords: Optional[SummaryTargetCoords] = None


class SummaryItem(BaseModel):
    """Capture and detection item for a waypoint in mission summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wp_index: int
    timestamp: float
    camera_x: float
    camera_y: float
    altitude_m: float
    detections: List[SummaryDetection] = Field(default_factory=list)


class SummaryDetections(BaseModel):
    """Complete summary structure returned by GET /runs/{id}/data?type=summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    coordinate_system: str
    detections_count: int
    items: List[SummaryItem]


class DeleteRunResponse(BaseModel):
    """Response model for DELETE /runs/{id} endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deleted_run_id: str
    freed_bytes: int
    status: str


def mission_to_start_request(task: MissionTask) -> StartRequest:
    """Convert GCS MissionTask domain model to onboard StartRequest DTO."""
    cs_name = "aruco_local" if task.coordinate_system == CoordinateSystem.LOCAL_ARUCO else "wgs84"
    use_nn = task.mode == MissionMode.INFERENCE_SEQUENTIAL
    return StartRequest(
        coordinate_system=cs_name,
        use_nn=use_nn,
        save_policy=task.save_policy.value,
        waypoints=list(task.waypoints),
    )


def summary_to_frame_payloads(
    summary: SummaryDetections, default_frame_size_px: tuple[int, int] = (640, 640)
) -> List[FramePayload]:
    """Convert onboard SummaryDetections to a list of FramePayload instances."""
    cs = (
        CoordinateSystem.LOCAL_ARUCO
        if summary.coordinate_system == "aruco_local"
        else CoordinateSystem.WGS84
    )
    img_w, img_h = default_frame_size_px

    payloads: List[FramePayload] = []
    for item in summary.items:
        frame_id = f"{summary.run_id}_wp{item.wp_index}"
        pose = CameraSpatialPose(
            coordinate_system=cs,
            x=item.camera_x,
            y=item.camera_y,
            altitude_m=item.altitude_m,
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
        )

        detections: List[DetectedWeed] = []
        for idx, det in enumerate(item.detections):
            # Convert pixel coords [x1, y1, x2, y2] to normalized [0..1]
            x1, y1, x2, y2 = det.bbox_px
            norm_x = min(1.0, max(0.0, float(x1) / img_w))
            norm_y = min(1.0, max(0.0, float(y1) / img_h))
            norm_w = min(1.0 - norm_x, max(0.0, float(x2 - x1) / img_w))
            norm_h = min(1.0 - norm_y, max(0.0, float(y2 - y1) / img_h))

            world_x = det.target_local_coords.x if det.target_local_coords else None
            world_y = det.target_local_coords.y if det.target_local_coords else None

            detected_weed = DetectedWeed(
                weed_id=f"{frame_id}_weed_{idx}",
                class_id=0,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=BBox(x=norm_x, y=norm_y, w=norm_w, h=norm_h),
                world_x=world_x,
                world_y=world_y,
            )
            detections.append(detected_weed)

        payload = FramePayload(
            frame_id=frame_id,
            source_path=None,
            image_size_px=default_frame_size_px,
            pose=pose,
            optics=None,
            detections=detections,
        )
        payloads.append(payload)

    return payloads
