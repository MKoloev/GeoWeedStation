"""Data schemas and transfer objects for GeoWeedStation."""

from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
    SensorOptics,
)
from geo_weed_station.schemas.map_models import (
    Bounds,
    FrameFootprint,
    FrameMosaicItem,
    HeatmapGrid,
    MapStatistics,
    PngExportOptions,
    PrescriptionZone,
    SummaryReport,
    WeedPoint,
)
from geo_weed_station.schemas.mission import (
    MissionMode,
    MissionTask,
    SavePolicy,
    Waypoint,
)
from geo_weed_station.schemas.onboard_api import (
    DeleteRunResponse,
    DroneRun,
    StartRequest,
    StartResponse,
    StopAction,
    StopRequest,
    StopResponse,
    SummaryDetection,
    SummaryDetections,
    SummaryItem,
    SummaryTargetCoords,
    mission_to_start_request,
    summary_to_frame_payloads,
)
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.schemas.telemetry import (
    ConnectionState,
    DronePose,
    DroneStatus,
    FlightState,
)

__all__ = [
    # entities
    "BBox",
    "CameraSpatialPose",
    "CoordinateSystem",
    "DetectedWeed",
    "SensorOptics",
    # payload
    "FramePayload",
    # mission
    "MissionMode",
    "MissionTask",
    "SavePolicy",
    "Waypoint",
    # telemetry
    "ConnectionState",
    "DronePose",
    "DroneStatus",
    "FlightState",
    # onboard_api
    "DeleteRunResponse",
    "DroneRun",
    "StartRequest",
    "StartResponse",
    "StopAction",
    "StopRequest",
    "StopResponse",
    "SummaryDetection",
    "SummaryDetections",
    "SummaryItem",
    "SummaryTargetCoords",
    "mission_to_start_request",
    "summary_to_frame_payloads",
    # map_models
    "Bounds",
    "FrameFootprint",
    "FrameMosaicItem",
    "HeatmapGrid",
    "MapStatistics",
    "PngExportOptions",
    "PrescriptionZone",
    "SummaryReport",
    "WeedPoint",
]
