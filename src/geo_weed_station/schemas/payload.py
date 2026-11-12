"""Unified frame payload data container for digital mapping and analytics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_weed_station.schemas.entities import CameraSpatialPose, DetectedWeed, SensorOptics


class FramePayload(BaseModel):
    """Normalized frame data container carrying pose, optics, and detection results."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    frame_id: str
    source_path: str | None = None
    image_size_px: tuple[int, int]  # (width, height)
    image: Any | None = Field(default=None, exclude=True)  # In-memory image buffer, excluded from serialization
    detector_image: Any | None = Field(default=None, exclude=True)
    pose: CameraSpatialPose | None = None
    optics: SensorOptics | None = None
    detections: list[DetectedWeed] = Field(default_factory=list)

    def has_georef(self) -> bool:
        """Check if frame contains valid georeferencing tags for mapping projection.
        
        Returns False if pose is None, altitude_m <= 0, or any spatial coordinate/attitude angle is missing.
        """
        if self.pose is None:
            return False
        if self.pose.altitude_m <= 0.0:
            return False
        if (
            self.pose.x is None
            or self.pose.y is None
            or self.pose.roll_deg is None
            or self.pose.pitch_deg is None
            or self.pose.yaw_deg is None
        ):
            return False
        return True
