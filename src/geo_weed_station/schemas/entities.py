"""Domain entities and basic geometric/spatial data structures."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CoordinateSystem(str, Enum):
    """Supported coordinate systems in GeoWeedStation."""

    LOCAL_ARUCO = "local_aruco"  # Internal metric coordinate system (XMP, MapEngine)
    WGS84 = "wgs84"  # Global WGS84 geographic coordinate system (Lat/Lon)


class BBox(BaseModel):
    """Normalized bounding box in range [0.0, 1.0]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = Field(..., description="Top-left corner X normalized [0..1]")
    y: float = Field(..., description="Top-left corner Y normalized [0..1]")
    w: float = Field(..., description="Bounding box width normalized [0..1]")
    h: float = Field(..., description="Bounding box height normalized [0..1]")

    @field_validator("x", "y", "w", "h")
    @classmethod
    def check_normalized(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"BBox coordinate must be in [0.0, 1.0], got {v}")
        return v

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert [x, y, w, h] to [xmin, ymin, xmax, ymax] normalized coordinates."""
        return (self.x, self.y, min(1.0, self.x + self.w), min(1.0, self.y + self.h))


class DetectedWeed(BaseModel):
    """Single detected weed object on a frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weed_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    world_x: float | None = None  # Metres in local_aruco or Longitude in wgs84
    world_y: float | None = None  # Metres in local_aruco or Latitude in wgs84

    @field_validator("confidence")
    @classmethod
    def check_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be in range [0.0, 1.0], got {v}")
        return v


class CameraSpatialPose(BaseModel):
    """Spatial position and attitude angles of the camera at exposure time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coordinate_system: CoordinateSystem
    x: float  # Metres (local_aruco) or Longitude (wgs84)
    y: float  # Metres (local_aruco) or Latitude (wgs84)
    altitude_m: float  # Relative altitude H above ground
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


class SensorOptics(BaseModel):
    """Camera lens and sensor optical parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    focal_length_mm: float
    fov_h_deg: float | None = None
    fov_v_deg: float | None = None
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None
    principal_point_px: tuple[float, float] | None = None
    band_names: list[str] = Field(default_factory=list)
