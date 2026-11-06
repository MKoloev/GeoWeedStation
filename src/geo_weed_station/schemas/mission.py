"""Flight mission planning schemas and task definitions."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from geo_weed_station.schemas.entities import CoordinateSystem


class Waypoint(BaseModel):
    """Target waypoint in metric local or geographic coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float
    z: float
    yaw: float = 0.0


class MissionMode(str, Enum):
    """Execution mode of the onboard module during mission."""

    COLLECTION_ONLY = "collection_only"  # Data capture only without onboard inference (use_nn=False)
    INFERENCE_SEQUENTIAL = "inference_sequential"  # Onboard inference at each waypoint (use_nn=True)


class SavePolicy(str, Enum):
    """Storage save policy on the onboard computer."""

    ALL = "all"  # Images + detections JSON
    IMAGES_ONLY = "images_only"  # Raw images only
    METADATA_ONLY = "metadata_only"  # Only JSON metadata and detections without raw images


class MissionTask(BaseModel):
    """Complete flight mission task configuration for drone onboard execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coordinate_system: CoordinateSystem
    mode: MissionMode
    save_policy: SavePolicy
    waypoints: list[Waypoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_task(self) -> MissionTask:
        if not self.waypoints:
            raise ValueError("MissionTask waypoints list cannot be empty")
        if self.mode == MissionMode.COLLECTION_ONLY and self.save_policy == SavePolicy.METADATA_ONLY:
            raise ValueError(
                "Combination COLLECTION_ONLY + METADATA_ONLY is invalid (no images or detections saved)"
            )
        return self
