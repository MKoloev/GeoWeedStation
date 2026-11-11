"""Data contracts for digital mapping, heatmap grids, statistics, and export options."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from geo_weed_station.schemas.entities import CoordinateSystem


class Bounds(BaseModel):
    """2D rectangular spatial bounding box."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        """Width of bounding box along X."""
        return max(0.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        """Height of bounding box along Y."""
        return max(0.0, self.max_y - self.min_y)


class WeedPoint(BaseModel):
    """Spatial point feature representing a localized weed on the map."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    weed_id: str
    class_name: str
    confidence: float
    x: float  # Metres in local metric plane (local_aruco or local metric projection when wgs84)
    y: float  # Metres in local metric plane
    coord_system: CoordinateSystem
    source_frame: str
    lat: float | None = None  # Global WGS84 Latitude in degrees
    lon: float | None = None  # Global WGS84 Longitude in degrees
    bbox: Any = None  # Normalized bbox (x, y, w, h) or BBox instance
    is_duplicate: bool = False
    best_source_frame: str | None = None
    best_frame_id: str | None = None
    duplicate_count: int = 1
    stage_name: str | None = None
    class_prob: float | None = None


class PrescriptionZone(BaseModel):
    """Vector polygon representing contiguous high-density infestation zone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    zone_id: str
    polygon: tuple[tuple[float, float], ...] | list[tuple[float, float]]  # [(x, y), ...] in local metric plane
    wgs84_polygon: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None = None  # [(lon, lat), ...]
    polygon_wgs84: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None = None  # alias
    area_m2: float
    mean_density_per_m2: float = 0.0
    mean_density: float = 0.0
    weed_count: int = 0
    critical_threshold: float = 3.0
    dominant_class: str = "weed_dicot"
    treatment_recommendation: str = "Spot Spray"


class FrameFootprint(BaseModel):
    """Spatial footprint and orientation of an ingested image frame for quick-stitch preview."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    frame_id: str
    source_path: str | None = None
    center_x: float  # Metres in local metric plane
    center_y: float  # Metres in local metric plane
    width_m: float
    height_m: float
    altitude_m: float
    yaw_deg: float
    polygon: tuple[tuple[float, float], ...] | list[tuple[float, float]]  # 4 corner coordinates [(x, y), ...]
    thumbnail_path: Path | str | None = None
    lat: float | None = None
    lon: float | None = None
    thumbnail: Any = Field(default=None, exclude=True)


FrameMosaicItem = FrameFootprint  # Alias for compatibility


class SummaryReport(BaseModel):
    """Summary report for complete surveyed mission and prescription zones."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_detections: int
    unique_weeds: int
    deduplication_ratio: float = 1.0
    surveyed_area_m2: float
    mean_density_per_m2: float
    critical_area_percent: float
    by_class: dict[str, int] = Field(default_factory=dict)
    prescription_zones_count: int = 0
    frames_count: int = 0
    zones: list[PrescriptionZone] = Field(default_factory=list)


class HeatmapGrid(BaseModel):
    """2D raster grid representing weed density distribution."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    origin: tuple[float, float]  # (min_x, min_y) bottom-left coordinate
    cell_size_m: float
    bounds: Bounds
    matrix: Any = Field(default=None, exclude=True)  # np.ndarray (rows, cols), excluded from JSON serialization


class MapStatistics(BaseModel):
    """Aggregated quantitative agronomic metrics for the surveyed field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_weeds: int
    by_class: dict[str, int]
    surveyed_area_m2: float
    mean_density_per_m2: float
    critical_area_percent: float


class PngExportOptions(BaseModel):
    """Configuration options for map raster rendering and PNG export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    background: Literal["white", "transparent", "orthophoto"] = "white"
    show_grid: bool = True
    show_legend: bool = True
    show_scalebar: bool = True
    show_route: bool = True
    dpi: int = 300
    orthophoto_path: Path | None = None
