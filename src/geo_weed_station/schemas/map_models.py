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

    model_config = ConfigDict(frozen=True, extra="forbid")

    weed_id: str
    class_name: str
    confidence: float
    x: float  # Metres (local_aruco) or Longitude (wgs84)
    y: float  # Metres (local_aruco) or Latitude (wgs84)
    coord_system: CoordinateSystem
    source_frame: str


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
    dpi: int = 300
    orthophoto_path: Path | None = None
