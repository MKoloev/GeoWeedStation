"""Digital mapping engine: coordinate conversions, spatial projections, heatmap, and exporters."""

from geo_weed_station.core.map_engine.coordinates import CoordinateConverter
from geo_weed_station.core.map_engine.errors import (
    GeoreferenceMissingError,
    OriginNotSetError,
)
from geo_weed_station.core.map_engine.exporter import MapExporter
from geo_weed_station.core.map_engine.heatmap import (
    CountHeatmap,
    HeatmapAlgorithm,
    KdeHeatmap,
)
from geo_weed_station.core.map_engine.projector import (
    DistortionModel,
    NadirProjector,
    ProjectionStrategy,
    get_projector,
    register_projector,
    validate_georef,
)
from geo_weed_station.core.map_engine.statistics import (
    calculate_statistics,
    convex_hull_area,
)
from geo_weed_station.core.map_engine.weed_map import MapEngine, WeedMap

__all__ = [
    # errors
    "GeoreferenceMissingError",
    "OriginNotSetError",
    # coordinates
    "CoordinateConverter",
    # projector
    "DistortionModel",
    "ProjectionStrategy",
    "NadirProjector",
    "validate_georef",
    "get_projector",
    "register_projector",
    # heatmap
    "HeatmapAlgorithm",
    "CountHeatmap",
    "KdeHeatmap",
    # statistics
    "convex_hull_area",
    "calculate_statistics",
    # weed_map
    "WeedMap",
    "MapEngine",
    # exporter
    "MapExporter",
]
