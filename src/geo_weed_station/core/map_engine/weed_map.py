"""Core WeedMap point layer aggregation and MapEngine coordinator."""

from __future__ import annotations

from collections.abc import Iterable

from geo_weed_station.config.settings import MapSettings
from geo_weed_station.core.map_engine.coordinates import CoordinateConverter
from geo_weed_station.core.map_engine.errors import GeoreferenceMissingError
from geo_weed_station.core.map_engine.heatmap import CountHeatmap, HeatmapAlgorithm
from geo_weed_station.core.map_engine.projector import ProjectionStrategy, validate_georef
from geo_weed_station.core.map_engine.statistics import calculate_statistics, convex_hull_area
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import Bounds, HeatmapGrid, MapStatistics, WeedPoint
from geo_weed_station.schemas.onboard_api import SummaryDetections, summary_to_frame_payloads
from geo_weed_station.schemas.payload import FramePayload


class WeedMap:
    """In-memory collection of georeferenced weed points for the active session."""

    def __init__(self, coord_system: CoordinateSystem = CoordinateSystem.LOCAL_ARUCO) -> None:
        self.points: list[WeedPoint] = []
        self.coord_system = coord_system

    def add_points(self, pts: Iterable[WeedPoint]) -> None:
        """Add new localized weed points to the map."""
        for p in pts:
            self.points.append(p)
            self.coord_system = p.coord_system

    def bounds(self, margin_m: float = 1.0) -> Bounds:
        """Calculate spatial bounding box of all points with optional margin."""
        if not self.points:
            return Bounds(min_x=0.0, min_y=0.0, max_x=1.0, max_y=1.0)

        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return Bounds(
            min_x=min(xs) - margin_m,
            min_y=min(ys) - margin_m,
            max_x=max(xs) + margin_m,
            max_y=max(ys) + margin_m,
        )

    def convex_hull_area(self) -> float:
        """Calculate surveyed polygon area in square meters."""
        return convex_hull_area(self.points)

    def clear(self) -> None:
        """Remove all points from the map."""
        self.points.clear()


class MapEngine:
    """Unified coordinator for frame ingestion, spatial projections, heatmap, and analytics."""

    def __init__(
        self,
        projector: ProjectionStrategy,
        settings: MapSettings,
        heatmap: HeatmapAlgorithm | None = None,
    ) -> None:
        self.projector = projector
        self.settings = settings
        self.heatmap_algo: HeatmapAlgorithm = heatmap if heatmap is not None else CountHeatmap()
        self.weed_map = WeedMap()
        self._is_locked: bool = False
        self._current_grid: HeatmapGrid | None = None
        self._wgs84_origin: tuple[float, float] | None = None
        self._converter: CoordinateConverter | None = None

    @property
    def wgs84_origin(self) -> tuple[float, float] | None:
        """Reference origin (latitude_0, longitude_0) for local metric projection in WGS84 mode."""
        return self._wgs84_origin

    def set_wgs84_origin(self, origin: tuple[float, float]) -> None:
        """Explicitly set WGS84 origin (lat, lon) for local metric projection."""
        self._wgs84_origin = origin
        self._converter = CoordinateConverter(wgs84_origin=origin)

    def is_locked(self) -> bool:
        """Return True if mapping engine is locked due to missing georeferencing."""
        return self._is_locked

    def reset(self) -> None:
        """Clear all map layers, reset statistics, and unlock mapping mode."""
        self.weed_map.clear()
        self._is_locked = False
        self._current_grid = None
        self._wgs84_origin = None
        self._converter = None

    def ingest_frame(self, payload: FramePayload) -> list[WeedPoint]:
        """Project detected weeds on frame and ingest them into WeedMap layer.
        
        Raises:
            GeoreferenceMissingError: If frame has H <= 0 or missing pose. Engine locks.
        """
        try:
            validate_georef(payload)
        except GeoreferenceMissingError as e:
            self._is_locked = True
            raise e

        assert payload.pose is not None
        projected_detections = self.projector.project(payload)

        new_points: list[WeedPoint] = []
        is_wgs84 = payload.pose.coordinate_system == CoordinateSystem.WGS84

        if is_wgs84 and self._wgs84_origin is None:
            # Anchor local metric coordinate origin to first frame position (lat, lon)
            self._wgs84_origin = (payload.pose.y, payload.pose.x)
            self._converter = CoordinateConverter(wgs84_origin=self._wgs84_origin)

        for det in projected_detections:
            if det.world_x is None or det.world_y is None:
                continue

            if is_wgs84:
                assert self._converter is not None
                weed_lon = det.world_x
                weed_lat = det.world_y
                local_x, local_y = self._converter.wgs84_to_local(lat=weed_lat, lon=weed_lon)
                wp = WeedPoint(
                    weed_id=det.weed_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    x=round(local_x, 3),
                    y=round(local_y, 3),
                    coord_system=CoordinateSystem.WGS84,
                    source_frame=payload.frame_id,
                    lat=round(weed_lat, 8),
                    lon=round(weed_lon, 8),
                )
            else:
                wp = WeedPoint(
                    weed_id=det.weed_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    x=det.world_x,
                    y=det.world_y,
                    coord_system=CoordinateSystem.LOCAL_ARUCO,
                    source_frame=payload.frame_id,
                )
            new_points.append(wp)

        self.weed_map.add_points(new_points)
        return new_points

    def ingest_summary(self, summary: SummaryDetections) -> list[WeedPoint]:
        """Ingest aggregated detections summary returned by onboard computer (GET /runs/{id}/data?type=summary)."""
        payloads = summary_to_frame_payloads(summary)
        all_new_points: list[WeedPoint] = []

        for p in payloads:
            if not p.has_georef():
                self._is_locked = True
                raise GeoreferenceMissingError(
                    f"Summary item '{p.frame_id}' lacks valid coordinates or altitude"
                )

            assert p.pose is not None
            is_wgs84 = p.pose.coordinate_system == CoordinateSystem.WGS84
            if is_wgs84 and self._wgs84_origin is None:
                self._wgs84_origin = (p.pose.y, p.pose.x)
                self._converter = CoordinateConverter(wgs84_origin=self._wgs84_origin)

            for det in p.detections:
                if det.world_x is not None and det.world_y is not None:
                    if is_wgs84:
                        assert self._converter is not None
                        weed_lon = det.world_x
                        weed_lat = det.world_y
                        local_x, local_y = self._converter.wgs84_to_local(lat=weed_lat, lon=weed_lon)
                        wp = WeedPoint(
                            weed_id=det.weed_id,
                            class_name=det.class_name,
                            confidence=det.confidence,
                            x=round(local_x, 3),
                            y=round(local_y, 3),
                            coord_system=CoordinateSystem.WGS84,
                            source_frame=p.frame_id,
                            lat=round(weed_lat, 8),
                            lon=round(weed_lon, 8),
                        )
                    else:
                        wp = WeedPoint(
                            weed_id=det.weed_id,
                            class_name=det.class_name,
                            confidence=det.confidence,
                            x=det.world_x,
                            y=det.world_y,
                            coord_system=p.pose.coordinate_system,
                            source_frame=p.frame_id,
                        )
                    all_new_points.append(wp)

        self.weed_map.add_points(all_new_points)
        return all_new_points

    def build_heatmap(self, cell_m: float | None = None) -> HeatmapGrid:
        """Compute raster weed density heatmap using configured algorithm."""
        cell_size = cell_m if cell_m is not None else self.settings.heatmap_cell_m
        bounds = self.weed_map.bounds()
        self._current_grid = self.heatmap_algo.compute(
            self.weed_map.points, cell_m=cell_size, bounds=bounds
        )
        return self._current_grid

    def statistics(self) -> MapStatistics:
        """Calculate aggregated field metrics and critical density percentage."""
        return calculate_statistics(
            self.weed_map.points,
            grid=self._current_grid,
            critical_threshold=self.settings.critical_density_threshold,
        )
