"""Unit tests for WeedMap and MapEngine coordinator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geo_weed_station.config.settings import MapSettings
from geo_weed_station.core.map_engine.errors import GeoreferenceMissingError
from geo_weed_station.core.map_engine.heatmap import KdeHeatmap
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
    SensorOptics,
)
from geo_weed_station.schemas.onboard_api import SummaryDetections
from geo_weed_station.schemas.payload import FramePayload

SAMPLES_DIR = Path(__file__).parent / "data" / "api_samples"


@pytest.fixture
def map_engine() -> MapEngine:
    settings = MapSettings(
        heatmap_cell_m=0.5,
        critical_density_threshold=3.0,
        max_local_extent_m=3.0,
    )
    projector = NadirProjector()
    return MapEngine(projector=projector, settings=settings)


def test_ingest_frame_valid(map_engine: MapEngine):
    """Verify ingesting a valid georeferenced frame adds points to WeedMap."""
    pose = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=1.5,
        y=2.0,
        altitude_m=2.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    optics = SensorOptics(focal_length_mm=3.6, principal_point_px=(320.0, 240.0))
    weed = DetectedWeed(
        weed_id="w1",
        class_id=0,
        class_name="weed_dicot",
        confidence=0.88,
        bbox=BBox(x=0.45, y=0.45, w=0.1, h=0.1),
    )
    payload = FramePayload(
        frame_id="frame_001",
        image_size_px=(640, 480),
        pose=pose,
        optics=optics,
        detections=[weed],
    )

    pts = map_engine.ingest_frame(payload)
    assert len(pts) == 1
    assert map_engine.is_locked() is False
    assert len(map_engine.weed_map.points) == 1

    p = pts[0]
    assert p.class_name == "weed_dicot"
    assert p.x == pytest.approx(1.5, abs=0.01)
    assert p.y == pytest.approx(2.0, abs=0.01)


def test_ingest_frame_invalid_locks_engine(map_engine: MapEngine):
    """Verify ingesting frame with altitude <= 0 raises GeoreferenceMissingError and locks engine."""
    pose_invalid = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=1.0,
        y=1.0,
        altitude_m=0.0,  # Invalid
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    payload_bad = FramePayload(
        frame_id="bad_frame",
        image_size_px=(640, 480),
        pose=pose_invalid,
    )

    with pytest.raises(GeoreferenceMissingError):
        map_engine.ingest_frame(payload_bad)

    assert map_engine.is_locked() is True

    # Reset clears locked state
    map_engine.reset()
    assert map_engine.is_locked() is False
    assert len(map_engine.weed_map.points) == 0


def test_ingest_summary(map_engine: MapEngine):
    """Verify ingesting onboard SummaryDetections adds points correctly."""
    summary_raw = json.loads((SAMPLES_DIR / "summary_response.json").read_text(encoding="utf-8"))
    summary = SummaryDetections.model_validate(summary_raw)

    pts = map_engine.ingest_summary(summary)
    assert len(pts) == 1
    assert pts[0].class_name == "weed_dicot"
    assert pts[0].x == 1.34
    assert pts[0].y == 0.12


def test_statistics_and_critical_density(map_engine: MapEngine):
    """Verify statistical metrics calculation including critical area percentage."""
    pose = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=0.0,
        y=0.0,
        altitude_m=2.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    # 4 detections in the same small cell area to exceed critical threshold (3.0)
    weeds = [
        DetectedWeed(
            weed_id=f"w_{i}",
            class_id=i % 2,
            class_name="weed_dicot" if i % 2 == 0 else "weed_monocot",
            confidence=0.9,
            bbox=BBox(x=0.45, y=0.45, w=0.05, h=0.05),
        )
        for i in range(4)
    ]
    payload = FramePayload(
        frame_id="f_stats",
        image_size_px=(640, 480),
        pose=pose,
        detections=weeds,
    )
    map_engine.ingest_frame(payload)
    grid = map_engine.build_heatmap(cell_m=0.5)

    stats = map_engine.statistics()
    assert stats.total_weeds == 4
    assert stats.by_class["weed_dicot"] == 2
    assert stats.by_class["weed_monocot"] == 2
    assert stats.surveyed_area_m2 > 0.0
    assert stats.mean_density_per_m2 > 0.0
    # Because all 4 points land in the center cell, density in that cell is 4 >= 3.0
    assert stats.critical_area_percent > 0.0


def test_heatmap_strategy_substitution(map_engine: MapEngine):
    """Verify pluggable heatmap algorithm substitution."""
    assert map_engine.heatmap_algo.__class__.__name__ == "CountHeatmap"
    map_engine.heatmap_algo = KdeHeatmap(bandwidth_m=0.8)
    assert map_engine.heatmap_algo.__class__.__name__ == "KdeHeatmap"


def test_wgs84_local_metric_projection_and_heatmap(map_engine: MapEngine, tmp_path: Path):
    """Verify that WGS84 frames are projected into local metric space under the hood.
    
    Verifies:
    1. Origin is anchored to first frame's GPS position.
    2. WeedPoint.x and y are in meters; lat and lon are in degrees.
    3. Heatmap grid has proper metric resolution and does not collapse to 1x1.
    4. ConvexHull area is calculated in real square meters.
    5. CSV and GeoJSON exports output GPS coordinates.
    """
    import numpy as np
    from geo_weed_station.core.map_engine.exporter import MapExporter

    optics = SensorOptics(focal_length_mm=40.0, sensor_width_mm=8.57, sensor_height_mm=6.43)

    # Frame 1: origin
    pose_1 = CameraSpatialPose(
        coordinate_system=CoordinateSystem.WGS84,
        x=55.304049,  # lon
        y=51.777788,  # lat
        altitude_m=10.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=-178.0,
    )
    w1 = DetectedWeed(
        weed_id="w1",
        class_id=0,
        class_name="weed_dicot",
        confidence=0.9,
        bbox=BBox(x=0.5, y=0.5, w=0.05, h=0.05),
    )
    f1 = FramePayload(
        frame_id="frame_0001",
        image_size_px=(8192, 6144),
        pose=pose_1,
        optics=optics,
        detections=[w1],
    )

    # Frame 2: ~50 meters South
    pose_2 = CameraSpatialPose(
        coordinate_system=CoordinateSystem.WGS84,
        x=55.304066,  # lon
        y=51.777339,  # lat (diff 0.000449 deg ~= 50 m)
        altitude_m=10.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=-178.0,
    )
    w2 = DetectedWeed(
        weed_id="w2",
        class_id=1,
        class_name="weed_monocot",
        confidence=0.88,
        bbox=BBox(x=0.5, y=0.5, w=0.05, h=0.05),
    )
    f2 = FramePayload(
        frame_id="frame_0002",
        image_size_px=(8192, 6144),
        pose=pose_2,
        optics=optics,
        detections=[w2],
    )

    pts1 = map_engine.ingest_frame(f1)
    pts2 = map_engine.ingest_frame(f2)

    assert map_engine.wgs84_origin == (51.777788, 55.304049)
    assert len(map_engine.weed_map.points) == 2

    p1, p2 = map_engine.weed_map.points

    # Point 1 is at Origin (0, 0 meters)
    assert p1.x == pytest.approx(0.0, abs=0.5)
    assert p1.y == pytest.approx(0.0, abs=0.5)
    assert p1.lat == pytest.approx(51.777788, abs=1e-5)
    assert p1.lon == pytest.approx(55.304049, abs=1e-5)

    # Point 2 is ~50 meters South (Y ~ -50m)
    assert p2.x == pytest.approx(1.17, abs=1.0)
    assert p2.y == pytest.approx(-49.9, abs=1.0)
    assert p2.lat == pytest.approx(51.777339, abs=1e-5)
    assert p2.lon == pytest.approx(55.304066, abs=1e-5)

    # Spatial bounds are in meters
    bounds = map_engine.weed_map.bounds(margin_m=1.0)
    assert bounds.height == pytest.approx(52.0, abs=2.0)  # ~50m distance + 2m margin

    # Heatmap resolution: with cell_m=1.0, rows should be ~52, NOT 1
    grid = map_engine.build_heatmap(cell_m=1.0)
    assert grid.matrix.shape[0] >= 45
    assert np.sum(grid.matrix) == 2.0

    # Statistics: area and density
    stats = map_engine.statistics()
    assert stats.total_weeds == 2
    assert stats.surveyed_area_m2 >= 1.0

    # Exports
    exporter = MapExporter()
    csv_file = tmp_path / "wgs_weeds.csv"
    geojson_file = tmp_path / "wgs_weeds.geojson"
    exporter.to_csv(map_engine.weed_map, csv_file)
    exporter.to_geojson(map_engine.weed_map, geojson_file)

    # Verify CSV has GPS coordinates
    csv_text = csv_file.read_text(encoding="utf-8")
    assert "55.3040" in csv_text
    assert "51.7777" in csv_text

    # Verify GeoJSON has EPSG:4326 coords [lon, lat]
    doc = json.loads(geojson_file.read_text(encoding="utf-8"))
    assert doc["features"][0]["geometry"]["coordinates"] == [pytest.approx(55.304048, abs=1e-5), pytest.approx(51.777788, abs=1e-5)]
