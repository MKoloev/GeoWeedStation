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
