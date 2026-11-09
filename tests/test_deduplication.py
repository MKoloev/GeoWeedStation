"""Comprehensive unit tests for weed spatial deduplication."""

from __future__ import annotations

from geo_weed_station.core.map_engine.deduplication import deduplicate_weeds
from geo_weed_station.schemas.entities import BBox, CoordinateSystem
from geo_weed_station.schemas.map_models import WeedPoint


def test_deduplicate_empty():
    assert deduplicate_weeds([]) == []


def test_deduplicate_single_point():
    p = WeedPoint(
        weed_id="w1",
        class_name="weed_dicot",
        confidence=0.85,
        x=5.0,
        y=5.0,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="frame_001",
        bbox=BBox(x=0.48, y=0.52, w=0.04, h=0.04),
    )
    res = deduplicate_weeds([p])
    assert len(res) == 1
    assert res[0].weed_id == "w1"
    assert res[0].best_frame_id == "frame_001"
    assert res[0].duplicate_count == 1


def test_deduplicate_overlapping_frames_picks_nadir_frame():
    """Verify that when the same weed is detected in two consecutive frames,
    the detection closest to nadir (image center [0.5, 0.5]) is selected as best_frame_id.
    """
    p1 = WeedPoint(
        weed_id="w_frame1",
        class_name="Ambrosia",
        confidence=0.75,
        x=10.02,
        y=15.01,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="DJI_0001",
        bbox=BBox(x=0.85, y=0.85, w=0.1, h=0.1),
    )
    p2 = WeedPoint(
        weed_id="w_frame2",
        class_name="Ambrosia",
        confidence=0.92,
        x=10.0,
        y=15.0,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="DJI_0002",
        bbox=BBox(x=0.48, y=0.48, w=0.04, h=0.04),
    )

    res = deduplicate_weeds([p1, p2], radius_m=0.35)
    assert len(res) == 1
    assert res[0].best_frame_id == "DJI_0002"
    assert res[0].duplicate_count == 2


def test_different_classes_not_merged():
    """Points very close spatially but with different class names should not merge when match_classes=True."""
    p1 = WeedPoint(
        weed_id="w1",
        class_name="weed_dicot",
        confidence=0.9,
        x=2.0,
        y=2.0,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="f1",
    )
    p2 = WeedPoint(
        weed_id="w2",
        class_name="weed_monocot",
        confidence=0.9,
        x=2.05,
        y=2.05,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="f2",
    )

    merged = deduplicate_weeds([p1, p2], radius_m=0.35, match_classes=True)
    assert len(merged) == 2


def test_same_frame_detections_not_merged():
    """Two detections found on the exact same frame are distinct plants and should not merge."""
    p1 = WeedPoint(
        weed_id="w1",
        class_name="weed_dicot",
        confidence=0.9,
        x=2.0,
        y=2.0,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="frame_A",
        bbox=BBox(x=0.2, y=0.2, w=0.05, h=0.05),
    )
    p2 = WeedPoint(
        weed_id="w2",
        class_name="weed_dicot",
        confidence=0.85,
        x=2.1,
        y=2.1,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="frame_A",
        bbox=BBox(x=0.25, y=0.25, w=0.05, h=0.05),
    )

    merged = deduplicate_weeds([p1, p2], radius_m=0.35)
    assert len(merged) == 2
