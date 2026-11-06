"""Tests for MapController coordinating MapEngine and UI signals."""

from __future__ import annotations

import numpy as np
import pytest

from geo_weed_station.config.settings import MapSettings
from geo_weed_station.controllers.map_controller import MAP_LOCKED_DEFAULT_REASON, MapController
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
    SensorOptics,
)
from geo_weed_station.schemas.map_models import MapStatistics
from geo_weed_station.schemas.payload import FramePayload


@pytest.fixture
def map_engine() -> MapEngine:
    projector = NadirProjector()
    settings = MapSettings(heatmap_cell_m=1.0, critical_density_threshold=2.0)
    return MapEngine(projector=projector, settings=settings)


def _make_georef_payload() -> FramePayload:
    pose = CameraSpatialPose(
        x=10.0,
        y=10.0,
        altitude_m=15.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
    )
    optics = SensorOptics(focal_length_mm=4.5, sensor_width_mm=6.4, sensor_height_mm=4.8)
    det = DetectedWeed(
        weed_id="w-1",
        class_id=0,
        class_name="Ambrosia",
        confidence=0.9,
        bbox=BBox(x=0.5, y=0.5, w=0.1, h=0.1),
    )
    return FramePayload(
        frame_id="frame_001",
        image_size_px=(4000, 3000),
        pose=pose,
        optics=optics,
        detections=[det],
    )


def _make_ungeoref_payload() -> FramePayload:
    return FramePayload(
        frame_id="frame_no_geo",
        image_size_px=(4000, 3000),
        pose=None,
        optics=None,
        detections=[],
    )


def test_map_controller_ingest_georef_frame(map_engine, qtbot):
    ctrl = MapController(engine=map_engine)

    updates = []
    stats_list = []
    locks = []

    ctrl.map_updated.connect(lambda: updates.append(True))
    ctrl.statistics_changed.connect(stats_list.append)
    ctrl.map_locked_changed.connect(lambda locked, reason: locks.append((locked, reason)))

    payload = _make_georef_payload()
    ctrl.on_frame(payload)

    assert len(updates) == 1
    assert len(stats_list) == 1
    assert isinstance(stats_list[0], MapStatistics)
    assert stats_list[0].total_weeds == 1
    assert locks == [(False, "")]
    assert not ctrl.is_locked


def test_map_controller_ingest_ungeoref_frame_locks_map(map_engine, qtbot):
    ctrl = MapController(engine=map_engine)

    locks = []
    ctrl.map_locked_changed.connect(lambda locked, reason: locks.append((locked, reason)))

    payload = _make_ungeoref_payload()
    ctrl.on_frame(payload)

    assert len(locks) == 1
    assert locks[0][0] is True
    assert MAP_LOCKED_DEFAULT_REASON in locks[0][1]


def test_map_controller_set_cell_size_and_reset(map_engine, qtbot):
    ctrl = MapController(engine=map_engine)
    ctrl.on_frame(_make_georef_payload())

    stats_before = map_engine.statistics()
    ctrl.set_cell_size(2.0)
    stats_after = map_engine.statistics()
    assert stats_after.total_weeds == stats_before.total_weeds

    ctrl.reset()
    assert len(ctrl.engine.weed_map.points) == 0
    assert not ctrl.is_locked
