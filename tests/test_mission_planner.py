"""Unit tests for mission_planner module: serpentine grid, steps, validation, and flight time estimation."""

from __future__ import annotations

import pytest

from geo_weed_station.core.mission_planner import (
    PlannerParams,
    build_mission,
    compute_footprint_m,
    compute_grid_step,
    estimate_flight_time_s,
    generate_waypoints,
    validate_params,
)
from geo_weed_station.schemas.entities import CoordinateSystem, SensorOptics
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy


@pytest.fixture
def default_optics() -> SensorOptics:
    return SensorOptics(
        focal_length_mm=3.6,
        sensor_width_mm=6.4,
        sensor_height_mm=4.8,
    )


def test_compute_footprint_and_grid_step(default_optics: SensorOptics):
    """Verify ground footprint and grid step scaling with altitude and overlap."""
    # H = 2.0m: footprint W = 2.0 * (6.4 / 3.6) = 3.555m, H = 2.0 * (4.8 / 3.6) = 2.666m
    w_fp, h_fp = compute_footprint_m(default_optics, altitude_m=2.0)
    assert w_fp == pytest.approx(3.555, rel=1e-2)
    assert h_fp == pytest.approx(2.666, rel=1e-2)

    # Overlap = 0.2: step should be 80% of footprint
    step_x, step_y = compute_grid_step(default_optics, altitude_m=2.0, overlap=0.2)
    assert step_x == pytest.approx(w_fp * 0.8, rel=1e-2)
    assert step_y == pytest.approx(h_fp * 0.8, rel=1e-2)


def test_serpentine_pattern_order(default_optics: SensorOptics):
    """Verify serpentine (lawnmower) path ordering: even rows left-to-right, odd rows right-to-left."""
    params = PlannerParams(
        corner_tl=(0.0, 3.0),
        corner_br=(3.0, 0.0),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        overlap=0.1,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        step_override_m=1.0,  # 3 rows, 3 columns
    )

    waypoints = generate_waypoints(params)
    assert len(waypoints) == 9

    # Row 0 (top row, y=2.5): X should increase (left-to-right)
    row0 = waypoints[0:3]
    assert row0[0].x < row0[1].x < row0[2].x
    assert row0[0].y == row0[1].y == row0[2].y

    # Row 1 (middle row, y=1.5): X should decrease (right-to-left)
    row1 = waypoints[3:6]
    assert row1[0].x > row1[1].x > row1[2].x
    assert row1[0].y == row1[1].y == row1[2].y

    # Row 2 (bottom row, y=0.5): X should increase (left-to-right)
    row2 = waypoints[6:9]
    assert row2[0].x < row2[1].x < row2[2].x
    assert row2[0].y == row2[1].y == row2[2].y

    # Check Z altitude on all points
    for wp in waypoints:
        assert wp.z == 1.5


def test_inverted_corners_give_same_bounds(default_optics: SensorOptics):
    """Verify planner handles inverted corner inputs identically."""
    p_normal = PlannerParams(
        corner_tl=(0.2, 2.8),
        corner_br=(2.8, 0.2),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        step_override_m=1.0,
    )
    p_inverted = PlannerParams(
        corner_tl=(2.8, 0.2),  # Inverted
        corner_br=(0.2, 2.8),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        step_override_m=1.0,
    )

    wps1 = generate_waypoints(p_normal)
    wps2 = generate_waypoints(p_inverted)

    assert len(wps1) == len(wps2)
    for wp1, wp2 in zip(wps1, wps2):
        assert wp1.x == wp2.x
        assert wp1.y == wp2.y


def test_single_row_and_column(default_optics: SensorOptics):
    """Verify single row or single column when area is smaller than grid step."""
    params_narrow = PlannerParams(
        corner_tl=(1.0, 3.0),
        corner_br=(1.2, 0.0),  # width 0.2 < step 1.0
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        step_override_m=1.0,
    )
    wps = generate_waypoints(params_narrow)
    # 3 points along single column (x = 1.1)
    assert len(wps) == 3
    for wp in wps:
        assert wp.x == pytest.approx(1.1)


def test_validate_params_rules(default_optics: SensorOptics):
    """Verify validation catches out of bounds, degenerate rects, and invalid modes."""
    # 1. Out of local cube (x > 3.0m)
    p_oob = PlannerParams(
        corner_tl=(0.0, 4.0),  # y > 3.0
        corner_br=(3.5, 0.0),  # x > 3.0
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
    )
    errs = validate_params(p_oob, max_extent_m=3.0)
    assert any("exceeds local arena limits" in e for e in errs)

    # 2. Altitude > 3.0 in local arena
    p_alt_oob = PlannerParams(
        corner_tl=(0.0, 2.0),
        corner_br=(2.0, 0.0),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=4.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
    )
    errs_alt = validate_params(p_alt_oob, max_extent_m=3.0)
    assert any("exceeds local arena height limit" in e for e in errs_alt)

    # 3. Degenerate rectangle
    p_degen = PlannerParams(
        corner_tl=(1.0, 1.0),
        corner_br=(1.0, 2.0),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
    )
    errs_degen = validate_params(p_degen)
    assert any("degenerate" in e for e in errs_degen)

    # 4. COLLECTION_ONLY + METADATA_ONLY conflict
    p_conflict = PlannerParams(
        corner_tl=(0.0, 2.0),
        corner_br=(2.0, 0.0),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.COLLECTION_ONLY,
        save_policy=SavePolicy.METADATA_ONLY,
    )
    errs_conflict = validate_params(p_conflict)
    assert any("COLLECTION_ONLY + METADATA_ONLY is invalid" in e for e in errs_conflict)


def test_build_mission_returns_valid_mission_task(default_optics: SensorOptics):
    """Verify build_mission generates a valid MissionTask DTO passing all schema validations."""
    params = PlannerParams(
        corner_tl=(0.5, 2.5),
        corner_br=(2.5, 0.5),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        step_override_m=1.0,
    )

    mission = build_mission(params)
    assert isinstance(mission, MissionTask)
    assert len(mission.waypoints) == 4
    assert mission.mode == MissionMode.INFERENCE_SEQUENTIAL
    assert mission.coordinate_system == CoordinateSystem.LOCAL_ARUCO


def test_estimate_flight_time(default_optics: SensorOptics):
    """Verify estimated flight time accounts for distance and hover dwells."""
    params = PlannerParams(
        corner_tl=(0.0, 2.0),
        corner_br=(2.0, 0.0),
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        altitude_m=1.5,
        optics=default_optics,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        step_override_m=1.0,
    )
    # 4 waypoints forming a 1x1m square: path length is 3m
    wps = generate_waypoints(params)
    assert len(wps) == 4

    # speed = 1.0 m/s -> transit = 3.0 s; hover = 4 * 2.0 = 8.0 s -> total = 11.0 s
    est_time = estimate_flight_time_s(wps, speed_mps=1.0, hover_s=2.0)
    assert est_time == pytest.approx(11.0, abs=0.5)
