"""Mission planner: grid step calculation, serpentine waypoints generation, and validation."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from geo_weed_station.schemas.entities import CoordinateSystem, SensorOptics
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy, Waypoint


class PlannerParams(BaseModel):
    """Parameters for mission survey grid generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corner_tl: Tuple[float, float]  # Top-left or first corner (x1, y1)
    corner_br: Tuple[float, float]  # Bottom-right or second corner (x2, y2)
    coordinate_system: CoordinateSystem
    altitude_m: float  # Survey altitude Z above ground
    optics: SensorOptics
    overlap: float = Field(default=0.1, ge=0.0, le=0.9)  # Forward/side overlap fraction
    mode: MissionMode
    save_policy: SavePolicy
    step_override_m: Optional[float] = None
    start_yaw_rad: float = 0.0


def compute_footprint_m(optics: SensorOptics, altitude_m: float) -> Tuple[float, float]:
    """Compute physical ground coverage footprint (width_m, height_m) for given camera optics and altitude."""
    alt = max(0.01, altitude_m)

    # 1. FOV-based computation
    if optics.fov_h_deg is not None and optics.fov_v_deg is not None:
        w_m = 2.0 * alt * math.tan(math.radians(optics.fov_h_deg) / 2.0)
        h_m = 2.0 * alt * math.tan(math.radians(optics.fov_v_deg) / 2.0)
        return max(0.1, w_m), max(0.1, h_m)

    # 2. Sensor dimensions & focal length
    if optics.focal_length_mm > 0:
        f = optics.focal_length_mm
        sw = optics.sensor_width_mm if optics.sensor_width_mm is not None else 6.4
        sh = optics.sensor_height_mm if optics.sensor_height_mm is not None else 4.8
        w_m = alt * (sw / f)
        h_m = alt * (sh / f)
        return max(0.1, w_m), max(0.1, h_m)

    # 3. Default UAV camera FOV
    w_m = 2.0 * alt * math.tan(math.radians(75.0) / 2.0)
    h_m = w_m * 0.75
    return max(0.1, w_m), max(0.1, h_m)


def compute_grid_step(optics: SensorOptics, altitude_m: float, overlap: float) -> Tuple[float, float]:
    """Calculate distance step between flight lines (step_x, step_y) accounting for overlap."""
    footprint_w, footprint_h = compute_footprint_m(optics, altitude_m)
    overlap_clamped = max(0.0, min(0.9, overlap))
    step_x = footprint_w * (1.0 - overlap_clamped)
    step_y = footprint_h * (1.0 - overlap_clamped)
    return max(0.1, step_x), max(0.1, step_y)


def validate_params(params: PlannerParams, max_extent_m: float = 3.0) -> List[str]:
    """Validate mission planning parameters. Returns a list of error descriptions (empty if valid)."""
    errors: List[str] = []

    if params.altitude_m <= 0.0:
        errors.append(f"Altitude must be > 0, got {params.altitude_m}")

    min_x = min(params.corner_tl[0], params.corner_br[0])
    max_x = max(params.corner_tl[0], params.corner_br[0])
    min_y = min(params.corner_tl[1], params.corner_br[1])
    max_y = max(params.corner_tl[1], params.corner_br[1])

    if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
        errors.append("Survey rectangle is degenerate (zero width or height)")

    if params.mode == MissionMode.COLLECTION_ONLY and params.save_policy == SavePolicy.METADATA_ONLY:
        errors.append("Combination COLLECTION_ONLY + METADATA_ONLY is invalid")

    if params.coordinate_system == CoordinateSystem.LOCAL_ARUCO:
        # Check flight arena cube bounds [0, max_extent_m]
        if min_x < -1e-4 or max_x > max_extent_m + 1e-4 or min_y < -1e-4 or max_y > max_extent_m + 1e-4:
            errors.append(
                f"Flight area [{min_x:.2f}, {max_x:.2f}] x [{min_y:.2f}, {max_y:.2f}] "
                f"exceeds local arena limits [0, {max_extent_m}] m"
            )
        if params.altitude_m > max_extent_m + 1e-4:
            errors.append(
                f"Altitude {params.altitude_m:.2f} m exceeds local arena height limit {max_extent_m} m"
            )

    return errors


def generate_waypoints(params: PlannerParams) -> List[Waypoint]:
    """Generate ordered serpentine (lawnmower) flight path waypoints.
    
    Even rows are traversed left-to-right, odd rows right-to-left.
    """
    min_x = min(params.corner_tl[0], params.corner_br[0])
    max_x = max(params.corner_tl[0], params.corner_br[0])
    min_y = min(params.corner_tl[1], params.corner_br[1])
    max_y = max(params.corner_tl[1], params.corner_br[1])

    rect_w = max_x - min_x
    rect_h = max_y - min_y

    if params.step_override_m is not None and params.step_override_m > 0:
        step_x = step_y = float(params.step_override_m)
    else:
        step_x, step_y = compute_grid_step(params.optics, params.altitude_m, params.overlap)

    # Determine discrete grid lines
    if rect_w < step_x:
        x_coords = [round((min_x + max_x) / 2.0, 4)]
    else:
        num_cols = max(1, int(round(rect_w / step_x)))
        if num_cols == 1:
            x_coords = [round((min_x + max_x) / 2.0, 4)]
        else:
            offset_x = (rect_w - (num_cols - 1) * step_x) / 2.0
            x_coords = [round(min_x + offset_x + i * step_x, 4) for i in range(num_cols)]

    if rect_h < step_y:
        y_coords = [round((min_y + max_y) / 2.0, 4)]
    else:
        num_rows = max(1, int(round(rect_h / step_y)))
        if num_rows == 1:
            y_coords = [round((min_y + max_y) / 2.0, 4)]
        else:
            offset_y = (rect_h - (num_rows - 1) * step_y) / 2.0
            # Traverse from top (max_y) down to bottom (min_y)
            y_coords = [round(max_y - offset_y - i * step_y, 4) for i in range(num_rows)]

    waypoints: List[Waypoint] = []
    yaw = params.start_yaw_rad

    for row_idx, y in enumerate(y_coords):
        # Even rows: left to right; Odd rows: right to left
        row_xs = list(x_coords) if row_idx % 2 == 0 else list(reversed(x_coords))
        for x in row_xs:
            wp = Waypoint(
                x=x,
                y=y,
                z=round(params.altitude_m, 4),
                yaw=round(yaw, 4),
            )
            waypoints.append(wp)

    return waypoints


def build_mission(params: PlannerParams) -> MissionTask:
    """Validate parameters, generate serpentine waypoints, and return valid MissionTask DTO.
    
    Raises:
        ValueError: If validation fails.
    """
    errors = validate_params(params)
    if errors:
        raise ValueError(f"Invalid mission parameters: {'; '.join(errors)}")

    waypoints = generate_waypoints(params)
    return MissionTask(
        coordinate_system=params.coordinate_system,
        mode=params.mode,
        save_policy=params.save_policy,
        waypoints=waypoints,
    )


def estimate_flight_time_s(
    waypoints: Sequence[Waypoint],
    speed_mps: float = 0.5,
    hover_s: float = 2.0,
) -> float:
    """Estimate total mission execution time including transit and waypoint hover dwell time.
    
    Args:
        waypoints: Ordered sequence of mission waypoints.
        speed_mps: Nominal drone ground speed in meters per second (default 0.5 m/s).
        hover_s: Dwell time in seconds at each waypoint for stabilization and capture/inference.
        
    Returns:
        float: Estimated duration in seconds.
    """
    if not waypoints:
        return 0.0

    total_dist_m = 0.0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)
        total_dist_m += dist

    flight_time = total_dist_m / max(0.1, speed_mps)
    hover_time = len(waypoints) * hover_s
    return round(flight_time + hover_time, 1)


# Alias for serpentine grid generation
generate_serpentine_grid = generate_waypoints
