"""Unit tests for heatmap computation algorithms (CountHeatmap and KdeHeatmap)."""

from __future__ import annotations

import numpy as np
import pytest

from geo_weed_station.core.map_engine.heatmap import CountHeatmap, KdeHeatmap
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import Bounds, WeedPoint


def make_dummy_point(x: float, y: float, class_name: str = "weed_dicot") -> WeedPoint:
    """Helper to create dummy WeedPoint instance."""
    return WeedPoint(
        weed_id=f"wp_{x}_{y}",
        class_name=class_name,
        confidence=0.9,
        x=x,
        y=y,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="frame_0",
    )


def test_count_heatmap_sum_equals_points_count():
    """Verify that CountHeatmap cell sum strictly equals total number of points."""
    bounds = Bounds(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0)
    points = [
        make_dummy_point(1.2, 1.4),
        make_dummy_point(1.8, 1.9),  # in same 1.0m cell as previous
        make_dummy_point(5.5, 4.2),
        make_dummy_point(9.9, 9.9),  # corner
        make_dummy_point(0.0, 0.0),  # origin
    ]

    # Test cell_size = 1.0m
    grid_1m = CountHeatmap().compute(points, cell_m=1.0, bounds=bounds)
    assert grid_1m.matrix.shape == (10, 10)
    assert np.sum(grid_1m.matrix) == len(points)
    # Cell at col 1, row 1 should contain 2 points
    assert grid_1m.matrix[1, 1] == 2.0

    # Test cell_size = 0.5m
    grid_05m = CountHeatmap().compute(points, cell_m=0.5, bounds=bounds)
    assert grid_05m.matrix.shape == (20, 20)
    assert np.sum(grid_05m.matrix) == len(points)


def test_count_heatmap_empty():
    """Verify CountHeatmap on empty points returns zero matrix without errors."""
    bounds = Bounds(min_x=0.0, min_y=0.0, max_x=4.0, max_y=4.0)
    grid = CountHeatmap().compute([], cell_m=1.0, bounds=bounds)
    assert grid.matrix.shape == (4, 4)
    assert np.sum(grid.matrix) == 0.0


def test_kde_heatmap_smoothing_and_normalization():
    """Verify KdeHeatmap applies Gaussian smoothing and preserves total volume."""
    bounds = Bounds(min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0)
    points = [
        make_dummy_point(5.0, 5.0),
        make_dummy_point(5.2, 5.1),
    ]

    kde_grid = KdeHeatmap(bandwidth_m=1.0).compute(points, cell_m=0.5, bounds=bounds)
    mat = kde_grid.matrix

    # Center cell should have high density, and neighboring cells should have non-zero spread
    col_c, row_c = int(5.0 / 0.5), int(5.0 / 0.5)
    assert mat[row_c, col_c] > 0.0
    assert mat[row_c + 1, col_c] > 0.0  # neighboring cell smoothed
    assert mat[0, 0] < mat[row_c, col_c]  # far corner much lower

    # Preserves volume
    assert np.sum(mat) == pytest.approx(len(points), rel=1e-3)
