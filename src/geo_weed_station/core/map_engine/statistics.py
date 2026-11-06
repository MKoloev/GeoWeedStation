"""Agronomic statistics calculation: Convex Hull area, weed density, and critical zone assessment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from geo_weed_station.schemas.map_models import HeatmapGrid, MapStatistics, WeedPoint


def convex_hull_area(points: Sequence[WeedPoint]) -> float:
    """Compute the 2D surveyed field area in square meters using Convex Hull.
    
    Args:
        points: Sequence of WeedPoint instances.
        
    Returns:
        float: Area of convex hull polygon in square meters.
    """
    if len(points) < 3:
        if len(points) == 0:
            return 0.0
        # For 1-2 points return small 1 m2 default buffer
        return 1.0

    coords = np.array([[pt.x, pt.y] for pt in points], dtype=np.float64)

    # Check for collinearity or zero variance
    if np.ptp(coords[:, 0]) < 1e-4 and np.ptp(coords[:, 1]) < 1e-4:
        return 1.0

    try:
        hull = ConvexHull(coords)
        # In SciPy 2D ConvexHull, 'volume' represents the 2D polygon area
        return float(hull.volume)
    except (QhullError, ValueError):
        # Fallback to simple bounding box area with minimum 1.0 m2
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        return max(1.0, float((max_x - min_x) * (max_y - min_y)))


def calculate_statistics(
    points: Sequence[WeedPoint],
    grid: HeatmapGrid | None = None,
    critical_threshold: float = 5.0,
) -> MapStatistics:
    """Calculate aggregated agronomic metrics for the surveyed field area.
    
    Args:
        points: Localized weed points.
        grid: Optional computed HeatmapGrid.
        critical_threshold: Weed count per cell threshold indicating critical infestation.
        
    Returns:
        MapStatistics summary object.
    """
    total_weeds = len(points)
    by_class: dict[str, int] = dict(Counter(pt.class_name for pt in points))

    area_m2 = max(0.01, convex_hull_area(points)) if total_weeds > 0 else 0.0
    mean_density = float(total_weeds / area_m2) if area_m2 > 0.0 else 0.0

    critical_percent = 0.0
    if grid is not None and grid.matrix is not None and grid.matrix.size > 0 and total_weeds > 0:
        critical_cells = int(np.sum(grid.matrix >= critical_threshold))
        total_cells = int(grid.matrix.size)
        critical_percent = float((critical_cells / total_cells) * 100.0)

    return MapStatistics(
        total_weeds=total_weeds,
        by_class=by_class,
        surveyed_area_m2=round(area_m2, 2),
        mean_density_per_m2=round(mean_density, 3),
        critical_area_percent=round(critical_percent, 2),
    )
