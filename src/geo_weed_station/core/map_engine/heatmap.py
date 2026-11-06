"""Heatmap generation algorithms: Direct Bin Counting and Gaussian KDE."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Sequence
import numpy as np
from scipy.ndimage import gaussian_filter

from geo_weed_station.schemas.map_models import Bounds, HeatmapGrid, WeedPoint


class HeatmapAlgorithm(ABC):
    """Abstract Strategy interface for raster heatmap computation from spatial point layers."""

    @abstractmethod
    def compute(
        self,
        points: Sequence[WeedPoint],
        cell_m: float,
        bounds: Bounds,
    ) -> HeatmapGrid:
        """Compute 2D raster grid of weed density across bounding extent.
        
        Args:
            points: Sequence of localized weed points.
            cell_m: Regular grid cell resolution in meters (e.g. 0.5 or 1.0).
            bounds: Spatial bounding box of the survey extent.
            
        Returns:
            HeatmapGrid containing 2D numpy matrix and spatial origin.
        """


class CountHeatmap(HeatmapAlgorithm):
    """Direct discrete histogram binning where each cell stores exact count of weed points."""

    def compute(
        self,
        points: Sequence[WeedPoint],
        cell_m: float,
        bounds: Bounds,
    ) -> HeatmapGrid:
        cols = max(1, int(math.ceil(bounds.width / cell_m)))
        rows = max(1, int(math.ceil(bounds.height / cell_m)))

        matrix = np.zeros((rows, cols), dtype=np.float32)

        for pt in points:
            # Determine cell column and row
            col = int((pt.x - bounds.min_x) / cell_m)
            row = int((pt.y - bounds.min_y) / cell_m)

            # Clamp index within valid array bounds
            col = min(cols - 1, max(0, col))
            row = min(rows - 1, max(0, row))
            matrix[row, col] += 1.0

        return HeatmapGrid(
            origin=(bounds.min_x, bounds.min_y),
            cell_size_m=cell_m,
            bounds=bounds,
            matrix=matrix,
        )


class KdeHeatmap(HeatmapAlgorithm):
    """Kernel Density Estimation (KDE) applying smooth 2D Gaussian kernels to weed points."""

    def __init__(self, bandwidth_m: float = 1.0) -> None:
        """Initialize KDE algorithm.
        
        Args:
            bandwidth_m: Spatial smoothing bandwidth in meters.
        """
        self.bandwidth_m = max(0.1, float(bandwidth_m))

    def compute(
        self,
        points: Sequence[WeedPoint],
        cell_m: float,
        bounds: Bounds,
    ) -> HeatmapGrid:
        # First compute base discrete bin counts
        base_grid = CountHeatmap().compute(points, cell_m, bounds)
        matrix = base_grid.matrix

        if len(points) == 0:
            return base_grid

        # Convert bandwidth from meters to grid cell units for Gaussian filter
        sigma_cells = max(0.5, self.bandwidth_m / cell_m)
        smoothed = gaussian_filter(matrix, sigma=sigma_cells, mode="constant", cval=0.0)

        # Normalize total volume so sum of smoothed values preserves total weed count
        smooth_sum = np.sum(smoothed)
        if smooth_sum > 0.0:
            smoothed = smoothed * (float(len(points)) / smooth_sum)

        return HeatmapGrid(
            origin=base_grid.origin,
            cell_size_m=cell_m,
            bounds=bounds,
            matrix=smoothed,
        )
