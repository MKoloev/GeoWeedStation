"""Prescription zones generation: vectorization of high-density weed clusters into treatment polygons."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import label

from geo_weed_station.core.map_engine.coordinates import CoordinateConverter
from geo_weed_station.schemas.map_models import HeatmapGrid, PrescriptionZone, WeedPoint

logger = logging.getLogger(__name__)


def generate_prescription_zones(
    grid: HeatmapGrid | None,
    points: Sequence[WeedPoint] | None = None,
    threshold: float = 3.0,
    min_area_m2: float = 1.0,
    converter: CoordinateConverter | None = None,
) -> list[PrescriptionZone]:
    """Generate vector prescription polygons from contiguous high-density heatmap clusters.

    Agronomists require prescription maps (VRA / Variable Rate Application) that delineate
    discrete management zones with high weed infestation rather than smooth raster surfaces.
    This algorithm isolates connected components above `threshold` density, vectorizes their
    boundaries into closed metric polygons, and assigns agronomic application recommendations.

    Args:
        grid: Computed HeatmapGrid containing 2D density matrix.
        points: Localized weed points for class attribution and density validation.
        threshold: Minimum cell weed count to qualify as an active weed patch.
        min_area_m2: Minimum zone area filter to discard isolated single-point noise.
        converter: Optional CoordinateConverter to produce WGS84 polygon vertices.

    Returns:
        List of PrescriptionZone instances ordered by severity (highest density first).
    """
    if grid is None or grid.matrix is None or grid.matrix.size == 0:
        return []

    matrix = np.asarray(grid.matrix, dtype=np.float32)
    binary_mask = (matrix >= threshold).astype(np.uint8)
    if not np.any(binary_mask):
        return []

    labeled_array, num_features = label(binary_mask)
    if num_features == 0:
        return []

    cell_m = grid.cell_size_m
    origin_x, origin_y = grid.origin
    cell_area = cell_m * cell_m
    zones: list[PrescriptionZone] = []

    for feat_id in range(1, num_features + 1):
        feat_mask = (labeled_array == feat_id).astype(np.uint8)
        num_cells = int(np.sum(feat_mask))
        zone_area = float(num_cells * cell_area)

        if zone_area < min_area_m2:
            continue

        padded = np.pad(feat_mask, pad_width=1, mode="constant", constant_values=0)
        contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        main_contour = max(contours, key=cv2.contourArea)
        pts = main_contour.reshape(-1, 2) - 1

        polygon_vertices: list[tuple[float, float]] = []
        for col_idx, row_idx in pts:
            wx = origin_x + float(col_idx) * cell_m
            wy = origin_y + float(row_idx) * cell_m
            polygon_vertices.append((round(wx, 3), round(wy, 3)))

        if len(polygon_vertices) < 3:
            rows, cols = np.where(feat_mask)
            min_c, max_c = int(np.min(cols)), int(np.max(cols)) + 1
            min_r, max_r = int(np.min(rows)), int(np.max(rows)) + 1
            p1 = (round(origin_x + min_c * cell_m, 3), round(origin_y + min_r * cell_m, 3))
            p2 = (round(origin_x + max_c * cell_m, 3), round(origin_y + min_r * cell_m, 3))
            p3 = (round(origin_x + max_c * cell_m, 3), round(origin_y + max_r * cell_m, 3))
            p4 = (round(origin_x + min_c * cell_m, 3), round(origin_y + max_r * cell_m, 3))
            polygon_vertices = [p1, p2, p3, p4]

        # Find contained weeds
        contained_weeds: list[WeedPoint] = []
        if points:
            rows, cols = np.where(feat_mask)
            cell_coords_set = set(zip(rows, cols))
            for pt in points:
                c = int((pt.x - origin_x) / cell_m)
                r = int((pt.y - origin_y) / cell_m)
                if (r, c) in cell_coords_set:
                    contained_weeds.append(pt)

        weed_count = len(contained_weeds)
        mean_density = weed_count / max(0.01, zone_area)

        if contained_weeds:
            class_counts = Counter(p.class_name for p in contained_weeds)
            dominant_class = class_counts.most_common(1)[0][0]
        else:
            dominant_class = "Mixed"

        if mean_density >= 5.0:
            rec = "Сплошная обработка (высокая норма гербицида)"
        elif mean_density >= 2.0:
            rec = "Дифференцированное опрыскивание (стандартная норма)"
        else:
            rec = "Локальная обработка (пониженная норма)"

        wgs84_poly: list[tuple[float, float]] | None = None
        if converter is not None and converter.wgs84_origin is not None:
            wgs84_poly = []
            for vx, vy in polygon_vertices:
                try:
                    lat, lon = converter.local_to_wgs84(vx, vy)
                    wgs84_poly.append((round(lon, 8), round(lat, 8)))
                except (ValueError, TypeError, RuntimeError) as exc:
                    logger.debug("Failed to project prescription zone vertex to WGS84: %s", exc)

        zone = PrescriptionZone(
            zone_id=f"zone_{feat_id:03d}",
            polygon=polygon_vertices,
            wgs84_polygon=wgs84_poly,
            area_m2=round(zone_area, 2),
            mean_density_per_m2=round(mean_density, 2),
            weed_count=weed_count,
            dominant_class=dominant_class,
            treatment_recommendation=rec,
        )
        zones.append(zone)

    zones.sort(key=lambda z: (z.weed_count, z.mean_density_per_m2), reverse=True)
    return zones
