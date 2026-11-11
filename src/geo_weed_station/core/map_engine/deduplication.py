"""Spatial clustering and deduplication of weed detections across overlapping aerial frames."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial import KDTree

from geo_weed_station.schemas.map_models import WeedPoint


def _nadir_distance(point: WeedPoint) -> float:
    """Calculate normalized distance from camera principal center [0.5, 0.5].

    Returns a value in [0.0, ~0.707]. Lower values indicate higher nadir quality.
    """
    if point.bbox is None:
        return 0.5

    if hasattr(point.bbox, "x") and hasattr(point.bbox, "w"):
        cx = point.bbox.x + point.bbox.w / 2.0
        cy = point.bbox.y + point.bbox.h / 2.0
    elif isinstance(point.bbox, (tuple, list)) and len(point.bbox) >= 4:
        cx = point.bbox[0] + point.bbox[2] / 2.0
        cy = point.bbox[1] + point.bbox[3] / 2.0
    else:
        return 0.5

    return math.hypot(cx - 0.5, cy - 0.5)


def deduplicate_weeds(
    points: Sequence[WeedPoint],
    radius_m: float = 0.35,
    match_classes: bool = True,
) -> list[WeedPoint]:
    """Deduplicate spatially overlapping weed points across consecutive UAV frames.

    When aerial surveys have 60-80% overlap, a physical weed is photographed and detected
    multiple times. This algorithm groups detections within `radius_m` using a KDTree,
    resolves the optimal detection (closest to nadir / optical center with highest confidence),
    and produces a unique, clean point layer with provenance attributes.

    Args:
        points: Raw sequence of WeedPoint instances from all ingested frames.
        radius_m: Spatial tolerance radius in meters (default 0.35m).
        match_classes: Whether points must share the same class name to be merged.

    Returns:
        List of unique, deduplicated WeedPoint instances.
    """
    if not points:
        return []

    if len(points) == 1:
        p = points[0]
        return [
            p.model_copy(
                update={
                    "best_frame_id": p.best_frame_id or p.source_frame,
                    "duplicate_count": p.duplicate_count,
                }
            )
        ]

    coords = np.array([[p.x, p.y] for p in points], dtype=np.float64)
    tree = KDTree(coords)
    neighbor_groups = tree.query_ball_tree(tree, r=max(0.01, float(radius_m)))

    parent = list(range(len(points)))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i, neighbors in enumerate(neighbor_groups):
        pt_i = points[i]
        for j in neighbors:
            if i >= j:
                continue
            pt_j = points[j]
            # Two detections found on the exact same frame are distinct plants and should not merge
            if pt_i.source_frame == pt_j.source_frame:
                continue
            # If match_classes is True, different species must not be merged
            if match_classes and pt_i.class_name != pt_j.class_name:
                continue
            union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(len(points)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    unique_points: list[WeedPoint] = []

    for cluster_indices in clusters.values():
        cluster_points = [points[k] for k in cluster_indices]

        def quality_key(pt: WeedPoint) -> tuple[float, float]:
            return (_nadir_distance(pt), -pt.confidence)

        best_pt = min(cluster_points, key=quality_key)

        total_conf = sum(p.confidence for p in cluster_points)
        if total_conf > 0:
            avg_x = sum(p.x * p.confidence for p in cluster_points) / total_conf
            avg_y = sum(p.y * p.confidence for p in cluster_points) / total_conf
            if best_pt.lat is not None:
                avg_lat = sum(p.lat * p.confidence for p in cluster_points if p.lat is not None) / total_conf
            else:
                avg_lat = None
            if best_pt.lon is not None:
                avg_lon = sum(p.lon * p.confidence for p in cluster_points if p.lon is not None) / total_conf
            else:
                avg_lon = None
        else:
            avg_x, avg_y = best_pt.x, best_pt.y
            avg_lat, avg_lon = best_pt.lat, best_pt.lon

        unique_pt = WeedPoint(
            weed_id=best_pt.weed_id,
            class_name=best_pt.class_name,
            confidence=round(max(p.confidence for p in cluster_points), 4),
            x=round(float(avg_x), 3),
            y=round(float(avg_y), 3),
            coord_system=best_pt.coord_system,
            source_frame=best_pt.source_frame,
            lat=round(float(avg_lat), 8) if avg_lat is not None else None,
            lon=round(float(avg_lon), 8) if avg_lon is not None else None,
            best_frame_id=best_pt.best_frame_id or best_pt.source_frame,
            bbox=best_pt.bbox,
            duplicate_count=len(cluster_points),
            stage_name=best_pt.stage_name,
            class_prob=best_pt.class_prob,
        )
        unique_points.append(unique_pt)

    return unique_points
