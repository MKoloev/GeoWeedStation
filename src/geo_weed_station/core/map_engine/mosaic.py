"""Quick-Stitch mosaic footprint geometry and lightweight thumbnail utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from geo_weed_station.core.map_engine.coordinates import CoordinateConverter
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import FrameFootprint
from geo_weed_station.schemas.payload import FramePayload


def compute_frame_footprint(
    payload: FramePayload,
    converter: CoordinateConverter | None = None,
    thumbnail_path: Path | str | None = None,
) -> FrameFootprint | None:
    """Calculate spatial ground footprint and rotated bounding polygon for an aerial frame.

    Quick-Stitch displays frames placed by GPS/Local coordinates, altitude H, and Yaw angle
    without computationally heavy structure-from-motion photogrammetry.

    Args:
        payload: Valid georeferenced FramePayload.
        converter: CoordinateConverter if coordinates are in WGS84.
        thumbnail_path: Optional path to stored low-resolution thumbnail.

    Returns:
        FrameFootprint instance with ground dimensions and 4 polygon vertices, or None if invalid.
    """
    if not payload.has_georef() or payload.pose is None:
        return None

    pose = payload.pose
    img_w, img_h = payload.image_size_px
    fw, fh = NadirProjector.ground_footprint_m(payload.optics, pose.altitude_m, (img_w, img_h))

    lat: float | None = None
    lon: float | None = None

    if pose.coordinate_system == CoordinateSystem.WGS84:
        if converter is None or converter.wgs84_origin is None:
            return None
        center_x, center_y = converter.wgs84_to_local(lat=pose.y, lon=pose.x)
        lat = pose.y
        lon = pose.x
    else:
        center_x = pose.x
        center_y = pose.y

    half_w = fw / 2.0
    half_h = fh / 2.0

    # 4 unrotated corners relative to center: top-left, top-right, bottom-right, bottom-left
    local_corners = [
        (-half_w, half_h),
        (half_w, half_h),
        (half_w, -half_h),
        (-half_w, -half_h),
    ]

    rotated_corners: list[tuple[float, float]] = []
    for dx, dy in local_corners:
        rx, ry = CoordinateConverter.rotate_by_yaw(dx, dy, pose.yaw_deg)
        rotated_corners.append((round(center_x + rx, 3), round(center_y + ry, 3)))

    thumb = make_thumbnail(payload.image) if payload.image is not None else None

    return FrameFootprint(
        frame_id=payload.frame_id,
        source_path=payload.source_path,
        center_x=round(center_x, 3),
        center_y=round(center_y, 3),
        width_m=round(fw, 3),
        height_m=round(fh, 3),
        altitude_m=round(pose.altitude_m, 2),
        yaw_deg=round(pose.yaw_deg, 2),
        polygon=tuple(rotated_corners),
        thumbnail_path=thumbnail_path,
        lat=round(lat, 8) if lat is not None else None,
        lon=round(lon, 8) if lon is not None else None,
        thumbnail=thumb,
    )


def make_thumbnail(image: np.ndarray, max_dim: int = 384) -> np.ndarray:
    """Generate high-performance low-resolution RGB thumbnail preserving aspect ratio."""
    if image is None or image.size == 0:
        return np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image.copy()

    scale = float(max_dim) / float(max(h, w))
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)
