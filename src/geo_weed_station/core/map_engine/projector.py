"""Spatial projection strategies for mapping weed detections onto field coordinate planes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Dict, List, Optional, Protocol, Tuple, Type

from geo_weed_station.core.map_engine.coordinates import CoordinateConverter
from geo_weed_station.core.map_engine.errors import GeoreferenceMissingError
from geo_weed_station.schemas.entities import CoordinateSystem, DetectedWeed, SensorOptics
from geo_weed_station.schemas.payload import FramePayload


class DistortionModel(Protocol):
    """Extension point protocol for camera optical distortion models."""

    def undistort_point(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Apply inverse lens distortion to pixel coordinates."""
        ...


def validate_georef(payload: FramePayload) -> None:
    """Validate that frame payload has sufficient georeferencing for spatial projection.
    
    Raises:
        GeoreferenceMissingError: If pose is missing, altitude <= 0, or attitude angles/coordinates are missing.
    """
    if not payload.has_georef():
        alt_str = str(payload.pose.altitude_m) if payload.pose else "None"
        raise GeoreferenceMissingError(
            f"Frame '{payload.frame_id}' lacks valid georeferencing tags (altitude={alt_str}, pose={payload.pose}). "
            "Digital mapping mode is blocked."
        )


class ProjectionStrategy(ABC):
    """Abstract Strategy interface for projecting camera pixel coordinates into field planes."""

    @abstractmethod
    def project(self, payload: FramePayload) -> List[DetectedWeed]:
        """Project detected weeds of a frame to world metric or geographic coordinates.
        
        Args:
            payload: FramePayload with valid pose, optics, and detections.
            
        Returns:
            New list of DetectedWeed instances with world_x and world_y populated.
        """


class NadirProjector(ProjectionStrategy):
    """Nadir (downward-looking, pitch ≈ -90°) pinhole perspective projector."""

    def __init__(self, distortion: Optional[DistortionModel] = None) -> None:
        self.distortion = distortion

    @staticmethod
    def ground_footprint_m(
        optics: Optional[SensorOptics],
        altitude_m: float,
        image_size_px: Tuple[int, int],
    ) -> Tuple[float, float]:
        """Calculate the physical ground footprint (width_m, height_m) of a frame at given altitude.
        
        Args:
            optics: Camera optical parameters (focal length, sensor dimensions or FOV).
            altitude_m: Relative flight altitude H > 0.
            image_size_px: (image_width_px, image_height_px).
            
        Returns:
            Tuple[float, float]: (footprint_width_m, footprint_height_m).
        """
        img_w, img_h = image_size_px

        if optics is not None:
            # 1. FOV-based calculation if available
            if optics.fov_h_deg is not None and optics.fov_v_deg is not None:
                w_m = 2.0 * altitude_m * math.tan(math.radians(optics.fov_h_deg) / 2.0)
                h_m = 2.0 * altitude_m * math.tan(math.radians(optics.fov_v_deg) / 2.0)
                return max(0.01, w_m), max(0.01, h_m)

            # 2. Physical sensor size + focal length
            if optics.focal_length_mm > 0:
                f_mm = optics.focal_length_mm
                sw_mm = optics.sensor_width_mm if optics.sensor_width_mm is not None else 6.4
                sh_mm = optics.sensor_height_mm if optics.sensor_height_mm is not None else (sw_mm * img_h / img_w)
                w_m = altitude_m * (sw_mm / f_mm)
                h_m = altitude_m * (sh_mm / f_mm)
                return max(0.01, w_m), max(0.01, h_m)

        # 3. Default UAV camera FOV (approx 80 deg horizontal)
        w_m = 2.0 * altitude_m * math.tan(math.radians(75.0) / 2.0)
        h_m = w_m * (img_h / img_w)
        return max(0.01, w_m), max(0.01, h_m)

    def project(self, payload: FramePayload) -> List[DetectedWeed]:
        """Project detected weeds to world coordinates (meters in local_aruco or lat/lon in wgs84).
        
        Raises:
            GeoreferenceMissingError: If frame georeference validation fails.
        """
        validate_georef(payload)
        pose = payload.pose
        assert pose is not None  # Guaranteed by validate_georef

        footprint_w, footprint_h = self.ground_footprint_m(
            payload.optics, pose.altitude_m, payload.image_size_px
        )

        img_w, img_h = payload.image_size_px
        has_pp = (
            payload.optics is not None and payload.optics.principal_point_px is not None
        )
        if has_pp:
            pp_x, pp_y = payload.optics.principal_point_px  # type: ignore

        projected_weeds: List[DetectedWeed] = []
        wgs84_conv: Optional[CoordinateConverter] = None

        if pose.coordinate_system == CoordinateSystem.WGS84:
            # Create local tangent converter centered at camera GPS position (lat, lon)
            wgs84_conv = CoordinateConverter(wgs84_origin=(pose.y, pose.x))

        for det in payload.detections:
            norm_cx = det.bbox.x + det.bbox.w / 2.0
            norm_cy = det.bbox.y + det.bbox.h / 2.0

            if has_pp:
                cx_px = norm_cx * img_w
                cy_px = norm_cy * img_h
                if self.distortion is not None:
                    cx_px, cy_px = self.distortion.undistort_point(cx_px, cy_px)
                dx_frac = (cx_px - pp_x) / img_w
                dy_frac = (pp_y - cy_px) / img_h
            else:
                dx_frac = norm_cx - 0.5
                dy_frac = 0.5 - norm_cy

            dx_cam = dx_frac * footprint_w
            dy_cam = dy_frac * footprint_h

            # Rotate displacement by camera Yaw angle
            dx_world, dy_world = CoordinateConverter.rotate_by_yaw(dx_cam, dy_cam, pose.yaw_deg)

            if pose.coordinate_system == CoordinateSystem.LOCAL_ARUCO:
                world_x = pose.x + dx_world
                world_y = pose.y + dy_world
            else:
                assert wgs84_conv is not None
                weed_lat, weed_lon = wgs84_conv.local_to_wgs84(dx_world, dy_world)
                world_x = weed_lon
                world_y = weed_lat

            # Immutable update: create a new DetectedWeed instance
            projected_weed = det.model_copy(
                update={
                    "world_x": round(world_x, 6),
                    "world_y": round(world_y, 6),
                }
            )
            projected_weeds.append(projected_weed)

        return projected_weeds


_PROJECTOR_REGISTRY: Dict[str, Type[ProjectionStrategy]] = {
    "nadir": NadirProjector,
}


def register_projector(name: str, strategy_cls: Type[ProjectionStrategy]) -> None:
    """Register a new projection strategy dynamically without modifying existing code."""
    _PROJECTOR_REGISTRY[name.lower()] = strategy_cls


def get_projector(name: str = "nadir", **kwargs) -> ProjectionStrategy:
    """Factory function to instantiate projection strategies by name."""
    cls = _PROJECTOR_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown projector strategy: '{name}'. Available strategies: {list(_PROJECTOR_REGISTRY.keys())}"
        )
    return cls(**kwargs)
