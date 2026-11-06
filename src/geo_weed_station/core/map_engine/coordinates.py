"""Coordinate system conversion utilities and 2D rotations."""

from __future__ import annotations

import math

import pyproj

from geo_weed_station.core.map_engine.errors import OriginNotSetError


class CoordinateConverter:
    """Handles projection between local flat metric space (local_aruco) and geographic WGS84."""

    def __init__(self, wgs84_origin: tuple[float, float] | None = None) -> None:
        """Initialize converter with an optional WGS84 reference origin.
        
        Args:
            wgs84_origin: Base reference point tuple (latitude_deg, longitude_deg).
        """
        self.wgs84_origin = wgs84_origin
        self._proj: pyproj.Proj | None = None

        if self.wgs84_origin is not None:
            lat0, lon0 = self.wgs84_origin
            # Azimuthal Equidistant projection provides millimeter accuracy around origin
            self._proj = pyproj.Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84")

    def local_to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        """Convert local flat coordinates (x, y in meters) to global WGS84 (latitude, longitude in degrees).
        
        Raises:
            OriginNotSetError: If wgs84_origin was not set during initialization.
        """
        if self._proj is None or self.wgs84_origin is None:
            raise OriginNotSetError(
                "wgs84_origin is not set. Automatic conversion from local_aruco to wgs84 is forbidden."
            )
        lon, lat = self._proj(x, y, inverse=True)
        return float(lat), float(lon)

    def wgs84_to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert global WGS84 (latitude, longitude in degrees) to local flat coordinates (x, y in meters).
        
        Raises:
            OriginNotSetError: If wgs84_origin was not set during initialization.
        """
        if self._proj is None or self.wgs84_origin is None:
            raise OriginNotSetError(
                "wgs84_origin is not set. Automatic conversion from wgs84 to local_aruco is forbidden."
            )
        x, y = self._proj(lon, lat, inverse=False)
        return float(x), float(y)

    @staticmethod
    def rotate_by_yaw(dx: float, dy: float, yaw_deg: float) -> tuple[float, float]:
        """Rotate camera displacement vector by yaw angle (clockwise from North/Forward).
        
        Args:
            dx: Horizontal camera frame offset (meters, positive to the right).
            dy: Vertical camera frame offset (meters, positive forward).
            yaw_deg: Heading angle in degrees.
            
        Returns:
            Tuple[float, float]: Rotated world offset (dx_world, dy_world).
        """
        rad = math.radians(yaw_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Standard clockwise heading rotation:
        # dx_world = dx * cos(yaw) + dy * sin(yaw)
        # dy_world = -dx * sin(yaw) + dy * cos(yaw)
        dx_world = dx * cos_a + dy * sin_a
        dy_world = -dx * sin_a + dy * cos_a
        return dx_world, dy_world
