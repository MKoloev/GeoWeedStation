"""Unit tests for CoordinateConverter and yaw rotations."""

from __future__ import annotations

import pytest

from geo_weed_station.core.map_engine.coordinates import CoordinateConverter
from geo_weed_station.core.map_engine.errors import OriginNotSetError


def test_rotate_by_yaw_cardinal_directions():
    """Verify 2D camera-to-world rotation by cardinal yaw angles."""
    # Yaw = 0 deg: Forward stays (0, 1), Right stays (1, 0)
    dx, dy = CoordinateConverter.rotate_by_yaw(0.0, 1.0, 0.0)
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy == pytest.approx(1.0, abs=1e-6)

    dx, dy = CoordinateConverter.rotate_by_yaw(1.0, 0.0, 0.0)
    assert dx == pytest.approx(1.0, abs=1e-6)
    assert dy == pytest.approx(0.0, abs=1e-6)

    # Yaw = 90 deg (facing East):
    # Forward (0, 1) -> East (1, 0)
    dx, dy = CoordinateConverter.rotate_by_yaw(0.0, 1.0, 90.0)
    assert dx == pytest.approx(1.0, abs=1e-6)
    assert dy == pytest.approx(0.0, abs=1e-6)
    # Right (1, 0) -> South (0, -1)
    dx, dy = CoordinateConverter.rotate_by_yaw(1.0, 0.0, 90.0)
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy == pytest.approx(-1.0, abs=1e-6)

    # Yaw = 180 deg (facing South):
    # Forward (0, 1) -> South (0, -1)
    dx, dy = CoordinateConverter.rotate_by_yaw(0.0, 1.0, 180.0)
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy == pytest.approx(-1.0, abs=1e-6)

    # Yaw = -90 deg (facing West):
    # Forward (0, 1) -> West (-1, 0)
    dx, dy = CoordinateConverter.rotate_by_yaw(0.0, 1.0, -90.0)
    assert dx == pytest.approx(-1.0, abs=1e-6)
    assert dy == pytest.approx(0.0, abs=1e-6)


def test_converter_without_origin_raises_error():
    """Verify OriginNotSetError is raised when origin is not set."""
    conv = CoordinateConverter(wgs84_origin=None)

    with pytest.raises(OriginNotSetError, match="wgs84_origin is not set"):
        conv.local_to_wgs84(10.0, 20.0)

    with pytest.raises(OriginNotSetError, match="wgs84_origin is not set"):
        conv.wgs84_to_local(55.75, 37.61)


def test_converter_roundtrip_precision():
    """Verify sub-millimeter (< 1 mm) roundtrip precision for coordinate conversion."""
    origin = (55.751244, 37.618423)
    conv = CoordinateConverter(wgs84_origin=origin)

    # Origin point should convert to exact (0, 0)
    lx, ly = conv.wgs84_to_local(origin[0], origin[1])
    assert lx == pytest.approx(0.0, abs=1e-6)
    assert ly == pytest.approx(0.0, abs=1e-6)

    # Test multiple displacements within field extent (up to 5 km)
    test_points = [
        (0.0, 0.0),
        (1.5, 0.8),
        (-25.4, 60.1),
        (500.0, -350.0),
        (-1200.5, 800.2),
    ]

    for orig_x, orig_y in test_points:
        lat, lon = conv.local_to_wgs84(orig_x, orig_y)
        back_x, back_y = conv.wgs84_to_local(lat, lon)

        # Distance error in meters must be < 0.001 m (1 mm)
        error_m = ((back_x - orig_x) ** 2 + (back_y - orig_y) ** 2) ** 0.5
        assert error_m < 0.001, f"Roundtrip error {error_m} m exceeds 1 mm for ({orig_x}, {orig_y})"
