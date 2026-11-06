"""MapEngine specific exceptions."""

from __future__ import annotations

from geo_weed_station.core.exceptions import MapEngineError


class GeoreferenceMissingError(MapEngineError):
    """Raised when spatial projection is attempted on a frame without valid georeferencing."""


class OriginNotSetError(MapEngineError):
    """Raised when converting between local metric and WGS84 coordinates without a reference origin point."""
