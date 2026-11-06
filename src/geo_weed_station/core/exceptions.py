"""Core domain exceptions for GeoWeedStation."""

from __future__ import annotations


class GeoWeedError(Exception):
    """Base exception for all GeoWeedStation errors."""


class XmpError(GeoWeedError):
    """Base error for XMP metadata extraction and parsing."""


class XmpNotFoundError(XmpError):
    """Raised when no XMP packet is found in an image file."""


class XmpMalformedError(XmpError):
    """Raised when XMP packet XML is corrupted or contains invalid tag values."""


class ImageReadError(GeoWeedError):
    """Raised when reading or decoding an image file fails."""


class DetectionError(GeoWeedError):
    """Base error for neural network detector."""


class MapEngineError(GeoWeedError):
    """Base error for map projection and coordinate transformations."""
