"""Core business logic package."""

from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.exceptions import (
    DetectionError,
    GeoWeedError,
    ImageReadError,
    MapEngineError,
    XmpError,
    XmpMalformedError,
    XmpNotFoundError,
)
from geo_weed_station.core.image_io import (
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    read_image,
)
from geo_weed_station.core.mission_planner import (
    PlannerParams,
    build_mission,
    compute_footprint_m,
    compute_grid_step,
    estimate_flight_time_s,
    generate_waypoints,
    validate_params,
)
from geo_weed_station.core.onnx_detector import (
    LetterboxResult,
    OnnxDetector,
    letterbox,
    load_class_names,
    nms,
)
from geo_weed_station.core.xmp_parser import (
    XmpSpatialData,
    build_frame_payload,
    extract_spatial_data,
    has_xmp,
    parse_xmp,
    read_xmp_packet,
)

__all__ = [
    # exceptions
    "GeoWeedError",
    "XmpError",
    "XmpNotFoundError",
    "XmpMalformedError",
    "ImageReadError",
    "DetectionError",
    "MapEngineError",
    # xmp_parser
    "XmpSpatialData",
    "read_xmp_packet",
    "parse_xmp",
    "extract_spatial_data",
    "has_xmp",
    "build_frame_payload",
    # image_io
    "read_image",
    "MAX_IMAGE_WIDTH",
    "MAX_IMAGE_HEIGHT",
    # detector
    "BaseDetector",
    "LetterboxResult",
    "OnnxDetector",
    "letterbox",
    "nms",
    "load_class_names",
    # mission_planner
    "PlannerParams",
    "compute_footprint_m",
    "compute_grid_step",
    "generate_waypoints",
    "build_mission",
    "validate_params",
    "estimate_flight_time_s",
]
