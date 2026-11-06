"""Unit tests for spatial projection strategies, nadir projection, and georeference validation."""

from __future__ import annotations

import math
import pytest

from geo_weed_station.core.map_engine.errors import GeoreferenceMissingError
from geo_weed_station.core.map_engine.projector import (
    NadirProjector,
    ProjectionStrategy,
    get_projector,
    register_projector,
    validate_georef,
)
from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
    SensorOptics,
)
from geo_weed_station.schemas.payload import FramePayload


def test_validate_georef_missing_raises_error():
    """Verify validate_georef raises GeoreferenceMissingError when georeferencing is missing or H <= 0."""
    # 1. No pose
    payload_no_pose = FramePayload(
        frame_id="f1",
        image_size_px=(1920, 1080),
        pose=None,
    )
    with pytest.raises(GeoreferenceMissingError, match="lacks valid georeferencing"):
        validate_georef(payload_no_pose)

    # 2. H <= 0
    pose_zero_alt = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=1.0,
        y=2.0,
        altitude_m=0.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    payload_zero_alt = FramePayload(
        frame_id="f2",
        image_size_px=(1920, 1080),
        pose=pose_zero_alt,
    )
    with pytest.raises(GeoreferenceMissingError):
        validate_georef(payload_zero_alt)


def test_ground_footprint_calculation():
    """Verify ground footprint computation via FOV and sensor parameters."""
    # 1. FOV-based: H=2.0m, FOV_h=90 deg -> W = 2*2*tan(45)=4.0m
    optics_fov = SensorOptics(
        focal_length_mm=4.0,
        fov_h_deg=90.0,
        fov_v_deg=90.0,
    )
    w_m, h_m = NadirProjector.ground_footprint_m(optics_fov, altitude_m=2.0, image_size_px=(1000, 1000))
    assert w_m == pytest.approx(4.0, rel=1e-3)
    assert h_m == pytest.approx(4.0, rel=1e-3)

    # 2. Sensor dimensions: H=2.0m, Sw=6.4mm, Sh=4.8mm, f=3.2mm -> W=2.0*(6.4/3.2)=4.0m, H=2.0*(4.8/3.2)=3.0m
    optics_sensor = SensorOptics(
        focal_length_mm=3.2,
        sensor_width_mm=6.4,
        sensor_height_mm=4.8,
    )
    w_m2, h_m2 = NadirProjector.ground_footprint_m(optics_sensor, altitude_m=2.0, image_size_px=(640, 480))
    assert w_m2 == pytest.approx(4.0, rel=1e-3)
    assert h_m2 == pytest.approx(3.0, rel=1e-3)


def test_nadir_projection_center_box_equals_camera_position():
    """Verify a detection at exact image center projects to exact camera coordinates regardless of Yaw."""
    for yaw in [0.0, 45.0, 90.0, 180.0, -90.0]:
        pose = CameraSpatialPose(
            coordinate_system=CoordinateSystem.LOCAL_ARUCO,
            x=10.0,
            y=20.0,
            altitude_m=2.0,
            roll_deg=0.0,
            pitch_deg=-90.0,
            yaw_deg=yaw,
        )
        optics = SensorOptics(
            focal_length_mm=3.2,
            sensor_width_mm=6.4,
            sensor_height_mm=4.8,
            principal_point_px=(320.0, 240.0),
        )
        # Box centered at (0.5, 0.5) -> center pixel = (320, 240)
        center_weed = DetectedWeed(
            weed_id="w_center",
            class_id=0,
            class_name="weed_dicot",
            confidence=0.9,
            bbox=BBox(x=0.45, y=0.45, w=0.1, h=0.1),
        )
        payload = FramePayload(
            frame_id="frame_center",
            image_size_px=(640, 480),
            pose=pose,
            optics=optics,
            detections=[center_weed],
        )

        projector = NadirProjector()
        projected = projector.project(payload)

        assert len(projected) == 1
        res = projected[0]
        assert res.world_x == pytest.approx(10.0, abs=0.005)
        assert res.world_y == pytest.approx(20.0, abs=0.005)

        # Immutability: original object world_x is unchanged
        assert center_weed.world_x is None


def test_nadir_projection_offset_and_yaw_rotation():
    """Verify off-center detection projection matches analytical geometry within 1 cm."""
    # H=2.0m, Sw=6.4mm, Sh=4.8mm, f=3.2mm -> footprint W=4.0m, H=3.0m
    # Camera at (x=0.0, y=0.0)
    optics = SensorOptics(
        focal_length_mm=3.2,
        sensor_width_mm=6.4,
        sensor_height_mm=4.8,
        principal_point_px=(320.0, 240.0),
    )
    # Box center at pixel (480, 240) -> offset dx_px = +160 (right), dy_px = 0
    # In meters: dx_cam = (160 / 640) * 4.0m = +1.0m, dy_cam = 0.0m
    offset_weed = DetectedWeed(
        weed_id="w_offset",
        class_id=0,
        class_name="weed_dicot",
        confidence=0.85,
        bbox=BBox(x=0.7, y=0.45, w=0.1, h=0.1),  # center_x = 0.75 -> 0.75 * 640 = 480
    )

    # 1. Yaw = 0 deg: Right offset in camera (+1.0m) -> World X = +1.0, World Y = 0.0
    pose_yaw0 = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=0.0,
        y=0.0,
        altitude_m=2.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    payload_yaw0 = FramePayload(
        frame_id="f_yaw0",
        image_size_px=(640, 480),
        pose=pose_yaw0,
        optics=optics,
        detections=[offset_weed],
    )
    res_yaw0 = NadirProjector().project(payload_yaw0)[0]
    assert res_yaw0.world_x == pytest.approx(1.0, abs=0.01)
    assert res_yaw0.world_y == pytest.approx(0.0, abs=0.01)

    # 2. Yaw = 90 deg: Facing East. Camera right (+1.0m) points South -> World X = 0.0, World Y = -1.0
    pose_yaw90 = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=0.0,
        y=0.0,
        altitude_m=2.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=90.0,
    )
    payload_yaw90 = FramePayload(
        frame_id="f_yaw90",
        image_size_px=(640, 480),
        pose=pose_yaw90,
        optics=optics,
        detections=[offset_weed],
    )
    res_yaw90 = NadirProjector().project(payload_yaw90)[0]
    assert res_yaw90.world_x == pytest.approx(0.0, abs=0.01)
    assert res_yaw90.world_y == pytest.approx(-1.0, abs=0.01)


def test_nadir_projection_wgs84():
    """Verify projection in WGS84 produces valid geographic coordinates."""
    pose_wgs = CameraSpatialPose(
        coordinate_system=CoordinateSystem.WGS84,
        x=37.618423,  # lon
        y=55.751244,  # lat
        altitude_m=10.0,
        roll_deg=0.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    weed = DetectedWeed(
        weed_id="w_wgs",
        class_id=0,
        class_name="weed_dicot",
        confidence=0.9,
        bbox=BBox(x=0.45, y=0.45, w=0.1, h=0.1),
    )
    payload = FramePayload(
        frame_id="f_wgs",
        image_size_px=(1920, 1080),
        pose=pose_wgs,
        detections=[weed],
    )
    res = NadirProjector().project(payload)[0]
    assert res.world_x == pytest.approx(37.618423, abs=1e-5)
    assert res.world_y == pytest.approx(55.751244, abs=1e-5)


def test_projector_factory_and_extension():
    """Verify projector factory and dynamic strategy registration."""
    p_nadir = get_projector("nadir")
    assert isinstance(p_nadir, NadirProjector)

    # Test unknown strategy raises ValueError
    with pytest.raises(ValueError, match="Unknown projector strategy"):
        get_projector("non_existent_strategy")

    # Extension test: register a custom mock strategy
    class CustomProjector(ProjectionStrategy):
        def project(self, payload: FramePayload) -> list[DetectedWeed]:
            return []

    register_projector("custom_mock", CustomProjector)
    custom_inst = get_projector("custom_mock")
    assert isinstance(custom_inst, CustomProjector)
