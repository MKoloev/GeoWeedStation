"""Comprehensive unit tests for GeoWeedStation schemas, DTOs, validations and converters."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest
from pydantic import ValidationError

from geo_weed_station.schemas import (
    BBox,
    CameraSpatialPose,
    ConnectionState,
    CoordinateSystem,
    DeleteRunResponse,
    DetectedWeed,
    DronePose,
    DroneRun,
    DroneStatus,
    FlightState,
    FramePayload,
    MissionMode,
    MissionTask,
    SavePolicy,
    SensorOptics,
    StartRequest,
    StartResponse,
    StopAction,
    StopRequest,
    StopResponse,
    SummaryDetection,
    SummaryDetections,
    SummaryItem,
    Waypoint,
    mission_to_start_request,
    summary_to_frame_payloads,
)

SAMPLES_DIR = Path(__file__).parent / "data" / "api_samples"


def test_no_forbidden_imports_in_schemas():
    """Verify schemas do not import PySide6, cv2, or onnxruntime."""
    forbidden = ["PySide6", "cv2", "onnxruntime"]
    for mod_name, module in list(sys.modules.items()):
        if mod_name.startswith("geo_weed_station.schemas"):
            for forb in forbidden:
                assert forb not in sys.modules or forb not in getattr(module, "__dict__", {}), (
                    f"Forbidden module '{forb}' imported in schemas ({mod_name})"
                )


def test_bbox_normalized_and_to_xyxy():
    """Verify BBox normalization validation and coordinate conversion."""
    bbox = BBox(x=0.1, y=0.2, w=0.3, h=0.4)
    assert bbox.to_xyxy() == pytest.approx((0.1, 0.2, 0.4, 0.6))

    # Invalid normalized coordinates
    with pytest.raises(ValidationError):
        BBox(x=-0.1, y=0.0, w=0.5, h=0.5)

    with pytest.raises(ValidationError):
        BBox(x=0.0, y=1.2, w=0.5, h=0.5)

    # Immutability check
    with pytest.raises(ValidationError):
        bbox.x = 0.5  # type: ignore


def test_detected_weed_validation():
    """Verify DetectedWeed confidence bounds and fields."""
    weed = DetectedWeed(
        weed_id="w1",
        class_id=0,
        class_name="weed_dicot",
        confidence=0.85,
        bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
        world_x=1.2,
        world_y=3.4,
    )
    assert weed.confidence == 0.85

    # Confidence outside [0, 1]
    with pytest.raises(ValidationError):
        DetectedWeed(
            weed_id="w2",
            class_id=0,
            class_name="weed_dicot",
            confidence=1.5,
            bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
        )

    # Forbid extra fields
    with pytest.raises(ValidationError):
        DetectedWeed(
            weed_id="w3",
            class_id=0,
            class_name="weed_dicot",
            confidence=0.5,
            bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
            unexpected_field="bad",  # type: ignore
        )


def test_frame_payload_has_georef():
    """Verify has_georef conditions for FramePayload."""
    # Without pose
    payload_no_pose = FramePayload(
        frame_id="f1",
        image_size_px=(1920, 1080),
    )
    assert payload_no_pose.has_georef() is False

    # Pose with altitude <= 0
    pose_zero_alt = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=1.0,
        y=2.0,
        altitude_m=0.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
    )
    payload_zero_alt = FramePayload(
        frame_id="f2",
        image_size_px=(1920, 1080),
        pose=pose_zero_alt,
    )
    assert payload_zero_alt.has_georef() is False

    # Valid georeferenced pose
    valid_pose = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=1.5,
        y=2.5,
        altitude_m=2.0,
        roll_deg=1.0,
        pitch_deg=-2.0,
        yaw_deg=45.0,
    )
    valid_payload = FramePayload(
        frame_id="f3",
        image_size_px=(1920, 1080),
        pose=valid_pose,
    )
    assert valid_payload.has_georef() is True


def test_mission_task_validation():
    """Verify MissionTask business validation rules."""
    wp = Waypoint(x=1.0, y=2.0, z=1.5, yaw=0.0)

    # Empty waypoints rejected
    with pytest.raises(ValidationError, match="waypoints list cannot be empty"):
        MissionTask(
            coordinate_system=CoordinateSystem.LOCAL_ARUCO,
            mode=MissionMode.INFERENCE_SEQUENTIAL,
            save_policy=SavePolicy.ALL,
            waypoints=[],
        )

    # COLLECTION_ONLY + METADATA_ONLY combination rejected
    with pytest.raises(ValidationError, match="COLLECTION_ONLY \\+ METADATA_ONLY is invalid"):
        MissionTask(
            coordinate_system=CoordinateSystem.LOCAL_ARUCO,
            mode=MissionMode.COLLECTION_ONLY,
            save_policy=SavePolicy.METADATA_ONLY,
            waypoints=[wp],
        )

    # Valid task
    task = MissionTask(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        waypoints=[wp],
    )
    assert len(task.waypoints) == 1


def test_parse_api_samples():
    """Verify parsing of real JSON samples from onboard REST specification."""
    # 1. status_response.json
    status_raw = json.loads((SAMPLES_DIR / "status_response.json").read_text(encoding="utf-8"))
    status = DroneStatus.model_validate(status_raw)
    assert status.flight_state == FlightState.FLYING_TO_WP
    assert status.current_wp_index == 2
    assert status.total_waypoints == 3
    assert status.current_pose is not None
    assert status.current_pose.coordinate_system == "aruco_local"

    # 2. start_request.json
    start_req_raw = json.loads((SAMPLES_DIR / "start_request.json").read_text(encoding="utf-8"))
    start_req = StartRequest.model_validate(start_req_raw)
    assert start_req.coordinate_system == "aruco_local"
    assert start_req.use_nn is True
    assert len(start_req.waypoints) == 3

    # 3. start_response.json
    start_resp_raw = json.loads((SAMPLES_DIR / "start_response.json").read_text(encoding="utf-8"))
    start_resp = StartResponse.model_validate(start_resp_raw)
    assert start_resp.status == "started"
    assert start_resp.total_waypoints == 3

    # 4. runs_response.json
    runs_raw = json.loads((SAMPLES_DIR / "runs_response.json").read_text(encoding="utf-8"))
    runs = [DroneRun.model_validate(r) for r in runs_raw]
    assert len(runs) == 1
    assert runs[0].status == "completed"

    # 5. summary_response.json
    summary_raw = json.loads((SAMPLES_DIR / "summary_response.json").read_text(encoding="utf-8"))
    summary = SummaryDetections.model_validate(summary_raw)
    assert summary.detections_count == 4
    assert len(summary.items) == 1
    assert summary.items[0].detections[0].class_name == "weed_dicot"

    # 6. stop_response.json
    stop_resp_raw = json.loads((SAMPLES_DIR / "stop_response.json").read_text(encoding="utf-8"))
    stop_resp = StopResponse.model_validate(stop_resp_raw)
    assert stop_resp.action_initiated == "land"

    # 7. delete_run_response.json
    del_resp_raw = json.loads((SAMPLES_DIR / "delete_run_response.json").read_text(encoding="utf-8"))
    del_resp = DeleteRunResponse.model_validate(del_resp_raw)
    assert del_resp.status == "success"
    assert del_resp.freed_bytes == 48234496


def test_mission_to_start_request_mapping():
    """Verify mapping of coordinate systems and mode to onboard StartRequest."""
    wp = Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0)

    # 1. local_aruco -> aruco_local, INFERENCE_SEQUENTIAL -> use_nn=True
    task_local = MissionTask(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        mode=MissionMode.INFERENCE_SEQUENTIAL,
        save_policy=SavePolicy.ALL,
        waypoints=[wp],
    )
    req1 = mission_to_start_request(task_local)
    assert req1.coordinate_system == "aruco_local"
    assert req1.use_nn is True
    assert req1.save_policy == "all"
    assert len(req1.waypoints) == 1

    # 2. wgs84 -> wgs84, COLLECTION_ONLY -> use_nn=False
    task_wgs = MissionTask(
        coordinate_system=CoordinateSystem.WGS84,
        mode=MissionMode.COLLECTION_ONLY,
        save_policy=SavePolicy.IMAGES_ONLY,
        waypoints=[wp],
    )
    req2 = mission_to_start_request(task_wgs)
    assert req2.coordinate_system == "wgs84"
    assert req2.use_nn is False
    assert req2.save_policy == "images_only"


def test_summary_to_frame_payloads_conversion():
    """Verify converting SummaryDetections into list of FramePayload instances."""
    summary_raw = json.loads((SAMPLES_DIR / "summary_response.json").read_text(encoding="utf-8"))
    summary = SummaryDetections.model_validate(summary_raw)

    payloads = summary_to_frame_payloads(summary, default_frame_size_px=(640, 640))
    assert len(payloads) == 1

    payload = payloads[0]
    assert payload.has_georef() is True
    assert payload.pose is not None
    assert payload.pose.coordinate_system == CoordinateSystem.LOCAL_ARUCO
    assert payload.pose.x == 1.20
    assert payload.pose.y == 0.00
    assert payload.pose.altitude_m == 1.50

    assert len(payload.detections) == 1
    det = payload.detections[0]
    assert det.class_name == "weed_dicot"
    assert det.confidence == 0.89
    assert det.world_x == 1.34
    assert det.world_y == 0.12

    # Check normalized BBox
    assert 0.0 <= det.bbox.x <= 1.0
    assert 0.0 <= det.bbox.y <= 1.0
    assert 0.0 < det.bbox.w <= 1.0
    assert 0.0 < det.bbox.h <= 1.0
