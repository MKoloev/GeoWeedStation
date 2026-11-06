"""Unit tests for BaseDetector, OnnxDetector, letterbox, NMS, and image_io."""

from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

from geo_weed_station.config.settings import DetectorSettings
from geo_weed_station.core.exceptions import DetectionError, ImageReadError
from geo_weed_station.core.image_io import MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, read_image
from geo_weed_station.core.onnx_detector import (
    OnnxDetector,
    letterbox,
    load_class_names,
    nms,
)
from tests.helpers.make_dummy_onnx import create_dummy_yolo_onnx


def test_no_pyside_in_core_detector():
    """Verify core detector and image_io modules do not import PySide6."""
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("geo_weed_station.core"):
            module = sys.modules[mod_name]
            assert "PySide6" not in getattr(module, "__dict__", {}), (
                f"Forbidden PySide6 import in {mod_name}"
            )


def test_letterbox_aspect_ratio_and_reverse():
    """Verify letterbox preserves aspect ratio and inverse coordinates match within 1 px."""
    orig_w, orig_h = 1920, 1080
    dummy_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    target_size = (640, 640)
    lb = letterbox(dummy_img, size=target_size)

    assert lb.tensor.shape == (1, 3, 640, 640)
    assert lb.tensor.dtype == np.float32
    assert 0.0 <= lb.tensor.min() <= lb.tensor.max() <= 1.0

    # Scale should be 640 / 1920 = 1/3
    expected_scale = 640.0 / 1920.0
    assert lb.scale == pytest.approx(expected_scale, rel=1e-3)

    # Pad X should be 0, Pad Y should be (640 - 1080 * (640/1920)) / 2 = 140
    pad_x, pad_y = lb.pad
    assert pad_x == pytest.approx(0.0, abs=1.0)
    assert pad_y == pytest.approx(140.0, abs=1.0)

    # Test coordinate round-trip: point on original image (960, 540)
    orig_px, orig_py = 960.0, 540.0
    lb_x = orig_px * lb.scale + pad_x
    lb_y = orig_py * lb.scale + pad_y

    rev_x = (lb_x - pad_x) / lb.scale
    rev_y = (lb_y - pad_y) / lb.scale

    assert abs(rev_x - orig_px) <= 1.0
    assert abs(rev_y - orig_py) <= 1.0


def test_nms_boxes():
    """Verify NMS suppresses overlapping boxes above threshold."""
    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],  # Box 0 (high score)
            [12.0, 12.0, 48.0, 48.0],  # Box 1 (high overlap with Box 0, lower score)
            [100.0, 100.0, 150.0, 150.0],  # Box 2 (distant box, no overlap)
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.75, 0.85], dtype=np.float32)

    keep = nms(boxes, scores, iou_thr=0.45)
    # Box 1 should be suppressed by Box 0; Box 2 must remain
    assert keep == [0, 2]

    # Empty input
    assert nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []


def test_read_image_success(tmp_path: Path):
    """Verify read_image loads valid image in RGB format."""
    img_path = tmp_path / "valid.jpg"
    test_img = np.full((100, 200, 3), (10, 20, 30), dtype=np.uint8)  # BGR
    cv2.imwrite(str(img_path), test_img)

    loaded = read_image(img_path)
    assert loaded.shape == (100, 200, 3)
    # OpenCV imwrite saved BGR (10, 20, 30); read_image converts to RGB (30, 20, 10)
    assert loaded[0, 0, 0] == 30
    assert loaded[0, 0, 2] == 10


def test_read_image_corrupted(tmp_path: Path):
    """Verify corrupted image raises ImageReadError without crash."""
    bad_path = tmp_path / "corrupted.jpg"
    bad_path.write_bytes(b"not an image at all header dummy data")

    with pytest.raises(ImageReadError):
        read_image(bad_path)


def test_read_image_missing(tmp_path: Path):
    """Verify missing file raises ImageReadError."""
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(ImageReadError):
        read_image(missing)


def test_read_image_downscale_large(tmp_path: Path):
    """Verify image exceeding MAX resolution is safely downscaled."""
    large_path = tmp_path / "huge.jpg"
    # Create image with width 9000 > 8000
    huge_img = np.zeros((1000, 9000, 3), dtype=np.uint8)
    cv2.imwrite(str(large_path), huge_img)

    loaded = read_image(large_path)
    assert loaded.shape[1] <= MAX_IMAGE_WIDTH
    assert loaded.shape[0] <= MAX_IMAGE_HEIGHT


def test_onnx_detector_lifecycle_and_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify lazy loading, inference on dummy ONNX, thresholds, and close()."""
    model_path = create_dummy_yolo_onnx(tmp_path / "dummy.onnx")
    classes_path = tmp_path / "classes.json"
    classes_path.write_text('{"0": "weed_dicot", "1": "weed_monocot"}', encoding="utf-8")

    out_arr = np.zeros((1, 6, 8400), dtype=np.float32)
    # Box 1: weed_dicot (class 0)
    out_arr[0, 0, 0] = 320.0
    out_arr[0, 1, 0] = 320.0
    out_arr[0, 2, 0] = 100.0
    out_arr[0, 3, 0] = 80.0
    out_arr[0, 4, 0] = 0.92
    out_arr[0, 5, 0] = 0.05

    # Box 2: weed_monocot (class 1)
    out_arr[0, 0, 1] = 120.0
    out_arr[0, 1, 1] = 180.0
    out_arr[0, 2, 1] = 60.0
    out_arr[0, 3, 1] = 50.0
    out_arr[0, 4, 1] = 0.08
    out_arr[0, 5, 1] = 0.88

    class DummyInputOutput:
        def __init__(self, name: str):
            self.name = name

    class MockSession:
        def __init__(self, *args, **kwargs):
            self._inputs = [DummyInputOutput("images")]
            self._outputs = [DummyInputOutput("output0")]

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def run(self, output_names, input_feed):
            assert "images" in input_feed
            return [out_arr]

    import onnxruntime as ort
    monkeypatch.setattr(ort, "InferenceSession", MockSession)

    settings = DetectorSettings(
        model_path=model_path,
        classes_path=classes_path,
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
        input_size=(640, 640),
        video_frame_skip=3,
    )

    detector = OnnxDetector(settings)

    # 1. Lazy loading check
    assert detector.is_loaded is False

    # 2. Providers check
    avail_providers = detector.available_providers()
    assert "CPUExecutionProvider" in avail_providers

    # 3. Inference on RGB image (640x640)
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_img)

    # Session should now be loaded
    assert detector.is_loaded is True
    assert len(detections) == 2

    # Verify detection classes and BBoxes
    d0 = detections[0]
    assert d0.class_name == "weed_dicot"
    assert d0.confidence == pytest.approx(0.92, abs=0.01)
    assert 0.0 <= d0.bbox.x <= 1.0
    assert 0.0 <= d0.bbox.y <= 1.0
    assert 0.0 < d0.bbox.w <= 1.0

    d1 = detections[1]
    assert d1.class_name == "weed_monocot"
    assert d1.confidence == pytest.approx(0.88, abs=0.01)

    # 4. Threshold tuning
    detector.set_thresholds(confidence=0.90, iou=0.45)
    filtered_detections = detector.detect(test_img)
    # Only d0 (0.92) exceeds 0.90
    assert len(filtered_detections) == 1
    assert filtered_detections[0].class_name == "weed_dicot"

    # 5. Close and reload
    detector.close()
    assert detector.is_loaded is False

    # Second load should work
    detector.load()
    assert detector.is_loaded is True
    detector.close()


def test_onnx_detector_missing_model(tmp_path: Path):
    """Verify DetectionError when loading non-existent ONNX model."""
    settings = DetectorSettings(
        model_path=tmp_path / "missing.onnx",
        classes_path=tmp_path / "classes.json",
    )
    detector = OnnxDetector(settings)
    with pytest.raises(DetectionError):
        detector.load()
