"""Unit tests for BaseDetector, OnnxDetector, letterbox, NMS, and image_io."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from geo_weed_station.config.settings import DetectorSettings
from geo_weed_station.core.exceptions import DetectionError, ImageReadError
from geo_weed_station.core.image_io import MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, read_image
from geo_weed_station.core.onnx_detector import (
    OnnxDetector,
    letterbox,
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


def test_resolve_classes_path_and_scan(tmp_path: Path):
    """Verify scanning models and auto-resolving corresponding classes.json."""
    from geo_weed_station.core.onnx_detector import resolve_classes_path, scan_available_models

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    m1 = models_dir / "alpha.onnx"
    m2 = models_dir / "beta.onnx"
    m1.touch()
    m2.touch()
    (models_dir / "not_model.txt").touch()

    scanned = scan_available_models(models_dir)
    assert [p.name for p in scanned] == ["alpha.onnx", "beta.onnx"]

    # Candidate 1: alpha.json next to alpha.onnx
    c_alpha = models_dir / "alpha.json"
    c_alpha.write_text('{"0": "Corn"}')
    assert resolve_classes_path(m1) == c_alpha

    # Candidate 2: beta_classes.json next to beta.onnx
    c_beta = models_dir / "beta_classes.json"
    c_beta.write_text('{"0": "Sunflower"}')
    assert resolve_classes_path(m2) == c_beta

    # Fallback to classes.json
    m3 = models_dir / "gamma.onnx"
    m3.touch()
    c_global = models_dir / "classes.json"
    c_global.write_text('{"0": "Global"}')
    assert resolve_classes_path(m3) == c_global


def test_load_model_classes_and_type(tmp_path: Path):
    """Verify parsing classes and model_type across flat and structured JSON formats."""
    from geo_weed_station.core.onnx_detector import ModelType, load_model_classes_and_type

    # 1. Flat JSON with explicit model_type
    f1 = tmp_path / "model1.json"
    f1.write_text('{"model_type": "yolox", "0": "plant"}', encoding="utf-8")
    cls1, type1 = load_model_classes_and_type(f1)
    assert cls1 == {0: "plant"}
    assert type1 == ModelType.YOLOX.value

    # 2. Structured JSON with nested classes and architecture
    f2 = tmp_path / "model2.json"
    f2.write_text('{"architecture": "yolo", "classes": {"0": "weed_a", "1": "weed_b"}}', encoding="utf-8")
    cls2, type2 = load_model_classes_and_type(f2)
    assert cls2 == {0: "weed_a", 1: "weed_b"}
    assert type2 == ModelType.YOLO.value

    # 3. Simple JSON without explicit model_type
    f3 = tmp_path / "model3.json"
    f3.write_text('{"0": "plant"}', encoding="utf-8")
    cls3, type3 = load_model_classes_and_type(f3)
    assert cls3 == {0: "plant"}
    assert type3 is None


def test_onnx_detector_yolox_inference_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify YOLOX multi-stride decoding, auto-detection, and NMS on mock session."""
    import onnxruntime as ort
    from geo_weed_station.core.onnx_detector import ModelType

    dummy_onnx = tmp_path / "dummy_yolox.onnx"
    dummy_onnx.touch()
    classes_json = tmp_path / "dummy_yolox.json"
    classes_json.write_text('{"0": "plant"}', encoding="utf-8")

    # Stride 8: (1, 6, 80, 80), Stride 16: (1, 6, 40, 40), Stride 32: (1, 6, 20, 20)
    out_stride8 = np.full((1, 6, 80, 80), -10.0, dtype=np.float32)
    out_stride16 = np.full((1, 6, 40, 40), -10.0, dtype=np.float32)
    out_stride32 = np.full((1, 6, 20, 20), -10.0, dtype=np.float32)

    # Place one detection in stride 8 at center (y=40, x=40)
    # Target in letterbox: grid_x=40 * 8 = 320, grid_y=40 * 8 = 320
    # exp(0) * 8 = 8 -> w=8, h=8
    out_stride8[0, 0, 40, 40] = 0.0   # reg dx
    out_stride8[0, 1, 40, 40] = 0.0   # reg dy
    out_stride8[0, 2, 40, 40] = 0.0   # reg dw -> exp(0) * 8 = 8
    out_stride8[0, 3, 40, 40] = 0.0   # reg dh -> exp(0) * 8 = 8
    out_stride8[0, 4, 40, 40] = 5.0   # obj_logits -> sigmoid(5.0) ~ 0.993
    out_stride8[0, 5, 40, 40] = 5.0   # cls_logits -> sigmoid(5.0) ~ 0.993

    class DummyInputOutput:
        def __init__(self, name: str, shape: list[int]):
            self.name = name
            self.shape = shape

    class MockYoloxSession:
        def __init__(self, *args, **kwargs):
            self._inputs = [DummyInputOutput("images", [1, 3, 640, 640])]
            self._outputs = [
                DummyInputOutput("out_stride8", [1, 6, 80, 80]),
                DummyInputOutput("out_stride16", [1, 6, 40, 40]),
                DummyInputOutput("out_stride32", [1, 6, 20, 20]),
            ]

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def get_modelmeta(self):
            class Meta:
                custom_metadata_map = {}
            return Meta()

        def run(self, output_names, input_feed):
            assert "images" in input_feed
            return [out_stride8, out_stride16, out_stride32]

    monkeypatch.setattr(ort, "InferenceSession", MockYoloxSession)

    settings = DetectorSettings(
        model_path=dummy_onnx,
        classes_path=classes_json,
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
        input_size=(640, 640),
    )

    detector = OnnxDetector(settings)
    detector.load()

    # Verify auto-detection picked YOLOX
    assert detector.model_type == ModelType.YOLOX.value

    # Run detect on 640x640 image
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = detector.detect(test_img)

    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "plant"
    assert d.class_id == 0
    assert d.confidence > 0.95
    # Center was at 320, 320 -> normalized 0.5, 0.5
    assert pytest.approx(d.bbox.x + d.bbox.w / 2.0, abs=0.02) == 0.5
    assert pytest.approx(d.bbox.y + d.bbox.h / 2.0, abs=0.02) == 0.5


def test_nms_class_aware_and_boxes_helpers():
    """Verify class-aware NMS preserves overlapping boxes of different classes."""
    from geo_weed_station.core.onnx_detector import nms, nms_boxes, nms_detected_weeds
    from geo_weed_station.schemas.entities import BBox, DetectedWeed

    # Two heavily overlapping boxes
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 48.0, 48.0],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)

    # 1. Class-agnostic (same class or class_ids=None): only highest score retained
    assert nms(boxes, scores, iou_thr=0.45) == [0]
    assert nms(boxes, scores, iou_thr=0.45, class_ids=np.array([0, 0])) == [0]

    # 2. Class-aware (different classes): both must be kept
    assert nms(boxes, scores, iou_thr=0.45, class_ids=np.array([0, 1])) == [0, 1]

    # 3. Test nms_boxes with xywh format
    boxes_xywh = [[10.0, 10.0, 40.0, 40.0], [12.0, 12.0, 36.0, 36.0]]
    assert nms_boxes(boxes_xywh, scores, iou_thr=0.45, box_format="xywh") == [0]
    assert nms_boxes(boxes_xywh, scores, iou_thr=0.45, class_ids=[0, 1], box_format="xywh") == [0, 1]

    # 4. Test nms_detected_weeds on entities
    w1 = DetectedWeed(
        weed_id="w1", class_id=0, class_name="dicot", confidence=0.95,
        bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2)
    )
    w2_same_cls = DetectedWeed(
        weed_id="w2", class_id=0, class_name="dicot", confidence=0.80,
        bbox=BBox(x=0.11, y=0.11, w=0.19, h=0.19)
    )
    w3_diff_cls = DetectedWeed(
        weed_id="w3", class_id=1, class_name="monocot", confidence=0.88,
        bbox=BBox(x=0.11, y=0.11, w=0.19, h=0.19)
    )

    # Suppress duplicate of same class
    filtered_same = nms_detected_weeds([w1, w2_same_cls], iou_thr=0.45)
    assert len(filtered_same) == 1
    assert filtered_same[0].weed_id == "w1"

    # Keep both when classes differ
    filtered_diff = nms_detected_weeds([w1, w3_diff_cls], iou_thr=0.45, class_aware=True)
    assert len(filtered_diff) == 2


