"""Unit tests for ONNX crop classification pipeline in GeoWeedStation."""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from geo_weed_station.config.settings import DetectorSettings
from geo_weed_station.core.annotation_io import parse_detection_item, save_annotations_to_folder
from geo_weed_station.core.onnx_detector import (
    C06_TAXONOMY,
    C07_CATEGORIES,
    C07_STAGES,
    OnnxDetector,
)
from geo_weed_station.schemas.entities import BBox, DetectedWeed


def test_detected_weed_entity_with_classification():
    """Verify DetectedWeed entity accepts and validates stage_name and class_prob."""
    weed = DetectedWeed(
        weed_id="det_0",
        class_id=1,
        class_name="Осот, Бодяк, Латук",
        confidence=0.9123,
        bbox=BBox(x=0.1, y=0.2, w=0.3, h=0.4),
        stage_name="2-4 листа",
        class_prob=0.8842,
    )
    assert weed.class_name == "Осот, Бодяк, Латук"
    assert weed.stage_name == "2-4 листа"
    assert weed.class_prob == 0.8842

    # Verify serialization / deserialization roundtrip
    data = weed.model_dump()
    assert data["stage_name"] == "2-4 листа"
    assert data["class_prob"] == 0.8842

    reconstructed = DetectedWeed.model_validate(data)
    assert reconstructed == weed


def test_c07_softmax_decoding_math():
    """Verify C07 dual-head Softmax math and taxonomy mapping."""
    logits_cat = np.array([[0.0, 5.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.2, 0.1, 0.0]], dtype=np.float32)
    logits_stg = np.array([[0.1, 4.0, 0.3, 0.1, 0.0]], dtype=np.float32)

    p_cat = np.exp(logits_cat[0] - np.max(logits_cat[0])) / np.sum(np.exp(logits_cat[0] - np.max(logits_cat[0])))
    p_stg = np.exp(logits_stg[0] - np.max(logits_stg[0])) / np.sum(np.exp(logits_stg[0] - np.max(logits_stg[0])))

    cat_id = int(np.argmax(p_cat))
    stg_id = int(np.argmax(p_stg))

    assert cat_id == 1
    assert stg_id == 1
    assert C07_CATEGORIES[cat_id] == "Осот, Бодяк, Латук"
    assert C07_STAGES[stg_id] == "2-4 листа"
    assert p_cat[cat_id] > 0.90
    assert p_stg[stg_id] > 0.90


def test_crop_preprocessing_nchw():
    """Verify 10% padding expansion, RGB ImageNet normalization, and NCHW shape."""
    orig_w, orig_h = 800, 600
    # Dummy box: x1=100, y1=150, x2=200, y2=250
    x1, y1, x2, y2 = 100.0, 150.0, 200.0, 250.0

    pad_w = int(round((x2 - x1) * 0.10))  # 10 px
    pad_h = int(round((y2 - y1) * 0.10))  # 10 px
    cx1 = max(0, int(round(x1)) - pad_w)  # 90
    cy1 = max(0, int(round(y1)) - pad_h)  # 140
    cx2 = min(orig_w, int(round(x2)) + pad_w)  # 210
    cy2 = min(orig_h, int(round(y2)) + pad_h)  # 260

    assert cx1 == 90 and cy1 == 140
    assert cx2 == 210 and cy2 == 260

    synthetic_frame = np.full((orig_h, orig_w, 3), 128, dtype=np.uint8)
    crop = synthetic_frame[cy1:cy2, cx1:cx2]
    crop_res = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop_norm = (crop_res.astype(np.float32) / 255.0 - mean) / std
    crop_tensor = np.transpose(crop_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    assert crop_tensor.shape == (1, 3, 224, 224)
    assert crop_tensor.dtype == np.float32


def test_annotation_io_preserves_classification(tmp_path: Path):
    """Verify that saving and loading annotations preserves stage_name and class_prob."""
    weed = DetectedWeed(
        weed_id="weed_123",
        class_id=2,
        class_name="Вьюнок",
        confidence=0.88,
        bbox=BBox(x=0.2, y=0.3, w=0.4, h=0.4),
        world_x=1.23,
        world_y=4.56,
        stage_name="всходы",
        class_prob=0.825,
    )

    dets_by_path = {"img1.jpg": [weed]}
    saved_file = save_annotations_to_folder(tmp_path, dets_by_path)
    assert saved_file.is_file()

    raw_json = json.loads(saved_file.read_text(encoding="utf-8"))
    frame_entry = raw_json["frames"]["img1.jpg"]
    raw_det = frame_entry["detections"][0]

    assert raw_det["class_name"] == "Вьюнок"
    assert raw_det["stage_name"] == "всходы"
    assert raw_det["class_prob"] == 0.825

    # Re-parse through parse_detection_item
    parsed = parse_detection_item(raw_det, (800, 600))
    assert parsed is not None
    assert parsed.class_name == "Вьюнок"
    assert parsed.stage_name == "всходы"
    assert parsed.class_prob == 0.825
    assert parsed.world_x == 1.23
    assert parsed.world_y == 4.56


def test_real_resnet50_dualhead_c07_onnx_inference():
    """Verify inference on real resnet50_dualhead_c07.onnx if file is present."""
    model_path = Path("/home/user/maximus/Ekat2026/flying_robotics/repositories/models/c07_rgb_geosaero/resnet50_dualhead_c07.onnx")
    if not model_path.is_file():
        pytest.skip("resnet50_dualhead_c07.onnx not found")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    assert inp_name == "images"

    # Feed dummy normalized batch [1, 3, 224, 224]
    dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
    outs = sess.run(None, {inp_name: dummy_input})

    assert len(outs) == 2
    logits_cat, logits_stg = outs[0], outs[1]
    assert logits_cat.shape == (1, 11)
    assert logits_stg.shape == (1, 5)

    cat_id = int(np.argmax(logits_cat[0]))
    stg_id = int(np.argmax(logits_stg[0]))
    assert 0 <= cat_id < len(C07_CATEGORIES)
    assert 0 <= stg_id < len(C07_STAGES)


def test_onnx_detector_two_stage_c07_integration(tmp_path: Path):
    """Verify OnnxDetector end-to-end detection and classification on image."""
    from tests.helpers.make_dummy_onnx import create_dummy_yolo_onnx

    det_model = create_dummy_yolo_onnx(tmp_path / "dummy_yolo.onnx")
    cands = [
        Path("GeoWeedStation/models/resnet50_dualhead_c07.onnx"),
        Path("models/resnet50_dualhead_c07.onnx"),
        Path("models/c07_rgb_geosaero/resnet50_dualhead_c07.onnx"),
        Path("/home/user/maximus/Ekat2026/flying_robotics/repositories/models/c07_rgb_geosaero/resnet50_dualhead_c07.onnx"),
    ]
    cls_model = next((c for c in cands if c.is_file()), None)
    if cls_model is None:
        pytest.skip("resnet50_dualhead_c07.onnx not found")

    settings = DetectorSettings(
        model_path=det_model,
        classifier_type="c07",
        classifier_path=cls_model,
        confidence_threshold=0.3,
    )
    detector = OnnxDetector(settings, providers=["CPUExecutionProvider"])
    detector.load()

    assert detector.is_loaded
    assert detector._classifier_session is not None

    # Synthetic RGB frame 640x640
    img_rgb = np.full((640, 640, 3), 120, dtype=np.uint8)

    detections = detector.detect(img_rgb)
    assert len(detections) > 0

    first = detections[0]
    assert first.class_name in C07_CATEGORIES
    assert first.stage_name in C07_STAGES
    assert first.class_prob is not None
    assert 0.0 <= first.class_prob <= 1.0

    # Toggle off classifier
    detector.set_classifier("none")
    detections_none = detector.detect(img_rgb)
    assert len(detections_none) > 0
    assert detections_none[0].stage_name is None
    assert detections_none[0].class_prob is None

    detector.close()
    assert not detector.is_loaded


def test_onnx_detector_filters_out_background_after_second_stage(tmp_path: Path):
    """Verify OnnxDetector.detect skips detections classified as background (cat_id=0 in C07, sp_id=0 in C06)."""
    from tests.helpers.make_dummy_onnx import create_dummy_yolo_onnx

    det_model = create_dummy_yolo_onnx(tmp_path / "dummy_yolo.onnx")
    settings = DetectorSettings(
        model_path=det_model,
        classifier_type="c07",
        confidence_threshold=0.3,
    )
    detector = OnnxDetector(settings, providers=["CPUExecutionProvider"])
    detector.load()

    class _DummyBgSession:
        def run(self, _output_names, _input_feed):
            # cat_id = 0 ("background") has highest logit
            logits_cat = np.array([[10.0] + [0.0] * 10], dtype=np.float32)
            logits_stg = np.array([[5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            return [logits_cat, logits_stg]

    detector._classifier_session = _DummyBgSession()
    detector._cls_input_name = "images"

    img_rgb = np.full((640, 640, 3), 120, dtype=np.uint8)
    dets_c07_bg = detector.detect(img_rgb)
    assert dets_c07_bg == []

    # Same for C06 when sp_id = 0 ("S000 Фон / Не растение")
    detector.classifier_type = "c06"

    class _DummyC06BgSession:
        def run(self, _output_names, _input_feed):
            logits = np.array([[10.0] + [0.0] * 23], dtype=np.float32)
            return [logits]

    detector._classifier_session = _DummyC06BgSession()
    dets_c06_bg = detector.detect(img_rgb)
    assert dets_c06_bg == []

    detector.close()
