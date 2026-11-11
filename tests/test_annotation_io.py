"""Unit tests for annotation_io module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geo_weed_station.core.annotation_io import (
    find_annotation_file,
    load_annotations_file,
    parse_detection_item,
    save_annotations_to_folder,
)
from geo_weed_station.schemas.entities import BBox, DetectedWeed
from geo_weed_station.schemas.session import FileEntry, ProcessingStatus


def test_find_annotation_file_priority(tmp_path: Path):
    assert find_annotation_file(tmp_path) is None

    # Summary.json present
    summary_f = tmp_path / "summary.json"
    summary_f.write_text("{}", encoding="utf-8")
    assert find_annotation_file(tmp_path) == summary_f

    # Detections.json overrides summary.json
    detections_f = tmp_path / "detections.json"
    detections_f.write_text("{}", encoding="utf-8")
    assert find_annotation_file(tmp_path) == detections_f

    # Annotations.json has highest priority
    annotations_f = tmp_path / "annotations.json"
    annotations_f.write_text("{}", encoding="utf-8")
    assert find_annotation_file(tmp_path) == annotations_f


def test_parse_detection_item():
    # Valid dict bbox
    raw = {
        "weed_id": "w1",
        "class_id": 1,
        "class_name": "Sunflower",
        "confidence": 0.85,
        "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        "world_x": 1.5,
        "world_y": 2.5,
    }
    det = parse_detection_item(raw)
    assert det is not None
    assert det.weed_id == "w1"
    assert det.class_name == "Sunflower"
    assert det.confidence == 0.85
    assert det.bbox.x == 0.1
    assert det.bbox.w == 0.3
    assert det.world_x == 1.5

    # Pixel bbox_px
    raw_px = {
        "class_name": "Corn",
        "confidence": 0.9,
        "bbox_px": [100, 200, 300, 400],
    }
    det_px = parse_detection_item(raw_px, img_size=(1000, 1000))
    assert det_px is not None
    assert det_px.class_name == "Corn"
    assert det_px.bbox.x == 0.1
    assert det_px.bbox.y == 0.2
    assert pytest.approx(det_px.bbox.w) == 0.2
    assert pytest.approx(det_px.bbox.h) == 0.2


def test_save_and_load_roundtrip(tmp_path: Path):
    folder = tmp_path / "images"
    folder.mkdir()
    img1 = folder / "frame_001.jpg"
    img2 = folder / "frame_002.jpg"
    img1.touch()
    img2.touch()

    dets1 = [
        DetectedWeed(
            weed_id="d1",
            class_id=0,
            class_name="Ambrosia",
            confidence=0.95,
            bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
            world_x=0.5,
            world_y=1.0,
        )
    ]
    dets2 = [
        DetectedWeed(
            weed_id="d2",
            class_id=1,
            class_name="Chenopodium",
            confidence=0.88,
            bbox=BBox(x=0.3, y=0.3, w=0.15, h=0.15),
        )
    ]

    detections_by_path = {
        str(img1): dets1,
        str(img2): dets2,
    }
    entries = [
        FileEntry(path=str(img1), size_bytes=100),
        FileEntry(path=str(img2), size_bytes=100),
    ]

    saved_file = save_annotations_to_folder(folder, detections_by_path, entries)
    assert saved_file.is_file()
    assert saved_file.name == "annotations.json"

    # Verify JSON content
    data = json.loads(saved_file.read_text(encoding="utf-8"))
    assert data["total_frames"] == 2
    assert data["total_detections"] == 2
    assert "Ambrosia" in data["classes"]
    assert "Chenopodium" in data["classes"]

    # Load back
    loaded = load_annotations_file(saved_file, [img1, img2])
    assert str(img1) in loaded
    assert len(loaded[str(img1)]) == 1
    assert loaded[str(img1)][0].class_name == "Ambrosia"
    assert loaded[str(img1)][0].world_x == 0.5

    assert str(img2) in loaded
    assert len(loaded[str(img2)]) == 1
    assert loaded[str(img2)][0].class_name == "Chenopodium"


def test_load_summary_json_format(tmp_path: Path):
    folder = tmp_path / "drone_run"
    folder.mkdir()
    img0 = folder / "capture_000.jpg"
    img1 = folder / "capture_001.jpg"
    img0.touch()
    img1.touch()

    summary_data = {
        "run_id": "run_test",
        "coordinate_system": "aruco_local",
        "detections_count": 2,
        "items": [
            {
                "wp_index": 0,
                "camera_x": 0.5,
                "camera_y": 0.5,
                "altitude_m": 1.5,
                "detections": [
                    {
                        "class_name": "weed_dicot",
                        "confidence": 0.91,
                        "bbox_px": [100, 100, 200, 200],
                        "target_local_coords": {"x": 0.4, "y": 0.6},
                    }
                ],
            },
            {
                "wp_index": 1,
                "camera_x": 1.5,
                "camera_y": 0.5,
                "altitude_m": 1.5,
                "detections": [
                    {
                        "class_name": "weed_monocot",
                        "confidence": 0.77,
                        "bbox_px": [300, 300, 450, 450],
                    }
                ],
            },
        ],
    }
    sum_file = folder / "summary.json"
    sum_file.write_text(json.dumps(summary_data), encoding="utf-8")

    loaded = load_annotations_file(sum_file, [img0, img1])
    assert str(img0) in loaded
    assert len(loaded[str(img0)]) == 1
    assert loaded[str(img0)][0].class_name == "weed_dicot"
    assert loaded[str(img0)][0].world_x == 0.4

    assert str(img1) in loaded
    assert len(loaded[str(img1)]) == 1
    assert loaded[str(img1)][0].class_name == "weed_monocot"


def test_save_and_load_status_preservation(tmp_path: Path):
    folder = tmp_path / "survey"
    folder.mkdir()

    f_done_dets = folder / "f_done_dets.jpg"
    f_done_zero = folder / "f_done_zero.jpg"
    f_no_meta_dets = folder / "f_no_meta_dets.jpg"
    f_no_meta_zero = folder / "f_no_meta_zero.jpg"
    f_pending = folder / "f_pending.jpg"
    f_error = folder / "f_error.jpg"

    for f in (f_done_dets, f_done_zero, f_no_meta_dets, f_no_meta_zero, f_pending, f_error):
        f.touch()

    det1 = DetectedWeed(
        weed_id="w1",
        class_id=0,
        class_name="WeedA",
        confidence=0.9,
        bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
    )
    det2 = DetectedWeed(
        weed_id="w2",
        class_id=1,
        class_name="WeedB",
        confidence=0.8,
        bbox=BBox(x=0.2, y=0.2, w=0.1, h=0.1),
    )

    detections_by_path = {
        str(f_done_dets): [det1],
        str(f_done_zero): [],
        str(f_no_meta_dets): [det2],
        str(f_no_meta_zero): [],
        str(f_pending): [],
        str(f_error): [],
    }

    entries = [
        FileEntry(path=str(f_done_dets), size_bytes=100, status=ProcessingStatus.DONE, has_xmp=True, detections_count=1),
        FileEntry(path=str(f_done_zero), size_bytes=100, status=ProcessingStatus.DONE, has_xmp=True, detections_count=0),
        FileEntry(path=str(f_no_meta_dets), size_bytes=100, status=ProcessingStatus.NO_METADATA, has_xmp=False, detections_count=1),
        FileEntry(path=str(f_no_meta_zero), size_bytes=100, status=ProcessingStatus.NO_METADATA, has_xmp=False, detections_count=0),
        FileEntry(path=str(f_pending), size_bytes=100, status=ProcessingStatus.PENDING, has_xmp=False, detections_count=None),
        FileEntry(path=str(f_error), size_bytes=100, status=ProcessingStatus.ERROR, has_xmp=False, detections_count=None),
    ]

    saved = save_annotations_to_folder(folder, detections_by_path, entries)
    assert saved.is_file()

    loaded = load_annotations_file(saved, [f_done_dets, f_done_zero, f_no_meta_dets, f_no_meta_zero, f_pending, f_error])

    assert loaded.get_status(f_done_dets) == ProcessingStatus.DONE
    assert loaded.get_detections_count(f_done_dets) == 1
    assert len(loaded[str(f_done_dets)]) == 1

    assert loaded.get_status(f_done_zero) == ProcessingStatus.DONE
    assert loaded.get_detections_count(f_done_zero) == 0
    assert len(loaded[str(f_done_zero)]) == 0

    assert loaded.get_status(f_no_meta_dets) == ProcessingStatus.NO_METADATA
    assert loaded.get_detections_count(f_no_meta_dets) == 1
    assert len(loaded[str(f_no_meta_dets)]) == 1

    assert loaded.get_status(f_no_meta_zero) == ProcessingStatus.NO_METADATA
    assert loaded.get_detections_count(f_no_meta_zero) == 0
    assert len(loaded[str(f_no_meta_zero)]) == 0

    assert loaded.get_status(f_pending) == ProcessingStatus.PENDING
    assert loaded.get_status(f_error) == ProcessingStatus.ERROR

    # Also test FolderSource.entries() restoration
    from geo_weed_station.core.source_factory import FolderSource
    src = FolderSource(folder)
    restored_entries = {Path(e.path).name: e for e in src.entries()}

    assert restored_entries["f_done_dets.jpg"].status == ProcessingStatus.DONE
    assert restored_entries["f_done_dets.jpg"].detections_count == 1

    assert restored_entries["f_done_zero.jpg"].status == ProcessingStatus.DONE
    assert restored_entries["f_done_zero.jpg"].detections_count == 0

    assert restored_entries["f_no_meta_dets.jpg"].status == ProcessingStatus.NO_METADATA
    assert restored_entries["f_no_meta_dets.jpg"].detections_count == 1

    assert restored_entries["f_no_meta_zero.jpg"].status == ProcessingStatus.NO_METADATA
    assert restored_entries["f_no_meta_zero.jpg"].detections_count == 0

    assert restored_entries["f_pending.jpg"].status == ProcessingStatus.PENDING
    assert restored_entries["f_pending.jpg"].detections_count is None

    assert restored_entries["f_error.jpg"].status == ProcessingStatus.ERROR
    assert restored_entries["f_error.jpg"].detections_count is None


def test_load_summary_json_uses_real_image_size_and_altitude(tmp_path: Path):
    """Pixel bbox_px in summary.json normalizes using actual JPEG dims and flags z<=0 as NO_METADATA."""
    from tests.helpers.make_media import make_real_jpeg

    folder = tmp_path / "run_summary"
    folder.mkdir()
    img0 = make_real_jpeg(folder / "capture_000.jpg", size=(1600, 1200))
    img1 = make_real_jpeg(folder / "capture_001.jpg", size=(1600, 1200))

    summary_data = {
        "run_id": "run_test_real_size",
        "coordinate_system": "aruco_local",
        "detections_count": 2,
        "items": [
            {
                "wp_index": 0,
                "camera_x": 0.5,
                "camera_y": 0.5,
                "altitude_m": 2.0,
                "detections": [
                    {
                        "class_name": "weed",
                        "confidence": 0.9,
                        "bbox_px": [400, 300, 800, 600],
                    }
                ],
            },
            {
                "wp_index": 1,
                "camera_x": 1.0,
                "camera_y": 0.5,
                "altitude_m": 0.0,
                "detections": [
                    {
                        "class_name": "weed",
                        "confidence": 0.8,
                        "bbox_px": [160, 120, 320, 240],
                    }
                ],
            },
        ],
    }
    sum_file = folder / "summary.json"
    sum_file.write_text(json.dumps(summary_data), encoding="utf-8")

    loaded = load_annotations_file(sum_file, [img0, img1])
    det0 = loaded[str(img0)][0]
    assert det0.bbox.x == pytest.approx(0.25)
    assert det0.bbox.y == pytest.approx(0.25)
    assert det0.bbox.w == pytest.approx(0.25)
    assert det0.bbox.h == pytest.approx(0.25)
    assert loaded.get_has_xmp(img0) is True
    assert loaded.get_status(img0) == ProcessingStatus.DONE

    det1 = loaded[str(img1)][0]
    assert det1.bbox.x == pytest.approx(0.1)
    assert det1.bbox.y == pytest.approx(0.1)
    assert loaded.get_has_xmp(img1) is False
    assert loaded.get_status(img1) == ProcessingStatus.NO_METADATA
