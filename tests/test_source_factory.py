"""Tests for SourceFactory and frame sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from geo_weed_station.core.exceptions import ImageReadError
from geo_weed_station.core.source_factory import (
    DroneRunSource,
    FolderSource,
    FrameSource,
    ImageSource,
    SourceFactory,
    VideoSource,
    load_image_payload,
)
from geo_weed_station.schemas.mission import Waypoint
from tests.helpers.make_media import make_mp4, make_real_jpeg
from tests.helpers.make_xmp_image import SAMPLE_XMP_LOCAL_ARUCO
from tests.mocks.fixtures_generator import make_summary_detections


def test_factory_kinds(tmp_path: Path):
    img = make_real_jpeg(tmp_path / "a.jpg", SAMPLE_XMP_LOCAL_ARUCO)
    vid = make_mp4(tmp_path / "v.mp4")
    assert isinstance(SourceFactory.create("image", img), ImageSource)
    assert isinstance(SourceFactory.create("folder", tmp_path), FolderSource)
    assert isinstance(SourceFactory.create("video", vid), VideoSource)
    assert isinstance(SourceFactory.create("drone_run", tmp_path), DroneRunSource)
    assert isinstance(SourceFactory.create("image", img), FrameSource)
    with pytest.raises(ValueError):
        SourceFactory.create("bogus", tmp_path)  # type: ignore[arg-type]


def test_folder_recursive_scan_extensions(tmp_path: Path):
    make_real_jpeg(tmp_path / "a.jpg")
    make_real_jpeg(tmp_path / "deep" / "er" / "b.JPEG")
    (tmp_path / "c.png").write_bytes(b"x")
    (tmp_path / "d.tif").write_bytes(b"x")
    (tmp_path / "e.tiff").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "meta.json").write_text("{}")
    entries = SourceFactory.create("folder", tmp_path).entries()
    names = sorted(Path(e.path).name for e in entries)
    assert names == ["a.jpg", "b.JPEG", "c.png", "d.tif", "e.tiff"]


def test_folder_scan_ignores_report_subfolder_and_report_pngs(tmp_path: Path):
    make_real_jpeg(tmp_path / "frame_01.jpg")
    # Report subfolder with PNG map inside the images directory
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "detection_report.png").write_bytes(b"png")
    (report_dir / "custom_map.png").write_bytes(b"png")
    # Legacy report PNG in root folder
    (tmp_path / "detection_report.png").write_bytes(b"png")
    (tmp_path / "weed_map_export.png").write_bytes(b"png")

    entries = SourceFactory.create("folder", tmp_path).entries()
    names = [Path(e.path).name for e in entries]
    assert names == ["frame_01.jpg"]


def test_image_load_uses_embedded_xmp(tmp_path: Path):
    img = make_real_jpeg(tmp_path / "a.jpg", SAMPLE_XMP_LOCAL_ARUCO)
    src = SourceFactory.create("image", img)
    payload = src.load(src.entries()[0])
    assert payload.image is not None and payload.image.shape == (48, 64, 3)
    assert payload.image_size_px == (64, 48)
    assert payload.pose is not None and payload.pose.altitude_m == pytest.approx(2.0)
    assert payload.has_georef()


def test_image_without_xmp_has_no_pose(tmp_path: Path):
    img = make_real_jpeg(tmp_path / "a.jpg")
    src = SourceFactory.create("image", img)
    assert src.load(src.entries()[0]).pose is None


def test_corrupted_image_raises(tmp_path: Path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    src = SourceFactory.create("image", bad)
    with pytest.raises(ImageReadError):
        src.load(src.entries()[0])


def test_missing_paths_raise(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SourceFactory.create("image", tmp_path / "nope.jpg")
    with pytest.raises(NotADirectoryError):
        SourceFactory.create("folder", tmp_path / "nope")


def test_video_source_first_frame(tmp_path: Path):
    vid = make_mp4(tmp_path / "v.mp4")
    src = SourceFactory.create("video", vid)
    payload = src.load(src.entries()[0])
    assert payload.image_size_px == (64, 48)
    assert payload.pose is None


def test_drone_run_attaches_summary_detections(tmp_path: Path):
    wps = [Waypoint(x=0.0, y=0.0, z=2.0), Waypoint(x=1.0, y=0.0, z=2.0)]
    summary = make_summary_detections("run1", wps)
    (tmp_path / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    for i in range(2):
        make_real_jpeg(tmp_path / f"wp_{i:04d}.jpg", SAMPLE_XMP_LOCAL_ARUCO)
    src = SourceFactory.create("drone_run", tmp_path)
    entries = src.entries()
    assert [e.detections_count for e in entries] == [1, 0]
    p0 = src.load(entries[0])
    assert len(p0.detections) == 1
    assert p0.detections[0].class_name == "taraxacum"
    assert len(src.load(entries[1]).detections) == 0


def test_drone_run_broken_summary_is_ignored(tmp_path: Path):
    (tmp_path / "summary.json").write_text("{broken", encoding="utf-8")
    make_real_jpeg(tmp_path / "wp_0000.jpg", SAMPLE_XMP_LOCAL_ARUCO)
    src = SourceFactory.create("drone_run", tmp_path)
    assert len(src.load(src.entries()[0]).detections) == 0


def test_load_image_payload_scales_principal_point_when_downscaled(tmp_path: Path):
    """When read_image downscales >8000x6000 images, principal_point_px scales proportionally."""
    xmp = SAMPLE_XMP_LOCAL_ARUCO.replace(
        'Camera:PrincipalPoint="1280.0,960.0"',
        'Camera:PrincipalPoint="4096.0,3072.0"',
    )
    big_img = make_real_jpeg(tmp_path / "big_50mp.jpg", xmp_xml=xmp, size=(8192, 6144))

    payload_meta = load_image_payload(big_img, load_pixels=False)
    assert payload_meta.image_size_px == (8192, 6144)
    assert payload_meta.optics.principal_point_px == pytest.approx((4096.0, 3072.0))

    payload_px = load_image_payload(big_img, load_pixels=True)
    assert payload_px.image_size_px == (8000, 6000)
    assert payload_px.optics.principal_point_px[0] == pytest.approx(4000.0, abs=0.5)
    assert payload_px.optics.principal_point_px[1] == pytest.approx(3000.0, abs=0.5)
