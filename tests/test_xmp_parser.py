"""Comprehensive unit tests for xmp_parser module (JPEG and TIFF)."""

from __future__ import annotations

import builtins
import time
from pathlib import Path

import pytest

from geo_weed_station.core.exceptions import (
    XmpMalformedError,
    XmpNotFoundError,
)
from geo_weed_station.core.xmp_parser import (
    build_frame_payload,
    extract_spatial_data,
    has_xmp,
    parse_xmp,
    read_xmp_packet,
)
from geo_weed_station.schemas.entities import CoordinateSystem
from tests.helpers.make_xmp_image import (
    SAMPLE_XMP_ELEMENT_FORM,
    SAMPLE_XMP_LOCAL_ARUCO,
    SAMPLE_XMP_WGS84,
    create_jpeg_with_xmp,
    create_tiff_with_xmp,
)


def test_parse_reference_local_aruco(tmp_path: Path):
    """Verify parsing reference XMP packet from technical specification (local_aruco)."""
    img_path = create_jpeg_with_xmp(tmp_path / "aruco.jpg", SAMPLE_XMP_LOCAL_ARUCO)

    assert has_xmp(img_path) is True
    data = extract_spatial_data(img_path)

    assert data.pose is not None
    assert data.pose.coordinate_system == CoordinateSystem.LOCAL_ARUCO
    assert data.pose.x == 1.50
    assert data.pose.y == 0.80
    assert data.pose.altitude_m == 2.00
    assert data.pose.yaw_deg == 15.0
    assert data.pose.pitch_deg == -90.0
    assert data.pose.roll_deg == 0.0

    assert data.optics is not None
    assert data.optics.focal_length_mm == 3.6
    assert data.optics.principal_point_px == (1280.0, 960.0)
    assert data.optics.band_names == ["RGB"]


def test_parse_wgs84(tmp_path: Path):
    """Verify parsing global geographic coordinates in WGS84."""
    img_path = create_jpeg_with_xmp(tmp_path / "wgs.jpg", SAMPLE_XMP_WGS84)
    data = extract_spatial_data(img_path)

    assert data.pose is not None
    assert data.pose.coordinate_system == CoordinateSystem.WGS84
    assert data.pose.x == pytest.approx(37.618423)  # Longitude
    assert data.pose.y == pytest.approx(55.751244)  # Latitude
    assert data.pose.altitude_m == 15.50
    assert data.pose.yaw_deg == -45.0
    assert data.pose.pitch_deg == -85.0
    assert data.pose.roll_deg == 1.2

    assert data.optics is not None
    assert data.optics.focal_length_mm == 8.8
    assert data.optics.principal_point_px == (2000.0, 1500.0)
    assert data.optics.band_names == ["RGB", "RedEdge"]


def test_parse_element_form_rdf(tmp_path: Path):
    """Verify parsing element-based RDF tags."""
    img_path = create_jpeg_with_xmp(tmp_path / "element.jpg", SAMPLE_XMP_ELEMENT_FORM)
    data = extract_spatial_data(img_path)

    assert data.pose is not None
    assert data.pose.coordinate_system == CoordinateSystem.LOCAL_ARUCO
    assert data.pose.x == 2.10
    assert data.pose.y == 1.80
    assert data.pose.altitude_m == 3.20


def test_jpeg_without_xmp(tmp_path: Path):
    """Verify that JPEG without XMP raises XmpNotFoundError and has_xmp is False."""
    img_path = create_jpeg_with_xmp(tmp_path / "no_xmp.jpg", xmp_xml=None)

    assert has_xmp(img_path) is False
    with pytest.raises(XmpNotFoundError):
        read_xmp_packet(img_path)

    with pytest.raises(XmpNotFoundError):
        extract_spatial_data(img_path)


def test_corrupted_xmp_xml(tmp_path: Path):
    """Verify that corrupted XML string raises XmpMalformedError."""
    corrupted_xml = "<x:xmpmeta><rdf:RDF><rdf:Description unclosed"
    img_path = create_jpeg_with_xmp(tmp_path / "corrupted.jpg", corrupted_xml)

    with pytest.raises(XmpMalformedError):
        extract_spatial_data(img_path)


def test_wgs84_out_of_bounds_validation():
    """Verify that WGS84 coordinates exceeding latitude/longitude boundaries are rejected."""
    invalid_lat_xml = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
      <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description
            xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
            xmlns:agro="http://geosaero.ru/agro/1.0/"
            drone-dji:RelativeAltitude="+10.0"
            drone-dji:GimbalYawDegree="0.0"
            drone-dji:GimbalPitchDegree="-90.0"
            drone-dji:GimbalRollDegree="0.0"
            agro:CoordinateSystem="wgs84"
            drone-dji:GpsLatitude="95.0"
            drone-dji:GpsLongitude="37.0"/>
      </rdf:RDF>
    </x:xmpmeta>"""
    with pytest.raises(XmpMalformedError, match="latitude out of range"):
        parse_xmp(invalid_lat_xml)


def test_zero_or_negative_altitude():
    """Verify that H <= 0 leads to pose=None."""
    zero_alt_xml = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
      <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description
            xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
            xmlns:agro="http://geosaero.ru/agro/1.0/"
            drone-dji:RelativeAltitude="0.0"
            drone-dji:GimbalYawDegree="0.0"
            drone-dji:GimbalPitchDegree="-90.0"
            drone-dji:GimbalRollDegree="0.0"
            agro:CoordinateSystem="local_aruco"
            agro:CameraX="1.0"
            agro:CameraY="2.0"/>
      </rdf:RDF>
    </x:xmpmeta>"""
    data = parse_xmp(zero_alt_xml)
    assert data.pose is None


def test_tiff_with_xmp(tmp_path: Path):
    """Verify extraction and parsing of XMP tag from TIFF file."""
    tiff_path = create_tiff_with_xmp(tmp_path / "test.tif", SAMPLE_XMP_LOCAL_ARUCO)

    assert has_xmp(tiff_path) is True
    data = extract_spatial_data(tiff_path)
    assert data.pose is not None
    assert data.pose.coordinate_system == CoordinateSystem.LOCAL_ARUCO
    assert data.pose.x == 1.50
    assert data.pose.y == 0.80
    assert data.pose.altitude_m == 2.00


def test_benchmark_large_file_under_50ms(tmp_path: Path):
    """Verify that extracting XMP from a large file (~5MB simulated 8000x6000) takes <= 50ms."""
    # Create large file with 5 MB of padding after header
    large_img = create_jpeg_with_xmp(
        tmp_path / "large_8000x6000.jpg",
        SAMPLE_XMP_LOCAL_ARUCO,
        padding_bytes=5 * 1024 * 1024,
    )

    start = time.perf_counter()
    data = extract_spatial_data(large_img)
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert data.pose is not None
    assert data.pose.altitude_m == 2.00
    assert duration_ms <= 50.0, f"Extraction took {duration_ms:.2f} ms (must be <= 50 ms)"


def test_no_sidecar_file_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify parser never opens external sidecar files (.txt, .xml, .json)."""
    img_path = create_jpeg_with_xmp(tmp_path / "photo.jpg", SAMPLE_XMP_LOCAL_ARUCO)

    # Place dummy sidecar files next to photo
    (tmp_path / "photo.txt").write_text("dummy sidecar", encoding="utf-8")
    (tmp_path / "photo.xml").write_text("<sidecar/>", encoding="utf-8")
    (tmp_path / "photo.json").write_text("{}", encoding="utf-8")

    opened_files: list[str] = []
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        path_str = str(file)
        opened_files.append(path_str)
        # Forbid opening sidecars
        if any(path_str.endswith(ext) for ext in [".txt", ".xml", ".json"]) and "api_samples" not in path_str:
            raise PermissionError(f"Sidecar access forbidden: {path_str}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    data = extract_spatial_data(img_path)
    assert data.pose is not None

    # Verify only target image was opened
    for opened in opened_files:
        assert not opened.endswith(".txt")
        assert not (opened.endswith(".xml") and not opened.startswith("/usr"))


def test_build_frame_payload_safe_fallback(tmp_path: Path):
    """Verify build_frame_payload safely handles files without XMP without crashing."""
    no_xmp = create_jpeg_with_xmp(tmp_path / "plain.jpg", xmp_xml=None)
    payload = build_frame_payload(no_xmp, image_size_px=(1920, 1080))

    assert payload.frame_id == "plain"
    assert payload.pose is None
    assert payload.has_georef() is False
    assert payload.image_size_px == (1920, 1080)

    # Valid XMP
    with_xmp = create_jpeg_with_xmp(tmp_path / "valid.jpg", SAMPLE_XMP_LOCAL_ARUCO)
    payload_valid = build_frame_payload(with_xmp, image_size_px=(1920, 1080))
    assert payload_valid.pose is not None
    assert payload_valid.has_georef() is True
