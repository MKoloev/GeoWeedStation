"""Unit tests for ReportGenerator and FolderDetectionReport exports."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from geo_weed_station.config.settings import DetectorSettings, MapSettings
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.core.report_generator import ReportGenerator
from geo_weed_station.schemas.entities import (
    BBox,
    CameraSpatialPose,
    CoordinateSystem,
    DetectedWeed,
    SensorOptics,
)
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.schemas.session import FileEntry, ProcessingStatus


def test_build_and_export_report_with_georef_and_stages(tmp_path: Path):
    engine = MapEngine(projector=NadirProjector(), settings=MapSettings(heatmap_cell_m=1.0))

    pose1 = CameraSpatialPose(
        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
        x=2.0,
        y=3.0,
        altitude_m=2.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
    )
    optics = SensorOptics(focal_length_mm=4.5, sensor_width_mm=6.4, sensor_height_mm=4.8)

    det1 = DetectedWeed(
        weed_id="w_01",
        class_id=1,
        class_name="Осот, Бодяк, Латук",
        confidence=0.91,
        bbox=BBox(x=0.4, y=0.4, w=0.1, h=0.1),
        stage_name="2-4 листа",
        class_prob=0.87,
    )
    det2 = DetectedWeed(
        weed_id="w_02",
        class_id=2,
        class_name="Вьюнок",
        confidence=0.78,
        bbox=BBox(x=0.1, y=0.2, w=0.1, h=0.1),
        stage_name="всходы",
        class_prob=0.82,
    )

    img1 = tmp_path / "frame_001.jpg"
    img2 = tmp_path / "frame_002.jpg"
    img1.write_bytes(b"fake_jpeg_1")
    img2.write_bytes(b"fake_jpeg_2")

    payload1 = FramePayload(
        frame_id="frame_001",
        source_path=str(img1),
        image_size_px=(1280, 960),
        pose=pose1,
        optics=optics,
        detections=[det1, det2],
    )
    engine.ingest_frame(payload1)
    engine.build_heatmap()
    engine.build_prescription_zones(threshold=1.0, min_area_m2=0.5)

    entries = [
        FileEntry(
            path=str(img1),
            size_bytes=1024,
            status=ProcessingStatus.DONE,
            has_xmp=True,
            detections_count=2,
        ),
        FileEntry(
            path=str(img2),
            size_bytes=2048,
            status=ProcessingStatus.DONE,
            has_xmp=True,
            detections_count=0,
        ),
    ]
    detections_by_path = {
        str(img1): [det1, det2],
        str(img2): [],
    }

    gen = ReportGenerator()
    report = gen.build_report(
        entries=entries,
        detections_by_path=detections_by_path,
        map_engine=engine,
        detector_settings=DetectorSettings(classifier_type="c07"),
        source_folder=tmp_path,
    )

    assert report.total_files == 2
    assert report.processed_files == 2
    assert report.frames_with_detections == 1
    assert report.empty_frames == 1
    assert report.total_detections == 2
    assert report.has_georef is True
    assert report.by_class == {"Осот, Бодяк, Латук": 1, "Вьюнок": 1}
    assert report.by_stage == {"2-4 листа": 1, "всходы": 1}
    assert report.class_stage_matrix["Осот, Бодяк, Латук"]["2-4 листа"] == 1

    # Export JSON
    json_out = gen.export_json(report, tmp_path / "out" / "report.json")
    assert json_out.is_file()
    parsed_json = json.loads(json_out.read_text(encoding="utf-8"))
    assert parsed_json["total_detections"] == 2
    assert parsed_json["by_stage"]["2-4 листа"] == 1

    # Export Frames CSV
    frames_csv = gen.export_frames_csv(report, tmp_path / "out" / "frames.csv")
    assert frames_csv.is_file()
    with open(frames_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["file_name"] == "frame_001.jpg"
    assert rows[0]["detections_count"] == "2"

    # Export Detections CSV
    dets_csv = gen.export_detections_csv(report, tmp_path / "out" / "detections.csv")
    assert dets_csv.is_file()
    with open(dets_csv, "r", encoding="utf-8") as f:
        det_rows = list(csv.DictReader(f))
    assert len(det_rows) == 2
    assert det_rows[0]["stage_name"] == "2-4 листа"
    assert float(det_rows[0]["class_prob"]) == 0.87

    # Export HTML with embedded map
    map_bytes = gen.render_map_png_bytes(engine)
    assert map_bytes is not None
    html_out = gen.export_html(report, tmp_path / "out" / "report.html", map_png_bytes=map_bytes)
    html_text = html_out.read_text(encoding="utf-8")
    assert "Осот, Бодяк, Латук" in html_text
    assert "2-4 листа" in html_text
    assert "data:image/png;base64," in html_text

    # Export ZIP bundle
    zip_out = gen.export_zip_bundle(report, tmp_path / "out" / "bundle.zip", map_engine=engine)
    assert zip_out.is_file()
    with zipfile.ZipFile(zip_out, "r") as zf:
        names = set(zf.namelist())
    assert {
        "report.html",
        "report_summary.json",
        "frames_summary.csv",
        "detections_table.csv",
        "weed_map.geojson",
        "weed_map.png",
    }.issubset(names)


def test_build_report_without_georef(tmp_path: Path):
    img = tmp_path / "plain.jpg"
    img.write_bytes(b"plain")

    det = DetectedWeed(
        weed_id="w_plain",
        class_id=0,
        class_name="Щирица (виды)",
        confidence=0.85,
        bbox=BBox(x=0.2, y=0.3, w=0.1, h=0.1),
        stage_name="взрослое растение",
        class_prob=0.93,
    )
    entries = [
        FileEntry(
            path=str(img),
            size_bytes=512,
            status=ProcessingStatus.NO_METADATA,
            has_xmp=False,
            detections_count=1,
        )
    ]

    gen = ReportGenerator()
    report = gen.build_report(
        entries=entries,
        detections_by_path={str(img): [det]},
        map_engine=None,
        source_folder=tmp_path,
    )

    assert report.has_georef is False
    assert report.total_files == 1
    assert report.files_without_xmp == 1
    assert report.total_detections == 1
    assert report.by_class == {"Щирица (виды)": 1}
    assert report.by_stage == {"взрослое растение": 1}

    zip_out = gen.export_zip_bundle(report, tmp_path / "no_georef_bundle.zip", map_engine=None)
    with zipfile.ZipFile(zip_out, "r") as zf:
        names = set(zf.namelist())
    assert "report.html" in names
    assert "report_summary.json" in names
    assert "frames_summary.csv" in names
    assert "detections_table.csv" in names
    assert "weed_map.geojson" not in names
