"""Tests for GIS GeoJSON export with prescription zones and summary reports."""

from __future__ import annotations

import json
from pathlib import Path

from geo_weed_station.core.map_engine.exporter import MapExporter
from geo_weed_station.core.map_engine.weed_map import WeedMap
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import (
    PrescriptionZone,
    SummaryReport,
    WeedPoint,
)


def test_export_geojson_with_prescription_zones(tmp_path: Path):
    exporter = MapExporter()
    wm = WeedMap(coord_system=CoordinateSystem.LOCAL_ARUCO)

    pt = WeedPoint(
        weed_id="w1",
        class_name="weed_dicot",
        confidence=0.9,
        x=2.0,
        y=3.0,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="f1",
        best_frame_id="f1",
        duplicate_count=2,
    )
    wm.add_points([pt])

    zone = PrescriptionZone(
        zone_id="zone_01",
        polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
        area_m2=25.0,
        mean_density_per_m2=4.0,
        weed_count=100,
        dominant_class="weed_dicot",
        treatment_recommendation="Full Spray",
    )
    wm.prescription_zones = [zone]

    target = tmp_path / "survey.geojson"
    exporter.to_geojson(wm, target, include_zones=True)

    assert target.exists()
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["type"] == "FeatureCollection"
    assert len(doc["features"]) == 2

    # Check zone feature
    zone_feat = next(f for f in doc["features"] if f["geometry"]["type"] == "Polygon")
    assert zone_feat["properties"]["zone_id"] == "zone_01"
    assert zone_feat["properties"]["area_m2"] == 25.0
    assert zone_feat["properties"]["weed_count"] == 100

    # Check point feature
    pt_feat = next(f for f in doc["features"] if f["geometry"]["type"] == "Point")
    assert pt_feat["properties"]["weed_id"] == "w1"
    assert pt_feat["properties"]["best_frame_id"] == "f1"
    assert pt_feat["properties"]["duplicate_count"] == 2


def test_export_summary_json_and_csv(tmp_path: Path):
    exporter = MapExporter()
    zone = PrescriptionZone(
        zone_id="zone_01",
        polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
        area_m2=25.0,
        mean_density_per_m2=4.0,
        weed_count=100,
        dominant_class="weed_dicot",
        treatment_recommendation="Full Spray",
    )
    report = SummaryReport(
        total_detections=150,
        unique_weeds=100,
        deduplication_ratio=1.5,
        surveyed_area_m2=500.0,
        mean_density_per_m2=0.2,
        critical_area_percent=5.0,
        by_class={"weed_dicot": 1},
        prescription_zones_count=1,
        frames_count=10,
        zones=[zone],
    )

    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"

    exporter.to_summary_json(report, json_path)
    exporter.to_summary_csv(report, csv_path)

    assert json_path.exists()
    assert csv_path.exists()

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["total_detections"] == 150
    assert json_data["unique_weeds"] == 100
    assert len(json_data["zones"]) == 1

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "total_detections,150" in csv_text
    assert "unique_weeds,100" in csv_text
    assert "zone_01,25.0,4.0,100,weed_dicot,Full Spray" in csv_text
