"""Unit tests for MapExporter: CSV, GeoJSON, and PNG raster exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from PIL import Image
import pytest

from geo_weed_station.core.map_engine.exporter import MapExporter
from geo_weed_station.core.map_engine.heatmap import CountHeatmap
from geo_weed_station.core.map_engine.weed_map import WeedMap
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import PngExportOptions, WeedPoint


@pytest.fixture
def sample_weed_map() -> WeedMap:
    wm = WeedMap(coord_system=CoordinateSystem.LOCAL_ARUCO)
    pts = [
        WeedPoint(
            weed_id="weed_01",
            class_name="weed_dicot",
            confidence=0.92,
            x=1.25,
            y=0.75,
            coord_system=CoordinateSystem.LOCAL_ARUCO,
            source_frame="frame_10",
        ),
        WeedPoint(
            weed_id="weed_02",
            class_name="weed_monocot",
            confidence=0.87,
            x=2.50,
            y=1.80,
            coord_system=CoordinateSystem.LOCAL_ARUCO,
            source_frame="frame_10",
        ),
    ]
    wm.add_points(pts)
    return wm


def test_export_to_csv(sample_weed_map: WeedMap, tmp_path: Path):
    """Verify CSV export format and column headers."""
    csv_path = tmp_path / "weeds.csv"
    exporter = MapExporter()
    exporter.to_csv(sample_weed_map, csv_path)

    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    row0 = reader[0]
    expected_headers = [
        "weed_id",
        "class_name",
        "confidence",
        "coord_x",
        "coord_y",
        "coord_system",
        "source_frame",
    ]
    assert list(row0.keys()) == expected_headers
    assert row0["weed_id"] == "weed_01"
    assert row0["class_name"] == "weed_dicot"
    assert float(row0["coord_x"]) == pytest.approx(1.25)
    assert float(row0["coord_y"]) == pytest.approx(0.75)


def test_export_to_geojson(sample_weed_map: WeedMap, tmp_path: Path):
    """Verify GeoJSON FeatureCollection structure and properties."""
    json_path = tmp_path / "weeds.geojson"
    exporter = MapExporter()
    exporter.to_geojson(sample_weed_map, json_path)

    assert json_path.exists()
    doc = json.loads(json_path.read_text(encoding="utf-8"))

    assert doc["type"] == "FeatureCollection"
    assert "crs" in doc
    assert len(doc["features"]) == 2

    f0 = doc["features"][0]
    assert f0["type"] == "Feature"
    assert f0["geometry"]["type"] == "Point"
    assert f0["geometry"]["coordinates"] == [1.25, 0.75]
    assert f0["properties"]["weed_id"] == "weed_01"
    assert f0["properties"]["class_name"] == "weed_dicot"


def test_export_to_png_white_and_transparent(sample_weed_map: WeedMap, tmp_path: Path):
    """Verify rendering map to PNG with white and transparent backgrounds."""
    exporter = MapExporter()
    grid = CountHeatmap().compute(sample_weed_map.points, cell_m=0.5, bounds=sample_weed_map.bounds())

    # 1. White background
    png_white = tmp_path / "map_white.png"
    exporter.to_png(
        sample_weed_map,
        grid,
        png_white,
        options=PngExportOptions(background="white", show_grid=True, show_legend=True),
    )
    assert png_white.exists()
    img_w = Image.open(png_white)
    assert img_w.format == "PNG"
    assert img_w.size == (1200, 900)

    # 2. Transparent background
    png_trans = tmp_path / "map_trans.png"
    exporter.to_png(
        sample_weed_map,
        grid,
        png_trans,
        options=PngExportOptions(background="transparent", show_grid=False, show_legend=False),
    )
    assert png_trans.exists()
    img_t = Image.open(png_trans)
    assert img_t.mode == "RGBA"
    # Corner pixel should be completely transparent (alpha = 0)
    assert img_t.getpixel((5, 5))[3] == 0
