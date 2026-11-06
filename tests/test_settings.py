"""Tests for AppSettings, validations, logging setup, and main entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path
import pytest
from pydantic import ValidationError

from geo_weed_station.config.logging_setup import setup_logging
from geo_weed_station.config.settings import (
    AppSettings,
    DetectorSettings,
    MapSettings,
    NetworkSettings,
    get_settings,
)
from main import main


def test_default_settings_load():
    """Verify that AppSettings.load() without args loads valid default_config.json values."""
    cfg = AppSettings.load()

    assert cfg.network.host == "192.168.4.1"
    assert cfg.network.port == 8000
    assert cfg.network.request_timeout_s == 2.0
    assert cfg.network.poll_rate_hz == 1.0
    assert cfg.network.download_chunk_bytes == 65536

    assert cfg.detector.confidence_threshold == 0.25
    assert cfg.detector.nms_iou_threshold == 0.45
    assert cfg.detector.input_size == (640, 640)
    assert cfg.detector.video_frame_skip == 3

    assert cfg.map.heatmap_cell_m == 0.5
    assert cfg.map.critical_density_threshold == 5.0
    assert cfg.map.max_local_extent_m == 3.0
    assert cfg.map.wgs84_origin is None


def test_validation_request_timeout():
    """Verify that request_timeout_s > 2.0 or <= 0 is rejected."""
    with pytest.raises(ValidationError):
        NetworkSettings(request_timeout_s=2.5)

    with pytest.raises(ValidationError):
        NetworkSettings(request_timeout_s=0.0)

    with pytest.raises(ValidationError):
        NetworkSettings(request_timeout_s=-1.0)

    # Valid values
    valid_cfg = NetworkSettings(request_timeout_s=1.5)
    assert valid_cfg.request_timeout_s == 1.5


def test_validation_poll_rate():
    """Verify that poll_rate_hz outside [0.5, 2.0] is rejected."""
    with pytest.raises(ValidationError):
        NetworkSettings(poll_rate_hz=0.2)

    with pytest.raises(ValidationError):
        NetworkSettings(poll_rate_hz=2.5)

    # Boundary valid values
    assert NetworkSettings(poll_rate_hz=0.5).poll_rate_hz == 0.5
    assert NetworkSettings(poll_rate_hz=2.0).poll_rate_hz == 2.0


def test_validation_heatmap_cell():
    """Verify that heatmap_cell_m <= 0 is rejected."""
    with pytest.raises(ValidationError):
        MapSettings(heatmap_cell_m=0.0)

    with pytest.raises(ValidationError):
        MapSettings(heatmap_cell_m=-0.5)

    assert MapSettings(heatmap_cell_m=1.0).heatmap_cell_m == 1.0


def test_validation_wgs84_origin():
    """Verify coordinate bounds for optional wgs84_origin."""
    # Valid
    s = MapSettings(wgs84_origin=(55.75, 37.61))
    assert s.wgs84_origin == (55.75, 37.61)

    # Invalid latitude
    with pytest.raises(ValidationError):
        MapSettings(wgs84_origin=(95.0, 37.61))

    # Invalid longitude
    with pytest.raises(ValidationError):
        MapSettings(wgs84_origin=(55.0, 190.0))


def test_settings_save_and_load_roundtrip(tmp_path: Path):
    """Verify saving configuration to a JSON file and loading it back."""
    custom_cfg = AppSettings.load()
    custom_cfg.network.host = "10.0.0.42"
    custom_cfg.network.port = 9000
    custom_cfg.network.request_timeout_s = 1.2
    custom_cfg.map.heatmap_cell_m = 1.0
    custom_cfg.map.wgs84_origin = (56.0, 38.0)

    save_path = tmp_path / "subdir" / "custom_config.json"
    custom_cfg.save(save_path)

    assert save_path.exists()

    loaded_cfg = AppSettings.load(save_path)
    assert loaded_cfg.network.host == "10.0.0.42"
    assert loaded_cfg.network.port == 9000
    assert loaded_cfg.network.request_timeout_s == 1.2
    assert loaded_cfg.map.heatmap_cell_m == 1.0
    assert loaded_cfg.map.wgs84_origin == (56.0, 38.0)


def test_load_non_existent_file(tmp_path: Path):
    """Verify FileNotFoundError when loading from missing explicit path."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        AppSettings.load(missing)


def test_get_settings_singleton():
    """Verify get_settings returns the singleton and supports reload."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

    s3 = get_settings(reload=True)
    assert isinstance(s3, AppSettings)


def test_setup_logging(tmp_path: Path):
    """Verify logging setup creates file and records logs."""
    log_file = tmp_path / "logs" / "test.log"
    setup_logging(level="DEBUG", log_file=log_file)

    logger = logging.getLogger("geo_weed_station.test")
    test_msg = "Test log message from test_setup_logging"
    logger.info(test_msg)

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_msg in content


def test_main_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify main entrypoint executes without error."""
    # Ensure temporary workdir
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    exit_code = main()
    assert exit_code == 0
