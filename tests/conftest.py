"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

# Ensure Qt runs offscreen during testing if no display is available
if os.environ.get("QT_QPA_PLATFORM") is None and not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from geo_weed_station.config.settings import AppSettings
from tests.mocks.mock_onboard_server import MockConfig, MockServerHandle, run_in_thread


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Ensure a global QApplication exists for offscreen Qt tests."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(["geo-weed-station-test"])
        return app
    except Exception:
        return None


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """Fixture providing a temporary working directory."""
    workdir = tmp_path / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


@pytest.fixture
def settings(tmp_workdir: Path) -> AppSettings:
    """Fixture providing a standard AppSettings instance configured with tmp_workdir."""
    app_settings = AppSettings.load()
    app_settings.workdir = tmp_workdir
    return app_settings


@pytest.fixture
def mock_server() -> Generator[MockServerHandle, None, None]:
    """Fixture running mock onboard drone server in background thread."""
    config = MockConfig(tick_s=0.1, initial_runs=1)
    handle = run_in_thread(config=config, port=0)
    try:
        yield handle
    finally:
        handle.stop()
