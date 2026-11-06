"""Tests for RunsController onboard runs management and post-flight processing (offscreen)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QMessageBox

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.controllers.runs_controller import RunsController
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.schemas.onboard_api import DeleteRunResponse, DroneRun
from geo_weed_station.ui.main_window import MainWindow


@pytest.fixture
def runs_setup(tmp_path):
    settings = AppSettings(workdir=tmp_path / "workdir")
    engine = MapEngine(projector=NadirProjector(), settings=settings.map)
    map_ctrl = MapController(engine=engine)
    qset = QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)
    window = MainWindow(qsettings=qset)

    mock_client = None
    mock_worker = None

    ctrl = RunsController(
        window=window,
        map_ctrl=map_ctrl,
        settings=settings,
        client_getter=lambda: mock_client,
        command_worker_getter=lambda: mock_worker,
    )
    return ctrl, map_ctrl, window, tmp_path


def test_runs_controller_update_and_selection(runs_setup, qtbot):
    ctrl, _map_ctrl, _window, _ = runs_setup

    sample_runs = [
        DroneRun(
            run_id="run_001",
            start_time="2026-11-06 10:00:00",
            status="completed",
            directory_path="/data/run_001",
            size_bytes=1024 * 1024 * 15,
            contents=["images"],
        )
    ]

    ctrl._on_runs_loaded(sample_runs)
    assert ctrl.runs_table.table.rowCount() == 1
    assert "run_001" in ctrl.runs_table.table.item(0, 0).text()
    assert "15.00 МБ" in ctrl.runs_table.table.item(0, 3).text()


def test_runs_controller_delete_with_confirmation(runs_setup, monkeypatch, qtbot):
    ctrl, _map_ctrl, _window, _ = runs_setup

    class MockWorker(QObject):
        run_deleted = Signal(object)
        runs_loaded = Signal(object)

        def __init__(self):
            super().__init__()
            self.deleted_id = None

        def delete_run(self, run_id: str):
            self.deleted_id = run_id

        def request_runs(self):
            pass

    mock_w = MockWorker()
    ctrl._command_worker_getter = lambda: mock_w
    ctrl._runs_cache["run_del"] = DroneRun(
        run_id="run_del",
        start_time="2026-11-06 11:00:00",
        status="completed",
        directory_path="/tmp/del",
        size_bytes=1024 * 1024 * 5,
        contents=[],
    )

    # User confirms deletion
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    ctrl.delete("run_del")
    assert mock_w.deleted_id == "run_del"

    # Handle deletion response
    ctrl._on_run_deleted(DeleteRunResponse(status="ok", deleted_run_id="run_del", freed_bytes=1024 * 1024 * 5))
    assert "Освобождено 5.0 МБ" in ctrl._window.status_bar_panel.label_message.text()
