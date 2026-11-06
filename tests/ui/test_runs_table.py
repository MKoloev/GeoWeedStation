"""Tests for RunsTable, TelemetryPanel and MapInfoPanel widgets (offscreen)."""

from __future__ import annotations

import numpy as np
import pytest

from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import MapStatistics, WeedPoint
from geo_weed_station.schemas.onboard_api import DroneRun
from geo_weed_station.schemas.telemetry import DronePose, DroneStatus, FlightState
from geo_weed_station.ui.widgets.map_info_panel import MapInfoPanel
from geo_weed_station.ui.widgets.runs_table import RunsTable
from geo_weed_station.ui.widgets.telemetry_panel import TelemetryPanel


def _sample_runs() -> list[DroneRun]:
    return [
        DroneRun(
            run_id="run_20261106_001",
            start_time="2026-11-06 08:30:00",
            status="completed",
            directory_path="/data/runs/run_001",
            size_bytes=1024 * 1024 * 45,  # 45 MB
            contents=["images", "metadata.json"],
        ),
        DroneRun(
            run_id="run_20261106_002",
            start_time="2026-11-06 09:15:00",
            status="aborted",
            directory_path="/data/runs/run_002",
            size_bytes=1024 * 512,  # 512 KB
            contents=["metadata.json"],
        ),
    ]


def test_runs_table_population_and_actions(qtbot):
    rt = RunsTable()
    qtbot.addWidget(rt)
    rt.resize(600, 300)
    rt.show()

    runs = _sample_runs()
    rt.set_runs(runs)

    assert rt.table.rowCount() == 2
    assert "run_20261106_001" in rt.table.item(0, 0).text()
    assert "45.00 МБ" in rt.table.item(0, 3).text()

    # Buttons initially disabled without selection
    assert not rt.btn_download.isEnabled()

    # Select row 0
    rt.table.selectRow(0)
    assert rt.btn_download.isEnabled()
    assert rt.selected_run_id() == "run_20261106_001"

    # Test signals
    downloads, deletes, processes, refreshes = [], [], [], []
    rt.download_requested.connect(downloads.append)
    rt.delete_requested.connect(deletes.append)
    rt.process_requested.connect(processes.append)
    rt.refresh_requested.connect(lambda: refreshes.append(True))

    rt.btn_download.click()
    assert downloads == ["run_20261106_001"]

    rt.btn_process.click()
    assert processes == ["run_20261106_001"]

    rt.btn_delete.click()
    assert deletes == ["run_20261106_001"]

    rt.btn_refresh.click()
    assert refreshes == [True]


def test_telemetry_panel_updates(qtbot):
    panel = TelemetryPanel()
    qtbot.addWidget(panel)

    status = DroneStatus(
        preset="dji_m300",
        flight_state=FlightState.FLYING_TO_WP,
        current_wp_index=4,
        total_waypoints=10,
        current_pose=DronePose(
            x=15.2,
            y=28.4,
            z=12.0,
            roll=0.0,
            pitch=0.0,
            yaw=45.0,
            coordinate_system="aruco_local",
            timestamp=1000.0,
        ),
        captures_count=18,
        is_data_ready=True,
        active_run_id="run_current_42",
    )

    panel.update_status(status)

    assert "Полет к точке" in panel.lbl_state.text()
    assert "4 из 10" in panel.lbl_progress.text()
    assert panel.bar_progress.value() == 40
    assert "15.20" in panel.lbl_pos_xy.text()
    assert "12.00" in panel.lbl_alt_z.text()
    assert "45.0°" in panel.lbl_yaw.text()
    assert panel.lbl_captures.text() == "18"
    assert panel.lbl_run_id.text() == "run_current_42"


def test_map_info_panel(qtbot):
    info = MapInfoPanel()
    qtbot.addWidget(info)
    info.show()

    stats = MapStatistics(
        total_weeds=84,
        by_class={"Ambrosia": 50, "Cirsium": 34},
        surveyed_area_m2=1200.0,
        mean_density_per_m2=0.07,
        critical_area_percent=12.5,
    )
    info.set_statistics(stats)

    assert info.lbl_total_weeds.text() == "84"
    assert "1200.0" in info.lbl_area.text()
    assert info.table_classes.rowCount() == 2

    # Point popup
    pt = WeedPoint(
        weed_id="pt-42",
        class_name="Ambrosia",
        confidence=0.91,
        x=12.5,
        y=8.3,
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        source_frame="DJI_0042.JPG",
    )
    thumb = np.zeros((80, 80, 3), dtype=np.uint8)
    info.show_point(pt, thumb)

    assert info.point_group.isVisible()
    assert info.lbl_point_id.text() == "pt-42"
    assert "91.0" in info.lbl_point_conf.text()
    assert not info.lbl_thumbnail.pixmap().isNull()
