"""Tests for MissionControl flight planning and emergency control widget (offscreen)."""

from __future__ import annotations

import pytest

from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.mission import MissionMode, MissionTask, SavePolicy, Waypoint
from geo_weed_station.schemas.onboard_api import StopAction
from geo_weed_station.ui.widgets.mission_control import MissionControl


def test_mission_control_validation_errors(qtbot):
    mc = MissionControl()
    qtbot.addWidget(mc)
    mc.show()

    # 1. Zero area conflict (corners equal)
    mc.spin_x1.setValue(10.0)
    mc.spin_y1.setValue(10.0)
    mc.spin_x2.setValue(10.0)
    mc.spin_y2.setValue(10.0)
    mc.btn_plan.click()
    assert mc.lbl_errors.isVisible()
    assert "нулевую ширину или высоту" in mc.lbl_errors.text()

    # 2. COLLECTION_ONLY + unchecked save_images conflict
    mc.spin_x2.setValue(50.0)
    mc.spin_y2.setValue(50.0)
    mc.combo_mode.setCurrentIndex(0)  # COLLECTION_ONLY
    mc.cb_save_images.setChecked(False)
    mc.cb_save_metadata.setChecked(True)
    mc.btn_plan.click()
    assert mc.lbl_errors.isVisible()
    assert "сохранение снимков обязательно" in mc.lbl_errors.text()


def test_mission_control_plan_and_send_signals(qtbot):
    mc = MissionControl()
    qtbot.addWidget(mc)

    # Valid inputs
    mc.spin_x1.setValue(0.0)
    mc.spin_y1.setValue(0.0)
    mc.spin_x2.setValue(40.0)
    mc.spin_y2.setValue(30.0)
    mc.spin_alt.setValue(12.0)
    mc.cb_save_images.setChecked(True)
    mc.cb_save_metadata.setChecked(True)

    plan_requests = []
    mc.plan_requested.connect(plan_requests.append)

    mc.btn_plan.click()
    assert len(plan_requests) == 1
    assert plan_requests[0].altitude_m == 12.0
    assert plan_requests[0].corner_br == (40.0, 30.0)
    assert not mc.lbl_errors.isVisible()

    # Now set calculated waypoints
    wps = [Waypoint(x=0.0, y=0.0, z=12.0), Waypoint(x=40.0, y=0.0, z=12.0)]
    mc.set_calculated_waypoints(wps)
    assert "2 точек" in mc.lbl_route_summary.text()

    send_tasks = []
    mc.send_mission_requested.connect(send_tasks.append)

    mc.btn_send.click()
    assert len(send_tasks) == 1
    task = send_tasks[0]
    assert isinstance(task, MissionTask)
    assert len(task.waypoints) == 2
    assert task.save_policy == SavePolicy.ALL


def test_mission_control_emergency_stops(qtbot):
    mc = MissionControl()
    qtbot.addWidget(mc)

    stops = []
    mc.stop_requested.connect(stops.append)

    mc.btn_land.click()
    assert stops == [StopAction.LAND]

    mc.btn_rth.click()
    assert stops == [StopAction.LAND, StopAction.RTH]


def test_mission_control_flight_locking(qtbot):
    mc = MissionControl()
    qtbot.addWidget(mc)

    mc.set_flight_locked(True)
    assert not mc.spin_x1.isEnabled()
    assert not mc.spin_alt.isEnabled()
    assert not mc.btn_plan.isEnabled()
    assert not mc.btn_send.isEnabled()

    # Stop buttons MUST remain enabled during flight!
    assert mc.btn_land.isEnabled()
    assert mc.btn_rth.isEnabled()

    mc.set_flight_locked(False)
    assert mc.spin_x1.isEnabled()
    assert mc.spin_alt.isEnabled()
    assert mc.btn_plan.isEnabled()
