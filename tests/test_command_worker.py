"""Unit and integration tests for CommandWorker using pytest-qt and mock server."""

from __future__ import annotations

from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.onboard_api import (
    DeleteRunResponse,
    StartRequest,
    StartResponse,
    StopAction,
    StopResponse,
    SummaryDetections,
)
from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.command_worker import CommandWorker


def test_command_worker_lifecycle(mock_server, qtbot):
    """Test start, stop, runs, summary, and delete commands in background thread."""
    client = DroneClient(base_url=mock_server.url, timeout_s=2.0)
    worker = CommandWorker(client=client)

    started_res = []
    stopped_res = []
    runs_res = []
    summary_res = []
    deleted_res = []
    failed_res = []

    worker.mission_started.connect(started_res.append)
    worker.mission_stopped.connect(stopped_res.append)
    worker.runs_loaded.connect(runs_res.append)
    worker.summary_loaded.connect(summary_res.append)
    worker.run_deleted.connect(deleted_res.append)
    worker.command_failed.connect(lambda cmd, err: failed_res.append((cmd, err)))

    thread = start_in_thread(worker, "test_cmd_thread")

    try:
        # 1. Request runs
        worker.request_runs()
        qtbot.waitUntil(lambda: len(runs_res) >= 1, timeout=3000)
        assert isinstance(runs_res[0], list)
        hist_run_id = runs_res[0][0].run_id

        # 2. Request summary
        worker.request_summary(hist_run_id)
        qtbot.waitUntil(lambda: len(summary_res) >= 1, timeout=3000)
        assert isinstance(summary_res[0], SummaryDetections)

        # 3. Start mission
        req = StartRequest(
            coordinate_system="aruco_local",
            use_nn=True,
            save_policy="all",
            waypoints=[
                Waypoint(x=0.0, y=0.0, z=2.0, yaw=0.0),
                Waypoint(x=1.0, y=1.0, z=2.0, yaw=0.0),
            ],
        )
        worker.start_mission(req)
        qtbot.waitUntil(lambda: len(started_res) >= 1, timeout=3000)
        assert isinstance(started_res[0], StartResponse)

        # 4. Attempt starting again -> should fail with conflict 409
        worker.start_mission(req)
        qtbot.waitUntil(lambda: len(failed_res) >= 1, timeout=3000)
        assert failed_res[0][0] == "start"
        assert "409" in failed_res[0][1]

        # 5. Stop mission
        worker.stop_mission(StopAction.LAND)
        qtbot.waitUntil(lambda: len(stopped_res) >= 1, timeout=3000)
        assert isinstance(stopped_res[0], StopResponse)

        # 6. Delete run
        worker.delete_run(hist_run_id)
        qtbot.waitUntil(lambda: len(deleted_res) >= 1, timeout=3000)
        assert isinstance(deleted_res[0], DeleteRunResponse)

    finally:
        worker.request_stop()
        thread.quit()
        thread.wait(3000)
        assert thread.isFinished()
