"""Command worker executing drone control and data requests outside the GUI thread."""

from __future__ import annotations

from PySide6.QtCore import Signal, Slot

from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.schemas.mission import MissionTask
from geo_weed_station.schemas.onboard_api import (
    StartRequest,
    StopAction,
)
from geo_weed_station.workers.base_worker import ManagedWorker


class CommandWorker(ManagedWorker):
    """Worker handling asynchronous discrete commands (start, stop, list runs, get summary, delete)."""

    mission_started = Signal(object)  # StartResponse
    mission_stopped = Signal(object)  # StopResponse
    runs_loaded = Signal(list)  # List[DroneRun]
    summary_loaded = Signal(object)  # SummaryDetections
    run_deleted = Signal(object)  # DeleteRunResponse
    command_failed = Signal(str, str)  # command_name, error_message

    def __init__(self, client: DroneClient) -> None:
        super().__init__()
        self.client = client

    @Slot(object)
    def start_mission(self, task: MissionTask | StartRequest) -> None:
        """Slot for mission start request."""
        try:
            resp = self.client.start_mission(task)
            self.mission_started.emit(resp)
        except Exception as exc:
            self.command_failed.emit("start", str(exc))

    @Slot()
    @Slot(object)
    def stop_mission(self, action: StopAction = StopAction.LAND) -> None:
        """Slot for mission stop/abort request."""
        try:
            resp = self.client.stop_mission(action)
            self.mission_stopped.emit(resp)
        except Exception as exc:
            self.command_failed.emit("stop", str(exc))

    @Slot()
    def request_runs(self) -> None:
        """Slot for querying runs catalog."""
        try:
            runs = self.client.list_runs()
            self.runs_loaded.emit(runs)
        except Exception as exc:
            self.command_failed.emit("runs", str(exc))

    @Slot(str)
    def request_summary(self, run_id: str) -> None:
        """Slot for fetching mission summary JSON."""
        try:
            summary = self.client.get_summary(run_id)
            self.summary_loaded.emit(summary)
        except Exception as exc:
            self.command_failed.emit("summary", str(exc))

    @Slot(str)
    def delete_run(self, run_id: str) -> None:
        """Slot for deleting run on drone."""
        try:
            resp = self.client.delete_run(run_id)
            self.run_deleted.emit(resp)
        except Exception as exc:
            self.command_failed.emit("delete", str(exc))
