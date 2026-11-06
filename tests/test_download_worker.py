"""Unit and integration tests for DownloadWorker using pytest-qt and mock server."""

from __future__ import annotations

from pathlib import Path

from geo_weed_station.network.download_manager import DownloadManager
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.download_worker import DownloadWorker


def test_download_worker_success(mock_server, tmp_path: Path, qtbot):
    """Test successful download and extraction with progress signals."""
    client = DroneClient(base_url=mock_server.url, timeout_s=2.0)
    dl_dir = tmp_path / "downloads"
    manager = DownloadManager(client=client, target_dir=dl_dir)
    worker = DownloadWorker(download_manager=manager)

    progress_events = []
    finished_dirs = []

    worker.progress.connect(lambda d, t: progress_events.append((d, t)))
    worker.download_finished.connect(finished_dirs.append)

    thread = start_in_thread(worker, "test_dl_thread")

    try:
        runs = client.list_runs()
        run_id = runs[0].run_id

        worker.download(run_id)
        qtbot.waitUntil(lambda: len(finished_dirs) >= 1, timeout=5000)

        assert len(progress_events) > 0
        extracted_path = Path(finished_dirs[0])
        assert extracted_path.exists()
        assert (extracted_path / "summary.json").exists()
        assert (extracted_path / "wp_0000.jpg").exists()

    finally:
        worker.request_stop()
        thread.quit()
        thread.wait(3000)
        assert thread.isFinished()


def test_download_worker_cancel(mock_server, tmp_path: Path, qtbot):
    """Test cancellation of download cleanly aborts without leaving corrupt files."""
    client = DroneClient(base_url=mock_server.url, timeout_s=2.0)
    dl_dir = tmp_path / "downloads_cancel"
    # Set small chunk bytes to ensure cancellation occurs during loop
    manager = DownloadManager(client=client, target_dir=dl_dir, chunk_bytes=10)
    worker = DownloadWorker(download_manager=manager)

    failed_reasons = []
    worker.failed.connect(failed_reasons.append)

    thread = start_in_thread(worker, "test_dl_cancel_thread")

    try:
        runs = client.list_runs()
        run_id = runs[0].run_id

        # Immediately cancel before or right as download starts
        worker.cancel()
        worker.download(run_id)

        qtbot.waitUntil(lambda: len(failed_reasons) >= 1, timeout=3000)
        assert failed_reasons[0] == "cancelled"

        # Check no .part or uncompleted zip left in dir
        part_files = list(dl_dir.glob("*.part"))
        assert len(part_files) == 0

    finally:
        worker.request_stop()
        thread.quit()
        thread.wait(3000)
        assert thread.isFinished()
