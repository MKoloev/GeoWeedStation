"""Unit tests for SshMissionWorker remote execution, streaming, and abort handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.ssh_mission_worker import SshMissionWorker


class DummyChannel:
    """Mock Paramiko channel delivering chunks and simulating PTY execution."""

    def __init__(self, output_lines: list[str], exit_code: int = 0) -> None:
        self.output_lines = output_lines
        self.exit_code = exit_code
        self._sent_bytes: list[bytes] = []
        self._idx = 0
        self.closed = False

    def get_pty(self, **kwargs) -> None:
        pass

    def set_combine_stderr(self, combine: bool) -> None:
        pass

    def exec_command(self, cmd: str) -> None:
        pass

    def recv_ready(self) -> bool:
        return self._idx < len(self.output_lines)

    def recv(self, nbytes: int) -> bytes:
        if self._idx < len(self.output_lines):
            line = self.output_lines[self._idx]
            self._idx += 1
            return line.encode("utf-8")
        return b""

    def send(self, data: bytes) -> int:
        self._sent_bytes.append(data)
        return len(data)

    def exit_status_ready(self) -> bool:
        return self._idx >= len(self.output_lines)

    def recv_exit_status(self) -> int:
        return self.exit_code

    def close(self) -> None:
        self.closed = True


def test_ssh_mission_worker_success_flow(tmp_path: Path, qtbot):
    """Test full successful mission run: log streaming, progress parsing, and summary download."""
    simulated_logs = [
        "Starting Skyris UAV mission...\n",
        "Created mission run directory: /home/orangepi/tsu/CollectOnboardImage/runs/run_001\n",
        "[WP 1/10] Moving to (x=0, y=0, z=2)\n",
        "[WP 2/10] Moving to (x=1, y=1, z=2)\n",
        "Saved summary: /home/orangepi/tsu/CollectOnboardImage/runs/run_001/summary.json\n",
        "MISSION FINISHED SUCCESSFULLY (Exit code 0)\n",
    ]

    mock_channel = DummyChannel(simulated_logs, exit_code=0)

    mock_transport = MagicMock()
    mock_transport.open_session.return_value = mock_channel

    mock_sftp = MagicMock()

    mock_client = MagicMock()
    mock_client.get_transport.return_value = mock_transport
    mock_client.open_sftp.return_value = mock_sftp

    runs_dir = tmp_path / "workdir" / "runs"
    worker = SshMissionWorker(
        host="10.42.0.1",
        port=22,
        username="orangepi",
        password="orangepi",
        local_runs_dir=runs_dir,
    )

    logs_received: list[str] = []
    progress_updates: list[tuple[int, int]] = []
    finished_results: list[tuple[bool, str]] = []
    summary_downloads: list[str] = []

    worker.log_received.connect(logs_received.append)
    worker.progress_updated.connect(lambda c, t: progress_updates.append((c, t)))
    worker.mission_finished.connect(lambda ok, msg: finished_results.append((ok, msg)))
    worker.summary_downloaded.connect(summary_downloads.append)

    with patch("geo_weed_station.workers.ssh_mission_worker.paramiko.SSHClient", return_value=mock_client):
        thread = start_in_thread(worker, "test_ssh_success")
        try:
            qtbot.waitUntil(lambda: len(finished_results) >= 1, timeout=4000)

            assert finished_results[0][0] is True
            assert "успешно завершена" in finished_results[0][1]

            # Verify progress tracking
            assert (1, 10) in progress_updates
            assert (2, 10) in progress_updates

            # Verify log lines streamed
            assert any("[WP 1/10]" in line for line in logs_received)
            assert any("[WP 2/10]" in line for line in logs_received)

            # Verify SFTP download was triggered
            mock_sftp.get.assert_called()
            assert len(summary_downloads) == 1
            assert "summary.json" in summary_downloads[0]

        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)


def test_ssh_mission_worker_abort_via_sigint(qtbot):
    """Test emergency abort: sends SIGINT 0x03 byte over channel and finishes with aborted status."""
    class RunningChannel(DummyChannel):
        def exit_status_ready(self) -> bool:
            # Stays running until SIGINT is received
            return b"\x03" in self._sent_bytes or self.closed

        def recv_ready(self) -> bool:
            return False

    mock_channel = RunningChannel([], exit_code=130)

    mock_transport = MagicMock()
    mock_transport.open_session.return_value = mock_channel

    mock_client = MagicMock()
    mock_client.get_transport.return_value = mock_transport

    worker = SshMissionWorker(host="10.42.0.1")

    started_events = []
    finished_results: list[tuple[bool, str]] = []
    worker.mission_started.connect(lambda: started_events.append(True))
    worker.mission_finished.connect(lambda ok, msg: finished_results.append((ok, msg)))

    with patch("geo_weed_station.workers.ssh_mission_worker.paramiko.SSHClient", return_value=mock_client):
        thread = start_in_thread(worker, "test_ssh_abort")
        try:
            # Wait for channel to be initialized and mission to start
            qtbot.waitUntil(lambda: len(started_events) >= 1, timeout=3000)

            # Trigger emergency abort
            worker.abort_mission()
            qtbot.waitUntil(lambda: len(finished_results) >= 1, timeout=4000)

            assert b"\x03" in mock_channel._sent_bytes
            assert finished_results[0][0] is False
            assert "прервана" in finished_results[0][1] or "130" in finished_results[0][1]

        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)


def test_ssh_mission_worker_connection_failure(qtbot):
    """Test connection timeout or network failure handling without crashing."""
    mock_client = MagicMock()
    mock_client.connect.side_effect = TimeoutError("Timed out connecting to 10.42.0.1:22")

    worker = SshMissionWorker(host="10.42.0.1")

    failed_messages: list[str] = []
    finished_results: list[tuple[bool, str]] = []

    worker.failed.connect(failed_messages.append)
    worker.mission_finished.connect(lambda ok, msg: finished_results.append((ok, msg)))

    with patch("geo_weed_station.workers.ssh_mission_worker.paramiko.SSHClient", return_value=mock_client):
        thread = start_in_thread(worker, "test_ssh_fail")
        try:
            qtbot.waitUntil(lambda: len(finished_results) >= 1, timeout=3000)

            assert finished_results[0][0] is False
            assert len(failed_messages) >= 1
            assert "Timed out" in failed_messages[0]

        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)


def test_default_ssh_command_has_optimal_detector_params():
    """Verify DEFAULT_SSH_COMMAND passes the tuned RKNN detector and transfer parameters."""
    from geo_weed_station.workers.ssh_mission_worker import DEFAULT_SSH_COMMAND

    assert "--gsd 0.00022" in DEFAULT_SSH_COMMAND
    assert "--iou-thresh 0.25" in DEFAULT_SSH_COMMAND
    assert "--conf-thresh 0.6" in DEFAULT_SSH_COMMAND
    assert "--containment-thresh 0.7" in DEFAULT_SSH_COMMAND
    assert "--transfer-max-dim 1600" in DEFAULT_SSH_COMMAND
    assert "--transfer-quality 85" in DEFAULT_SSH_COMMAND


def test_ssh_mission_worker_downloads_full_run_tree_sftp(qtbot, tmp_path: Path):
    """Verify SshMissionWorker recursively downloads local_aruco and preserve_original artifacts."""
    import stat
    from types import SimpleNamespace

    remote_root = "/home/orangepi/tsu/CollectOnboardImage/runs/run_20261111_120000"
    remote_fs: dict[str, bytes | list[SimpleNamespace]] = {
        remote_root: [
            SimpleNamespace(filename=".", st_mode=stat.S_IFDIR),
            SimpleNamespace(filename="..", st_mode=stat.S_IFDIR),
            SimpleNamespace(filename="summary.json", st_mode=stat.S_IFREG),
            SimpleNamespace(filename="summary.json.tmp", st_mode=stat.S_IFREG),
            SimpleNamespace(filename="manifest.json", st_mode=stat.S_IFREG),
            SimpleNamespace(filename="local_aruco", st_mode=stat.S_IFDIR),
            SimpleNamespace(filename="preserve_original", st_mode=stat.S_IFDIR),
        ],
        f"{remote_root}/local_aruco": [
            SimpleNamespace(filename="capture_000.jpg", st_mode=stat.S_IFREG),
            SimpleNamespace(filename="annotations.json", st_mode=stat.S_IFREG),
            SimpleNamespace(filename="summary.json", st_mode=stat.S_IFREG),
        ],
        f"{remote_root}/preserve_original": [
            SimpleNamespace(filename="capture_000.jpg", st_mode=stat.S_IFREG),
            SimpleNamespace(filename="annotations.json", st_mode=stat.S_IFREG),
            SimpleNamespace(filename="summary.json", st_mode=stat.S_IFREG),
        ],
        f"{remote_root}/summary.json": b'{"run_id":"run_20261111_120000","coordinate_system":"aruco_local","detections_count":0,"items":[]}',
        f"{remote_root}/manifest.json": b'{"status":"completed"}',
        f"{remote_root}/local_aruco/capture_000.jpg": b"LOCAL_JPG",
        f"{remote_root}/local_aruco/annotations.json": b'{"format":"geoweed_annotations","frames":{}}',
        f"{remote_root}/local_aruco/summary.json": b'{"run_id":"run_20261111_120000","coordinate_system":"aruco_local","detections_count":0,"items":[]}',
        f"{remote_root}/preserve_original/capture_000.jpg": b"ORIG_JPG",
        f"{remote_root}/preserve_original/annotations.json": b'{"format":"geoweed_annotations","frames":{}}',
        f"{remote_root}/preserve_original/summary.json": b'{"run_id":"run_20261111_120000","coordinate_system":"wgs84","detections_count":0,"items":[]}',
    }

    mock_sftp = MagicMock()
    mock_sftp.listdir_attr.side_effect = lambda rdir: remote_fs[rdir]

    def _fake_get(remote_path: str, local_path: str) -> None:
        if remote_path not in remote_fs or not isinstance(remote_fs[remote_path], bytes):
            raise FileNotFoundError(remote_path)
        Path(local_path).write_bytes(remote_fs[remote_path])  # type: ignore[arg-type]

    mock_sftp.get.side_effect = _fake_get

    output_lines = [
        f"Created mission run directory: {remote_root}\n",
        f"Saved summary: {remote_root}/summary.json\n",
    ]
    mock_channel = DummyChannel(output_lines, exit_code=0)
    mock_transport = MagicMock()
    mock_transport.open_session.return_value = mock_channel

    mock_client = MagicMock()
    mock_client.get_transport.return_value = mock_transport
    mock_client.open_sftp.return_value = mock_sftp

    worker = SshMissionWorker(host="10.42.0.1", local_runs_dir=tmp_path)
    downloaded_summaries: list[str] = []
    downloaded_runs: list[str] = []
    finished_results: list[tuple[bool, str]] = []
    worker.summary_downloaded.connect(downloaded_summaries.append)
    worker.run_downloaded.connect(downloaded_runs.append)
    worker.mission_finished.connect(lambda ok, msg: finished_results.append((ok, msg)))

    with patch("geo_weed_station.workers.ssh_mission_worker.paramiko.SSHClient", return_value=mock_client):
        thread = start_in_thread(worker, "test_ssh_recursive_sftp")
        try:
            qtbot.waitUntil(lambda: len(finished_results) >= 1, timeout=3000)
        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)

    local_run = tmp_path / "run_20261111_120000"
    assert finished_results == [(True, "Миссия успешно завершена (код 0)")]
    assert downloaded_runs == [str(local_run)]
    assert downloaded_summaries == [str(local_run / "summary.json")]

    assert (local_run / "summary.json").read_bytes() == remote_fs[f"{remote_root}/summary.json"]
    assert (local_run / "manifest.json").read_bytes() == remote_fs[f"{remote_root}/manifest.json"]
    assert not (local_run / "summary.json.tmp").exists()
    assert not list(local_run.rglob("*.part"))

    assert (local_run / "local_aruco" / "capture_000.jpg").read_bytes() == b"LOCAL_JPG"
    assert (local_run / "local_aruco" / "annotations.json").is_file()
    assert (local_run / "local_aruco" / "summary.json").is_file()
    assert (local_run / "preserve_original" / "capture_000.jpg").read_bytes() == b"ORIG_JPG"
    assert (local_run / "preserve_original" / "annotations.json").is_file()
    assert (local_run / "preserve_original" / "summary.json").is_file()


def test_ssh_mission_worker_sftp_missing_summary_no_empty_file(qtbot, tmp_path: Path):
    """Ensure failed SFTP summary.json download does not leave an empty 0-byte local file."""
    remote_root = "/home/orangepi/tsu/CollectOnboardImage/runs/run_20261111_130000"

    mock_sftp = MagicMock()
    mock_sftp.listdir_attr.return_value = []
    mock_sftp.get.side_effect = FileNotFoundError("No such remote file")

    mock_channel = DummyChannel(
        [f"Created mission run directory: {remote_root}\n"],
        exit_code=1,
    )
    mock_transport = MagicMock()
    mock_transport.open_session.return_value = mock_channel
    mock_client = MagicMock()
    mock_client.get_transport.return_value = mock_transport
    mock_client.open_sftp.return_value = mock_sftp

    worker = SshMissionWorker(host="10.42.0.1", local_runs_dir=tmp_path)
    downloaded_summaries: list[str] = []
    finished_results: list[tuple[bool, str]] = []
    worker.summary_downloaded.connect(downloaded_summaries.append)
    worker.mission_finished.connect(lambda ok, msg: finished_results.append((ok, msg)))

    with patch("geo_weed_station.workers.ssh_mission_worker.paramiko.SSHClient", return_value=mock_client):
        thread = start_in_thread(worker, "test_ssh_missing_summary")
        try:
            qtbot.waitUntil(lambda: len(finished_results) >= 1, timeout=3000)
        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)

    local_run = tmp_path / "run_20261111_130000"
    assert downloaded_summaries == []
    assert not (local_run / "summary.json").exists()
    assert not (local_run / "summary.json.part").exists()


def test_ssh_mission_worker_logs_progress_and_skips_fullres(qtbot, tmp_path: Path):
    """Verify SFTP download logs per-file progress and skips any fullres artifacts."""
    import stat
    from types import SimpleNamespace

    remote_root = "/home/orangepi/tsu/CollectOnboardImage/runs/run_20261111_140000"
    remote_fs: dict[str, bytes | list[SimpleNamespace]] = {
        remote_root: [
            SimpleNamespace(filename="preserve_original", st_mode=stat.S_IFDIR),
            SimpleNamespace(filename="local_aruco", st_mode=stat.S_IFDIR),
            SimpleNamespace(filename="fullres", st_mode=stat.S_IFDIR),
            SimpleNamespace(filename="summary.json", st_mode=stat.S_IFREG, st_size=128),
        ],
        f"{remote_root}/local_aruco": [
            SimpleNamespace(filename="capture_000.jpg", st_mode=stat.S_IFREG, st_size=512),
            SimpleNamespace(filename="annotations.json", st_mode=stat.S_IFREG, st_size=64),
        ],
        f"{remote_root}/preserve_original": [
            SimpleNamespace(filename="capture_000.jpg", st_mode=stat.S_IFREG, st_size=512),
            SimpleNamespace(filename="annotations.json", st_mode=stat.S_IFREG, st_size=64),
        ],
        f"{remote_root}/fullres": [
            SimpleNamespace(filename="raw_50mp.jpg", st_mode=stat.S_IFREG, st_size=19_000_000),
        ],
        f"{remote_root}/summary.json": b'{"run_id":"run_20261111_140000","coordinate_system":"aruco_local","detections_count":0,"items":[]}',
        f"{remote_root}/local_aruco/capture_000.jpg": b"A" * 512,
        f"{remote_root}/local_aruco/annotations.json": b"{}",
        f"{remote_root}/preserve_original/capture_000.jpg": b"B" * 512,
        f"{remote_root}/preserve_original/annotations.json": b"{}",
        f"{remote_root}/fullres/raw_50mp.jpg": b"C" * 100,
    }

    mock_sftp = MagicMock()
    mock_sftp.listdir_attr.side_effect = lambda rdir: remote_fs[rdir]

    def _fake_get(remote_path: str, local_path: str) -> None:
        if remote_path not in remote_fs or not isinstance(remote_fs[remote_path], bytes):
            raise FileNotFoundError(remote_path)
        Path(local_path).write_bytes(remote_fs[remote_path])  # type: ignore[arg-type]

    mock_sftp.get.side_effect = _fake_get

    mock_channel = DummyChannel([f"Created mission run directory: {remote_root}\n"], exit_code=0)
    mock_transport = MagicMock()
    mock_transport.open_session.return_value = mock_channel
    mock_client = MagicMock()
    mock_client.get_transport.return_value = mock_transport
    mock_client.open_sftp.return_value = mock_sftp

    worker = SshMissionWorker(host="10.42.0.1", local_runs_dir=tmp_path)
    logs: list[str] = []
    finished: list[tuple[bool, str]] = []
    worker.log_received.connect(logs.append)
    worker.mission_finished.connect(lambda ok, msg: finished.append((ok, msg)))

    with patch("geo_weed_station.workers.ssh_mission_worker.paramiko.SSHClient", return_value=mock_client):
        thread = start_in_thread(worker, "test_ssh_progress_skip_fullres")
        try:
            qtbot.waitUntil(lambda: len(finished) >= 1, timeout=3000)
        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)

    local_run = tmp_path / "run_20261111_140000"
    assert (local_run / "local_aruco" / "capture_000.jpg").is_file()
    assert (local_run / "preserve_original" / "capture_000.jpg").is_file()
    assert not (local_run / "fullres").exists()
    assert any("[1/5]" in line for line in logs)
    assert any("МБ за" in line for line in logs)
