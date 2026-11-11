"""Background worker executing Skyris UAV collection mission over SSH/SFTP via Paramiko."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from geo_weed_station.workers.base_worker import ManagedWorker

logger = logging.getLogger(__name__)

try:
    import paramiko
except ImportError:  # pragma: no cover
    paramiko = None  # type: ignore[assignment]


DEFAULT_SSH_HOST = "10.42.0.1"
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_USER = "orangepi"
DEFAULT_SSH_PASSWORD = "orangepi"
DEFAULT_REMOTE_WORKDIR = "/home/orangepi/tsu/CollectOnboardImage"

DEFAULT_SSH_COMMAND = (
    'bash -l -c "cd /home/orangepi/tsu/CollectOnboardImage && python3 main.py '
    '--input-dir data_image/06_geosaero_corner_25_wgs84_10pct_aruc_ex '
    '--model models/yolox_m_rk3588_int8.rknn '
    '--waypoints waypoints/zigzag_skyris.json '
    '--geotag-mode drone_telemetry '
    '--altitude 2.0 '
    '--speed 0.5 '
    '--hover-delay 1.0 '
    '--nav-timeout 20.0 '
    '--conf-thresh 0.35 '
    '--gsd 0.00027 '
    '--core-mask 0_1_2"'
)


def resolve_remote_path(remote_path: str, command: str) -> str:
    """Resolve a relative remote path against the `cd <dir>` target in an SSH command."""
    import posixpath
    import re

    if remote_path.startswith("/"):
        return remote_path
    match = re.search(r'(?:^|[;&\s"\'])cd\s+([^\s;&"\']+)', command)
    workdir = match.group(1).rstrip("/") if match else DEFAULT_REMOTE_WORKDIR
    return posixpath.normpath(posixpath.join(workdir, remote_path))

_WP_REGEX = re.compile(r"\[WP\s+(\d+)\s*/\s*(\d+)\]", re.IGNORECASE)
_SUMMARY_REGEX = re.compile(r"Saved summary:\s*(\S+)", re.IGNORECASE)
_RUN_DIR_REGEX = re.compile(r"Created mission run directory:\s*(\S+)", re.IGNORECASE)


class SshMissionWorker(ManagedWorker):
    """Executes remote flight mission on Skyris Orange Pi SBC over SSH.
    
    Streams real-time console logs, tracks waypoint progress, supports safe abort
    via SIGINT (Ctrl+C), and optionally downloads mission summary artifacts over SFTP.
    """

    log_received = Signal(str)
    progress_updated = Signal(int, int)  # current_wp, total_wps
    mission_started = Signal()
    mission_finished = Signal(bool, str)  # success, message
    summary_downloaded = Signal(str)  # local summary.json path

    def __init__(
        self,
        host: str = DEFAULT_SSH_HOST,
        port: int = DEFAULT_SSH_PORT,
        username: str = DEFAULT_SSH_USER,
        password: str = DEFAULT_SSH_PASSWORD,
        command: str = DEFAULT_SSH_COMMAND,
        connect_timeout_s: float = 10.0,
        download_artifacts: bool = True,
        local_runs_dir: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.command = command
        self.connect_timeout_s = connect_timeout_s
        self.download_artifacts = download_artifacts
        self.local_runs_dir = Path(local_runs_dir) if local_runs_dir is not None else None

        self._ssh_client: Any | None = None
        self._channel: Any | None = None
        self._remote_summary_path: str | None = None
        self._remote_run_dir: str | None = None
        self._exit_status: int | None = None

    @Slot()
    def start_work(self) -> None:
        """Connect to Skyris SBC over SSH, launch mission command, and stream logs."""
        self._running = True
        self.started_ok.emit()

        if paramiko is None:
            err = "Библиотека paramiko не найдена в окружении Python!"
            logger.error(err)
            self.log_received.emit(f"❌ {err}")
            self.failed.emit(err)
            self.mission_finished.emit(False, err)
            self.finished.emit()
            return

        self.log_received.emit(f"🔌 Подключение к борту Skyris ({self.host}:{self.port}) по SSH...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh_client = client

        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=self.connect_timeout_s,
                banner_timeout=self.connect_timeout_s,
                auth_timeout=self.connect_timeout_s,
            )
            self.log_received.emit(f"✅ SSH соединение установлено с {self.host}:{self.port}")
            logger.info("SSH connection established with %s:%d", self.host, self.port)

            transport = client.get_transport()
            if transport is None:
                raise RuntimeError("Не удалось открыть SSH transport")

            channel = transport.open_session()
            channel.get_pty(term="xterm", width=120, height=40)
            channel.set_combine_stderr(True)
            self._channel = channel

            self.log_received.emit(f"🚀 Запуск команды миссии на дроне:\n{self.command}\n")
            channel.exec_command(self.command)
            self.mission_started.emit()

            buffer = ""
            while not self.is_stop_requested or channel.recv_ready():
                if channel.exit_status_ready() and not channel.recv_ready():
                    break

                if channel.recv_ready():
                    try:
                        chunk = channel.recv(4096)
                    except OSError as exc:
                        logger.debug("Channel recv exception: %s", exc)
                        break

                    if not chunk:
                        break

                    text = chunk.decode("utf-8", errors="replace")
                    buffer += text
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        clean_line = line.rstrip("\r")
                        self.log_received.emit(clean_line)
                        self._parse_line(clean_line)
                else:
                    time.sleep(0.04)

            # Flush remaining buffer if any
            if buffer:
                for line in buffer.splitlines():
                    clean_line = line.rstrip("\r")
                    if clean_line:
                        self.log_received.emit(clean_line)
                        self._parse_line(clean_line)

            # Determine exit status
            if channel.exit_status_ready():
                self._exit_status = channel.recv_exit_status()
            else:
                self._exit_status = -1

            logger.info("Remote process terminated with exit code %s", self._exit_status)
            self.log_received.emit(f"\n[SSH] Процесс завершился с кодом: {self._exit_status}")

            # Optional artifact download over SFTP before emitting mission_finished
            if self.download_artifacts and self.local_runs_dir is not None:
                self._download_artifacts_sftp(client)

            # Check success / abort
            if self._exit_status == 0:
                msg = "Миссия успешно завершена (код 0)"
                self.log_received.emit(f"🎉 {msg}")
                self.mission_finished.emit(True, msg)
            elif self.is_stop_requested or self._exit_status == 130:
                msg = "Миссия прервана оператором (выполнена посадка, код 130)"
                self.log_received.emit(f"🛑 {msg}")
                self.mission_finished.emit(False, msg)
            else:
                msg = f"Миссия завершилась с ошибкой (код {self._exit_status})"
                self.log_received.emit(f"⚠️ {msg}")
                self.mission_finished.emit(False, msg)

        except Exception as exc:
            err_msg = f"Ошибка выполнения SSH миссии: {exc}"
            logger.exception("SSH mission error")
            self.log_received.emit(f"\n❌ {err_msg}")
            self.failed.emit(str(exc))
            self.mission_finished.emit(False, str(exc))
        finally:
            self._cleanup()
            self._running = False
            self.finished.emit()

    @Slot()
    def abort_mission(self) -> None:
        """Send SIGINT (Ctrl+C) to the remote process to trigger safe controller.land()."""
        logger.warning("Operator requested emergency abort for SSH mission")
        self.request_stop()
        if self._channel is not None and not self._channel.closed:
            try:
                self.log_received.emit("\n⚠️ [GCS] Отправлен сигнал экстренной посадки SIGINT (Ctrl+C)...")
                # Send ASCII 0x03 (ETX / Ctrl+C) over PTY
                self._channel.send(b"\x03")
            except OSError as exc:
                logger.warning("Failed to send SIGINT over SSH channel: %s", exc)

    def _parse_line(self, line: str) -> None:
        """Extract waypoint progress and run directory metadata from console output."""
        # Check waypoint progress [WP 2/10]
        wp_match = _WP_REGEX.search(line)
        if wp_match:
            try:
                curr = int(wp_match.group(1))
                total = int(wp_match.group(2))
                self.progress_updated.emit(curr, total)
            except (ValueError, IndexError):
                pass

        # Check run directory
        run_match = _RUN_DIR_REGEX.search(line)
        if run_match:
            self._remote_run_dir = run_match.group(1).strip()
            logger.info("Detected remote run directory: %s", self._remote_run_dir)

        # Check saved summary
        sum_match = _SUMMARY_REGEX.search(line)
        if sum_match:
            self._remote_summary_path = sum_match.group(1).strip()
            logger.info("Detected remote summary path: %s", self._remote_summary_path)

    def _download_artifacts_sftp(self, client: Any) -> None:
        """Download summary.json and manifest.json from drone SBC via SFTP."""
        try:
            self.log_received.emit("📥 Проверка и загрузка артефактов полета (SFTP)...")
            sftp = client.open_sftp()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open SFTP session: %s", exc)
            self.log_received.emit(f"⚠️ Не удалось открыть SFTP сессию: {exc}")
            return

        try:
            remote_summary = self._remote_summary_path
            # If not explicitly captured, check remote_run_dir
            if remote_summary is None and self._remote_run_dir is not None:
                remote_summary = f"{self._remote_run_dir}/summary.json"
            if remote_summary is not None:
                remote_summary = resolve_remote_path(remote_summary, self.command)

            if remote_summary is None:
                # Try finding the latest run directory on remote
                remote_base = "/home/orangepi/tsu/CollectOnboardImage/runs"
                try:
                    entries = sftp.listdir(remote_base)
                    run_dirs = sorted([e for e in entries if e.startswith("run_")])
                    if run_dirs:
                        remote_summary = f"{remote_base}/{run_dirs[-1]}/summary.json"
                except OSError as err:
                    logger.debug("Could not list remote runs: %s", err)

            if remote_summary is None:
                self.log_received.emit("ℹ️ Артефакты summary.json на борту не найдены")
                return

            # Prepare local destination
            assert self.local_runs_dir is not None
            run_name = Path(remote_summary).parent.name
            local_target_dir = self.local_runs_dir / run_name
            local_target_dir.mkdir(parents=True, exist_ok=True)
            local_summary_file = local_target_dir / "summary.json"

            self.log_received.emit(f"📥 Скачивание: {remote_summary} -> {local_summary_file}")
            sftp.get(remote_summary, str(local_summary_file))

            # Also attempt manifest.json
            remote_manifest = f"{Path(remote_summary).parent}/manifest.json"
            local_manifest = local_target_dir / "manifest.json"
            try:
                sftp.get(remote_manifest, str(local_manifest))
            except OSError as err:
                logger.debug("Could not get manifest: %s", err)

            self.log_received.emit(f"✅ Артефакты миссии успешно сохранены: {local_summary_file}")
            self.summary_downloaded.emit(str(local_summary_file))
        except OSError as exc:
            logger.warning("Failed to download artifacts via SFTP: %s", exc)
            self.log_received.emit(f"⚠️ Ошибка загрузки артефактов SFTP: {exc}")
        finally:
            try:
                sftp.close()
            except OSError:
                pass

    def _cleanup(self) -> None:
        """Close remote channel and SSH connection."""
        if self._channel is not None:
            try:
                self._channel.close()
            except OSError:
                pass
            self._channel = None

        if self._ssh_client is not None:
            try:
                self._ssh_client.close()
            except OSError:
                pass
            self._ssh_client = None
