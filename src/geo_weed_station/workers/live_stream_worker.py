"""Background worker polling debug video snapshots and frames from Skyris web_video_server."""

from __future__ import annotations

import logging
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QPixmap

from geo_weed_station.workers.base_worker import ManagedWorker

logger = logging.getLogger(__name__)

STATUS_LIVE = "LIVE"
STATUS_WAITING = "WAITING"
STATUS_DISCONNECTED = "DISCONNECTED"

STATUS_TEXT_LIVE = "🟢 В эфире"
STATUS_TEXT_WAITING = "🟡 Ожидание потока"
STATUS_TEXT_DISCONNECTED = "🔴 Нет связи с бортом"

DEFAULT_STREAM_HOST = "10.42.0.1"
DEFAULT_STREAM_PORT = 8080
DEFAULT_STREAM_TOPIC = "/yolo/debug"

# Opener ignoring http_proxy/https_proxy environment variables (direct connection to the drone).
_DIRECT_OPENER = build_opener(ProxyHandler({}))


def urlopen(req: Request, timeout: float):  # noqa: ANN201
    """Open request directly (no system proxy) -- drone is on a local Wi-Fi network."""
    return _DIRECT_OPENER.open(req, timeout=timeout)


class LiveStreamWorker(ManagedWorker):
    """Background QThread worker polling JPEG snapshots from ROS web_video_server (port 8080).
    
    Streams decoded QPixmap frames to UI, tracks board reachability, and manages status badges.
    """

    frame_ready = Signal(QPixmap)
    raw_frame_ready = Signal(bytes)
    status_changed = Signal(str, str)  # status_key, display_text
    connection_error = Signal(str)

    def __init__(
        self,
        host: str = DEFAULT_STREAM_HOST,
        port: int = DEFAULT_STREAM_PORT,
        topic: str = DEFAULT_STREAM_TOPIC,
        poll_rate_hz: float = 3.0,
        timeout_s: float = 1.5,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._port = port
        self._topic = topic
        self._poll_rate_hz = max(0.5, min(10.0, poll_rate_hz))
        self._timeout_s = timeout_s

        self._lock = threading.Lock()
        self._current_status: str | None = None

    # -------------------------------------------------------------------------
    # Properties & Setters
    # -------------------------------------------------------------------------
    @property
    def host(self) -> str:
        with self._lock:
            return self._host

    @property
    def port(self) -> int:
        with self._lock:
            return self._port

    @property
    def topic(self) -> str:
        with self._lock:
            return self._topic

    @property
    def poll_rate_hz(self) -> float:
        with self._lock:
            return self._poll_rate_hz

    @Slot(str)
    def set_topic(self, topic: str) -> None:
        """Update active ROS image topic in a thread-safe manner."""
        with self._lock:
            if self._topic != topic:
                logger.info("LiveStreamWorker topic changed to: %s", topic)
                self._topic = topic
                self._update_status(STATUS_WAITING, STATUS_TEXT_WAITING, force=True)

    @Slot(str, int)
    def set_endpoint(self, host: str, port: int) -> None:
        """Update stream host and port."""
        with self._lock:
            self._host = host
            self._port = port

    @Slot(float)
    def set_poll_rate(self, rate_hz: float) -> None:
        """Adjust polling frequency (clamped between 0.5 and 10.0 Hz)."""
        with self._lock:
            self._poll_rate_hz = max(0.5, min(10.0, rate_hz))

    # -------------------------------------------------------------------------
    # Main Polling Loop
    # -------------------------------------------------------------------------
    @Slot()
    def start_work(self) -> None:
        """Execute continuous snapshot polling loop in worker thread."""
        self._running = True
        self.started_ok.emit()
        self._update_status(STATUS_WAITING, STATUS_TEXT_WAITING, force=True)

        logger.info(
            "LiveStreamWorker started: http://%s:%d/snapshot?topic=%s at %.1f Hz",
            self.host,
            self.port,
            self.topic,
            self.poll_rate_hz,
        )

        while not self.is_stop_requested:
            if self.is_paused:
                time.sleep(0.1)
                continue

            t_start = time.time()
            self._fetch_single_frame()

            # Target interval calculation
            with self._lock:
                interval = 1.0 / self._poll_rate_hz

            elapsed = time.time() - t_start
            sleep_time = max(0.01, interval - elapsed)

            # Responsive sleep chunking
            chunk = 0.05
            while sleep_time > 0 and not self.is_stop_requested:
                time.sleep(min(chunk, sleep_time))
                sleep_time -= chunk

        self._running = False
        self.finished.emit()
        logger.info("LiveStreamWorker finished.")

    def _fetch_single_frame(self) -> None:
        """Issue HTTP GET /snapshot?topic=... and process image data."""
        with self._lock:
            url = f"http://{self._host}:{self._port}/snapshot?topic={self._topic}"
            timeout = self._timeout_s

        try:
            req = Request(url, headers={"User-Agent": "GeoWeedStation-GCS/1.0"})
            # The drone lives on a direct local Wi-Fi link (10.42.0.0/24): never route it through
            # system HTTP(S) proxies (e.g. Clash/V2Ray via http_proxy), which return 502/timeouts.
            with urlopen(req, timeout=timeout) as resp:
                status_code = resp.status
                if status_code == 200:
                    data = resp.read()
                    if len(data) > 100:
                        pixmap = QPixmap()
                        loaded = pixmap.loadFromData(data)
                        if loaded and not pixmap.isNull():
                            self.frame_ready.emit(pixmap)
                            self.raw_frame_ready.emit(data)
                            self._update_status(STATUS_LIVE, STATUS_TEXT_LIVE)
                        else:
                            self._update_status(STATUS_WAITING, STATUS_TEXT_WAITING)
                    else:
                        self._update_status(STATUS_WAITING, STATUS_TEXT_WAITING)
                else:
                    self._update_status(STATUS_WAITING, f"{STATUS_TEXT_WAITING} ({status_code})")

        except HTTPError as http_err:
            # e.g., 404 if topic not registered yet
            logger.debug("HTTP error from web_video_server: %s", http_err)
            self._update_status(STATUS_WAITING, f"{STATUS_TEXT_WAITING} (HTTP {http_err.code})")
        except (URLError, TimeoutError, OSError) as net_err:
            # Board unreachable or connection refused
            logger.debug("Network error contacting web_video_server: %s", net_err)
            self._update_status(STATUS_DISCONNECTED, STATUS_TEXT_DISCONNECTED)
            self.connection_error.emit(str(net_err))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unexpected error fetching snapshot: %s", exc)
            self._update_status(STATUS_DISCONNECTED, STATUS_TEXT_DISCONNECTED)

    def _update_status(self, key: str, text: str, force: bool = False) -> None:
        """Emit status_changed signal when status changes to avoid UI spam."""
        if force or self._current_status != key:
            self._current_status = key
            self.status_changed.emit(key, text)
