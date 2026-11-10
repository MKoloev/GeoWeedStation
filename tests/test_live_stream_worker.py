"""Unit tests for LiveStreamWorker snapshot polling and resilience."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QColor, QImage, QPixmap

from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.live_stream_worker import (
    STATUS_DISCONNECTED,
    STATUS_LIVE,
    STATUS_WAITING,
    LiveStreamWorker,
)


def _create_sample_jpeg_bytes() -> bytes:
    """Create real valid JPEG image bytes in memory."""
    img = QImage(320, 240, QImage.Format.Format_RGB32)
    img.fill(QColor(40, 180, 40))
    byte_array = QByteArray()
    buf = QBuffer(byte_array)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPEG")
    return bytes(byte_array.data())


def test_live_stream_worker_receives_valid_frames(qtbot):
    """Test receiving valid JPEG snapshots transitions status to LIVE and emits frames."""
    jpeg_bytes = _create_sample_jpeg_bytes()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = jpeg_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    worker = LiveStreamWorker(host="10.42.0.1", port=8080, poll_rate_hz=5.0)

    frames: list[QPixmap] = []
    statuses: list[tuple[str, str]] = []

    worker.frame_ready.connect(frames.append)
    worker.status_changed.connect(lambda k, t: statuses.append((k, t)))

    with patch("geo_weed_station.workers.live_stream_worker.urlopen", return_value=mock_resp):
        thread = start_in_thread(worker, "test_stream_live")
        try:
            qtbot.waitUntil(lambda: len(frames) >= 1, timeout=3000)

            assert len(frames) >= 1
            assert not frames[0].isNull()
            assert frames[0].width() == 320
            assert frames[0].height() == 240

            assert any(k == STATUS_LIVE for k, _ in statuses)

        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)


def test_live_stream_worker_handles_404_waiting(qtbot):
    """Test 404 response (topic not yet published) emits WAITING status."""
    http_404 = HTTPError(url="http://10.42.0.1:8080/snapshot", code=404, msg="Not Found", hdrs={}, fp=io.BytesIO())

    worker = LiveStreamWorker(host="10.42.0.1", port=8080, poll_rate_hz=5.0)

    statuses: list[tuple[str, str]] = []
    worker.status_changed.connect(lambda k, t: statuses.append((k, t)))

    with patch("geo_weed_station.workers.live_stream_worker.urlopen", side_effect=http_404):
        thread = start_in_thread(worker, "test_stream_waiting")
        try:
            qtbot.waitUntil(lambda: any(k == STATUS_WAITING for k, _ in statuses), timeout=3000)
            assert any(k == STATUS_WAITING for k, _ in statuses)

        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)


def test_live_stream_worker_handles_connection_refused(qtbot):
    """Test connection refused / timeout transitions status to DISCONNECTED."""
    net_err = URLError("Connection refused")

    worker = LiveStreamWorker(host="10.42.0.1", port=8080, poll_rate_hz=5.0)

    statuses: list[tuple[str, str]] = []
    errs: list[str] = []

    worker.status_changed.connect(lambda k, t: statuses.append((k, t)))
    worker.connection_error.connect(errs.append)

    with patch("geo_weed_station.workers.live_stream_worker.urlopen", side_effect=net_err):
        thread = start_in_thread(worker, "test_stream_disconnected")
        try:
            qtbot.waitUntil(lambda: any(k == STATUS_DISCONNECTED for k, _ in statuses), timeout=3000)

            assert any(k == STATUS_DISCONNECTED for k, _ in statuses)
            assert len(errs) >= 1

        finally:
            worker.request_stop()
            thread.quit()
            thread.wait(3000)


def test_live_stream_worker_topic_switch(qtbot):
    """Test dynamic topic changing."""
    worker = LiveStreamWorker(topic="/yolo/debug")
    assert worker.topic == "/yolo/debug"

    worker.set_topic("/collect_onboard_image/debug")
    assert worker.topic == "/collect_onboard_image/debug"

    worker.set_poll_rate(4.0)
    assert worker.poll_rate_hz == 4.0
