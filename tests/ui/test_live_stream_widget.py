"""Tests for LiveStreamWidget UI controls, frame rendering and snapshot export."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QImage, QPixmap

from geo_weed_station.ui.widgets.live_stream_widget import LiveStreamWidget
from geo_weed_station.workers.live_stream_worker import (
    STATUS_DISCONNECTED,
    STATUS_LIVE,
    STATUS_TEXT_DISCONNECTED,
    STATUS_TEXT_LIVE,
)


@pytest.fixture
def sample_pixmap() -> QPixmap:
    img = QImage(640, 480, QImage.Format.Format_RGB32)
    img.fill(QColor(50, 150, 50))
    return QPixmap.fromImage(img)


def test_live_stream_widget_init(qtbot):
    """Test initial widget layout and default status."""
    widget = LiveStreamWidget(host="10.42.0.1", port=8080)
    qtbot.addWidget(widget)
    widget.show()

    assert "10.42.0.1:8080" in widget.lbl_endpoint_info.text()
    assert "/yolo/debug" in widget.combo_topic.currentText()
    assert "Ожидание" in widget.lbl_status.text()
    assert "Ожидание подключения" in widget.lbl_viewport.text()


def test_live_stream_widget_frame_rendering(sample_pixmap, qtbot):
    """Test setting a new frame renders it and updates frame info."""
    widget = LiveStreamWidget()
    qtbot.addWidget(widget)
    widget.show()

    widget.set_frame(sample_pixmap)
    assert widget.lbl_viewport.pixmap() is not None
    assert "640×480" in widget.lbl_frame_info.text()


def test_live_stream_widget_status_transitions(qtbot):
    """Test status badge transitions between LIVE and DISCONNECTED."""
    widget = LiveStreamWidget()
    qtbot.addWidget(widget)

    # Set LIVE
    widget.set_status(STATUS_LIVE, STATUS_TEXT_LIVE)
    assert "В эфире" in widget.lbl_status.text()

    # Set DISCONNECTED
    widget.set_status(STATUS_DISCONNECTED, STATUS_TEXT_DISCONNECTED)
    assert "Нет связи" in widget.lbl_status.text()
    assert "Нет связи" in widget.lbl_viewport.text()


def test_live_stream_widget_topic_change_signal(qtbot):
    """Test changing topic in combobox emits topic_changed signal."""
    widget = LiveStreamWidget()
    qtbot.addWidget(widget)

    topics: list[str] = []
    widget.topic_changed.connect(topics.append)

    widget.combo_topic.setCurrentIndex(1)
    assert len(topics) == 1
    assert topics[0] == "/collect_onboard_image/debug"


def test_live_stream_widget_save_snapshot(sample_pixmap, tmp_path: Path, qtbot):
    """Test saving current frame to disk."""
    widget = LiveStreamWidget()
    qtbot.addWidget(widget)
    widget._snapshots_dir = tmp_path / "snapshots"

    saved_paths: list[str] = []
    widget.snapshot_saved.connect(saved_paths.append)

    # With no frame -> returns None
    assert widget.save_current_snapshot() is None

    # With frame -> saves JPG file
    widget.set_frame(sample_pixmap)
    saved = widget.save_current_snapshot()
    assert saved is not None
    assert Path(saved).exists()
    assert len(saved_paths) == 1
    assert "snap_" in saved


def test_live_stream_widget_auto_snapshot(sample_pixmap, tmp_path: Path, qtbot):
    """Test auto-snapshot timer triggering and toggle button."""
    widget = LiveStreamWidget()
    qtbot.addWidget(widget)
    widget._snapshots_dir = tmp_path / "snapshots"

    saved_paths: list[str] = []
    widget.snapshot_saved.connect(saved_paths.append)

    # Set frame
    widget.set_frame(sample_pixmap)

    # Start auto snapshot with very short interval for test
    widget.start_auto_snapshot(interval_s=0.05)
    assert widget.is_auto_snapshot_active() is True
    assert widget.btn_auto_snapshot.isChecked() is True

    # Wait for timer ticks
    qtbot.waitUntil(lambda: len(saved_paths) >= 2, timeout=2000)

    # Stop auto snapshot
    widget.stop_auto_snapshot()
    assert widget.is_auto_snapshot_active() is False
    assert widget.btn_auto_snapshot.isChecked() is False

    initial_count = len(saved_paths)
    qtbot.wait(100)
    assert len(saved_paths) == initial_count

