"""Live video stream monitoring widget displaying drone detection stream with HUD."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.workers.live_stream_worker import (
    DEFAULT_STREAM_HOST,
    DEFAULT_STREAM_PORT,
    DEFAULT_STREAM_TOPIC,
    STATUS_DISCONNECTED,
    STATUS_LIVE,
    STATUS_TEXT_WAITING,
    STATUS_WAITING,
)

logger = logging.getLogger(__name__)


class LiveStreamWidget(QWidget):
    """Widget displaying the live camera video stream with YOLO weed detections and HUD."""

    topic_changed = Signal(str)
    snapshot_saved = Signal(str)

    def __init__(
        self,
        host: str = DEFAULT_STREAM_HOST,
        port: int = DEFAULT_STREAM_PORT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._port = port
        self._current_pixmap: QPixmap | None = None
        self._current_status_key = STATUS_WAITING
        self._current_status_text = STATUS_TEXT_WAITING
        self._snapshots_dir = Path("workdir/snapshots")

        self._init_ui()
        self.set_status(STATUS_WAITING, STATUS_TEXT_WAITING)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # 1. Top control toolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 2, 2, 2)
        top_bar.setSpacing(8)

        # Status Badge
        self.lbl_status = QLabel(STATUS_TEXT_WAITING, self)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setFixedHeight(26)
        top_bar.addWidget(self.lbl_status)

        # Topic Selector Label & Combo
        lbl_topic_title = QLabel("Топик:", self)
        top_bar.addWidget(lbl_topic_title)

        self.combo_topic = QComboBox(self)
        self.combo_topic.addItem("/yolo/debug (YOLO Детекция + HUD)", "/yolo/debug")
        self.combo_topic.addItem(
            "/collect_onboard_image/debug (Бортовая камера)", "/collect_onboard_image/debug"
        )
        self.combo_topic.setEditable(True)
        self.combo_topic.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_topic.currentIndexChanged.connect(self._on_topic_index_changed)
        self.combo_topic.lineEdit().returnPressed.connect(self._on_custom_topic_entered)
        top_bar.addWidget(self.combo_topic)

        # Save snapshot button
        self.btn_save_snapshot = QPushButton("💾 Снимок", self)
        self.btn_save_snapshot.setToolTip("Сохранить текущий кадр детекции на диск станции")
        self.btn_save_snapshot.clicked.connect(self.save_current_snapshot)
        top_bar.addWidget(self.btn_save_snapshot)

        main_layout.addLayout(top_bar)

        # 2. Video viewport label inside container frame
        self.viewport_frame = QFrame(self)
        self.viewport_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.viewport_frame.setStyleSheet(
            "background-color: #121212; border: 1px solid #333333; border-radius: 4px;"
        )
        viewport_layout = QVBoxLayout(self.viewport_frame)
        viewport_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_viewport = QLabel(self.viewport_frame)
        self.lbl_viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_viewport.setMinimumSize(320, 240)
        viewport_layout.addWidget(self.lbl_viewport)

        main_layout.addWidget(self.viewport_frame, 1)

        # 3. Bottom info footer
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(2, 0, 2, 0)
        self.lbl_endpoint_info = QLabel(
            f"Источник: http://{self._host}:{self._port}/snapshot?topic={DEFAULT_STREAM_TOPIC}", self
        )
        self.lbl_endpoint_info.setStyleSheet("color: #888888; font-size: 11px;")
        footer_layout.addWidget(self.lbl_endpoint_info)

        self.lbl_frame_info = QLabel("Нет кадров", self)
        self.lbl_frame_info.setStyleSheet("color: #888888; font-size: 11px;")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignRight)
        footer_layout.addWidget(self.lbl_frame_info)

        main_layout.addLayout(footer_layout)

    # -------------------------------------------------------------------------
    # Public Slots & Methods
    # -------------------------------------------------------------------------
    @Slot(QPixmap)
    def set_frame(self, pixmap: QPixmap) -> None:
        """Display incoming frame while preserving aspect ratio."""
        if pixmap.isNull():
            return

        self._current_pixmap = pixmap
        self.lbl_frame_info.setText(f"{pixmap.width()}×{pixmap.height()} px")
        self._render_current_frame()

    @Slot(str, str)
    def set_status(self, status_key: str, display_text: str) -> None:
        """Update status badge text and appearance."""
        self._current_status_key = status_key
        self._current_status_text = display_text
        self.lbl_status.setText(f" {display_text} ")

        if status_key == STATUS_LIVE:
            self.lbl_status.setStyleSheet(
                "background-color: #2b8a3e; color: #ffffff; font-weight: bold; "
                "border-radius: 4px; padding: 2px 8px;"
            )
        elif status_key == STATUS_WAITING:
            self.lbl_status.setStyleSheet(
                "background-color: #d97706; color: #ffffff; font-weight: bold; "
                "border-radius: 4px; padding: 2px 8px;"
            )
        else:  # STATUS_DISCONNECTED
            self.lbl_status.setStyleSheet(
                "background-color: #c92a2a; color: #ffffff; font-weight: bold; "
                "border-radius: 4px; padding: 2px 8px;"
            )

        # If disconnected or waiting without frame, update placeholder
        if self._current_pixmap is None or status_key == STATUS_DISCONNECTED:
            self._render_placeholder()

    @Slot()
    def save_current_snapshot(self) -> str | None:
        """Save the currently displayed frame to disk."""
        if self._current_pixmap is None or self._current_pixmap.isNull():
            logger.warning("No frame available to save snapshot")
            return None

        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_sanitized = self.current_topic().replace("/", "_").strip("_")
        out_path = self._snapshots_dir / f"snap_{topic_sanitized}_{timestamp}.jpg"

        ok = self._current_pixmap.save(str(out_path), "JPEG", 95)
        if ok:
            logger.info("Snapshot saved to: %s", out_path)
            self.snapshot_saved.emit(str(out_path))
            self.lbl_frame_info.setText(f"💾 Сохранен: {out_path.name}")
            return str(out_path)
        else:
            logger.error("Failed to save snapshot to: %s", out_path)
            return None

    def set_endpoint(self, host: str, port: int) -> None:
        """Update displayed host/port endpoint."""
        self._host = host
        self._port = port
        topic = self.current_topic()
        self.lbl_endpoint_info.setText(
            f"Источник: http://{self._host}:{self._port}/snapshot?topic={topic}"
        )
        if self._current_pixmap is None:
            self._render_placeholder()

    def current_topic(self) -> str:
        """Get the currently selected ROS image topic."""
        data = self.combo_topic.currentData()
        if data:
            return str(data)
        return self.combo_topic.currentText().strip()

    # -------------------------------------------------------------------------
    # Internal Rendering & Events
    # -------------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle resize to re-render frame or placeholder with correct dimensions."""
        super().resizeEvent(event)
        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            self._render_current_frame()
        else:
            self._render_placeholder()

    def _render_current_frame(self) -> None:
        """Scale and render current pixmap centered in viewport."""
        if self._current_pixmap is None or self._current_pixmap.isNull():
            self._render_placeholder()
            return

        target_size = self.lbl_viewport.size()
        if target_size.width() < 10 or target_size.height() < 10:
            return

        scaled = self._current_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_viewport.setPixmap(scaled)

    def _render_placeholder(self) -> None:
        """Draw an informative dark placeholder when waiting or disconnected."""
        topic = self.current_topic()
        url = f"http://{self._host}:{self._port}/snapshot?topic={topic}"

        msg = (
            f"Ожидание подключения к {url}\n"
            f"Статус: {self._current_status_text}"
        )
        self.lbl_viewport.clear()
        self.lbl_viewport.setText(msg)
        self.lbl_viewport.setStyleSheet(
            "color: #aaaaaa; font-size: 13px; font-weight: normal; background-color: transparent;"
        )

    def _on_topic_index_changed(self, index: int) -> None:
        topic = self.current_topic()
        self.lbl_endpoint_info.setText(
            f"Источник: http://{self._host}:{self._port}/snapshot?topic={topic}"
        )
        self._current_pixmap = None
        self._render_placeholder()
        self.topic_changed.emit(topic)

    def _on_custom_topic_entered(self) -> None:
        topic = self.current_topic()
        self.lbl_endpoint_info.setText(
            f"Источник: http://{self._host}:{self._port}/snapshot?topic={topic}"
        )
        self._current_pixmap = None
        self._render_placeholder()
        self.topic_changed.emit(topic)
