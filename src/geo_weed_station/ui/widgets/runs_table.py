"""Table widget displaying onboard mission runs history with download, delete and process actions."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.schemas.onboard_api import DroneRun

logger = logging.getLogger(__name__)

COL_RUN_ID = 0
COL_TIME = 1
COL_STATUS = 2
COL_SIZE = 3
COL_CONTENTS = 4
HEADERS = ["ID запуска", "Время старта", "Статус", "Размер", "Содержимое"]


def _format_size_mb(size_bytes: int) -> str:
    mb = size_bytes / (1024.0 * 1024.0)
    if mb < 0.1:
        kb = size_bytes / 1024.0
        return f"{kb:.1f} КБ"
    return f"{mb:.2f} МБ"


class RunsTable(QWidget):
    """Panel showing past onboard flight runs with download, local processing, and deletion triggers."""

    refresh_requested = Signal()
    download_requested = Signal(str)  # run_id
    delete_requested = Signal(str)  # run_id
    process_requested = Signal(str)  # run_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._runs: List[DroneRun] = []
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Table
        self.table = QTableWidget(0, len(HEADERS), self)
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_RUN_ID, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_TIME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_CONTENTS, QHeaderView.ResizeMode.Stretch)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, stretch=1)

        # 2. Action Buttons Toolbar
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_refresh = QPushButton("🔄 Обновить список", self)
        self.btn_refresh.clicked.connect(self.refresh_requested.emit)
        btn_layout.addWidget(self.btn_refresh)

        self.btn_download = QPushButton("📥 Скачать архив", self)
        self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self._on_download_clicked)
        btn_layout.addWidget(self.btn_download)

        self.btn_process = QPushButton("⚡ Обработать локально", self)
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self._on_process_clicked)
        btn_layout.addWidget(self.btn_process)

        self.btn_delete = QPushButton("🗑 Удалить с борта", self)
        self.btn_delete.setEnabled(False)
        self.btn_delete.setStyleSheet("color: #ff6b6b;")
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        btn_layout.addWidget(self.btn_delete)

        layout.addLayout(btn_layout)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def set_runs(self, runs: Sequence[DroneRun]) -> None:
        """Populate table with onboard mission runs list."""
        self._runs = list(runs)
        self.table.setRowCount(len(self._runs))

        for row, run in enumerate(self._runs):
            item_id = QTableWidgetItem(run.run_id)
            item_id.setData(Qt.ItemDataRole.UserRole, run.run_id)
            self.table.setItem(row, COL_RUN_ID, item_id)

            item_time = QTableWidgetItem(run.start_time)
            self.table.setItem(row, COL_TIME, item_time)

            item_status = QTableWidgetItem(run.status)
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, COL_STATUS, item_status)

            item_size = QTableWidgetItem(_format_size_mb(run.size_bytes))
            item_size.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.table.setItem(row, COL_SIZE, item_size)

            contents_str = ", ".join(run.contents) if run.contents else "—"
            item_contents = QTableWidgetItem(contents_str)
            self.table.setItem(row, COL_CONTENTS, item_contents)

        self._update_buttons_state()

    def selected_run_id(self) -> Optional[str]:
        """Return run_id of currently selected row or None."""
        selected_items = self.table.selectedItems()
        if not selected_items:
            return None
        row = selected_items[0].row()
        item = self.table.item(row, COL_RUN_ID)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    # -------------------------------------------------------------------------
    # Internal handlers
    # -------------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        self._update_buttons_state()

    def _update_buttons_state(self) -> None:
        has_sel = self.selected_run_id() is not None
        self.btn_download.setEnabled(has_sel)
        self.btn_process.setEnabled(has_sel)
        self.btn_delete.setEnabled(has_sel)

    def _on_download_clicked(self) -> None:
        run_id = self.selected_run_id()
        if run_id:
            self.download_requested.emit(run_id)

    def _on_process_clicked(self) -> None:
        run_id = self.selected_run_id()
        if run_id:
            self.process_requested.emit(run_id)

    def _on_delete_clicked(self) -> None:
        run_id = self.selected_run_id()
        if run_id:
            self.delete_requested.emit(run_id)
