"""File list panel presenting offline batch items with status and selection signals."""

from __future__ import annotations

from typing import Optional, Sequence
from PySide6.QtCore import QItemSelection, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.schemas.session import FileEntry, ProcessingStatus
from geo_weed_station.ui.models.file_table_model import (
    COL_DETECTIONS,
    COL_NAME,
    COL_SIZE,
    COL_STATUS,
    COL_XMP,
    FileTableModel,
)


class FileListPanel(QWidget):
    """Panel displaying the batch file table and control buttons."""

    file_selected = Signal(int)  # Emitted with row index when user selects an item
    process_requested = Signal()  # Emitted when user clicks 'Start Processing'

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = FileTableModel(self)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Table View
        self.table_view = QTableView(self)
        self.table_view.setModel(self._model)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.verticalHeader().setVisible(False)

        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_XMP, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_DETECTIONS, QHeaderView.ResizeMode.ResizeToContents)

        self.table_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table_view, stretch=1)

        # 2. Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_process = QPushButton("⚡ Запустить обработку", self)
        self.btn_process.clicked.connect(self.process_requested.emit)
        btn_layout.addWidget(self.btn_process)

        self.btn_clear = QPushButton("Очистить", self)
        self.btn_clear.clicked.connect(self.clear)
        btn_layout.addWidget(self.btn_clear)

        layout.addLayout(btn_layout)

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    @property
    def model(self) -> FileTableModel:
        """Return the underlying FileTableModel."""
        return self._model

    def set_entries(self, entries: Sequence[FileEntry]) -> None:
        """Populate the table with new file entries."""
        self._model.set_entries(entries)
        if entries:
            self.table_view.selectRow(0)

    def append_entry(self, entry: FileEntry) -> None:
        """Append a file entry to table."""
        self._model.append_entry(entry)

    def update_status(self, row: int, status: ProcessingStatus) -> None:
        """Update file processing status at row."""
        self._model.update_status(row, status)

    def update_entry(self, row: int, entry: FileEntry) -> None:
        """Update entire file entry at row."""
        self._model.update_entry(row, entry)

    def selected_row(self) -> Optional[int]:
        """Return currently selected row index or None."""
        selection = self.table_view.selectionModel().selectedRows()
        if selection:
            return selection[0].row()
        return None

    def selected_entry(self) -> Optional[FileEntry]:
        """Return currently selected FileEntry or None."""
        row = self.selected_row()
        if row is not None:
            return self._model.entry_at(row)
        return None

    def set_process_enabled(self, enabled: bool) -> None:
        """Enable or disable the process button."""
        self.btn_process.setEnabled(enabled)

    def clear(self) -> None:
        """Clear table entries."""
        self._model.clear()

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------
    def _on_selection_changed(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        rows = self.table_view.selectionModel().selectedRows()
        if rows:
            self.file_selected.emit(rows[0].row())
