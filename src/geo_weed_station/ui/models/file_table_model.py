"""Table model for displaying offline file batch entries in PySide6."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt

from geo_weed_station.schemas.session import FileEntry, ProcessingStatus

COL_NAME = 0
COL_SIZE = 1
COL_STATUS = 2
COL_XMP = 3
COL_DETECTIONS = 4
COLUMN_HEADERS = ["Имя файла", "Размер", "Статус", "XMP", "Сорняки"]

STATUS_LABELS = {
    ProcessingStatus.PENDING: "Ожидание",
    ProcessingStatus.PROCESSING: "Обработка",
    ProcessingStatus.DONE: "Готово",
    ProcessingStatus.ERROR: "Ошибка",
    ProcessingStatus.NO_METADATA: "Без XMP",
}


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable KiB/MiB/GiB."""
    if size_bytes <= 0:
        return "0 Б"
    units = ["Б", "КБ", "МБ", "ГБ"]
    size = float(size_bytes)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {units[idx]}" if idx > 0 else f"{int(size)} {units[idx]}"


class FileTableModel(QAbstractTableModel):
    """QAbstractTableModel representing a list of FileEntry objects for offline batch views."""

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._entries: List[FileEntry] = []

    # -------------------------------------------------------------------------
    # QAbstractTableModel interface overrides
    # -------------------------------------------------------------------------
    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._entries)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(COLUMN_HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if 0 <= section < len(COLUMN_HEADERS):
                return COLUMN_HEADERS[section]
        return None

    def data(  # noqa: N802
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None

        entry = self._entries[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == COL_NAME:
                return os.path.basename(entry.path)
            if col == COL_SIZE:
                return _format_size(entry.size_bytes)
            if col == COL_STATUS:
                return STATUS_LABELS.get(entry.status, str(entry.status))
            if col == COL_XMP:
                return "Да" if entry.has_xmp else "Нет"
            if col == COL_DETECTIONS:
                return str(entry.detections_count) if entry.detections_count is not None else "-"

        elif role == Qt.ItemDataRole.ToolTipRole:
            if col == COL_NAME:
                return entry.path
            if col == COL_STATUS:
                return f"Статус: {entry.status.value}"

        elif role == Qt.ItemDataRole.UserRole:
            return entry

        return None

    # -------------------------------------------------------------------------
    # Public domain methods
    # -------------------------------------------------------------------------
    def set_entries(self, entries: Sequence[FileEntry]) -> None:
        """Replace the entire list of file entries."""
        self.beginResetModel()
        self._entries = list(entries)
        self.endResetModel()

    def append_entry(self, entry: FileEntry) -> None:
        """Append a single file entry to the model."""
        pos = len(self._entries)
        self.beginInsertRows(QModelIndex(), pos, pos)
        self._entries.append(entry)
        self.endInsertRows()

    def update_status(self, row: int, status: ProcessingStatus) -> None:
        """Update status of a file entry at the given row."""
        if 0 <= row < len(self._entries):
            self._entries[row].status = status
            top_left = self.index(row, COL_STATUS)
            bottom_right = self.index(row, COL_STATUS)
            self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.DisplayRole])

    def update_entry(self, row: int, entry: FileEntry) -> None:
        """Update entire file entry at row (e.g. after extraction or inference)."""
        if 0 <= row < len(self._entries):
            self._entries[row] = entry
            top_left = self.index(row, 0)
            bottom_right = self.index(row, len(COLUMN_HEADERS) - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole])

    def entry_at(self, row: int) -> Optional[FileEntry]:
        """Return FileEntry at given row if in bounds, else None."""
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def clear(self) -> None:
        """Clear all entries."""
        self.beginResetModel()
        self._entries.clear()
        self.endResetModel()

    @property
    def entries(self) -> List[FileEntry]:
        """Return copy of entries."""
        return list(self._entries)
