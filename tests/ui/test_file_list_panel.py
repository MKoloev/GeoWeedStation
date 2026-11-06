"""Tests for FileTableModel and FileListPanel widgets (offscreen)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from geo_weed_station.schemas.session import FileEntry, ProcessingStatus
from geo_weed_station.ui.models.file_table_model import (
    COL_DETECTIONS,
    COL_NAME,
    COL_SIZE,
    COL_STATUS,
    COL_XMP,
    FileTableModel,
)
from geo_weed_station.ui.widgets.file_list_panel import FileListPanel


def _sample_entries() -> list[FileEntry]:
    return [
        FileEntry(
            path="/data/DJI_0001.JPG",
            size_bytes=1024 * 1024 * 5,  # 5 MB
            status=ProcessingStatus.PENDING,
            has_xmp=True,
            detections_count=None,
        ),
        FileEntry(
            path="/data/DJI_0002.JPG",
            size_bytes=1024 * 500,  # 500 KB
            status=ProcessingStatus.DONE,
            has_xmp=False,
            detections_count=12,
        ),
    ]


def test_file_table_model_data():
    model = FileTableModel()
    entries = _sample_entries()
    model.set_entries(entries)

    assert model.rowCount() == 2
    assert model.columnCount() == 5

    # Check headers
    assert model.headerData(COL_NAME, Qt.Orientation.Horizontal) == "Имя файла"
    assert model.headerData(COL_SIZE, Qt.Orientation.Horizontal) == "Размер"

    # Check row 0 data
    idx_name = model.index(0, COL_NAME)
    assert model.data(idx_name, Qt.ItemDataRole.DisplayRole) == "DJI_0001.JPG"

    idx_size = model.index(0, COL_SIZE)
    assert "МБ" in model.data(idx_size, Qt.ItemDataRole.DisplayRole)

    idx_status = model.index(0, COL_STATUS)
    assert model.data(idx_status, Qt.ItemDataRole.DisplayRole) == "Ожидание"

    idx_xmp = model.index(0, COL_XMP)
    assert model.data(idx_xmp, Qt.ItemDataRole.DisplayRole) == "Да"

    idx_det = model.index(0, COL_DETECTIONS)
    assert model.data(idx_det, Qt.ItemDataRole.DisplayRole) == "-"

    # Check row 1 detections
    idx_det1 = model.index(1, COL_DETECTIONS)
    assert model.data(idx_det1, Qt.ItemDataRole.DisplayRole) == "12"


def test_file_table_model_status_update():
    model = FileTableModel()
    model.set_entries(_sample_entries())

    model.update_status(0, ProcessingStatus.PROCESSING)
    idx_status = model.index(0, COL_STATUS)
    assert model.data(idx_status, Qt.ItemDataRole.DisplayRole) == "Обработка"


def test_file_list_panel_selection_and_signals(qtbot):
    panel = FileListPanel()
    qtbot.addWidget(panel)
    panel.resize(500, 300)
    panel.show()

    entries = _sample_entries()
    panel.set_entries(entries)

    selected_rows = []
    panel.file_selected.connect(selected_rows.append)

    # Programmatic selection
    panel.table_view.selectRow(1)
    assert 1 in selected_rows
    assert panel.selected_row() == 1
    selected = panel.selected_entry()
    assert selected is not None
    assert selected.path == "/data/DJI_0002.JPG"


def test_file_list_panel_process_signal(qtbot):
    panel = FileListPanel()
    qtbot.addWidget(panel)

    process_signals = []
    panel.process_requested.connect(lambda: process_signals.append(True))

    panel.btn_process.click()
    assert process_signals == [True]


def test_file_list_panel_clear(qtbot):
    panel = FileListPanel()
    qtbot.addWidget(panel)
    panel.set_entries(_sample_entries())
    assert panel.model.rowCount() == 2

    panel.btn_clear.click()
    assert panel.model.rowCount() == 0
