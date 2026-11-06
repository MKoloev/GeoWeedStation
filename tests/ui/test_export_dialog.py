"""Tests for ExportDialog and ConfirmDialog (offscreen)."""

from __future__ import annotations

from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.onboard_api import StopAction
from geo_weed_station.ui.dialogs.confirm_dialog import ConfirmDialog
from geo_weed_station.ui.dialogs.export_dialog import ExportDialog, ExportFormat, ExportRequest


def test_export_dialog_formats_and_options(tmp_path, qtbot):
    dialog = ExportDialog(
        coord_system=CoordinateSystem.LOCAL_ARUCO,
        has_orthophoto=False,
        default_dir=tmp_path,
    )
    qtbot.addWidget(dialog)

    # Initial state: CSV, GeoJSON, PNG all checked
    assert dialog.cb_csv.isChecked()
    assert dialog.cb_geojson.isChecked()
    assert dialog.cb_png.isChecked()
    assert dialog.png_group.isEnabled()

    # Without orthophoto, background combo has only 2 items
    items = [dialog.combo_bg.itemText(i) for i in range(dialog.combo_bg.count())]
    assert "Подложка ортофотоплана" not in items

    # Accept request
    dialog._on_accept()
    req = dialog.result_request()
    assert isinstance(req, ExportRequest)
    assert req.formats == {ExportFormat.CSV, ExportFormat.GEOJSON, ExportFormat.PNG}
    assert req.target_dir == tmp_path
    assert req.base_name == "weed_map_export"
    assert req.png_options is not None
    assert req.png_options.dpi == 300


def test_export_dialog_with_orthophoto(tmp_path, qtbot):
    dialog = ExportDialog(
        coord_system=CoordinateSystem.WGS84,
        has_orthophoto=True,
        default_dir=tmp_path,
    )
    qtbot.addWidget(dialog)

    items = [dialog.combo_bg.itemText(i) for i in range(dialog.combo_bg.count())]
    assert "Подложка ортофотоплана" in items


def test_export_dialog_validation_no_formats(tmp_path, qtbot):
    dialog = ExportDialog(default_dir=tmp_path)
    qtbot.addWidget(dialog)
    dialog.show()

    ok_btn = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_btn.isEnabled()

    # Uncheck all formats
    dialog.cb_csv.setChecked(False)
    dialog.cb_geojson.setChecked(False)
    dialog.cb_png.setChecked(False)

    assert not ok_btn.isEnabled()
    assert dialog.lbl_error.isVisible()

    # Check one format back
    dialog.cb_csv.setChecked(True)
    assert ok_btn.isEnabled()
    assert not dialog.lbl_error.isVisible()


def test_confirm_dialog(monkeypatch):
    # Test delete confirmation
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    assert ConfirmDialog.confirm_delete_run("run_123", 1024 * 1024 * 10) is True

    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    assert ConfirmDialog.confirm_delete_run("run_123", 1024 * 1024 * 10) is False

    # Test abort confirmation
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    assert ConfirmDialog.confirm_abort(StopAction.LAND) is True

    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Cancel)
    assert ConfirmDialog.confirm_abort(StopAction.RTH) is False
