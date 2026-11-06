"""Tests for SettingsDialog configuration management and validation (offscreen)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialogButtonBox

from geo_weed_station.config.settings import (
    AppSettings,
    DetectorSettings,
    MapSettings,
    NetworkSettings,
)
from geo_weed_station.ui.dialogs.settings_dialog import SettingsDialog


@pytest.fixture
def sample_settings() -> AppSettings:
    return AppSettings(
        network=NetworkSettings(
            host="192.168.1.100",
            port=8080,
            request_timeout_s=1.5,
            poll_rate_hz=1.0,
        ),
        detector=DetectorSettings(
            model_path=Path("models/test.onnx"),
            classes_path=Path("models/classes.json"),
            confidence_threshold=0.35,
        ),
        map=MapSettings(
            heatmap_cell_m=0.8,
            critical_density_threshold=4.0,
            wgs84_origin=(55.75, 37.61),
        ),
        workdir=Path("/tmp/workdir"),
    )


def test_settings_dialog_initial_population(sample_settings, qtbot):
    dialog = SettingsDialog(sample_settings)
    qtbot.addWidget(dialog)

    assert dialog.edit_host.text() == "192.168.1.100"
    assert dialog.spin_port.value() == 8080
    assert dialog.spin_timeout.value() == 1.5
    assert dialog.spin_poll_hz.value() == 1.0
    assert dialog.spin_conf_thresh.value() == 0.35
    assert dialog.spin_cell_m.value() == 0.8
    assert dialog.cb_has_origin.isChecked()
    assert dialog.spin_lat.value() == 55.75
    assert dialog.spin_lon.value() == 37.61


def test_settings_dialog_modify_and_accept(sample_settings, qtbot):
    dialog = SettingsDialog(sample_settings)
    qtbot.addWidget(dialog)

    # Change some values
    dialog.edit_host.setText("10.0.0.50")
    dialog.spin_port.setValue(9000)
    dialog.spin_timeout.setValue(1.8)
    dialog.spin_conf_thresh.setValue(0.50)
    dialog.spin_cell_m.setValue(1.2)

    dialog._on_accept()

    res = dialog.result_settings()
    assert res.network.host == "10.0.0.50"
    assert res.network.port == 9000
    assert res.network.request_timeout_s == 1.8
    assert res.detector.confidence_threshold == 0.50
    assert res.map.heatmap_cell_m == 1.2


def test_settings_dialog_validation_blocks_ok(sample_settings, qtbot):
    dialog = SettingsDialog(sample_settings)
    qtbot.addWidget(dialog)
    dialog.show()

    ok_btn = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_btn.isEnabled()

    # 1. Timeout > 2.0s -> invalid!
    dialog.spin_timeout.setValue(2.5)
    assert not ok_btn.isEnabled()
    assert dialog.lbl_error.isVisible()

    # Restore timeout
    dialog.spin_timeout.setValue(1.5)
    assert ok_btn.isEnabled()
    assert not dialog.lbl_error.isVisible()

    # 2. Poll rate < 0.5 Hz -> invalid!
    dialog.spin_poll_hz.setValue(0.2)
    assert not ok_btn.isEnabled()
    assert dialog.lbl_error.isVisible()


def test_settings_dialog_browse_model_auto_resolves_classes(sample_settings, tmp_path, monkeypatch, qtbot):
    dialog = SettingsDialog(sample_settings)
    qtbot.addWidget(dialog)

    model_file = tmp_path / "custom_yolo.onnx"
    model_file.touch()
    classes_file = tmp_path / "custom_yolo.json"
    classes_file.write_text('{"0": "Corn"}')

    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(model_file), "ONNX models (*.onnx)"))

    dialog._browse_model()

    assert dialog.edit_model_path.text() == str(model_file)
    assert dialog.edit_classes_path.text() == str(classes_file)
