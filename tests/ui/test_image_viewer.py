"""Tests for ImageViewer and DetectionFilterPanel widgets (offscreen)."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent

from geo_weed_station.schemas.entities import BBox, DetectedWeed
from geo_weed_station.ui.widgets.detection_filter_panel import (
    DetectionFilterPanel,
    DetectionFilterState,
)
from geo_weed_station.ui.widgets.image_viewer import ImageViewer


def _dummy_image(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _sample_detections() -> list[DetectedWeed]:
    return [
        DetectedWeed(
            weed_id="w-1",
            class_id=0,
            class_name="Ambrosia",
            confidence=0.85,
            bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
        ),
        DetectedWeed(
            weed_id="w-2",
            class_id=1,
            class_name="Cirsium",
            confidence=0.40,
            bbox=BBox(x=0.5, y=0.5, w=0.15, h=0.15),
        ),
        DetectedWeed(
            weed_id="w-3",
            class_id=0,
            class_name="Ambrosia",
            confidence=0.15,
            bbox=BBox(x=0.7, y=0.7, w=0.1, h=0.1),
        ),
    ]


def test_image_viewer_set_image_and_fit(qtbot):
    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(800, 600)
    viewer.show()

    img = _dummy_image(640, 480)
    viewer.set_image(img)

    rect = viewer.scene().sceneRect()
    assert rect.width() == 640
    assert rect.height() == 480
    assert viewer._current_zoom > 0


def test_image_viewer_downscale_extreme_dimensions(qtbot):
    """Large images (e.g. 8000x6000) are downscaled for scene to prevent memory exhaustion."""
    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(800, 600)
    viewer.show()

    # Create dummy 8000x6000 array with stride-friendly pattern
    large_img = np.zeros((6000, 8000, 3), dtype=np.uint8)
    viewer.set_image(large_img)

    rect = viewer.scene().sceneRect()
    assert rect.width() <= viewer.MAX_PIXMAP_DIM
    assert rect.height() <= viewer.MAX_PIXMAP_DIM


def test_image_viewer_zoom_wheel(qtbot):
    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(800, 600)
    viewer.show()

    viewer.set_image(_dummy_image(640, 480))
    initial_zoom = viewer._current_zoom

    # Zoom in wheel event
    wheel_in = QWheelEvent(
        QPoint(400, 300),
        viewer.mapToGlobal(QPoint(400, 300)),
        QPoint(0, 0),
        QPoint(0, 120),  # positive delta -> zoom in
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    viewer.wheelEvent(wheel_in)
    assert viewer._current_zoom > initial_zoom

    # Zoom out wheel event
    wheel_out = QWheelEvent(
        QPoint(400, 300),
        viewer.mapToGlobal(QPoint(400, 300)),
        QPoint(0, 0),
        QPoint(0, -120),  # negative delta -> zoom out
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    viewer.wheelEvent(wheel_out)
    assert pytest.approx(viewer._current_zoom, rel=1e-2) == initial_zoom


def test_image_viewer_detection_filtering(qtbot):
    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(800, 600)
    viewer.show()

    viewer.set_image(_dummy_image(640, 480))
    viewer.set_detections(_sample_detections())
    assert len(viewer._detection_items) == 3

    # Default filter has min_confidence=0.25 -> items with 0.85 and 0.40 visible, 0.15 hidden
    visible = [it for it in viewer._detection_items if it.isVisible()]
    assert len(visible) == 2

    # Filter with higher threshold 0.50 -> only 0.85 visible
    state_high = DetectionFilterState(min_confidence=0.50)
    viewer.apply_filter(state_high)
    visible_high = [it for it in viewer._detection_items if it.isVisible()]
    assert len(visible_high) == 1
    assert visible_high[0].weed.weed_id == "w-1"

    # Hide class 'Ambrosia' -> only 'Cirsium' with conf >= 0.25 visible (w-2)
    state_class = DetectionFilterState(
        min_confidence=0.25,
        hidden_classes=frozenset(["Ambrosia"]),
    )
    viewer.apply_filter(state_class)
    visible_class = [it for it in viewer._detection_items if it.isVisible()]
    assert len(visible_class) == 1
    assert visible_class[0].weed.weed_id == "w-2"

    # Hide all boxes
    state_no_boxes = DetectionFilterState(show_boxes=False)
    viewer.apply_filter(state_no_boxes)
    assert all(not it.isVisible() for it in viewer._detection_items)


def test_detection_filter_panel_signals(qtbot):
    panel = DetectionFilterPanel()
    qtbot.addWidget(panel)
    panel.set_classes(["Ambrosia", "Cirsium", "Chenopodium"])

    emitted_states: list[DetectionFilterState] = []
    panel.filter_changed.connect(emitted_states.append)

    # Change confidence threshold
    panel._slider_conf.setValue(50)
    assert len(emitted_states) >= 1
    assert emitted_states[-1].min_confidence == 0.50

    # Toggle boxes checkbox
    panel._cb_boxes.setChecked(False)
    assert emitted_states[-1].show_boxes is False

    # Uncheck first class
    item = panel._list_classes.item(0)
    item.setCheckState(Qt.CheckState.Unchecked)
    assert "Ambrosia" in emitted_states[-1].hidden_classes


def test_detection_filter_panel_model_selector(tmp_path: Path, qtbot):
    panel = DetectionFilterPanel()
    qtbot.addWidget(panel)

    m1 = tmp_path / "model1.onnx"
    m2 = tmp_path / "model2.onnx"
    m1.touch()
    m2.touch()

    # Model 1 has custom classes
    c1 = tmp_path / "model1.json"
    c1.write_text('{"0": "WeedA", "1": "WeedB"}', encoding="utf-8")

    panel.set_available_models([m1, m2], active=m1)
    panel.load_classes_from_file(c1)

    assert panel._combo_model.count() == 2
    assert panel.selected_model() == m1
    assert panel._class_names == ["WeedA", "WeedB"]

    switched_models = []
    panel.model_changed.connect(switched_models.append)

    # Switch to model 2
    panel._combo_model.setCurrentIndex(1)
    assert len(switched_models) == 1
    assert switched_models[0] == m2


def test_image_viewer_sanity_grab(qtbot):
    viewer = ImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(400, 300)
    viewer.show()
    viewer.set_image(_dummy_image(200, 200))
    viewer.set_detections(_sample_detections())

    pix = viewer.grab()
    assert not pix.isNull()
    assert pix.width() == 400
    assert pix.height() == 300
