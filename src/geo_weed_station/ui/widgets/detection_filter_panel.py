"""Panel for interactive filtering of detection bounding boxes and labels."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set
from pydantic import BaseModel, ConfigDict, Field
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class DetectionFilterState(BaseModel):
    """Immutable state describing visual detection filters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    show_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    hidden_classes: frozenset[str] = frozenset()


class DetectionFilterPanel(QWidget):
    """Widget allowing the user to toggle boxes, labels, confidence threshold and per-class visibility."""

    filter_changed = Signal(DetectionFilterState)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._block_signals = False
        self._class_names: List[str] = []
        self._hidden_classes: Set[str] = set()

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # 1. Visibility group
        vis_group = QGroupBox("Отображение детекций", self)
        vis_layout = QVBoxLayout(vis_group)
        vis_layout.setContentsMargins(6, 6, 6, 6)

        self._cb_boxes = QCheckBox("Показывать рамки (BBox)", self)
        self._cb_boxes.setChecked(True)
        self._cb_boxes.toggled.connect(self._on_control_changed)
        vis_layout.addWidget(self._cb_boxes)

        self._cb_labels = QCheckBox("Названия классов", self)
        self._cb_labels.setChecked(True)
        self._cb_labels.toggled.connect(self._on_control_changed)
        vis_layout.addWidget(self._cb_labels)

        self._cb_confidence = QCheckBox("Оценка уверенности (Confidence)", self)
        self._cb_confidence.setChecked(True)
        self._cb_confidence.toggled.connect(self._on_control_changed)
        vis_layout.addWidget(self._cb_confidence)

        main_layout.addWidget(vis_group)

        # 2. Confidence threshold group
        conf_group = QGroupBox("Порог уверенности", self)
        conf_layout = QVBoxLayout(conf_group)
        conf_layout.setContentsMargins(6, 6, 6, 6)

        row_layout = QHBoxLayout()
        self._slider_conf = QSlider(Qt.Orientation.Horizontal, self)
        self._slider_conf.setRange(0, 100)
        self._slider_conf.setValue(25)

        self._spin_conf = QDoubleSpinBox(self)
        self._spin_conf.setRange(0.0, 1.0)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setValue(0.25)
        self._spin_conf.setDecimals(2)

        self._slider_conf.valueChanged.connect(self._on_slider_changed)
        self._spin_conf.valueChanged.connect(self._on_spin_changed)

        row_layout.addWidget(self._slider_conf)
        row_layout.addWidget(self._spin_conf)
        conf_layout.addLayout(row_layout)

        main_layout.addWidget(conf_group)

        # 3. Classes checklist group
        self._classes_group = QGroupBox("Классы сорняков", self)
        classes_layout = QVBoxLayout(self._classes_group)
        classes_layout.setContentsMargins(6, 6, 6, 6)

        self._list_classes = QListWidget(self)
        self._list_classes.itemChanged.connect(self._on_class_item_changed)
        classes_layout.addWidget(self._list_classes)

        main_layout.addWidget(self._classes_group)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def set_classes(self, names: Sequence[str]) -> None:
        """Populate the list of classes, preserving unchecked state for known names."""
        self._class_names = list(names)
        self._block_signals = True
        self._list_classes.clear()

        for name in self._class_names:
            item = QListWidgetItem(name, self._list_classes)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if name in self._hidden_classes:
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setCheckState(Qt.CheckState.Checked)

        self._block_signals = False

    def current_state(self) -> DetectionFilterState:
        """Return current filter configuration."""
        return DetectionFilterState(
            show_boxes=self._cb_boxes.isChecked(),
            show_labels=self._cb_labels.isChecked(),
            show_confidence=self._cb_confidence.isChecked(),
            min_confidence=float(self._spin_conf.value()),
            hidden_classes=frozenset(self._hidden_classes),
        )

    def set_state(self, state: DetectionFilterState) -> None:
        """Apply a filter configuration state."""
        self._block_signals = True
        self._cb_boxes.setChecked(state.show_boxes)
        self._cb_labels.setChecked(state.show_labels)
        self._cb_confidence.setChecked(state.show_confidence)
        conf = max(0.0, min(1.0, float(state.min_confidence)))
        self._spin_conf.setValue(conf)
        self._slider_conf.setValue(int(round(conf * 100)))

        self._hidden_classes = set(state.hidden_classes)
        for i in range(self._list_classes.count()):
            item = self._list_classes.item(i)
            if item.text() in self._hidden_classes:
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setCheckState(Qt.CheckState.Checked)

        self._block_signals = False
        self.filter_changed.emit(self.current_state())

    # -------------------------------------------------------------------------
    # Internal event handlers
    # -------------------------------------------------------------------------
    def _on_slider_changed(self, value: int) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        self._spin_conf.setValue(value / 100.0)
        self._block_signals = False
        self._emit_filter_changed()

    def _on_spin_changed(self, value: float) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        self._slider_conf.setValue(int(round(value * 100)))
        self._block_signals = False
        self._emit_filter_changed()

    def _on_control_changed(self) -> None:
        if self._block_signals:
            return
        self._emit_filter_changed()

    def _on_class_item_changed(self, item: QListWidgetItem) -> None:
        if self._block_signals:
            return
        name = item.text()
        if item.checkState() == Qt.CheckState.Unchecked:
            self._hidden_classes.add(name)
        else:
            self._hidden_classes.discard(name)
        self._emit_filter_changed()

    def _emit_filter_changed(self) -> None:
        self.filter_changed.emit(self.current_state())
