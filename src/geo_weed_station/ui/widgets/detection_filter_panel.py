"""Panel for interactive filtering of detection bounding boxes and labels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
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
    nms_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    hidden_classes: frozenset[str] = frozenset()


class DetectionFilterPanel(QWidget):
    """Widget allowing the user to toggle boxes, labels, confidence threshold, model, classifier, and per-class visibility."""

    model_changed = Signal(Path)
    classifier_changed = Signal(str)
    filter_changed = Signal(DetectionFilterState)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._block_signals = False
        self._base_classes: list[str] = []
        self._extra_classes: list[str] = []
        self._class_names: list[str] = []
        self._hidden_classes: set[str] = set()

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # 0. Active model selector group
        model_group = QGroupBox("Нейросетевая модель", self)
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(6, 6, 6, 6)
        model_layout.setSpacing(4)

        row_model = QHBoxLayout()
        self._combo_model = QComboBox(self)
        self._combo_model.currentIndexChanged.connect(self._on_combo_model_changed)
        self._btn_browse_model = QPushButton("...", self)
        self._btn_browse_model.setToolTip("Выбрать ONNX-модель с диска")
        self._btn_browse_model.setMaximumWidth(36)
        self._btn_browse_model.clicked.connect(self._on_browse_model)

        row_model.addWidget(self._combo_model, stretch=1)
        row_model.addWidget(self._btn_browse_model)
        model_layout.addLayout(row_model)

        main_layout.addWidget(model_group)

        # 0.5 Crop classifier selector group
        cls_group = QGroupBox("Классификация видов", self)
        cls_layout = QVBoxLayout(cls_group)
        cls_layout.setContentsMargins(6, 6, 6, 6)
        cls_layout.setSpacing(4)

        self._combo_classifier = QComboBox(self)
        self._combo_classifier.addItem("Отключена", userData="none")
        self._combo_classifier.addItem("C07 (10 категорий + фазы)", userData="c07")
        self._combo_classifier.addItem("C06 (23 вида)", userData="c06")
        self._combo_classifier.currentIndexChanged.connect(self._on_combo_classifier_changed)
        cls_layout.addWidget(self._combo_classifier)

        main_layout.addWidget(cls_group)

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

        # 2. Confidence and NMS threshold group
        thresh_group = QGroupBox("Пороги детекции", self)
        thresh_layout = QVBoxLayout(thresh_group)
        thresh_layout.setContentsMargins(6, 6, 6, 6)
        thresh_layout.setSpacing(4)

        # 2.1 Confidence row
        lbl_conf = QLabel("Уверенность (Confidence):", self)
        thresh_layout.addWidget(lbl_conf)

        row_conf = QHBoxLayout()
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

        row_conf.addWidget(self._slider_conf, stretch=1)
        row_conf.addWidget(self._spin_conf)
        thresh_layout.addLayout(row_conf)

        # 2.2 NMS IoU threshold row
        lbl_iou = QLabel("Подавление наложений (NMS IoU):", self)
        thresh_layout.addWidget(lbl_iou)

        row_iou = QHBoxLayout()
        self._slider_iou = QSlider(Qt.Orientation.Horizontal, self)
        self._slider_iou.setRange(0, 100)
        self._slider_iou.setValue(45)

        self._spin_iou = QDoubleSpinBox(self)
        self._spin_iou.setRange(0.0, 1.0)
        self._spin_iou.setSingleStep(0.05)
        self._spin_iou.setValue(0.45)
        self._spin_iou.setDecimals(2)

        self._slider_iou.valueChanged.connect(self._on_slider_iou_changed)
        self._spin_iou.valueChanged.connect(self._on_spin_iou_changed)

        row_iou.addWidget(self._slider_iou, stretch=1)
        row_iou.addWidget(self._spin_iou)
        thresh_layout.addLayout(row_iou)

        main_layout.addWidget(thresh_group)

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
    def set_available_models(self, models: Sequence[Path], active: Path | None = None) -> None:
        """Populate the model selector combobox with available ONNX models."""
        self._block_signals = True
        self._combo_model.clear()
        active_idx = -1
        active_resolved = active.resolve() if active is not None and active.exists() else None

        for i, m in enumerate(models):
            p = Path(m)
            self._combo_model.addItem(p.name, userData=str(p))
            if active_resolved is not None and p.exists() and p.resolve() == active_resolved:
                active_idx = i

        if active_idx == -1 and active is not None and Path(active).is_file():
            p = Path(active)
            self._combo_model.addItem(p.name, userData=str(p))
            active_idx = self._combo_model.count() - 1

        if active_idx >= 0:
            self._combo_model.setCurrentIndex(active_idx)

        self._block_signals = False

    def selected_model(self) -> Path | None:
        """Return currently selected ONNX model path or None."""
        data = self._combo_model.currentData()
        return Path(data) if data else None

    def selected_classifier_type(self) -> str:
        """Return currently selected classifier type: 'none', 'c07', or 'c06'."""
        data = self._combo_classifier.currentData()
        return str(data) if data else "none"

    def set_classifier_type(self, cls_type: str) -> None:
        """Set active classifier type in the combobox."""
        target = (cls_type or "none").lower()
        self._block_signals = True
        for i in range(self._combo_classifier.count()):
            if str(self._combo_classifier.itemData(i)).lower() == target:
                self._combo_classifier.setCurrentIndex(i)
                break
        self._block_signals = False

    def load_classes_from_file(self, classes_path: Path) -> None:
        """Load and display classes for current mode using JSON file metadata and thresholds."""
        self.load_classes_for_mode(
            cls_type=self.selected_classifier_type(),
            classes_path=classes_path,
            update_thresholds=True,
        )

    def load_classes_for_mode(
        self,
        cls_type: str | None = None,
        classes_path: Path | None = None,
        *,
        update_thresholds: bool = True,
    ) -> None:
        """Populate base classes according to active classifier mode ('none', 'c07', 'c06') and detector metadata."""
        from geo_weed_station.core.onnx_detector import (
            get_classifier_class_names,
            load_model_metadata,
        )

        active_cls = (cls_type if cls_type is not None else self.selected_classifier_type()).lower().strip()
        meta = None
        if classes_path is not None and Path(classes_path).is_file():
            meta = load_model_metadata(Path(classes_path))

        if active_cls in ("c07", "c06"):
            base_names = get_classifier_class_names(active_cls)
        elif meta is not None:
            base_names = list(meta.classes.values())
        else:
            base_names = []

        self.set_classes(base_names, preserve_hidden=False)

        if update_thresholds and meta is not None and (
            meta.confidence_threshold is not None or meta.nms_iou_threshold is not None
        ):
            self._block_signals = True
            if meta.confidence_threshold is not None:
                conf = max(0.0, min(1.0, float(meta.confidence_threshold)))
                self._spin_conf.setValue(conf)
                self._slider_conf.setValue(int(round(conf * 100)))
            if meta.nms_iou_threshold is not None:
                iou = max(0.0, min(1.0, float(meta.nms_iou_threshold)))
                self._spin_iou.setValue(iou)
                self._slider_iou.setValue(int(round(iou * 100)))
            self._block_signals = False

        self._emit_filter_changed()

    def set_classes(self, names: Sequence[str], *, preserve_hidden: bool = True) -> None:
        """Populate the base list of classes, pruning stale hidden classes."""
        from geo_weed_station.core.onnx_detector import is_background_class

        clean_names: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name and not is_background_class(name) and name not in seen:
                clean_names.append(name)
                seen.add(name)

        self._base_classes = list(clean_names)
        self._extra_classes = []
        self._class_names = list(self._base_classes)
        if preserve_hidden:
            self._hidden_classes &= set(self._class_names)
        else:
            self._hidden_classes.clear()

        self._rebuild_class_list_widget()

    def append_extra_classes(self, extra_names: Iterable[str]) -> bool:
        """Append non-standard classes (e.g. from preloaded annotations) after base model classes."""
        from geo_weed_station.core.onnx_detector import is_background_class

        cls_type = self.selected_classifier_type()
        known = set(self._base_classes) | set(self._extra_classes)
        new_extras: list[str] = []
        for raw_name in sorted({str(n).strip() for n in extra_names if n}):
            if not raw_name or is_background_class(raw_name):
                continue
            if cls_type in ("c07", "c06") and raw_name.lower() == "plant":
                continue
            if raw_name not in known:
                new_extras.append(raw_name)
                known.add(raw_name)

        if not new_extras:
            return False

        self._extra_classes.extend(new_extras)
        self._class_names = list(self._base_classes) + list(self._extra_classes)
        self._rebuild_class_list_widget()
        return True

    def reset_extra_classes(self) -> None:
        """Remove any dynamically appended extra classes, restoring only the active base classes."""
        if not self._extra_classes:
            return
        self._extra_classes = []
        self._class_names = list(self._base_classes)
        self._hidden_classes &= set(self._class_names)
        self._rebuild_class_list_widget()

    def _rebuild_class_list_widget(self) -> None:
        """Rebuild QListWidget items from _class_names while keeping check states."""
        self._block_signals = True
        self._list_classes.blockSignals(True)
        self._list_classes.setUpdatesEnabled(False)
        try:
            self._list_classes.clear()
            for name in self._class_names:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if name in self._hidden_classes:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
                self._list_classes.addItem(item)
        finally:
            self._list_classes.setUpdatesEnabled(True)
            self._list_classes.blockSignals(False)
            self._block_signals = False

    def current_state(self) -> DetectionFilterState:
        """Return current filter configuration."""
        return DetectionFilterState(
            show_boxes=self._cb_boxes.isChecked(),
            show_labels=self._cb_labels.isChecked(),
            show_confidence=self._cb_confidence.isChecked(),
            min_confidence=float(self._spin_conf.value()),
            nms_iou_threshold=float(self._spin_iou.value()),
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

        iou = max(0.0, min(1.0, float(state.nms_iou_threshold)))
        self._spin_iou.setValue(iou)
        self._slider_iou.setValue(int(round(iou * 100)))

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

    def _on_slider_iou_changed(self, value: int) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        self._spin_iou.setValue(value / 100.0)
        self._block_signals = False
        self._emit_filter_changed()

    def _on_spin_iou_changed(self, value: float) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        self._slider_iou.setValue(int(round(value * 100)))
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

    def _on_combo_model_changed(self, index: int) -> None:
        if self._block_signals or index < 0:
            return
        model_path = self.selected_model()
        if model_path is not None:
            self.model_changed.emit(model_path)

    def _on_combo_classifier_changed(self, index: int) -> None:
        if self._block_signals or index < 0:
            return
        cls_type = self.selected_classifier_type()
        self.classifier_changed.emit(cls_type)

    def _on_browse_model(self) -> None:
        from geo_weed_station.core.onnx_detector import get_models_dir

        current_model = self.selected_model()
        start_dir = str(get_models_dir(current_model))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать ONNX-модель",
            start_dir,
            "ONNX Models (*.onnx);;All Files (*.*)",
        )
        if not path:
            return
        model_p = Path(path)
        # Check if already in combobox
        idx = -1
        for i in range(self._combo_model.count()):
            if self._combo_model.itemData(i) == str(model_p):
                idx = i
                break
        if idx == -1:
            self._combo_model.addItem(model_p.name, userData=str(model_p))
            idx = self._combo_model.count() - 1

        self._combo_model.setCurrentIndex(idx)

    def _emit_filter_changed(self) -> None:
        self.filter_changed.emit(self.current_state())
