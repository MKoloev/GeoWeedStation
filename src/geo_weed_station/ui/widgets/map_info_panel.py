"""Side panel displaying agronomic field statistics and selected weed inspection details with thumbnail."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geo_weed_station.schemas.map_models import MapStatistics, WeedPoint

logger = logging.getLogger(__name__)


class MapInfoPanel(QWidget):
    """Panel showing aggregated field metrics and inspection popup for a clicked weed point."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 1. Summary statistics group
        stats_group = QGroupBox("Сводные агрономические метрики", container)
        stats_layout = QFormLayout(stats_group)
        stats_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_total_weeds = QLabel("0", stats_group)
        self.lbl_area = QLabel("0.0 м²", stats_group)
        self.lbl_density = QLabel("0.00 шт/м²", stats_group)
        self.lbl_critical = QLabel("0.0 %", stats_group)

        stats_layout.addRow("Всего сорняков:", self.lbl_total_weeds)
        stats_layout.addRow("Обследованная площадь:", self.lbl_area)
        stats_layout.addRow("Средняя плотность:", self.lbl_density)
        stats_layout.addRow("Критическая зона (> порога):", self.lbl_critical)

        layout.addWidget(stats_group)

        # 2. Breakdown by classes table
        classes_group = QGroupBox("Распределение по видам", container)
        classes_layout = QVBoxLayout(classes_group)
        classes_layout.setContentsMargins(4, 4, 4, 4)

        self.table_classes = QTableWidget(0, 2, classes_group)
        self.table_classes.setHorizontalHeaderLabels(["Класс сорняка", "Кол-во"])
        self.table_classes.horizontalHeader().setStretchLastSection(True)
        self.table_classes.verticalHeader().setVisible(False)
        self.table_classes.setMaximumHeight(150)
        classes_layout.addWidget(self.table_classes)

        layout.addWidget(classes_group)

        # 3. Selected Point Inspector Group
        self.point_group = QGroupBox("Детали выбранного объекта", container)
        point_layout = QVBoxLayout(self.point_group)
        point_layout.setContentsMargins(6, 6, 6, 6)

        form_point = QFormLayout()
        self.lbl_point_id = QLabel("-", self.point_group)
        self.lbl_point_class = QLabel("-", self.point_group)
        self.lbl_point_conf = QLabel("-", self.point_group)
        self.lbl_point_coords = QLabel("-", self.point_group)
        self.lbl_point_frame = QLabel("-", self.point_group)

        form_point.addRow("ID объекта:", self.lbl_point_id)
        form_point.addRow("Класс:", self.lbl_point_class)
        form_point.addRow("Уверенность:", self.lbl_point_conf)
        form_point.addRow("Координаты (X, Y):", self.lbl_point_coords)
        form_point.addRow("Кадр-источник:", self.lbl_point_frame)
        point_layout.addLayout(form_point)

        # Thumbnail image preview
        self.lbl_thumbnail = QLabel("Нет миниатюры", self.point_group)
        self.lbl_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_thumbnail.setMinimumHeight(120)
        self.lbl_thumbnail.setStyleSheet("background-color: #222; border: 1px solid #444;")
        point_layout.addWidget(self.lbl_thumbnail)

        self.btn_clear_selection = QPushButton("Сбросить выбор", self.point_group)
        self.btn_clear_selection.clicked.connect(self.clear_selection)
        point_layout.addWidget(self.btn_clear_selection)

        self.point_group.setVisible(False)  # Initially hidden until point selected
        layout.addWidget(self.point_group)

        layout.addStretch(1)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def set_statistics(self, stats: MapStatistics) -> None:
        """Update field agronomic metrics and class distribution table."""
        self.lbl_total_weeds.setText(str(stats.total_weeds))
        self.lbl_area.setText(f"{stats.surveyed_area_m2:.1f} м²")
        self.lbl_density.setText(f"{stats.mean_density_per_m2:.2f} шт/м²")
        self.lbl_critical.setText(f"{stats.critical_area_percent:.1f} %")

        # Update classes table
        self.table_classes.setRowCount(len(stats.by_class))
        for row, (c_name, count) in enumerate(sorted(stats.by_class.items())):
            item_name = QTableWidgetItem(c_name)
            item_count = QTableWidgetItem(str(count))
            item_count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_classes.setItem(row, 0, item_name)
            self.table_classes.setItem(row, 1, item_count)

    def show_point(self, point: WeedPoint, thumbnail: np.ndarray | None = None) -> None:
        """Display inspector details and thumbnail for the selected weed point."""
        self.point_group.setVisible(True)
        self.lbl_point_id.setText(point.weed_id)
        self.lbl_point_class.setText(point.class_name)
        self.lbl_point_conf.setText(f"{point.confidence * 100.0:.1f} %")
        self.lbl_point_coords.setText(f"X: {point.x:.3f}, Y: {point.y:.3f}")
        self.lbl_point_frame.setText(point.source_frame)

        if thumbnail is not None and thumbnail.size > 0:
            h, w = thumbnail.shape[:2]
            arr = np.ascontiguousarray(thumbnail)
            if arr.ndim == 2:
                qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
            elif arr.shape[2] == 3:
                qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            else:
                qimg = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()

            pixmap = QPixmap.fromImage(qimg).scaled(
                200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_thumbnail.setPixmap(pixmap)
        else:
            self.lbl_thumbnail.clear()
            self.lbl_thumbnail.setText("Миниатюра отсутствует")

    def clear_selection(self) -> None:
        """Hide inspector panel."""
        self.point_group.setVisible(False)
        self.lbl_thumbnail.clear()
        self.lbl_thumbnail.setText("Нет миниатюры")
