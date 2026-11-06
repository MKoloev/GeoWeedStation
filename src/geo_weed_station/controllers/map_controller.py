"""Map controller coordinating MapEngine data ingestion, heatmap generation and locked state."""

from __future__ import annotations

import logging
from typing import Optional
from PySide6.QtCore import QObject, Signal, Slot

from geo_weed_station.core.map_engine.errors import GeoreferenceMissingError
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.schemas.map_models import HeatmapGrid, MapStatistics
from geo_weed_station.schemas.onboard_api import SummaryDetections
from geo_weed_station.schemas.payload import FramePayload

logger = logging.getLogger(__name__)

MAP_LOCKED_DEFAULT_REASON = "Геопривязка отсутствует. Доступен только просмотр исходных изображений"


class MapController(QObject):
    """Bridge coordinating spatial projections, heatmap calculation and lock signals."""

    map_updated = Signal()
    statistics_changed = Signal(MapStatistics)
    map_locked_changed = Signal(bool, str)

    def __init__(self, engine: MapEngine, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._engine = engine

    @property
    def engine(self) -> MapEngine:
        return self._engine

    @property
    def is_locked(self) -> bool:
        return self._engine.is_locked()

    @Slot(FramePayload)
    def on_frame(self, payload: FramePayload) -> None:
        """Ingest a single frame payload, recomputing map layer or triggering lock if unreferenced."""
        if not payload.has_georef():
            logger.warning("Frame %s has no georeferencing metadata", payload.frame_id)
            self._engine._is_locked = True
            self.map_locked_changed.emit(True, MAP_LOCKED_DEFAULT_REASON)
            return

        try:
            self._engine.ingest_frame(payload)
            self._engine.build_heatmap()
            self.map_updated.emit()
            self.statistics_changed.emit(self._engine.statistics())
            self.map_locked_changed.emit(False, "")
        except GeoreferenceMissingError as exc:
            logger.warning("Georeference missing error: %s", exc)
            self.map_locked_changed.emit(True, str(exc))

    @Slot(SummaryDetections)
    def on_summary(self, summary: SummaryDetections) -> None:
        """Ingest mission detections summary from onboard computer."""
        try:
            self._engine.ingest_summary(summary)
            self._engine.build_heatmap()
            self.map_updated.emit()
            self.statistics_changed.emit(self._engine.statistics())
            self.map_locked_changed.emit(False, "")
        except GeoreferenceMissingError as exc:
            logger.warning("Summary lacks georeference: %s", exc)
            self.map_locked_changed.emit(True, str(exc))

    @Slot(float)
    def set_cell_size(self, cell_m: float) -> None:
        """Update heatmap grid cell resolution."""
        if self._engine.weed_map.points:
            self._engine.build_heatmap(cell_m=cell_m)
            self.map_updated.emit()
            self.statistics_changed.emit(self._engine.statistics())

    def reset(self) -> None:
        """Clear all mapping data and unlock state."""
        self._engine.reset()
        self.map_locked_changed.emit(False, "")
        self.map_updated.emit()
        self.statistics_changed.emit(self._engine.statistics())
