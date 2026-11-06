"""Background worker for offline batch image processing (XMP + optional ONNX inference)."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional
from PySide6.QtCore import Signal, Slot

from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.session import ProcessingSession
from geo_weed_station.core.source_factory import FrameSource
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.schemas.session import FileEntry, ProcessingStatus
from geo_weed_station.workers.base_worker import ManagedWorker


class BatchImageWorker(ManagedWorker):
    """Processes a batch of file entries sequentially in a background thread."""

    progress = Signal(int, int, str)  # i, N, status text
    frame_processed = Signal(FramePayload)
    entry_status_changed = Signal(int, object)  # index, ProcessingStatus
    batch_finished = Signal(int, int)  # succeeded, failed

    def __init__(
        self,
        source: FrameSource,
        detector_factory: Optional[Callable[[], BaseDetector]] = None,
        session: Optional[ProcessingSession] = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._detector_factory = detector_factory
        self._detector: Optional[BaseDetector] = None  # lazily created
        self._session = session
        self._cancel_event = threading.Event()

    @property
    def detector_created(self) -> bool:
        """True once the detector was instantiated (first inference)."""
        return self._detector is not None

    def _ensure_detector(self) -> BaseDetector:
        if self._detector is None:
            if self._detector_factory is None:
                raise RuntimeError("Inference requested but no detector factory is configured")
            detector = self._detector_factory()
            detector.load()
            self._detector = detector
        return self._detector

    def _set_status(self, index: int, status: ProcessingStatus, count: Optional[int] = None) -> None:
        if self._session is not None:
            self._session.update_entry(index, status, count)
        self.entry_status_changed.emit(index, status)

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set() or self.is_stop_requested

    @Slot(list, bool)
    def process(self, entries: List[FileEntry], run_inference: bool) -> None:
        """Process entries one by one; corrupted files become ERROR and the batch continues."""
        total = len(entries)
        ok = 0
        failed = 0
        try:
            for i, entry in enumerate(entries):
                if self._cancelled():
                    break
                self.progress.emit(i, total, f"инференс кадра {i + 1} из {total}")
                self._set_status(i, ProcessingStatus.PROCESSING)
                try:
                    payload = self._source.load(entry)
                    count: Optional[int] = len(payload.detections) if payload.detections else None
                    if run_inference:
                        detector = self._ensure_detector()
                        detections = detector.detect(payload.image)
                        payload = payload.model_copy(update={"detections": detections})
                        count = len(detections)
                    status = (
                        ProcessingStatus.NO_METADATA
                        if payload.pose is None
                        else ProcessingStatus.DONE
                    )
                    # Frames cross the thread boundary as copies only
                    safe = payload.model_copy(
                        update={"image": None if payload.image is None else payload.image.copy()}
                    )
                    self._set_status(i, status, count)
                    self.frame_processed.emit(safe)
                    ok += 1
                except Exception as exc:  # noqa: BLE001 - batch must survive any single failure
                    self._set_status(i, ProcessingStatus.ERROR)
                    self.failed.emit(f"{entry.path}: {exc}")
                    failed += 1
            else:
                self.progress.emit(total, total, f"инференс кадра {total} из {total}")
        finally:
            if self._detector is not None:
                self._detector.close()
                self._detector = None
            if self._session is not None:
                self._session.flush()
            self.batch_finished.emit(ok, failed)
            self.finished.emit()

    @Slot()
    def cancel(self) -> None:
        """Cooperatively cancel the batch; processed results stay in the session."""
        self._cancel_event.set()
