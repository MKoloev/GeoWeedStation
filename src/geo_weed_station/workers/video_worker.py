"""Background video playback worker with frame skipping and bounded in-flight frames."""

from __future__ import annotations

from collections.abc import Callable

import cv2
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, QMetaObject, Qt, Q_ARG

from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.schemas.entities import DetectedWeed
from geo_weed_station.workers.base_worker import ManagedWorker
from geo_weed_station.workers.frame_queue import FrameQueue


class VideoWorker(ManagedWorker):
    """Decodes video (cv2.VideoCapture) and runs inference strictly inside its own thread.

    Inference runs on every (skip + 1)-th frame; on intermediate frames the last detections
    are held. The UI should call `frame_consumed()` after rendering each frame; if more than
    `max_inflight` frames are unconsumed, new frames are dropped instead of piling up.
    """

    frame_ready = Signal(object, list)  # ndarray (copy, RGB), list[DetectedWeed]
    position_changed = Signal(int, int)  # frame index, total frames
    playback_ended = Signal()

    def __init__(
        self,
        detector_factory: Callable[[], BaseDetector] | None = None,
        max_inflight: int = 3,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._detector_factory = detector_factory
        self._detector: BaseDetector | None = None
        self._reload_detector: bool = False
        self._cap: cv2.VideoCapture | None = None
        self._total = 0
        self._fps = 25.0
        self._next_index = 0
        self._skip = 0
        self._last_dets: list[DetectedWeed] = []
        self._inflight = FrameQueue(maxlen=max_inflight)
        self.dropped_frames = 0
        self._at_eof = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # --------------------------------------------------------------- helpers
    @property
    def detector_created(self) -> bool:
        return self._detector is not None

    @property
    def is_playing(self) -> bool:
        return self._timer.isActive()

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _ensure_detector(self) -> BaseDetector | None:
        if self._reload_detector:
            self._reload_detector = False
            if self._detector is not None:
                try:
                    self._detector.close()
                except Exception:
                    pass
                self._detector = None
        if self._detector is None and self._detector_factory is not None:
            detector = self._detector_factory()
            detector.load()
            self._detector = detector
        return self._detector

    def _step(self, force_inference: bool = False) -> bool:
        """Decode one frame and emit it. Returns False at end of stream."""
        if self._cap is None or self.is_stop_requested:
            return False
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._at_eof = True
            return False
        self._at_eof = False
        index = self._next_index
        self._next_index += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._detector_factory is not None and (
            force_inference or index % (self._skip + 1) == 0
        ):
            try:
                detector = self._ensure_detector()
                if detector is not None:
                    self._last_dets = list(detector.detect(rgb))
            except Exception as exc:  # noqa: BLE001
                logger.error("Video inference failed on frame %d: %s", index, exc)
                self.failed.emit(f"Inference failed: {exc}")

        self.position_changed.emit(index, self._total)
        if self._inflight.is_full:
            self.dropped_frames += 1  # UI is behind: drop instead of piling up
        else:
            self._inflight.put(index)
            self.frame_ready.emit(rgb.copy(), list(self._last_dets))
        return True

    # ----------------------------------------------------------------- slots
    @Slot(str)
    def open(self, path: str) -> None:
        """Open a video file and show its first frame."""
        if self.is_stop_requested:
            return
        if self.thread() != QThread.currentThread():
            QMetaObject.invokeMethod(self, "open", Qt.ConnectionType.QueuedConnection, Q_ARG(str, str(path)))
            return
        if self.is_stop_requested:
            return
        self._timer.stop()
        self._release_capture()
        self._inflight.clear()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            self.failed.emit(f"Cannot open video: {path}")
            return
        self._cap = cap
        self._total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 1e-3 else 25.0
        self._next_index = 0
        self._last_dets = []
        self._at_eof = False
        self._step(force_inference=True)

    @Slot(str)
    def open_video(self, path: str) -> None:
        """Alias for open(path)."""
        self.open(path)

    @Slot()
    def play(self) -> None:
        if self._cap is None:
            return
        if self._at_eof or (self._total > 0 and self._next_index >= self._total):
            self.seek(0)
        self._timer.start(max(1, int(1000.0 / self._fps)))

    @Slot()
    def pause(self) -> None:
        self._timer.stop()
        super().pause()

    @Slot(int)
    def seek(self, frame_index: int) -> None:
        if self._cap is None:
            return
        self._inflight.clear()
        target = max(0, min(int(frame_index), max(0, self._total - 1)))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self._next_index = target
        self._at_eof = False
        self._step(force_inference=True)

    @Slot(int)
    def set_frame_skip(self, n: int) -> None:
        self._skip = max(0, int(n))

    @Slot()
    def frame_consumed(self) -> None:
        """UI acknowledgement that a frame has been rendered."""
        self._inflight.get(timeout=0)

    @Slot(object)
    def set_detector_factory(self, factory: Callable[[], BaseDetector] | None) -> None:
        """Update detector factory dynamically and reload detector on next step."""
        self._detector_factory = factory
        if self.thread() == QThread.currentThread():
            if self._detector is not None:
                try:
                    self._detector.close()
                except Exception:
                    pass
                self._detector = None
        else:
            self._reload_detector = True
        self._last_dets = []

    @Slot()
    def close_resources(self) -> None:
        """Release VideoCapture and ONNX session; safe to call repeatedly."""
        self._timer.stop()
        try:
            self._release_capture()
        finally:
            if self._detector is not None:
                self._detector.close()
                self._detector = None
            if self.is_stop_requested:
                self.finished.emit()

    @Slot()
    def _tick(self) -> None:
        if self.is_stop_requested:
            self.close_resources()
            self.finished.emit()
            return
        if not self._step():
            self._timer.stop()
            self.playback_ended.emit()

    @Slot()
    def request_stop(self) -> None:
        super().request_stop()
        self._inflight.clear()
        if self.thread() != QThread.currentThread():
            QMetaObject.invokeMethod(self, "close_resources", Qt.ConnectionType.QueuedConnection)
        else:
            self.close_resources()
