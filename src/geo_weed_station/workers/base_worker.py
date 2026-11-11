"""Base classes and utilities for background worker threads in GeoWeedStation."""

from __future__ import annotations

import encodings.idna  # noqa: F401
import gc
import netrc  # noqa: F401
import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot


class ManagedWorker(QObject):
    """Abstract base class for managed background workers providing standard lifecycle signals."""

    started_ok = Signal()
    finished = Signal()
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._is_paused = threading.Event()
        self._stop_event = threading.Event()
        self._running = False

    @Slot()
    def start_work(self) -> None:
        """Entry slot invoked when worker thread starts. Should be overridden by subclasses."""
        self.started_ok.emit()

    @Slot()
    def pause(self) -> None:
        """Pause worker processing loop if supported."""
        self._is_paused.set()

    @Slot()
    def resume(self) -> None:
        """Resume paused worker loop."""
        self._is_paused.clear()

    @Slot()
    def request_stop(self) -> None:
        """Signal safe cooperative worker termination."""
        self._stop_event.set()
        self.resume()  # Unblock if paused

    @property
    def is_stop_requested(self) -> bool:
        """Check whether stop was requested."""
        return self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        """Check whether worker is currently paused."""
        return self._is_paused.is_set()


def start_in_thread(worker: ManagedWorker, name: str) -> QThread:
    """Move worker to a new QThread, wire lifecycle signals, and start the thread."""
    gc.collect()
    thread = QThread()
    thread.setObjectName(name)
    worker.moveToThread(thread)

    thread.started.connect(worker.start_work, Qt.DirectConnection)
    worker.finished.connect(thread.quit)

    thread.start()
    return thread
