"""QRunnable tasks indexing XMP presence of files via a thread pool."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from geo_weed_station.core.xmp_parser import has_xmp


class XmpIndexSignals(QObject):
    """Signal carrier for XmpIndexTask (QRunnable is not a QObject)."""

    indexed = Signal(int, bool)  # entry index, has_xmp
    finished = Signal()  # emitted once when all chunks are done


class XmpIndexTask(QRunnable):
    """Checks XMP presence for a chunk of (index, path) pairs."""

    def __init__(
        self,
        items: Sequence[tuple[int, str]],
        signals: XmpIndexSignals,
        cancel_event: threading.Event,
        on_done=None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._items = list(items)
        self._signals = signals
        self._cancel = cancel_event
        self._on_done = on_done

    def run(self) -> None:
        try:
            for index, path in self._items:
                if self._cancel.is_set():
                    break
                self._signals.indexed.emit(index, has_xmp(Path(path)))
        finally:
            if self._on_done is not None:
                self._on_done()


class XmpIndexJob:
    """Handle of a running indexing job: signals + cancellation."""

    def __init__(
        self,
        signals: XmpIndexSignals,
        cancel_event: threading.Event,
        submit=lambda: None,
    ) -> None:
        self.signals = signals
        self._cancel = cancel_event
        self._submit = submit

    def cancel(self) -> None:
        self._cancel.set()

    def start(self) -> None:
        """Submit all chunks to the thread pool."""
        self._submit()


def start_xmp_indexing(
    items: Sequence[tuple[int, str]],
    pool: QThreadPool | None = None,
    chunk_size: int = 16,
) -> XmpIndexJob:
    """Prepare chunked XMP indexing; connect to `job.signals`, then call `job.start()`.

    Tasks are not submitted until `start()` so that no signal can be emitted before
    the caller has connected its slots.
    """
    pool = pool or QThreadPool.globalInstance()
    signals = XmpIndexSignals()
    cancel = threading.Event()
    items = list(items)
    chunks: list[list[tuple[int, str]]] = [
        items[i : i + chunk_size] for i in range(0, len(items), chunk_size)
    ]
    remaining = {"n": len(chunks)}
    lock = threading.Lock()

    def on_done() -> None:
        with lock:
            remaining["n"] -= 1
            last = remaining["n"] == 0
        if last:
            signals.finished.emit()

    def submit() -> None:
        if not chunks:
            signals.finished.emit()
            return
        for chunk in chunks:
            pool.start(XmpIndexTask(chunk, signals, cancel, on_done))

    return XmpIndexJob(signals, cancel, submit)
