"""Dispatch background thumbnails to surviving graphics items in the GUI thread."""
from pathlib import Path
import weakref

from PySide6.QtCore import QObject, QThreadPool, Qt, Signal, Slot
from shiboken6 import isValid

from geo_weed_station.workers.thumbnail_loader import ThumbnailJob


class ThumbnailController(QObject):
    result = Signal(object)

    def __init__(self, path, callback):
        super().__init__()
        self.path = Path(path)
        self.callback = weakref.WeakMethod(callback)
        self.result.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    def start(self):
        QThreadPool.globalInstance().start(ThumbnailJob(self))

    @Slot(object)
    def _deliver(self, thumbnail):
        callback = self.callback()
        if thumbnail is not None and callback is not None and isValid(callback.__self__):
            callback(thumbnail)
