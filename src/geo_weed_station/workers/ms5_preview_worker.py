"""MS5 preview decoding outside the GUI thread."""
from pathlib import Path

from PySide6.QtCore import Signal, Slot

from geo_weed_station.core.image_io import read_ms5_image, ms5_rgb_preview
from geo_weed_station.workers.base_worker import ManagedWorker


class Ms5PreviewWorker(ManagedWorker):
    ready = Signal(int, str, object)

    def __init__(self, path: Path, generation: int) -> None:
        super().__init__()
        self.path = path
        self.generation = generation

    @Slot()
    def start_work(self) -> None:
        try:
            if not self.is_stop_requested:
                image = ms5_rgb_preview(read_ms5_image(self.path))
                image.setflags(write=False)
                if not self.is_stop_requested:
                    self.ready.emit(self.generation, str(self.path), image)
        except Exception as exc:
            if not self.is_stop_requested:
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()
