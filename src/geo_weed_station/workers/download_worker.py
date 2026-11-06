"""Background worker managing run archive downloads and unpacking without UI freezing."""

from __future__ import annotations

import threading

from PySide6.QtCore import Signal, Slot

from geo_weed_station.network.download_manager import DownloadManager
from geo_weed_station.network.exceptions import DownloadCancelled
from geo_weed_station.workers.base_worker import ManagedWorker


class DownloadWorker(ManagedWorker):
    """Worker managing streaming download and decompression of mission archive."""

    progress = Signal(int, int)  # (bytes_downloaded, total_bytes)
    download_finished = Signal(str)  # absolute path to extracted directory

    def __init__(self, download_manager: DownloadManager) -> None:
        super().__init__()
        self.download_manager = download_manager
        self._cancel_event = threading.Event()
        self._active_run_id: str | None = None

    @Slot(str)
    def download(self, run_id: str) -> None:
        """Download and unpack run archive in background."""
        self._active_run_id = run_id

        def progress_callback(downloaded: int, total: int | None) -> None:
            tot = -1 if total is None else total
            self.progress.emit(downloaded, tot)

        def is_cancelled() -> bool:
            return self._cancel_event.is_set() or self.is_stop_requested

        try:
            zip_path = self.download_manager.download_run(
                run_id=run_id,
                progress_cb=progress_callback,
                cancel_flag=is_cancelled,
            )
            extracted_dir = self.download_manager.extract_run(zip_path)
            self.download_finished.emit(str(extracted_dir))
        except DownloadCancelled:
            self.failed.emit("cancelled")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._active_run_id = None

    @Slot()
    def cancel(self) -> None:
        """Cancel ongoing download."""
        self._cancel_event.set()
