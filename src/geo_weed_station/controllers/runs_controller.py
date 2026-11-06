"""Controller managing onboard runs catalog, archive streaming download, local inference, and deletion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from geo_weed_station.config.settings import AppSettings
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.source_factory import FolderSource
from geo_weed_station.network.download_manager import DownloadManager
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.schemas.onboard_api import DeleteRunResponse, DroneRun
from geo_weed_station.ui.dialogs.confirm_dialog import ConfirmDialog
from geo_weed_station.ui.main_window import TAB_RUNS, MainWindow
from geo_weed_station.ui.widgets.runs_table import RunsTable
from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.batch_image_worker import BatchImageWorker
from geo_weed_station.workers.command_worker import CommandWorker
from geo_weed_station.workers.download_worker import DownloadWorker

logger = logging.getLogger(__name__)


class RunsController(QObject):
    """Coordinates the runs history catalog, archive download, post-flight local inference, and deletion."""

    runs_updated = Signal(list)
    download_finished = Signal(str)

    def __init__(
        self,
        window: MainWindow,
        map_ctrl: MapController,
        settings: AppSettings,
        client_getter: Callable[[], DroneClient | None],
        command_worker_getter: Callable[[], CommandWorker | None],
        detector_factory: Callable[[], BaseDetector] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._map_ctrl = map_ctrl
        self._settings = settings
        self._client_getter = client_getter
        self._command_worker_getter = command_worker_getter
        self._detector_factory = detector_factory

        self.runs_table = RunsTable()
        self._download_worker: DownloadWorker | None = None
        self._batch_worker: BatchImageWorker | None = None
        self._runs_cache: dict[str, DroneRun] = {}

        self._embed_widgets_in_ui()
        self._wire_signals()

    def _embed_widgets_in_ui(self) -> None:
        self._window.set_tab_widget(TAB_RUNS, self.runs_table)

    def _wire_signals(self) -> None:
        self.runs_table.refresh_requested.connect(self.refresh)
        self.runs_table.download_requested.connect(self.download)
        self.runs_table.delete_requested.connect(self.delete)
        self.runs_table.process_requested.connect(self.process_locally)

    # -------------------------------------------------------------------------
    # Public Slots
    # -------------------------------------------------------------------------
    @Slot()
    def refresh(self) -> None:
        """Query onboard REST API for list of stored runs (GET /runs)."""
        worker = self._command_worker_getter()
        if worker is None:
            self._window.status_bar_panel.set_message("Невозможно обновить список: нет подключения к БПЛА")
            return

        worker.runs_loaded.connect(self._on_runs_loaded)
        self._window.status_bar_panel.set_message("Запрос списка запусков с борта...")
        worker.request_runs()

    @Slot(str)
    def download(self, run_id: str) -> None:
        """Download and unpack run ZIP archive in background thread."""
        client = self._client_getter()
        if client is None:
            self._window.status_bar_panel.set_message("Ошибка: отсутствует сетевое подключение к БПЛА")
            return

        dest_dir = self._settings.workdir / "runs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dm = DownloadManager(client=client, base_download_dir=dest_dir)

        worker = DownloadWorker(download_manager=dm)
        self._download_worker = worker

        worker.progress.connect(self._on_download_progress)
        worker.download_finished.connect(self._on_download_finished)
        worker.failed.connect(self._on_download_failed)

        thread = start_in_thread(worker, "download_worker_thread")
        self._window.register_thread(thread, worker)

        self._window.status_bar_panel.set_cancel_enabled(True)
        self._window.status_bar_panel.set_progress(0)
        self._window.status_bar_panel.set_message(f"Скачивание архива запуска {run_id}...")

        worker.download(run_id)

    @Slot(str)
    def process_locally(self, run_id: str) -> None:
        """Perform secondary local ONNX detection and mapping on downloaded run frames."""
        run_dir = self._settings.workdir / "runs" / run_id
        if not run_dir.exists():
            self._window.status_bar_panel.set_message(
                f"Сначала скачайте запуск {run_id} перед локальной обработкой"
            )
            return

        source = FolderSource(run_dir)
        entries = source.entries()
        if not entries:
            self._window.status_bar_panel.set_message(f"В папке {run_id} нет снимков для обработки")
            return

        worker = BatchImageWorker(source=source, detector_factory=self._detector_factory)
        self._batch_worker = worker

        worker.progress.connect(
            lambda i, total, text: self._window.status_bar_panel.set_progress(
                int(round((i / float(max(1, total))) * 100))
            )
        )
        worker.frame_processed.connect(self._map_ctrl.on_frame)
        worker.batch_finished.connect(
            lambda s, f: self._window.status_bar_panel.set_message(
                f"Локальная обработка запуска {run_id} завершена: {s} кадров, {f} ошибок"
            )
        )

        thread = start_in_thread(worker, "batch_run_worker_thread")
        self._window.register_thread(thread, worker)

        self._window.status_bar_panel.set_message(f"Вторичная обработка запуска {run_id}...")
        worker.process(entries, run_inference=True)

    @Slot(str)
    def delete(self, run_id: str) -> None:
        """Delete mission run from onboard drone computer after confirmation."""
        worker = self._command_worker_getter()
        if worker is None:
            self._window.status_bar_panel.set_message("Ошибка: нет подключения к БПЛА")
            return

        run_info = self._runs_cache.get(run_id)
        size_bytes = run_info.size_bytes if run_info else 0

        if not ConfirmDialog.confirm_delete_run(run_id, size_bytes, parent=self._window):
            return

        worker.run_deleted.connect(self._on_run_deleted)
        self._window.status_bar_panel.set_message(f"Удаление запуска {run_id} с борта...")
        worker.delete_run(run_id)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def _on_runs_loaded(self, runs: list[DroneRun]) -> None:
        self._runs_cache = {r.run_id: r for r in runs}
        self.runs_table.set_runs(runs)
        self.runs_updated.emit(runs)
        self._window.status_bar_panel.set_message(f"Список запусков обновлен: {len(runs)} записей")

    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            pct = int(min(100, round((downloaded / float(total)) * 100)))
            self._window.status_bar_panel.set_progress(pct)
            mb = downloaded / (1024.0 * 1024.0)
            tot_mb = total / (1024.0 * 1024.0)
            self._window.status_bar_panel.set_message(f"Скачивание: {mb:.1f} / {tot_mb:.1f} МБ ({pct}%)")

    def _on_download_finished(self, extracted_path: str) -> None:
        self._window.status_bar_panel.set_cancel_enabled(False)
        self._window.status_bar_panel.set_progress(100)
        self._window.status_bar_panel.set_message(f"Архив успешно распакован: {Path(extracted_path).name}")
        self.download_finished.emit(extracted_path)
        self._download_worker = None

    def _on_download_failed(self, err: str) -> None:
        self._window.status_bar_panel.set_cancel_enabled(False)
        self._window.status_bar_panel.set_message(f"Ошибка скачивания: {err}")
        self._download_worker = None

    def _on_run_deleted(self, resp: DeleteRunResponse) -> None:
        self.refresh()
        freed_mb = resp.freed_bytes / (1024.0 * 1024.0)
        self._window.status_bar_panel.set_message(
            f"Запуск {resp.deleted_run_id} удален. Освобождено {freed_mb:.1f} МБ"
        )
