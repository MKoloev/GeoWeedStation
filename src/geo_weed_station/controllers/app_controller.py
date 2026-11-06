"""Root application controller orchestrating windows, controllers, and thread lifecycle."""

from __future__ import annotations

import logging
from typing import Callable, Optional
from PySide6.QtCore import QObject, QSettings
from PySide6.QtWidgets import QDialog

from geo_weed_station.config.settings import AppSettings, get_settings
from geo_weed_station.controllers.export_controller import ExportController
from geo_weed_station.controllers.map_controller import MapController
from geo_weed_station.controllers.offline_controller import OfflineController
from geo_weed_station.controllers.online_controller import OnlineController
from geo_weed_station.controllers.runs_controller import RunsController
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.ui.app_mode import AppMode
from geo_weed_station.ui.dialogs.settings_dialog import SettingsDialog
from geo_weed_station.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class AppController(QObject):
    """Top-level MVC coordinator managing application shell, domain controllers, and background threads."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        detector_factory: Optional[Callable[[], BaseDetector]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings or get_settings()
        self._detector_factory = detector_factory

        # 1. Initialize core MapEngine
        projector = NadirProjector()
        self.map_engine = MapEngine(projector=projector, settings=self.settings.map)

        # 2. Initialize UI Shell (MainWindow)
        self.settings.workdir.mkdir(parents=True, exist_ok=True)
        qsettings = QSettings(str(self.settings.workdir / "ui_state.ini"), QSettings.IniFormat)
        self.window = MainWindow(qsettings=qsettings)

        # 3. Initialize Domain Controllers
        self.map_controller = MapController(engine=self.map_engine, parent=self)
        self.offline_controller = OfflineController(
            window=self.window,
            map_ctrl=self.map_controller,
            settings=self.settings,
            detector_factory=self._detector_factory,
            parent=self,
        )
        self.online_controller = OnlineController(
            window=self.window,
            map_ctrl=self.map_controller,
            settings=self.settings,
            map_canvas=self.offline_controller.map_canvas,
            parent=self,
        )
        self.runs_controller = RunsController(
            window=self.window,
            map_ctrl=self.map_controller,
            settings=self.settings,
            client_getter=lambda: self.online_controller._client,
            command_worker_getter=lambda: self.online_controller._command_worker,
            detector_factory=self._detector_factory,
            parent=self,
        )
        self.export_controller = ExportController(
            window=self.window,
            map_ctrl=self.map_controller,
            settings=self.settings,
            parent=self,
        )

        # 4. Wire global window signals & mode switches
        self.window.request_settings.connect(self._open_settings_dialog)
        self.window.mode_manager.mode_changed.connect(self._on_mode_changed)

    def _on_mode_changed(self, mode: AppMode) -> None:
        if mode == AppMode.ONLINE:
            net = self.settings.network
            self.online_controller.connect_to(net.host, net.port)
        else:
            self.online_controller.disconnect()

    # -------------------------------------------------------------------------
    # Public Lifecycle
    # -------------------------------------------------------------------------
    def start(self) -> None:
        """Display the main application window."""
        self.window.show()

    def shutdown(self) -> None:
        """Stop all background worker threads and close windows."""
        logger.info("Initiating application shutdown...")
        self.window.shutdown_threads(timeout_ms=1500)
        self.window.close()

    # -------------------------------------------------------------------------
    # Settings Handler
    # -------------------------------------------------------------------------
    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings, parent=self.window)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.result_settings()
            self.map_engine.settings = self.settings.map
            self.window.status_bar_panel.set_message("Настройки успешно применены")
