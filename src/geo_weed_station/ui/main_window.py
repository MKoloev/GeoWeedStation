"""Main application window: mode switch, tabs, mode panels, status bar. Signals only, no business logic."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QWidget,
)

from geo_weed_station.ui.app_mode import AppMode, ModeManager
from geo_weed_station.ui.theme import icon
from geo_weed_station.ui.widgets.status_bar_panel import StatusBarPanel

logger = logging.getLogger(__name__)

MAP_LOCKED_TEXT = "Геопривязка отсутствует. Доступен только просмотр исходных изображений"

TAB_IMAGE = "image"
TAB_VIDEO = "video"
TAB_MAP = "map"
TAB_MISSION = "mission"
TAB_RUNS = "runs"
_TABS: list[tuple[str, str]] = [
    (TAB_IMAGE, "Изображение"),
    (TAB_VIDEO, "Видео"),
    (TAB_MAP, "Карта"),
    (TAB_MISSION, "Миссия"),
    (TAB_RUNS, "Запуски"),
]

_PAGE_CONTENT = 0
_PAGE_LOCKED = 1


class MainWindow(QMainWindow):
    """Application shell. Menus/toolbar only emit signals (MVC)."""

    request_settings = Signal()
    request_export = Signal()
    export_report_requested = Signal()
    open_image_requested = Signal()
    open_folder_requested = Signal()
    open_video_requested = Signal()
    save_annotations_requested = Signal()
    closing = Signal()

    SHUTDOWN_TIMEOUT_MS = 5000

    def __init__(
        self,
        mode_manager: ModeManager | None = None,
        qsettings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("mainWindow")
        self.setWindowTitle("GeoWeedStation")
        self.resize(1280, 800)

        self._qsettings = qsettings
        self._mode_manager = mode_manager or ModeManager(qsettings, parent=self)
        self._panels: dict[AppMode, list[QDockWidget]] = {AppMode.ONLINE: [], AppMode.OFFLINE: []}
        self._threads: list[tuple[QThread, QObject | None]] = []
        self._pages: dict[str, QStackedWidget] = {}
        self._map_locked = False

        self._build_tabs()
        self._build_actions()
        self._build_toolbar()
        self._build_menu()
        self._build_status_bar()
        self._build_default_panels()

        self._mode_manager.mode_changed.connect(self._apply_mode)
        self._apply_mode(self._mode_manager.mode)
        self._restore_geometry()

    # ----------------------------------------------------------------- build
    def _build_tabs(self) -> None:
        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("centralTabs")
        for key, title in _TABS:
            stack = QStackedWidget(self._tabs)
            placeholder = QLabel(f"{title}", stack)
            placeholder.setObjectName("tabPlaceholder")
            placeholder.setAlignment(Qt.AlignCenter)
            locked = QLabel(MAP_LOCKED_TEXT, stack)
            locked.setObjectName("mapLockedLabel")
            locked.setAlignment(Qt.AlignCenter)
            locked.setWordWrap(True)
            stack.addWidget(placeholder)  # _PAGE_CONTENT
            stack.addWidget(locked)  # _PAGE_LOCKED
            self._pages[key] = stack
            self._tabs.addTab(stack, title)
        self.setCentralWidget(self._tabs)

    def _build_actions(self) -> None:
        self.action_mode_online = QAction(icon("online"), "Онлайн", self)
        self.action_mode_offline = QAction(icon("offline"), "Офлайн", self)
        self.action_mode_online.setObjectName("actionModeOnline")
        self.action_mode_offline.setObjectName("actionModeOffline")
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)
        for act, mode in (
            (self.action_mode_online, AppMode.ONLINE),
            (self.action_mode_offline, AppMode.OFFLINE),
        ):
            act.setCheckable(True)
            self._mode_group.addAction(act)
            act.triggered.connect(lambda _checked=False, m=mode: self._mode_manager.set_mode(m))

        self.action_open_image = QAction(icon("image"), "Открыть изображение…", self)
        self.action_open_folder = QAction(icon("folder"), "Открыть папку…", self)
        self.action_open_video = QAction(icon("video"), "Открыть видео…", self)
        self.action_save_annotations = QAction("Сохранить разметку", self)
        self.action_save_annotations.setShortcut("Ctrl+S")
        self.action_export_report = QAction(icon("export"), "Выгрузить отчет…", self)
        self.action_export_report.setShortcut("Ctrl+R")
        self.action_export = QAction(icon("export"), "Экспорт…", self)
        self.action_settings = QAction(icon("settings"), "Настройки…", self)
        self.action_quit = QAction("Выход", self)
        self.action_open_image.triggered.connect(self.open_image_requested)
        self.action_open_folder.triggered.connect(self.open_folder_requested)
        self.action_open_video.triggered.connect(self.open_video_requested)
        self.action_save_annotations.triggered.connect(self.save_annotations_requested)
        self.action_export_report.triggered.connect(self.export_report_requested)
        self.action_export.triggered.connect(self.request_export)
        self.action_settings.triggered.connect(self.request_settings)
        self.action_quit.triggered.connect(self.close)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Режим", self)
        bar.setObjectName("modeToolbar")
        bar.setMovable(False)
        bar.addAction(self.action_mode_online)
        bar.addAction(self.action_mode_offline)
        bar.addSeparator()
        bar.addAction(self.action_open_image)
        bar.addAction(self.action_open_folder)
        bar.addAction(self.action_open_video)
        bar.addAction(self.action_save_annotations)
        bar.addAction(self.action_export_report)
        bar.addSeparator()
        bar.addAction(self.action_export)
        bar.addAction(self.action_settings)
        self.addToolBar(Qt.TopToolBarArea, bar)
        self._toolbar = bar
        self._offline_actions = (
            self.action_open_image,
            self.action_open_folder,
            self.action_open_video,
            self.action_save_annotations,
        )

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.action_open_image)
        file_menu.addAction(self.action_open_folder)
        file_menu.addAction(self.action_open_video)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save_annotations)
        file_menu.addAction(self.action_export_report)
        file_menu.addSeparator()
        file_menu.addAction(self.action_export)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)
        mode_menu = self.menuBar().addMenu("Режим")
        mode_menu.addAction(self.action_mode_online)
        mode_menu.addAction(self.action_mode_offline)
        self.menuBar().addMenu("Настройки").addAction(self.action_settings)

    def _build_status_bar(self) -> None:
        self.status_panel = StatusBarPanel(self)
        self.status_bar_panel = self.status_panel
        self.statusBar().addWidget(self.status_panel, 1)

    def _build_default_panels(self) -> None:
        net = QLabel("Сеть и телеметрия борта", self)
        net.setAlignment(Qt.AlignCenter)
        self.add_mode_panel(AppMode.ONLINE, net, "Сеть и телеметрия")
        files = QLabel("Проводник локальных файлов", self)
        files.setAlignment(Qt.AlignCenter)
        self.add_mode_panel(AppMode.OFFLINE, files, "Файлы")

    # ------------------------------------------------------------ public API
    @property
    def mode_manager(self) -> ModeManager:
        return self._mode_manager

    def central_tabs(self) -> QTabWidget:
        """Tab widget: Изображение / Видео / Карта / Миссия / Запуски."""
        return self._tabs

    def tab_index(self, key: str) -> int:
        return self._tabs.indexOf(self._pages[key])

    def set_tab_widget(self, key: str, widget: QWidget) -> None:
        """Install a real content widget into a tab (replaces the placeholder)."""
        stack = self._pages[key]
        old = stack.widget(_PAGE_CONTENT)
        stack.removeWidget(old)
        old.deleteLater()
        stack.insertWidget(_PAGE_CONTENT, widget)
        stack.setCurrentIndex(_PAGE_LOCKED if (key == TAB_MAP and self._map_locked) else _PAGE_CONTENT)

    def add_mode_panel(self, mode: AppMode, widget: QWidget, title: str = "") -> None:
        """Register a side panel that is visible only in the given mode."""
        mode = AppMode(mode)
        dock = QDockWidget(title or widget.windowTitle() or mode.value, self)
        dock.setObjectName(f"panel_{mode.value}_{len(self._panels[mode])}")
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        if self._panels[mode]:
            first_dock = self._panels[mode][0]
            self.tabifyDockWidget(first_dock, dock)
            first_dock.raise_()
        self._panels[mode].append(dock)
        dock.setVisible(mode == self._mode_manager.mode)

    def clear_default_panels(self, mode: AppMode) -> None:
        """Remove initial placeholder panels for the given mode when real controllers mount."""
        mode = AppMode(mode)
        for dock in list(self._panels[mode]):
            if isinstance(dock.widget(), QLabel):
                self._panels[mode].remove(dock)
                self.removeDockWidget(dock)
                dock.deleteLater()

    def mode_panels(self, mode: AppMode) -> list[QDockWidget]:
        return list(self._panels[AppMode(mode)])

    def set_map_tab_locked(self, locked: bool, reason: str | None = None) -> None:
        """Lock the map tab and show the georeference warning instead of its content."""
        self._map_locked = bool(locked)
        stack = self._pages[TAB_MAP]
        if locked:
            label = stack.widget(_PAGE_LOCKED)
            if isinstance(label, QLabel):
                label.setText(reason or MAP_LOCKED_TEXT)
            stack.setCurrentIndex(_PAGE_LOCKED)
            self._tabs.setTabToolTip(self.tab_index(TAB_MAP), reason or MAP_LOCKED_TEXT)
        else:
            stack.setCurrentIndex(_PAGE_CONTENT)
            self._tabs.setTabToolTip(self.tab_index(TAB_MAP), "")

    @property
    def is_map_tab_locked(self) -> bool:
        return self._map_locked

    def map_locked_text(self) -> str:
        label = self._pages[TAB_MAP].widget(_PAGE_LOCKED)
        return label.text() if isinstance(label, QLabel) else ""

    def register_thread(self, thread: QThread, worker: QObject | None = None) -> None:
        """Track a worker thread so closeEvent can stop it and wait for completion."""
        self._threads.append((thread, worker))

    def shutdown_threads(self, timeout_ms: int | None = None) -> bool:
        """Request workers to stop, quit their threads and wait. Returns True if all stopped."""
        timeout = self.SHUTDOWN_TIMEOUT_MS if timeout_ms is None else timeout_ms
        for _thread, worker in self._threads:
            stop = getattr(worker, "request_stop", None)
            if callable(stop):
                stop()
        all_ok = True
        remaining = []
        for thread, worker in self._threads:
            thread.quit()
            stopped = thread.wait(timeout)
            is_alive = (not stopped) or thread.isRunning() or (worker is not None and getattr(worker, "_running", False))
            if is_alive:
                logger.error("Thread %s did not stop within %d ms", thread.objectName(), timeout)
                all_ok = False
                remaining.append((thread, worker))
        self._threads = remaining
        return all_ok

    # --------------------------------------------------------------- private
    def _apply_mode(self, mode: AppMode) -> None:
        mode = AppMode(mode)
        for m, docks in self._panels.items():
            for dock in docks:
                dock.setVisible(m == mode)
        if self._panels[mode]:
            self._panels[mode][0].raise_()
        self.action_mode_online.setChecked(mode == AppMode.ONLINE)
        self.action_mode_offline.setChecked(mode == AppMode.OFFLINE)
        for act in self._offline_actions:
            act.setEnabled(mode == AppMode.OFFLINE)

    def _restore_geometry(self) -> None:
        if self._qsettings is None:
            return
        geometry = self._qsettings.value("ui/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closing.emit()
        self.shutdown_threads()
        if self._qsettings is not None:
            self._qsettings.setValue("ui/geometry", self.saveGeometry())
            self._qsettings.sync()
        super().closeEvent(event)
