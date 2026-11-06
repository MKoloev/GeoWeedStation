"""Tests for MainWindow, ModeManager and UI shell behavior (offscreen)."""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Slot
from PySide6.QtWidgets import QApplication, QLabel

from geo_weed_station.ui import main_window as mw_module
from geo_weed_station.ui.app_mode import AppMode, ModeManager
from geo_weed_station.ui.main_window import (
    MAP_LOCKED_TEXT,
    TAB_IMAGE,
    TAB_MAP,
    MainWindow,
)
from geo_weed_station.ui.theme import ICONS_DIR, icon, load_stylesheet
from geo_weed_station.workers.base_worker import ManagedWorker, start_in_thread


def _ini(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat)


@pytest.fixture
def window(qtbot, tmp_path):
    w = MainWindow(qsettings=_ini(tmp_path))
    qtbot.addWidget(w)
    w.show()
    return w


def test_tabs_present(window):
    tabs = window.central_tabs()
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Изображение", "Видео", "Карта", "Миссия", "Запуски",
    ]


def test_mode_switch_toggles_panels(window):
    online = window.mode_panels(AppMode.ONLINE)
    offline = window.mode_panels(AppMode.OFFLINE)
    window.mode_manager.set_mode(AppMode.OFFLINE)
    assert all(not d.isVisible() for d in online)
    assert all(d.isVisible() for d in offline)
    window.mode_manager.set_mode(AppMode.ONLINE)
    assert all(d.isVisible() for d in online)
    assert all(not d.isVisible() for d in offline)
    assert window.action_mode_online.isChecked() and not window.action_mode_offline.isChecked()


def test_toolbar_action_switches_mode(window):
    window.mode_manager.set_mode(AppMode.OFFLINE)
    window.action_mode_online.trigger()
    assert window.mode_manager.mode == AppMode.ONLINE


def test_add_mode_panel_visibility(window):
    from PySide6.QtWidgets import QLabel

    window.mode_manager.set_mode(AppMode.OFFLINE)
    extra = QLabel("x")
    window.add_mode_panel(AppMode.ONLINE, extra, "extra")
    assert not extra.isVisible()
    window.mode_manager.set_mode(AppMode.ONLINE)
    assert extra.isVisible()


def test_offline_file_actions_disabled_online(window):
    window.mode_manager.set_mode(AppMode.ONLINE)
    assert not window.action_open_image.isEnabled()
    window.mode_manager.set_mode(AppMode.OFFLINE)
    assert window.action_open_folder.isEnabled()


def test_mode_persists_between_sessions(qtbot, tmp_path):
    w1 = MainWindow(qsettings=_ini(tmp_path))
    qtbot.addWidget(w1)
    w1.mode_manager.set_mode(AppMode.ONLINE)
    w1.close()
    w2 = MainWindow(qsettings=_ini(tmp_path))
    qtbot.addWidget(w2)
    assert w2.mode_manager.mode == AppMode.ONLINE
    w2.show()
    assert all(d.isVisible() for d in w2.mode_panels(AppMode.ONLINE))
    assert all(not d.isVisible() for d in w2.mode_panels(AppMode.OFFLINE))


def test_mode_manager_emits_only_on_change(qtbot):
    mm = ModeManager(default=AppMode.OFFLINE)
    with qtbot.assertNotEmitted(mm.mode_changed):
        mm.set_mode(AppMode.OFFLINE)
    with qtbot.waitSignal(mm.mode_changed, timeout=500) as blocker:
        mm.set_mode("online")
    assert blocker.args == [AppMode.ONLINE]


def test_mode_manager_ignores_garbage_setting(tmp_path):
    s = _ini(tmp_path)
    s.setValue(ModeManager.SETTINGS_KEY, "bogus")
    assert ModeManager(s).mode == AppMode.OFFLINE


def test_menu_and_toolbar_only_emit_signals(window, qtbot):
    pairs = [
        (window.action_open_image, window.open_image_requested),
        (window.action_open_folder, window.open_folder_requested),
        (window.action_open_video, window.open_video_requested),
        (window.action_export, window.request_export),
        (window.action_settings, window.request_settings),
    ]
    window.mode_manager.set_mode(AppMode.OFFLINE)
    for action, signal in pairs:
        with qtbot.waitSignal(signal, timeout=500):
            action.trigger()


def test_map_lock_default_text(window):
    assert not window.is_map_tab_locked
    window.set_map_tab_locked(True)
    assert window.is_map_tab_locked
    assert window.map_locked_text() == MAP_LOCKED_TEXT
    assert MAP_LOCKED_TEXT == "Геопривязка отсутствует. Доступен только просмотр исходных изображений"
    stack = window._pages[TAB_MAP]
    assert stack.currentIndex() == 1
    assert window.central_tabs().tabToolTip(window.tab_index(TAB_MAP)) == MAP_LOCKED_TEXT


def test_map_lock_custom_reason_and_unlock(window):
    window.set_map_tab_locked(True, "Высота H ≤ 0")
    assert window.map_locked_text() == "Высота H ≤ 0"
    window.set_map_tab_locked(False)
    assert window._pages[TAB_MAP].currentIndex() == 0
    assert window.central_tabs().tabToolTip(window.tab_index(TAB_MAP)) == ""


def test_set_tab_widget_respects_lock(window):
    window.set_map_tab_locked(True)
    widget = QLabel("map!")
    window.set_tab_widget(TAB_MAP, widget)
    assert window._pages[TAB_MAP].currentIndex() == 1
    window.set_map_tab_locked(False)
    assert window._pages[TAB_MAP].currentWidget() is widget
    window.set_tab_widget(TAB_IMAGE, QLabel("img"))


def test_stylesheet_applied_and_icons_load(qtbot):
    qss = load_stylesheet()
    assert "QMainWindow" in qss
    app = QApplication.instance()
    app.setStyleSheet(qss)
    names = ["online", "offline", "image", "folder", "video", "settings", "export"]
    for n in names:
        assert (ICONS_DIR / f"{n}.svg").is_file()
        assert not icon(n).pixmap(16, 16).isNull(), n
    app.setStyleSheet("")


class _LoopWorker(ManagedWorker):
    def start_work(self) -> None:
        while not self.is_stop_requested:
            time.sleep(0.01)
        self.finished.emit()


def test_close_event_stops_running_worker(qtbot, tmp_path):
    w = MainWindow(qsettings=_ini(tmp_path))
    qtbot.addWidget(w)
    w.show()
    worker = _LoopWorker()
    thread = start_in_thread(worker, "fake_loop")
    w.register_thread(thread, worker)
    qtbot.waitUntil(thread.isRunning, timeout=2000)
    assert thread.isRunning()

    w.close()
    assert not thread.isRunning()
    assert thread.isFinished()


def test_shutdown_threads_without_workers(qtbot, tmp_path):
    w = MainWindow(qsettings=_ini(tmp_path))
    qtbot.addWidget(w)
    assert w.shutdown_threads() is True


def test_shutdown_reports_stuck_thread(qtbot, tmp_path):
    started_ev = threading.Event()
    release_ev = threading.Event()

    class Stuck(ManagedWorker):
        @Slot()
        def start_work(self) -> None:
            self._running = True
            started_ev.set()
            release_ev.wait(5.0)  # ignores stop request until released
            self._running = False
            self.finished.emit()

    w = MainWindow(qsettings=_ini(tmp_path))
    qtbot.addWidget(w)
    worker = Stuck()
    thread = start_in_thread(worker, "stuck")
    w.register_thread(thread, worker)
    qtbot.waitUntil(started_ev.is_set, timeout=2000)
    assert w.shutdown_threads(timeout_ms=50) is False
    release_ev.set()
    assert w.shutdown_threads(timeout_ms=3000) is True


def test_main_window_module_has_no_forbidden_imports():
    forbidden = {"cv2", "onnxruntime", "requests", "socket", "urllib", "http"}
    for path in (
        Path(mw_module.__file__),
        Path(mw_module.__file__).with_name("app_mode.py"),
        Path(mw_module.__file__).parent / "widgets" / "status_bar_panel.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not (forbidden & set(names)), f"{path.name} imports {forbidden & set(names)}"


def test_main_launches_window(tmp_path, monkeypatch):
    import main as app_main
    from geo_weed_station.config.settings import AppSettings

    s = AppSettings.load()
    s.workdir = tmp_path
    monkeypatch.setattr(app_main, "get_settings", lambda *a, **kw: s)
    monkeypatch.setattr("geo_weed_station.main.get_settings", lambda *a, **kw: s)
    assert app_main.main() == 0
    assert (tmp_path / "app.log").exists()
