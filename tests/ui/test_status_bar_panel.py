"""Tests for StatusBarPanel."""

from __future__ import annotations

import pytest

from geo_weed_station.schemas.telemetry import ConnectionState
from geo_weed_station.ui.theme import load_stylesheet
from geo_weed_station.ui.widgets.status_bar_panel import StatusBarPanel


@pytest.fixture
def panel(qtbot):
    p = StatusBarPanel()
    qtbot.addWidget(p)
    return p


def test_defaults(panel):
    assert panel.progress == 0
    assert not panel.cancel_enabled
    assert panel.connection_state == ConnectionState.DISCONNECTED
    assert panel.counters_text == "Снимков: —   Сорняков: —"


def test_message(panel):
    panel.set_message("инференс кадра 3 из 10")
    assert panel.message == "инференс кадра 3 из 10"


@pytest.mark.parametrize("value,expected", [(-10, 0), (0, 0), (37, 37), (100, 100), (250, 100)])
def test_progress_bounds(panel, value, expected):
    panel.set_progress(value)
    assert panel.progress == expected


def test_cancel_requested_signal(panel, qtbot):
    panel.set_cancel_enabled(True)
    assert panel.cancel_enabled
    with qtbot.waitSignal(panel.cancel_requested, timeout=1000):
        panel._cancel.click()


def test_cancel_disabled_does_not_emit(panel, qtbot):
    with qtbot.assertNotEmitted(panel.cancel_requested):
        panel._cancel.click()


@pytest.mark.parametrize("state", list(ConnectionState))
def test_all_connection_states(panel, state):
    panel.set_connection_state(state)
    assert panel.connection_state == state
    assert panel._connection.property("state") == state.value
    assert panel.connection_text.startswith("●")


def test_connection_states_have_distinct_texts(panel):
    texts = set()
    for s in ConnectionState:
        panel.set_connection_state(s)
        texts.add(panel.connection_text)
    assert len(texts) == len(ConnectionState)


def test_counters(panel):
    panel.set_counters(12, 5)
    assert panel.counters_text == "Снимков: 12   Сорняков: 5"
    panel.set_counters(0, None)
    assert panel.counters_text == "Снимков: 0   Сорняков: —"


def test_slots_via_signals(panel, qtbot):
    from PySide6.QtCore import QObject, Signal

    class Src(QObject):
        msg = Signal(str)
        prog = Signal(int)
        conn = Signal(object)

    s = Src()
    s.msg.connect(panel.set_message)
    s.prog.connect(panel.set_progress)
    s.conn.connect(panel.set_connection_state)
    s.msg.emit("x")
    s.prog.emit(55)
    s.conn.emit(ConnectionState.CONNECTED)
    assert (panel.message, panel.progress, panel.connection_state) == ("x", 55, ConnectionState.CONNECTED)


def test_stylesheet_has_all_connection_states():
    qss = load_stylesheet()
    for s in ConnectionState:
        assert f'[state="{s.value}"]' in qss
