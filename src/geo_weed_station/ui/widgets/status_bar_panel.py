"""Status bar panel: operation text, progress, cancel button, connection indicator, counters."""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

from geo_weed_station.schemas.telemetry import ConnectionState

_CONNECTION_TEXT = {
    ConnectionState.DISCONNECTED: "Нет связи",
    ConnectionState.CONNECTING: "Подключение…",
    ConnectionState.CONNECTED: "Подключено",
    ConnectionState.RECONNECTING: "Переподключение…",
    ConnectionState.TIMEOUT: "Таймаут",
    ConnectionState.ERROR: "Ошибка связи",
}


class StatusBarPanel(QWidget):
    """Composite status widget; contains no business logic."""

    cancel_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusBarPanel")

        self._message = QLabel("Готово", self)
        self._message.setObjectName("statusMessage")
        self.label_message = self._message

        self._progress = QProgressBar(self)
        self._progress.setObjectName("statusProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedWidth(180)
        self._progress.setTextVisible(True)

        self._cancel = QPushButton("Отмена", self)
        self._cancel.setObjectName("statusCancelButton")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self.cancel_requested)

        self._counters = QLabel("", self)
        self._counters.setObjectName("statusCounters")

        self._connection = QLabel(self)
        self._connection.setObjectName("connectionIndicator")
        self._connection_state = ConnectionState.DISCONNECTED

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(10)
        layout.addWidget(self._message, 1)
        layout.addWidget(self._progress)
        layout.addWidget(self._cancel)
        layout.addWidget(self._counters)
        layout.addWidget(self._connection)

        self.set_connection_state(ConnectionState.DISCONNECTED)
        self.set_counters(None, None)

    # ------------------------------------------------------------- accessors
    @property
    def message(self) -> str:
        return self._message.text()

    @property
    def progress(self) -> int:
        return self._progress.value()

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state

    @property
    def cancel_button(self) -> QPushButton:
        return self._cancel

    @property
    def cancel_enabled(self) -> bool:
        return self._cancel.isEnabled()

    @property
    def counters_text(self) -> str:
        return self._counters.text()

    @property
    def connection_text(self) -> str:
        return self._connection.text()

    # ----------------------------------------------------------------- slots
    @Slot(str)
    def set_message(self, text: str) -> None:
        self._message.setText(text)

    @Slot(int)
    def set_progress(self, percent: int) -> None:
        """Set progress, clamped to 0..100."""
        self._progress.setValue(max(0, min(100, int(percent))))

    @Slot(object)
    def set_connection_state(self, state: ConnectionState) -> None:
        state = ConnectionState(state)
        self._connection_state = state
        self._connection.setText(f"● {_CONNECTION_TEXT[state]}")
        self._connection.setProperty("state", state.value)
        style = self._connection.style()
        style.unpolish(self._connection)
        style.polish(self._connection)

    def set_cancel_enabled(self, enabled: bool) -> None:
        self._cancel.setEnabled(bool(enabled))

    def set_counters(self, captures: Optional[int], weeds: Optional[int]) -> None:
        """Show capture / weed counters; None renders as an em dash."""
        c = "—" if captures is None else str(captures)
        w = "—" if weeds is None else str(weeds)
        self._counters.setText(f"Снимков: {c}   Сорняков: {w}")
