"""Application operating mode (online / offline) and its persistent manager."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union
from PySide6.QtCore import QObject, QSettings, Signal, Slot


class AppMode(str, Enum):
    """Operating mode of the ground station."""

    ONLINE = "online"
    OFFLINE = "offline"


class ModeManager(QObject):
    """Holds the current AppMode, emits changes and persists it between sessions."""

    SETTINGS_KEY = "ui/mode"

    mode_changed = Signal(object)  # AppMode

    def __init__(
        self,
        settings: Optional[QSettings] = None,
        default: AppMode = AppMode.OFFLINE,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._mode = default
        if settings is not None:
            stored = settings.value(self.SETTINGS_KEY, None)
            try:
                self._mode = AppMode(str(stored)) if stored is not None else default
            except ValueError:
                self._mode = default

    @property
    def mode(self) -> AppMode:
        return self._mode

    @Slot(object)
    def set_mode(self, mode: Union[AppMode, str]) -> None:
        """Switch mode; emits `mode_changed` only on real change."""
        new_mode = AppMode(mode)
        if new_mode == self._mode:
            return
        self._mode = new_mode
        if self._settings is not None:
            self._settings.setValue(self.SETTINGS_KEY, new_mode.value)
            self._settings.sync()
        self.mode_changed.emit(new_mode)
