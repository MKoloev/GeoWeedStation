"""Confirmation dialogs for irreversible and emergency actions (deletion, abort)."""

from __future__ import annotations

import logging
from typing import Optional
from PySide6.QtWidgets import QMessageBox, QWidget

from geo_weed_station.schemas.onboard_api import StopAction

logger = logging.getLogger(__name__)


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    mb = size_bytes / (1024.0 * 1024.0)
    if mb < 0.1:
        kb = size_bytes / 1024.0
        return f"{kb:.1f} КБ"
    return f"{mb:.2f} МБ"


class ConfirmDialog:
    """Static utility methods showing confirmation dialogs with clear consequences."""

    @staticmethod
    def confirm_delete_run(run_id: str, size_bytes: int, parent: Optional[QWidget] = None) -> bool:
        """Ask user confirmation before deleting a mission run from onboard drone storage."""
        size_str = _format_size(size_bytes)
        text = (
            f"Вы действительно хотите удалить запуск <b>{run_id}</b> с борта БПЛА?<br><br>"
            f"Будет безвозвратно освобождено: <b>{size_str}</b> дискового пространства."
        )

        res = QMessageBox.question(
            parent,
            "Подтверждение удаления запуска",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return res == QMessageBox.StandardButton.Yes

    @staticmethod
    def confirm_abort(action: StopAction, parent: Optional[QWidget] = None) -> bool:
        """Ask user confirmation before issuing an emergency flight abort command."""
        if action == StopAction.LAND:
            act_desc = "немедленную экстренную посадку (Land)"
        elif action == StopAction.RTH:
            act_desc = "прерывание миссии и возврат в точку взлета (RTH)"
        else:
            act_desc = f"действие {action.value}"

        text = (
            f"Внимание! Вы запрашиваете <b>{act_desc}</b>.<br><br>"
            f"Текущее полетное задание будет прервано. Подтверждаете отправку команды?"
        )

        res = QMessageBox.warning(
            parent,
            "Подтверждение аварийной команды",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return res == QMessageBox.StandardButton.Yes
