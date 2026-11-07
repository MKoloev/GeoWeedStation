"""Theme helpers: stylesheet and icon loading from package resources."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
ICONS_DIR = RESOURCES_DIR / "icons"


def load_stylesheet() -> str:
    """Return the QSS theme text (empty string if the file is missing)."""
    path = RESOURCES_DIR / "styles.qss"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def apply_theme(app: QApplication) -> None:
    """Apply dark Fusion palette and QSS stylesheet to application."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#182234"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1e293b"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1e293b"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#5eead4"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#14b8a6"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#134e4a"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#5eead4"))
    app.setPalette(palette)
    app.setStyleSheet(load_stylesheet())


def icon(name: str) -> QIcon:
    """Load an SVG icon by base name from resources/icons."""
    return QIcon(str(ICONS_DIR / f"{name}.svg"))
