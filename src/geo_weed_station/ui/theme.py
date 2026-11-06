"""Theme helpers: stylesheet and icon loading from package resources."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
ICONS_DIR = RESOURCES_DIR / "icons"


def load_stylesheet() -> str:
    """Return the QSS theme text (empty string if the file is missing)."""
    path = RESOURCES_DIR / "styles.qss"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def icon(name: str) -> QIcon:
    """Load an SVG icon by base name from resources/icons."""
    return QIcon(str(ICONS_DIR / f"{name}.svg"))
