"""GeoWeedStation package entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from geo_weed_station.config.logging_setup import setup_logging
from geo_weed_station.config.settings import AppSettings, get_settings


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="geo-weed-station",
        description="GeoWeedStation - Ground Control Station & Agronomic Weed Mapping",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom config JSON file")
    parser.add_argument("--offscreen", action="store_true", help="Force offscreen headless platform")
    parser.add_argument("--selftest", action="store_true", help="Run internal self-test and exit")
    return parser.parse_args(argv if argv is not None else [])


def run_selftest() -> int:
    """Self-test verifying imports, settings loading, and headless UI initialization."""
    print("[SELFTEST] Checking core configuration and imports...")
    try:
        settings = get_settings()
        assert settings is not None, "Failed to load AppSettings"

        import cv2
        import numpy as np
        import onnxruntime
        from PySide6 import QtCore, QtGui, QtWidgets
        from geo_weed_station.core.map_engine.weed_map import MapEngine
        from geo_weed_station.core.xmp_parser import parse_xmp
        from geo_weed_station.ui.theme import load_stylesheet

        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        sheet = load_stylesheet()
        assert len(sheet) > 0, "Stylesheet is empty"

        print("[SELFTEST] All components and assets verified successfully.")
        return 0
    except Exception as exc:
        print(f"[SELFTEST] ERROR: {exc}", file=sys.stderr)
        return 1


def main(argv: Optional[Sequence[str]] = None, run_event_loop: bool = False) -> int:
    """Initialize settings, logging, Qt application and the AppController."""
    args = parse_args(argv)

    if args.selftest:
        return run_selftest()

    if args.offscreen or (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    settings = get_settings(Path(args.config) if args.config else None)
    setup_logging(settings)

    from PySide6.QtWidgets import QApplication
    from geo_weed_station.controllers.app_controller import AppController
    from geo_weed_station.ui.theme import load_stylesheet

    app = QApplication.instance()
    if app is None:
        qt_args = [sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])
        app = QApplication(qt_args)
    app.setStyleSheet(load_stylesheet())

    controller = AppController(settings)

    if not run_event_loop:
        controller.shutdown()
        return 0

    controller.start()
    return app.exec()


def cli_main() -> int:
    """CLI binary entrypoint running interactive Qt event loop."""
    return main(argv=sys.argv[1:], run_event_loop=True)


if __name__ == "__main__":
    sys.exit(cli_main())
