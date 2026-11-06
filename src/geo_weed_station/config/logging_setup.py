"""Logging configuration setup for GeoWeedStation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str | object = "INFO", log_file: Path | None = None) -> None:
    """Configure root logger with console and optional file handlers.
    
    Args:
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR') or AppSettings instance.
        log_file: Optional path to log file. Directory will be created if needed.
    """
    if not isinstance(level, str):
        if log_file is None and hasattr(level, "workdir"):
            log_file = Path(level.workdir) / "app.log"
        level_str = str(getattr(level, "log_level", "INFO"))
    else:
        level_str = level

    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on reconfiguration
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
