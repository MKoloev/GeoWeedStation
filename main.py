"""GeoWeedStation application entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence

# Ensure src/ is on sys.path even when running without pip install -e .
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from geo_weed_station.config.settings import AppSettings, get_settings
from geo_weed_station.config.logging_setup import setup_logging
from geo_weed_station.main import cli_main, main, parse_args, run_selftest

if __name__ == "__main__":
    sys.exit(cli_main())
