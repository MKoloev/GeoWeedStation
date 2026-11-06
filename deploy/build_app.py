"""PyInstaller build script and smoke testing utility for GeoWeedStation."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_app")

ROOT_DIR = Path(__file__).resolve().parent.parent


def collect_data_files() -> List[Tuple[str, str]]:
    """Collect non-code static assets: QSS, icons, default config, models."""
    datas = [
        (
            str(ROOT_DIR / "src" / "geo_weed_station" / "ui" / "resources"),
            os.path.join("geo_weed_station", "ui", "resources"),
        ),
        (
            str(ROOT_DIR / "src" / "geo_weed_station" / "config" / "default_config.json"),
            os.path.join("geo_weed_station", "config"),
        ),
    ]

    models_dir = ROOT_DIR / "models"
    if models_dir.exists():
        datas.append((str(models_dir), "models"))

    return datas


def build(onefile: bool = False, clean: bool = True, dist_dir: Optional[Path] = None) -> Path:
    """Run PyInstaller build process."""
    dist_path = dist_dir or (ROOT_DIR / "dist")
    build_path = ROOT_DIR / "build"
    main_script = ROOT_DIR / "main.py"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        "GeoWeedStation",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(build_path),
        "--noconfirm",
    ]

    if clean:
        cmd.append("--clean")
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Add data files
    for src, dst in collect_data_files():
        if Path(src).exists():
            cmd.extend(["--add-data", f"{src}:{dst}"])

    # Hidden imports and paths
    cmd.extend(
        [
            "--paths",
            str(ROOT_DIR / "src"),
            "--hidden-import",
            "PySide6.QtCore",
            "--hidden-import",
            "PySide6.QtGui",
            "--hidden-import",
            "PySide6.QtWidgets",
            str(main_script),
        ]
    )

    logger.info("Executing PyInstaller command: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
    if res.returncode != 0:
        raise RuntimeError(f"PyInstaller build failed with return code {res.returncode}")

    if onefile:
        binary = dist_path / "GeoWeedStation"
    else:
        binary = dist_path / "GeoWeedStation" / "GeoWeedStation"

    logger.info("Build artifact generated at: %s", binary)
    return binary


def smoke_test(binary: Path) -> bool:
    """Execute built binary with --selftest flag."""
    if not binary.exists():
        logger.error("Binary not found: %s", binary)
        return False

    cmd = [str(binary), "--selftest", "--offscreen"]
    logger.info("Running smoke test: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, timeout=30, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            logger.info("Smoke test PASSED successfully.")
            return True
        logger.error("Smoke test FAILED (code %d):\n%s", proc.returncode, proc.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Smoke test timed out after 30 seconds.")
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GeoWeedStation build & packaging tool")
    parser.add_argument("--onefile", action="store_true", help="Package as single binary file")
    parser.add_argument("--smoke-test", action="store_true", help="Run self-test on built binary")
    parser.add_argument("--dist", type=str, default=None, help="Custom dist directory")
    args = parser.parse_args(argv)

    dist_path = Path(args.dist) if args.dist else None
    binary = build(onefile=args.onefile, dist_dir=dist_path)

    if args.smoke_test:
        ok = smoke_test(binary)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
