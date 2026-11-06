"""Tests for deployment, packaging assets and selftest runner."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import pytest

from deploy.build_app import collect_data_files
from main import parse_args, run_selftest


def test_collect_data_files_contains_assets():
    datas = collect_data_files()
    assert len(datas) >= 2

    # Check QSS resources and default config exist
    paths = [p[0] for p in datas]
    assert any("resources" in p for p in paths)
    assert any("default_config.json" in p for p in paths)


def test_parse_args_options():
    args = parse_args(["--selftest", "--offscreen"])
    assert args.selftest is True
    assert args.offscreen is True
    assert args.config is None

    args_custom = parse_args(["--config", "/tmp/custom.json"])
    assert args_custom.config == "/tmp/custom.json"
    assert args_custom.selftest is False


def test_main_selftest_executes_successfully():
    ret = run_selftest()
    assert ret == 0


def test_main_subprocess_selftest():
    """Verify running python main.py --selftest in a separate process returns 0."""
    root_dir = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root_dir / "main.py"), "--selftest"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root_dir / 'src'}:{env.get('PYTHONPATH', '')}"

    res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "[SELFTEST]" in res.stdout
