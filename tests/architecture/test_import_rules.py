"""Automated architectural tests enforcing layer isolation, clean boundaries, and license compliance."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Set
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "geo_weed_station"


def collect_module_imports(py_file: Path) -> Set[str]:
    """Parse a python file using AST and return a set of all imported root module names."""
    imports = set()
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def collect_layer_imports(subfolder: str) -> Dict[str, Set[str]]:
    """Map each file in subfolder to its imported modules."""
    res = {}
    target = SRC_DIR / subfolder
    if not target.exists():
        return res

    for py_file in target.rglob("*.py"):
        res[py_file.name] = collect_module_imports(py_file)
    return res


def test_core_and_schemas_have_no_pyside6():
    """Core and Schemas MUST NOT import PySide6 (headless pure-python rule)."""
    for layer in ["core", "schemas"]:
        layer_imports = collect_layer_imports(layer)
        for fname, imports in layer_imports.items():
            assert "PySide6" not in imports, f"Violation: {layer}/{fname} imports PySide6!"


def test_network_has_no_ui_or_workers():
    """Network layer MUST NOT depend on UI or Workers."""
    net_imports = collect_layer_imports("network")
    for fname, imports in net_imports.items():
        assert "ui" not in imports, f"Violation: network/{fname} imports ui!"
        assert "workers" not in imports, f"Violation: network/{fname} imports workers!"


def test_workers_have_no_ui():
    """Workers layer MUST NOT import UI widgets or windows."""
    workers_imports = collect_layer_imports("workers")
    for fname, imports in workers_imports.items():
        assert "ui" not in imports, f"Violation: workers/{fname} imports ui!"


def test_ui_has_no_direct_network():
    """UI widgets MUST NOT import network client directly (MVC rule)."""
    ui_imports = collect_layer_imports("ui")
    for fname, imports in ui_imports.items():
        assert "network" not in imports, f"Violation: ui/{fname} imports network directly!"


def test_license_compliance_no_agpl_or_pyqt():
    """Strictly forbid AGPL libraries (e.g. ultralytics) and PyQt across entire codebase."""
    forbidden = {"ultralytics", "PyQt5", "PyQt6"}
    for py_file in SRC_DIR.rglob("*.py"):
        imports = collect_module_imports(py_file)
        intersect = imports & forbidden
        assert not intersect, f"Violation in {py_file}: contains forbidden library {intersect}"
