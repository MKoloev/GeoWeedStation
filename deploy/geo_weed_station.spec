# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os
import sys

block_cipher = None

root_dir = Path(SPECPATH).resolve().parent
src_dir = root_dir / "src"

sys.path.insert(0, str(src_dir))

datas = [
    (str(src_dir / "geo_weed_station" / "ui" / "resources"), os.path.join("geo_weed_station", "ui", "resources")),
    (str(src_dir / "geo_weed_station" / "config" / "default_config.json"), os.path.join("geo_weed_station", "config")),
]

models_dir = root_dir / "models"
if models_dir.exists():
    datas.append((str(models_dir), "models"))

a = Analysis(
    [str(root_dir / "main.py")],
    pathex=[str(src_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "pydantic",
        "pydantic_settings",
        "scipy.spatial",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GeoWeedStation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GeoWeedStation',
)
