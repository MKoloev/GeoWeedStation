"""Helpers producing real decodable JPEGs (optionally with XMP) and mp4 videos for tests."""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

_XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def make_real_jpeg(path: Path, xmp_xml: str | None = None, size=(64, 48), seed: int = 0) -> Path:
    """Write a decodable JPEG; if xmp_xml given, insert an APP1 XMP segment after SOI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    data = buf.tobytes()
    if xmp_xml is not None:
        payload = _XMP_JPEG_HEADER + xmp_xml.encode("utf-8")
        segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
        data = data[:2] + segment + data[2:]
    path.write_bytes(data)
    return path


def make_mp4(path: Path, n_frames: int = 10, size=(64, 48), fps: int = 10) -> Path:
    """Write a small mp4 video with n_frames distinct frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
