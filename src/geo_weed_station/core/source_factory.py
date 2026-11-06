"""Frame source abstractions and factory: single image, folder, video and drone run (no Qt dependencies)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import cv2

from geo_weed_station.core.exceptions import ImageReadError
from geo_weed_station.core.image_io import read_image
from geo_weed_station.core.xmp_parser import build_frame_payload
from geo_weed_station.schemas.onboard_api import SummaryDetections, summary_to_frame_payloads
from geo_weed_station.schemas.payload import FramePayload
from geo_weed_station.schemas.session import FileEntry

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})

SourceKind = Literal["image", "folder", "video", "drone_run"]


@runtime_checkable
class FrameSource(Protocol):
    """Provider of file entries and decoded frame payloads."""

    def entries(self) -> list[FileEntry]: ...

    def load(self, entry: FileEntry) -> FramePayload: ...


def scan_images(root: Path) -> list[Path]:
    """Recursively list supported image files in deterministic order."""
    root = Path(root)
    found = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(found, key=lambda p: str(p).lower())


def _entry_for(path: Path) -> FileEntry:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return FileEntry(path=str(path), size_bytes=size)


def load_image_payload(path: Path) -> FramePayload:
    """Decode an image and build a payload using only embedded XMP metadata."""
    image = read_image(path)
    h, w = image.shape[:2]
    payload = build_frame_payload(path, (w, h))
    return payload.model_copy(update={"image": image})


class ImageSource:
    """Single image source."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Image not found: {self.path}")

    def entries(self) -> list[FileEntry]:
        return [_entry_for(self.path)]

    def load(self, entry: FileEntry) -> FramePayload:
        return load_image_payload(Path(entry.path))


class FolderSource:
    """Recursive folder of images."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(f"Folder not found: {self.path}")

    def entries(self) -> list[FileEntry]:
        return [_entry_for(p) for p in scan_images(self.path)]

    def load(self, entry: FileEntry) -> FramePayload:
        return load_image_payload(Path(entry.path))


class VideoSource:
    """Video file source; `load` returns the first frame (streaming is done by VideoWorker)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")

    def entries(self) -> list[FileEntry]:
        return [_entry_for(self.path)]

    def load(self, entry: FileEntry) -> FramePayload:
        cap = cv2.VideoCapture(entry.path)
        try:
            if not cap.isOpened():
                raise ImageReadError(f"Cannot open video: {entry.path}")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ImageReadError(f"Cannot decode video frame: {entry.path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        h, w = rgb.shape[:2]
        return FramePayload(
            frame_id=Path(entry.path).stem,
            source_path=entry.path,
            image_size_px=(w, h),
            image=rgb,
        )


_WP_RE = re.compile(r"wp_?(\d+)", re.IGNORECASE)


class DroneRunSource:
    """Extracted drone run directory: images (XMP) plus optional summary.json with detections."""

    SUMMARY_NAME = "summary.json"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(f"Run directory not found: {self.path}")
        self._summary: SummaryDetections | None = None
        summary_path = self.path / self.SUMMARY_NAME
        if summary_path.is_file():
            try:
                self._summary = SummaryDetections.model_validate_json(
                    summary_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._summary = None
        self._counts: dict[int, int] = {}
        if self._summary is not None:
            self._counts = {it.wp_index: len(it.detections) for it in self._summary.items}

    @staticmethod
    def _wp_index(path: Path) -> int | None:
        m = _WP_RE.search(path.stem)
        return int(m.group(1)) if m else None

    def entries(self) -> list[FileEntry]:
        result: list[FileEntry] = []
        for p in scan_images(self.path):
            entry = _entry_for(p)
            idx = self._wp_index(p)
            if idx is not None and idx in self._counts:
                entry.detections_count = self._counts[idx]
            result.append(entry)
        return result

    def load(self, entry: FileEntry) -> FramePayload:
        path = Path(entry.path)
        payload = load_image_payload(path)
        idx = self._wp_index(path)
        if self._summary is not None and idx is not None:
            for fp in summary_to_frame_payloads(self._summary, payload.image_size_px):
                if fp.frame_id.endswith(f"_wp{idx}"):
                    return payload.model_copy(update={"detections": list(fp.detections)})
        return payload


class SourceFactory:
    """Factory creating frame sources by kind."""

    @staticmethod
    def create(kind: SourceKind, path: Path) -> FrameSource:
        if kind == "image":
            return ImageSource(path)
        if kind == "folder":
            return FolderSource(path)
        if kind == "video":
            return VideoSource(path)
        if kind == "drone_run":
            return DroneRunSource(path)
        raise ValueError(f"Unknown source kind: {kind!r}")
