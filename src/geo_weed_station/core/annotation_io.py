"""Annotation persistence and serialization utilities for GeoWeedStation (pure Python)."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_weed_station.core.image_io import get_image_size
from geo_weed_station.schemas.entities import BBox, DetectedWeed
from geo_weed_station.schemas.session import FileEntry, ProcessingStatus

logger = logging.getLogger(__name__)

DEFAULT_ANNOTATION_NAME = "annotations.json"
KNOWN_ANNOTATION_NAMES = ("annotations.json", "detections.json", "summary.json")

_WP_RE = re.compile(r"wp_?(\d+)|frame_?(\d+)|capture_?(\d+)", re.IGNORECASE)


class AnnotationsMap(dict[str, list[DetectedWeed]]):
    """Dictionary mapping image paths to lists of DetectedWeed, preserving frame status and metadata."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.statuses: dict[str, ProcessingStatus] = {}
        self.xmp_flags: dict[str, bool] = {}
        self.detections_counts: dict[str, int | None] = {}

    def get_status(
        self,
        path: str | Path,
        default: ProcessingStatus | None = None,
    ) -> ProcessingStatus | None:
        """Lookup saved processing status by path, filename, or stem."""
        p = Path(path)
        for key in (str(p), str(p.resolve() if p.is_absolute() else p), p.name, p.name.lower(), p.stem, p.stem.lower()):
            if key in self.statuses:
                return self.statuses[key]
        return default

    def get_has_xmp(
        self,
        path: str | Path,
        default: bool | None = None,
    ) -> bool | None:
        """Lookup saved XMP flag by path, filename, or stem."""
        p = Path(path)
        for key in (str(p), str(p.resolve() if p.is_absolute() else p), p.name, p.name.lower(), p.stem, p.stem.lower()):
            if key in self.xmp_flags:
                return self.xmp_flags[key]
        return default

    def get_detections_count(
        self,
        path: str | Path,
        default: int | None = None,
    ) -> int | None:
        """Lookup saved detections count by path, filename, or stem."""
        p = Path(path)
        for key in (str(p), str(p.resolve() if p.is_absolute() else p), p.name, p.name.lower(), p.stem, p.stem.lower()):
            if key in self.detections_counts:
                return self.detections_counts[key]
        return default


def find_annotation_file(folder: Path) -> Path | None:
    """Check whether a recognized annotation file exists in the directory.
    
    Checks in priority order:
    1. annotations.json
    2. detections.json
    3. summary.json
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None

    for name in KNOWN_ANNOTATION_NAMES:
        candidate = folder / name
        if candidate.is_file():
            return candidate

    return None


def _extract_index_from_name(name: str) -> int | None:
    """Extract numeric frame index from filename like frame_002.jpg or capture_005.jpg."""
    m = _WP_RE.search(name)
    if m:
        for g in m.groups():
            if g is not None:
                try:
                    return int(g)
                except ValueError:
                    pass
    return None


def parse_detection_item(
    item: dict[str, Any],
    default_class: str = "weed",
    default_id: str = "det_0",
    img_size: tuple[int, int] = (1024, 1024),
) -> DetectedWeed | None:
    """Parse a single raw detection dictionary into a validated DetectedWeed entity."""
    try:
        weed_id = str(item.get("weed_id") or item.get("id") or default_id)
        class_id = int(item.get("class_id", 0))
        class_name = str(item.get("class_name") or item.get("name") or default_class)
        conf = float(item.get("confidence") if item.get("confidence") is not None else item.get("score", 1.0))
        conf = max(0.0, min(1.0, conf))

        bbox_raw = item.get("bbox")
        bbox_px = item.get("bbox_px")

        x, y, w, h = 0.0, 0.0, 0.0, 0.0

        if isinstance(bbox_raw, dict):
            x = float(bbox_raw.get("x", 0.0))
            y = float(bbox_raw.get("y", 0.0))
            w = float(bbox_raw.get("w", 0.0))
            h = float(bbox_raw.get("h", 0.0))
        elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            b = [float(v) for v in bbox_raw]
            if max(b) <= 1.0 and min(b) >= 0.0:
                # Normalized [x, y, w, h] or [x1, y1, x2, y2]
                if b[2] >= b[0] and b[3] >= b[1] and (b[2] > 0.0 or b[3] > 0.0):
                    # Assume [x1, y1, x2, y2]
                    x, y, w, h = b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
                else:
                    x, y, w, h = b[0], b[1], b[2], b[3]
            else:
                # Pixel coordinates [x1, y1, x2, y2]
                img_w, img_h = max(1, img_size[0]), max(1, img_size[1])
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                x = min(1.0, max(0.0, x1 / img_w))
                y = min(1.0, max(0.0, y1 / img_h))
                w = min(1.0 - x, max(0.0, (x2 - x1) / img_w))
                h = min(1.0 - y, max(0.0, (y2 - y1) / img_h))
        elif isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4:
            img_w, img_h = max(1, img_size[0]), max(1, img_size[1])
            x1, y1, x2, y2 = float(bbox_px[0]), float(bbox_px[1]), float(bbox_px[2]), float(bbox_px[3])
            x = min(1.0, max(0.0, x1 / img_w))
            y = min(1.0, max(0.0, y1 / img_h))
            w = min(1.0 - x, max(0.0, (x2 - x1) / img_w))
            h = min(1.0 - y, max(0.0, (y2 - y1) / img_h))
        else:
            return None

        # Clamp BBox bounds
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))

        bbox = BBox(x=x, y=y, w=w, h=h)

        world_x = item.get("world_x")
        world_y = item.get("world_y")
        target_local = item.get("target_local_coords")
        if isinstance(target_local, dict):
            if world_x is None:
                world_x = target_local.get("x")
            if world_y is None:
                world_y = target_local.get("y")

        stage_name = item.get("stage_name")
        class_prob = item.get("class_prob")

        return DetectedWeed(
            weed_id=weed_id,
            class_id=class_id,
            class_name=class_name,
            confidence=round(conf, 4),
            bbox=bbox,
            world_x=float(world_x) if world_x is not None else None,
            world_y=float(world_y) if world_y is not None else None,
            stage_name=str(stage_name) if stage_name is not None else None,
            class_prob=float(class_prob) if class_prob is not None else None,
        )
    except Exception as exc:
        logger.debug("Could not parse detection item: %s (%s)", item, exc)
        return None


class _RawFrameRecord:
    """Internal container for parsed raw frame annotation record."""

    def __init__(
        self,
        raw_dets: list[dict[str, Any]],
        status: ProcessingStatus | None = None,
        has_xmp: bool | None = None,
        count: int | None = None,
    ) -> None:
        self.raw_dets = raw_dets
        self.status = status
        self.has_xmp = has_xmp
        self.count = count


def save_annotations_to_folder(
    folder: Path,
    detections_by_path: dict[str, list[DetectedWeed]],
    entries: Sequence[FileEntry] | None = None,
    file_name: str = DEFAULT_ANNOTATION_NAME,
) -> Path:
    """Save in-memory detections and frame states into a structured annotations.json file inside folder.
    
    Uses atomic write via temporary file to prevent file corruption.
    Preserves exact ProcessingStatus, XMP metadata flag, and detection count for each frame.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / file_name
    tmp_path = folder / f"{file_name}.tmp"

    # Build path-to-entry lookup map to preserve statuses and metadata
    entry_by_path: dict[str, FileEntry] = {}
    candidate_paths: list[Path] = []
    if entries:
        for e in entries:
            ep = Path(e.path)
            candidate_paths.append(ep)
            entry_by_path[str(ep)] = e
            try:
                entry_by_path[str(ep.resolve())] = e
            except Exception:
                pass
            entry_by_path[ep.name] = e
            entry_by_path[ep.name.lower()] = e
            entry_by_path[ep.stem] = e
            entry_by_path[ep.stem.lower()] = e
    else:
        seen = set()
        for p_str in detections_by_path:
            p = Path(p_str)
            if p not in seen:
                seen.add(p)
                candidate_paths.append(p)

    frames_dict: dict[str, dict[str, Any]] = {}
    classes_set: set[str] = set()
    total_detections = 0

    for p in candidate_paths:
        norm_str = str(p)
        resolved_str = str(p.resolve())
        dets = detections_by_path.get(norm_str) or detections_by_path.get(resolved_str) or []

        try:
            rel_key = str(p.relative_to(folder))
        except ValueError:
            rel_key = p.name

        serialized_dets: list[dict[str, Any]] = []
        for d in dets:
            classes_set.add(d.class_name)
            det_dict: dict[str, Any] = {
                "weed_id": d.weed_id,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 4),
                "bbox": {
                    "x": round(float(d.bbox.x), 5),
                    "y": round(float(d.bbox.y), 5),
                    "w": round(float(d.bbox.w), 5),
                    "h": round(float(d.bbox.h), 5),
                },
            }
            if d.world_x is not None:
                det_dict["world_x"] = round(float(d.world_x), 4)
            if d.world_y is not None:
                det_dict["world_y"] = round(float(d.world_y), 4)
            if d.stage_name is not None:
                det_dict["stage_name"] = d.stage_name
            if d.class_prob is not None:
                det_dict["class_prob"] = round(float(d.class_prob), 4)
            serialized_dets.append(det_dict)

        entry = (
            entry_by_path.get(norm_str)
            or entry_by_path.get(resolved_str)
            or entry_by_path.get(p.name)
            or entry_by_path.get(p.name.lower())
            or entry_by_path.get(p.stem)
            or entry_by_path.get(p.stem.lower())
        )

        if entry is not None:
            if entry.status != ProcessingStatus.PENDING:
                status_str = entry.status.value
            elif dets:
                status_str = ProcessingStatus.DONE.value if entry.has_xmp else ProcessingStatus.NO_METADATA.value
            else:
                status_str = entry.status.value
            has_xmp = entry.has_xmp
            if entry.status in (ProcessingStatus.PENDING, ProcessingStatus.ERROR) and not dets:
                count = None
            else:
                count = entry.detections_count if entry.detections_count is not None else len(serialized_dets)
        else:
            status_str = ProcessingStatus.DONE.value if dets else ProcessingStatus.PENDING.value
            has_xmp = False
            count = len(serialized_dets) if dets else None

        frames_dict[rel_key] = {
            "status": status_str,
            "has_xmp": has_xmp,
            "detections_count": count,
            "detections": serialized_dets,
        }
        total_detections += len(serialized_dets)

    payload = {
        "version": "1.0",
        "format": "geoweed_annotations",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folder": folder.name,
        "total_frames": len(frames_dict),
        "total_detections": total_detections,
        "classes": sorted(classes_set),
        "frames": frames_dict,
    }

    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)
    logger.info("Saved %d annotations for %d frames to %s", total_detections, len(frames_dict), out_path)
    return out_path


def load_annotations_file(
    file_path: Path,
    image_paths: Sequence[Path | str] | None = None,
) -> AnnotationsMap:
    """Load and parse detections and statuses from annotations.json, detections.json, or summary.json.
    
    Returns an AnnotationsMap (dict mapping image path to list of DetectedWeed,
    with preserved statuses, xmp flags, and detection counts).
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        return AnnotationsMap()

    try:
        content = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read JSON from %s: %s", file_path, exc)
        return AnnotationsMap()

    if not isinstance(content, dict):
        return AnnotationsMap()

    by_filename: dict[str, _RawFrameRecord] = {}
    by_stem: dict[str, _RawFrameRecord] = {}
    by_index: dict[int, _RawFrameRecord] = {}

    # Case 1: Standard GeoWeed format with top-level "frames"
    if "frames" in content and isinstance(content["frames"], dict):
        for key, val in content["frames"].items():
            key_p = Path(key)
            if isinstance(val, dict):
                raw_dets = val.get("detections", [])
                st_raw = val.get("status")
                st = None
                if st_raw is not None:
                    try:
                        st = ProcessingStatus(str(st_raw).lower())
                    except ValueError:
                        st = None
                has_xmp = val.get("has_xmp")
                cnt = val.get("detections_count")
                if st in (ProcessingStatus.PENDING, ProcessingStatus.ERROR) and not raw_dets:
                    cnt = None
                elif cnt is None and isinstance(raw_dets, list):
                    cnt = len(raw_dets)
                rec = _RawFrameRecord(
                    raw_dets=raw_dets if isinstance(raw_dets, list) else [],
                    status=st,
                    has_xmp=has_xmp if isinstance(has_xmp, bool) else None,
                    count=cnt,
                )
            elif isinstance(val, list):
                rec = _RawFrameRecord(raw_dets=val, status=None, has_xmp=None, count=len(val))
            else:
                continue

            by_filename[key_p.name.lower()] = rec
            by_stem[key_p.stem.lower()] = rec
            idx = _extract_index_from_name(key_p.stem)
            if idx is not None:
                by_index[idx] = rec

    # Case 2: summary.json format with "items" (wp_index, camera_x, camera_y, detections)
    elif "items" in content and isinstance(content["items"], list):
        for it in content["items"]:
            if isinstance(it, dict) and "wp_index" in it:
                try:
                    idx = int(it["wp_index"])
                    dets = it.get("detections", [])
                    raw_dets = dets if isinstance(dets, list) else []
                    if "altitude_m" in it:
                        alt_raw = it.get("altitude_m")
                        has_geo = (
                            isinstance(alt_raw, (int, float))
                            and not isinstance(alt_raw, bool)
                            and math.isfinite(float(alt_raw))
                            and float(alt_raw) > 0.0
                        )
                    else:
                        has_geo = True
                    rec = _RawFrameRecord(
                        raw_dets=raw_dets,
                        status=ProcessingStatus.DONE if has_geo else ProcessingStatus.NO_METADATA,
                        has_xmp=has_geo,
                        count=len(raw_dets),
                    )
                    by_index[idx] = rec
                    by_filename[f"wp_{idx}"] = rec
                    by_stem[f"wp_{idx}"] = rec
                    by_filename[f"frame_{idx:03d}.jpg"] = rec
                    by_stem[f"frame_{idx:03d}"] = rec
                    by_filename[f"capture_{idx:03d}.jpg"] = rec
                    by_stem[f"capture_{idx:03d}"] = rec
                except (ValueError, TypeError):
                    pass

    # Case 3: Flat dictionary {filename: [detections] or {status: ..., detections: ...}}
    else:
        for k, v in content.items():
            key_p = Path(k)
            if isinstance(v, list):
                rec = _RawFrameRecord(raw_dets=v, status=None, has_xmp=None, count=len(v))
            elif isinstance(v, dict):
                raw_dets = v.get("detections", [])
                st_raw = v.get("status")
                st = None
                if st_raw is not None:
                    try:
                        st = ProcessingStatus(str(st_raw).lower())
                    except ValueError:
                        st = None
                rec = _RawFrameRecord(
                    raw_dets=raw_dets if isinstance(raw_dets, list) else [],
                    status=st,
                    has_xmp=v.get("has_xmp"),
                    count=v.get("detections_count"),
                )
            else:
                continue

            by_filename[key_p.name.lower()] = rec
            by_stem[key_p.stem.lower()] = rec
            idx = _extract_index_from_name(key_p.stem)
            if idx is not None:
                by_index[idx] = rec

    if not by_filename and not by_index:
        return AnnotationsMap()

    # Determine candidates to resolve
    candidates: list[Path]
    if image_paths:
        candidates = [Path(p) for p in image_paths]
    else:
        candidates = [Path(k) for k in by_filename]

    result = AnnotationsMap()

    for img_p in candidates:
        rec = None

        # 1. Direct filename match
        if img_p.name.lower() in by_filename:
            rec = by_filename[img_p.name.lower()]
        # 2. Stem match
        elif img_p.stem.lower() in by_stem:
            rec = by_stem[img_p.stem.lower()]
        # 3. Numeric index match
        else:
            idx = _extract_index_from_name(img_p.stem)
            if idx is not None and idx in by_index:
                rec = by_index[idx]

        if rec is not None:
            resolved_img_size: tuple[int, int] | None = None
            parsed: list[DetectedWeed] = []
            for d_idx, d_dict in enumerate(rec.raw_dets):
                if isinstance(d_dict, dict):
                    if resolved_img_size is None and not isinstance(d_dict.get("bbox"), dict):
                        resolved_img_size = get_image_size(img_p) if img_p.is_file() else (1024, 1024)
                    weed = parse_detection_item(
                        d_dict,
                        default_class="weed",
                        default_id=f"det_{d_idx}",
                        img_size=resolved_img_size or (1024, 1024),
                    )
                    if weed is not None:
                        parsed.append(weed)

            norm_str = str(img_p)
            result[norm_str] = parsed
            try:
                resolved_str = str(img_p.resolve())
                result[resolved_str] = parsed
            except Exception:
                resolved_str = norm_str

            for k in (norm_str, resolved_str, img_p.name, img_p.name.lower(), img_p.stem, img_p.stem.lower()):
                if rec.status is not None:
                    result.statuses[k] = rec.status
                if rec.has_xmp is not None:
                    result.xmp_flags[k] = rec.has_xmp
                if rec.count is not None:
                    result.detections_counts[k] = rec.count

    return result
