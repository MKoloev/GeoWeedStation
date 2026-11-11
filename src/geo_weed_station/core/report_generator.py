"""Comprehensive report generator for folder object detection and agronomic analysis (pure Python)."""

from __future__ import annotations

import base64
import csv
import html
import io
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from geo_weed_station.config.settings import DetectorSettings
from geo_weed_station.core.map_engine.deduplication import deduplicate_weeds
from geo_weed_station.core.map_engine.exporter import MapExporter
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.core.onnx_detector import is_background_class
from geo_weed_station.core.xmp_parser import extract_spatial_data
from geo_weed_station.schemas.entities import DetectedWeed
from geo_weed_station.schemas.map_models import PngExportOptions, PrescriptionZone, WeedPoint
from geo_weed_station.schemas.report import (
    DetectionReportRow,
    FolderDetectionReport,
    FrameReportItem,
)
from geo_weed_station.schemas.session import FileEntry, ProcessingStatus


def _lookup_detections(
    path_str: str,
    detections_by_path: dict[str, list[DetectedWeed]],
) -> list[DetectedWeed]:
    """Lookup detections list for a file path across raw, resolved, and filename keys."""
    if path_str in detections_by_path:
        return list(detections_by_path[path_str])
    p = Path(path_str)
    for candidate in (str(p), p.name, p.stem):
        if candidate in detections_by_path:
            return list(detections_by_path[candidate])
    try:
        resolved = str(p.resolve())
        if resolved in detections_by_path:
            return list(detections_by_path[resolved])
    except Exception:
        pass
    return []


class ReportGenerator:
    """Builds and exports structured analytical reports from folder detection sessions."""

    def __init__(self, exporter: MapExporter | None = None) -> None:
        self._exporter = exporter or MapExporter()

    def build_report(
        self,
        entries: Sequence[FileEntry],
        detections_by_path: dict[str, list[DetectedWeed]],
        map_engine: MapEngine | None = None,
        detector_settings: DetectorSettings | None = None,
        source_folder: Path | None = None,
        hidden_classes: Iterable[str] | None = None,
        min_confidence: float | None = None,
    ) -> FolderDetectionReport:
        """Aggregate file entries, detections, and optional map analytics into FolderDetectionReport."""
        now_utc = datetime.now(timezone.utc)
        report_id = f"report_{now_utc.strftime('%Y%m%d_%H%M%S')}"
        created_at = now_utc.isoformat()

        if source_folder is None and entries:
            source_folder = Path(entries[0].path).parent
        folder_path_str = str(source_folder) if source_folder is not None else ""
        folder_name = source_folder.name if source_folder is not None and source_folder.name else "session"

        model_name = "unknown"
        classifier_type = "none"
        conf_thr = 0.25
        iou_thr = 0.45
        if detector_settings is not None:
            model_name = Path(detector_settings.model_path).name
            classifier_type = str(detector_settings.classifier_type or "none")
            conf_thr = float(detector_settings.confidence_threshold)
            iou_thr = float(detector_settings.nms_iou_threshold)

        active_hidden: frozenset[str] = (
            frozenset(hidden_classes)
            if hidden_classes is not None
            else (map_engine.hidden_classes if map_engine is not None else frozenset())
        )
        active_min_conf: float = (
            float(min_confidence)
            if min_confidence is not None
            else (map_engine.min_confidence if map_engine is not None else 0.0)
        )
        if min_confidence is not None:
            conf_thr = active_min_conf

        def _is_visible(cls_name: str, conf: float) -> bool:
            return (
                not is_background_class(cls_name)
                and cls_name not in active_hidden
                and conf >= active_min_conf
            )

        # Build footprint lookup from MapEngine if available
        footprint_by_id: dict[str, object] = {}
        raw_points_by_frame_and_id: dict[tuple[str, str], WeedPoint] = {}
        dedup_lookup: dict[tuple[str, str], WeedPoint] = {}
        map_points: list[WeedPoint] = []
        dedup_points: list[WeedPoint] = []

        if map_engine is not None:
            for fp in map_engine.weed_map.footprints:
                footprint_by_id[fp.frame_id] = fp
            map_points = [
                pt
                for pt in map_engine.weed_map.points
                if _is_visible(pt.class_name, pt.confidence)
            ]
            dedup_points = deduplicate_weeds(map_points)
            for pt in map_points:
                raw_points_by_frame_and_id[(pt.source_frame, pt.weed_id)] = pt
            for dpt in dedup_points:
                dedup_lookup[(dpt.source_frame, dpt.weed_id)] = dpt
                if dpt.best_frame_id:
                    dedup_lookup[(dpt.best_frame_id, dpt.weed_id)] = dpt

        frame_items: list[FrameReportItem] = []
        detection_rows: list[DetectionReportRow] = []

        by_class_counter: Counter[str] = Counter()
        by_stage_counter: Counter[str] = Counter()
        class_stage_matrix: dict[str, dict[str, int]] = {}
        all_confidences: list[float] = []

        total_files = len(entries)
        processed_files = 0
        files_with_xmp = 0
        files_without_xmp = 0
        error_files = 0
        frames_with_detections = 0
        empty_frames = 0
        total_size_bytes = 0

        # If entries is empty but map_engine has points, synthesize from map_points
        if not entries and map_points:
            frames_grouped: dict[str, list[WeedPoint]] = {}
            for pt in map_points:
                frames_grouped.setdefault(pt.source_frame, []).append(pt)
            for frame_id, pts in frames_grouped.items():
                f_cls: Counter[str] = Counter()
                f_stg: Counter[str] = Counter()
                confs = [p.confidence for p in pts]
                for p in pts:
                    f_cls[p.class_name] += 1
                    stg = p.stage_name or "не определена"
                    f_stg[stg] += 1
                    by_class_counter[p.class_name] += 1
                    if p.stage_name:
                        by_stage_counter[p.stage_name] += 1
                    class_stage_matrix.setdefault(p.class_name, {})
                    class_stage_matrix[p.class_name][stg] = class_stage_matrix[p.class_name].get(stg, 0) + 1
                    all_confidences.append(p.confidence)

                    dpt = dedup_lookup.get((frame_id, p.weed_id))
                    bx = p.bbox.x if hasattr(p.bbox, "x") else 0.0
                    by = p.bbox.y if hasattr(p.bbox, "y") else 0.0
                    bw = p.bbox.w if hasattr(p.bbox, "w") else 0.0
                    bh = p.bbox.h if hasattr(p.bbox, "h") else 0.0
                    detection_rows.append(
                        DetectionReportRow(
                            weed_id=p.weed_id,
                            source_frame=frame_id,
                            class_id=0,
                            class_name=p.class_name,
                            stage_name=p.stage_name,
                            confidence=round(float(p.confidence), 4),
                            class_prob=round(float(p.class_prob), 4) if p.class_prob is not None else None,
                            bbox_x=round(float(bx), 5),
                            bbox_y=round(float(by), 5),
                            bbox_w=round(float(bw), 5),
                            bbox_h=round(float(bh), 5),
                            coord_system=p.coord_system.value,
                            world_x=round(float(p.x), 4),
                            world_y=round(float(p.y), 4),
                            lat=round(float(p.lat), 8) if p.lat is not None else None,
                            lon=round(float(p.lon), 8) if p.lon is not None else None,
                            is_unique=dpt is not None,
                            duplicate_count=dpt.duplicate_count if dpt is not None else 1,
                            best_frame_id=dpt.best_frame_id if dpt is not None else frame_id,
                        )
                    )

                frame_items.append(
                    FrameReportItem(
                        frame_id=frame_id,
                        file_name=frame_id,
                        file_path=frame_id,
                        size_bytes=0,
                        status=ProcessingStatus.DONE.value,
                        has_xmp=True,
                        coordinate_system=pts[0].coord_system.value if pts else None,
                        detections_count=len(pts),
                        mean_confidence=round(sum(confs) / len(confs), 4) if confs else 0.0,
                        max_confidence=round(max(confs), 4) if confs else 0.0,
                        by_class=dict(f_cls),
                        by_stage=dict(f_stg),
                    )
                )
            total_files = len(frame_items)
            processed_files = total_files
            files_with_xmp = total_files
            frames_with_detections = sum(1 for f in frame_items if f.detections_count > 0)
            empty_frames = total_files - frames_with_detections

        for entry in entries:
            p = Path(entry.path)
            frame_id = p.stem
            raw_dets = _lookup_detections(entry.path, detections_by_path)
            dets = [d for d in raw_dets if _is_visible(d.class_name, d.confidence)]
            has_det_record = bool(raw_dets) or (
                entry.path in detections_by_path
                or str(p) in detections_by_path
                or p.name in detections_by_path
                or p.stem in detections_by_path
            )
            det_count = (
                len(dets)
                if (has_det_record or active_hidden or active_min_conf > 0.0)
                else (entry.detections_count or 0)
            )

            total_size_bytes += max(0, entry.size_bytes)
            if entry.has_xmp:
                files_with_xmp += 1
            else:
                files_without_xmp += 1

            if entry.status in (ProcessingStatus.DONE, ProcessingStatus.NO_METADATA) or raw_dets:
                processed_files += 1
            elif entry.status == ProcessingStatus.ERROR:
                error_files += 1

            if det_count > 0:
                frames_with_detections += 1
            else:
                empty_frames += 1

            # Extract camera pose from footprint or XMP if available
            cs_str: str | None = None
            cam_x: float | None = None
            cam_y: float | None = None
            alt_m: float | None = None
            yaw_deg: float | None = None

            if entry.has_xmp and p.is_file():
                try:
                    sp = extract_spatial_data(p)
                    if sp.pose is not None:
                        cs_str = sp.pose.coordinate_system.value
                        cam_x = round(float(sp.pose.x), 6)
                        cam_y = round(float(sp.pose.y), 6)
                        alt_m = round(float(sp.pose.altitude_m), 3)
                        yaw_deg = round(float(sp.pose.yaw_deg), 2)
                except Exception:
                    pass

            f_cls_counter: Counter[str] = Counter()
            f_stg_counter: Counter[str] = Counter()
            f_confs: list[float] = []

            for det in dets:
                f_cls_counter[det.class_name] += 1
                by_class_counter[det.class_name] += 1
                f_confs.append(float(det.confidence))
                all_confidences.append(float(det.confidence))

                stg_label = det.stage_name if det.stage_name else "не определена"
                if det.stage_name:
                    f_stg_counter[det.stage_name] += 1
                    by_stage_counter[det.stage_name] += 1

                class_stage_matrix.setdefault(det.class_name, {})
                class_stage_matrix[det.class_name][stg_label] = (
                    class_stage_matrix[det.class_name].get(stg_label, 0) + 1
                )

                # Match with projected WeedPoint if ingested in MapEngine
                wp = raw_points_by_frame_and_id.get((frame_id, det.weed_id))
                dpt = dedup_lookup.get((frame_id, det.weed_id))

                wx = wp.x if wp is not None else det.world_x
                wy = wp.y if wp is not None else det.world_y
                lat = wp.lat if wp is not None else None
                lon = wp.lon if wp is not None else None
                row_cs = wp.coord_system.value if wp is not None else cs_str

                detection_rows.append(
                    DetectionReportRow(
                        weed_id=det.weed_id,
                        source_frame=p.name,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        stage_name=det.stage_name,
                        confidence=round(float(det.confidence), 4),
                        class_prob=round(float(det.class_prob), 4) if det.class_prob is not None else None,
                        bbox_x=round(float(det.bbox.x), 5),
                        bbox_y=round(float(det.bbox.y), 5),
                        bbox_w=round(float(det.bbox.w), 5),
                        bbox_h=round(float(det.bbox.h), 5),
                        coord_system=row_cs,
                        world_x=round(float(wx), 4) if wx is not None else None,
                        world_y=round(float(wy), 4) if wy is not None else None,
                        lat=round(float(lat), 8) if lat is not None else None,
                        lon=round(float(lon), 8) if lon is not None else None,
                        is_unique=(dpt is not None) if map_points else True,
                        duplicate_count=dpt.duplicate_count if dpt is not None else 1,
                        best_frame_id=dpt.best_frame_id if dpt is not None else frame_id,
                    )
                )

            frame_items.append(
                FrameReportItem(
                    frame_id=frame_id,
                    file_name=p.name,
                    file_path=str(p),
                    size_bytes=entry.size_bytes,
                    status=entry.status.value,
                    has_xmp=entry.has_xmp,
                    coordinate_system=cs_str,
                    camera_x=cam_x,
                    camera_y=cam_y,
                    altitude_m=alt_m,
                    yaw_deg=yaw_deg,
                    detections_count=det_count,
                    mean_confidence=round(sum(f_confs) / len(f_confs), 4) if f_confs else 0.0,
                    max_confidence=round(max(f_confs), 4) if f_confs else 0.0,
                    by_class=dict(f_cls_counter),
                    by_stage=dict(f_stg_counter),
                )
            )

        total_detections = len(detection_rows)
        mean_conf = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0

        # Spatial & Agronomic metrics
        has_georef = bool(
            (map_engine is not None and not map_engine.is_locked() and (map_points or map_engine.weed_map.footprints))
            or files_with_xmp > 0
        )
        coord_sys_global: str | None = None
        surveyed_area_m2 = 0.0
        mean_density_per_m2 = 0.0
        critical_area_percent = 0.0
        zones: list[PrescriptionZone] = []

        if map_engine is not None and not map_engine.is_locked() and (map_points or map_engine.weed_map.points):
            coord_sys_global = map_engine.weed_map.coord_system.value
            prev_hidden = map_engine.hidden_classes
            prev_conf = map_engine.min_confidence
            map_engine.set_filter(hidden_classes=active_hidden, min_confidence=active_min_conf)
            map_engine.build_heatmap()
            stats = map_engine.statistics()
            surveyed_area_m2 = stats.surveyed_area_m2
            mean_density_per_m2 = stats.mean_density_per_m2
            critical_area_percent = stats.critical_area_percent
            zones = map_engine.build_prescription_zones()
            unique_weeds = len(dedup_points) if dedup_points else len(map_points)
            if prev_hidden != active_hidden or abs(prev_conf - active_min_conf) > 1e-6:
                map_engine.set_filter(hidden_classes=prev_hidden, min_confidence=prev_conf)
        else:
            for f in frame_items:
                if f.coordinate_system:
                    coord_sys_global = f.coordinate_system
                    break
            unique_weeds = total_detections

        dedup_ratio = round(total_detections / max(1, unique_weeds), 2) if total_detections > 0 else 1.0

        return FolderDetectionReport(
            report_id=report_id,
            created_at=created_at,
            folder_name=folder_name,
            folder_path=folder_path_str,
            model_name=model_name,
            classifier_type=classifier_type,
            confidence_threshold=conf_thr,
            nms_iou_threshold=iou_thr,
            total_files=total_files,
            processed_files=processed_files,
            files_with_xmp=files_with_xmp,
            files_without_xmp=files_without_xmp,
            error_files=error_files,
            frames_with_detections=frames_with_detections,
            empty_frames=empty_frames,
            total_size_bytes=total_size_bytes,
            total_detections=total_detections,
            unique_weeds=unique_weeds,
            deduplication_ratio=dedup_ratio,
            mean_confidence=mean_conf,
            by_class=dict(by_class_counter),
            by_stage=dict(by_stage_counter),
            class_stage_matrix=class_stage_matrix,
            has_georef=has_georef,
            coordinate_system=coord_sys_global,
            surveyed_area_m2=surveyed_area_m2,
            mean_density_per_m2=mean_density_per_m2,
            critical_area_percent=critical_area_percent,
            prescription_zones_count=len(zones),
            zones=zones,
            frames=frame_items,
            detections=detection_rows,
        )

    def export_json(self, report: FolderDetectionReport, path: Path) -> Path:
        """Write complete FolderDetectionReport to a formatted JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return path

    def export_frames_csv(self, report: FolderDetectionReport, path: Path) -> Path:
        """Export per-frame summary table to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "frame_id",
            "file_name",
            "status",
            "size_bytes",
            "has_xmp",
            "coordinate_system",
            "camera_x",
            "camera_y",
            "altitude_m",
            "yaw_deg",
            "detections_count",
            "mean_confidence",
            "max_confidence",
            "classes_summary",
            "stages_summary",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in report.frames:
                cls_summary = "; ".join(f"{k}: {v}" for k, v in sorted(item.by_class.items()))
                stg_summary = "; ".join(f"{k}: {v}" for k, v in sorted(item.by_stage.items()))
                writer.writerow(
                    [
                        item.frame_id,
                        item.file_name,
                        item.status,
                        item.size_bytes,
                        item.has_xmp,
                        item.coordinate_system or "",
                        item.camera_x if item.camera_x is not None else "",
                        item.camera_y if item.camera_y is not None else "",
                        item.altitude_m if item.altitude_m is not None else "",
                        item.yaw_deg if item.yaw_deg is not None else "",
                        item.detections_count,
                        f"{item.mean_confidence:.4f}",
                        f"{item.max_confidence:.4f}",
                        cls_summary,
                        stg_summary,
                    ]
                )
        return path

    def export_detections_csv(self, report: FolderDetectionReport, path: Path) -> Path:
        """Export detailed per-object detections table (with classification & coordinates) to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "weed_id",
            "source_frame",
            "class_id",
            "class_name",
            "stage_name",
            "confidence",
            "class_prob",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "coord_system",
            "world_x",
            "world_y",
            "lat",
            "lon",
            "is_unique",
            "duplicate_count",
            "best_frame_id",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for d in report.detections:
                writer.writerow(
                    [
                        d.weed_id,
                        d.source_frame,
                        d.class_id,
                        d.class_name,
                        d.stage_name or "",
                        f"{d.confidence:.4f}",
                        f"{d.class_prob:.4f}" if d.class_prob is not None else "",
                        f"{d.bbox_x:.5f}",
                        f"{d.bbox_y:.5f}",
                        f"{d.bbox_w:.5f}",
                        f"{d.bbox_h:.5f}",
                        d.coord_system or "",
                        f"{d.world_x:.4f}" if d.world_x is not None else "",
                        f"{d.world_y:.4f}" if d.world_y is not None else "",
                        f"{d.lat:.8f}" if d.lat is not None else "",
                        f"{d.lon:.8f}" if d.lon is not None else "",
                        d.is_unique,
                        d.duplicate_count,
                        d.best_frame_id or "",
                    ]
                )
        return path

    def render_map_png_bytes(
        self,
        map_engine: MapEngine | None,
        options: PngExportOptions | None = None,
    ) -> bytes | None:
        """Render map PNG into memory bytes if map_engine has georeferenced points or footprints."""
        if (
            map_engine is None
            or map_engine.is_locked()
            or (not map_engine.weed_map.points and not map_engine.weed_map.footprints)
        ):
            return None
        try:
            grid = map_engine._current_grid or map_engine.build_heatmap()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                tmp_path = Path(tmp.name)
                self._exporter.to_png(
                    wm=map_engine.filtered_weed_map(),
                    grid=grid,
                    path=tmp_path,
                    options=options or PngExportOptions(),
                )
                return tmp_path.read_bytes()
        except Exception:
            return None

    def export_html(
        self,
        report: FolderDetectionReport,
        path: Path,
        map_png_bytes: bytes | None = None,
    ) -> Path:
        """Render a self-contained, print-ready HTML analytical report in Russian."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        esc = html.escape
        total_mb = report.total_size_bytes / (1024.0 * 1024.0)

        # Class breakdown rows
        class_rows_html = ""
        for cls_name, cnt in sorted(report.by_class.items(), key=lambda kv: kv[1], reverse=True):
            pct = (cnt / max(1, report.total_detections)) * 100.0
            class_rows_html += (
                f"<tr>"
                f"<td><strong>{esc(cls_name)}</strong></td>"
                f"<td>{cnt}</td>"
                f"<td>{pct:.1f}%</td>"
                f"<td><div class='bar-bg'><div class='bar-fill' style='width:{min(100.0, pct):.1f}%'></div></div></td>"
                f"</tr>"
            )
        if not class_rows_html:
            class_rows_html = "<tr><td colspan='4' class='muted'>Объекты не обнаружены</td></tr>"

        # Stage breakdown rows
        stage_rows_html = ""
        total_staged = sum(report.by_stage.values())
        for stg_name, cnt in sorted(report.by_stage.items(), key=lambda kv: kv[1], reverse=True):
            pct = (cnt / max(1, total_staged)) * 100.0
            stage_rows_html += (
                f"<tr>"
                f"<td><strong>{esc(stg_name)}</strong></td>"
                f"<td>{cnt}</td>"
                f"<td>{pct:.1f}%</td>"
                f"<td><div class='bar-bg'><div class='bar-fill bar-stage' style='width:{min(100.0, pct):.1f}%'></div></div></td>"
                f"</tr>"
            )
        if not stage_rows_html:
            stage_rows_html = "<tr><td colspan='4' class='muted'>Классификация стадий не выполнялась или нет данных</td></tr>"

        # Cross-matrix Class x Stage
        matrix_rows_html = ""
        for cls_name, stg_dict in sorted(report.class_stage_matrix.items()):
            stg_badges = ", ".join(f"{esc(s)}: <b>{c}</b>" for s, c in sorted(stg_dict.items(), key=lambda kv: kv[1], reverse=True))
            matrix_rows_html += f"<tr><td>{esc(cls_name)}</td><td>{stg_badges}</td></tr>"
        if not matrix_rows_html:
            matrix_rows_html = "<tr><td colspan='2' class='muted'>Нет данных</td></tr>"

        # Map embed
        map_section_html = ""
        if map_png_bytes:
            b64_map = base64.b64encode(map_png_bytes).decode("ascii")
            map_section_html = f"""
            <div class="section">
                <h2>🗺 Карта засоренности и тепловая карта плотности</h2>
                <div class="map-box">
                    <img src="data:image/png;base64,{b64_map}" alt="Карта засоренности" />
                </div>
            </div>
            """

        # Prescription zones table
        zones_rows_html = ""
        for z in report.zones:
            zones_rows_html += (
                f"<tr>"
                f"<td><strong>{esc(z.zone_id)}</strong></td>"
                f"<td>{z.area_m2:.2f} м²</td>"
                f"<td>{z.weed_count}</td>"
                f"<td>{z.mean_density_per_m2:.2f} шт/м²</td>"
                f"<td>{esc(z.dominant_class)}</td>"
                f"<td><span class='badge badge-warn'>{esc(z.treatment_recommendation)}</span></td>"
                f"</tr>"
            )
        if not zones_rows_html:
            zones_rows_html = "<tr><td colspan='6' class='muted'>Зоны критического засорения не выделены</td></tr>"

        # Frames table
        frames_rows_html = ""
        for f in report.frames:
            coords_txt = (
                f"X={f.camera_x:.2f}, Y={f.camera_y:.2f} (H={f.altitude_m:.1f}м)"
                if f.camera_x is not None and f.camera_y is not None and f.altitude_m is not None
                else "Нет XMP"
            )
            cls_txt = ", ".join(f"{esc(k)}: {v}" for k, v in sorted(f.by_class.items())) or "—"
            stg_txt = ", ".join(f"{esc(k)}: {v}" for k, v in sorted(f.by_stage.items())) or "—"
            frames_rows_html += (
                f"<tr>"
                f"<td><strong>{esc(f.file_name)}</strong></td>"
                f"<td>{esc(f.status)}</td>"
                f"<td>{'Да' if f.has_xmp else 'Нет'}</td>"
                f"<td>{esc(coords_txt)}</td>"
                f"<td><b>{f.detections_count}</b></td>"
                f"<td>{f.mean_confidence * 100.0:.1f}%</td>"
                f"<td>{cls_txt}</td>"
                f"<td>{stg_txt}</td>"
                f"</tr>"
            )
        if not frames_rows_html:
            frames_rows_html = "<tr><td colspan='8' class='muted'>Список кадров пуст</td></tr>"

        html_doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Отчет детекции GeoWeedStation — {esc(report.folder_name)}</title>
<style>
  :root {{
    --bg: #f8f9fa;
    --card: #ffffff;
    --border: #dee2e6;
    --text: #212529;
    --muted: #6c757d;
    --primary: #2b8a3e;
    --accent: #1971c2;
    --warn: #e67700;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .header {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 6px solid var(--primary);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
  }}
  .header h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
  .meta-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px 16px;
    font-size: 13px;
    color: var(--muted);
  }}
  .meta-grid b {{ color: var(--text); }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }}
  .kpi-title {{ font-size: 12px; color: var(--muted); text-transform: uppercase; }}
  .kpi-val {{ font-size: 24px; font-weight: 700; color: var(--primary); margin-top: 4px; }}
  .section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 20px;
  }}
  .section h2 {{ margin: 0 0 14px 0; font-size: 17px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ background: #f1f3f5; font-weight: 600; }}
  .muted {{ color: var(--muted); text-align: center; font-style: italic; }}
  .bar-bg {{ background: #e9ecef; height: 10px; border-radius: 5px; overflow: hidden; width: 100%; min-width: 80px; }}
  .bar-fill {{ background: var(--primary); height: 100%; }}
  .bar-stage {{ background: var(--accent); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-warn {{ background: #fff3bf; color: #b45309; }}
  .map-box {{ text-align: center; }}
  .map-box img {{ max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .section, .header, .kpi-card {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📊 Аналитический отчет по результатам детекции: {esc(report.folder_name)}</h1>
    <div class="meta-grid">
      <div>ID отчета: <b>{esc(report.report_id)}</b></div>
      <div>Дата формирования: <b>{esc(report.created_at[:19].replace('T', ' '))} UTC</b></div>
      <div>Каталог: <b>{esc(report.folder_path)}</b></div>
      <div>Модель детектора: <b>{esc(report.model_name)}</b></div>
      <div>Классификатор: <b>{esc(report.classifier_type.upper())}</b></div>
      <div>Пороги (Conf / IoU): <b>{report.confidence_threshold:.2f} / {report.nms_iou_threshold:.2f}</b></div>
      <div>Геопривязка: <b>{'Активна (' + esc(report.coordinate_system or 'XMP') + ')' if report.has_georef else 'Отсутствует'}</b></div>
      <div>Объем данных: <b>{total_mb:.2f} МБ ({report.total_files} файлов)</b></div>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">Кадров обработано</div>
      <div class="kpi-val">{report.processed_files} / {report.total_files}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Кадров с сорняками</div>
      <div class="kpi-val">{report.frames_with_detections}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Всего детекций</div>
      <div class="kpi-val">{report.total_detections}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Уникальных растений</div>
      <div class="kpi-val">{report.unique_weeds}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Средняя уверенность</div>
      <div class="kpi-val">{report.mean_confidence * 100.0:.1f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Площадь обследования</div>
      <div class="kpi-val">{report.surveyed_area_m2:.1f} м²</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Средняя плотность</div>
      <div class="kpi-val">{report.mean_density_per_m2:.2f} шт/м²</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Критическая зона</div>
      <div class="kpi-val">{report.critical_area_percent:.1f}%</div>
    </div>
  </div>

  <div class="two-col">
    <div class="section">
      <h2>🌿 Распределение по видам / классам</h2>
      <table>
        <thead><tr><th>Класс / таксон</th><th>Кол-во</th><th>Доля</th><th>Шкала</th></tr></thead>
        <tbody>{class_rows_html}</tbody>
      </table>
    </div>
    <div class="section">
      <h2>🌱 Распределение по фенологическим стадиям</h2>
      <table>
        <thead><tr><th>Стадия развития</th><th>Кол-во</th><th>Доля</th><th>Шкала</th></tr></thead>
        <tbody>{stage_rows_html}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>🔬 Матрица «Вид сорняка × Стадия развития»</h2>
    <table>
      <thead><tr><th>Вид / группа сорняка</th><th>Разбивка по стадиям роста</th></tr></thead>
      <tbody>{matrix_rows_html}</tbody>
    </table>
  </div>

  {map_section_html}

  <div class="section">
    <h2>🚜 Зоны предписаний для дифференцированной обработки ({report.prescription_zones_count})</h2>
    <table>
      <thead>
        <tr>
          <th>ID зоны</th>
          <th>Площадь</th>
          <th>Сорняков</th>
          <th>Плотность</th>
          <th>Доминирующий вид</th>
          <th>Рекомендация</th>
        </tr>
      </thead>
      <tbody>{zones_rows_html}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>📷 Покадровая ведомость снимков ({len(report.frames)})</h2>
    <table>
      <thead>
        <tr>
          <th>Файл</th>
          <th>Статус</th>
          <th>XMP</th>
          <th>Координаты камеры</th>
          <th>Детекций</th>
          <th>Ср. Conf</th>
          <th>Классы</th>
          <th>Стадии</th>
        </tr>
      </thead>
      <tbody>{frames_rows_html}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
        path.write_text(html_doc, encoding="utf-8")
        return path

    def export_zip_bundle(
        self,
        report: FolderDetectionReport,
        zip_path: Path,
        map_engine: MapEngine | None = None,
        png_options: PngExportOptions | None = None,
    ) -> Path:
        """Create a complete downloadable ZIP report archive containing HTML, JSON, CSVs, and GIS layers."""
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            map_png_bytes = self.render_map_png_bytes(map_engine, options=png_options)

            html_file = self.export_html(report, tmp_dir / "report.html", map_png_bytes=map_png_bytes)
            json_file = self.export_json(report, tmp_dir / "report_summary.json")
            frames_csv = self.export_frames_csv(report, tmp_dir / "frames_summary.csv")
            dets_csv = self.export_detections_csv(report, tmp_dir / "detections_table.csv")

            files_to_pack = [html_file, json_file, frames_csv, dets_csv]

            if map_engine is not None and not map_engine.is_locked() and map_engine.weed_map.points:
                geojson_file = tmp_dir / "weed_map.geojson"
                self._exporter.to_geojson(map_engine.filtered_weed_map(), geojson_file, include_zones=True)
                files_to_pack.append(geojson_file)

                if map_png_bytes is not None:
                    png_file = tmp_dir / "weed_map.png"
                    png_file.write_bytes(map_png_bytes)
                    files_to_pack.append(png_file)

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fpath in files_to_pack:
                    zf.write(fpath, arcname=fpath.name)

        return zip_path
