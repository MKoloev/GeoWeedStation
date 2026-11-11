"""Map export facilities: CSV tabular export, GeoJSON vector export, and PNG raster rendering."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geo_weed_station.core.map_engine.weed_map import WeedMap
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import (
    HeatmapGrid,
    PngExportOptions,
    PrescriptionZone,
    SummaryReport,
    WeedPoint,
)

# Class colors for vector markers on map (RGB)
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "weed_dicot": (230, 40, 40),      # Bright Red
    "weed_monocot": (40, 120, 240),   # Bright Blue
}
_PALETTE_COLORS: list[tuple[int, int, int]] = [
    (230, 40, 40),
    (40, 120, 240),
    (240, 140, 20),
    (155, 89, 182),
    (22, 160, 133),
    (211, 84, 0),
    (41, 128, 185),
    (192, 57, 43),
]
_DEFAULT_COLOR = (240, 140, 20)      # Orange

_ROUTE_COLOR = (25, 113, 194)         # Deep Sky Blue for path trajectory
_START_COLOR = (18, 184, 134)         # Emerald Green for Start line & marker
_END_COLOR = (201, 42, 42)            # Crimson Red for Finish line & marker


def _load_font(size: int = 12, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a Cyrillic-capable TrueType font with fallback to PIL default font."""
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    )
    for path_str in candidates:
        p = Path(path_str)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _extract_route_coords(wm: WeedMap) -> list[tuple[float, float]]:
    """Extract ordered flight trajectory coordinates (x, y) from footprints or weed points."""
    if wm.footprints:
        return [(fp.center_x, fp.center_y) for fp in wm.footprints]

    if not wm.points:
        return []

    # Group points by source_frame in order of appearance
    frame_groups: dict[str, list[tuple[float, float]]] = {}
    for pt in wm.points:
        key = pt.source_frame or ""
        frame_groups.setdefault(key, []).append((pt.x, pt.y))

    if len(frame_groups) >= 2:
        route: list[tuple[float, float]] = []
        for coords in frame_groups.values():
            avg_x = sum(c[0] for c in coords) / len(coords)
            avg_y = sum(c[1] for c in coords) / len(coords)
            route.append((avg_x, avg_y))
        return route

    # Fallback when all points belong to a single frame: connect points in order
    return [(pt.x, pt.y) for pt in wm.points]


class MapExporter:
    """Exports WeedMap spatial point layers and raster heatmaps to disk."""

    def to_csv(self, wm: WeedMap | Sequence[WeedPoint], path: Path) -> None:
        """Export all weed detections to a standardized CSV table.
        
        Columns: weed_id, class_name, confidence, coord_x, coord_y, coord_system, source_frame
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = wm.points if isinstance(wm, WeedMap) else wm

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "weed_id",
                    "class_name",
                    "confidence",
                    "coord_x",
                    "coord_y",
                    "coord_system",
                    "source_frame",
                ]
            )
            for pt in pts:
                if pt.coord_system == CoordinateSystem.WGS84 and pt.lon is not None and pt.lat is not None:
                    out_x = f"{pt.lon:.8f}"
                    out_y = f"{pt.lat:.8f}"
                else:
                    out_x = f"{pt.x:.6f}"
                    out_y = f"{pt.y:.6f}"

                writer.writerow(
                    [
                        pt.weed_id,
                        pt.class_name,
                        f"{pt.confidence:.4f}",
                        out_x,
                        out_y,
                        pt.coord_system.value,
                        pt.source_frame,
                    ]
                )

    def to_geojson(
        self,
        wm: WeedMap | Sequence[WeedPoint],
        path: Path,
        include_zones: bool = True,
    ) -> None:
        """Export weed detections and prescription zones as a valid RFC 7946 GeoJSON FeatureCollection."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = wm.points if isinstance(wm, WeedMap) else wm
        cs = wm.coord_system if isinstance(wm, WeedMap) else (pts[0].coord_system if pts else CoordinateSystem.LOCAL_ARUCO)

        features = []

        # 1. Export Prescription Zones if available and requested
        if include_zones and isinstance(wm, WeedMap) and hasattr(wm, "prescription_zones") and wm.prescription_zones:
            for zone in wm.prescription_zones:
                if cs == CoordinateSystem.WGS84 and zone.wgs84_polygon:
                    poly_coords = zone.wgs84_polygon
                else:
                    poly_coords = zone.polygon

                ring = [
                    [round(c[0], 8 if cs == CoordinateSystem.WGS84 else 6), round(c[1], 8 if cs == CoordinateSystem.WGS84 else 6)]
                    for c in poly_coords
                ]
                if ring and ring[0] != ring[-1]:
                    ring.append(ring[0])

                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring],
                    },
                    "properties": {
                        "feature_type": "prescription_zone",
                        "zone_id": zone.zone_id,
                        "area_m2": round(zone.area_m2, 2),
                        "mean_density_per_m2": round(zone.mean_density_per_m2, 2),
                        "weed_count": zone.weed_count,
                        "dominant_class": zone.dominant_class,
                        "treatment_recommendation": zone.treatment_recommendation,
                    },
                }
                features.append(feat)

        # 2. Export Weed Points
        for pt in pts:
            if pt.coord_system == CoordinateSystem.WGS84 and pt.lon is not None and pt.lat is not None:
                coords = [round(pt.lon, 8), round(pt.lat, 8)]
            else:
                coords = [round(pt.x, 6), round(pt.y, 6)]

            props = {
                "weed_id": pt.weed_id,
                "class_name": pt.class_name,
                "confidence": round(pt.confidence, 4),
                "coord_system": pt.coord_system.value,
                "source_frame": pt.source_frame,
            }
            if pt.best_frame_id is not None:
                props["best_frame_id"] = pt.best_frame_id
            if pt.duplicate_count > 1:
                props["duplicate_count"] = pt.duplicate_count
            if pt.stage_name is not None:
                props["stage_name"] = pt.stage_name
            if pt.class_prob is not None:
                props["class_prob"] = round(pt.class_prob, 4)

            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coords,
                },
                "properties": props,
            }
            features.append(feat)

        crs_name = (
            "local_aruco_meters"
            if cs == CoordinateSystem.LOCAL_ARUCO
            else "urn:ogc:def:crs:OGC:1.3:CRS84"
        )

        geojson_doc = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs_name}},
            "features": features,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, indent=2)

    export_csv = to_csv
    export_geojson = to_geojson

    def to_summary_json(self, report: SummaryReport, path: Path) -> None:
        """Export agronomic summary report and prescription zones to structured JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

    def to_summary_csv(self, report: SummaryReport, path: Path) -> None:
        """Export agronomic summary metrics and prescription zones table to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_detections", report.total_detections])
            writer.writerow(["unique_weeds", report.unique_weeds])
            writer.writerow(["deduplication_ratio", report.deduplication_ratio])
            writer.writerow(["surveyed_area_m2", report.surveyed_area_m2])
            writer.writerow(["mean_density_per_m2", report.mean_density_per_m2])
            writer.writerow(["critical_area_percent", report.critical_area_percent])
            writer.writerow(["prescription_zones_count", report.prescription_zones_count])
            writer.writerow(["frames_count", report.frames_count])
            writer.writerow([])
            writer.writerow(
                [
                    "zone_id",
                    "area_m2",
                    "mean_density_per_m2",
                    "weed_count",
                    "dominant_class",
                    "treatment_recommendation",
                ]
            )
            for z in report.zones:
                writer.writerow(
                    [
                        z.zone_id,
                        z.area_m2,
                        z.mean_density_per_m2,
                        z.weed_count,
                        z.dominant_class,
                        z.treatment_recommendation,
                    ]
                )

    export_summary_json = to_summary_json
    export_summary_csv = to_summary_csv

    def export_png(
        self,
        points: WeedMap | Sequence[WeedPoint],
        output_path: Path,
        grid: HeatmapGrid | None = None,
        options: PngExportOptions | None = None,
    ) -> None:
        """Alias for to_png supporting points list or WeedMap directly."""
        if isinstance(points, WeedMap):
            wm = points
        else:
            wm = WeedMap()
            wm.add_points(list(points))
        return self.to_png(wm=wm, grid=grid, path=output_path, options=options or PngExportOptions())

    @staticmethod
    def _draw_perpendicular_gate_line(
        draw: ImageDraw.ImageDraw,
        center_px: tuple[int, int],
        dir_vec: tuple[float, float],
        color: tuple[int, int, int],
        half_len: float = 16.0,
    ) -> None:
        """Draw a perpendicular gate line (start or finish line) across the flight path."""
        cx, cy = center_px
        dx, dy = dir_vec
        norm = math.hypot(dx, dy)
        if norm < 1e-5:
            nx, ny = 0.0, -1.0
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / norm, dy / norm
            nx, ny = -uy, ux

        x1 = int(round(cx - nx * half_len))
        y1 = int(round(cy - ny * half_len))
        x2 = int(round(cx + nx * half_len))
        y2 = int(round(cy + ny * half_len))

        # White outer casing for contrast
        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 235), width=6)
        # Main gate line
        draw.line([(x1, y1), (x2, y2)], fill=(*color, 255), width=4)

        # Small perpendicular end-caps (ticks) on the gate line
        tick = 5.0
        for ex, ey in ((x1, y1), (x2, y2)):
            tx1 = int(round(ex - ux * tick))
            ty1 = int(round(ey - uy * tick))
            tx2 = int(round(ex + ux * tick))
            ty2 = int(round(ey + uy * tick))
            draw.line([(tx1, ty1), (tx2, ty2)], fill=(*color, 255), width=3)

    @staticmethod
    def _draw_route_overlay(
        draw: ImageDraw.ImageDraw,
        route_px: list[tuple[int, int]],
        font_bold: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> None:
        """Draw flight path trajectory, directional arrows, start line/marker, and finish line/marker."""
        if not route_px:
            return

        # 1. Connecting path lines and directional arrows
        if len(route_px) >= 2:
            # White halo under path line
            draw.line(route_px, fill=(255, 255, 255, 210), width=6, joint="curve")
            # Main trajectory line
            draw.line(route_px, fill=(*_ROUTE_COLOR, 240), width=3, joint="curve")

            # Draw direction arrowheads on segments
            for i in range(len(route_px) - 1):
                x1, y1 = route_px[i]
                x2, y2 = route_px[i + 1]
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                seg_len = math.hypot(dx, dy)
                if seg_len < 24.0:
                    continue
                ux, uy = dx / seg_len, dy / seg_len
                nx, ny = -uy, ux

                # Place arrowhead at 55% of segment
                ax = x1 + dx * 0.55
                ay = y1 + dy * 0.55
                arrow_len = 9.0
                arrow_w = 5.0
                tip = (int(round(ax + ux * (arrow_len * 0.5))), int(round(ay + uy * (arrow_len * 0.5))))
                left = (
                    int(round(ax - ux * (arrow_len * 0.5) + nx * arrow_w)),
                    int(round(ay - uy * (arrow_len * 0.5) + ny * arrow_w)),
                )
                right = (
                    int(round(ax - ux * (arrow_len * 0.5) - nx * arrow_w)),
                    int(round(ay - uy * (arrow_len * 0.5) - ny * arrow_w)),
                )
                draw.polygon([tip, left, right], fill=(*_ROUTE_COLOR, 255), outline=(255, 255, 255, 240))

            # Intermediate waypoint stations
            for wx, wy in route_px[1:-1]:
                r = 4
                draw.ellipse(
                    [wx - r, wy - r, wx + r, wy + r],
                    fill=(*_ROUTE_COLOR, 255),
                    outline=(255, 255, 255, 255),
                    width=1,
                )

        # 2. Start Gate Line & Marker
        start_pt = route_px[0]
        if len(route_px) >= 2:
            start_dir = (float(route_px[1][0] - start_pt[0]), float(route_px[1][1] - start_pt[1]))
        else:
            start_dir = (1.0, 0.0)

        MapExporter._draw_perpendicular_gate_line(draw, start_pt, start_dir, _START_COLOR, half_len=16.0)
        sx, sy = start_pt
        sr = 7
        draw.ellipse(
            [sx - sr, sy - sr, sx + sr, sy + sr],
            fill=(*_START_COLOR, 255),
            outline=(255, 255, 255, 255),
            width=2,
        )
        # Start label pill
        draw.rectangle(
            [sx + 10, sy - 18, sx + 74, sy - 1],
            fill=(18, 184, 134, 235),
            outline=(255, 255, 255, 240),
            width=1,
        )
        draw.text((sx + 15, sy - 16), "СТАРТ", fill=(255, 255, 255, 255), font=font_bold)

        # 3. Finish Gate Line & Marker (when at least 2 route vertices exist)
        if len(route_px) >= 2:
            end_pt = route_px[-1]
            prev_pt = route_px[-2]
            end_dir = (float(end_pt[0] - prev_pt[0]), float(end_pt[1] - prev_pt[1]))
            if math.hypot(end_dir[0], end_dir[1]) < 1e-3 and len(route_px) >= 3:
                end_dir = (float(end_pt[0] - route_px[0][0]), float(end_pt[1] - route_px[0][1]))

            MapExporter._draw_perpendicular_gate_line(draw, end_pt, end_dir, _END_COLOR, half_len=16.0)
            ex, ey = end_pt
            er = 7
            # Draw square/diamond finish marker
            draw.rectangle(
                [ex - er, ey - er, ex + er, ey + er],
                fill=(*_END_COLOR, 255),
                outline=(255, 255, 255, 255),
                width=2,
            )
            # Offset finish label so it doesn't collide with start label if start == end
            lbl_y_offset = 4 if math.hypot(ex - sx, ey - sy) < 30 else -18
            draw.rectangle(
                [ex + 10, ey + lbl_y_offset, ex + 82, ey + lbl_y_offset + 17],
                fill=(201, 42, 42, 235),
                outline=(255, 255, 255, 240),
                width=1,
            )
            draw.text((ex + 15, ey + lbl_y_offset + 2), "ФИНИШ", fill=(255, 255, 255, 255), font=font_bold)

    def to_png(
        self,
        wm: WeedMap,
        grid: HeatmapGrid | None,
        path: Path,
        options: PngExportOptions,
    ) -> None:
        """Render and save complete map composition as a high-resolution PNG image."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        canvas_w, canvas_h = 1200, 900
        margin = 60
        plot_w = canvas_w - 2 * margin
        plot_h = canvas_h - 2 * margin

        bounds = wm.bounds()
        b_w = max(0.01, bounds.width)
        b_h = max(0.01, bounds.height)

        font_reg = _load_font(size=12, bold=False)
        font_bold = _load_font(size=12, bold=True)

        # Coordinate to pixel transform
        def to_px(x: float, y: float) -> tuple[int, int]:
            px = int(margin + ((x - bounds.min_x) / b_w) * plot_w)
            py = int(margin + ((bounds.max_y - y) / b_h) * plot_h)
            return px, py

        # 1. Base canvas creation
        if options.background == "transparent":
            img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        elif options.background == "orthophoto" and options.orthophoto_path and options.orthophoto_path.is_file():
            ortho = Image.open(options.orthophoto_path).convert("RGBA")
            img = ortho.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        else:
            img = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

        # 2. Render Heatmap layer if available
        if grid is not None and grid.matrix is not None and grid.matrix.size > 0:
            mat = grid.matrix
            max_val = float(np.max(mat))
            if max_val > 0.0:
                rows, cols = mat.shape
                norm_mat = mat / max_val  # 0..1
                # Create RGBA colormap array: green(0.0) -> yellow(0.5) -> red(1.0)
                heatmap_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
                for r in range(rows):
                    for c in range(cols):
                        val = norm_mat[r, c]
                        if val > 0.0:
                            # Gradient: val in [0..1]
                            red = int(min(255, val * 2.0 * 255))
                            green = int(min(255, (1.0 - max(0.0, val - 0.5) * 2.0) * 255))
                            alpha = int(40 + val * 160)  # semi-transparent
                            heatmap_rgba[rows - 1 - r, c] = [red, green, 0, alpha]

                hm_img = Image.fromarray(heatmap_rgba, mode="RGBA")
                hm_resized = hm_img.resize((plot_w, plot_h), Image.Resampling.BILINEAR)
                img.paste(hm_resized, (margin, margin), hm_resized)

        draw = ImageDraw.Draw(img)

        # 3. Draw coordinate grid
        if options.show_grid:
            grid_color = (200, 200, 200, 160) if options.background != "transparent" else (160, 160, 160, 180)
            draw.rectangle([margin, margin, margin + plot_w, margin + plot_h], outline=(80, 80, 80, 255), width=2)
            for i in range(1, 5):
                gx = margin + int(plot_w * i / 5)
                gy = margin + int(plot_h * i / 5)
                draw.line([(gx, margin), (gx, margin + plot_h)], fill=grid_color, width=1)
                draw.line([(margin, gy), (margin + plot_w, gy)], fill=grid_color, width=1)

        # 4. Draw flight route (start line, trajectory path line, finish line)
        route_coords = _extract_route_coords(wm) if getattr(options, "show_route", True) else []
        route_px = [to_px(rx, ry) for rx, ry in route_coords]
        if route_px:
            self._draw_route_overlay(draw, route_px, font_bold=font_bold)

        # Build dynamic class color mapping for all classes present in wm.points
        active_class_colors: dict[str, tuple[int, int, int]] = {}
        for pt in wm.points:
            if pt.class_name not in active_class_colors:
                if pt.class_name in _CLASS_COLORS:
                    active_class_colors[pt.class_name] = _CLASS_COLORS[pt.class_name]
                else:
                    c_idx = len(active_class_colors) % len(_PALETTE_COLORS)
                    active_class_colors[pt.class_name] = _PALETTE_COLORS[c_idx]
        if not active_class_colors:
            active_class_colors = dict(_CLASS_COLORS)

        # 5. Draw weed point markers
        marker_radius = 5
        for pt in wm.points:
            px, py = to_px(pt.x, pt.y)
            color = active_class_colors.get(pt.class_name, _DEFAULT_COLOR)
            draw.ellipse(
                [px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius],
                fill=(*color, 240),
                outline=(20, 20, 20, 255),
                width=1,
            )

        # 6. Draw Legend (classes + route start/finish/path indicators)
        if options.show_legend:
            legend_x, legend_y = margin + 15, margin + 15
            route_legend_rows = 3 if route_px else 0
            total_rows = len(active_class_colors) + route_legend_rows
            legend_w = 270
            legend_h = 36 + total_rows * 24 + (10 if route_legend_rows else 0)
            draw.rectangle(
                [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
                fill=(255, 255, 255, 230),
                outline=(100, 100, 100, 255),
                width=1,
            )
            draw.text((legend_x + 10, legend_y + 8), "Легенда карты", fill=(20, 20, 20, 255), font=font_bold)

            curr_y = legend_y + 32
            for cls_name, col in active_class_colors.items():
                draw.ellipse(
                    [legend_x + 15, curr_y + 2, legend_x + 27, curr_y + 14],
                    fill=(*col, 255),
                    outline=(0, 0, 0, 255),
                )
                count = sum(1 for p in wm.points if p.class_name == cls_name)
                draw.text(
                    (legend_x + 35, curr_y),
                    f"{cls_name} ({count})",
                    fill=(30, 30, 30, 255),
                    font=font_reg,
                )
                curr_y += 24

            if route_px:
                curr_y += 4
                draw.line(
                    [(legend_x + 10, curr_y - 2), (legend_x + legend_w - 10, curr_y - 2)],
                    fill=(200, 200, 200, 255),
                    width=1,
                )
                # Path line item
                draw.line(
                    [(legend_x + 12, curr_y + 8), (legend_x + 30, curr_y + 8)],
                    fill=(*_ROUTE_COLOR, 255),
                    width=3,
                )
                draw.text((legend_x + 35, curr_y + 1), "Линия пути (маршрут)", fill=(30, 30, 30, 255), font=font_reg)
                curr_y += 24

                # Start line & marker item
                draw.line(
                    [(legend_x + 14, curr_y + 2), (legend_x + 14, curr_y + 14)],
                    fill=(*_START_COLOR, 255),
                    width=3,
                )
                draw.ellipse(
                    [legend_x + 18, curr_y + 4, legend_x + 26, curr_y + 12],
                    fill=(*_START_COLOR, 255),
                    outline=(0, 0, 0, 255),
                )
                draw.text((legend_x + 35, curr_y + 1), "Начало пути (СТАРТ)", fill=(30, 30, 30, 255), font=font_reg)
                curr_y += 24

                # Finish line & marker item
                draw.line(
                    [(legend_x + 14, curr_y + 2), (legend_x + 14, curr_y + 14)],
                    fill=(*_END_COLOR, 255),
                    width=3,
                )
                draw.rectangle(
                    [legend_x + 18, curr_y + 4, legend_x + 26, curr_y + 12],
                    fill=(*_END_COLOR, 255),
                    outline=(0, 0, 0, 255),
                )
                draw.text((legend_x + 35, curr_y + 1), "Конец пути (ФИНИШ)", fill=(30, 30, 30, 255), font=font_reg)

        # 7. Draw Scalebar
        if options.show_scalebar:
            sb_x, sb_y = margin + 20, margin + plot_h - 35
            scale_len_px = 100
            meters_per_px = b_w / plot_w
            scale_m = round(scale_len_px * meters_per_px, 1)

            draw.rectangle(
                [sb_x - 5, sb_y - 18, sb_x + scale_len_px + 65, sb_y + 10],
                fill=(255, 255, 255, 210),
                outline=(120, 120, 120, 255),
            )
            draw.line([(sb_x, sb_y), (sb_x + scale_len_px, sb_y)], fill=(0, 0, 0, 255), width=3)
            draw.line([(sb_x, sb_y - 4), (sb_x, sb_y + 4)], fill=(0, 0, 0, 255), width=2)
            draw.line([(sb_x + scale_len_px, sb_y - 4), (sb_x + scale_len_px, sb_y + 4)], fill=(0, 0, 0, 255), width=2)
            draw.text((sb_x + scale_len_px + 8, sb_y - 8), f"{scale_m} m", fill=(10, 10, 10, 255), font=font_reg)

        img.save(path, "PNG", dpi=(options.dpi, options.dpi))
