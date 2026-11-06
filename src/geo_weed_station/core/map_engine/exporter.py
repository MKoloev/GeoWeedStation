"""Map export facilities: CSV tabular export, GeoJSON vector export, and PNG raster rendering."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geo_weed_station.core.map_engine.weed_map import WeedMap
from geo_weed_station.schemas.entities import CoordinateSystem
from geo_weed_station.schemas.map_models import HeatmapGrid, PngExportOptions

# Class colors for vector markers on map (RGB)
_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "weed_dicot": (230, 40, 40),      # Bright Red
    "weed_monocot": (40, 120, 240),   # Bright Blue
}
_DEFAULT_COLOR = (240, 140, 20)      # Orange


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
                writer.writerow(
                    [
                        pt.weed_id,
                        pt.class_name,
                        f"{pt.confidence:.4f}",
                        f"{pt.x:.6f}",
                        f"{pt.y:.6f}",
                        pt.coord_system.value,
                        pt.source_frame,
                    ]
                )

    def to_geojson(self, wm: WeedMap | Sequence[WeedPoint], path: Path) -> None:
        """Export weed detections as a valid RFC 7946 GeoJSON FeatureCollection."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = wm.points if isinstance(wm, WeedMap) else wm
        cs = wm.coord_system if isinstance(wm, WeedMap) else (pts[0].coord_system if pts else CoordinateSystem.LOCAL_ARUCO)

        features = []
        for pt in pts:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(pt.x, 6), round(pt.y, 6)],
                },
                "properties": {
                    "weed_id": pt.weed_id,
                    "class_name": pt.class_name,
                    "confidence": round(pt.confidence, 4),
                    "coord_system": pt.coord_system.value,
                    "source_frame": pt.source_frame,
                },
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

        # Coordinate to pixel transform
        def to_px(x: float, y: float) -> Tuple[int, int]:
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

        # 4. Draw weed point markers
        marker_radius = 5
        for pt in wm.points:
                    px, py = to_px(pt.x, pt.y)
                    color = _CLASS_COLORS.get(pt.class_name, _DEFAULT_COLOR)
                    # Filled circle with dark outline
                    draw.ellipse(
                        [px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius],
                        fill=(*color, 240),
                        outline=(20, 20, 20, 255),
                        width=1,
                    )

        # 5. Draw Legend
        if options.show_legend:
            legend_x, legend_y = margin + 15, margin + 15
            legend_w, legend_h = 220, 30 + len(_CLASS_COLORS) * 25
            draw.rectangle(
                [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
                fill=(255, 255, 255, 220),
                outline=(100, 100, 100, 255),
                width=1,
            )
            draw.text((legend_x + 10, legend_y + 8), "Weed Classes", fill=(20, 20, 20, 255))
            for idx, (cls_name, col) in enumerate(_CLASS_COLORS.items()):
                item_y = legend_y + 32 + idx * 24
                draw.ellipse(
                    [legend_x + 15, item_y, legend_x + 27, item_y + 12],
                    fill=(*col, 255),
                    outline=(0, 0, 0, 255),
                )
                count = sum(1 for p in wm.points if p.class_name == cls_name)
                draw.text((legend_x + 35, item_y), f"{cls_name} ({count})", fill=(30, 30, 30, 255))

        # 6. Draw Scalebar
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
            draw.text((sb_x + scale_len_px + 8, sb_y - 8), f"{scale_m} m", fill=(10, 10, 10, 255))

        img.save(path, "PNG", dpi=(options.dpi, options.dpi))
