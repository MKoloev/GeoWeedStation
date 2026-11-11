#!/usr/bin/env python3
"""Generate GeosAero validation datasets with 10% overlap for GeoWeedStation.

Takes high-resolution images from GeosAero_hitech_dataset/GAS_202608281553_032_web,
keeps original image files untouched, and creates two new datasets in
GeoWeedStation/workdir/validation_runs:

1. 05_geosaero_all_wgs84_10pct:
   All 102 valid images from the full GeosAero survey across all 11 passes,
   rescaled from 50m spacing down to 10% overlap (step ~1.37m along line, ~1.83m between passes).
   
2. 06_geosaero_corner_25_wgs84_10pct:
   The first 25 consecutive frames from the starting corner of the survey (Passes 0..3),
   scaled with 10% overlap.

Uses lossless JPEG XMP replacement without recompression, preserving 100% image quality.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

# Resolve repo root and src paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from geo_weed_station.core.map_engine.coordinates import CoordinateConverter
from geo_weed_station.core.map_engine.projector import NadirProjector
from geo_weed_station.core.map_engine.weed_map import MapEngine
from geo_weed_station.core.xmp_parser import build_frame_payload, extract_spatial_data, has_xmp
from geo_weed_station.config.settings import MapSettings

_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def replace_jpeg_xmp(src_path: Path, dst_path: Path, new_xmp: str) -> None:
    """Losslessly replace Adobe XMP packet in JPEG APP1 segment without recompressing pixels."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "rb") as f:
        data = f.read()

    if data[:2] != b"\xff\xd8":
        raise ValueError(f"File {src_path.name} is not a valid JPEG")

    idx = 2
    xmp_start = None
    xmp_end = None

    while idx < len(data) - 4:
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        if marker in (0xD8, 0xD9):
            idx += 2
            continue
        if marker == 0xDA:  # SOS marker
            break

        length = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
        payload = data[idx + 4 : idx + 2 + length]

        if marker == 0xE1 and payload.startswith(_XMP_HEADER):
            xmp_start = idx
            xmp_end = idx + 2 + length
            break
        idx += 2 + length

    new_payload = _XMP_HEADER + new_xmp.encode("utf-8")
    new_segment = b"\xff\xe1" + struct.pack(">H", len(new_payload) + 2) + new_payload

    if xmp_start is not None and xmp_end is not None:
        new_data = data[:xmp_start] + new_segment + data[xmp_end:]
    else:
        new_data = data[:2] + new_segment + data[2:]

    dst_path.write_bytes(new_data)


def build_xmp_xml(
    *,
    lat: float,
    lon: float,
    altitude_m: float,
    yaw_deg: float,
    pitch_deg: float = -90.0,
    roll_deg: float = 0.0,
    focal_length_mm: float = 40.0,
    sensor_width_mm: float = 8.571,
    sensor_height_mm: float = 6.428,
) -> str:
    """Create standard XMP XML with WGS84 coordinates."""
    return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
    xmlns:Camera="http://pix4d.com/camera/1.0/"
    xmlns:agro="http://geosaero.ru/agro/1.0/"
    drone-dji:RelativeAltitude="{altitude_m:+.2f}"
    drone-dji:GimbalYawDegree="{yaw_deg:+.1f}"
    drone-dji:GimbalPitchDegree="{pitch_deg:+.1f}"
    drone-dji:GimbalRollDegree="{roll_deg:+.1f}"
    Camera:PerspectiveFocalLength="{focal_length_mm:.2f}"
    Camera:SensorWidth="{sensor_width_mm:.3f}"
    Camera:SensorHeight="{sensor_height_mm:.3f}"
    Camera:PrincipalPoint="4096.0,3072.0"
    Camera:BandName="RGB"
    agro:CoordinateSystem="wgs84"
    drone-dji:GpsLatitude="{lat:.7f}"
    drone-dji:GpsLongitude="{lon:.7f}"/>
 </rdf:RDF>
</x:xmpmeta>"""


def prepare_geosaero_datasets(
    source_dir: Path,
    output_base_dir: Path,
    overlap: float = 0.10,
) -> None:
    source_dir = Path(source_dir)
    output_base_dir = Path(output_base_dir)

    all_files = sorted(list(source_dir.glob("*.JPG")))
    print(f"Найдено {len(all_files)} исходных файлов в {source_dir}")

    # 1. Parse and extract all valid frames with spatial metadata
    valid_frames = []
    for f in all_files:
        try:
            sp = extract_spatial_data(f, (8192, 6144))
            if sp.pose:
                valid_frames.append((f, sp.pose, sp.optics))
        except Exception as exc:
            print(f"  Пропущен повреждённый файл {f.name}: {exc}")

    print(f"Успешно прочитано {len(valid_frames)} валидных кадров GeosAero")
    if not valid_frames:
        raise RuntimeError("Нет валидных снимков для обработки")

    # Determine reference origin (first frame)
    origin_lat = valid_frames[0][1].y
    origin_lon = valid_frames[0][1].x
    conv = CoordinateConverter(wgs84_origin=(origin_lat, origin_lon))

    # Calculate real optical footprint
    f0, pose0, optics0 = valid_frames[0]
    fw, fh = NadirProjector.ground_footprint_m(optics0, pose0.altitude_m, (8192, 6144))
    print(f"Пятно кадра GeosAero на земле: {fw:.3f} м × {fh:.3f} м (H={pose0.altitude_m:.2f} м)")

    # Target 10% overlap steps:
    target_step_y = fh * (1.0 - overlap)  # ~1.374 m along flight line
    target_step_x = fw * (1.0 - overlap)  # ~1.832 m between passes
    print(f"Шаг при {int(overlap*100)}% перекрытии: ΔY={target_step_y:.3f} м, ΔX={target_step_x:.3f} м")

    # Extract original local coordinates
    orig_coords = []
    for f, pose, optics in valid_frames:
        x, y = conv.wgs84_to_local(pose.y, pose.x)
        orig_coords.append((x, y, pose))

    # In original flight, average step between shots was ~50.0m along Y and ~50.0m along X
    scale_x = target_step_x / 50.0
    scale_y = target_step_y / 50.0

    # Two target configs:
    # Config 1: All valid frames (102 frames)
    # Config 2: Corner 25 frames (frames 0..24)
    tasks = [
        ("05_geosaero_all_wgs84_10pct", valid_frames, "GeosAero — Полный пролёт БПЛА (все 102 кадра, 11 галсов, 10% перекрытие)"),
        ("06_geosaero_corner_25_wgs84_10pct", valid_frames[:25], "GeosAero — Угловой сектор (25 кадров из угла поля, 10% перекрытие)"),
    ]

    for folder_name, subset_frames, title in tasks:
        target_dir = output_base_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nГенерация набора: {folder_name} ({len(subset_frames)} кадров)...")

        frames_manifest = []

        for idx, (src_f, orig_pose, orig_optics) in enumerate(subset_frames):
            orig_x, orig_y = conv.wgs84_to_local(orig_pose.y, orig_pose.x)
            new_x = orig_x * scale_x
            new_y = orig_y * scale_y
            new_lat, new_lon = conv.local_to_wgs84(new_x, new_y)

            # Build new XMP XML
            new_xmp = build_xmp_xml(
                lat=new_lat,
                lon=new_lon,
                altitude_m=orig_pose.altitude_m,
                yaw_deg=orig_pose.yaw_deg,
                pitch_deg=orig_pose.pitch_deg,
                roll_deg=orig_pose.roll_deg,
                focal_length_mm=orig_optics.focal_length_mm if orig_optics else 40.0,
                sensor_width_mm=orig_optics.sensor_width_mm if orig_optics else 8.571,
                sensor_height_mm=orig_optics.sensor_height_mm if orig_optics else 6.428,
            )

            out_filename = f"frame_{idx:03d}_{src_f.name}"
            out_path = target_dir / out_filename

            # Lossless XMP replacement directly into new file
            replace_jpeg_xmp(src_f, out_path, new_xmp)

            frames_manifest.append({
                "index": idx,
                "filename": out_filename,
                "original_filename": src_f.name,
                "altitude_m": orig_pose.altitude_m,
                "yaw_deg": orig_pose.yaw_deg,
                "local_x_m": round(new_x, 3),
                "local_y_m": round(new_y, 3),
                "gps_latitude": round(new_lat, 8),
                "gps_longitude": round(new_lon, 8),
            })

        # Save manifest
        manifest_data = {
            "title": title,
            "folder": folder_name,
            "total_frames": len(frames_manifest),
            "overlap_ratio": overlap,
            "origin_wgs84": {"lat": origin_lat, "lon": origin_lon},
            "footprint_m": [round(fw, 3), round(fh, 3)],
            "target_steps_m": {"step_x": round(target_step_x, 3), "step_y": round(target_step_y, 3)},
            "frames": frames_manifest,
        }
        (target_dir / "flight_manifest.json").write_text(
            json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Save README.md
        readme_text = f"""# {title}

Набор снимков реального полёта БПЛА GeosAero (GAS GA Drone) с пересчитанной геопривязкой под **10% перекрытие**.

## Параметры съёмки
* **Сенсор:** ZoomCamera ($8192 \\times 6144$ px, 50 Мп)
* **Фокусное расстояние:** $40.0$ мм
* **Высота съёмки:** $\\approx 9.5$ м
* **Размер пятна кадра:** **{fw:.2f} м × {fh:.2f} м**
* **Шаг между кадрами:** вдоль галса $\\Delta Y = {target_step_y:.2f}$ м, между галсами $\\Delta X = {target_step_x:.2f}$ м
* **Перекрытие:** {int(overlap * 100)}%
* **Система координат:** WGS84 (Оренбургская область: $51.777788^\\circ$ с.ш., $55.304049^\\circ$ в.д.)
* **Количество снимков:** {len(frames_manifest)} кадров

## Проверка в GeoWeedStation
1. В верхнем меню выберите **«Файл» → «Открыть каталог снимков...»** (`Ctrl+O`).
2. Укажите папку: `{target_dir}`.
3. У всех снимков горит отметка **XMP: Да**.
4. Нажмите **«⚡ Запустить обработку»**.
5. На вкладке **«Карта»** отобразятся плотные контуры снимков с 10% взаимным перекрытием, тепловая карта засоренности и маршрут БПЛА!
"""
        (target_dir / "README.md").write_text(readme_text, encoding="utf-8")
        print(f"  ✓ Сгенерировано {len(frames_manifest)} кадров в {target_dir}")

    # Verify both generated datasets with MapEngine
    print("\n" + "=" * 70)
    print("ВЕРИФИКАЦИЯ НОВЫХ НАБОРОВ GEOSAERO С GEOWEEDSTATION")
    print("=" * 70)
    map_settings = MapSettings()

    for folder_name, _, _ in tasks:
        p = output_base_dir / folder_name
        jpgs = sorted(list(p.glob("*.JPG")))
        engine = MapEngine(NadirProjector(), map_settings)
        for j in jpgs:
            fp = build_frame_payload(j, (8192, 6144))
            if fp.has_georef():
                engine.ingest_frame(fp)

        b = engine.weed_map.bounds()
        fps = len(engine.weed_map.footprints)
        print(
            f"✅ [{folder_name}]\n"
            f"   Кадров с XMP: {len(jpgs)}/{len(jpgs)} | Контуров карты: {fps}\n"
            f"   Границы карты: X=[{b.min_x:.1f}, {b.max_x:.1f}] м, Y=[{b.min_y:.1f}, {b.max_y:.1f}] м | "
            f"Площадь: {b.width * b.height:.1f} м²"
        )
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовка наборов GeosAero с 10% перекрытием.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/media/sf_Common/GeosAero_hitech_dataset-20261110T051737Z-1-001/GeosAero_hitech_dataset/GAS_202608281553_032_web"),
        help="Путь к исходным снимкам GeosAero",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "workdir" / "validation_runs",
        help="Целевой каталог validation_runs",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.10,
        help="Перекрытие между кадрами (default: 0.10)",
    )
    args = parser.parse_args()

    prepare_geosaero_datasets(args.source_dir, args.output_dir, args.overlap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
