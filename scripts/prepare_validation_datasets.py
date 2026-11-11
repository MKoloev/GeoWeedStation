#!/usr/bin/env python3
"""Prepare geotagged validation datasets for GeoWeedStation with TRUE physical scale and GSD.

Takes raw PNG benchmark images from validation_benchmark, computes realistic
UAV flight geometry adhering strictly to each dataset's passport GSD and camera optics,
applies 10% spatial overlap between adjacent frames, embeds standard XMP spatial/optical
metadata into JPEG APP1 markers, and creates 4 ready-to-test dataset folders for GeoWeedStation:

1. 01_phenobench_aruco_5x5:
   PhenoBench 25 samples, GSD 1.0 mm/px -> footprint 1.024 x 1.024 m, local_aruco, 5x5 grid.
2. 02_uavweedseg_aruco_linear:
   UAVWeedSegmentation 25 samples, GSD 1.0 mm/px -> footprint 0.256 x 0.256 m, local_aruco, linear transect.
3. 03_weedmap_wgs84_5x5:
   WeedMap RedEdge 25 samples, GSD 2.2 mm/px -> footprint 1.056 x 0.792 m, WGS84, 5x5 grid.
4. 04_weedsgalore_wgs84_linear:
   WeedsGalore 25 samples, GSD 2.5 mm/px -> footprint 1.500 x 1.500 m, WGS84, linear transect.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

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
from geo_weed_station.schemas.entities import CoordinateSystem

_XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    folder_name: str
    source_subpath: str
    coord_system: Literal["local_aruco", "wgs84"]
    pattern: Literal["grid_5x5", "linear_25"]
    gsd_m_per_px: float       # Exact ground sampling distance from dataset passport
    altitude_m: float         # Real flight altitude (meters)
    focal_length_mm: float    # Real camera focal length (mm)
    sensor_name: str          # Physical UAV sensor / camera
    recommended_model: str
    title: str
    description: str


DATASET_CONFIGS: list[DatasetConfig] = [
    DatasetConfig(
        name="phenobench",
        folder_name="01_phenobench_aruco_5x5",
        source_subpath="phenobench/images",
        coord_system="local_aruco",
        pattern="grid_5x5",
        gsd_m_per_px=0.0010,  # 1.0 mm/px -> 1024x1024 px = 1.024 x 1.024 m
        altitude_m=2.00,      # Ultra-low altitude survey (Germany, University of Bonn)
        focal_length_mm=12.0, # 12mm high-resolution lens
        sensor_name="FLIR Blackfly S (RGB 1024x1024)",
        recommended_model="d01_rgb_plant.onnx",
        title="PhenoBench — Локальная метрическая сетка 5×5 («змейка») в реальном масштабе (GSD 1.0 мм/px)",
        description=(
            "25 высокодетальных RGB-снимков сахарной свёклы и сорняков (1024×1024 px). "
            "Паспортный GSD: 0.0010 м/px (1 мм/пиксель). Физический размер одного кадра: ровно 1.024 × 1.024 м. "
            "Сетка 1×1 м на карте идеально совпадает с размером кадра! Сетка 5×5 со змейкой и 10% перекрытием."
        ),
    ),
    DatasetConfig(
        name="uavweedsegmentation",
        folder_name="02_uavweedseg_aruco_linear",
        source_subpath="uavweedsegmentation/images",
        coord_system="local_aruco",
        pattern="linear_25",
        gsd_m_per_px=0.0010,  # 1.0 mm/px -> 256x256 px = 0.256 x 0.256 m
        altitude_m=4.25,      # DJI Mavic 2 Pro flight altitude for 1 mm/px
        focal_length_mm=10.26,# Hasselblad L1D-20c 28mm equivalent
        sensor_name="DJI Mavic 2 Pro (Hasselblad L1D-20c)",
        recommended_model="d01_rgb_plant.onnx",
        title="UAVWeedSegmentation — Локальный линейный трансект (GSD 1.0 мм/px, патчи 25.6 см)",
        description=(
            "25 RGB-патчей сорго и сорняков (256×256 px). "
            "Паспортный GSD: 0.0010 м/px (1 мм/пиксель). Физический размер одного патча: 25.6 × 25.6 см. "
            "В 1 метре координатной сетки укладывается ровно 4 патча! Линейный пролёт с 10% перекрытием."
        ),
    ),
    DatasetConfig(
        name="weedmap_rededge",
        folder_name="03_weedmap_wgs84_5x5",
        source_subpath="weedmap_rededge/composite_rgb",
        coord_system="wgs84",
        pattern="grid_5x5",
        gsd_m_per_px=0.0022,  # 2.2 mm/px -> 480x360 px = 1.056 x 0.792 m
        altitude_m=20.00,     # AscTec Falcon 8 survey altitude (ETH Zurich)
        focal_length_mm=5.4,  # MicaSense RedEdge-M
        sensor_name="MicaSense RedEdge-M (480x360 tiles)",
        recommended_model="d01_rgb_plant.onnx",
        title="WeedMap RedEdge — WGS84 координаты, сетка 5×5 («змейка») в реальном масштабе (GSD 2.2 мм/px)",
        description=(
            "25 композитных RGB-тайлов сцены RedEdge 002 (480×360 px, формат 4:3). "
            "Паспортный GSD: 0.0022 м/px (2.2 мм/пиксель). Физический размер одного тайла: 1.056 × 0.792 м. "
            "Глобальные координаты WGS84, сетка 5×5 с 10% перекрытием."
        ),
    ),
    DatasetConfig(
        name="weedsgalore",
        folder_name="04_weedsgalore_wgs84_linear",
        source_subpath="weedsgalore/composite_rgb",
        coord_system="wgs84",
        pattern="linear_25",
        gsd_m_per_px=0.0025,  # 2.5 mm/px -> 600x600 px = 1.500 x 1.500 m
        altitude_m=4.70,      # DJI Phantom 4 Multispectral survey altitude
        focal_length_mm=5.74, # DJI P4 Multispectral camera
        sensor_name="DJI Phantom 4 Multispectral (P4M)",
        recommended_model="d01_rgb_plant.onnx",
        title="WeedsGalore — WGS84 координаты, линейный маршрут (GSD 2.5 мм/px, сцены 1.5×1.5 м)",
        description=(
            "25 композитных RGB-снимков кукурузы и сорняков (600×600 px). "
            "Паспортный GSD: 0.0025 м/px (2.5 мм/пиксель). Физический размер одной сцены: 1.50 × 1.50 м. "
            "Глобальные координаты WGS84, прямой маршрут БПЛА 25 кадров с 10% перекрытием."
        ),
    ),
]


def build_xmp_xml(
    *,
    cs: str,
    x: float,
    y: float,
    altitude_m: float,
    yaw_deg: float,
    pitch_deg: float = -90.0,
    roll_deg: float = 0.0,
    focal_length_mm: float = 4.5,
    sensor_width_mm: float = 6.4,
    sensor_height_mm: float = 4.8,
    principal_point: tuple[float, float] | None = None,
    band_name: str = "RGB",
) -> str:
    """Construct an Adobe XMP metadata packet adhering to GeoWeedStation xmp_parser expectations."""
    if cs == "local_aruco":
        coord_xml = f'agro:CoordinateSystem="local_aruco" agro:CameraX="{x:.3f}" agro:CameraY="{y:.3f}"'
    else:
        # For WGS84: x is longitude, y is latitude
        coord_xml = f'agro:CoordinateSystem="wgs84" drone-dji:GpsLatitude="{y:.7f}" drone-dji:GpsLongitude="{x:.7f}"'

    pp_str = f"{principal_point[0]:.1f},{principal_point[1]:.1f}" if principal_point else "0.0,0.0"

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
        Camera:SensorWidth="{sensor_width_mm:.4f}"
        Camera:SensorHeight="{sensor_height_mm:.4f}"
        Camera:PrincipalPoint="{pp_str}"
        Camera:BandName="{band_name}"
        {coord_xml}/>
  </rdf:RDF>
</x:xmpmeta>"""


def embed_xmp_into_jpeg(image_bgr: np.ndarray, xmp_xml: str, output_path: Path, quality: int = 95) -> None:
    """Encode OpenCV BGR image as JPEG and insert XMP APP1 segment right after SOI marker."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG for {output_path.name}")

    raw_jpeg = buf.tobytes()
    payload = _XMP_JPEG_HEADER + xmp_xml.encode("utf-8")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    final_data = raw_jpeg[:2] + segment + raw_jpeg[2:]
    output_path.write_bytes(final_data)


def compute_trajectory_25(
    pattern: Literal["grid_5x5", "linear_25"],
    step_x_m: float,
    step_y_m: float,
) -> list[tuple[float, float, float]]:
    """Compute (x_m, y_m, yaw_deg) for 25 frames."""
    points: list[tuple[float, float, float]] = []

    if pattern == "linear_25":
        for i in range(25):
            x = i * step_x_m
            y = 0.0
            yaw = 90.0  # Heading East along flight line
            points.append((round(x, 4), round(y, 4), yaw))

    elif pattern == "grid_5x5":
        for i in range(25):
            row = i // 5
            col_in_row = i % 5

            if row % 2 == 0:
                col = col_in_row
                yaw = 90.0
            else:
                col = 4 - col_in_row
                yaw = 270.0

            x = col * step_x_m
            y = row * step_y_m
            points.append((round(x, 4), round(y, 4), yaw))

    return points


def process_dataset(
    cfg: DatasetConfig,
    benchmark_dir: Path,
    output_base_dir: Path,
    *,
    overlap: float = 0.10,
    origin_lat: float = 56.800000,
    origin_lon: float = 60.600000,
) -> dict:
    """Generate geotagged JPEG dataset with XMP adhering to physical GSD."""
    source_dir = benchmark_dir / cfg.source_subpath
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    source_images = sorted(list(source_dir.glob("*.png")))
    if len(source_images) < 25:
        raise ValueError(f"Expected at least 25 images in {source_dir}, found {len(source_images)}")
    source_images = source_images[:25]

    target_dir = output_base_dir / cfg.folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Peek first image dimensions to calculate exact physical footprint
    sample_img = cv2.imread(str(source_images[0]))
    h_px, w_px = sample_img.shape[:2]

    # TRUE ground footprint derived strictly from image pixels and passport GSD:
    footprint_w = w_px * cfg.gsd_m_per_px
    footprint_h = h_px * cfg.gsd_m_per_px

    # Calculate exact virtual sensor dimensions matching NadirProjector formula:
    # footprint_w = altitude * (sensor_w / focal) -> sensor_w = footprint_w * focal / altitude
    sensor_w = (footprint_w * cfg.focal_length_mm) / cfg.altitude_m
    sensor_h = (footprint_h * cfg.focal_length_mm) / cfg.altitude_m

    step_x = footprint_w * (1.0 - overlap)
    step_y = footprint_h * (1.0 - overlap)

    converter = CoordinateConverter(wgs84_origin=(origin_lat, origin_lon))
    trajectory = compute_trajectory_25(cfg.pattern, step_x, step_y)

    frames_metadata = []

    bench_root = benchmark_dir / cfg.name
    labels_crop_weed_dir = bench_root / "labels_yolo_crop_weed"
    labels_plant_dir = bench_root / "labels_yolo_plant"
    annot_dest_dir = target_dir / "annotations"
    has_annots = labels_crop_weed_dir.is_dir()

    if has_annots:
        (annot_dest_dir / "labels_yolo_crop_weed").mkdir(parents=True, exist_ok=True)
        (annot_dest_dir / "labels_yolo_plant").mkdir(parents=True, exist_ok=True)

    for idx, (img_path, (lx, ly, yaw)) in enumerate(zip(source_images, trajectory, strict=True)):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Could not read source image: {img_path}")
        cur_h, cur_w = img_bgr.shape[:2]
        pp = (cur_w / 2.0, cur_h / 2.0)

        if cfg.coord_system == "wgs84":
            lat, lon = converter.local_to_wgs84(lx, ly)
            coord_x = lon
            coord_y = lat
        else:
            lat, lon = None, None
            coord_x = lx
            coord_y = ly

        xmp_xml = build_xmp_xml(
            cs=cfg.coord_system,
            x=coord_x,
            y=coord_y,
            altitude_m=cfg.altitude_m,
            yaw_deg=yaw,
            focal_length_mm=cfg.focal_length_mm,
            sensor_width_mm=sensor_w,
            sensor_height_mm=sensor_h,
            principal_point=pp,
            band_name="RGB",
        )

        out_name = f"frame_{idx:03d}_{img_path.stem}.jpg"
        out_path = target_dir / out_name
        embed_xmp_into_jpeg(img_bgr, xmp_xml, out_path, quality=95)

        # Copy matching annotations if present
        txt_name = f"{img_path.stem}.txt"
        if has_annots:
            cw_txt = labels_crop_weed_dir / txt_name
            if cw_txt.is_file():
                shutil.copy(cw_txt, annot_dest_dir / "labels_yolo_crop_weed" / f"frame_{idx:03d}_{txt_name}")
            pl_txt = labels_plant_dir / txt_name
            if pl_txt.is_file():
                shutil.copy(pl_txt, annot_dest_dir / "labels_yolo_plant" / f"frame_{idx:03d}_{txt_name}")

        frame_meta = {
            "index": idx,
            "filename": out_name,
            "source_image": img_path.name,
            "image_size_px": [cur_w, cur_h],
            "gsd_m_per_px": cfg.gsd_m_per_px,
            "local_x_m": round(lx, 4),
            "local_y_m": round(ly, 4),
            "yaw_deg": yaw,
            "altitude_m": cfg.altitude_m,
            "footprint_m": [round(footprint_w, 4), round(footprint_h, 4)],
        }
        if cfg.coord_system == "wgs84":
            frame_meta["gps_latitude"] = round(lat, 8)
            frame_meta["gps_longitude"] = round(lon, 8)

        frames_metadata.append(frame_meta)

    # Save flight_manifest.json
    manifest = {
        "dataset_name": cfg.name,
        "folder": cfg.folder_name,
        "title": cfg.title,
        "sensor": cfg.sensor_name,
        "gsd_m_per_px": cfg.gsd_m_per_px,
        "gsd_mm_per_px": round(cfg.gsd_m_per_px * 1000.0, 2),
        "coordinate_system": cfg.coord_system,
        "flight_pattern": cfg.pattern,
        "frames_count": len(frames_metadata),
        "overlap_ratio": overlap,
        "step_x_m": round(step_x, 4),
        "step_y_m": round(step_y, 4),
        "altitude_m": cfg.altitude_m,
        "optics": {
            "focal_length_mm": cfg.focal_length_mm,
            "sensor_width_mm": round(sensor_w, 4),
            "sensor_height_mm": round(sensor_h, 4),
            "footprint_w_m": round(footprint_w, 4),
            "footprint_h_m": round(footprint_h, 4),
        },
        "recommended_model": cfg.recommended_model,
        "frames": frames_metadata,
    }
    if cfg.coord_system == "wgs84":
        manifest["origin_wgs84"] = {"lat": origin_lat, "lon": origin_lon}

    manifest_path = target_dir / "flight_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save README.md for this specific folder
    readme_content = f"""# {cfg.title}

{cfg.description}

## Физические характеристики съёмки и паспорт данных
* **Сенсор камеры:** {cfg.sensor_name}
* **Паспортный GSD:** {cfg.gsd_m_per_px * 1000.0:.1f} мм/пиксель ({cfg.gsd_m_per_px} м/px)
* **Разрешение кадра:** {w_px} × {h_px} px
* **Физический размер пятна на земле:** **{footprint_w:.3f} м × {footprint_h:.3f} м**
* **Высота съёмки:** {cfg.altitude_m} м
* **Фокусное расстояние:** {cfg.focal_length_mm} мм
* **Система координат:** `{cfg.coord_system}`
* **Траектория:** `{cfg.pattern}` ({'Сетка 5×5 со змейкой' if cfg.pattern == 'grid_5x5' else 'Прямой линейный пролёт'})
* **Количество снимков:** 25 кадров (JPEG с вшитым XMP)
* **Перекрытие между смежными кадрами:** {int(overlap * 100)}%
* **Шаг между центрами кадров:** $\\Delta X = {step_x:.3f}$ м, $\\Delta Y = {step_y:.3f}$ м
* **Рекомендуемая модель для теста:** `{cfg.recommended_model}`

## Соответствие масштабу координатной сетки карты
Координатная сетка в GeoWeedStation отображается с шагом 1.0 метр:
* В этом датасете кадр занимает **{footprint_w:.2f} м × {footprint_h:.2f} м**, поэтому линии сетки 1×1 м идеально соответствуют реальным размерам растений и почвы.
* Любое детектированное растение отображается в своих реальных физических габаритах (10–30 см).

## Как проверить в GeoWeedStation
1. Запустите GeoWeedStation:
   ```bash
   ./run_geoweed.sh
   ```
2. В верхнем меню выберите **«Файл» → «Открыть каталог снимков...»** (`Ctrl+O`).
3. Укажите эту папку: `{target_dir}`.
4. В панели «Пакетные файлы» проверьте, что у всех 25 снимков горит зелёная отметка **XMP: Да**.
5. Нажмите **«⚡ Запустить обработку»**.
6. Перейдите на вкладку **«Карта»** — на ней отобразится сетка кадров в физически точном масштабе, сорняки, тепловая карта и зоны опрыскивания.
"""
    (target_dir / "README.md").write_text(readme_content, encoding="utf-8")

    return manifest


def verify_generated_datasets(output_base_dir: Path) -> bool:
    """Verify generated datasets with GeoWeedStation core parsing and MapEngine."""
    print("\n" + "=" * 75)
    print("ВЕРИФИКАЦИЯ ФИЗИЧЕСКИ ТОЧНЫХ НАБОРОВ ДАННЫХ ДЛЯ GEOWEEDSTATION")
    print("=" * 75)

    all_passed = True
    map_settings = MapSettings()

    for cfg in DATASET_CONFIGS:
        target_dir = output_base_dir / cfg.folder_name
        jpgs = sorted(list(target_dir.glob("*.jpg")))
        if len(jpgs) != 25:
            print(f"❌ [{cfg.name}] Ошибка: найдено {len(jpgs)} файлов, ожидалось 25")
            all_passed = False
            continue

        xmp_ok_count = 0
        for p in jpgs:
            if has_xmp(p):
                xmp_ok_count += 1
            else:
                print(f"   Файл без XMP: {p.name}")

        if xmp_ok_count != 25:
            print(f"❌ [{cfg.name}] XMP-валидность: {xmp_ok_count}/25")
            all_passed = False
            continue

        engine = MapEngine(NadirProjector(), map_settings)
        ingested_count = 0
        for p in jpgs:
            sp = extract_spatial_data(p)
            fp = build_frame_payload(p, (1024, 1024))
            if fp.has_georef():
                engine.ingest_frame(fp)
                ingested_count += 1

        bounds = engine.weed_map.bounds()
        footprints_count = len(engine.weed_map.footprints)
        fp0 = engine.weed_map.footprints[0]

        if ingested_count == 25 and footprints_count == 25:
            print(
                f"✅ [{cfg.folder_name}]\n"
                f"   Сенсор: {cfg.sensor_name} | GSD: {cfg.gsd_m_per_px*1000:.1f} мм/px\n"
                f"   Кадр: {fp0.width_m:.3f} × {fp0.height_m:.3f} м (H={fp0.altitude_m:.1f} м) | СК: {cfg.coord_system:11s} | Траектория: {cfg.pattern}\n"
                f"   Охват карты: X=[{bounds.min_x:.2f}, {bounds.max_x:.2f}] м, Y=[{bounds.min_y:.2f}, {bounds.max_y:.2f}] м | Площадь: {bounds.width * bounds.height:.2f} м²"
            )
        else:
            print(f"❌ [{cfg.name}] Ошибка картографического движка: ingested={ingested_count}, footprints={footprints_count}")
            all_passed = False

    print("=" * 75)
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Генерация геопривязанных тестовых наборов с реальным физическим GSD и 10% перекрытием."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=_REPO_ROOT / "workdir" / "validation_benchmark-20261109T095055Z-1-001" / "validation_benchmark",
        help="Путь к исходному каталогу validation_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "workdir" / "validation_runs",
        help="Целевой каталог для подготовленных тестовых наборов (default: workdir/validation_runs)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.10,
        help="Доля взаимного перекрытия между смежными кадрами (default: 0.10 = 10%)",
    )
    parser.add_argument(
        "--origin-lat",
        type=float,
        default=56.800000,
        help="Базовая широта WGS84 для географических наборов (default: 56.800000)",
    )
    parser.add_argument(
        "--origin-lon",
        type=float,
        default=60.600000,
        help="Базовая долгота WGS84 для географических наборов (default: 60.600000)",
    )

    args = parser.parse_args()

    bench_dir = args.benchmark_dir.resolve()
    out_dir = args.output_dir.resolve()

    if not bench_dir.is_dir():
        print(f"Ошибка: Каталог бенчмарка не найден: {bench_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Начало подготовки геопривязанных наборов данных с физически точным GSD:")
    print(f"  Исходный бенчмарк: {bench_dir}")
    print(f"  Целевой каталог:   {out_dir}")
    print(f"  Перекрытие:        {int(args.overlap * 100)}%")
    print(f"  Базовая WGS84:     ({args.origin_lat:.6f}, {args.origin_lon:.6f})\n")

    summary_reports = []

    for cfg in DATASET_CONFIGS:
        print(f"Обработка датасета: {cfg.name} -> {cfg.folder_name} (GSD {cfg.gsd_m_per_px*1000:.1f} мм/px, {cfg.coord_system}, {cfg.pattern})...")
        meta = process_dataset(
            cfg,
            bench_dir,
            out_dir,
            overlap=args.overlap,
            origin_lat=args.origin_lat,
            origin_lon=args.origin_lon,
        )
        summary_reports.append(meta)
        print(f"  ✓ Сгенерировано 25 снимков в {out_dir / cfg.folder_name}")

    # Generate master index README.md
    index_md = f"""# Валидационные наборы данных AeroWeed 2026 для GeoWeedStation (Физически точный масштаб)

Сгенерированные тестовые наборы с точной физической геопривязкой (XMP-метаданные),
строгим соответствием паспортному **GSD (Ground Sampling Distance)** каждого сенсора
и **10% перекрытием** между смежными кадрами.

## Сводная таблица параметров масштабирования

| Каталог | Сенсор | Исходное разрешение | Паспортный GSD | Физический размер кадра $W \\times H$ | Шаг $\\Delta X, \\Delta Y$ (10% нахлёст) | СК | Траектория |
|---|---|---|---|---|---|---|---|
| `01_phenobench_aruco_5x5/` | FLIR Blackfly S RGB | $1024 \\times 1024$ px | **1.0 мм/px** | **1.024 × 1.024 м** | 0.922 м, 0.922 м | `local_aruco` | Сетка 5×5 («змейка») |
| `02_uavweedseg_aruco_linear/` | DJI Mavic 2 Pro (патчи) | $256 \\times 256$ px | **1.0 мм/px** | **0.256 × 0.256 м** | 0.230 м, — | `local_aruco` | Линейный трансект (25) |
| `03_weedmap_wgs84_5x5/` | MicaSense RedEdge-M | $480 \\times 360$ px | **2.2 мм/px** | **1.056 × 0.792 м** | 0.950 м, 0.713 м | `wgs84` | Сетка 5×5 («змейка») |
| `04_weedsgalore_wgs84_linear/` | DJI Phantom 4 Multispectral | $600 \\times 600$ px | **2.5 мм/px** | **1.500 × 1.500 м** | 1.350 м, — | `wgs84` | Линейный трансект (25) |

## Соответствие координатной сетке 1×1 метр
* **PhenoBench:** Кадр $1.024 \\times 1.024$ м практически идеально вписывается в одну ячейку 1-метровой координатной сетки карты.
* **UAVWeedSegmentation:** Ровно 4 патча ($25.6 \\times 25.6$ см) укладываются в 1 метр сетки.
* **WeedMap:** Тайл $1.056 \\times 0.792$ м сопоставим с одной 1-метровой ячейкой.
* **WeedsGalore:** Сцена $1.50 \\times 1.50$ м занимает $1.5$ ячейки сетки.
"""
    (out_dir / "README.md").write_text(index_md, encoding="utf-8")

    # Run self-verification
    ok = verify_generated_datasets(out_dir)
    if not ok:
        print("Ошибка верификации наборов данных!", file=sys.stderr)
        return 1

    print("\nГенерация и верификация всех 4 наборов данных успешно завершена!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
