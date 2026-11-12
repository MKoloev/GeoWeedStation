#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_unified_dataset.py - Сборка единого мастер-датасета засоренностей в формате JSON
для задач детекции и видовой классификации на аэрофотоснимках БПЛА (AeroWeed 2026).

Скрипт объединяет данные всех 10 моделей и всех используемых датасетов:
  - Детекция (D01-D06): RGB, MS3, MS4, MS5, MS7, GeosAero RGB
  - Классификация (C06-C09): RGB 24 вида, GeosAero DualHead, MS3 4 вида, MS4 4 вида

Выходной файл: master_dataset.json
Формат совместим со стандартным COCO JSON (секции images, annotations, categories),
а также содержит расширенные спектральные и масштабные метаданные (GSD, scale_factor, input_bands).
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
PREPARED_DIR = PROJECT_ROOT / "prepared"
OUTPUT_JSON = ROOT_DIR / "master_dataset.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def build_unified_dataset(output_path: Path = OUTPUT_JSON) -> Dict[str, Any]:
    print("=" * 70)
    print("🌾 СБОРКА ЕДИНОГО JSON-ДАТАСЕТА ЗАСОРЕННОСТЕЙ (AEROWEED 2026)")
    print("=" * 70)

    # 1. Сбор детекционных сэмплов по моделям
    det_models = [
        ("d01_rgb_plant", ["R", "G", "B"]),
        ("d02_ms3_plant", ["MS_R", "MS_G", "MS_NIR"]),
        ("d03_ms4_plant", ["MS_R", "MS_G", "MS_RE", "MS_NIR"]),
        ("d04_ms5_plant", ["R", "G", "B", "RE", "NIR"]),
        ("d05_ms7_plant", ["RGB_R", "RGB_G", "RGB_B", "MS_R", "MS_G", "MS_RE", "MS_NIR"]),
        ("d06_rgb_geosaero_plant", ["R", "G", "B"]),
    ]

    unique_detection_samples: Dict[str, Dict[str, Any]] = {}
    det_model_sample_ids: Dict[str, List[str]] = {m_id: [] for m_id, _ in det_models}

    for model_id, default_bands in det_models:
        manifest_p = PREPARED_DIR / model_id / "dataset_manifest.jsonl"
        records = load_jsonl(manifest_p)
        print(f"📦 Модель {model_id}: загружено {len(records)} детекционных записей")
        for rec in records:
            s_id = rec.get("sample_id")
            if not s_id:
                continue

            # Добавляем привязку модели
            if s_id not in unique_detection_samples:
                item = dict(rec)
                item["applicable_models"] = [model_id]
                if "input_bands" not in item:
                    item["input_bands"] = default_bands
                unique_detection_samples[s_id] = item
            else:
                if model_id not in unique_detection_samples[s_id]["applicable_models"]:
                    unique_detection_samples[s_id]["applicable_models"].append(model_id)

            det_model_sample_ids[model_id].append(s_id)

    # 2. Сбор классификационных сэмплов (C06, C07, C08, C09)
    cls_models = [
        "c06_rgb_species",
        "c07_rgb_geosaero",
        "c08_ms3_species",
        "c09_ms4_species"
    ]
    unique_classification_samples: Dict[str, Dict[str, Any]] = {}
    cls_model_sample_ids: Dict[str, List[str]] = {m_id: [] for m_id in cls_models}

    for model_id in cls_models:
        manifest_p = PREPARED_DIR / model_id / "dataset_manifest.jsonl"
        records = load_jsonl(manifest_p)
        print(f"🌿 Модель {model_id}: загружено {len(records)} классификационных кропов")
        for rec in records:
            s_id = rec.get("sample_id")
            if not s_id:
                continue
            if s_id not in unique_classification_samples:
                item = dict(rec)
                item["applicable_models"] = [model_id]
                unique_classification_samples[s_id] = item
            else:
                if model_id not in unique_classification_samples[s_id]["applicable_models"]:
                    unique_classification_samples[s_id]["applicable_models"].append(model_id)
            cls_model_sample_ids[model_id].append(s_id)

    det_list = list(unique_detection_samples.values())
    cls_list = list(unique_classification_samples.values())

    # 3. Построение стандартных секций COCO (images, annotations, categories)
    coco_categories = [
        {"id": 0, "name": "plant", "supercategory": "vegetation"}
    ]
    coco_images = []
    coco_annotations = []
    ann_id_counter = 1

    for img_idx, s in enumerate(det_list, start=1):
        s["coco_image_id"] = img_idx
        # Исходные файлы
        src = s.get("source_files", [])
        primary_file = src[0] if src else s.get("image_path", "")
        sh = s.get("shape", [640, 640, 3])
        h, w = (sh[0], sh[1]) if len(sh) >= 2 else (640, 640)

        coco_images.append({
            "id": img_idx,
            "sample_id": s.get("sample_id"),
            "file_name": str(primary_file),
            "source_files": [str(x) for x in src] if src else [str(primary_file)],
            "width": int(w),
            "height": int(h),
            "channels": int(sh[2]) if len(sh) >= 3 else 3,
            "input_bands": s.get("input_bands", ["R", "G", "B"]),
            "channel_source": s.get("channel_source", "RGB"),
            "dataset_id": s.get("dataset_id", "unknown"),
            "assigned_split": s.get("assigned_split", "train"),
            "scale_factor": float(s.get("scale_factor", 1.0)),
            "is_partial": bool(s.get("is_partial", False) or s.get("annotation_completeness") == "partial_plant"),
            "applicable_models": s.get("applicable_models", [])
        })

        boxes = s.get("boxes_xyxy", [])
        for box in boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                coco_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_idx,
                    "category_id": 0,
                    "bbox": [x1, y1, bw, bh],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "area": float(bw * bh),
                    "iscrowd": 0
                })
                ann_id_counter += 1

    # Статистика по наборам данных
    datasets_stats = {}
    for s in det_list:
        d_name = s.get("dataset_id", "Unknown")
        if d_name not in datasets_stats:
            datasets_stats[d_name] = {"samples": 0, "bboxes": 0, "splits": {"train": 0, "val": 0, "test": 0}}
        datasets_stats[d_name]["samples"] += 1
        datasets_stats[d_name]["bboxes"] += len(s.get("boxes_xyxy", []))
        sp = s.get("assigned_split", "train")
        datasets_stats[d_name]["splits"][sp] = datasets_stats[d_name]["splits"].get(sp, 0) + 1

    # 4. Итоговая структура JSON
    master_dataset = {
        "info": {
            "title": "AeroWeed 2026 Unified Weed & Plant Detection Dataset",
            "version": "1.0",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Единый объединенный JSON-датасет засоренностей для задач детекции и видовой классификации на аэрофотоснимках БПЛА.",
            "reference_gsd_m_per_px": 0.0018,
            "target_scale_description": "Все размеры согласованы с эталонным масштабом CoFly-WeedDB (1.8 мм/пиксель). Окно детектора 640x640 охватывает ровно 1.152 x 1.152 м на земле.",
            "total_detection_samples": len(det_list),
            "total_detection_bboxes": len(coco_annotations),
            "total_classification_crops": len(cls_list),
            "datasets_count": len(datasets_stats),
            "datasets_summary": datasets_stats,
            "models_sample_index": {
                "detection": {k: len(v) for k, v in det_model_sample_ids.items()},
                "classification": {k: len(v) for k, v in cls_model_sample_ids.items()}
            }
        },
        "categories": coco_categories,
        "detection_samples": det_list,
        "classification_samples": cls_list,
        "images": coco_images,
        "annotations": coco_annotations
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_dataset, f, ensure_ascii=False, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"✅ ЕДИНЫЙ JSON-ДАТАСЕТ УСПЕШНО СФОРМИРОВАН: {output_path}")
    print(f"📊 Размер файла: {size_mb:.2f} МБ")
    print(f"🖼  Всего детекционных кадров/тайлов: {len(det_list)}")
    print(f"📦 Всего bounding box разметок: {len(coco_annotations)}")
    print(f"🌱 Всего классификационных кропов: {len(cls_list)}")
    print(f"📚 Объединено источников данных: {len(datasets_stats)}")
    print("=" * 70 + "\n")

    return master_dataset


if __name__ == "__main__":
    build_unified_dataset()
