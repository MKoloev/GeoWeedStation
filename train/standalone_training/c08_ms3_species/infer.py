#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c08_ms3_species/infer.py - Инференс классификатора видов MS3 (4 класса: S000, S001, S009, S022)
и вычисление метрик (Accuracy, Macro F1, Precision, Recall).
"""

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import load_master_dataset, get_classification_samples, PlantCropDataset, collate_fn_classification
from common.metrics import ClassificationEvaluator
from common.models.resnet import build_resnet50

MODEL_ID = "c08_ms3_species"
IN_CHANNELS = 3
NUM_CLASSES = 4
RESOLUTION = 224

SPECIES_TAXONOMY = {
    0: {"id": "S000", "latin": "Background", "ru": "Фон / Не растение"},
    1: {"id": "S001", "latin": "Zea mays", "ru": "Кукуруза сахарная"},
    2: {"id": "S009", "latin": "Beta vulgaris", "ru": "Свёкла обыкновенная"},
    3: {"id": "S022", "latin": "Datura stramonium", "ru": "Дурман обыкновенный"}
}


def run_evaluation(model, val_samples, device, output_dir):
    print(f"\n📊 Оценка {MODEL_ID} на {len(val_samples)} кропах...")
    species_to_idx = {"S000": 0, "S001": 1, "S009": 2, "S022": 3}
    val_ds = PlantCropDataset(val_samples, species_to_idx=species_to_idx, input_size=(RESOLUTION, RESOLUTION), is_train=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn_classification)

    evaluator = ClassificationEvaluator(num_classes=NUM_CLASSES)
    model.eval()

    with torch.no_grad():
        for images, targets, _ in tqdm(val_loader, desc="Оценка точности"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            evaluator.update(outputs, targets)

    metrics = evaluator.compute()
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("macro_f1", 0.0)
    print("\n" + "=" * 50)
    print(f"📈 РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ {MODEL_ID}:")
    print(f"  • Accuracy (Общая точность): {acc:.4f} ({acc*100:.1f}%)")
    print(f"  • Macro F1-Score:           {f1:.4f}")
    print(f"  • Macro Precision:         {metrics.get('macro_precision', 0):.4f}")
    print(f"  • Macro Recall:            {metrics.get('macro_recall', 0):.4f}")
    print("=" * 50 + "\n")

    out_f = output_dir / "metrics_evaluation.json"
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 Метрики сохранены: {out_f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=f"Инференс и метрики {MODEL_ID}")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else (ROOT_DIR / "runs" / MODEL_ID / "infer_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (ROOT_DIR / "runs" / MODEL_ID / "best.pt")
    if not ckpt_path.exists():
        fallback = ROOT_DIR.parent / "runs" / MODEL_ID / "best.pt"
        if fallback.exists():
            ckpt_path = fallback

    print(f"🔧 Загрузка {MODEL_ID} на {device}...")
    model = build_resnet50(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, pretrained=False).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        st = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(st, strict=False)
        print(f"✅ Веса загружены из: {ckpt_path}")
    else:
        print("⚠️ Чекпоинт не найден, запуск со случайными весами.")

    model.eval()

    try:
        master = load_master_dataset()
        val_samples = get_classification_samples(master, model_id=MODEL_ID, split="val")
        if not val_samples:
            val_samples = get_classification_samples(master, model_id=MODEL_ID, split="train")[:30]
        run_evaluation(model, val_samples, device, out_dir)
    except Exception as e:
        print(f"⚠️ Ошибка при оценке: {e}")


if __name__ == "__main__":
    main()
