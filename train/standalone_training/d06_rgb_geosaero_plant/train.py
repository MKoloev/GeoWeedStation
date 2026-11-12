#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d01_rgb_plant/train.py - Обучение RGB-детектора растений YOLOX-M (D01-RGB-PLANT)
на основе единого JSON-датасета master_dataset.json.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import load_master_dataset, get_detection_samples, PlantPatchDataset, collate_fn_detection
from common.losses import YOLOXLoss
from common.metrics import DetectionEvaluator
from common.models.yolox import build_yolox_m
from common.transforms import (
    AffineAugmentation,
    RandomNoiseSP,
    SyncRandomBrightnessContrastTarget,
    SyncRandomCropTarget,
    SyncRandomHorizontalFlipTarget,
    SyncRandomVerticalFlipTarget,
    SyncResize,
    SyncRotateTarget,
    SyncToTensor,
    TrickyResize_UpDwn,
)

MODEL_ID = "d06_rgb_geosaero_plant"
IN_CHANNELS = 3
NUM_CLASSES = 1
RESOLUTION = 640


def build_augmentations(resolution: int = 640):
    train_transforms = [
        SyncRandomBrightnessContrastTarget(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
        SyncRandomHorizontalFlipTarget(p=0.5),
        SyncRandomVerticalFlipTarget(p=0.5),
        SyncRotateTarget(p=0.8, angle_range=(-45, 45), resolution=resolution),
        SyncRandomCropTarget(resolution=resolution),
        TrickyResize_UpDwn(resolution=resolution, minmax_size_up=[117, 200]),
        AffineAugmentation(p=0.75, scale=(0.35, 2.0)),
        SyncResize(size=resolution),
        SyncToTensor(bbox_padding=0, plant_label=0),
        RandomNoiseSP(p=0.5)
    ]
    val_transforms = [
        SyncResize(size=resolution),
        SyncToTensor(bbox_padding=0, plant_label=0)
    ]
    return train_transforms, val_transforms


def decode_predictions_for_eval(
    outputs: List[torch.Tensor],
    score_thresh: float = 0.25,
    nms_thresh: float = 0.50,
    strides: Tuple[int, ...] = (8, 16, 32)
) -> List[Dict[str, torch.Tensor]]:
    device = outputs[0].device
    batch_size = outputs[0].shape[0]
    all_decoded = []

    for b_idx in range(batch_size):
        b_boxes, b_scores, b_labels = [], [], []
        for out, stride in zip(outputs, strides):
            reg = out[b_idx, :4]
            obj = torch.sigmoid(out[b_idx, 4:5])
            cls_prob = torch.sigmoid(out[b_idx, 5:])
            conf = obj * cls_prob
            h, w = out.shape[2:]

            yv, xv = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            pos_mask = (conf[0] >= score_thresh)
            if not pos_mask.any():
                continue

            sub_xv = xv[pos_mask]
            sub_yv = yv[pos_mask]
            sub_reg = reg[:, pos_mask]
            sub_scores = conf[0][pos_mask]

            cx = (sub_reg[0] + sub_xv) * stride
            cy = (sub_reg[1] + sub_yv) * stride
            bw = torch.exp(sub_reg[2]) * stride
            bh = torch.exp(sub_reg[3]) * stride

            boxes_k = torch.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dim=-1)
            b_boxes.append(boxes_k)
            b_scores.append(sub_scores)
            b_labels.append(torch.zeros(len(sub_scores), dtype=torch.int64, device=device))

        if len(b_boxes) > 0:
            b_boxes = torch.cat(b_boxes, dim=0)
            b_scores = torch.cat(b_scores, dim=0)
            b_labels = torch.cat(b_labels, dim=0)
            keep = nms(b_boxes, b_scores, iou_threshold=nms_thresh)
            b_boxes = b_boxes[keep]
            b_scores = b_scores[keep]
            b_labels = b_labels[keep]
        else:
            b_boxes = torch.empty((0, 4), device=device)
            b_scores = torch.empty((0,), device=device)
            b_labels = torch.empty((0,), dtype=torch.int64, device=device)

        all_decoded.append({'boxes': b_boxes, 'scores': b_scores, 'labels': b_labels})
    return all_decoded


def train(
    master_json: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 1.0e-3,
    device_name: Optional[str] = None,
    dry_run: bool = False,
    weights_path: Optional[str] = None,
    output_dir: Optional[str] = None
):
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    runs_dir = Path(output_dir) if output_dir else ROOT_DIR / "runs" / MODEL_ID
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Запуск обучения: {MODEL_ID} (YOLOX-M, {IN_CHANNELS} кан.)")
    print(f"Устройство: {device} | Эпох: {epochs} | Batch size: {batch_size}")
    print("=" * 60)

    # 1. Загрузка выборки из единого JSON-датасета
    master_data = load_master_dataset(master_json)
    train_samples = get_detection_samples(master_data, model_id=MODEL_ID, split="train")
    val_samples = get_detection_samples(master_data, model_id=MODEL_ID, split="val")

    # Если сплит val пуст, используем срез train
    if not val_samples and len(train_samples) > 10:
        val_samples = train_samples[-10:]
        train_samples = train_samples[:-10]

    if dry_run:
        print("⚡ [DRY-RUN]: Ограничение до 4 train и 2 val сэмплов, 1 эпоха")
        train_samples = train_samples[:4]
        val_samples = val_samples[:2]
        epochs = 1

    print(f"Выборка: Train = {len(train_samples)}, Val = {len(val_samples)} сэмплов.")

    train_tf, val_tf = build_augmentations(resolution=RESOLUTION)
    train_ds = PlantPatchDataset(train_samples, resolution=RESOLUTION, transforms=train_tf, is_train=True)
    val_ds = PlantPatchDataset(val_samples, resolution=RESOLUTION, transforms=val_tf, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2 if not dry_run else 0, collate_fn=collate_fn_detection)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if not dry_run else 0, collate_fn=collate_fn_detection)

    # 2. Инициализация модели
    model = build_yolox_m(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=False).to(device)

    # Загрузка весов при наличии
    init_weights = weights_path or (ROOT_DIR.parent / "runs" / MODEL_ID / "best.pt")
    if init_weights and Path(init_weights).exists():
        print(f"📥 Загрузка начальных весов из: {init_weights}")
        ckpt = torch.load(str(init_weights), map_location=device)
        st = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(st, strict=False)
            print("✅ Веса успешно загружены!")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки весов: {e}")

    criterion = YOLOXLoss(num_classes=NUM_CLASSES, strides=(8, 16, 32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)

    best_ap50 = 0.0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1:02d}/{epochs:02d} [Train]")
        train_loss = 0.0
        for step, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets_gpu = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device), 'is_partial': t.get('is_partial', False)} for t in targets]

            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets_gpu, imgs=images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            if dry_run and step >= 1:
                break

        # Валидация
        model.eval()
        evaluator = DetectionEvaluator(score_threshold=0.25)
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = decode_predictions_for_eval(outputs, score_thresh=0.25, nms_thresh=0.50)
                evaluator.update(preds, targets)
                if dry_run:
                    break

        metrics = evaluator.compute()
        val_ap50 = metrics.get('val_mean_ap50', metrics.get('ap50', 0.0))
        val_f1 = metrics.get('val_mean_f1', metrics.get('f1', 0.0))
        val_map = metrics.get('val_mean_map50_95', metrics.get('map50_95', 0.0))
        print(f"📊 Эпоха {epoch+1:02d}: AP50 = {val_ap50:.4f}, mAP50:95 = {val_map:.4f}, F1 = {val_f1:.4f}")

        # Сохранение чекпоинтов
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, runs_dir / "last.pt")
        if val_ap50 >= best_ap50:
            best_ap50 = val_ap50
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, runs_dir / "best.pt")
            print(f"⭐ Новый лучший результат! Сохранено: {runs_dir / 'best.pt'}")

    print(f"\n✅ Обучение {MODEL_ID} завершено! Результаты в {runs_dir}")


def main():
    parser = argparse.ArgumentParser(description=f"Обучение {MODEL_ID}")
    parser.add_argument("--master-json", type=str, default=None, help="Путь к master_dataset.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    train(
        master_json=args.master_json,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
        dry_run=args.dry_run,
        weights_path=args.weights,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
