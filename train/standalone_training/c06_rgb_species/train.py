#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c06_rgb_species/train.py - Обучение RGB-классификатора ботанических видов (24 класса)
на основе единого JSON-датасета master_dataset.json.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common.dataset import load_master_dataset, get_classification_samples, PlantCropDataset, collate_fn_classification
from common.metrics import ClassificationEvaluator
from common.models.resnet import build_resnet50

MODEL_ID = "c06_rgb_species"
IN_CHANNELS = 3
NUM_CLASSES = 24
RESOLUTION = 224


def build_crop_transforms(is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train(
    master_json: Optional[str] = None,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 5.0e-4,
    device_name: Optional[str] = None,
    dry_run: bool = False,
    weights_path: Optional[str] = None,
    output_dir: Optional[str] = None
):
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    runs_dir = Path(output_dir) if output_dir else ROOT_DIR / "runs" / MODEL_ID
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Обучение {MODEL_ID} (ResNet-50, {NUM_CLASSES} классов видов)")
    print(f"Устройство: {device} | Эпох: {epochs} | Batch size: {batch_size}")
    print("=" * 60)

    master_data = load_master_dataset(master_json)
    train_samples = get_classification_samples(master_data, model_id=MODEL_ID, split="train")
    val_samples = get_classification_samples(master_data, model_id=MODEL_ID, split="val")

    if not val_samples and len(train_samples) > 10:
        val_samples = train_samples[-10:]
        train_samples = train_samples[:-10]

    if dry_run:
        print("⚡ [DRY-RUN]: Ограничение выборки")
        train_samples = train_samples[:8]
        val_samples = val_samples[:4]
        epochs = 1

    print(f"Выборка: Train = {len(train_samples)}, Val = {len(val_samples)} кропов.")

    species_to_idx = {f"S{i:03d}": i for i in range(NUM_CLASSES)}

    train_tf = build_crop_transforms(is_train=True)
    val_tf = build_crop_transforms(is_train=False)

    train_ds = PlantCropDataset(train_samples, species_to_idx=species_to_idx, input_size=(RESOLUTION, RESOLUTION), transforms=train_tf, is_train=True)
    val_ds = PlantCropDataset(val_samples, species_to_idx=species_to_idx, input_size=(RESOLUTION, RESOLUTION), transforms=val_tf, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2 if not dry_run else 0, collate_fn=collate_fn_classification)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if not dry_run else 0, collate_fn=collate_fn_classification)

    model = build_resnet50(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, pretrained=False).to(device)

    init_weights = weights_path or (ROOT_DIR.parent / "runs" / MODEL_ID / "best.pt")
    if init_weights and Path(init_weights).exists():
        print(f"📥 Загрузка чекпоинта из: {init_weights}")
        ckpt = torch.load(str(init_weights), map_location=device)
        st = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(st, strict=False)
            print("✅ Веса успешно загружены!")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки весов: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1:02d}/{epochs:02d} [Train]")
        train_loss = 0.0
        correct, total = 0, 0
        for step, (images, targets, _) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/max(1,total):.3f}"})
            if dry_run and step >= 1:
                break

        # Валидация
        model.eval()
        evaluator = ClassificationEvaluator(num_classes=NUM_CLASSES)
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                evaluator.update(outputs, targets)
                if dry_run:
                    break

        metrics = evaluator.compute()
        val_acc = metrics.get("accuracy", 0.0)
        print(f"📊 Эпоха {epoch+1:02d}: Accuracy = {val_acc:.4f} ({val_acc*100:.1f}%), Macro F1 = {metrics.get('macro_f1', 0):.4f}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, runs_dir / "last.pt")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, runs_dir / "best.pt")
            print(f"⭐ Новый лучший результат: {runs_dir / 'best.pt'}")

    print(f"\n✅ Обучение {MODEL_ID} завершено! Результаты в {runs_dir}")


def main():
    parser = argparse.ArgumentParser(description=f"Обучение {MODEL_ID}")
    parser.add_argument("--master-json", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5.0e-4)
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
