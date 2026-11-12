#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c07_rgb_geosaero/train.py - Обучение двухголового классификатора GeosAero (11 категорий + 5 стадий)
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

from common.dataset import load_master_dataset, get_classification_samples, GeosAeroDualHeadDataset, collate_fn_dual_head
from common.models.resnet import build_resnet50_dual_head

MODEL_ID = "c07_rgb_geosaero"
IN_CHANNELS = 3
NUM_CATS = 11
NUM_STAGES = 5
RESOLUTION = 224


def build_transforms(is_train: bool = True):
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
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 3.0e-4,
    stage_weight: float = 1.5,
    device_name: Optional[str] = None,
    dry_run: bool = False,
    weights_path: Optional[str] = None,
    output_dir: Optional[str] = None
):
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    runs_dir = Path(output_dir) if output_dir else ROOT_DIR / "runs" / MODEL_ID
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Обучение {MODEL_ID} (ResNet-50 DualHead, 11 кат. + 5 стадий)")
    print(f"Устройство: {device} | Эпох: {epochs} | Batch size: {batch_size}")
    print("=" * 60)

    master_data = load_master_dataset(master_json)
    samples = get_classification_samples(master_data, model_id=MODEL_ID)
    if not samples:
        samples = [s for s in master_data.get("classification_samples", []) if s.get("dataset_id") == "GeosAero_hitech_dataset"]

    if dry_run:
        print("⚡ [DRY-RUN]: Ограничение выборки")
        samples = samples[:4]
        epochs = 1

    train_samples = samples
    val_samples = samples

    cat_to_idx = {f"cat_{i}": i for i in range(NUM_CATS)}
    stage_to_idx = {f"stage_{i}": i for i in range(NUM_STAGES)}

    train_ds = GeosAeroDualHeadDataset(train_samples, cat_to_idx, stage_to_idx, input_size=(RESOLUTION, RESOLUTION), transforms=build_transforms(True), is_train=True)
    val_ds = GeosAeroDualHeadDataset(val_samples, cat_to_idx, stage_to_idx, input_size=(RESOLUTION, RESOLUTION), transforms=build_transforms(False), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2 if not dry_run else 0, collate_fn=collate_fn_dual_head)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if not dry_run else 0, collate_fn=collate_fn_dual_head)

    model = build_resnet50_dual_head(num_classes_category=NUM_CATS, num_classes_stage=NUM_STAGES, in_channels=IN_CHANNELS, pretrained=False).to(device)

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

    crit_cat = nn.CrossEntropyLoss()
    crit_stage = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)

    best_combined_acc = 0.0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1:02d}/{epochs:02d} [Train]")
        train_loss = 0.0
        for step, (images, cat_targets, stage_targets, _) in enumerate(pbar):
            images, cat_targets, stage_targets = images.to(device), cat_targets.to(device), stage_targets.to(device)

            optimizer.zero_grad()
            out_cat, out_stage = model(images)
            l_cat = crit_cat(out_cat, cat_targets)
            l_stage = crit_stage(out_stage, stage_targets)
            loss = l_cat + stage_weight * l_stage
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            if dry_run and step >= 1:
                break

        # Валидация
        model.eval()
        c_corr, s_corr, tot = 0, 0, 0
        with torch.no_grad():
            for images, cat_targets, stage_targets, _ in val_loader:
                images, cat_targets, stage_targets = images.to(device), cat_targets.to(device), stage_targets.to(device)
                out_cat, out_stage = model(images)
                c_corr += (torch.argmax(out_cat, dim=1) == cat_targets).sum().item()
                s_corr += (torch.argmax(out_stage, dim=1) == stage_targets).sum().item()
                tot += images.size(0)
                if dry_run:
                    break

        cat_acc = c_corr / max(1, tot)
        stg_acc = s_corr / max(1, tot)
        comb_acc = (cat_acc + stg_acc) / 2.0
        print(f"📊 Эпоха {epoch+1:02d}: Категории Acc = {cat_acc:.4f} ({cat_acc*100:.1f}%), Стадии Acc = {stg_acc:.4f} ({stg_acc*100:.1f}%)")

        metrics = {"category_accuracy": cat_acc, "stage_accuracy": stg_acc, "combined_accuracy": comb_acc}
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, runs_dir / "last.pt")
        if comb_acc >= best_combined_acc:
            best_combined_acc = comb_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, runs_dir / "best.pt")
            print(f"⭐ Новый лучший результат: {runs_dir / 'best.pt'}")

    print(f"\n✅ Обучение {MODEL_ID} завершено! Результаты в {runs_dir}")


def main():
    parser = argparse.ArgumentParser(description=f"Обучение {MODEL_ID}")
    parser.add_argument("--master-json", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3.0e-4)
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
