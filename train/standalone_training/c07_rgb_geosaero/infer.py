#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c07_rgb_geosaero/infer.py - Инференс двухголового классификатора GeosAero
(11 категорий + 5 стадий роста) и вычисление метрик точности.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from PIL import Image
import torch
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

GEOSAERO_CATEGORIES = {
    0: "background (фон)",
    1: "Осот / Бодяк / Латук",
    2: "Вьюнок",
    3: "Щирица",
    4: "Маревые",
    5: "Канатник / просвирник",
    6: "Крестоцветные",
    7: "Горец",
    8: "Злаковые",
    9: "Полынь / Ромашка",
    10: "Гвоздичные"
}

GEOSAERO_STAGES = {
    0: "background",
    1: "всходы (проростки)",
    2: "2-4 настоящих листа",
    3: "взрослое растение",
    4: "переросшие (цветение/семена)"
}


def run_evaluation(model, samples, device, output_dir):
    print(f"\n📊 Оценка GeosAero DualHead на {len(samples)} кропах...")
    eval_tf = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cat_to_idx = {f"cat_{i}": i for i in range(NUM_CATS)}
    stage_to_idx = {f"stage_{i}": i for i in range(NUM_STAGES)}

    ds = GeosAeroDualHeadDataset(samples, cat_to_idx, stage_to_idx, input_size=(RESOLUTION, RESOLUTION), transforms=eval_tf, is_train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn_dual_head)

    model.eval()
    c_corr, s_corr, tot = 0, 0, 0
    with torch.no_grad():
        for images, cat_targets, stage_targets, _ in tqdm(loader, desc="Оценка"):
            images, cat_targets, stage_targets = images.to(device), cat_targets.to(device), stage_targets.to(device)
            out_cat, out_stage = model(images)
            c_corr += (torch.argmax(out_cat, dim=1) == cat_targets).sum().item()
            s_corr += (torch.argmax(out_stage, dim=1) == stage_targets).sum().item()
            tot += images.size(0)

    cat_acc = c_corr / max(1, tot)
    stg_acc = s_corr / max(1, tot)
    comb_acc = (cat_acc + stg_acc) / 2.0

    print("\n" + "=" * 50)
    print(f"📈 РЕЗУЛЬТАТЫ ОЦЕНКИ {MODEL_ID}:")
    print(f"  • Точность категорий сорняков (Accuracy): {cat_acc:.4f} ({cat_acc*100:.1f}%)")
    print(f"  • Точность стадий развития (Accuracy):   {stg_acc:.4f} ({stg_acc*100:.1f}%)")
    print(f"  • Средняя совместная точность:          {comb_acc:.4f} ({comb_acc*100:.1f}%)")
    print("=" * 50 + "\n")

    metrics = {"category_accuracy": cat_acc, "stage_accuracy": stg_acc, "combined_accuracy": comb_acc}
    out_f = output_dir / "metrics_evaluation.json"
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 Метрики сохранены: {out_f}")
    return metrics


def predict_single(model, img_path, device):
    pil_im = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = tf(pil_im).unsqueeze(0).to(device)
    with torch.no_grad():
        out_cat, out_stage = model(tensor)
        cat_probs = torch.softmax(out_cat, dim=1)[0].cpu().numpy()
        stg_probs = torch.softmax(out_stage, dim=1)[0].cpu().numpy()

    best_cat = int(np.argmax(cat_probs))
    best_stg = int(np.argmax(stg_probs))

    print(f"\n🔍 Результат распознавания GeosAero ({img_path}):")
    print(f"  • Категория сорняка: {GEOSAERO_CATEGORIES.get(best_cat, str(best_cat))} ({cat_probs[best_cat]*100:.1f}%)")
    print(f"  • Стадия развития:  {GEOSAERO_STAGES.get(best_stg, str(best_stg))} ({stg_probs[best_stg]*100:.1f}%)")


def main():
    # Explicit original-dataset evaluation, with reproducible predictions and full metrics.
    if "--dataset" in sys.argv or "--geosaero-eval" in sys.argv or "--eval" in sys.argv or len(sys.argv) == 1:
        from common.geosaero_evaluation import main as evaluate_geosaero
        sys.argv[:] = [arg for arg in sys.argv if arg not in ("--eval", "--geosaero-eval")]
        evaluate_geosaero(MODEL_ID)
        return
    parser = argparse.ArgumentParser(description=f"Инференс и метрики {MODEL_ID}")
    parser.add_argument("--geosaero-eval", action="store_true", help="Полная оценка исходного GeosAero, JSON метрик и предсказания")
    parser.add_argument("--dataset", help="Корень исходного GeosAero; включает полную оценку")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else (ROOT_DIR / "runs" / MODEL_ID / "infer_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (ROOT_DIR.parent / "runs" / MODEL_ID / "best.pt")
    if not ckpt_path.exists():
        fallback = ROOT_DIR.parent / "runs" / MODEL_ID / "best.pt"
        if fallback.exists():
            ckpt_path = fallback

    print(f"🔧 Загрузка {MODEL_ID} на {device}...")
    model = build_resnet50_dual_head(num_classes_category=NUM_CATS, num_classes_stage=NUM_STAGES, in_channels=IN_CHANNELS, pretrained=False).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        st = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(st, strict=True)
        print(f"✅ Веса загружены из: {ckpt_path}")
    else:
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    model.eval()

    if args.eval or args.input is None:
        try:
            master = load_master_dataset()
            samples = get_classification_samples(master, model_id=MODEL_ID)
            if not samples:
                samples = [s for s in master.get("classification_samples", []) if s.get("dataset_id") == "GeosAero_hitech_dataset"]
            run_evaluation(model, samples, device, out_dir)
        except Exception as e:
            raise RuntimeError("Оценка датасета не выполнена") from e

    if args.input:
        predict_single(model, args.input, device)


if __name__ == "__main__":
    main()
