#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c06_rgb_species/infer.py - Инференс классификатора ботанических видов C06-RGB-SPECIES
и вычисление метрик (Accuracy, Macro F1, Precision, Recall).
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

from common.dataset import load_master_dataset, get_classification_samples, PlantCropDataset, collate_fn_classification
from common.metrics import ClassificationEvaluator
from common.models.resnet import build_resnet50

MODEL_ID = "c06_rgb_species"
IN_CHANNELS = 3
NUM_CLASSES = 24
RESOLUTION = 224

SPECIES_TAXONOMY = {0: {'id': 'S000', 'latin': 'Background', 'ru': 'Фон / Не растение'},
 1: {'id': 'S001', 'latin': 'Zea mays', 'ru': 'Кукуруза сахарная'},
 2: {'id': 'S002', 'latin': 'Lolium rigidum', 'ru': 'Плевел жесткий'},
 3: {'id': 'S003', 'latin': 'Sorghum halepense', 'ru': 'Гумай (сорго алеппское)'},
 4: {'id': 'S004', 'latin': 'Avena fatua', 'ru': 'Овсюг обыкновенный'},
 5: {'id': 'S005', 'latin': 'Hordeum vulgare', 'ru': 'Ячмень обыкновенный'},
 6: {'id': 'S006', 'latin': 'Atriplex patula', 'ru': 'Лебеда раскидистая'},
 7: {'id': 'S007', 'latin': 'Chenopodium album', 'ru': 'Марь белая'},
 8: {'id': 'S008', 'latin': 'Salsola kali', 'ru': 'Солянка русская (курай)'},
 9: {'id': 'S009', 'latin': 'Beta vulgaris', 'ru': 'Свёкла обыкновенная'},
 10: {'id': 'S010', 'latin': 'Matricaria discoidea', 'ru': 'Ромашка пахучая (безъязычковая)'},
 11: {'id': 'S011', 'latin': 'Onopordum acanthium', 'ru': 'Татарник колючий'},
 12: {'id': 'S012', 'latin': 'Silybum marianum', 'ru': 'Расторопша пятнистая'},
 13: {'id': 'S013', 'latin': 'Convolvulus arvensis', 'ru': 'Вьюнок полевой (берёзка)'},
 14: {'id': 'S014', 'latin': 'Malva sylvestris', 'ru': 'Мальва лесная (просвирник)'},
 15: {'id': 'S015', 'latin': 'Gossypium hirsutum', 'ru': 'Хлопчатник обыкновенный'},
 16: {'id': 'S016', 'latin': 'Sinapis arvensis', 'ru': 'Горчица полевая'},
 17: {'id': 'S017', 'latin': 'Cyperus rotundus', 'ru': 'Сыть круглая'},
 18: {'id': 'S018', 'latin': 'Solanum lycopersicum', 'ru': 'Томат обыкновенный'},
 19: {'id': 'S019', 'latin': 'Solanum nigrum', 'ru': 'Паслён чёрный'},
 20: {'id': 'S020', 'latin': 'Solanum elaeagnifolium', 'ru': 'Паслён линейнолистный'},
 21: {'id': 'S021', 'latin': 'Datura ferox', 'ru': 'Дурман свирепый'},
 22: {'id': 'S022', 'latin': 'Datura stramonium', 'ru': 'Дурман обыкновенный'},
 23: {'id': 'S023', 'latin': 'Portulaca oleracea', 'ru': 'Портулак огородный'}}


def run_evaluation(model, val_samples, device, output_dir):
    print(f"\n📊 Оценка классификатора видов на {len(val_samples)} кропах...")
    eval_tf = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    species_to_idx = {f"S{i:03d}": i for i in range(NUM_CLASSES)}
    val_ds = PlantCropDataset(val_samples, species_to_idx=species_to_idx, input_size=(RESOLUTION, RESOLUTION), transforms=eval_tf, is_train=False)
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


def predict_single(model, img_path, device):
    pil_im = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = tf(pil_im).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    conf = float(probs[top_idx])
    sp_info = SPECIES_TAXONOMY.get(top_idx, {"id": f"S{top_idx:03d}", "latin": "Unknown", "ru": "Неизвестный"})

    print(f"\n🔍 Результат классификации ({img_path}):")
    print(f"  • Вид: {sp_info['id']} — {sp_info['latin']} ({sp_info['ru']})")
    print(f"  • Уверенность (вероятность): {conf*100:.2f}%")


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
    model = build_resnet50(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, pretrained=False).to(device)

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
            val_samples = get_classification_samples(master, model_id=MODEL_ID, split="val")
            if not val_samples:
                val_samples = get_classification_samples(master, model_id=MODEL_ID, split="train")[:30]
            run_evaluation(model, val_samples, device, out_dir)
        except Exception as e:
            raise RuntimeError("Оценка датасета не выполнена") from e

    if args.input:
        predict_single(model, args.input, device)


if __name__ == "__main__":
    main()
