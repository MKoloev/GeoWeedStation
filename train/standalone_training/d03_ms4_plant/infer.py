#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d01_rgb_plant/infer.py - Инференс RGB-детектора растений D01-RGB-PLANT и вычисление метрик точности.

Режимы:
  1. Вычисление метрик на тестовой/валидационной выборке датасета:
     python infer.py --eval [--checkpoint runs/d01_rgb_plant/best.pt]
  2. Инференс на отдельном изображении с отрисовкой детекций:
     python infer.py --input sample.jpg [--checkpoint runs/d01_rgb_plant/best.pt]
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import load_master_dataset, get_detection_samples, PlantPatchDataset, collate_fn_detection
from common.metrics import DetectionEvaluator
from common.models.yolox import build_yolox_m
from common.transforms import SyncResize, SyncToTensor

MODEL_ID = "d03_ms4_plant"
IN_CHANNELS = 4
NUM_CLASSES = 1
RESOLUTION = 640


def decode_and_nms(
    outputs: List[torch.Tensor],
    score_thresh: float = 0.30,
    nms_thresh: float = 0.45,
    strides: Tuple[int, ...] = (8, 16, 32)
) -> Tuple[np.ndarray, np.ndarray]:
    device = outputs[0].device
    all_boxes, all_scores = [], []

    for out, stride in zip(outputs, strides):
        reg = out[0, :4]
        obj = torch.sigmoid(out[0, 4:5])
        cls_prob = torch.sigmoid(out[0, 5:])
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
        all_boxes.append(boxes_k)
        all_scores.append(sub_scores)

    if len(all_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    keep = nms(boxes, scores, nms_thresh)
    return boxes[keep].detach().cpu().numpy(), scores[keep].detach().cpu().numpy()


def run_evaluation(
    model: torch.nn.Module,
    val_samples: List[Dict[str, Any]],
    device: torch.device,
    output_dir: Path
) -> Dict[str, float]:
    print(f"\n📊 Запуск вычисления метрик на {len(val_samples)} сэмплах...")
    val_tf = [SyncResize(size=RESOLUTION), SyncToTensor(bbox_padding=0, plant_label=0)]
    val_ds = PlantPatchDataset(val_samples, resolution=RESOLUTION, transforms=val_tf, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn_detection)

    evaluator = DetectionEvaluator(score_threshold=0.25)
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Оценка метрик"):
            images = images.to(device)
            outputs = model(images)

            # Декодирование батча
            batch_size = images.shape[0]
            for b_idx in range(batch_size):
                b_outs = [out[b_idx:b_idx+1] for out in outputs]
                boxes, scores = decode_and_nms(b_outs, score_thresh=0.25, nms_thresh=0.50)
                pred_dict = {
                    'boxes': torch.from_numpy(boxes).to(device) if len(boxes) > 0 else torch.empty((0, 4), device=device),
                    'scores': torch.from_numpy(scores).to(device) if len(scores) > 0 else torch.empty((0,), device=device),
                    'labels': torch.zeros(len(scores), dtype=torch.int64, device=device)
                }
                evaluator.update([pred_dict], [targets[b_idx]])

    metrics = evaluator.compute()
    print("\n" + "=" * 50)
    print(f"📈 РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ {MODEL_ID}:")
    print(f"  • AP50:      {metrics.get('ap50', 0.0):.4f} ({metrics.get('ap50', 0.0)*100:.1f}%)")
    print(f"  • mAP50:95:  {metrics.get('map50_95', 0.0):.4f}")
    print(f"  • Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"  • Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"  • F1-Score:  {metrics.get('f1', 0.0):.4f}")
    print("=" * 50 + "\n")

    out_metrics_file = output_dir / "metrics_evaluation.json"
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 Метрики сохранены в: {out_metrics_file}")
    return metrics


def run_single_inference(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    output_dir: Path,
    score_thresh: float = 0.30
):
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Файл {image_path} не найден!")

    h_orig, w_orig = orig_bgr.shape[:2]
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (RESOLUTION, RESOLUTION))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        boxes_640, scores = decode_and_nms(outputs, score_thresh=score_thresh)

    rx = w_orig / float(RESOLUTION)
    ry = h_orig / float(RESOLUTION)

    vis = orig_bgr.copy()
    for box, score in zip(boxes_640, scores):
        x1 = int(round(box[0] * rx))
        y1 = int(round(box[1] * ry))
        x2 = int(round(box[2] * rx))
        y2 = int(round(box[3] * ry))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(vis, f"plant: {score:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)

    out_file = output_dir / f"pred_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_file), vis)
    print(f"✅ Обнаружено {len(boxes_640)} растений. Результат сохранен: {out_file}")


def main():
    parser = argparse.ArgumentParser(description=f"Инференс и метрики {MODEL_ID}")
    parser.add_argument("--checkpoint", type=str, default=None, help="Путь к весам best.pt")
    parser.add_argument("--eval", action="store_true", help="Вычислить метрики на валидационном датасете")
    parser.add_argument("--input", type=str, default=None, help="Путь к изображению для инференса")
    parser.add_argument("--score-thresh", type=float, default=0.30)
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

    print(f"🔧 Загрузка модели {MODEL_ID} на {device}...")
    model = build_yolox_m(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=False).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        st = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(st, strict=False)
        print(f"✅ Веса успешно загружены из: {ckpt_path}")
    else:
        print("⚠️ Чекпоинт не найден, запуск со случайными весами.")

    model.eval()

    if args.eval or args.input is None:
        try:
            master = load_master_dataset()
            val_samples = get_detection_samples(master, model_id=MODEL_ID, split="val")
            if not val_samples:
                val_samples = get_detection_samples(master, model_id=MODEL_ID, split="train")[:20]
            run_evaluation(model, val_samples, device, out_dir)
        except Exception as e:
            print(f"⚠️ Ошибка при оценке датасета: {e}")

    if args.input:
        run_single_inference(model, args.input, device, out_dir, score_thresh=args.score_thresh)


if __name__ == "__main__":
    main()
