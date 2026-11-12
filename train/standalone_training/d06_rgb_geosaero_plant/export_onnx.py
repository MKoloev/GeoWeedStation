#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d01_rgb_plant/export_onnx.py - Экспорт RGB-детектора растений YOLOX-M в формат ONNX FP32.
"""

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common.models.yolox import build_yolox_m

MODEL_ID = "d06_rgb_geosaero_plant"
IN_CHANNELS = 3
NUM_CLASSES = 1
RESOLUTION = 640


def export_onnx(
    checkpoint = None,
    output_onnx = None,
    opset: int = 12
):
    ckpt_path = Path(checkpoint) if checkpoint else (ROOT_DIR / "runs" / MODEL_ID / "best.pt")
    if not ckpt_path.exists():
        fallback = ROOT_DIR.parent / "runs" / MODEL_ID / "best.pt"
        if fallback.exists():
            ckpt_path = fallback

    out_p = Path(output_onnx) if output_onnx else (ROOT_DIR / "runs" / MODEL_ID / f"{MODEL_ID}_best.onnx")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"📦 Экспорт {MODEL_ID} в ONNX...")
    print(f"Чекпоинт: {ckpt_path}")
    print(f"Выходной файл: {out_p}")

    model = build_yolox_m(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=False)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        st = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(st, strict=False)
        print("✅ Веса чекпоинта успешно загружены!")
    else:
        print("⚠️ Чекпоинт не найден, экспорт со случайными весами для проверки пайплайна.")

    model.eval()
    dummy_input = torch.randn(1, IN_CHANNELS, RESOLUTION, RESOLUTION)
    output_names = ["out_stride8", "out_stride16", "out_stride32"]

    torch.onnx.export(
        model,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names
    )

    # Проверка структуры ONNX
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)
    print("✅ Граф ONNX успешно верифицирован onnx.checker!")

    # Численная проверка с ONNX Runtime
    session = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_outs = model(dummy_input)
    ort_outs = session.run(None, {"images": dummy_input.numpy()})

    max_err = max([float(np.max(np.abs(p.numpy() - o))) for p, o in zip(pt_outs, ort_outs)])
    print(f"🔍 Максимальное численное расхождение (PyTorch vs ONNX Runtime): {max_err:.2e}")
    assert max_err < 1e-3, f"Слишком большая погрешность: {max_err}"

    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"✅ ONNX-модель готова: {out_p} ({size_mb:.2f} МБ)")


def main():
    parser = argparse.ArgumentParser(description=f"Экспорт {MODEL_ID} в ONNX")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    export_onnx(checkpoint=args.checkpoint, output_onnx=args.output, opset=args.opset)


if __name__ == "__main__":
    main()
