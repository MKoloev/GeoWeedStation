#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d01_rgb_plant/export_rknn.py - Квантование ONNX в RKNN INT8 под NPU Rockchip RK3588 (Orange Pi 5 Pro).

Запуск:
  /home/workerfit/PycharmProjects/AeroWeed_Train/.venv_rknn230/bin/python export_rknn.py
"""

import argparse
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Автоматическое переключение на .venv_rknn230 при наличии
RKNN_PYTHON = ROOT_DIR.parent / ".venv_rknn230" / "bin" / "python"
if RKNN_PYTHON.exists() and not sys.executable.startswith(str(ROOT_DIR.parent / ".venv_rknn230")):
    print(f"🔄 Переключение на окружение RKNN: {RKNN_PYTHON}")
    os.execv(str(RKNN_PYTHON), [str(RKNN_PYTHON)] + sys.argv)

import cv2
import numpy as np

try:
    from rknn.api import RKNN
except ImportError:
    print("❌ rknn-toolkit2 не установлен в текущем окружении!")
    print("Запустите скрипт через .venv_rknn230/bin/python")
    sys.exit(1)

from common.dataset import load_master_dataset, get_detection_samples

MODEL_ID = "d04_ms5_plant"
IN_CHANNELS = 5
RESOLUTION = 640


def prepare_calib_dataset(calib_dir: Path, num_samples: int = 50) -> Path:
    calib_dir.mkdir(parents=True, exist_ok=True)
    txt_path = calib_dir / "dataset.txt"

    samples = []
    try:
        master = load_master_dataset()
        samples = get_detection_samples(master, model_id=MODEL_ID)
    except Exception:
        pass

    paths = []
    for i in range(num_samples):
        img_out = calib_dir / f"calib_{i:03d}.jpg"
        if i < len(samples) and Path(samples[i].get("source_files", [""])[0]).exists():
            orig = cv2.imread(samples[i]["source_files"][0])
            if orig is not None:
                resized = cv2.resize(orig, (RESOLUTION, RESOLUTION))
                cv2.imwrite(str(img_out), resized)
            else:
                cv2.imwrite(str(img_out), np.random.randint(0, 256, (RESOLUTION, RESOLUTION, 3), dtype=np.uint8))
        else:
            cv2.imwrite(str(img_out), np.random.randint(0, 256, (RESOLUTION, RESOLUTION, 3), dtype=np.uint8))
        paths.append(str(img_out.resolve()))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")

    return txt_path


def export_rknn(
    onnx_path = None,
    output_rknn = None,
    target_platform: str = "rk3588",
    num_calib: int = 50
):
    in_onnx = Path(onnx_path) if onnx_path else (ROOT_DIR / "runs" / MODEL_ID / f"{MODEL_ID}_best.onnx")
    if not in_onnx.exists():
        fallback = ROOT_DIR.parent / "runs" / MODEL_ID / f"{MODEL_ID}_best.onnx"
        if fallback.exists():
            in_onnx = fallback

    if not in_onnx.exists():
        raise FileNotFoundError(f"ONNX файл {in_onnx} не найден! Сначала выполните: python export_onnx.py")

    out_p = Path(output_rknn) if output_rknn else (ROOT_DIR / "runs" / MODEL_ID / f"{MODEL_ID}_best_int8.rknn")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Конвертация {MODEL_ID} -> RKNN INT8 ({target_platform})")
    print(f"Входной ONNX: {in_onnx}")
    print(f"Выходной RKNN: {out_p}")
    print("=" * 60)

    calib_txt = prepare_calib_dataset(out_p.parent / "calib", num_samples=num_calib)

    rknn = RKNN(verbose=False)
    # Нормализация [0..1] float32: mean=0, std=255
    rknn.config(
        target_platform=target_platform,
        optimization_level=3,
        mean_values=[[0, 0, 0, 0, 0]],
        std_values=[[255, 255, 255, 255, 255]],
        quantized_dtype="asymmetric_quantized-8"
    )

    ret = rknn.load_onnx(model=str(in_onnx))
    if ret != 0:
        raise RuntimeError("Ошибка загрузки ONNX модели в RKNN!")

    ret = rknn.build(do_quantization=True, dataset=str(calib_txt))
    if ret != 0:
        raise RuntimeError("Ошибка построения квантованной RKNN модели!")

    ret = rknn.export_rknn(str(out_p))
    if ret != 0:
        raise RuntimeError("Ошибка экспорта RKNN файла!")

    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"✅ Успешно экспортирована модель RKNN INT8: {out_p} ({size_mb:.2f} МБ)")
    rknn.release()


def main():
    parser = argparse.ArgumentParser(description=f"Экспорт {MODEL_ID} в RKNN INT8")
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--target", type=str, default="rk3588")
    parser.add_argument("--calib-samples", type=int, default=50)
    args = parser.parse_args()

    export_rknn(onnx_path=args.onnx, output_rknn=args.output, target_platform=args.target, num_calib=args.calib_samples)


if __name__ == "__main__":
    main()
