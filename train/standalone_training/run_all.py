#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py - Единый диспетчер запуска обучения, экспорта и инференса всех 10 моделей
в автономном пакете standalone_training.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent

ALL_MODELS = [
    "d01_rgb_plant",
    "d02_ms3_plant",
    "d03_ms4_plant",
    "d04_ms5_plant",
    "d05_ms7_plant",
    "d06_rgb_geosaero_plant",
    "c06_rgb_species",
    "c07_rgb_geosaero",
    "c08_ms3_species",
    "c09_ms4_species"
]


def run_command(cmd, desc):
    print("\n" + "=" * 70)
    print(f"▶ {desc}")
    print(f"Команда: {' '.join(cmd)}")
    print("=" * 70)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"⚠️ Ошибка выполнения: код возврата {res.returncode}")
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Автономный диспетчер запуска моделей AeroWeed 2026")
    parser.add_argument("--action", type=str, choices=["train", "export_onnx", "export_rknn", "eval"], default="train", help="Действие: train, export_onnx, export_rknn, eval")
    parser.add_argument("--only", type=str, choices=ALL_MODELS, default=None, help="Запустить только для одной модели")
    parser.add_argument("--dry-run", action="store_true", help="Быстрый проверочный прогон")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--master-json", type=str, default=str(ROOT_DIR / "master_dataset.json"))
    args = parser.parse_args()

    models_to_run = [args.only] if args.only else ALL_MODELS
    python_bin = sys.executable

    print(f"\n🚀 Запуск диспетчера: действие = {args.action}")
    print(f"Выбранные модели: {', '.join(models_to_run)}")
    print(f"Единый JSON-датасет: {args.master_json}\n")

    for m_id in models_to_run:
        m_dir = ROOT_DIR / m_id
        if args.action == "train":
            script = m_dir / "train.py"
            cmd = [python_bin, str(script), "--master-json", args.master_json]
            if args.dry_run:
                cmd.append("--dry-run")
            if args.epochs:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.batch_size:
                cmd.extend(["--batch-size", str(args.batch_size)])
            run_command(cmd, f"Обучение модели {m_id}")

        elif args.action == "export_onnx":
            script = m_dir / "export_onnx.py"
            cmd = [python_bin, str(script)]
            run_command(cmd, f"Экспорт в ONNX модели {m_id}")

        elif args.action == "export_rknn":
            script = m_dir / "export_rknn.py"
            cmd = [python_bin, str(script)]
            run_command(cmd, f"Экспорт в RKNN INT8 модели {m_id}")

        elif args.action == "eval":
            script = m_dir / "infer.py"
            cmd = [python_bin, str(script), "--eval"]
            run_command(cmd, f"Вычисление метрик для модели {m_id}")

    print("\n✅ Выполнение всех задач завершено!\n")


if __name__ == "__main__":
    main()
