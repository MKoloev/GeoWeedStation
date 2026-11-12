#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_models.py - Генератор скриптов (train, export_onnx, export_rknn, infer)
для всех моделей в standalone_training на основе единых шаблонов.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DET_MODELS = [
    ("d02_ms3_plant", 3, "Детектор растений MS3 [R, G, NIR] (YOLOX-M, 3 канала)"),
    ("d03_ms4_plant", 4, "Детектор растений MS4 [R, G, RE, NIR] (YOLOX-M, 4 канала)"),
    ("d04_ms5_plant", 5, "Детектор растений MS5 [R, G, B, RE, NIR] (YOLOX-M, 5 каналов)"),
    ("d05_ms7_plant", 7, "Детектор растений MS7 [RGB + 4 MS] (YOLOX-M, 7 каналов)"),
    ("d06_rgb_geosaero_plant", 3, "RGB-детектор растений GeosAero (YOLOX-M, 3 канала)"),
]

# Считываем эталонные файлы D01
d01_dir = ROOT / "d01_rgb_plant"
train_d01 = (d01_dir / "train.py").read_text(encoding="utf-8")
onnx_d01 = (d01_dir / "export_onnx.py").read_text(encoding="utf-8")
rknn_d01 = (d01_dir / "export_rknn.py").read_text(encoding="utf-8")
infer_d01 = (d01_dir / "infer.py").read_text(encoding="utf-8")

for model_id, channels, desc in DET_MODELS:
    m_dir = ROOT / model_id
    m_dir.mkdir(parents=True, exist_ok=True)

    # 1. train.py
    t_code = train_d01.replace('MODEL_ID = "d01_rgb_plant"', f'MODEL_ID = "{model_id}"')
    t_code = t_code.replace('IN_CHANNELS = 3', f'IN_CHANNELS = {channels}')
    t_code = t_code.replace('D01-RGB-PLANT', model_id.upper())
    (m_dir / "train.py").write_text(t_code, encoding="utf-8")

    # 2. export_onnx.py
    o_code = onnx_d01.replace('MODEL_ID = "d01_rgb_plant"', f'MODEL_ID = "{model_id}"')
    o_code = o_code.replace('IN_CHANNELS = 3', f'IN_CHANNELS = {channels}')
    (m_dir / "export_onnx.py").write_text(o_code, encoding="utf-8")

    # 3. export_rknn.py
    r_code = rknn_d01.replace('MODEL_ID = "d01_rgb_plant"', f'MODEL_ID = "{model_id}"')
    r_code = r_code.replace('IN_CHANNELS = 3', f'IN_CHANNELS = {channels}')
    if channels != 3:
        r_code = r_code.replace('mean_values=[[0, 0, 0]]', f'mean_values=[{[0]*channels}]')
        r_code = r_code.replace('std_values=[[255, 255, 255]]', f'std_values=[{[255]*channels}]')
    (m_dir / "export_rknn.py").write_text(r_code, encoding="utf-8")

    # 4. infer.py
    i_code = infer_d01.replace('MODEL_ID = "d01_rgb_plant"', f'MODEL_ID = "{model_id}"')
    i_code = i_code.replace('IN_CHANNELS = 3', f'IN_CHANNELS = {channels}')
    (m_dir / "infer.py").write_text(i_code, encoding="utf-8")

    print(f"✅ Созданы скрипты для {model_id} (каналов: {channels})")
