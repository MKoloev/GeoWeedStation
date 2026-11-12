"""
standalone_training/common/dataset.py - Загрузчик данных на основе единого JSON-датасета
(master_dataset.json) для задач детекции и видовой классификации.

Полностью автономный модуль без зависимостей от внешнего проекта.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from common.transforms import SyncCompose, n_e_augs_shuffle

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASTER_JSON = PACKAGE_ROOT / "master_dataset.json"


def load_master_dataset(json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Загружает единый мастер-датасет из JSON."""
    p = Path(json_path) if json_path else DEFAULT_MASTER_JSON
    if not p.exists():
        raise FileNotFoundError(f"Файл единого датасета не найден: {p}. Запустите build_unified_dataset.py!")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_detection_samples(
    master_data: Dict[str, Any],
    model_id: str,
    split: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Извлекает выборку сэмплов детекции для конкретной модели (D01-D06)
    с фильтрацией по сплиту (train / val / test).
    """
    all_det = master_data.get("detection_samples", [])
    filtered = []
    for s in all_det:
        # Проверяем вхождение модели в applicable_models
        models = s.get("applicable_models", [])
        if model_id in models:
            if split is None or s.get("assigned_split") == split:
                filtered.append(s)
    return filtered


def get_classification_samples(
    master_data: Dict[str, Any],
    model_id: str,
    split: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Извлекает выборку сэмплов классификации для конкретной модели (C06-C09)
    с фильтрацией по сплиту (train / val).
    """
    all_cls = master_data.get("classification_samples", [])
    filtered = []
    for s in all_cls:
        models = s.get("applicable_models", [])
        if model_id in models:
            if split is None or s.get("assigned_split") == split:
                filtered.append(s)
    return filtered


class PlantPatchDataset(Dataset):
    """
    Датасет тайлов 640x640 для обучения детектора растений YOLOX-M.
    Работает с единым JSON-датасетом master_dataset.json.
    
    Поддерживает:
      - 3 канала RGB
      - 3 канала MS3 [MS_R, MS_G, MS_NIR]
      - 4 канала MS4 [MS_R, MS_G, MS_RE, MS_NIR]
      - 5 каналов MS5 [R, G, B, RE, NIR]
      - 7 каналов MS7 [RGB_R, RGB_G, RGB_B, MS_R, MS_G, MS_RE, MS_NIR]
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        resolution: int = 640,
        transforms = None,
        is_train: bool = True,
        min_box_size: float = 6.0
    ):
        self.samples = samples
        self.resolution = resolution
        self.transforms = transforms
        self.is_train = is_train
        self.min_box_size = min_box_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        meta = self.samples[idx]
        source_files = meta.get('source_files', [])
        if not source_files and 'image_path' in meta:
            source_files = [meta['image_path']]

        scale_factor = float(meta.get('scale_factor', 1.0))
        raw_boxes = meta.get('boxes_xyxy', meta.get('boxes', []))
        is_partial = meta.get('is_partial', False) or (meta.get('annotation_completeness') == 'partial_plant')
        dataset_id = meta.get('dataset_id', '')

        orig_h, orig_w = None, None

        # 1. Чтение изображения
        is_hybrid_rgb_ms = (len(source_files) == 3 and len(meta.get('input_bands', [])) == 5)
        is_hybrid_rgb_ms7 = (len(source_files) == 5 and len(meta.get('input_bands', [])) == 7)

        if len(source_files) == 1:
            img = cv2.imread(str(source_files[0]))
            if img is None:
                try:
                    pil_im = Image.open(source_files[0]).convert('RGB')
                    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    # Если файл недоступен (например, отключен внешний диск), создаем заглушку
                    img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
        elif is_hybrid_rgb_ms:
            rgb_arr = cv2.imread(str(source_files[0]))
            if rgb_arr is None:
                rgb_arr = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            else:
                rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = rgb_arr.shape[:2]

            re_arr = cv2.imread(str(source_files[1]), cv2.IMREAD_UNCHANGED)
            if re_arr is None:
                re_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if len(re_arr.shape) == 3:
                re_arr = re_arr[:, :, 0]
            re_u8 = (re_arr.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8) if re_arr.dtype == np.uint16 else re_arr.astype(np.uint8)

            nir_arr = cv2.imread(str(source_files[2]), cv2.IMREAD_UNCHANGED)
            if nir_arr is None:
                nir_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if len(nir_arr.shape) == 3:
                nir_arr = nir_arr[:, :, 0]
            nir_u8 = (nir_arr.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8) if nir_arr.dtype == np.uint16 else nir_arr.astype(np.uint8)

            if re_u8.shape[:2] == (orig_h, orig_w):
                img = np.dstack([rgb_arr, re_u8, nir_u8])
            else:
                re_u8 = cv2.resize(re_u8, (orig_w, orig_h))
                nir_u8 = cv2.resize(nir_u8, (orig_w, orig_h))
                img = np.dstack([rgb_arr, re_u8, nir_u8])
        elif is_hybrid_rgb_ms7:
            rgb_arr = cv2.imread(str(source_files[0]))
            if rgb_arr is None:
                rgb_arr = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            else:
                rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = rgb_arr.shape[:2]

            ms_bands = []
            for bf in source_files[1:]:
                b_arr = cv2.imread(str(bf), cv2.IMREAD_UNCHANGED)
                if b_arr is None:
                    b_u8 = np.zeros((orig_h, orig_w), dtype=np.uint8)
                else:
                    if len(b_arr.shape) == 3:
                        b_arr = b_arr[:, :, 0]
                    b_u8 = (b_arr.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8) if b_arr.dtype == np.uint16 else b_arr.astype(np.uint8)
                    if b_u8.shape[:2] != (orig_h, orig_w):
                        b_u8 = cv2.resize(b_u8, (orig_w, orig_h))
                ms_bands.append(b_u8)
            img = np.dstack([rgb_arr] + ms_bands)
        else:
            bands = []
            for bf in source_files:
                b_arr = cv2.imread(str(bf), cv2.IMREAD_UNCHANGED)
                if b_arr is None:
                    b_f = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
                else:
                    if len(b_arr.shape) == 3:
                        b_arr = b_arr[:, :, 0]
                    b_f = (b_arr.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8) if b_arr.dtype == np.uint16 else b_arr.astype(np.uint8)
                bands.append(b_f)
            img = np.stack(bands, axis=-1)
            orig_h, orig_w = img.shape[:2]

        win_size = int(round(self.resolution / max(scale_factor, 0.01)))

        # 2. Вырезка или ресайз
        if win_size >= max(orig_w, orig_h) or (orig_w <= self.resolution and orig_h <= self.resolution and scale_factor <= 1.0):
            new_w = max(16, int(round(orig_w * scale_factor)))
            new_h = max(16, int(round(orig_h * scale_factor)))

            if new_w > self.resolution or new_h > self.resolution:
                scale_fit = min(self.resolution / float(new_w), self.resolution / float(new_h))
                new_w = int(round(new_w * scale_fit))
                new_h = int(round(new_h * scale_fit))
                eff_scale_x = new_w / float(orig_w)
                eff_scale_y = new_h / float(orig_h)
            else:
                eff_scale_x = scale_factor
                eff_scale_y = scale_factor

            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pad_w = self.resolution - new_w
            pad_h = self.resolution - new_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            final_img = np.pad(
                scaled_img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )

            final_boxes = []
            for b in raw_boxes:
                bx1 = b[0] * eff_scale_x + pad_left
                by1 = b[1] * eff_scale_y + pad_top
                bx2 = b[2] * eff_scale_x + pad_left
                by2 = b[3] * eff_scale_y + pad_top
                if bx2 - bx1 >= self.min_box_size and by2 - by1 >= self.min_box_size:
                    final_boxes.append([bx1, by1, bx2, by2])
        else:
            crop_w = min(win_size, orig_w)
            crop_h = min(win_size, orig_h)

            if len(raw_boxes) > 0 and self.is_train:
                target_box = raw_boxes[np.random.randint(len(raw_boxes))]
                bcx = (target_box[0] + target_box[2]) / 2.0
                bcy = (target_box[1] + target_box[3]) / 2.0
                x0 = int(round(bcx - crop_w / 2.0))
                y0 = int(round(bcy - crop_h / 2.0))
            else:
                x0 = (orig_w - crop_w) // 2
                y0 = (orig_h - crop_h) // 2

            x0 = max(0, min(orig_w - crop_w, x0))
            y0 = max(0, min(orig_h - crop_h, y0))
            x1 = x0 + crop_w
            y1 = y0 + crop_h

            crop = img[y0:y1, x0:x1]
            final_img = cv2.resize(crop, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

            rx = self.resolution / float(crop_w)
            ry = self.resolution / float(crop_h)

            final_boxes = []
            for b in raw_boxes:
                inter_x1 = max(float(x0), b[0])
                inter_y1 = max(float(y0), b[1])
                inter_x2 = min(float(x1), b[2])
                inter_y2 = min(float(y1), b[3])

                if inter_x2 - inter_x1 >= self.min_box_size and inter_y2 - inter_y1 >= self.min_box_size:
                    nx1 = (inter_x1 - x0) * rx
                    ny1 = (inter_y1 - y0) * ry
                    nx2 = (inter_x2 - x0) * rx
                    ny2 = (inter_y2 - y0) * ry
                    if nx2 - nx1 >= self.min_box_size and ny2 - ny1 >= self.min_box_size:
                        final_boxes.append([nx1, ny1, nx2, ny2])

        # 3. Формирование instance_map и target
        instance_map = np.zeros((self.resolution, self.resolution), dtype=np.uint16)
        for i, b in enumerate(final_boxes):
            bx1 = int(round(max(0, min(self.resolution - 1, b[0]))))
            by1 = int(round(max(0, min(self.resolution - 1, b[1]))))
            bx2 = int(round(max(0, min(self.resolution, b[2]))))
            by2 = int(round(max(0, min(self.resolution, b[3]))))
            if bx2 > bx1 and by2 > by1:
                instance_map[by1:by2, bx1:bx2] = (i + 1)

        target = {
            'instance_map': instance_map,
            'boxes': np.array(final_boxes, dtype=np.float32) if len(final_boxes) > 0 else np.empty((0, 4), dtype=np.float32),
            'is_partial': is_partial,
            'dataset_id': dataset_id,
            'sample_id': meta.get('sample_id', str(idx))
        }

        # 4. Аугментации
        if final_img.shape[-1] <= 4:
            pil_img = Image.fromarray(final_img)
            if self.transforms is not None:
                if self.is_train and isinstance(self.transforms, list):
                    aug_pipeline = SyncCompose(n_e_augs_shuffle(list(self.transforms)))
                elif isinstance(self.transforms, SyncCompose):
                    aug_pipeline = self.transforms
                else:
                    aug_pipeline = SyncCompose(list(self.transforms))
                pil_img, target = aug_pipeline(pil_img, target)

            if not isinstance(pil_img, torch.Tensor):
                img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = pil_img
        else:
            if self.transforms is not None:
                if self.is_train and isinstance(self.transforms, list):
                    aug_pipeline = SyncCompose(n_e_augs_shuffle(list(self.transforms)))
                elif isinstance(self.transforms, SyncCompose):
                    aug_pipeline = self.transforms
                else:
                    aug_pipeline = SyncCompose(list(self.transforms))
                img_aug, target = aug_pipeline(final_img, target)
            else:
                img_aug = final_img

            if isinstance(img_aug, torch.Tensor):
                img_tensor = img_aug
            elif isinstance(img_aug, np.ndarray):
                img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = torch.from_numpy(np.array(img_aug)).permute(2, 0, 1).float() / 255.0

        if 'boxes' not in target:
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            target['labels'] = torch.empty((0,), dtype=torch.int64)
        else:
            if not isinstance(target['boxes'], torch.Tensor):
                b_arr = np.array(target['boxes'], dtype=np.float32)
                target['boxes'] = torch.from_numpy(b_arr).float() if len(b_arr) > 0 else torch.empty((0, 4), dtype=torch.float32)
            if 'labels' not in target:
                target['labels'] = torch.zeros(len(target['boxes']), dtype=torch.int64)

        return img_tensor, target


def collate_fn_detection(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


class PlantCropDataset(Dataset):
    """Датасет кропов для классификации видов (C06, C08, C09)."""
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        species_to_idx: Optional[Dict[str, int]] = None,
        input_size: Tuple[int, int] = (224, 224),
        transforms = None,
        is_train: bool = True
    ):
        self.samples = samples
        self.species_to_idx = species_to_idx or {}
        self.input_size = input_size
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        meta = self.samples[idx]
        img_path = meta.get("image_path", "")
        source_files = meta.get("source_files", [])
        crop_box = meta.get("crop_box", None)
        crop_file = meta.get("crop_file", None)

        if crop_file and Path(crop_file).exists():
            pil_img = Image.open(crop_file)
            if pil_img.mode not in ("RGB", "RGBA"):
                pil_img = pil_img.convert("RGB")
        elif source_files and len(source_files) in (3, 4):
            bands = []
            for bf in source_files:
                b_arr = cv2.imread(str(bf), cv2.IMREAD_GRAYSCALE)
                if b_arr is None:
                    b_arr = np.zeros(self.input_size, dtype=np.uint8)
                bands.append(b_arr)
            full_img = cv2.merge(bands)
            if crop_box is not None:
                h, w = full_img.shape[:2]
                x1, y1, x2, y2 = [int(round(coord)) for coord in crop_box]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 1, min(w, x2))
                y2 = max(y1 + 1, min(h, y2))
                full_img = full_img[y1:y2, x1:x2]
            mode = "RGBA" if len(source_files) == 4 else "RGB"
            pil_img = Image.fromarray(full_img, mode=mode)
        elif crop_box is not None and img_path and Path(img_path).exists():
            full_img = cv2.imread(str(img_path))
            if full_img is None:
                full_img = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            else:
                full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                h, w = full_img.shape[:2]
                x1, y1, x2, y2 = [int(round(coord)) for coord in crop_box]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 1, min(w, x2))
                y2 = max(y1 + 1, min(h, y2))
                full_img = full_img[y1:y2, x1:x2]
            pil_img = Image.fromarray(full_img)
        else:
            if img_path and Path(img_path).exists():
                pil_img = Image.open(img_path).convert("RGB")
            else:
                pil_img = Image.new("RGB", self.input_size, (0, 0, 0))

        if self.transforms is not None:
            tensor = self.transforms(pil_img)
        else:
            pil_img = pil_img.resize(self.input_size, Image.BILINEAR)
            tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        sp_id = meta.get("species_id", "")
        label_idx = self.species_to_idx.get(sp_id, meta.get("label_idx", 0))

        return tensor, label_idx, meta


def collate_fn_classification(batch):
    tensors = []
    labels = []
    metas = []
    for tensor, label_idx, meta in batch:
        tensors.append(tensor)
        labels.append(label_idx)
        metas.append(meta)
    return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.int64), metas


class GeosAeroDualHeadDataset(Dataset):
    """Датасет для двухзадачной классификации C07-RGB-GEOSAERO."""
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        category_to_idx: Dict[str, int],
        stage_to_idx: Dict[str, int],
        input_size: Tuple[int, int] = (224, 224),
        transforms = None,
        is_train: bool = True
    ):
        self.samples = samples
        self.category_to_idx = category_to_idx
        self.stage_to_idx = stage_to_idx
        self.input_size = input_size
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, Dict[str, Any]]:
        meta = self.samples[idx]
        crop_file = meta.get("crop_file")
        img_path = meta.get("image_path")
        crop_box = meta.get("crop_box")

        if crop_file and Path(crop_file).exists():
            pil_img = Image.open(crop_file).convert("RGB")
        elif img_path and Path(img_path).exists():
            full_bgr = cv2.imread(str(img_path))
            if full_bgr is None:
                full_bgr = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            h, w = full_bgr.shape[:2]
            if crop_box:
                x1, y1, x2, y2 = [int(round(c)) for c in crop_box]
                x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
                crop_bgr = full_bgr[y1:y2, x1:x2]
            else:
                crop_bgr = full_bgr
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
        else:
            pil_img = Image.new("RGB", self.input_size, (0, 0, 0))

        if self.transforms is not None:
            tensor = self.transforms(pil_img)
        else:
            pil_img = pil_img.resize(self.input_size, Image.BILINEAR)
            tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        cat_name = meta.get("category_name", "background")
        stg_name = meta.get("stage_name", "background")
        cat_idx = self.category_to_idx.get(cat_name, meta.get("category_idx", 0))
        stg_idx = self.stage_to_idx.get(stg_name, meta.get("stage_idx", 0))

        return tensor, cat_idx, stg_idx, meta


def collate_fn_dual_head(batch):
    tensors = []
    cat_labels = []
    stg_labels = []
    metas = []
    for tensor, cat_idx, stg_idx, meta in batch:
        tensors.append(tensor)
        cat_labels.append(cat_idx)
        stg_labels.append(stg_idx)
        metas.append(meta)
    return (
        torch.stack(tensors, dim=0),
        torch.tensor(cat_labels, dtype=torch.int64),
        torch.tensor(stg_labels, dtype=torch.int64),
        metas
    )
