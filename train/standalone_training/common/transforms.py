import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode, ToTensor
from skimage.util import random_noise
import cv2
import albumentations as A

imodes = [
    InterpolationMode.NEAREST,
    InterpolationMode.BILINEAR,
    InterpolationMode.BICUBIC,
    InterpolationMode.BOX,
    InterpolationMode.HAMMING,
    InterpolationMode.LANCZOS,
]


def n_e_augs_shuffle(augs_list, n=-4, skip_first=False):
    """Случайное перемешивание порядка аугментаций как в шаблоне."""
    if len(augs_list) > 3:
        s = 1 if skip_first else 0
        first_n = augs_list[s:n]
        random.shuffle(first_n)
        augs_list[s:n] = first_n
    return augs_list


class SyncCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None, **kwargs):
        for t in self.transforms:
            if hasattr(t, 'target') or 'target' in t.__call__.__code__.co_varnames:
                img, target = t(img, target, **kwargs)
            else:
                img = t(img, **kwargs)
        return img, target


class SyncRandomBrightnessContrastTarget:
    """
    Бережная коррекция яркости и контраста, не искажающая естественный цвет хлорофилла
    (по согласованию с пользователем).
    """
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img: PILImage.Image, target: Dict[str, Any], **kwargs) -> tuple:
        if random.random() > self.p:
            return img, target

        if random.random() > 0.5:
            jittered_img = self.jitter(img)
            return jittered_img, target

        enhanced_img = F.adjust_brightness(img, 1.0 + random.uniform(-self.brightness, self.brightness))
        enhanced_img = F.adjust_contrast(enhanced_img, 1.0 + random.uniform(-self.contrast, self.contrast))
        return enhanced_img, target


class SyncRandomHorizontalFlipTarget:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, **kwargs):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                w = img.shape[1]
                img = np.fliplr(img).copy()
            else:
                w = img.size[0]
                img = F.hflip(img)
            if target is not None:
                if 'instance_map' in target:
                    target['instance_map'] = np.fliplr(target['instance_map']).copy()
                if 'ignore_mask' in target and target['ignore_mask'] is not None:
                    target['ignore_mask'] = np.fliplr(target['ignore_mask']).copy()
                if 'boxes' in target and len(target['boxes']) > 0:
                    boxes = np.array(target['boxes'], dtype=np.float32).copy()
                    x1 = boxes[:, 0].copy()
                    x2 = boxes[:, 2].copy()
                    boxes[:, 0] = w - x2
                    boxes[:, 2] = w - x1
                    target['boxes'] = boxes
        return img, target


class SyncRandomVerticalFlipTarget:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, **kwargs):
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                h = img.shape[0]
                img = np.flipud(img).copy()
            else:
                h = img.size[1]
                img = F.vflip(img)
            if target is not None:
                if 'instance_map' in target:
                    target['instance_map'] = np.flipud(target['instance_map']).copy()
                if 'ignore_mask' in target and target['ignore_mask'] is not None:
                    target['ignore_mask'] = np.flipud(target['ignore_mask']).copy()
                if 'boxes' in target and len(target['boxes']) > 0:
                    boxes = np.array(target['boxes'], dtype=np.float32).copy()
                    y1 = boxes[:, 1].copy()
                    y2 = boxes[:, 3].copy()
                    boxes[:, 1] = h - y2
                    boxes[:, 3] = h - y1
                    target['boxes'] = boxes
        return img, target


class SyncRotateTarget:
    def __init__(self, p=0.8, angle_range=(-45, 45), resolution=640, expand_prob=0.0,
                 img_fill=0, mask_fill=0):
        self.p = p
        self.angle_range = angle_range
        self.resolution = resolution
        self.expand_prob = expand_prob
        self.img_fill = img_fill
        self.mask_fill = mask_fill

    def __call__(self, img, target: Dict[str, Any], **kwargs):
        if random.random() > self.p:
            return img, target

        angle = random.uniform(*self.angle_range)
        expand = random.random() < self.expand_prob

        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.img_fill)
            if target is not None and 'instance_map' in target:
                target['instance_map'] = cv2.warpAffine(target['instance_map'].astype(np.uint16), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.mask_fill)
            if target is not None and 'ignore_mask' in target and target['ignore_mask'] is not None:
                target['ignore_mask'] = cv2.warpAffine(target['ignore_mask'].astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        else:
            w, h = img.size
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            rotated_img = F.rotate(
                img,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=expand,
                fill=self.img_fill,
                center=None
            )
            if target is not None and 'instance_map' in target:
                mask_pil = PILImage.fromarray(target['instance_map'].astype(np.int32), mode='I')
                rotated_mask_pil = F.rotate(
                    mask_pil,
                    angle=angle,
                    expand=expand,
                    interpolation=InterpolationMode.NEAREST,
                    fill=self.mask_fill
                )
                target['instance_map'] = np.array(rotated_mask_pil).astype(np.uint16)

            if target is not None and 'ignore_mask' in target and target['ignore_mask'] is not None:
                ign_pil = PILImage.fromarray(target['ignore_mask'].astype(np.uint8), mode='L')
                rot_ign = F.rotate(
                    ign_pil,
                    angle=angle,
                    expand=expand,
                    interpolation=InterpolationMode.NEAREST,
                    fill=255
                )
                target['ignore_mask'] = np.array(rot_ign).astype(np.uint8)

        # Геометрический пересчет прямоугольных рамок с сохранением целостности объектов
        if target is not None and 'boxes' in target and len(target['boxes']) > 0:
            boxes = np.array(target['boxes'], dtype=np.float32)
            rot_boxes = []
            for b in boxes:
                # 4 угла исходной рамки
                corners = np.array([
                    [b[0], b[1], 1.0],
                    [b[2], b[1], 1.0],
                    [b[2], b[3], 1.0],
                    [b[0], b[3], 1.0]
                ], dtype=np.float32)
                transformed = corners @ M.T
                rx1 = float(np.clip(transformed[:, 0].min(), 0, w))
                ry1 = float(np.clip(transformed[:, 1].min(), 0, h))
                rx2 = float(np.clip(transformed[:, 0].max(), 0, w))
                ry2 = float(np.clip(transformed[:, 1].max(), 0, h))
                if rx2 > rx1 + 1.0 and ry2 > ry1 + 1.0:
                    rot_boxes.append([rx1, ry1, rx2, ry2])
            target['boxes'] = np.array(rot_boxes, dtype=np.float32) if len(rot_boxes) > 0 else np.empty((0, 4), dtype=np.float32)

        return rotated_img, target


class SyncRandomCropTarget:
    def __init__(self, resolution=640, p=1.0):
        self.resolution = resolution
        self.p = p

    def __call__(self, img, target: Dict[str, Any], **kwargs):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        else:
            w, h = img.size

        if (w == self.resolution and h == self.resolution) or (w <= self.resolution and h <= self.resolution):
            return img, target
        if random.random() > self.p:
            return img, target

        max_x = max(0, w - self.resolution)
        max_y = max(0, h - self.resolution)

        left = random.randint(0, max_x) if max_x > 0 else 0
        top = random.randint(0, max_y) if max_y > 0 else 0
        right = left + self.resolution
        bottom = top + self.resolution

        if isinstance(img, np.ndarray):
            cropped_img = img[top:bottom, left:right].copy()
        else:
            cropped_img = img.crop((left, top, right, bottom))

        if target is not None and 'instance_map' in target:
            target['instance_map'] = target['instance_map'][top:bottom, left:right].copy()
        if target is not None and 'ignore_mask' in target and target['ignore_mask'] is not None:
            target['ignore_mask'] = target['ignore_mask'][top:bottom, left:right].copy()

        # Геометрический пересчет прямоугольных рамок при кропе
        if target is not None and 'boxes' in target and len(target['boxes']) > 0:
            boxes = np.array(target['boxes'], dtype=np.float32)
            crop_boxes = []
            for b in boxes:
                ix1 = max(float(left), float(b[0]))
                iy1 = max(float(top), float(b[1]))
                ix2 = min(float(right), float(b[2]))
                iy2 = min(float(bottom), float(b[3]))
                if ix2 > ix1 + 1.0 and iy2 > iy1 + 1.0:
                    crop_boxes.append([ix1 - left, iy1 - top, ix2 - left, iy2 - top])
            target['boxes'] = np.array(crop_boxes, dtype=np.float32) if len(crop_boxes) > 0 else np.empty((0, 4), dtype=np.float32)

        return cropped_img, target


class TrickyResize_UpDwn:
    """
    Имитация изменения оптического разрешения и вариаций GSD БПЛА без изменения размера кадра
    (по согласованию с пользователем).
    """
    def __init__(self, resolution=640, minmax_size_up=[117, 200]):
        self.resolution = resolution
        self.minmax_size_up = minmax_size_up

    def __call__(self, img, target: Dict[str, Any], **kwargs):
        if np.random.rand() < 0.8:
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                up_l = np.random.randint(self.minmax_size_up[0], self.minmax_size_up[1])
                rand_w, rand_h = int((w * up_l) / 100), int((h * up_l) / 100)
                img_up = cv2.resize(img, (rand_w, rand_h), interpolation=cv2.INTER_LINEAR)
                img = cv2.resize(img_up, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                size_down = np.array([img.size[1], img.size[0]])
                up_l = np.random.randint(self.minmax_size_up[0], self.minmax_size_up[1])
                rand_size_up = np.array([int((size_down[0] * up_l) / 100), int((size_down[1] * up_l) / 100)])

                interpolation = random.choice(imodes)
                img = F.resize(img, tuple(rand_size_up), interpolation=interpolation, antialias=True)
                interpolation = random.choice(imodes)
                img = F.resize(img, tuple(size_down), interpolation=interpolation, antialias=True)
        return img, target


class AffineAugmentation:
    """Аффинные трансформации с расширенным диапазоном масштабов (scale: 0.35..2.0) и фильтром min_box_size."""
    def __init__(
        self,
        p=0.75,
        p_affine=1.0,
        pad_to_original=True,
        pad_border_mode=cv2.BORDER_CONSTANT,
        pad_value=0,
        rotate=(-15, 15),
        scale=(0.35, 2.0),
        translate_percent=(-0.05, 0.05),
        shear=(-5, 5),
        min_box_size=6.0
    ):
        self.affine = A.Affine(
            p=p_affine,
            rotate=rotate,
            scale=scale,
            translate_percent=translate_percent,
            shear=shear
        )
        self.pad_to_original = pad_to_original
        self.pad_border_mode = pad_border_mode
        self.pad_value = pad_value
        self.p = p
        self.min_box_size = float(min_box_size)

    def __call__(self, img_pil, target: Dict[str, Any], **kwargs):
        if random.random() > self.p:
            return img_pil, target

        is_np = isinstance(img_pil, np.ndarray)
        img_np = img_pil if is_np else np.array(img_pil)
        h0, w0 = img_np.shape[:2]

        transforms = [self.affine]
        if self.pad_to_original:
            transforms += [
                A.PadIfNeeded(
                    min_height=h0,
                    min_width=w0,
                    border_mode=self.pad_border_mode,
                    fill=self.pad_value,
                    fill_mask=0,
                    p=1.0,
                ),
                A.CenterCrop(height=h0, width=w0, p=1.0),
            ]

        has_boxes = (target is not None and 'boxes' in target and len(target['boxes']) > 0)
        boxes_list = []
        if has_boxes:
            b_np = np.array(target['boxes'], dtype=np.float32)
            # Фильтруем рамки, выходящие за пределы или вырожденные
            for b in b_np:
                x1 = float(np.clip(b[0], 0, w0))
                y1 = float(np.clip(b[1], 0, h0))
                x2 = float(np.clip(b[2], 0, w0))
                y2 = float(np.clip(b[3], 0, h0))
                if (x2 - x1) >= self.min_box_size and (y2 - y1) >= self.min_box_size:
                    boxes_list.append([x1, y1, x2, y2])

        min_area = self.min_box_size * self.min_box_size
        bbox_params = A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=min_area,
            min_width=self.min_box_size,
            min_height=self.min_box_size
        ) if len(boxes_list) > 0 else None
        pipeline = A.Compose(transforms, mask_interpolation=cv2.INTER_NEAREST, bbox_params=bbox_params)

        instance_map_np = target.get('instance_map')
        call_kwargs = {"image": img_np}
        if instance_map_np is not None:
            if isinstance(instance_map_np, torch.Tensor):
                instance_map_np = instance_map_np.numpy()
            call_kwargs["mask"] = instance_map_np.astype(np.uint16)
        if len(boxes_list) > 0:
            call_kwargs["bboxes"] = boxes_list
            call_kwargs["labels"] = [0] * len(boxes_list)

        res = pipeline(**call_kwargs)
        img_aug = res["image"]
        if "mask" in res:
            target['instance_map'] = res["mask"]
        if "bboxes" in res:
            filtered_boxes = []
            for b in res["bboxes"]:
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                if bw >= self.min_box_size and bh >= self.min_box_size:
                    filtered_boxes.append([b[0], b[1], b[2], b[3]])
            target['boxes'] = np.array(filtered_boxes, dtype=np.float32) if len(filtered_boxes) > 0 else np.empty((0, 4), dtype=np.float32)

        if is_np or img_aug.shape[-1] > 4:
            return img_aug, target
        return PILImage.fromarray(img_aug), target


class SyncResize:
    def __init__(self, size=640):
        self.size = size

    def __call__(self, img, target: Dict[str, Any], **kwargs):
        if isinstance(img, np.ndarray):
            old_h, old_w = img.shape[:2]
            target_size = (self.size, self.size) if isinstance(self.size, int) else (self.size[1], self.size[0])
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            w, h = target_size
        else:
            old_w, old_h = img.size
            img = F.resize(img, self.size, interpolation=F.InterpolationMode.BILINEAR)
            w, h = (img.size[0], img.size[1]) if isinstance(self.size, int) else (self.size[1], self.size[0])

        if target is not None and 'instance_map' in target:
            target['instance_map'] = cv2.resize(
                target['instance_map'].astype(np.uint16),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        if target is not None and 'ignore_mask' in target and target['ignore_mask'] is not None:
            target['ignore_mask'] = cv2.resize(
                target['ignore_mask'].astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        # Геометрическое масштабирование рамок
        if target is not None and 'boxes' in target and len(target['boxes']) > 0 and old_w > 0 and old_h > 0:
            sx = float(w) / float(old_w)
            sy = float(h) / float(old_h)
            boxes = np.array(target['boxes'], dtype=np.float32).copy()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target['boxes'] = boxes

        return img, target


class RandomNoiseSP:
    """Импульсный шум соль-перец, настройки сохранены как в шаблоне."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_tensor: torch.Tensor, target: Dict[str, Any], **kwargs):
        if not isinstance(img_tensor, torch.Tensor):
            return img_tensor, target
        if np.random.rand() < self.p:
            mode = random.choice(['salt', 'pepper', 's&p'])
            amount = np.random.randint(0, 11) / 1000.0
            arr = img_tensor.permute(1, 2, 0).cpu().numpy()
            noisy = random_noise(arr, mode=mode, amount=amount)
            img_tensor = torch.tensor(noisy, dtype=torch.float32).permute(2, 0, 1)
        return img_tensor, target


class SyncToTensor:
    """
    Преобразует вход в тензор PyTorch [C, H, W] и разворачивает
    bounding boxes в формате [x1, y1, x2, y2] и метки классов с гарантией целостности перекрывающихся рамок.
    """
    def __init__(self, bbox_padding=0, plant_label=0):
        self.tt = ToTensor()
        self.bbox_padding = bbox_padding
        self.plant_label = plant_label

    def __call__(self, img, target: Dict[str, Any], **kwargs):
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        elif isinstance(img, torch.Tensor):
            img_tensor = img.float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
        else:
            img_tensor = self.tt(img)  # [C, H, W], float32 [0.0, 1.0]

        cur_h = img_tensor.shape[1]
        cur_w = img_tensor.shape[2]
        new_target = {}

        # 1. Приоритет отдается геометрически пересчитанным рамкам (не теряются при перекрытиях)
        if target is not None and 'boxes' in target and len(target['boxes']) > 0:
            b_arr = np.array(target['boxes'], dtype=np.float32).copy()
            b_arr[:, [0, 2]] = np.clip(b_arr[:, [0, 2]], 0, cur_w)
            b_arr[:, [1, 3]] = np.clip(b_arr[:, [1, 3]], 0, cur_h)
            valid = (b_arr[:, 2] > b_arr[:, 0] + 1.0) & (b_arr[:, 3] > b_arr[:, 1] + 1.0)
            if valid.any():
                new_target['boxes'] = torch.from_numpy(b_arr[valid]).float()
                new_target['labels'] = torch.zeros(valid.sum(), dtype=torch.int64)
            else:
                new_target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                new_target['labels'] = torch.empty((0,), dtype=torch.int64)
        elif target is not None and 'instance_map' in target:
            # Fallback: восстановление из instance_map
            instance_map = target['instance_map']
            if isinstance(instance_map, torch.Tensor):
                instance_map = instance_map.numpy()
            obj_ids = np.unique(instance_map)
            obj_ids = obj_ids[obj_ids != 0]

            boxes = []
            labels = []
            for obj_id in obj_ids:
                mask = (instance_map == obj_id)
                coords = np.where(mask)
                if len(coords[0]) == 0:
                    continue
                y_min, x_min = coords[0].min(), coords[1].min()
                y_max, x_max = coords[0].max(), coords[1].max()

                if x_max <= x_min or y_max <= y_min:
                    continue

                boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                labels.append(self.plant_label)

            if len(boxes) > 0:
                new_target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                new_target['labels'] = torch.tensor(labels, dtype=torch.int64)
            else:
                new_target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                new_target['labels'] = torch.empty((0,), dtype=torch.int64)
        else:
            new_target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            new_target['labels'] = torch.empty((0,), dtype=torch.int64)

        if target is not None and 'ignore_mask' in target and target['ignore_mask'] is not None:
            new_target['ignore_mask'] = torch.from_numpy(target['ignore_mask']).to(torch.uint8)

        # Перенос прочих метаданных
        if target is not None:
            for k, v in target.items():
                if k not in ['instance_map', 'boxes', 'labels', 'ignore_mask']:
                    new_target[k] = v

        return img_tensor, new_target


