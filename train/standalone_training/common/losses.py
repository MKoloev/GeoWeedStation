"""
common/losses.py - Функция потерь для YOLOX с поддержкой частичной разметки (Partial Plant Annotations)
согласно ТЗ § 3 и правилам проекта.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def bboxes_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy: bool = True) -> torch.Tensor:
    """
    Вычисляет IoU между двумя наборами рамок [N, 4] и [M, 4].
    Возвращает матрицу IoU [N, M].
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError("Размер рамок должен быть [*, 4]")

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i + 1e-16)


class IOUloss(nn.Module):
    """
    IoU loss для регрессии bounding box.
    loss = -ln(IoU) или 1 - IoU
    """
    def __init__(self, reduction="none", loss_type="iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1.0 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / (area_c + 1e-16)
            loss = 1.0 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise NotImplementedError

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class YOLOXLoss(nn.Module):
    """
    Функция потерь YOLOX с поддержкой:
    1. SimOTA / Center sampling динамического назначения положительных анкоров.
    2. Регрессии IoU / GIoU.
    3. Objectness BCE.
    4. Классификации BCE.
    5. Специального флага is_partial:
       Для частично размеченных изображений (CamelinaWeed, CoFly, Campaneta, WeedyRice)
       отрицательный штраф фоновых анкоров отключается, чтобы не штрафовать необнаруженные
       культурные растения и сорняки в фоне (ТЗ § 3).
    """
    def __init__(
        self,
        num_classes: int = 1,
        strides: Tuple[int, ...] = (8, 16, 32),
        iou_loss_weight: float = 5.0,
        obj_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        partial_bg_weight: float = 0.5,
        candidate_plant_thresh: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.iou_loss_weight = iou_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.partial_bg_weight = partial_bg_weight
        self.candidate_plant_thresh = candidate_plant_thresh

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type="iou")

    def _get_grids_and_strides(
        self, feature_shapes: List[Tuple[int, int]], dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grids = []
        strides = []
        for shape, stride in zip(feature_shapes, self.strides):
            h, w = shape
            yv, xv = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 2).view(-1, 2)
            grids.append(grid)
            shape_tensor = torch.full((h * w, 1), stride, dtype=dtype, device=device)
            strides.append(shape_tensor)

        return torch.cat(grids, dim=0), torch.cat(strides, dim=0)

    def forward(
        self,
        outputs: List[torch.Tensor],
        targets: List[Dict[str, Any]],
        imgs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        outputs: список выходов голов на страйдах 8, 16, 32:
                 каждый тензор формы [B, 4 + 1 + num_classes, H, W]
        targets: список длины B словарей:
                 'boxes': [N, 4] в формате xyxy
                 'labels': [N]
                 'is_partial': bool
        imgs:    тензор исходных батчей [B, C, H, W] для оптического вычисления
                 вегетативных индексов (ExG / NDVI) на частично размеченных кадрах.
        """
        device = outputs[0].device
        dtype = outputs[0].dtype
        batch_size = outputs[0].shape[0]

        feature_shapes = [out.shape[2:] for out in outputs]
        grids, strides = self._get_grids_and_strides(feature_shapes, dtype, device)
        total_anchors = grids.shape[0]

        # Разворачиваем выходы голов: [B, total_anchors, 4 + 1 + num_classes]
        flatten_outputs = []
        for out in outputs:
            b, c, h, w = out.shape
            out_flat = out.view(b, c, -1).permute(0, 2, 1)
            flatten_outputs.append(out_flat)
        flatten_outputs = torch.cat(flatten_outputs, dim=1)

        pred_reg = flatten_outputs[:, :, :4]  # [B, A, 4]: tx, ty, tw, th
        pred_obj = flatten_outputs[:, :, 4:5]  # [B, A, 1]: objectness logits
        pred_cls = flatten_outputs[:, :, 5:]  # [B, A, C]: class logits

        # Декодирование центров и размеров рамок
        pred_xy = (pred_reg[:, :, :2] + grids[None, :, :]) * strides[None, :, :]
        pred_wh = torch.exp(pred_reg[:, :, 2:]) * strides[None, :, :]
        pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)  # [B, A, 4]: xc, yc, w, h

        # Декодированные рамки в xyxy для IoU
        pred_boxes_xyxy = torch.cat([
            pred_xy - pred_wh / 2.0,
            pred_xy + pred_wh / 2.0
        ], dim=-1)

        total_num_fg = 0
        loss_iou = torch.tensor(0.0, device=device, dtype=dtype)
        loss_cls = torch.tensor(0.0, device=device, dtype=dtype)
        loss_obj = torch.tensor(0.0, device=device, dtype=dtype)

        # Вычисление потерь по каждому изображению батча
        for b_idx in range(batch_size):
            target_i = targets[b_idx]
            gt_boxes = target_i.get("boxes", torch.empty((0, 4), device=device)).to(device)
            gt_labels = target_i.get("labels", torch.empty((0,), dtype=torch.int64, device=device)).to(device)
            is_partial = target_i.get("is_partial", False)

            p_boxes_xywh = pred_boxes[b_idx]  # [A, 4]
            p_boxes_xyxy = pred_boxes_xyxy[b_idx]  # [A, 4]
            p_obj = pred_obj[b_idx]  # [A, 1]
            p_cls = pred_cls[b_idx]  # [A, C]

            num_gt = gt_boxes.shape[0]

            if num_gt == 0:
                # Нет положительных объектов
                if not is_partial:
                    # Для полностью размеченных датасетов штрафуем все анкоры как фон
                    loss_obj += self.bce_loss(p_obj, torch.zeros_like(p_obj)).sum()
                # Для is_partial=True отрицательный штраф не накладывается
                continue

            # Преобразуем GT boxes из xyxy в xywh
            gt_xy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
            gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
            gt_boxes_xywh = torch.cat([gt_xy, gt_wh], dim=-1)

            # Назначение анкоров (Center-in-box matching + Top-K IoU + distance)
            # 1. Проверяем попадание центров анкоров внутрь GT рамок
            anchor_centers = grids * strides  # [A, 2]
            c_x = anchor_centers[:, 0]
            c_y = anchor_centers[:, 1]

            # [N_gt, A]
            in_gt_mask = (
                (c_x[None, :] >= gt_boxes[:, 0:1]) &
                (c_x[None, :] <= gt_boxes[:, 2:3]) &
                (c_y[None, :] >= gt_boxes[:, 1:2]) &
                (c_y[None, :] <= gt_boxes[:, 3:4])
            )

            # Вычисляем IoU между всеми анкорами внутри GT и самими GT
            ious = bboxes_iou(gt_boxes, p_boxes_xyxy, xyxy=True)  # [N_gt, A]

            # Расстояние от центров анкоров до центров GT, нормализованное диагональю GT
            dist_centers = torch.sqrt(torch.sum((anchor_centers[None, :, :] - gt_xy[:, None, :]) ** 2, dim=-1))
            gt_diags = torch.sqrt(gt_wh[:, 0] ** 2 + gt_wh[:, 1] ** 2)[:, None].clamp(min=1e-3)
            norm_dist = dist_centers / gt_diags  # [N_gt, A]

            # Стоимость сопоставления: комбинируем IoU и близость к центру (для надежного старта при IoU=0)
            match_score = in_gt_mask.float() * (ious + 0.5 * torch.exp(-2.0 * norm_dist))

            # Динамический выбор Top-K для каждого GT
            fg_mask = torch.zeros(total_anchors, dtype=torch.bool, device=device)
            matched_gt_idx = torch.full((total_anchors,), -1, dtype=torch.int64, device=device)

            for gt_k in range(num_gt):
                scores_k = match_score[gt_k]
                valid_mask = in_gt_mask[gt_k]
                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) == 0:
                    # Если ни один центр не попал внутрь маленького объекта, берём ближайший анкор
                    dist = torch.sum((anchor_centers - gt_xy[gt_k:gt_k+1]) ** 2, dim=-1)
                    best_idx = torch.argmin(dist)
                    fg_mask[best_idx] = True
                    matched_gt_idx[best_idx] = gt_k
                    continue

                topk = min(10, len(valid_indices))
                _, topk_sub_idx = torch.topk(scores_k[valid_indices], k=topk)
                topk_anchor_idx = valid_indices[topk_sub_idx]

                fg_mask[topk_anchor_idx] = True
                matched_gt_idx[topk_anchor_idx] = gt_k

            num_fg = fg_mask.sum().item()
            total_num_fg += max(num_fg, 1)

            # Objectness target
            obj_target = torch.zeros_like(p_obj)
            if num_fg > 0:
                # Положительные анкоры
                assigned_gt_boxes = gt_boxes_xywh[matched_gt_idx[fg_mask]]
                assigned_pred_boxes = p_boxes_xywh[fg_mask]

                # Bbox loss
                iou_loss_fg = self.iou_loss(assigned_pred_boxes, assigned_gt_boxes)
                loss_iou += iou_loss_fg.sum()

                # Class loss
                assigned_gt_labels = gt_labels[matched_gt_idx[fg_mask]]
                cls_target = F.one_hot(assigned_gt_labels, num_classes=self.num_classes).float()
                loss_cls += self.bce_loss(p_cls[fg_mask], cls_target).sum()

                # Целевое значение objectness для положительных анкоров равно строго 1.0
                obj_target[fg_mask] = 1.0

            # Objectness loss:
            bce_all = self.bce_loss(p_obj, obj_target).squeeze(-1)
            if not is_partial:
                # Для полностью размеченных изображений штрафуются ВСЕ анкоры (и fg, и bg)
                loss_obj += bce_all.sum()
            else:
                # ДЛЯ ЧАСТИЧНО РАЗМЕЧЕННЫХ ИЗОБРАЖЕНИЙ (ТЗ § 1 п. 90, § 3):
                # 1. Положительные анкоры штрафуются в полную силу
                loss_obj += bce_all[fg_mask].sum()

                # 2. Определение растительности (ExG для RGB, NDVI для мультиспектра)
                anchor_is_veg = torch.zeros(total_anchors, dtype=torch.bool, device=device)
                if imgs is not None and b < imgs.shape[0]:
                    img_b = imgs[b]  # [C, H, W]
                    img_c, img_h, img_w = img_b.shape

                    if img_c >= 5:  # Мультиспектр (RE, NIR в конце каналов)
                        nir = img_b[-1].float()
                        red = img_b[0].float()
                        ndvi = (nir - red) / (nir + red + 1e-6)
                        is_veg_map = ndvi > 0.15
                    elif img_c >= 3:  # RGB
                        red = img_b[0].float()
                        green = img_b[1].float()
                        blue = img_b[2].float()
                        exg = 2.0 * green - red - blue
                        is_veg_map = exg > 0.0
                    else:
                        is_veg_map = torch.zeros((img_h, img_w), dtype=torch.bool, device=device)

                    px = c_x.long().clamp(0, img_w - 1)
                    py = c_y.long().clamp(0, img_h - 1)
                    anchor_is_veg = is_veg_map[py, px]

                # 3. Модельная оценка уверенности вне GT (PU-Learning ignore)
                p_obj_prob = torch.sigmoid(p_obj.detach()).squeeze(-1)
                anchor_high_conf = (p_obj_prob > self.candidate_plant_thresh)

                # 4. Буферная зона вокруг подтвержденных GT-растений
                if num_gt > 0:
                    buf_x1 = gt_boxes[:, 0] - 0.20 * gt_wh[:, 0]
                    buf_y1 = gt_boxes[:, 1] - 0.20 * gt_wh[:, 1]
                    buf_x2 = gt_boxes[:, 2] + 0.20 * gt_wh[:, 0]
                    buf_y2 = gt_boxes[:, 3] + 0.20 * gt_wh[:, 1]
                    in_buffer = (
                        (c_x[None, :] >= buf_x1[:, None]) &
                        (c_x[None, :] <= buf_x2[:, None]) &
                        (c_y[None, :] >= buf_y1[:, None]) &
                        (c_y[None, :] <= buf_y2[:, None])
                    ).any(dim=0)
                else:
                    in_buffer = torch.zeros(total_anchors, dtype=torch.bool, device=device)

                # Кандидаты в неразмеченные растения (зеленая масса ИЛИ высокая уверенность ИЛИ буфер GT):
                # Для них лосс отключается (IGNORE zone, нулевой штраф за предсказание неразмеченных культур/сорняков)
                is_unlabeled_plant_candidate = anchor_is_veg | anchor_high_conf | in_buffer

                # Истинный фон (чистая почва / земля вне растений):
                # Только те анкоры, где НЕТ зеленой биомассы, НЕТ высокой уверенности и НЕТ буфера GT
                pure_soil_bg_mask = (~fg_mask) & (~is_unlabeled_plant_candidate)
                if pure_soil_bg_mask.any() and self.partial_bg_weight > 0.0:
                    loss_obj += self.partial_bg_weight * bce_all[pure_soil_bg_mask].sum()

        num_fg_norm = max(total_num_fg, 1.0)
        reg_loss = self.iou_loss_weight * (loss_iou / num_fg_norm)
        obj_loss = self.obj_loss_weight * (loss_obj / num_fg_norm)
        cls_loss = self.cls_loss_weight * (loss_cls / num_fg_norm)

        total_loss = reg_loss + obj_loss + cls_loss

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_reg": reg_loss.item(),
            "loss_obj": obj_loss.item(),
            "loss_cls": cls_loss.item(),
            "num_fg": total_num_fg,
        }

        return total_loss, loss_dict
