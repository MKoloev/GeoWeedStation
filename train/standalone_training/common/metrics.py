"""
common/metrics.py - Модуль вычисления метрик детекции (Precision, Recall, F1, AP50, mAP50:95)
с раздельным протоколом для валидируемых и train_only датасетов (ТЗ § 3.1, § 8).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


def box_iou_np(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """
    Вычисляет IoU между двумя наборами рамок [N, 4] и [M, 4] в формате [x1, y1, x2, y2].
    Возвращает матрицу IoU [N, M].
    """
    if len(box_a) == 0 or len(box_b) == 0:
        return np.zeros((len(box_a), len(box_b)), dtype=np.float32)

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    iw = np.maximum(
        0.0,
        np.minimum(box_a[:, 2:3], box_b[:, 2:3].T) - np.maximum(box_a[:, 0:1], box_b[:, 0:1].T)
    )
    ih = np.maximum(
        0.0,
        np.minimum(box_a[:, 3:4], box_b[:, 3:4].T) - np.maximum(box_a[:, 1:2], box_b[:, 1:2].T)
    )
    intersection = iw * ih
    union = area_a[:, None] + area_b[None, :] - intersection + 1e-12
    return intersection / union


def compute_ap_single_threshold(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    iou_thresh: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Вычисляет AP, Precision, Recall и F1 для одного порога IoU (например, 0.50).
    """
    all_scores = []
    all_tp = []
    all_fp = []
    total_gt = 0

    for p_b, p_s, g_b in zip(pred_boxes, pred_scores, gt_boxes):
        n_gt = len(g_b)
        n_pred = len(p_b)
        total_gt += n_gt

        if n_pred == 0:
            continue

        if n_gt == 0:
            for s in p_s:
                all_scores.append(s)
                all_tp.append(0)
                all_fp.append(1)
            continue

        # Сортировка предсказаний по убыванию уверенности
        order = np.argsort(-p_s)
        p_b = p_b[order]
        p_s = p_s[order]

        ious = box_iou_np(p_b, g_b)
        matched_gt = set()

        for pred_i in range(n_pred):
            all_scores.append(p_s[pred_i])
            best_iou = -1.0
            best_gt = -1

            for gt_i in range(n_gt):
                if gt_i in matched_gt:
                    continue
                if ious[pred_i, gt_i] > best_iou:
                    best_iou = ious[pred_i, gt_i]
                    best_gt = gt_i

            if best_iou >= iou_thresh:
                all_tp.append(1)
                all_fp.append(0)
                matched_gt.add(best_gt)
            else:
                all_tp.append(0)
                all_fp.append(1)

    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(all_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)
    all_fp = np.array(all_fp)

    sort_idx = np.argsort(-all_scores)
    tp_sorted = np.cumsum(all_tp[sort_idx])
    fp_sorted = np.cumsum(all_fp[sort_idx])

    recalls = tp_sorted / float(total_gt)
    precisions = tp_sorted / (tp_sorted + fp_sorted + 1e-12)

    # 11-point interpolation or standard continuous AP
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i_indices + 1] - mrec[i_indices]) * mpre[i_indices + 1])

    # Precision, Recall, F1 при пороге уверенности score_thresh = 0.25
    pos_mask = all_scores >= 0.25
    if np.any(pos_mask):
        tp_at_thresh = all_tp[pos_mask].sum()
        fp_at_thresh = all_fp[pos_mask].sum()
        p_eval = tp_at_thresh / (tp_at_thresh + fp_at_thresh + 1e-12)
        r_eval = tp_at_thresh / float(total_gt)
        f1_eval = 2.0 * p_eval * r_eval / (p_eval + r_eval + 1e-12)
    else:
        p_eval, r_eval, f1_eval = 0.0, 0.0, 0.0

    return float(ap), float(p_eval), float(r_eval), float(f1_eval)


class DetectionEvaluator:
    """
    Оценщик качества детектора растений.
    Вычисляет AP50, mAP50:95, Precision, Recall, F1
    с раздельным представлением для валидируемых и train_only датасетов (ТЗ § 3.1).
    """
    def __init__(self, score_threshold: float = 0.25, iou_thresholds: Optional[List[float]] = None):
        self.score_threshold = score_threshold
        self.iou_thresholds = iou_thresholds or [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]
        self.reset()

    def reset(self):
        self.records_by_dataset: Dict[str, Dict[str, List]] = {}

    def update(self, predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        return self.add_batch(predictions, targets)

    def add_batch(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ):
        """
        predictions: список словарей {'boxes': Tensor[N, 4], 'scores': Tensor[N], 'labels': Tensor[N]}
        targets: список словарей {'boxes': Tensor[M, 4], 'labels': Tensor[M], 'dataset_id': str, 'is_partial': bool}
        """
        for pred, target in zip(predictions, targets):
            d_id = target.get('dataset_id', 'unknown')
            if d_id not in self.records_by_dataset:
                self.records_by_dataset[d_id] = {
                    'pred_boxes': [],
                    'pred_scores': [],
                    'gt_boxes': [],
                    'is_partial': target.get('is_partial', False)
                }

            p_b = pred.get('boxes', torch.empty((0, 4)))
            p_s = pred.get('scores', torch.empty((0,)))
            g_b = target.get('boxes', torch.empty((0, 4)))

            if isinstance(p_b, torch.Tensor):
                p_b = p_b.detach().cpu().numpy()
            if isinstance(p_s, torch.Tensor):
                p_s = p_s.detach().cpu().numpy()
            if isinstance(g_b, torch.Tensor):
                g_b = g_b.detach().cpu().numpy()

            self.records_by_dataset[d_id]['pred_boxes'].append(p_b)
            self.records_by_dataset[d_id]['pred_scores'].append(p_s)
            self.records_by_dataset[d_id]['gt_boxes'].append(g_b)

    def compute(self) -> Dict[str, Any]:
        """
        Вычисляет метрики по каждому датасету и итоговые агрегированные показатели.
        """
        results_per_dataset = {}
        val_maps = []
        val_ap50s = []
        val_f1s = []

        train_only_maps = []

        for d_id, data in self.records_by_dataset.items():
            pred_boxes = data['pred_boxes']
            pred_scores = data['pred_scores']
            gt_boxes = data['gt_boxes']
            is_partial = data['is_partial']

            # AP50
            ap50, prec, rec, f1 = compute_ap_single_threshold(
                pred_boxes, pred_scores, gt_boxes, iou_thresh=0.50
            )

            # mAP50:95
            aps = []
            for iou_th in self.iou_thresholds:
                ap_k, _, _, _ = compute_ap_single_threshold(
                    pred_boxes, pred_scores, gt_boxes, iou_thresh=iou_th
                )
                aps.append(ap_k)
            map50_95 = float(np.mean(aps))

            total_gt_count = sum(len(b) for b in gt_boxes)
            total_pred_count = sum(len(b) for b in pred_boxes)

            entry = {
                'dataset_id': d_id,
                'is_partial': is_partial,
                'samples_count': len(gt_boxes),
                'gt_boxes_count': total_gt_count,
                'pred_boxes_count': total_pred_count,
                'ap50': ap50,
                'map50_95': map50_95,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            results_per_dataset[d_id] = entry

            # Разделение согласно протоколу ТЗ § 3.1
            if not is_partial:
                val_maps.append(map50_95)
                val_ap50s.append(ap50)
                val_f1s.append(f1)
            else:
                train_only_maps.append(map50_95)

        all_ap50s = [d['ap50'] for d in results_per_dataset.values()]
        all_maps = [d['map50_95'] for d in results_per_dataset.values()]
        all_f1s = [d['f1'] for d in results_per_dataset.values()]

        summary = {
            'by_dataset': results_per_dataset,
            'val_mean_ap50': float(np.mean(val_ap50s)) if len(val_ap50s) > 0 else (float(np.mean(all_ap50s)) if len(all_ap50s) > 0 else 0.0),
            'val_mean_map50_95': float(np.mean(val_maps)) if len(val_maps) > 0 else (float(np.mean(all_maps)) if len(all_maps) > 0 else 0.0),
            'val_mean_f1': float(np.mean(val_f1s)) if len(val_f1s) > 0 else (float(np.mean(all_f1s)) if len(all_f1s) > 0 else 0.0),
            'val_datasets_count': len(val_maps) if len(val_maps) > 0 else len(all_ap50s),
            'train_only_datasets_count': len(train_only_maps)
        }

        return summary


class ClassificationEvaluator:
    """
    Оценщик качества классификации ботанических видов и категорий (ТЗ § 7).
    Вычисляет accuracy, balanced accuracy, macro-F1, per-class Precision/Recall/F1,
    матрицу ошибок (confusion matrix) и разбивку по датасетам.
    """
    def __init__(self, class_names: Optional[List[str]] = None, num_classes: Optional[int] = None):
        if class_names is not None:
            self.class_names = class_names
        elif num_classes is not None:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]
        else:
            self.class_names = []
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.dataset_ids = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metas: Optional[List[Dict[str, Any]]] = None
    ):
        return self.add_batch(predictions, targets, metas)

    def add_batch(
        self,
        predictions: torch.Tensor,  # [B] или [B, C] (logits/probs)
        targets: torch.Tensor,      # [B]
        metas: Optional[List[Dict[str, Any]]] = None
    ):
        if predictions.dim() == 2:
            preds = predictions.argmax(dim=1).detach().cpu().numpy()
        else:
            preds = predictions.detach().cpu().numpy()

        gts = targets.detach().cpu().numpy()

        self.y_pred.extend(preds.tolist())
        self.y_true.extend(gts.tolist())

        if metas is not None:
            for m in metas:
                self.dataset_ids.append(m.get("dataset_id", "unknown"))
        else:
            self.dataset_ids.extend(["unknown"] * len(preds))

    def compute(self) -> Dict[str, Any]:
        if len(self.y_true) == 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "per_class": {},
                "confusion_matrix": []
            }

        y_t = np.array(self.y_true)
        y_p = np.array(self.y_pred)

        num_classes = max(
            len(self.class_names),
            int(max(y_t.max() if len(y_t) > 0 else 0, y_p.max() if len(y_p) > 0 else 0) + 1)
        )

        # Accuracy
        acc = float(np.mean(y_t == y_p))

        # Confusion Matrix [num_classes, num_classes] (строки: true, столбцы: pred)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_t, y_p):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

        per_class = {}
        recalls = []
        precisions = []
        f1s = []

        for c in range(num_classes):
            c_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            tp = float(cm[c, c])
            fp = float(cm[:, c].sum() - tp)
            fn = float(cm[c, :].sum() - tp)
            support = int(cm[c, :].sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if support > 0:
                recalls.append(rec)
                precisions.append(prec)
                f1s.append(f1)

            per_class[c_name] = {
                "class_idx": c,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": support
            }

        balanced_acc = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
        macro_prec = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
        macro_rec = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
        macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

        return {
            "total_samples": len(y_t),
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": cm.tolist()
        }


