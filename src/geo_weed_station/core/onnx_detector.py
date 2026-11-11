"""ONNX Runtime based neural network detector for weed and plant identification."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort

from geo_weed_station.config.settings import DetectorSettings
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.exceptions import DetectionError
from geo_weed_station.schemas.entities import BBox, DetectedWeed

logger = logging.getLogger(__name__)

# =============================================================================
# TAXONOMIES AND CLASSIFICATION DICTIONARIES (matching pipeline_orange_pi.py & MODELS.md)
# =============================================================================

C07_CATEGORIES = [
    "background",
    "Осот, Бодяк, Латук",
    "Вьюнок",
    "Щирица (виды)",
    "Маревые (Марь, Лебеда, Кохия)",
    "Канатник, просвирник, гибискус",
    "Крестоцветные (Капустные)",
    "Горец (виды)",
    "Злаковые",
    "Полынь, Ромашка",
    "Гвоздичные",
]

C07_CATEGORIES_EN = [
    "Background",
    "Osot, Bodyak, Latuk",
    "Vyunok (Convolvulus)",
    "Shchiritsa (Amaranthus)",
    "Marevye (Chenopodiaceae)",
    "Kanatnik, Malva",
    "Krestotsvetnye (Brassicaceae)",
    "Gorets (Polygonum)",
    "Zlakovye (Poaceae)",
    "Polyn, Romashka",
    "Gvozdichnye (Caryophyllaceae)",
]

C07_STAGES = [
    "всходы",
    "2-4 листа",
    "взрослое растение",
    "переросшие",
    "не определена",
]

C06_TAXONOMY: dict[int, tuple[str, str, str]] = {
    0: ("S000", "Фон / Не растение", "Background"),
    1: ("S001", "Кукуруза", "Zea mays"),
    2: ("S002", "Плевел жесткий", "Lolium rigidum"),
    3: ("S003", "Гумай", "Sorghum halepense"),
    4: ("S004", "Овсюг обыкновенный", "Avena fatua"),
    5: ("S005", "Ячмень обыкновенный", "Hordeum vulgare"),
    6: ("S006", "Лебеда раскидистая", "Atriplex patula"),
    7: ("S007", "Марь белая", "Chenopodium album"),
    8: ("S008", "Солянка русская (курай)", "Salsola kali"),
    9: ("S009", "Свёкла обыкновенная", "Beta vulgaris"),
    10: ("S010", "Ромашка пахучая", "Matricaria discoidea"),
    11: ("S011", "Татарник колючий", "Onopordum acanthium"),
    12: ("S012", "Расторопша пятнистая", "Silybum marianum"),
    13: ("S013", "Вьюнок полевой (берёзка)", "Convolvulus arvensis"),
    14: ("S014", "Мальва лесная", "Malva sylvestris"),
    15: ("S015", "Хлопчатник", "Gossypium hirsutum"),
    16: ("S016", "Горчица полевая", "Sinapis arvensis"),
    17: ("S017", "Сыть круглая", "Cyperus rotundus"),
    18: ("S018", "Томат обыкновенный", "Solanum lycopersicum"),
    19: ("S019", "Паслён чёрный", "Solanum nigrum"),
    20: ("S020", "Паслён линейнолистный", "Solanum elaeagnifolium"),
    21: ("S021", "Дурман свирепый", "Datura ferox"),
    22: ("S022", "Дурман обыкновенный", "Datura stramonium"),
    23: ("S023", "Портулак огородный", "Portulaca oleracea"),
}


class ModelType(str, Enum):
    """Supported model detection architectures."""

    AUTO = "auto"
    YOLO = "yolo"    # Ultralytics YOLOv5 / YOLOv8 (single 3D output tensor)
    YOLOX = "yolox"  # YOLOX Decoupled Head (multi-stride 4D output tensors)


@dataclass(frozen=True)
class LetterboxResult:
    """Result of image letterbox preprocessing for neural network input."""

    tensor: np.ndarray  # Shape: (1, 3, target_h, target_w), dtype float32, normalized [0..1]
    scale: float
    pad: tuple[float, float]  # (pad_x, pad_y)


def letterbox(
    image: np.ndarray,
    size: tuple[int, int] = (640, 640),
    pad_color: int = 114,
) -> LetterboxResult:
    """Resize and pad image with preserved aspect ratio to target size (width, height).
    
    Args:
        image: Source RGB numpy array (H, W, 3).
        size: Target (width, height).
        pad_color: Padding border color value.
        
    Returns:
        LetterboxResult containing normalized NCHW tensor, scale factor, and padding offsets.
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = size

    # Scale with preserved aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # Compute symmetric padding
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0

    # Resize image
    if (orig_w, orig_h) != (new_w, new_h):
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    # Create padded canvas
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    top = int(round(pad_y))
    left = int(round(pad_x))
    canvas[top : top + new_h, left : left + new_w] = resized

    # Convert HWC uint8 [0..255] to NCHW float32 [0.0..1.0]
    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # (3, H, W)
    tensor = np.expand_dims(tensor, axis=0)  # (1, 3, H, W)
    tensor = np.ascontiguousarray(tensor)

    return LetterboxResult(tensor=tensor, scale=scale, pad=(float(left), float(top)))


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
    class_ids: np.ndarray | None = None,
) -> list[int]:
    """Non-Maximum Suppression (NMS) on bounding boxes with optional class grouping.
    
    Args:
        boxes: Numpy array of shape (N, 4) in format [x1, y1, x2, y2].
        scores: Numpy array of shape (N,) confidence scores.
        iou_thr: IoU threshold above which overlapping boxes are suppressed.
        class_ids: Optional array of shape (N,) class IDs. If provided, NMS operates
                   independently per class (class-aware NMS).
        
    Returns:
        List of retained box indices.
    """
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # Class-aware NMS via coordinate offset to prevent inter-class suppression
    if class_ids is not None and len(class_ids) == len(boxes):
        class_ids_arr = np.asarray(class_ids)
        max_coord = float(np.max(boxes)) if boxes.size > 0 else 0.0
        offset = (max_coord + 1000.0) * class_ids_arr[:, None].astype(boxes.dtype)
        eval_boxes = boxes + offset
    else:
        eval_boxes = boxes

    x1 = eval_boxes[:, 0]
    y1 = eval_boxes[:, 1]
    x2 = eval_boxes[:, 2]
    y2 = eval_boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(inter)
        valid_mask = union > 0.0
        iou[valid_mask] = inter[valid_mask] / union[valid_mask]

        remaining_indices = np.where(iou <= iou_thr)[0]
        order = order[remaining_indices + 1]

    return keep


def nms_boxes(
    boxes: np.ndarray | Sequence[Sequence[float]],
    scores: np.ndarray | Sequence[float],
    iou_thr: float = 0.45,
    class_ids: np.ndarray | Sequence[int] | None = None,
    box_format: str = "xyxy",
) -> list[int]:
    """Universal NMS helper supporting multiple box formats (xyxy, xywh) and class grouping.
    
    Args:
        boxes: (N, 4) array or list of boxes.
        scores: (N,) array or list of confidence scores.
        iou_thr: IoU suppression threshold (e.g. 0.45).
        class_ids: Optional class IDs for per-class NMS.
        box_format: Either 'xyxy' ([x1, y1, x2, y2]) or 'xywh' ([x, y, w, h]).
        
    Returns:
        List of retained indices sorted by descending confidence.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    cls_arr = np.asarray(class_ids, dtype=np.int32) if class_ids is not None else None

    if box_format.lower() == "xywh":
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = x1 + boxes_arr[:, 2]
        y2 = y1 + boxes_arr[:, 3]
        boxes_arr = np.stack([x1, y1, x2, y2], axis=1)

    return nms(boxes_arr, scores_arr, iou_thr=iou_thr, class_ids=cls_arr)


def nms_detected_weeds(
    weeds: Sequence[DetectedWeed],
    iou_thr: float = 0.45,
    class_aware: bool = True,
) -> list[DetectedWeed]:
    """Filter a list of DetectedWeed entities using Non-Maximum Suppression.
    
    Args:
        weeds: Sequence of DetectedWeed objects (normalized BBox [0..1]).
        iou_thr: IoU threshold above which overlapping weeds are suppressed.
        class_aware: If True, suppression is applied per-class. If False,
                     highest confidence weed suppresses any overlapping weed.
                     
    Returns:
        List of retained DetectedWeed entities sorted by confidence.
    """
    if not weeds:
        return []

    boxes = [[w.bbox.x, w.bbox.y, w.bbox.w, w.bbox.h] for w in weeds]
    scores = [w.confidence for w in weeds]
    class_ids = [w.class_id for w in weeds] if class_aware else None

    keep_indices = nms_boxes(
        boxes=boxes,
        scores=scores,
        iou_thr=iou_thr,
        class_ids=class_ids,
        box_format="xywh",
    )
    return [weeds[i] for i in keep_indices]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _decode_yolo(
    preds: np.ndarray,
    conf_threshold: float,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLOv8 / YOLOv5 single 2D/3D output into boxes, scores, and class IDs.
    
    Returns:
        boxes: (N, 4) in [x1, y1, x2, y2] format on letterbox canvas.
        scores: (N,) float32 confidence scores.
        class_ids: (N,) int32 class IDs.
    """
    if preds.ndim == 3:
        # If shape is (1, C, N) where C < N (e.g. YOLOv8 1x6x8400), transpose to (1, N, C)
        if preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))
        preds = preds[0]  # (N, C)
    elif preds.ndim != 2:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    if preds.shape[0] == 0 or preds.shape[1] < 5:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # Check for objectness (YOLOv5 has objectness at col 4; YOLOv8 does not)
    has_objectness = preds.shape[1] > (num_classes + 4) and num_classes > 0

    if has_objectness:
        obj_conf = preds[:, 4]
        class_scores = preds[:, 5:] * obj_conf[:, None]
    else:
        class_scores = preds[:, 4:]

    best_classes = np.argmax(class_scores, axis=-1).astype(np.int32)
    best_scores = np.max(class_scores, axis=-1).astype(np.float32)

    mask = best_scores >= conf_threshold
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    cx = preds[mask, 0]
    cy = preds[mask, 1]
    bw = preds[mask, 2]
    bh = preds[mask, 3]

    bx1 = cx - bw / 2.0
    by1 = cy - bh / 2.0
    bx2 = cx + bw / 2.0
    by2 = cy + bh / 2.0

    boxes = np.stack([bx1, by1, bx2, by2], axis=1).astype(np.float32)
    return boxes, best_scores[mask], best_classes[mask]


def _decode_yolox(
    outputs: list[np.ndarray],
    input_size: tuple[int, int],
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLOX Decoupled Head multi-stride outputs into boxes, scores, and class IDs.
    
    Args:
        outputs: List of stride feature maps, typically shapes (1, 6, H, W).
        input_size: (width, height) of network input.
        conf_threshold: Confidence filtering threshold.
        
    Returns:
        boxes: (N, 4) in [x1, y1, x2, y2] on letterbox canvas.
        scores: (N,) float32.
        class_ids: (N,) int32.
    """
    target_w, target_h = input_size
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    for out in outputs:
        # Expected shape (1, C, H, W) or (C, H, W)
        feat = out[0] if out.ndim == 4 else out
        if feat.ndim != 3 or feat.shape[0] < 5:
            continue

        c, h, w = feat.shape
        stride_x = target_w / float(w)
        stride_y = target_h / float(h)

        # Transpose (C, H, W) -> (H, W, C)
        feat_hwc = np.transpose(feat, (1, 2, 0))

        # YOLOX Decoupled head activations:
        # ch 0..3: reg (dx, dy, dw, dh)
        # ch 4: obj_logits -> sigmoid
        # ch 5..: cls_logits -> sigmoid
        obj_conf = _sigmoid(feat_hwc[..., 4])
        cls_scores = _sigmoid(feat_hwc[..., 5:])

        # Combined confidence
        combined_scores = obj_conf[..., None] * cls_scores
        best_cls = np.argmax(combined_scores, axis=-1).astype(np.int32)
        best_scores = np.max(combined_scores, axis=-1).astype(np.float32)

        mask = best_scores >= conf_threshold
        if not np.any(mask):
            continue

        y_idx, x_idx = np.where(mask)
        selected_scores = best_scores[mask]
        selected_classes = best_cls[mask]

        reg = feat_hwc[y_idx, x_idx, :4]
        dx = reg[:, 0]
        dy = reg[:, 1]
        dw = np.clip(reg[:, 2], -10.0, 10.0)
        dh = np.clip(reg[:, 3], -10.0, 10.0)

        cx = (x_idx + dx) * stride_x
        cy = (y_idx + dy) * stride_y
        bw = np.exp(dw) * stride_x
        bh = np.exp(dh) * stride_y

        bx1 = cx - bw / 2.0
        by1 = cy - bh / 2.0
        bx2 = cx + bw / 2.0
        by2 = cy + bh / 2.0

        boxes = np.stack([bx1, by1, bx2, by2], axis=1).astype(np.float32)
        all_boxes.append(boxes)
        all_scores.append(selected_scores)
        all_classes.append(selected_classes)

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.vstack(all_boxes), np.concatenate(all_scores), np.concatenate(all_classes)


@dataclass(frozen=True)
class ModelMetadata:
    """Parsed model metadata including classes, architecture, and NMS thresholds."""

    classes: dict[int, str]
    model_type: str | None = None
    confidence_threshold: float | None = None
    nms_iou_threshold: float | None = None


def load_model_metadata(path: Path) -> ModelMetadata:
    """Load model classes and optional detection hyperparameters from JSON file."""
    path = Path(path)
    if not path.is_file():
        return ModelMetadata(classes={0: "weed_dicot", 1: "weed_monocot"})

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return ModelMetadata(classes={0: "weed_dicot", 1: "weed_monocot"})

        # Detect explicit model_type if present
        raw_type = (
            data.get("model_type")
            or data.get("type")
            or data.get("arch")
            or data.get("architecture")
        )
        model_type: str | None = None
        if isinstance(raw_type, str):
            clean_type = raw_type.strip().lower()
            if "yolox" in clean_type:
                model_type = ModelType.YOLOX.value
            elif "yolo" in clean_type:
                model_type = ModelType.YOLO.value

        # Optional threshold overrides
        raw_conf = data.get("confidence_threshold") or data.get("conf_threshold") or data.get("conf")
        conf_thr = float(raw_conf) if raw_conf is not None else None

        raw_iou = data.get("nms_iou_threshold") or data.get("iou_threshold") or data.get("iou") or data.get("nms")
        iou_thr = float(raw_iou) if raw_iou is not None else None

        # Parse classes: either in nested "classes" key or flat dict
        cls_dict = data.get("classes") if isinstance(data.get("classes"), dict) else data
        classes: dict[int, str] = {}
        for k, v in cls_dict.items():
            try:
                classes[int(k)] = str(v)
            except ValueError:
                pass

        if not classes:
            classes = {0: "weed_dicot", 1: "weed_monocot"}

        return ModelMetadata(
            classes=classes,
            model_type=model_type,
            confidence_threshold=conf_thr,
            nms_iou_threshold=iou_thr,
        )
    except Exception:
        return ModelMetadata(classes={0: "weed_dicot", 1: "weed_monocot"})


def load_model_classes_and_type(path: Path) -> tuple[dict[int, str], str | None]:
    """Load class mapping and optional explicit model_type from JSON file."""
    meta = load_model_metadata(path)
    return meta.classes, meta.model_type


def load_class_names(path: Path) -> dict[int, str]:
    """Load class ID to class name mapping from JSON file (backwards-compatible)."""
    meta = load_model_metadata(path)
    return meta.classes


def resolve_classes_path(model_path: Path, fallback_classes_path: Path | None = None) -> Path:
    """Find corresponding classes JSON file for a given model path.
    
    Checks in priority order:
    1. <model_stem>.json next to model
    2. <model_stem>_classes.json next to model
    3. fallback_classes_path if existing
    4. classes.json in model directory
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        candidate_model = get_models_dir() / model_path.name
        if candidate_model.is_file():
            model_path = candidate_model
    model_dir = model_path.parent
    stem = model_path.stem

    candidate1 = model_dir / f"{stem}.json"
    if candidate1.is_file():
        return candidate1

    candidate2 = model_dir / f"{stem}_classes.json"
    if candidate2.is_file():
        return candidate2

    if fallback_classes_path is not None:
        fb = Path(fallback_classes_path)
        if fb.is_file():
            return fb
        fb_candidate = model_dir / fb.name
        if fb_candidate.is_file():
            return fb_candidate

    candidate3 = model_dir / "classes.json"
    if candidate3.is_file():
        return candidate3

    return fallback_classes_path or candidate1


def get_models_dir(custom_path: Path | None = None) -> Path:
    """Resolve the models directory robustly across PyInstaller bundle, source root, and CWD."""
    if custom_path is not None:
        p = Path(custom_path)
        if p.is_dir():
            return p.resolve()
        if p.parent.is_dir():
            return p.parent.resolve()

    if hasattr(sys, "_MEIPASS"):
        meipass_models = Path(sys._MEIPASS) / "models"
        if meipass_models.is_dir():
            return meipass_models.resolve()

    pkg_models = Path(__file__).resolve().parent.parent.parent.parent / "models"
    if pkg_models.is_dir():
        return pkg_models.resolve()

    cwd_models = Path.cwd() / "models"
    if cwd_models.is_dir():
        return cwd_models.resolve()

    return pkg_models.resolve()


def scan_available_models(directory: Path | None = None) -> list[Path]:
    """Scan directory for valid .onnx model files in deterministic order."""
    dir_path = get_models_dir(directory) if directory is None else Path(directory)
    if not dir_path.is_dir():
        return []
    models = [p for p in dir_path.glob("*.onnx") if p.is_file()]
    return sorted(models, key=lambda p: p.name.lower())


class OnnxDetector(BaseDetector):
    """YOLO and YOLOX object detector implementation running on ONNX Runtime."""

    def __init__(
        self,
        settings: DetectorSettings,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.settings = settings
        self._custom_providers = list(providers) if providers is not None else None
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._output_names: list[str] = []
        self._classifier_session: ort.InferenceSession | None = None
        self._cls_input_name: str | None = None
        self._cls_output_names: list[str] = []
        self.classifier_type: str = (self.settings.classifier_type or "none").lower()

        classes_p = Path(self.settings.classes_path)
        if not classes_p.is_file():
            cand_cls = get_models_dir() / classes_p.name
            if cand_cls.is_file():
                classes_p = cand_cls

        meta = load_model_metadata(classes_p)
        self._classes = meta.classes
        self._explicit_model_type = meta.model_type
        self.model_type: str = self._explicit_model_type or ModelType.AUTO.value

        self.confidence_threshold = (
            float(meta.confidence_threshold)
            if meta.confidence_threshold is not None
            else float(self.settings.confidence_threshold)
        )
        self.nms_iou_threshold = (
            float(meta.nms_iou_threshold)
            if meta.nms_iou_threshold is not None
            else float(self.settings.nms_iou_threshold)
        )

    def _get_providers(self) -> list[str]:
        """Return execution providers in priority order."""
        if self._custom_providers:
            return self._custom_providers
        avail = ort.get_available_providers()
        preferred_order = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in preferred_order if p in avail]
        return providers if providers else ["CPUExecutionProvider"]

    def _resolve_classifier_path(self) -> Path | None:
        """Resolve path to classifier ONNX model file."""
        cls_p = self.settings.classifier_path
        if cls_p is not None:
            p = Path(cls_p)
            if p.is_file():
                return p
            cand = get_models_dir() / p.name
            if cand.is_file():
                return cand
            cand_repo = Path(__file__).resolve().parent.parent.parent.parent / "models" / "c07_rgb_geosaero" / p.name
            if cand_repo.is_file():
                return cand_repo

        if self.classifier_type == "c07":
            cands = [
                get_models_dir() / "resnet50_dualhead_c07.onnx",
                get_models_dir() / "best.onnx",
                Path(__file__).resolve().parent.parent.parent.parent / "models" / "c07_rgb_geosaero" / "resnet50_dualhead_c07.onnx",
            ]
            for c in cands:
                if c.is_file():
                    return c
        elif self.classifier_type == "c06":
            cands = [
                get_models_dir() / "resnet50_c06.onnx",
                Path(__file__).resolve().parent.parent.parent.parent / "models" / "c06_rgb_species" / "resnet50_c06.onnx",
            ]
            for c in cands:
                if c.is_file():
                    return c
        return None

    def _load_classifier(self) -> None:
        """Initialize ONNX InferenceSession for crop classification."""
        if self._classifier_session is not None:
            return
        cls_path = self._resolve_classifier_path()
        if cls_path is None or not cls_path.is_file():
            logger.info("Classifier model for '%s' not found. Crop classification disabled.", self.classifier_type)
            self._classifier_session = None
            return

        try:
            providers = self._get_providers()
            opts = ort.SessionOptions()
            opts.enable_mem_pattern = True
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._classifier_session = ort.InferenceSession(
                str(cls_path),
                sess_options=opts,
                providers=providers,
            )
            self._cls_input_name = self._classifier_session.get_inputs()[0].name
            self._cls_output_names = [o.name for o in self._classifier_session.get_outputs()]
            logger.info(
                "Loaded ONNX classifier '%s' (%s) with %d outputs",
                cls_path.name,
                self.classifier_type.upper(),
                len(self._cls_output_names),
            )
        except Exception as e:
            logger.warning("Failed to load classifier from %s: %s", cls_path, e)
            self._classifier_session = None

    def set_classifier(self, classifier_type: str, classifier_path: Path | None = None) -> None:
        """Dynamically configure or switch classifier model."""
        self.classifier_type = classifier_type.lower()
        if classifier_path is not None:
            self.settings = self.settings.model_copy(update={"classifier_path": Path(classifier_path)})
        self._classifier_session = None
        if self.is_loaded and self.classifier_type != "none":
            self._load_classifier()

    @staticmethod
    def available_providers() -> list[str]:
        """Return list of execution providers available in this ONNX Runtime environment."""
        return ort.get_available_providers()

    @property
    def is_loaded(self) -> bool:
        """Return True if ONNX InferenceSession is currently loaded."""
        return self._session is not None

    def set_thresholds(self, confidence: float, iou: float) -> None:
        """Update detection confidence and NMS IoU thresholds dynamically."""
        self.confidence_threshold = float(confidence)
        self.nms_iou_threshold = float(iou)

    def _determine_model_type(self) -> str:
        """Determine whether model is YOLO (v5/v8) or YOLOX multi-stride.
        
        Checks:
        1. Explicit type from classes/metadata JSON if specified.
        2. Model metadata in ONNX session (custom_metadata_map).
        3. Output tensor signatures (number of outputs, rank, and names/shapes).
        """
        # 1. Check explicit configuration from JSON
        if self._explicit_model_type in (ModelType.YOLO.value, ModelType.YOLOX.value):
            return self._explicit_model_type

        if self._session is None:
            return ModelType.YOLO.value

        # 2. Check ONNX model metadata
        try:
            meta = self._session.get_modelmeta().custom_metadata_map
            desc = (meta.get("description", "") + meta.get("author", "") + meta.get("task", "")).lower()
            if "yolox" in desc:
                return ModelType.YOLOX.value
            if "ultralytics" in desc or "yolov8" in desc or "yolov5" in desc:
                return ModelType.YOLO.value
        except Exception:
            pass

        # 3. Inspect output shapes and names
        outputs = self._session.get_outputs()
        if len(outputs) >= 3:
            has_stride_name = any("stride" in out.name.lower() for out in outputs)
            is_4d = any(len(out.shape) == 4 for out in outputs)
            if has_stride_name or is_4d:
                return ModelType.YOLOX.value

        # Default single output YOLO (v8 / v5)
        return ModelType.YOLO.value

    def load(self) -> None:
        """Initialize ONNX Runtime InferenceSession with preferred providers."""
        if self._session is not None:
            return

        model_path = Path(self.settings.model_path)
        if not model_path.is_file():
            cand_model = get_models_dir() / model_path.name
            if cand_model.is_file():
                model_path = cand_model
            else:
                raise DetectionError(f"ONNX model file not found: {model_path}")

        providers = self._get_providers()

        try:
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [out.name for out in self._session.get_outputs()]
            self.model_type = self._determine_model_type()
            logger.info(
                "Loaded ONNX detector '%s' as %s with %d classes",
                model_path.name,
                self.model_type,
                len(self._classes),
            )
        except Exception as e:
            self._session = None
            raise DetectionError(f"Failed to create ONNX InferenceSession from {model_path.name}: {e}") from e

        # Optional crop classifier initialization
        if self.classifier_type != "none":
            self._load_classifier()

    def close(self) -> None:
        """Release ONNX InferenceSession and free GPU/system memory."""
        self._session = None
        self._input_name = None
        self._output_names = []
        self._classifier_session = None
        self._cls_input_name = None
        self._cls_output_names = []
        if self._explicit_model_type:
            self.model_type = self._explicit_model_type
        else:
            self.model_type = ModelType.AUTO.value

    def detect(self, image: np.ndarray) -> list[DetectedWeed]:
        """Run object detection on an RGB image array and return normalized weed detections."""
        if not self.is_loaded:
            self.load()

        if self._session is None or self._input_name is None:
            raise DetectionError("Inference session is not loaded.")

        orig_h, orig_w = image.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            return []

        # Letterbox preprocessing
        lb = letterbox(image, size=self.settings.input_size)
        pad_x, pad_y = lb.pad
        scale = lb.scale

        try:
            outputs = self._session.run(None, {self._input_name: lb.tensor})
        except Exception as e:
            raise DetectionError(f"Inference execution failed: {e}") from e

        if not outputs:
            return []

        # Decode based on detected architecture
        if self.model_type == ModelType.YOLOX.value:
            boxes_arr, scores_arr, class_ids = _decode_yolox(
                outputs,
                input_size=self.settings.input_size,
                conf_threshold=self.confidence_threshold,
            )
        else:
            boxes_arr, scores_arr, class_ids = _decode_yolo(
                outputs[0],
                conf_threshold=self.confidence_threshold,
                num_classes=len(self._classes),
            )

        if len(boxes_arr) == 0:
            return []

        # Apply Non-Maximum Suppression (class-aware)
        keep_indices = nms(boxes_arr, scores_arr, self.nms_iou_threshold, class_ids=class_ids)

        detected_weeds: list[DetectedWeed] = []
        mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for idx in keep_indices:
            bx1, by1, bx2, by2 = boxes_arr[idx]
            score = float(scores_arr[idx])
            c_id = int(class_ids[idx])
            class_name = self._classes.get(c_id, f"class_{c_id}")

            # Inverse letterbox transformation back to original image space
            orig_x1 = (bx1 - pad_x) / scale
            orig_y1 = (by1 - pad_y) / scale
            orig_x2 = (bx2 - pad_x) / scale
            orig_y2 = (by2 - pad_y) / scale

            # Clip to [0, orig_w] and [0, orig_h]
            orig_x1 = max(0.0, min(float(orig_w), orig_x1))
            orig_y1 = max(0.0, min(float(orig_h), orig_y1))
            orig_x2 = max(orig_x1, min(float(orig_w), orig_x2))
            orig_y2 = max(orig_y1, min(float(orig_h), orig_y2))

            stage_name: str | None = None
            class_prob: float | None = None

            # Optional second-stage crop classification
            if self._classifier_session is not None and self._cls_input_name is not None:
                pad_w = int(round((orig_x2 - orig_x1) * 0.10))
                pad_h = int(round((orig_y2 - orig_y1) * 0.10))
                cx1 = max(0, int(round(orig_x1)) - pad_w)
                cy1 = max(0, int(round(orig_y1)) - pad_h)
                cx2 = min(orig_w, int(round(orig_x2)) + pad_w)
                cy2 = min(orig_h, int(round(orig_y2)) + pad_h)

                if (cx2 - cx1) >= 4 and (cy2 - cy1) >= 4:
                    crop_rgb = image[cy1:cy2, cx1:cx2]
                    target_cls_w, target_cls_h = self.settings.classifier_input_size
                    crop_res = cv2.resize(crop_rgb, (target_cls_w, target_cls_h), interpolation=cv2.INTER_LINEAR)
                    crop_norm = (crop_res.astype(np.float32) / 255.0 - mean_rgb) / std_rgb
                    # NCHW format: (1, 3, target_cls_h, target_cls_w)
                    crop_tensor = np.transpose(crop_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

                    try:
                        cls_outs = self._classifier_session.run(None, {self._cls_input_name: crop_tensor})
                        if self.classifier_type == "c07" and len(cls_outs) >= 2:
                            logits_cat = cls_outs[0][0]
                            logits_stg = cls_outs[1][0]
                            p_cat = np.exp(logits_cat - np.max(logits_cat)) / np.sum(np.exp(logits_cat - np.max(logits_cat)))
                            p_stg = np.exp(logits_stg - np.max(logits_stg)) / np.sum(np.exp(logits_stg - np.max(logits_stg)))
                            cat_id = int(np.argmax(p_cat))
                            stg_id = int(np.argmax(p_stg))

                            class_name = C07_CATEGORIES[cat_id] if cat_id < len(C07_CATEGORIES) else class_name
                            stage_name = C07_STAGES[stg_id] if stg_id < len(C07_STAGES) else "не определена"
                            class_prob = float(p_cat[cat_id])
                            c_id = cat_id
                        elif self.classifier_type == "c06" and len(cls_outs) >= 1:
                            logits = cls_outs[0][0]
                            p = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
                            sp_id = int(np.argmax(p))
                            code, ru_name, _ = C06_TAXONOMY.get(sp_id, (f"S{sp_id:03d}", "Неизвестный сорняк", ""))
                            class_name = f"{ru_name} ({code})"
                            class_prob = float(p[sp_id])
                            c_id = sp_id
                    except Exception as exc:
                        logger.warning("Error running ONNX crop classification: %s", exc)

            # Normalize to [0.0..1.0]
            norm_x = min(1.0, max(0.0, orig_x1 / orig_w))
            norm_y = min(1.0, max(0.0, orig_y1 / orig_h))
            norm_w = min(1.0 - norm_x, max(0.0, (orig_x2 - orig_x1) / orig_w))
            norm_h = min(1.0 - norm_y, max(0.0, (orig_y2 - orig_y1) / orig_h))

            bbox = BBox(x=norm_x, y=norm_y, w=norm_w, h=norm_h)
            weed_obj = DetectedWeed(
                weed_id=f"det_{len(detected_weeds)}",
                class_id=c_id,
                class_name=class_name,
                confidence=round(score, 4),
                bbox=bbox,
                stage_name=stage_name,
                class_prob=round(class_prob, 4) if class_prob is not None else None,
            )
            detected_weeds.append(weed_obj)

        return detected_weeds
