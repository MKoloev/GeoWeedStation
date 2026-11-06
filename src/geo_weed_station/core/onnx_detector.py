"""ONNX Runtime based neural network detector for weed identification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from geo_weed_station.config.settings import DetectorSettings
from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.exceptions import DetectionError
from geo_weed_station.schemas.entities import BBox, DetectedWeed


@dataclass(frozen=True)
class LetterboxResult:
    """Result of image letterbox preprocessing for neural network input."""

    tensor: np.ndarray  # Shape: (1, 3, target_h, target_w), dtype float32, normalized [0..1]
    scale: float
    pad: Tuple[float, float]  # (pad_x, pad_y)


def letterbox(
    image: np.ndarray,
    size: Tuple[int, int] = (640, 640),
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

    return LetterboxResult(tensor=tensor, scale=scale, pad=(pad_x, pad_y))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Non-Maximum Suppression (NMS) on bounding boxes.
    
    Args:
        boxes: Numpy array of shape (N, 4) in format [x1, y1, x2, y2].
        scores: Numpy array of shape (N,) confidence scores.
        iou_thr: IoU threshold above which overlapping boxes are suppressed.
        
    Returns:
        List of retained box indices.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep: List[int] = []

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


def load_class_names(path: Path) -> Dict[int, str]:
    """Load class ID to class name mapping from JSON file."""
    path = Path(path)
    if not path.is_file():
        return {0: "weed_dicot", 1: "weed_monocot"}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes: Dict[int, str] = {}
        for k, v in data.items():
            try:
                classes[int(k)] = str(v)
            except ValueError:
                pass
        return classes if classes else {0: "weed_dicot", 1: "weed_monocot"}
    except Exception:
        return {0: "weed_dicot", 1: "weed_monocot"}


class OnnxDetector(BaseDetector):
    """YOLO detector implementation running on ONNX Runtime."""

    def __init__(
        self,
        settings: DetectorSettings,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.settings = settings
        self._custom_providers = list(providers) if providers is not None else None
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_names: List[str] = []
        self._classes = load_class_names(self.settings.classes_path)
        self.confidence_threshold = float(self.settings.confidence_threshold)
        self.nms_iou_threshold = float(self.settings.nms_iou_threshold)

    @staticmethod
    def available_providers() -> List[str]:
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

    def load(self) -> None:
        """Initialize ONNX Runtime InferenceSession with preferred providers."""
        if self._session is not None:
            return

        model_path = Path(self.settings.model_path)
        if not model_path.is_file():
            raise DetectionError(f"ONNX model file not found: {model_path}")

        if self._custom_providers:
            providers = self._custom_providers
        else:
            avail = ort.get_available_providers()
            # Prioritize hardware acceleration if present
            preferred_order = [
                "CUDAExecutionProvider",
                "ROCMExecutionProvider",
                "TensorrtExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preferred_order if p in avail]
            if not providers:
                providers = ["CPUExecutionProvider"]

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
        except Exception as e:
            self._session = None
            raise DetectionError(f"Failed to create ONNX InferenceSession from {model_path.name}: {e}") from e

    def close(self) -> None:
        """Release ONNX InferenceSession and free GPU/system memory."""
        self._session = None
        self._input_name = None
        self._output_names = []

    def detect(self, image: np.ndarray) -> List[DetectedWeed]:
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

        # Standard YOLO output shape: (1, num_channels, num_boxes) or (1, num_boxes, num_channels)
        preds = outputs[0]
        if preds.ndim == 3:
            # If shape is (1, C, N) where C < N (e.g. YOLOv8 1x6x8400), transpose to (1, N, C)
            if preds.shape[1] < preds.shape[2]:
                preds = np.transpose(preds, (0, 2, 1))
            preds = preds[0]  # (N, C)
        elif preds.ndim == 2:
            pass
        else:
            return []

        num_proposals, num_features = preds.shape
        if num_features < 5:
            return []

        boxes_xyxy_letterbox: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        # Identify YOLO format (v8 vs v5)
        # In YOLOv8: cols 0..3 are cx, cy, w, h; cols 4.. are class logits
        # In YOLOv5: cols 0..3 are cx, cy, w, h; col 4 is obj_conf; cols 5.. are class scores
        has_objectness = num_features > (len(self._classes) + 4) and len(self._classes) > 0

        for row in preds:
            cx, cy, bw, bh = row[0:4]

            if has_objectness:
                obj_conf = float(row[4])
                class_scores = row[5:] * obj_conf
            else:
                class_scores = row[4:]

            best_class_id = int(np.argmax(class_scores))
            score = float(class_scores[best_class_id])

            if score < self.confidence_threshold:
                continue

            # Convert center-size to corner coordinates in letterbox canvas
            bx1 = cx - bw / 2.0
            by1 = cy - bh / 2.0
            bx2 = cx + bw / 2.0
            by2 = cy + bh / 2.0

            boxes_xyxy_letterbox.append([bx1, by1, bx2, by2])
            confidences.append(score)
            class_ids.append(best_class_id)

        if not boxes_xyxy_letterbox:
            return []

        boxes_arr = np.array(boxes_xyxy_letterbox, dtype=np.float32)
        scores_arr = np.array(confidences, dtype=np.float32)

        # Apply Non-Maximum Suppression
        keep_indices = nms(boxes_arr, scores_arr, self.nms_iou_threshold)

        detected_weeds: List[DetectedWeed] = []
        for idx in keep_indices:
            bx1, by1, bx2, by2 = boxes_arr[idx]
            score = float(scores_arr[idx])
            c_id = class_ids[idx]
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
            )
            detected_weeds.append(weed_obj)

        return detected_weeds
