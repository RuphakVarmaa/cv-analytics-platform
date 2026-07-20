"""
YOLOv8 ONNX ModelManager — handles model loading, export, warmup,
pre/post-processing, and COCO class lookup.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class names (80 classes, indices 0-79)
# ---------------------------------------------------------------------------
COCO_CLASSES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

INPUT_SIZE = 640  # YOLOv8 default input resolution


def get_coco_classes() -> dict[int, str]:
    """Return a copy of the COCO class-name mapping."""
    return dict(COCO_CLASSES)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
    """
    Letterbox-resize *img* (H×W×3 BGR uint8) to INPUT_SIZE×INPUT_SIZE,
    normalise to [0, 1], convert to NCHW float32.

    Returns
    -------
    blob          : np.ndarray, shape (1, 3, INPUT_SIZE, INPUT_SIZE), dtype float32
    original_size : (orig_h, orig_w) before any resize
    pad           : (pad_top, pad_left) applied during letterbox
    """
    orig_h, orig_w = img.shape[:2]

    # Scale factor that fits both dimensions within INPUT_SIZE
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding to reach INPUT_SIZE × INPUT_SIZE (grey = 114)
    pad_top = (INPUT_SIZE - new_h) // 2
    pad_left = (INPUT_SIZE - new_w) // 2
    pad_bottom = INPUT_SIZE - new_h - pad_top
    pad_right = INPUT_SIZE - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )

    # BGR -> RGB, uint8 -> float32 [0, 1], HWC -> NCHW
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    return blob, (orig_h, orig_w), (pad_top, pad_left)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] -> [x1, y1, x2, y2]."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def postprocess(
    outputs: List[np.ndarray],
    original_shape: Tuple[int, int],
    pad: Tuple[int, int],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
) -> List[dict]:
    """
    Decode raw ONNX outputs from YOLOv8 into a list of detection dicts.

    Parameters
    ----------
    outputs        : list of arrays as returned by onnxruntime session.run()
                     YOLOv8 exports a single output of shape (1, 84, 8400).
    original_shape : (orig_h, orig_w) of the image fed to preprocess().
    pad            : (pad_top, pad_left) returned by preprocess().
    conf_thresh    : minimum object confidence score to keep.
    iou_thresh     : IoU threshold for NMS.

    Returns
    -------
    List of dicts with keys: class_id, class_name, confidence, bbox (normalized).
    """
    # YOLOv8 ONNX output shape: (1, 84, 8400) — 4 box coords + 80 class scores
    raw = outputs[0]  # (1, 84, 8400) or (1, num_classes+4, num_anchors)
    if raw.ndim == 3:
        raw = raw[0]  # (84, 8400)

    # Transpose to (8400, 84)
    preds = raw.T  # (8400, 84)

    boxes_xywh = preds[:, :4]      # (8400, 4) — cx, cy, w, h in INPUT_SIZE coords
    class_scores = preds[:, 4:]    # (8400, 80)

    class_ids = np.argmax(class_scores, axis=1)          # (8400,)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]  # (8400,)

    # Confidence filter
    mask = confidences >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # Convert to [x1, y1, x2, y2] in INPUT_SIZE pixel space
    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    # NMS per class using cv2
    orig_h, orig_w = original_shape
    pad_top, pad_left = pad
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)

    detections = []
    unique_classes = np.unique(class_ids)
    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_boxes = boxes_xyxy[cls_mask]
        cls_confs = confidences[cls_mask]
        cls_ids_subset = class_ids[cls_mask]

        # cv2 NMS expects [x, y, w, h] integers
        nms_boxes = []
        for b in cls_boxes:
            x1, y1, x2, y2 = b
            nms_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            cls_confs.tolist(),
            conf_thresh,
            iou_thresh,
        )
        if len(indices) == 0:
            continue

        # cv2.dnn.NMSBoxes may return shape (N, 1) or (N,)
        if hasattr(indices, "flatten"):
            indices = indices.flatten()

        for idx in indices:
            b = cls_boxes[idx]
            x1, y1, x2, y2 = b

            # Remove letterbox padding and scale back to original image coords
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale

            # Clamp to [0, original dimension]
            x1 = max(0.0, min(float(x1), float(orig_w)))
            y1 = max(0.0, min(float(y1), float(orig_h)))
            x2 = max(0.0, min(float(x2), float(orig_w)))
            y2 = max(0.0, min(float(y2), float(orig_h)))

            # Normalise to [0, 1]
            nx1 = x1 / orig_w
            ny1 = y1 / orig_h
            nx2 = x2 / orig_w
            ny2 = y2 / orig_h

            class_id = int(cls_ids_subset[idx])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": COCO_CLASSES.get(class_id, f"class_{class_id}"),
                    "confidence": float(cls_confs[idx]),
                    "bbox": {
                        "x1": nx1,
                        "y1": ny1,
                        "x2": nx2,
                        "y2": ny2,
                    },
                }
            )

    return detections


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Manages the YOLOv8n ONNX model lifecycle:
    - Downloading / exporting from Ultralytics
    - Loading into ONNXRuntime
    - Warmup inferences
    """

    def __init__(
        self,
        model_path: str | None = None,
        pt_model_name: str = "yolov8n.pt",
    ) -> None:
        self.model_path = Path(
            model_path or os.environ.get("MODEL_PATH", "/app/models/yolov8n.onnx")
        )
        self.pt_model_name = pt_model_name
        self.session = None
        self.input_name: str = ""
        self.model_loaded: bool = False
        self.warmup_done: bool = False
        self.device: str = "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed), export to ONNX, load into ONNXRuntime."""
        self._ensure_onnx_model()
        self._load_onnxruntime_session()
        logger.info("Model loaded from %s", self.model_path)

    def warmup(self, n: int = 5) -> None:
        """Run *n* dummy inferences to warm up the ONNX runtime."""
        if self.session is None:
            raise RuntimeError("Model not loaded; call load() first.")
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for i in range(n):
            t0 = time.perf_counter()
            self.session.run(None, {self.input_name: dummy})
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Warmup inference %d/%d: %.1f ms", i + 1, n, elapsed)
        self.warmup_done = True
        logger.info("Warmup complete (%d inferences).", n)

    def infer(self, blob: np.ndarray) -> List[np.ndarray]:
        """Run inference and return raw ONNX outputs."""
        if self.session is None:
            raise RuntimeError("Model not loaded.")
        return self.session.run(None, {self.input_name: blob})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_onnx_model(self) -> None:
        """Export YOLOv8n to ONNX if the file does not already exist."""
        if self.model_path.exists():
            logger.info("ONNX model already cached at %s", self.model_path)
            return

        logger.info(
            "ONNX model not found at %s — downloading and exporting from Ultralytics…",
            self.model_path,
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from ultralytics import YOLO  # type: ignore

            yolo = YOLO(self.pt_model_name)
            # Export to ONNX; fp16=False for CPU compatibility
            export_path = yolo.export(
                format="onnx",
                imgsz=INPUT_SIZE,
                dynamic=False,
                simplify=True,
                opset=12,
                fp16=False,
            )
            exported = Path(export_path)
            if not exported.exists():
                raise FileNotFoundError(
                    f"ONNX export succeeded but file not found at {export_path}"
                )
            # Move to the desired location if different
            if exported.resolve() != self.model_path.resolve():
                import shutil
                shutil.move(str(exported), str(self.model_path))
            logger.info("ONNX model exported to %s", self.model_path)
        except Exception as exc:
            logger.exception("Failed to export ONNX model: %s", exc)
            raise

    def _load_onnxruntime_session(self) -> None:
        import onnxruntime as ort  # type: ignore

        # Prefer CUDA if available, fall back to CPU
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "cuda"
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu"

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.model_loaded = True
        logger.info(
            "ONNXRuntime session ready. Device=%s, input_name=%s",
            self.device,
            self.input_name,
        )
