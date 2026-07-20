"""
YOLOv8 ONNX Inference Microservice
FastAPI app — POST /infer, GET /health
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from app.model import ModelManager, get_coco_classes, postprocess, preprocess

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MODEL_VERSION = "yolov8n-onnx"
MAX_EXECUTOR_WORKERS = 4
WARMUP_INFERENCES = 5

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float  # all normalised [0, 1]


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BBox
    track_id: Optional[int] = None


class DetectionResult(BaseModel):
    frame_id: str
    inference_time_ms: float
    detections: List[Detection]
    model_version: str = MODEL_VERSION


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: str
    warmup_done: bool


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model_manager: ModelManager = ModelManager()
executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=MAX_EXECUTOR_WORKERS)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
      1. Download + export YOLOv8n to ONNX if not cached.
      2. Load ONNXRuntime session.
      3. Run warmup inferences.
      4. Log device info.
    On shutdown: graceful executor shutdown.
    """
    logger.info("=== ML Inference Service — starting up ===")
    try:
        loop = asyncio.get_event_loop()

        # Load model (blocking I/O; run in thread so event loop stays alive)
        await loop.run_in_executor(executor, model_manager.load)
        logger.info("Device: %s", model_manager.device)

        # Warmup
        await loop.run_in_executor(
            executor, lambda: model_manager.warmup(WARMUP_INFERENCES)
        )

        logger.info("=== Service ready ===")
    except Exception:
        logger.exception("Failed to initialise model — service will return 503")

    yield

    logger.info("=== ML Inference Service — shutting down ===")
    executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="YOLOv8 Inference Service",
    description="Production ONNX-accelerated object detection microservice.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper: decode image bytes -> numpy BGR array
# ---------------------------------------------------------------------------

def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw JPEG/PNG bytes to a BGR numpy array via Pillow."""
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Convert RGB -> BGR (OpenCV convention)
    arr = np.array(pil_img, dtype=np.uint8)
    bgr = arr[:, :, ::-1].copy()
    return bgr


# ---------------------------------------------------------------------------
# Helper: synchronous inference pipeline (run in executor)
# ---------------------------------------------------------------------------

def _run_inference(image_bytes: bytes) -> tuple[List[dict], float]:
    """
    Decode image, preprocess, run ONNX inference, postprocess.
    Returns (detections_list, inference_time_ms).
    """
    bgr = _decode_image(image_bytes)

    blob, original_shape, pad = preprocess(bgr)

    t0 = time.perf_counter()
    outputs = model_manager.infer(blob)
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    detections = postprocess(
        outputs,
        original_shape,
        pad,
        conf_thresh=CONF_THRESHOLD,
        iou_thresh=IOU_THRESHOLD,
    )
    return detections, inference_time_ms


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["meta"],
)
async def health() -> HealthResponse:
    """Returns service health and model metadata."""
    return HealthResponse(
        status="ok" if model_manager.model_loaded else "degraded",
        model_loaded=model_manager.model_loaded,
        device=model_manager.device,
        model_version=MODEL_VERSION,
        warmup_done=model_manager.warmup_done,
    )


@app.post(
    "/infer",
    response_model=DetectionResult,
    summary="Run YOLOv8 object detection",
    tags=["inference"],
    status_code=status.HTTP_200_OK,
)
async def infer(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
) -> DetectionResult:
    """
    Accept image data as either:
    - multipart/form-data with field name ``file``
    - raw request body (application/octet-stream or image/jpeg)

    Returns a DetectionResult containing all detections above the
    confidence threshold.
    """
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not yet loaded. Please retry in a few seconds.",
        )

    # --- Read image bytes ---
    if file is not None:
        image_bytes = await file.read()
    else:
        # Raw body
        image_bytes = await request.body()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No image data received. Send as multipart file or raw body.",
        )

    # --- Run inference in thread pool ---
    loop = asyncio.get_event_loop()
    try:
        detections_raw, inference_time_ms = await loop.run_in_executor(
            executor, _run_inference, image_bytes
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. See server logs.",
        ) from exc

    # --- Build response ---
    detections = [
        Detection(
            class_name=d["class_name"],
            confidence=round(d["confidence"], 6),
            bbox=BBox(
                x1=round(d["bbox"]["x1"], 6),
                y1=round(d["bbox"]["y1"], 6),
                x2=round(d["bbox"]["x2"], 6),
                y2=round(d["bbox"]["y2"], 6),
            ),
        )
        for d in detections_raw
    ]

    return DetectionResult(
        frame_id=str(uuid.uuid4()),
        inference_time_ms=round(inference_time_ms, 3),
        detections=detections,
        model_version=MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Exception handler: catch-all JSON error body
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )
