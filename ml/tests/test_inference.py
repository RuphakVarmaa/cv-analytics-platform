"""
pytest test suite for the YOLOv8 ONNX inference microservice.

Tests:
  1.  DetectionResult / BBox / Detection schema validation (Pydantic)
  2.  preprocess() produces correct output shape (1, 3, 640, 640)
  3.  postprocess() with mock ONNX outputs returns correct Detection dicts
  4.  GET /health returns HTTP 200 with expected JSON keys
  5.  POST /infer with a synthetic JPEG returns a valid DetectionResult JSON

Run with:
    pytest tests/test_inference.py -v
"""
from __future__ import annotations

import io
import sys
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ── Make sure the project root is on sys.path ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Imports from the service ──────────────────────────────────────────────────
from app.model import (
    INPUT_SIZE,
    ModelManager,
    get_coco_classes,
    postprocess,
    preprocess,
)
from app.main import BBox, Detection, DetectionResult


# =============================================================================
# 1. Pydantic schema validation
# =============================================================================

class TestSchemas:
    def test_bbox_valid(self):
        bbox = BBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
        assert bbox.x1 == pytest.approx(0.1)
        assert bbox.y2 == pytest.approx(0.8)

    def test_bbox_zero_area(self):
        """A degenerate box (point) should still validate."""
        bbox = BBox(x1=0.5, y1=0.5, x2=0.5, y2=0.5)
        assert bbox.x1 == bbox.x2

    def test_detection_valid(self):
        det = Detection(
            class_name="person",
            confidence=0.87,
            bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
        )
        assert det.class_name == "person"
        assert det.track_id is None

    def test_detection_with_track_id(self):
        det = Detection(
            class_name="car",
            confidence=0.55,
            bbox=BBox(x1=0.1, y1=0.1, x2=0.4, y2=0.4),
            track_id=42,
        )
        assert det.track_id == 42

    def test_detection_result_valid(self):
        result = DetectionResult(
            frame_id="abc-123",
            inference_time_ms=15.7,
            detections=[
                Detection(
                    class_name="dog",
                    confidence=0.92,
                    bbox=BBox(x1=0.2, y1=0.3, x2=0.6, y2=0.9),
                )
            ],
        )
        assert result.model_version == "yolov8n-onnx"
        assert len(result.detections) == 1

    def test_detection_result_empty_detections(self):
        result = DetectionResult(
            frame_id="empty",
            inference_time_ms=8.0,
            detections=[],
        )
        assert result.detections == []

    def test_detection_result_serialises_to_dict(self):
        result = DetectionResult(
            frame_id="x",
            inference_time_ms=10.0,
            detections=[],
        )
        d = result.model_dump()
        assert "frame_id" in d
        assert "detections" in d
        assert isinstance(d["detections"], list)

    def test_get_coco_classes_returns_80_entries(self):
        classes = get_coco_classes()
        assert len(classes) == 80
        assert classes[0] == "person"
        assert classes[79] == "toothbrush"


# =============================================================================
# 2. preprocess() — shape checks
# =============================================================================

class TestPreprocess:
    def _make_bgr(self, h: int, w: int) -> np.ndarray:
        """Return a random BGR uint8 image."""
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_output_shape_square(self):
        img = self._make_bgr(640, 640)
        blob, orig_shape, pad = preprocess(img)
        assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_output_shape_portrait(self):
        img = self._make_bgr(1080, 720)
        blob, orig_shape, pad = preprocess(img)
        assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_output_shape_landscape(self):
        img = self._make_bgr(480, 1280)
        blob, orig_shape, pad = preprocess(img)
        assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_output_shape_small_image(self):
        img = self._make_bgr(32, 32)
        blob, orig_shape, pad = preprocess(img)
        assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_dtype_is_float32(self):
        img = self._make_bgr(224, 224)
        blob, _, _ = preprocess(img)
        assert blob.dtype == np.float32

    def test_values_normalised_0_1(self):
        img = self._make_bgr(300, 400)
        blob, _, _ = preprocess(img)
        assert float(blob.min()) >= 0.0
        assert float(blob.max()) <= 1.0

    def test_original_shape_preserved(self):
        h, w = 720, 1280
        img = self._make_bgr(h, w)
        _, orig_shape, _ = preprocess(img)
        assert orig_shape == (h, w)

    def test_pad_non_negative(self):
        img = self._make_bgr(300, 200)
        _, _, pad = preprocess(img)
        assert pad[0] >= 0 and pad[1] >= 0


# =============================================================================
# 3. postprocess() with mock ONNX outputs
# =============================================================================

class TestPostprocess:
    """
    Build synthetic ONNX output (1, 84, 8400) that mimics YOLOv8 format.
    We inject a single high-confidence box and verify it is returned.
    """

    # YOLOv8 output shape
    NUM_ANCHORS = 8400
    NUM_CLASSES = 80

    def _make_empty_output(self) -> List[np.ndarray]:
        """All-zero output — no detections expected."""
        raw = np.zeros((1, 4 + self.NUM_CLASSES, self.NUM_ANCHORS), dtype=np.float32)
        return [raw]

    def _make_single_detection_output(
        self,
        cx: float = 320.0,
        cy: float = 320.0,
        w: float = 100.0,
        h: float = 80.0,
        class_id: int = 0,
        conf: float = 0.9,
    ) -> List[np.ndarray]:
        """
        Output with a single above-threshold box for *class_id*.
        All other anchors are zero.
        """
        raw = np.zeros((1, 4 + self.NUM_CLASSES, self.NUM_ANCHORS), dtype=np.float32)
        # Box at anchor index 0
        raw[0, 0, 0] = cx
        raw[0, 1, 0] = cy
        raw[0, 2, 0] = w
        raw[0, 3, 0] = h
        raw[0, 4 + class_id, 0] = conf
        return [raw]

    def test_empty_output_returns_empty_list(self):
        outputs = self._make_empty_output()
        dets = postprocess(outputs, (640, 640), (0, 0))
        assert dets == []

    def test_single_detection_returned(self):
        outputs = self._make_single_detection_output(
            cx=320.0, cy=320.0, w=200.0, h=150.0,
            class_id=0, conf=0.9,
        )
        dets = postprocess(outputs, (640, 640), (0, 0), conf_thresh=0.25)
        assert len(dets) == 1
        d = dets[0]
        assert d["class_name"] == "person"
        assert d["confidence"] == pytest.approx(0.9, abs=1e-4)

    def test_detection_bbox_normalised(self):
        outputs = self._make_single_detection_output(
            cx=320.0, cy=320.0, w=200.0, h=150.0,
            class_id=2, conf=0.88,
        )
        dets = postprocess(outputs, (640, 640), (0, 0), conf_thresh=0.25)
        assert len(dets) == 1
        bbox = dets[0]["bbox"]
        assert 0.0 <= bbox["x1"] <= 1.0
        assert 0.0 <= bbox["y1"] <= 1.0
        assert 0.0 <= bbox["x2"] <= 1.0
        assert 0.0 <= bbox["y2"] <= 1.0
        assert bbox["x1"] < bbox["x2"]
        assert bbox["y1"] < bbox["y2"]

    def test_below_threshold_not_returned(self):
        outputs = self._make_single_detection_output(
            cx=100.0, cy=100.0, w=50.0, h=50.0,
            class_id=1, conf=0.10,  # below default 0.25
        )
        dets = postprocess(outputs, (640, 640), (0, 0), conf_thresh=0.25)
        assert dets == []

    def test_class_name_lookup(self):
        outputs = self._make_single_detection_output(class_id=16, conf=0.95)
        dets = postprocess(outputs, (640, 640), (0, 0))
        assert len(dets) == 1
        assert dets[0]["class_name"] == "dog"

    def test_correct_class_id_in_result(self):
        outputs = self._make_single_detection_output(class_id=63, conf=0.95)
        dets = postprocess(outputs, (640, 640), (0, 0))
        assert len(dets) == 1
        assert dets[0]["class_id"] == 63
        assert dets[0]["class_name"] == "laptop"

    def test_with_letterbox_padding(self):
        """
        Simulate a 480×640 original image padded to 640×640.
        pad_top=80, pad_left=0.  Box at centre should still map into [0,1].
        """
        outputs = self._make_single_detection_output(
            cx=320.0, cy=320.0, w=100.0, h=100.0,
            class_id=0, conf=0.95,
        )
        # Original 480h x 640w image, scale = min(640/640, 640/480) = 1.0 (w) or 1.333 (h)
        # scale = 1.0 → new_h=480, pad_top=(640-480)//2=80, pad_left=0
        dets = postprocess(outputs, (480, 640), (80, 0), conf_thresh=0.25)
        assert len(dets) == 1
        bbox = dets[0]["bbox"]
        for coord in (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]):
            assert 0.0 <= coord <= 1.0


# =============================================================================
# 4 & 5. FastAPI endpoint tests (async, with TestClient)
# =============================================================================

def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return a minimal valid JPEG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8), mode="RGB"
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """
    Provide a synchronous TestClient with a mocked ModelManager so no
    real model download / ONNX session is needed during testing.
    """
    from fastapi.testclient import TestClient
    import app.main as main_module

    # Patch model_manager on the module so the lifespan code is bypassed
    mock_manager = MagicMock(spec=ModelManager)
    mock_manager.model_loaded = True
    mock_manager.warmup_done = True
    mock_manager.device = "cpu"

    # infer returns zero-output array (no detections)
    empty_output = np.zeros((1, 84, 8400), dtype=np.float32)
    mock_manager.infer.return_value = [empty_output]

    with patch.object(main_module, "model_manager", mock_manager):
        # Skip the lifespan startup completely
        app_no_lifespan = main_module.app
        with TestClient(app_no_lifespan, raise_server_exceptions=True) as c:
            yield c, mock_manager


class TestEndpoints:
    def test_health_returns_200(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        tc, _ = client
        data = tc.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "model_version" in data
        assert "warmup_done" in data

    def test_health_model_loaded_true(self, client):
        tc, _ = client
        data = tc.get("/health").json()
        assert data["model_loaded"] is True

    def test_infer_with_multipart_jpeg(self, client):
        tc, _ = client
        jpeg = _make_jpeg_bytes(64, 64)
        resp = tc.post(
            "/infer",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_infer_response_is_detection_result(self, client):
        tc, _ = client
        jpeg = _make_jpeg_bytes(128, 128)
        resp = tc.post(
            "/infer",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        data = resp.json()
        assert "frame_id" in data
        assert "inference_time_ms" in data
        assert "detections" in data
        assert "model_version" in data
        assert isinstance(data["detections"], list)

    def test_infer_model_version_field(self, client):
        tc, _ = client
        jpeg = _make_jpeg_bytes()
        resp = tc.post(
            "/infer",
            files={"file": ("f.jpg", jpeg, "image/jpeg")},
        )
        assert resp.json()["model_version"] == "yolov8n-onnx"

    def test_infer_inference_time_positive(self, client):
        tc, _ = client
        jpeg = _make_jpeg_bytes()
        resp = tc.post(
            "/infer",
            files={"file": ("f.jpg", jpeg, "image/jpeg")},
        )
        assert resp.json()["inference_time_ms"] >= 0.0

    def test_infer_with_raw_body(self, client):
        tc, _ = client
        jpeg = _make_jpeg_bytes(32, 32)
        resp = tc.post(
            "/infer",
            content=jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
        assert resp.status_code == 200

    def test_infer_empty_body_returns_422(self, client):
        tc, _ = client
        resp = tc.post(
            "/infer",
            content=b"",
            headers={"Content-Type": "image/jpeg"},
        )
        assert resp.status_code == 422

    def test_infer_invalid_bytes_returns_422(self, client):
        tc, _ = client
        resp = tc.post(
            "/infer",
            content=b"not-an-image!!!",
            headers={"Content-Type": "image/jpeg"},
        )
        assert resp.status_code == 422

    def test_infer_detections_when_model_returns_hits(self, client):
        """
        Inject a mock ONNX output containing one detection and verify
        that /infer propagates it into the DetectionResult.
        """
        tc, mock_manager = client

        # Single high-confidence person box at centre
        raw = np.zeros((1, 84, 8400), dtype=np.float32)
        raw[0, 0, 0] = 320.0  # cx
        raw[0, 1, 0] = 320.0  # cy
        raw[0, 2, 0] = 200.0  # w
        raw[0, 3, 0] = 150.0  # h
        raw[0, 4, 0] = 0.9    # person class score
        mock_manager.infer.return_value = [raw]

        jpeg = _make_jpeg_bytes(640, 640)
        resp = tc.post(
            "/infer",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["detections"]) >= 1
        det = data["detections"][0]
        assert det["class_name"] == "person"
        assert 0.0 < det["confidence"] <= 1.0
        assert "bbox" in det
