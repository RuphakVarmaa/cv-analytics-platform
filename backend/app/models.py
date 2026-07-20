from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResult(BaseModel):
    frame_id: str
    timestamp_ms: int
    detections: List[DetectionBox]
    inference_time_ms: float


class TelemetrySnapshot(BaseModel):
    session_id: str
    recorded_at: datetime
    fps: float
    p50_ms: float
    p95_ms: float
    queue_depth: int
    class_counts: Dict[str, int]


class Session(BaseModel):
    id: str
    user_id: Optional[str]
    source_type: str
    source_url: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    total_frames: int = 0
    dropped_frames: int = 0
    total_detections: int = 0
    is_deleted: bool = False


class SessionCreate(BaseModel):
    source_type: str = Field(..., pattern="^(demo|webcam|rtsp)$")
    source_url: Optional[str] = None


class PaginatedSessions(BaseModel):
    items: List[Session]
    total: int
    page: int
    per_page: int
    pages: int


class MetricsPayload(BaseModel):
    type: str = "METRICS_UPDATE"
    fps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    queue_depth: int
    drop_rate: float
    total_detections: int
    active_connections: int


class QueueStatusPayload(BaseModel):
    type: str = "QUEUE_STATUS"
    depth: int
    warning: bool


class FrameResultPayload(BaseModel):
    type: str = "FRAME_RESULT"
    frame_id: str
    timestamp_ms: int
    detections: List[Dict[str, Any]]
    inference_time_ms: float
