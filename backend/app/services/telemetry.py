import asyncio
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from prometheus_client import Counter, Gauge
import structlog

from app.models import TelemetrySnapshot

logger = structlog.get_logger(__name__)

# Prometheus metrics (module-level singletons)
frames_processed_total = Counter(
    "frames_processed_total",
    "Total number of video frames processed",
    ["session_id"],
)
detections_total = Counter(
    "detections_total",
    "Total number of object detections",
    ["session_id", "class_name"],
)
active_sessions_gauge = Gauge(
    "active_sessions",
    "Number of currently active streaming sessions",
)


class TelemetryService:
    """
    In-memory telemetry store with Redis pub/sub fanout.

    Stores up to 3600 TelemetrySnapshot objects per session (1 per second = 1 hour).
    Publishes each snapshot to a Redis channel so external consumers can subscribe.
    """

    def __init__(self, redis_client=None):
        # session_id -> deque of TelemetrySnapshot
        self._snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3600))
        self._redis = redis_client
        self._lock = asyncio.Lock()

    def set_redis(self, redis_client) -> None:
        self._redis = redis_client

    async def record_snapshot(
        self,
        session_id: str,
        fps: float,
        latencies: List[float],
        queue_depth: int,
        class_counts: Dict[str, int],
        p50_ms: float = 0.0,
        p95_ms: float = 0.0,
    ) -> TelemetrySnapshot:
        """Record a telemetry snapshot and publish to Redis."""
        import numpy as np

        if latencies:
            arr = [float(x) for x in latencies]
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
        else:
            p50 = p50_ms
            p95 = p95_ms

        snap = TelemetrySnapshot(
            session_id=session_id,
            recorded_at=datetime.now(tz=timezone.utc),
            fps=fps,
            p50_ms=p50,
            p95_ms=p95,
            queue_depth=queue_depth,
            class_counts=class_counts,
        )

        async with self._lock:
            self._snapshots[session_id].append(snap)

        # Update Prometheus counters
        total_detections = sum(class_counts.values())
        frames_processed_total.labels(session_id=session_id).inc()
        for cls_name, count in class_counts.items():
            detections_total.labels(session_id=session_id, class_name=cls_name).inc(count)

        # Publish to Redis channel
        if self._redis is not None:
            channel = f"session:{session_id}"
            payload = json.dumps({
                "fps": snap.fps,
                "p50_ms": snap.p50_ms,
                "p95_ms": snap.p95_ms,
                "queue_depth": snap.queue_depth,
                "class_counts": snap.class_counts,
                "recorded_at": snap.recorded_at.isoformat(),
            })
            try:
                await self._redis.publish(channel, payload)
            except Exception as exc:
                logger.warning("redis_publish_failed", channel=channel, error=str(exc))

        return snap

    async def get_snapshots(
        self, session_id: str, limit: int = 60
    ) -> List[TelemetrySnapshot]:
        """Return the most recent `limit` snapshots for a session."""
        async with self._lock:
            snaps = list(self._snapshots.get(session_id, []))
        return snaps[-limit:]

    def increment_active_sessions(self) -> None:
        active_sessions_gauge.inc()

    def decrement_active_sessions(self) -> None:
        active_sessions_gauge.dec()

    async def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all active sessions."""
        async with self._lock:
            all_sessions = dict(self._snapshots)

        if not all_sessions:
            return {
                "total_fps": 0.0,
                "avg_p50_ms": 0.0,
                "avg_p95_ms": 0.0,
                "total_detections": 0,
                "active_sessions": 0,
            }

        fps_values = []
        p50_values = []
        p95_values = []
        total_dets = 0

        for snaps in all_sessions.values():
            if snaps:
                last = snaps[-1]
                fps_values.append(last.fps)
                p50_values.append(last.p50_ms)
                p95_values.append(last.p95_ms)
                total_dets += sum(last.class_counts.values())

        return {
            "total_fps": sum(fps_values),
            "avg_p50_ms": sum(p50_values) / len(p50_values) if p50_values else 0.0,
            "avg_p95_ms": sum(p95_values) / len(p95_values) if p95_values else 0.0,
            "total_detections": total_dets,
            "active_sessions": len(all_sessions),
        }


# Global singleton, wired up in main.py lifespan
telemetry_service = TelemetryService()
