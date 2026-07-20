import asyncio
import json
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import msgpack
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.middleware.auth import validate_token
from app.models import DetectionBox
from app.services.frame_capture import capture_frames, FrameCaptureError
from app.services.inference_client import InferenceClient, InferenceClientError
from app.services.queue_manager import QueueManager
from app.services.telemetry import telemetry_service
from app.state import app_state

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["stream"])

# Track active connection count
_active_connections: int = 0


@router.websocket("/ws/stream")
async def stream_endpoint(
    websocket: WebSocket,
    token: Optional[str] = None,
    source: str = "demo",
    source_url: Optional[str] = None,
):
    global _active_connections

    # ── 1. Authenticate ──────────────────────────────────────────────────────
    user_id: Optional[str] = None
    if source == "demo" and token is None:
        # Demo mode: unauthenticated access allowed
        user_id = None
    elif token is not None:
        try:
            payload = validate_token(token)
            user_id = payload.get("sub")
        except Exception:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
    else:
        await websocket.close(code=4001, reason="Authentication required")
        return

    # ── 2. Capacity check ────────────────────────────────────────────────────
    if _active_connections >= settings.MAX_CONCURRENT_SESSIONS:
        await websocket.close(code=4003, reason="Server at capacity")
        return

    await websocket.accept()
    _active_connections += 1
    telemetry_service.increment_active_sessions()

    # ── 3. Create DB session ─────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    session_store = app_state.get("session_store")
    if session_store is not None:
        try:
            session_id = await session_store.create_session(
                user_id=user_id,
                source_type=source,
                source_url=source_url,
            )
        except Exception as exc:
            logger.warning("session_create_failed", error=str(exc))

    log = logger.bind(session_id=session_id, source=source)
    log.info("ws_session_started")

    # ── 4. Per-session state ──────────────────────────────────────────────────
    queue = QueueManager(maxsize=settings.QUEUE_MAXSIZE)
    inference_client = InferenceClient()
    await inference_client.start()

    # Latency circular buffer for this session (separate from client's global)
    latency_buf: deque = deque(maxlen=1000)
    total_detections = 0
    total_frames = 0
    frame_timestamps: deque = deque(maxlen=300)  # for fps calc

    producer_done = asyncio.Event()
    consumer_done = asyncio.Event()
    shutdown = asyncio.Event()

    # ── 5. Producer task ─────────────────────────────────────────────────────
    async def producer_task():
        try:
            async for jpeg in capture_frames(source, source_url):
                if shutdown.is_set():
                    break
                await queue.put_frame(jpeg)
        except FrameCaptureError as exc:
            log.error("frame_capture_error", error=str(exc))
        except asyncio.CancelledError:
            pass
        finally:
            producer_done.set()

    # ── 6. Consumer task ─────────────────────────────────────────────────────
    async def consumer_task():
        nonlocal total_detections, total_frames
        while not shutdown.is_set() or queue.queue_depth > 0:
            try:
                frame = await asyncio.wait_for(queue.get_frame(), timeout=0.5)
            except asyncio.TimeoutError:
                if shutdown.is_set() and queue.queue_depth == 0:
                    break
                continue
            except asyncio.CancelledError:
                break

            frame_id = str(uuid.uuid4())
            t0 = time.monotonic()
            try:
                result = await inference_client.infer(frame, frame_id)
            except InferenceClientError as exc:
                log.warning("inference_error", error=str(exc))
                continue
            except asyncio.CancelledError:
                break

            elapsed_ms = (time.monotonic() - t0) * 1000
            latency_buf.append(elapsed_ms)
            total_frames += 1
            frame_timestamps.append(time.monotonic())
            total_detections += len(result.detections)

            # Pack and send binary msgpack frame result
            payload = {
                "type": "FRAME_RESULT",
                "frame_id": result.frame_id,
                "timestamp_ms": result.timestamp_ms,
                "detections": [
                    {
                        "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                        "confidence": d.confidence,
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                    }
                    for d in result.detections
                ],
                "inference_time_ms": result.inference_time_ms,
            }
            try:
                await websocket.send_bytes(msgpack.packb(payload, use_bin_type=True))
            except (WebSocketDisconnect, RuntimeError):
                shutdown.set()
                break

        consumer_done.set()

    # ── 7. Metrics task ───────────────────────────────────────────────────────
    async def metrics_task():
        import numpy as np

        while not shutdown.is_set():
            await asyncio.sleep(1.0)
            if shutdown.is_set():
                break

            # FPS: frames in last second
            now = time.monotonic()
            recent = [t for t in frame_timestamps if t >= now - 1.0]
            fps = float(len(recent))

            lats = list(latency_buf)
            if lats:
                arr = np.array(lats)
                p50 = float(np.percentile(arr, 50))
                p95 = float(np.percentile(arr, 95))
                p99 = float(np.percentile(arr, 99))
            else:
                p50 = p95 = p99 = 0.0

            depth = queue.queue_depth
            drop_rate = queue.drop_rate

            metrics_msg = json.dumps({
                "type": "METRICS_UPDATE",
                "fps": fps,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "queue_depth": depth,
                "drop_rate": drop_rate,
                "total_detections": total_detections,
                "active_connections": _active_connections,
            })
            try:
                await websocket.send_text(metrics_msg)
            except (WebSocketDisconnect, RuntimeError):
                shutdown.set()
                break

            if depth > 10:
                queue_msg = json.dumps({
                    "type": "QUEUE_STATUS",
                    "depth": depth,
                    "warning": True,
                })
                try:
                    await websocket.send_text(queue_msg)
                except (WebSocketDisconnect, RuntimeError):
                    shutdown.set()
                    break

            # Persist snapshot to telemetry service
            class_counts: dict = {}
            try:
                await telemetry_service.record_snapshot(
                    session_id=session_id,
                    fps=fps,
                    latencies=lats,
                    queue_depth=depth,
                    class_counts=class_counts,
                )
            except Exception as exc:
                log.warning("telemetry_record_failed", error=str(exc))

    # ── 8. Run all three tasks concurrently ───────────────────────────────────
    prod = asyncio.create_task(producer_task(), name=f"producer-{session_id}")
    cons = asyncio.create_task(consumer_task(), name=f"consumer-{session_id}")
    metr = asyncio.create_task(metrics_task(), name=f"metrics-{session_id}")

    try:
        # Wait for disconnect (receive loop)
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Keepalive: no-op, loop continues
                continue
            except (WebSocketDisconnect, RuntimeError):
                break
    except asyncio.CancelledError:
        pass
    finally:
        shutdown.set()
        for task in (prod, cons, metr):
            task.cancel()
        await asyncio.gather(prod, cons, metr, return_exceptions=True)
        await inference_client.stop()
        await queue.drain(timeout=2.0)

        # Flush final stats to DB
        if session_store is not None:
            try:
                await session_store.update_session(
                    session_id=session_id,
                    total_frames=total_frames,
                    dropped_frames=queue._drop_counter,
                    total_detections=total_detections,
                    ended_at=datetime.now(tz=timezone.utc),
                )
            except Exception as exc:
                log.warning("session_update_failed", error=str(exc))

        _active_connections -= 1
        telemetry_service.decrement_active_sessions()
        log.info("ws_session_ended", total_frames=total_frames, total_detections=total_detections)
