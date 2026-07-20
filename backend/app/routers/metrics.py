import asyncio
import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request, Response
from sse_starlette.sse import EventSourceResponse

from app.services.telemetry import telemetry_service
from app.state import app_state

router = APIRouter(tags=["metrics"])

# Record process start time for uptime
_START_TIME = time.time()


@router.get("/api/metrics/stream")
async def metrics_stream(request: Request):
    """
    Server-Sent Events endpoint that streams aggregate metrics every second.
    """

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            metrics = await telemetry_service.get_aggregate_metrics()
            metrics["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

            yield {
                "event": "metrics",
                "data": json.dumps(metrics),
            }
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


@router.get("/api/metrics/totals")
async def metrics_totals():
    """
    Lifetime aggregate stats: total frames processed, uptime, start time.
    """
    uptime_seconds = time.time() - _START_TIME
    aggregate = await telemetry_service.get_aggregate_metrics()

    return {
        "uptime_seconds": uptime_seconds,
        "started_at": datetime.fromtimestamp(_START_TIME, tz=timezone.utc).isoformat(),
        "active_sessions": aggregate.get("active_sessions", 0),
        "total_fps": aggregate.get("total_fps", 0.0),
        "avg_p50_ms": aggregate.get("avg_p50_ms", 0.0),
        "avg_p95_ms": aggregate.get("avg_p95_ms", 0.0),
    }
