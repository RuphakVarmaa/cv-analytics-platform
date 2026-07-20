from datetime import datetime, timezone

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.state import app_state

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


@router.get("/ready")
async def readiness_check(response: Response):
    """
    Checks live connectivity to both PostgreSQL and Redis.
    Returns 200 when ready, 503 when any dependency is unavailable.
    """
    checks = {}
    healthy = True

    # PostgreSQL check
    try:
        pool = app_state.get("db_pool")
        if pool is None:
            raise RuntimeError("pool not initialized")
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        healthy = False

    # Redis check
    try:
        redis = app_state.get("redis")
        if redis is None:
            raise RuntimeError("redis not initialized")
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"
        healthy = False

    if not healthy:
        response.status_code = 503

    return {
        "status": "ready" if healthy else "unavailable",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "checks": checks,
    }


@router.get("/metrics")
async def prometheus_metrics(response: Response):
    """Expose Prometheus metrics in text format."""
    data = generate_latest()
    response.headers["Content-Type"] = CONTENT_TYPE_LATEST
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
