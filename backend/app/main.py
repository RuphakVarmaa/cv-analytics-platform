import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.middleware.rate_limit import TokenBucketRateLimiter
from app.routers import health, stream, metrics, sessions
from app.services.session_store import SessionStore
from app.services.telemetry import telemetry_service
from app.state import app_state

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), settings.LOG_LEVEL, 20)
    ),
)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("startup_begin", ml_service=settings.ML_SERVICE_URL)

    # PostgreSQL connection pool
    try:
        pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        await SessionStore.create_tables(pool)
        app_state["db_pool"] = pool
        app_state["session_store"] = SessionStore(pool)
        logger.info("db_connected")
    except Exception as exc:
        logger.error("db_connect_failed", error=str(exc))
        app_state["db_pool"] = None
        app_state["session_store"] = None

    # Redis connection
    try:
        redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,
            socket_connect_timeout=5,
        )
        await redis.ping()
        app_state["redis"] = redis
        telemetry_service.set_redis(redis)
        logger.info("redis_connected")
    except Exception as exc:
        logger.error("redis_connect_failed", error=str(exc))
        app_state["redis"] = None

    logger.info("startup_complete")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("shutdown_begin")
    pool = app_state.get("db_pool")
    if pool:
        await pool.close()
    redis = app_state.get("redis")
    if redis:
        await redis.aclose()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CV Analytics Platform",
        description="Real-time computer vision analytics with WebSocket streaming",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = (
        ["*"]
        if settings.ALLOWED_ORIGINS == "*"
        else [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiter ──────────────────────────────────────────────────────────
    app.add_middleware(TokenBucketRateLimiter, rate=30, window=1.0)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(stream.router)
    app.include_router(metrics.router)
    app.include_router(sessions.router)

    # ── Static files (demo video) ─────────────────────────────────────────────
    import os
    demo_dir = os.path.dirname(settings.DEMO_VIDEO_PATH)
    if os.path.isdir(demo_dir):
        app.mount("/static", StaticFiles(directory=demo_dir), name="static")

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            path=str(request.url.path),
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
