import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.config import settings
from app.main import create_app
from app.models import DetectionBox, DetectionResult
from app.state import app_state


# ── pytest-asyncio mode ───────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


# ── Override settings for tests ───────────────────────────────────────────────
settings.DATABASE_URL = "postgresql://test:test@localhost:5432/cv_test"
settings.REDIS_URL = "redis://localhost:6379/1"
settings.JWT_SECRET = "test-secret-key-for-unit-tests-only"
settings.MAX_CONCURRENT_SESSIONS = 5
settings.QUEUE_MAXSIZE = 30
settings.DEMO_VIDEO_PATH = "/tmp/test-demo.mp4"


# ── Fake ML service response ──────────────────────────────────────────────────
FAKE_DETECTION = {
    "x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0,
    "confidence": 0.92,
    "class_id": 0,
    "class_name": "person",
}

FAKE_INFER_RESPONSE = {
    "detections": [FAKE_DETECTION],
    "inference_time_ms": 12.5,
    "timestamp_ms": 1700000000000,
}


@pytest.fixture
def fake_detection_result() -> DetectionResult:
    return DetectionResult(
        frame_id=str(uuid.uuid4()),
        timestamp_ms=1700000000000,
        detections=[DetectionBox(**FAKE_DETECTION)],
        inference_time_ms=12.5,
    )


# ── Mock asyncpg pool ─────────────────────────────────────────────────────────
class FakeAsyncpgConn:
    async def execute(self, query, *args):
        return "OK"

    async def fetchrow(self, query, *args):
        sid = uuid.uuid4()
        return {"id": sid, "user_id": None, "source_type": "demo",
                "source_url": None, "started_at": datetime.now(tz=timezone.utc),
                "ended_at": None, "total_frames": 0, "dropped_frames": 0,
                "total_detections": 0, "is_deleted": False}

    async def fetch(self, query, *args):
        return []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


class FakePool:
    def acquire(self):
        return FakeAsyncpgConn()

    async def fetchrow(self, query, *args):
        return await FakeAsyncpgConn().fetchrow(query, *args)

    async def fetch(self, query, *args):
        return []

    async def execute(self, query, *args):
        return "OK"

    async def close(self):
        pass


@pytest.fixture
def fake_pool() -> FakePool:
    return FakePool()


# ── Mock Redis ────────────────────────────────────────────────────────────────
class FakeRedis:
    async def ping(self):
        return True

    async def publish(self, channel: str, message: str):
        return 0

    async def aclose(self):
        pass


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


# ── Mock InferenceClient ──────────────────────────────────────────────────────
@pytest.fixture
def mock_inference_client(fake_detection_result):
    client = AsyncMock()
    client.infer = AsyncMock(return_value=fake_detection_result)
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.p50 = MagicMock(return_value=12.5)
    client.p95 = MagicMock(return_value=18.0)
    client.p99 = MagicMock(return_value=22.0)
    return client


# ── Mock SessionStore ─────────────────────────────────────────────────────────
@pytest.fixture
def mock_session_store():
    store = AsyncMock()
    store.create_session = AsyncMock(return_value=str(uuid.uuid4()))
    store.update_session = AsyncMock(return_value=None)
    store.get_sessions = AsyncMock(return_value=[])
    store.count_sessions = AsyncMock(return_value=0)
    store.get_session = AsyncMock(return_value=None)
    store.soft_delete_session = AsyncMock(return_value=True)
    store.save_telemetry_snapshot = AsyncMock(return_value=None)
    store.get_telemetry_snapshots = AsyncMock(return_value=[])
    return store


# ── App fixture with mocked dependencies ─────────────────────────────────────
@pytest.fixture
def app_with_mocks(fake_pool, fake_redis, mock_session_store):
    from app.services.session_store import SessionStore

    app_state["db_pool"] = fake_pool
    app_state["redis"] = fake_redis
    app_state["session_store"] = mock_session_store

    application = create_app()
    yield application

    app_state.clear()


@pytest.fixture
def sync_client(app_with_mocks):
    with TestClient(app_with_mocks, raise_server_exceptions=False) as client:
        yield client


@pytest_asyncio.fixture
async def async_client(app_with_mocks) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks), base_url="http://test"
    ) as client:
        yield client


# ── JWT token helper ──────────────────────────────────────────────────────────
def make_jwt(user_id: str = "user-123", expire_minutes: int = 60) -> str:
    from datetime import timedelta
    from jose import jwt as jose_jwt

    now = datetime.now(tz=timezone.utc)
    claims = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=expire_minutes),
    }
    return jose_jwt.encode(claims, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


@pytest.fixture
def valid_token() -> str:
    return make_jwt()


@pytest.fixture
def expired_token() -> str:
    return make_jwt(expire_minutes=-1)
