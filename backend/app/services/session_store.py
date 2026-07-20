import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any

import asyncpg
import structlog

from app.models import Session, TelemetrySnapshot

logger = structlog.get_logger(__name__)

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username    TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS inference_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    source_type     TEXT NOT NULL,
    source_url      TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    total_frames    INT NOT NULL DEFAULT 0,
    dropped_frames  INT NOT NULL DEFAULT 0,
    total_detections INT NOT NULL DEFAULT 0,
    is_deleted      BOOL NOT NULL DEFAULT FALSE
);
"""

CREATE_TELEMETRY_TABLE = """
CREATE TABLE IF NOT EXISTS telemetry_snapshots (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES inference_sessions(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fps         FLOAT NOT NULL DEFAULT 0,
    p50_ms      FLOAT NOT NULL DEFAULT 0,
    p95_ms      FLOAT NOT NULL DEFAULT 0,
    queue_depth INT NOT NULL DEFAULT 0,
    class_counts JSONB NOT NULL DEFAULT '{}'::jsonb
);
"""

CREATE_TELEMETRY_IDX = """
CREATE INDEX IF NOT EXISTS idx_telemetry_session_recorded
    ON telemetry_snapshots (session_id, recorded_at DESC);
"""

CREATE_SESSIONS_IDX = """
CREATE INDEX IF NOT EXISTS idx_sessions_user_started
    ON inference_sessions (user_id, started_at DESC);
"""


class SessionStore:
    """asyncpg-backed persistent session and telemetry store."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def create_tables(cls, pool: asyncpg.Pool) -> None:
        """Idempotent DDL – safe to call on every startup."""
        async with pool.acquire() as conn:
            await conn.execute(CREATE_USERS_TABLE)
            await conn.execute(CREATE_SESSIONS_TABLE)
            await conn.execute(CREATE_TELEMETRY_TABLE)
            await conn.execute(CREATE_TELEMETRY_IDX)
            await conn.execute(CREATE_SESSIONS_IDX)
        logger.info("db_tables_ready")

    async def create_session(
        self,
        user_id: Optional[str],
        source_type: str,
        source_url: Optional[str] = None,
    ) -> str:
        """Insert a new inference session and return its UUID string."""
        row = await self._pool.fetchrow(
            """
            INSERT INTO inference_sessions (user_id, source_type, source_url)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            uuid.UUID(user_id) if user_id else None,
            source_type,
            source_url,
        )
        return str(row["id"])

    async def update_session(
        self,
        session_id: str,
        total_frames: int,
        dropped_frames: int,
        total_detections: int,
        ended_at: Optional[datetime] = None,
    ) -> None:
        """Update counters and mark ended_at for a finished session."""
        if ended_at is None:
            ended_at = datetime.now(tz=timezone.utc)
        await self._pool.execute(
            """
            UPDATE inference_sessions
            SET total_frames     = $2,
                dropped_frames   = $3,
                total_detections = $4,
                ended_at         = $5
            WHERE id = $1
            """,
            uuid.UUID(session_id),
            total_frames,
            dropped_frames,
            total_detections,
            ended_at,
        )

    async def get_sessions(
        self,
        user_id: Optional[str],
        page: int = 1,
        per_page: int = 20,
    ) -> List[Session]:
        """Paginated session list; filters by user_id if provided."""
        offset = (page - 1) * per_page
        if user_id:
            rows = await self._pool.fetch(
                """
                SELECT * FROM inference_sessions
                WHERE user_id = $1 AND is_deleted = FALSE
                ORDER BY started_at DESC
                LIMIT $2 OFFSET $3
                """,
                uuid.UUID(user_id),
                per_page,
                offset,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM inference_sessions
                WHERE is_deleted = FALSE
                ORDER BY started_at DESC
                LIMIT $1 OFFSET $2
                """,
                per_page,
                offset,
            )
        return [_row_to_session(r) for r in rows]

    async def count_sessions(self, user_id: Optional[str]) -> int:
        if user_id:
            row = await self._pool.fetchrow(
                "SELECT COUNT(*) AS cnt FROM inference_sessions WHERE user_id=$1 AND is_deleted=FALSE",
                uuid.UUID(user_id),
            )
        else:
            row = await self._pool.fetchrow(
                "SELECT COUNT(*) AS cnt FROM inference_sessions WHERE is_deleted=FALSE"
            )
        return row["cnt"]

    async def get_session(self, session_id: str) -> Optional[Session]:
        row = await self._pool.fetchrow(
            "SELECT * FROM inference_sessions WHERE id=$1 AND is_deleted=FALSE",
            uuid.UUID(session_id),
        )
        if row is None:
            return None
        return _row_to_session(row)

    async def soft_delete_session(self, session_id: str) -> bool:
        result = await self._pool.execute(
            "UPDATE inference_sessions SET is_deleted=TRUE WHERE id=$1",
            uuid.UUID(session_id),
        )
        return result == "UPDATE 1"

    async def save_telemetry_snapshot(
        self,
        session_id: str,
        fps: float,
        p50: float,
        p95: float,
        queue_depth: int,
        class_counts: dict,
    ) -> None:
        import json as _json
        await self._pool.execute(
            """
            INSERT INTO telemetry_snapshots
                (session_id, fps, p50_ms, p95_ms, queue_depth, class_counts)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            uuid.UUID(session_id),
            fps,
            p50,
            p95,
            queue_depth,
            _json.dumps(class_counts),
        )

    async def get_telemetry_snapshots(
        self,
        session_id: str,
        limit: int = 300,
    ) -> List[dict]:
        rows = await self._pool.fetch(
            """
            SELECT recorded_at, fps, p50_ms, p95_ms, queue_depth, class_counts
            FROM telemetry_snapshots
            WHERE session_id = $1
            ORDER BY recorded_at ASC
            LIMIT $2
            """,
            uuid.UUID(session_id),
            limit,
        )
        return [dict(r) for r in rows]


def _row_to_session(row: asyncpg.Record) -> Session:
    return Session(
        id=str(row["id"]),
        user_id=str(row["user_id"]) if row["user_id"] else None,
        source_type=row["source_type"],
        source_url=row["source_url"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        total_frames=row["total_frames"],
        dropped_frames=row["dropped_frames"],
        total_detections=row["total_detections"],
        is_deleted=row["is_deleted"],
    )
