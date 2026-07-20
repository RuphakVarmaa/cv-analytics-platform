import csv
import io
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.middleware.auth import validate_token
from app.models import PaginatedSessions, Session
from app.state import app_state

router = APIRouter(tags=["sessions"])
_bearer = HTTPBearer(auto_error=False)


def _get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    if creds is None:
        return None
    payload = validate_token(creds.credentials)
    return payload.get("sub")


def _require_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    if creds is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = validate_token(creds.credentials)
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token subject")
    return uid


@router.get("/api/sessions", response_model=PaginatedSessions)
async def list_sessions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user_id: str = Depends(_require_user),
):
    store = app_state.get("session_store")
    if store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    sessions = await store.get_sessions(user_id=user_id, page=page, per_page=per_page)
    total = await store.count_sessions(user_id=user_id)
    pages = max(1, (total + per_page - 1) // per_page)

    return PaginatedSessions(
        items=sessions,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
    )


@router.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str,
    user_id: str = Depends(_require_user),
):
    store = app_state.get("session_store")
    if store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    session = await store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id and session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    snapshots = await store.get_telemetry_snapshots(session_id)
    return {
        "session": session.model_dump(),
        "telemetry": snapshots,
    }


@router.get("/api/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = Query("json", pattern="^(csv|json)$"),
    user_id: str = Depends(_require_user),
):
    store = app_state.get("session_store")
    if store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    session = await store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id and session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    snapshots = await store.get_telemetry_snapshots(session_id)

    if format == "json":
        content = json.dumps(
            {
                "session": session.model_dump(mode="json"),
                "telemetry": [
                    {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in s.items()}
                    for s in snapshots
                ],
            },
            indent=2,
        )
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="session-{session_id}.json"'
            },
        )

    # CSV export
    output = io.StringIO()
    if snapshots:
        fieldnames = list(snapshots[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in snapshots:
            writer.writerow(
                {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
            )
    csv_content = output.getvalue()

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="session-{session_id}.csv"'
        },
    )


@router.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    user_id: str = Depends(_require_user),
):
    store = app_state.get("session_store")
    if store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    session = await store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id and session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    await store.soft_delete_session(session_id)
    return Response(status_code=204)
