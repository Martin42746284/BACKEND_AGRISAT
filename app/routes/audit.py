"""Routes d'audit : journal des actions utilisateurs sur les parcelles.

- GET /api/audit : liste les entrées d'audit (récentes par défaut).
- POST /api/audit : crée une entrée d'audit (correction, rejet, observation, connexion, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import desc, select

from app import cache
from app.db import async_session_maker
from app.errors import parse_or_400
from app.models import AuditEntry
from app.schemas import AuditEntryCreate

logger = logging.getLogger(__name__)

router = APIRouter()


def _serialize_audit(row: AuditEntry) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "at": row.created_at.isoformat(),
        "user_name": row.user_name,
        "action": row.action,
        "object": row.object,
    }


@router.get("/api/audit")
async def list_audit() -> JSONResponse:
    """Liste les dernières entrées d'audit (50 maximum par défaut)."""
    try:
        limit = 50
        async with async_session_maker() as session:
            result = await session.execute(
                select(AuditEntry).order_by(desc(AuditEntry.created_at)).limit(limit)
            )
            rows = result.scalars().all()
        return JSONResponse(content=[_serialize_audit(r) for r in rows])
    except Exception as error:  # noqa: BLE001
        logger.error("audit: read failed: %s", error)
        return JSONResponse(status_code=503, content={"error": "L'audit est temporairement indisponible."})


@router.post("/api/audit")
async def create_audit(request: Request) -> JSONResponse:
    """Crée une entrée d'audit (correction, rejet, observation, connexion, etc.)."""
    try:
        parsed = parse_or_400(AuditEntryCreate, await request.json(), "Données d'audit invalides")
    except ValidationError:
        return JSONResponse(status_code=400, content={"error": "Données d'audit invalides"})

    async with async_session_maker() as session:
        row = AuditEntry(
            user_name=parsed.user_name,
            action=parsed.action,
            object=parsed.object,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        result = _serialize_audit(row)

    return JSONResponse(status_code=201, content=result)
