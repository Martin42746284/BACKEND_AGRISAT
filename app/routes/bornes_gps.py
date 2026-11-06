"""Bornes GPS : points de repère nommés, enregistrés par les utilisateurs.

- GET    /api/bornes-gps       : liste (filtre optionnel `q` sur le nom)
- POST   /api/bornes-gps       : création d'une borne
- PATCH  /api/bornes-gps/{id}  : renommage
- DELETE /api/bornes-gps/{id}  : suppression
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import delete as sa_delete, select

from app.db import async_session_maker
from app.errors import parse_or_400
from app.models import BorneGps
from app.schemas import BorneGpsCreate, BorneGpsUpdate

logger = logging.getLogger("agrisat.routes.bornes_gps")

router = APIRouter()


def _serialize(row: BorneGps) -> dict[str, Any]:
    def iso(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    return {
        "id": str(row.id),
        "name": row.name,
        "lat": row.lat,
        "lng": row.lng,
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


@router.get("/api/bornes-gps")
async def list_bornes_gps(q: str | None = None) -> JSONResponse:
    try:
        async with async_session_maker() as session:
            statement = select(BorneGps).order_by(BorneGps.name.asc(), BorneGps.created_at.asc())
            needle = (q or "").strip()
            if needle:
                statement = statement.where(BorneGps.name.ilike(f"%{needle}%"))
            rows = (await session.execute(statement)).scalars().all()
        return JSONResponse(content=[_serialize(row) for row in rows])
    except Exception as error:  # noqa: BLE001
        logger.error("bornes GPS: lecture impossible : %s", error)
        return JSONResponse(
            status_code=503,
            content={"error": "Les bornes GPS sont temporairement indisponibles."},
        )


@router.post("/api/bornes-gps")
async def create_borne_gps(request: Request) -> JSONResponse:
    parsed = parse_or_400(BorneGpsCreate, await request.json(), "Données de borne GPS invalides")

    async with async_session_maker() as session:
        row = BorneGps(name=parsed.name, lat=parsed.lat, lng=parsed.lng)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        result = _serialize(row)

    return JSONResponse(status_code=201, content=result)


@router.patch("/api/bornes-gps/{borne_id}")
async def update_borne_gps(borne_id: str, request: Request) -> JSONResponse:
    try:
        borne_uuid = uuid.UUID(borne_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Identifiant de borne invalide"})

    parsed = parse_or_400(BorneGpsUpdate, await request.json(), "Données de borne GPS invalides")

    async with async_session_maker() as session:
        existing = await session.get(BorneGps, borne_uuid)
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "Borne GPS introuvable"})

        existing.name = parsed.name
        await session.commit()
        await session.refresh(existing)
        result = _serialize(existing)

    return JSONResponse(content=result)


@router.delete("/api/bornes-gps/{borne_id}")
async def delete_borne_gps(borne_id: str) -> Response:
    try:
        borne_uuid = uuid.UUID(borne_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Identifiant de borne invalide"})

    async with async_session_maker() as session:
        existing = await session.get(BorneGps, borne_uuid)
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "Borne GPS introuvable"})

        await session.execute(sa_delete(BorneGps).where(BorneGps.id == borne_uuid))
        await session.commit()

    return Response(status_code=204)
