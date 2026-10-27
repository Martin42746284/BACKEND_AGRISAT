import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from sqlalchemy import delete as sa_delete, update

from app import cache
from app.db import async_session_maker
from app.errors import parse_or_400
from app.models import Parcelle
from app.schemas import DeleteParcelleParams, ParcelleCreate, ParcelleUpdate

logger = logging.getLogger("agrisat.routes.parcelles")

router = APIRouter()

# Champs exposés par l'API, identiques à ceux connus de Prisma (schema.prisma) — exclut
# volontairement `updated_at`, une colonne présente en base mais jamais déclarée côté Prisma
# et donc jamais renvoyée par l'ancien backend.
_PARCELLE_FIELDS = (
    "id", "label", "coordinates", "center_lat", "center_lng", "surface_ha",
    "culture_declared", "culture_detected", "ndvi_percentage", "ndre", "spectral_bands",
    "confidence", "verdict", "details", "saison", "soil_type", "risk_factors",
    "recommendations", "data_source", "owner_name", "notes", "time_series_s1",
    "time_series_s2", "time_series_rain", "estimated_planting_date", "estimated_harvest_date",
    "days_since_planting", "growth_stage", "planting_confidence", "evi", "savi", "ndwi",
    "agro_score", "hybrid_score", "cnn_prob_barley", "cnn_prob_non_barley", "created_at",
)


def _serialize(row: Parcelle) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in _PARCELLE_FIELDS:
        value = getattr(row, field)
        if isinstance(value, datetime):
            value = value.isoformat()
        else:
            value = str(value) if field == "id" else value
        out[field] = value
    return out


@router.get("/api/parcelles")
async def list_parcelles() -> JSONResponse:
    try:
        rows = await cache.load_parcelles()
        return JSONResponse(content=[_serialize(r) for r in rows])
    except Exception as error:  # noqa: BLE001
        logger.error(
            "parcelles: read failed (%s): %s",
            type(error).__name__,
            cache.describe_error(error),
        )
        return JSONResponse(status_code=503, content={"error": "Les parcelles sont temporairement indisponibles."})


@router.post("/api/parcelles")
async def create_parcelle(request: Request) -> JSONResponse:
    body = await request.json()
    parsed = parse_or_400(ParcelleCreate, body, "Données de parcelle invalides")

    values = {
        "label": parsed.label,
        "coordinates": [p.model_dump() for p in parsed.coordinates],
        "center_lat": parsed.center_lat,
        "center_lng": parsed.center_lng,
        "surface_ha": parsed.surface_ha,
        "culture_declared": parsed.culture_declared,
        "culture_detected": parsed.culture_detected,
        "ndvi_percentage": parsed.ndvi_percentage,
        "ndre": parsed.ndre,
        "spectral_bands": parsed.spectral_bands,
        "confidence": parsed.confidence,
        "verdict": parsed.verdict,
        "details": parsed.details,
        "saison": parsed.saison,
        "soil_type": parsed.soil_type,
        "risk_factors": parsed.risk_factors,
        "recommendations": parsed.recommendations,
        "data_source": parsed.data_source,
        "owner_name": parsed.owner_name,
        "notes": parsed.notes,
        "time_series_s1": parsed.time_series_s1,
        "time_series_s2": parsed.time_series_s2,
        "time_series_rain": parsed.time_series_rain,
        "estimated_planting_date": parsed.estimated_planting_date,
        "estimated_harvest_date": parsed.estimated_harvest_date,
        "days_since_planting": int(parsed.days_since_planting) if parsed.days_since_planting is not None else None,
        "growth_stage": parsed.growth_stage,
        "planting_confidence": parsed.planting_confidence,
        "evi": parsed.evi,
        "savi": parsed.savi,
        "ndwi": parsed.ndwi,
        "agro_score": parsed.agro_score,
        "hybrid_score": parsed.hybrid_score,
        "cnn_prob_barley": parsed.cnn_prob_barley,
        "cnn_prob_non_barley": parsed.cnn_prob_non_barley,
    }

    async with async_session_maker() as session:
        row = Parcelle(**values)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        result = _serialize(row)

    cache.invalidate()
    return JSONResponse(status_code=201, content=result)


# Données de culture attachées à une parcelle : effacées quand l'utilisateur supprime la
# culture, alors que le contour, l'identité et la localisation de la parcelle restent.
_CULTURE_FIELDS = (
    "culture_declared", "culture_detected", "ndvi_percentage", "ndre", "spectral_bands",
    "confidence", "verdict", "details", "saison", "soil_type", "risk_factors",
    "recommendations", "time_series_s1", "time_series_s2", "time_series_rain",
    "estimated_planting_date", "estimated_harvest_date", "days_since_planting",
    "growth_stage", "planting_confidence", "evi", "savi", "ndwi", "agro_score",
    "hybrid_score", "cnn_prob_barley", "cnn_prob_non_barley",
)
# Colonnes JSONB de culture : une liste vide est plus sûre qu'un NULL pour les consommateurs.
_CULTURE_LIST_FIELDS = {"risk_factors", "time_series_s1", "time_series_s2", "time_series_rain"}


@router.delete("/api/parcelles/{parcelle_id}/culture")
async def delete_parcelle_culture(parcelle_id: str) -> Response:
    """Supprime la culture d'une parcelle **sans** supprimer la parcelle.

    Le contour, la superficie et l'identité de la parcelle restent en base (donc sur
    la carte) ; seules les données de culture sont effacées : classification, indice
    de confiance, indices spectraux, cycle cultural, sol, risques, recommandations et
    séries temporelles. L'opération est idempotente.
    """
    try:
        parsed = DeleteParcelleParams.model_validate({"id": parcelle_id})
    except ValidationError:
        return JSONResponse(status_code=400, content={"error": "Identifiant invalide"})

    parcelle_uuid = uuid.UUID(parsed.id)
    async with async_session_maker() as session:
        existing = await session.get(Parcelle, parcelle_uuid)
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "Parcelle introuvable"})

        await session.execute(
            update(Parcelle)
            .where(Parcelle.id == parcelle_uuid)
            .values(**{
                field: [] if field in _CULTURE_LIST_FIELDS else None
                for field in _CULTURE_FIELDS
            })
        )
        await session.commit()

    cache.invalidate()
    return Response(status_code=204)


@router.delete("/api/parcelles/{parcelle_id}")
async def delete_parcelle(parcelle_id: str) -> Response:
    try:
        parsed = DeleteParcelleParams.model_validate({"id": parcelle_id})
    except ValidationError:
        return JSONResponse(status_code=400, content={"error": "Identifiant invalide"})

    async with async_session_maker() as session:
        existing = await session.get(Parcelle, uuid.UUID(parsed.id))
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "Parcelle introuvable"})

        if existing.data_source == "Tracé automatique corrigé":
            return JSONResponse(status_code=409, content={"error": "Cette parcelle corrigée ne peut pas être supprimée."})

        result = await session.execute(sa_delete(Parcelle).where(Parcelle.id == uuid.UUID(parsed.id)))
        await session.commit()
        deleted_count = result.rowcount

    if deleted_count == 0:
        return JSONResponse(status_code=404, content={"error": "Parcelle introuvable"})

    cache.invalidate()
    return Response(status_code=204)


@router.patch("/api/parcelles/{parcelle_id}")
async def update_parcelle(parcelle_id: str, request: Request) -> JSONResponse:
    try:
        parsed = DeleteParcelleParams.model_validate({"id": parcelle_id})
    except ValidationError:
        return JSONResponse(status_code=400, content={"error": "Identifiant invalide"})

    parsed_update = parse_or_400(ParcelleUpdate, await request.json(), "Données de mise à jour invalides")

    async with async_session_maker() as session:
        existing = await session.get(Parcelle, uuid.UUID(parsed.id))
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "Parcelle introuvable"})

        update_data = parsed_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(existing, field, value)

        await session.commit()
        await session.refresh(existing)
        result = _serialize(existing)

    cache.invalidate()
    return JSONResponse(content=result)
