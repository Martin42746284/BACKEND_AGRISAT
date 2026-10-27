"""Route /api/gemini-parcels : capture satellite + détection de parcelles par Gemini.

La route ne fait que traduire la requête et les erreurs du service ; toute la
logique (capture Static Maps, appel Gemini, géoréférencement, déduplication,
écriture dans `parcelles`) vit dans `app/services/gemini_parcels.py`.
"""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import GeminiParcelsRequest
from app.services.gemini_parcels import (
    GeminiConfigurationError,
    GeminiParcelsError,
    GeminiUnavailable,
    detect_parcels_gemini,
)

logger = logging.getLogger("agrisat.routes.gemini_parcels")

router = APIRouter()


@router.post("/api/gemini-parcels")
async def post_gemini_parcels(payload: GeminiParcelsRequest) -> JSONResponse:
    """Détecte les parcelles d'une capture satellite et les enregistre."""
    span_m = payload.span_m if payload.span_m is not None else settings.gemini_capture_span_m
    tile_span_m = (
        payload.tile_span_m
        if payload.tile_span_m is not None
        else settings.gemini_capture_tile_span_m
    )
    if not all(
        math.isfinite(value)
        for value in (payload.latitude, payload.longitude, span_m, tile_span_m)
    ):
        return JSONResponse(status_code=400, content={"error": "Coordonnées invalides."})

    try:
        result = await detect_parcels_gemini(
            latitude=payload.latitude,
            longitude=payload.longitude,
            span_m=span_m,
            tile_span_m=tile_span_m,
            zoom=payload.zoom,
            persist=payload.persist,
            viewport_width=payload.viewport_width,
            viewport_height=payload.viewport_height,
            bounds=payload.bounds.model_dump() if payload.bounds else None,
        )
    except GeminiConfigurationError as exc:
        logger.warning("gemini-parcels configuration : %s", exc)
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except GeminiUnavailable as exc:
        logger.warning("gemini-parcels indisponible : %s", exc)
        return JSONResponse(status_code=502, content={"error": str(exc)})
    except GeminiParcelsError as exc:
        logger.info("gemini-parcels refusé : %s", exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001 — l'interface affiche le message
        logger.exception("gemini-parcels en erreur")
        return JSONResponse(
            status_code=500,
            content={"error": f"Détection Gemini impossible : {exc}"},
        )

    return JSONResponse(content=result)
