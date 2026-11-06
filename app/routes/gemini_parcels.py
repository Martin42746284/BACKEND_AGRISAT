"""Routes /api/gemini-parcels : capture satellite + détection de parcelles.

La route JSON reste le contrat historique ; sa variante `/stream` relaie les
mêmes paliers de progression en NDJSON pour l'interface de chargement. Les deux
routes ne font que traduire la requête et les erreurs du service ; toute la
logique (capture Static Maps, appel Gemini, géoréférencement, déduplication,
écriture dans `parcelles`) vit dans `app/services/gemini_parcels.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse, StreamingResponse

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


@router.post("/api/gemini-parcels/stream")
async def post_gemini_parcels_stream(
    payload: GeminiParcelsRequest,
) -> Response:
    """Variante NDJSON de la détection, pour afficher la progression réelle.

    Un événement `progress` est émis à chaque palier effectivement atteint
    (tuiles téléchargées, fenêtres Gemini terminées, écriture en base), puis un
    événement `result` ou `error`. Le calcul reste strictement le même que sur
    `/api/gemini-parcels`.
    """
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

    async def events() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def report(percent: int, label: str) -> None:
            await queue.put({"type": "progress", "percent": percent, "label": label})

        async def run() -> None:
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
                    progress=report,
                )
                await queue.put({"type": "result", "result": result})
            except GeminiConfigurationError as exc:
                logger.warning("gemini-parcels stream configuration : %s", exc)
                await queue.put({"type": "error", "error": str(exc)})
            except GeminiUnavailable as exc:
                logger.warning("gemini-parcels stream indisponible : %s", exc)
                await queue.put({"type": "error", "error": str(exc)})
            except GeminiParcelsError as exc:
                logger.info("gemini-parcels stream refusé : %s", exc)
                await queue.put({"type": "error", "error": str(exc)})
            except Exception as exc:  # noqa: BLE001 — l'interface affiche le message
                logger.exception("gemini-parcels stream en erreur")
                await queue.put(
                    {"type": "error", "error": f"Détection Gemini impossible : {exc}"}
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
