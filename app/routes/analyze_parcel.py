from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app import cache
from app.errors import parse_or_400
from app.schemas import AnalyzeParcelRequest
from app.services.analyze_parcel import (
    analyze_parcel,
    approximate_polygon_area_m2,
    js_round,
    normalize_polygon,
    polygon_centroid,
)
from app.services.parcel_registry import (
    STORE_ALREADY_ANALYZED,
    STORE_CREATED,
    STORE_NOT_FOUND,
    find_parcel_for_analysis,
    store_parcel_analysis,
    values_have_analysis,
)

router = APIRouter()

ALREADY_ANALYZED_MESSAGE = (
    "Cette parcelle a déjà été analysée : ses données enregistrées restent inchangées. "
    "Relancez l'analyse pour les remplacer."
)


@router.post("/api/analyze-parcel")
async def analyze_parcel_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(error) or "Unknown error"})
    status, data = await analyze_parcel(body)
    return JSONResponse(status_code=status, content=data)


@router.post("/api/parcelles/analyze")
async def analyze_single_parcel(request: Request) -> JSONResponse:
    """Analyse **une seule** parcelle et enregistre son résultat.

    Seule la parcelle transmise est analysée : les parcelles voisines ne sont ni
    cherchées ni modifiées, même si elles tombent dans le même rayon.

    Une parcelle déjà analysée conserve ses données : la route répond 409 tant que
    l'utilisateur n'a pas explicitement demandé de relancer l'analyse (`force`).
    """
    try:
        body = await request.json()
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": "Corps de requête invalide."})
    parsed = parse_or_400(AnalyzeParcelRequest, body, "Données d'analyse de parcelle invalides.")

    coordinates = normalize_polygon([{"lat": point.lat, "lng": point.lng} for point in parsed.polygon])
    if coordinates is None:
        return JSONResponse(status_code=400, content={"error": "Le contour de la parcelle est invalide."})

    center = polygon_centroid(coordinates)
    surface_ha = js_round((approximate_polygon_area_m2(coordinates) / 10_000) * 100) / 100

    existing, unknown_id = await find_parcel_for_analysis(parsed.parcel_id, center["lat"], center["lng"])
    if unknown_id:
        return JSONResponse(status_code=404, content={"error": "Parcelle introuvable."})

    if existing is not None and values_have_analysis(existing) and not parsed.force:
        return JSONResponse(
            status_code=409,
            content={
                "error": ALREADY_ANALYZED_MESSAGE,
                "already_analyzed": True,
                "parcelle": existing,
            },
        )

    status, analysis = await analyze_parcel(
        {
            "lat": center["lat"],
            "lng": center["lng"],
            "zoom": parsed.zoom,
            "polygon": coordinates,
        }
    )
    if status != 200:
        return JSONResponse(status_code=status, content=analysis)

    row, outcome = await store_parcel_analysis(
        parcel_id=parsed.parcel_id,
        coordinates=coordinates,
        center_lat=center["lat"],
        center_lng=center["lng"],
        surface_ha=surface_ha,
        analysis=analysis,
        force=parsed.force,
        label=existing.get("label") if existing else None,
        owner_name=existing.get("owner_name") if existing else None,
        notes=existing.get("notes") if existing else None,
    )
    if outcome == STORE_NOT_FOUND:
        return JSONResponse(status_code=404, content={"error": "Parcelle introuvable."})
    if outcome == STORE_ALREADY_ANALYZED:
        return JSONResponse(
            status_code=409,
            content={
                "error": ALREADY_ANALYZED_MESSAGE,
                "already_analyzed": True,
                "parcelle": row,
            },
        )

    cache.invalidate()
    return JSONResponse(
        content={
            "analysis": analysis,
            "parcelle": row,
            "created": outcome == STORE_CREATED,
            "already_analyzed": False,
        }
    )
