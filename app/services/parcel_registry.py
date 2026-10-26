"""Registre des parcelles : garde-fou « ne pas réanalyser » et écriture des résultats.

Règle centrale : **une parcelle déjà analysée ne l'est plus automatiquement.**

Dès qu'une parcelle porte un résultat de classification en base, les analyses de
rayon (`POST /api/analyze`, `POST /api/detect-parcels`) l'ignorent et son
enregistrement n'est jamais écrasé : les données affichées restent celles de la
première analyse, même si une nouvelle analyse passe dans le même rayon.

La seule voie qui puisse réécrire ces données est l'analyse explicite d'une
seule parcelle (bouton « Analyser » du frontend) — et uniquement si
l'utilisateur le demande (`force=True`).

Le module centralise aussi la sérialisation des lignes `parcelles` (déplacée
depuis `app/routes/parcelles.py`) afin que les routes et les services parlent
du même format.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.db import async_session_maker
from app.models import Parcelle

logger = logging.getLogger("agrisat.parcel_registry")


# Marqueur de suppression persistée d'un contour automatique (cf. routes/parcelles.py).
# Une ligne portant ce marqueur n'est pas une parcelle analysée : c'est une trace
# de suppression, elle ne doit jamais être mise à jour par une analyse.
REJECTED_PARCEL_SOURCE = "Tracé automatique supprimé"

# Deux parcelles dont les centroïdes sont à moins de 20 m désignent le même champ :
# le contour tracé à l'écran et la ligne enregistrée en base ne coïncident jamais
# exactement (même source de vérité, pas les mêmes sommets).
ANALYSIS_MATCH_TOLERANCE_M = 20.0

# Champs exposés par l'API, identiques à ceux connus de Prisma (schema.prisma) — exclut
# volontairement `updated_at`, une colonne présente en base mais jamais déclarée côté
# Prisma et donc jamais renvoyée par l'ancien backend.
PARCELLE_FIELDS = (
    "id", "label", "coordinates", "center_lat", "center_lng", "surface_ha",
    "culture_declared", "culture_detected", "ndvi_percentage", "ndre", "spectral_bands",
    "confidence", "verdict", "details", "saison", "soil_type", "risk_factors",
    "recommendations", "data_source", "owner_name", "notes", "time_series_s1",
    "time_series_s2", "time_series_rain", "estimated_planting_date", "estimated_harvest_date",
    "days_since_planting", "growth_stage", "planting_confidence", "evi", "savi", "ndwi",
    "agro_score", "hybrid_score", "cnn_prob_barley", "cnn_prob_non_barley", "created_at",
)


def serialize_parcelle(row: Parcelle) -> dict[str, Any]:
    """Ligne `parcelles` → objet JSON renvoyé par l'API."""
    out: dict[str, Any] = {}
    for field in PARCELLE_FIELDS:
        value = getattr(row, field)
        if isinstance(value, datetime):
            value = value.isoformat()
        else:
            value = str(value) if field == "id" else value
        out[field] = value
    return out


# ── Petites conversions partagées ──


def as_number(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def as_string(value: Any) -> str | None:
    return value if isinstance(value, str) and len(value) > 0 else None


def as_string_array(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


# ── Détection d'une parcelle déjà analysée ──


def row_has_analysis(row: Parcelle) -> bool:
    """Vrai si la ligne porte déjà un résultat d'analyse (données de culture).

    Un contour simplement tracé (`culture_declared = « Inconnue »`, aucun indice)
    n'est pas une parcelle analysée : il peut donc encore être analysé.
    """
    if row.data_source == REJECTED_PARCEL_SOURCE:
        return False
    if (row.culture_detected or "").strip():
        return True
    return any(
        value is not None
        for value in (row.confidence, row.ndvi_percentage, row.agro_score, row.hybrid_score, row.cnn_prob_barley)
    )


def values_have_analysis(values: dict[str, Any]) -> bool:
    """Vrai si une ligne sérialisée (donc en JSON) porte déjà une analyse."""
    if values.get("data_source") == REJECTED_PARCEL_SOURCE:
        return False
    detected = values.get("culture_detected")
    if isinstance(detected, str) and detected.strip():
        return True
    return any(
        values.get(field) is not None
        for field in ("confidence", "ndvi_percentage", "agro_score", "hybrid_score", "cnn_prob_barley")
    )


def distance_m(lat_a: float, lng_a: float, lat_b: float, lng_b: float) -> float:
    """Distance approchée (m) entre deux points, à la latitude de référence."""
    meters_per_deg_lat = 111_320
    meters_per_deg_lng = 111_320 * math.cos((lat_a * math.pi) / 180)
    dx = (lng_b - lng_a) * meters_per_deg_lng
    dy = (lat_b - lat_a) * meters_per_deg_lat
    return math.hypot(dx, dy)


def nearest_analyzed_parcel(
    rows: list[Parcelle],
    lat: float,
    lng: float,
    tolerance_m: float = ANALYSIS_MATCH_TOLERANCE_M,
) -> Parcelle | None:
    """Ligne déjà analysée la plus proche du point, si elle est à portée."""
    best: Parcelle | None = None
    best_distance = tolerance_m
    for row in rows:
        if not row_has_analysis(row):
            continue
        candidate_distance = distance_m(lat, lng, row.center_lat, row.center_lng)
        if candidate_distance <= best_distance:
            best, best_distance = row, candidate_distance
    return best


def nearest_parcel(
    rows: list[Parcelle],
    lat: float,
    lng: float,
    tolerance_m: float = ANALYSIS_MATCH_TOLERANCE_M,
) -> Parcelle | None:
    """Ligne la plus proche du point (analysée ou non), si elle est à portée."""
    best: Parcelle | None = None
    best_distance = tolerance_m
    for row in rows:
        candidate_distance = distance_m(lat, lng, row.center_lat, row.center_lng)
        if candidate_distance <= best_distance:
            best, best_distance = row, candidate_distance
    return best


async def load_parcels_in_area(lat: float, lng: float, radius_m: float) -> list[Parcelle]:
    """Lignes `parcelles` dont le centre tombe dans le disque (lat, lng, radius_m).

    La base est lente à réveiller : une requête filtrée reste bien plus rapide
    qu'une lecture complète (celle du cache, 20–35 s). En cas d'échec, on
    renvoie une liste vide : l'analyse continue, elle est simplement moins
    discriminante (le garde-fou ne peut alors pas s'appuyer sur la base).
    """
    try:
        async with async_session_maker() as session:
            return await _load_parcels_in_area(session, lat, lng, radius_m)
    except Exception as error:  # noqa: BLE001 — un registre indisponible n'arrête pas l'analyse
        logger.warning("Registre des parcelles indisponible : %s", error)
        return []


def bbox_deltas(lat: float, radius_m: float) -> tuple[float, float]:
    """Demi-dimensions (degrés) de la boîte englobante d'un disque."""
    lat_delta = radius_m / 110.574
    lng_delta = radius_m / (111.32 * max(abs(math.cos((lat * math.pi) / 180)), 0.1))
    return lat_delta, lng_delta


async def _load_parcels_in_area(session, lat: float, lng: float, radius_m: float) -> list[Parcelle]:
    lat_delta, lng_delta = bbox_deltas(lat, radius_m)
    stmt = select(Parcelle).where(
        Parcelle.center_lat >= lat - lat_delta,
        Parcelle.center_lat <= lat + lat_delta,
        Parcelle.center_lng >= lng - lng_delta,
        Parcelle.center_lng <= lng + lng_delta,
    )
    return list((await session.execute(stmt)).scalars().all())


async def find_analyzed_parcel_in_session(
    session,
    lat: float,
    lng: float,
    tolerance_m: float = ANALYSIS_MATCH_TOLERANCE_M,
) -> Parcelle | None:
    """Parcelle déjà analysée proche du point, recherchée dans une session ouverte."""
    rows = await _load_parcels_in_area(session, lat, lng, max(tolerance_m * 3, 100.0))
    return nearest_analyzed_parcel(rows, lat, lng, tolerance_m)


async def find_analyzed_parcel_near(
    lat: float,
    lng: float,
    tolerance_m: float = ANALYSIS_MATCH_TOLERANCE_M,
) -> Parcelle | None:
    """Parcelle déjà analysée proche du point (None si la base est muette)."""
    rows = await load_parcels_in_area(lat, lng, max(tolerance_m * 3, 100.0))
    return nearest_analyzed_parcel(rows, lat, lng, tolerance_m)


async def filter_analyzed_candidates(
    candidates: list[dict[str, Any]],
    lat: float,
    lng: float,
    radius_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sépare les candidats déjà analysés des candidats à analyser.

    Retourne (candidats_à_analyser, candidats_ignorés). Un candidat dont le
    centre tombe sur une parcelle déjà analysée en base est écarté : il ne sera
    ni reclassé, ni réécrit.
    """
    if not candidates:
        return [], []

    rows = await load_parcels_in_area(lat, lng, radius_m)
    if not rows:
        return candidates, []

    rows_by_id = {str(row.id): row for row in rows}
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in candidates:
        # Parcelle déjà enregistrée et clairement identifiée (import KML, tracé manuel) :
        # seule sa propre ligne fait foi. La comparaison de proximité ne doit pas lui
        # opposer le résultat d'une parcelle voisine, sans quoi elle ne serait jamais
        # analysée alors que sa ligne en base est vide.
        stored = rows_by_id.get(str(candidate.get("id") or ""))
        if stored is not None:
            (skipped if row_has_analysis(stored) else kept).append(candidate)
            continue
        center = candidate.get("center") or {}
        candidate_lat, candidate_lng = center.get("lat"), center.get("lng")
        if not isinstance(candidate_lat, (int, float)) or not isinstance(candidate_lng, (int, float)):
            kept.append(candidate)
            continue
        match = nearest_analyzed_parcel(rows, float(candidate_lat), float(candidate_lng))
        if match is None:
            kept.append(candidate)
        else:
            skipped.append(candidate)
    return kept, skipped


# ── Écriture d'une analyse dans `parcelles` ──


def analysis_to_values(
    analysis: dict[str, Any],
    *,
    label: str,
    coordinates: list[dict[str, float]],
    center_lat: float,
    center_lng: float,
    surface_ha: float | None,
    owner_name: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Résultat d'analyse → colonnes de la table `parcelles`."""
    days_since_planting = analysis.get("days_since_planting")
    return {
        "label": label,
        "coordinates": coordinates,
        "center_lat": center_lat,
        "center_lng": center_lng,
        "surface_ha": surface_ha,
        "culture_declared": None,
        "culture_detected": as_string(analysis.get("culture_detected")),
        "ndvi_percentage": as_number(analysis.get("percentage")),
        "ndre": as_number(analysis.get("ndre")),
        "spectral_bands": analysis.get("spectral_bands") if isinstance(analysis.get("spectral_bands"), dict) else None,
        "confidence": as_number(analysis.get("confidence")),
        "verdict": as_string(analysis.get("verdict")),
        "details": as_string(analysis.get("details")),
        "saison": as_string(analysis.get("saison")),
        "soil_type": as_string(analysis.get("soil_type")),
        "risk_factors": as_string_array(analysis.get("risk_factors")),
        "recommendations": as_string(analysis.get("recommendations")),
        "data_source": as_string(analysis.get("data_source")),
        "owner_name": owner_name,
        "notes": notes,
        "time_series_s1": analysis.get("time_series_s1") if isinstance(analysis.get("time_series_s1"), list) else [],
        "time_series_s2": analysis.get("time_series_s2") if isinstance(analysis.get("time_series_s2"), list) else [],
        "time_series_rain": analysis.get("time_series_rain") if isinstance(analysis.get("time_series_rain"), list) else [],
        "estimated_planting_date": as_string(analysis.get("estimated_planting_date")),
        "estimated_harvest_date": as_string(analysis.get("estimated_harvest_date")),
        "days_since_planting": int(days_since_planting) if isinstance(days_since_planting, (int, float)) and not isinstance(days_since_planting, bool) else None,
        "growth_stage": as_string(analysis.get("growth_stage")),
        "planting_confidence": as_number(analysis.get("planting_confidence")),
        "evi": as_number(analysis.get("evi")),
        "savi": as_number(analysis.get("savi")),
        "ndwi": as_number(analysis.get("ndwi")),
        "agro_score": as_number(analysis.get("agro_score")),
        "hybrid_score": as_number(analysis.get("hybrid_score")),
        "cnn_prob_barley": as_number(analysis.get("cnn_prob_barley")),
        "cnn_prob_non_barley": as_number(analysis.get("cnn_prob_non_barley")),
    }


# Résultat de `store_parcel_analysis` : ligne enregistrée + statut.
STORE_CREATED = "created"
STORE_UPDATED = "updated"
STORE_ALREADY_ANALYZED = "already_analyzed"
STORE_NOT_FOUND = "not_found"


def _as_parcel_uuid(parcel_id: str | None) -> uuid.UUID | None:
    """UUID de la ligne `parcelles`, ou None si l'identifiant n'en est pas un.

    Les contours tracés à l'écran portent un identifiant de session (`tile-…-VHR-…`)
    qui ne correspond à aucune ligne en base : ils sont créés à la première analyse.
    """
    if not parcel_id:
        return None
    try:
        return uuid.UUID(str(parcel_id))
    except (ValueError, AttributeError, TypeError):
        return None


async def find_parcel_for_analysis(
    parcel_id: str | None,
    lat: float,
    lng: float,
) -> tuple[dict[str, Any] | None, bool]:
    """Ligne visée par une analyse de parcelle unique.

    Retourne (ligne sérialisée ou None, identifiant_inconnu). `identifiant_inconnu`
    vaut True quand un identifiant a été fourni mais ne correspond à aucune ligne :
    l'appelant doit alors répondre 404.

    Sans identifiant (contour tracé à l'écran), la ligne déjà analysée la plus
    proche est retenue en priorité : plusieurs lignes peuvent partager le même
    centroïde (contour FTW + contour JECAM-VHR) et c'est la ligne porteuse de
    l'analyse qui fait foi.
    """
    parcel_uuid = _as_parcel_uuid(parcel_id)
    async with async_session_maker() as session:
        if parcel_uuid is not None:
            row = await session.get(Parcelle, parcel_uuid)
            if row is None:
                return None, True
            return serialize_parcelle(row), False

        nearby = await _load_parcels_in_area(session, lat, lng, max(ANALYSIS_MATCH_TOLERANCE_M * 3, 100.0))
        row = nearest_analyzed_parcel(nearby, lat, lng) or nearest_parcel(nearby, lat, lng)
        return (serialize_parcelle(row) if row is not None else None), False


async def store_parcel_analysis(
    *,
    parcel_id: str | None,
    coordinates: list[dict[str, float]],
    center_lat: float,
    center_lng: float,
    surface_ha: float | None,
    analysis: dict[str, Any],
    force: bool = False,
    label: str | None = None,
    owner_name: str | None = None,
    notes: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Enregistre le résultat d'une analyse dans `parcelles`.

    - `parcel_id` donné (UUID) : la ligne est mise à jour (404 si elle n'existe pas).
    - sinon : la ligne la plus proche du centre est reprise si elle existe, sinon
      une nouvelle ligne est créée (contour tracé à l'écran).

    Une ligne déjà analysée n'est jamais écrasée implicitement : sans `force`,
    la fonction renvoie `already_analyzed` et laisse les données inchangées.
    """
    values = analysis_to_values(
        analysis,
        label=label or f"analyse-{center_lat:.5f}-{center_lng:.5f}",
        coordinates=coordinates,
        center_lat=center_lat,
        center_lng=center_lng,
        surface_ha=surface_ha,
        owner_name=owner_name,
        notes=notes,
    )

    parcel_uuid = _as_parcel_uuid(parcel_id)
    async with async_session_maker() as session:
        row: Parcelle | None = None

        if parcel_uuid is not None:
            row = await session.get(Parcelle, parcel_uuid)
            if row is None:
                return None, STORE_NOT_FOUND
        else:
            nearby = await _load_parcels_in_area(session, center_lat, center_lng, max(ANALYSIS_MATCH_TOLERANCE_M * 3, 100.0))
            # La ligne déjà analysée prime : c'est elle qui porte les données à
            # préserver (et à remplacer en cas de relance explicite).
            row = nearest_analyzed_parcel(nearby, center_lat, center_lng) or nearest_parcel(nearby, center_lat, center_lng)

        if row is not None and row_has_analysis(row) and not force:
            return serialize_parcelle(row), STORE_ALREADY_ANALYZED

        if row is None:
            row = Parcelle(**values)
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return serialize_parcelle(row), STORE_CREATED

        values["label"] = row.label
        for key, value in values.items():
            setattr(row, key, value)
        await session.commit()
        await session.refresh(row)
        return serialize_parcelle(row), STORE_UPDATED
