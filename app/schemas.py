"""Pydantic — miroir des schémas zod de src/server.ts et src/routes/planet.routes.ts."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("doit être un nombre fini")
    return v


class Point(BaseModel):
    lat: float
    lng: float


MAX_TRACED_PARCELS = 2_000


class TracedParcel(BaseModel):
    """Parcelle à classer : contour tracé à l'écran (JECAM-VHR) ou ligne déjà
    enregistrée dans `parcelles` (import KML, tracé manuel).

    `id` est l'identifiant de la ligne en base : quand il est fourni, l'analyse
    met cette ligne à jour au lieu de créer un doublon au même endroit. Il reste
    absent pour les contours qui n'existent qu'à l'écran.
    """

    coordinates: list[Point] = Field(min_length=3, max_length=5_000)
    id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "coordinates": [point.model_dump() for point in self.coordinates],
        }


class AnalyzeSimpleRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius: float = Field(ge=50, le=20_000)
    confidence_threshold: float | None = Field(default=None, ge=0.5, le=0.95, alias="confidenceThreshold")
    min_area_ha: float | None = Field(default=None, ge=0.01, le=5, alias="minAreaHa")
    base_temperature: float | None = Field(default=None, ge=-20, le=30, alias="baseTemperature")
    threshold: float | None = Field(default=None, ge=2000, le=10_000)
    period_days: int | None = Field(default=None, ge=1, le=730, alias="periodDays")
    traced_parcels: list[TracedParcel] = Field(default_factory=list, max_length=MAX_TRACED_PARCELS, alias="tracedParcels")

    _v_lat = field_validator("latitude", "longitude", "radius")(_finite)


class AutomaticDetectionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    radius_km: float = Field(ge=0.05, le=20, alias="radiusKm")
    base_temperature: float = Field(ge=-20, le=30, alias="baseTemperature")
    threshold: float = Field(ge=2000, le=10_000)
    period_days: int = Field(ge=1, le=730, alias="periodDays")
    traced_parcels: list[TracedParcel] = Field(default_factory=list, max_length=MAX_TRACED_PARCELS, alias="tracedParcels")

    _v = field_validator("lat", "lng", "radius_km", "base_temperature", "threshold")(_finite)


class FieldSegmentationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_base64: str = Field(min_length=1, alias="fileBase64")
    filename: str = Field(min_length=1)
    threshold: float | None = Field(default=None, ge=0, le=1)

    _v = field_validator("threshold")(lambda v: _finite(v) if v is not None else v)


class ParcelleCreate(BaseModel):
    label: str = Field(max_length=200)
    coordinates: list[Point] = Field(min_length=3)

    center_lat: float
    center_lng: float

    surface_ha: float | None

    culture_declared: str
    culture_detected: str | None

    ndvi_percentage: float | None
    ndre: float | None

    spectral_bands: dict[str, float | None] | None

    confidence: float | None
    verdict: str | None
    details: str | None

    saison: str | None
    soil_type: str | None

    risk_factors: list[str]

    recommendations: str | None
    data_source: str | None

    owner_name: str
    notes: str

    time_series_s2: list[Any]
    time_series_s1: list[Any]
    time_series_rain: list[Any]

    estimated_planting_date: str | None
    estimated_harvest_date: str | None

    days_since_planting: float | None

    growth_stage: str | None
    planting_confidence: float | None

    evi: float | None
    savi: float | None
    ndwi: float | None

    agro_score: float | None
    hybrid_score: float | None

    cnn_prob_barley: float | None
    cnn_prob_non_barley: float | None


class SentinelTileParams(BaseModel):
    z: int = Field(ge=10, le=19)
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class SentinelTileQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_timestamp_ms: float = Field(gt=0, alias="imageTimestampMs")

    _v = field_validator("image_timestamp_ms")(_finite)


class PlanetTileParams(BaseModel):
    z: int = Field(ge=0, le=22)
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class PlanetTileQuery(BaseModel):
    mosaic: str | None = Field(default=None, min_length=1, max_length=300)


class AnalyzeLatestQuery(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)

    _v = field_validator("lat", "lng")(_finite)


class AnalyzeParcelRequest(BaseModel):
    """Analyse d'une seule parcelle (bouton « Analyser » du frontend).

    Le contour est obligatoire : l'analyse porte sur cette parcelle uniquement,
    jamais sur les parcelles voisines. `parcel_id` identifie la ligne à mettre à
    jour dans `parcelles` ; s'il est absent (contour tracé à l'écran, pas encore
    en base), la ligne est créée.
    """

    model_config = ConfigDict(populate_by_name=True)

    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    zoom: float | None = Field(default=None, ge=1, le=21)
    polygon: list[Point] = Field(min_length=3, max_length=5_000)
    parcel_id: str | None = Field(default=None, alias="parcelId")
    force: bool = False

    _v = field_validator("lat", "lng")(_finite)


class GeminiViewportBounds(BaseModel):
    south: float = Field(ge=-90, le=90)
    west: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)

    _v = field_validator("south", "west", "north", "east")(_finite)


class GeminiParcelsRequest(BaseModel):
    """Capture satellite (zone écran visible ou rayon autour d'un point) et détection Gemini.

    Si `viewport_width` et `viewport_height` (ou `bounds`) sont fournis, la capture
    couvre exactement l'intégralité du champ de vision visible à l'écran.
    Sinon, `span_m` et `tile_span_m` définissent la zone capturée autour du centre.
    """

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    span_m: float | None = Field(default=None, ge=10.0, le=10000.0)
    tile_span_m: float | None = Field(default=None, ge=10.0, le=10000.0)
    zoom: int | None = Field(default=None, ge=10, le=22)
    persist: bool = True
    viewport_width: int | None = Field(default=None, ge=100, le=4000)
    viewport_height: int | None = Field(default=None, ge=100, le=4000)
    bounds: GeminiViewportBounds | None = None

    _v = field_validator("latitude", "longitude")(_finite)

    @field_validator("span_m", "tile_span_m")
    @classmethod
    def _finite_span(cls, value: float | None) -> float | None:
        return None if value is None else _finite(value)


class DeleteParcelleParams(BaseModel):
    id: str

    @field_validator("id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str:
        import uuid

        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError("doit être un UUID valide") from exc
        return v


class ParcelleUpdate(BaseModel):
    """Champs modifiables lors d'une correction humaine sur une parcelle."""
    culture_detected: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=100.0)
    verdict: str | None = None
    details: str | None = None
    saison: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class AuditEntryCreate(BaseModel):
    """Création d'une entrée d'audit (journal des actions utilisateurs)."""
    user_name: str = Field(min_length=1, max_length=200)
    action: str = Field(min_length=1, max_length=100)  # ex. "correction", "rejet", "observation", "connexion"
    object: str = Field(min_length=1, max_length=300)  # ex. "Parcelle <id>", "Session analyste"


class AuditEntryResponse(BaseModel):
    """Entrée d'audit renvoyée par l'API."""
    id: str
    at: str
    user_name: str
    action: str
    object: str

    model_config = ConfigDict(populate_by_name=True)
