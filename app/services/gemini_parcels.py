"""Détection de parcelles par Gemini à partir d'une capture satellite.

Principe
--------
1. Une image satellite est demandée à Google Static Maps (même clé que le
   frontend), centrée sur le centre de la carte. Pour couvrir environ
   `GEMINI_CAPTURE_SPAN_M` mètres en conservant le détail métrique d'une tuile
   de `GEMINI_CAPTURE_TILE_SPAN_M` mètres (le zoom Static Maps est plafonné à
   21), plusieurs tuiles contiguës sont assemblées en une mosaïque.
2. L'image est envoyée à Gemini avec une réponse JSON structurée : chaque
   parcelle est un anneau de points normalisés (0–1) dans l'image.
3. Les points sont géoréférencés par la projection Web Mercator du zoom Static
   Maps (mêmes bornes que l'image), puis convertis en polygones métriques pour
   valider la forme et calculer surface/centroïde.
4. Les contours déjà présents en base (IoU ou centroïde proche) sont écartés :
   aucune ligne en double n'est créée.
5. Les nouveaux contours sont enregistrés dans `parcelles` avec un UUID et
   `data_source = « Détection Gemini »`, tous les champs de culture à NULL.
   Ils suivent alors exactement le même chemin qu'un contour importé par KML :
   affichage sur la carte, sélection, analyse unitaire ou par lot.

La clé Gemini est distincte de la clé Maps : une clé Google Maps n'a pas accès
à l'API Generative Language. `GEMINI_API_KEY` est donc attendue ; à défaut la
clé Maps est tentée en repli (et Gemini la refuse le plus souvent).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import httpx
from PIL import Image
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from sqlalchemy import select

from app import cache
from app.config import settings
from app.db import async_session_maker
from app.models import Parcelle
from app.services.gemini_parcels_image import (
    framing_warning,
    parcel_rings_from_red_lines,
)
from app.services.parcel_registry import (
    REJECTED_PARCEL_SOURCE,
    load_parcels_in_area,
    serialize_parcelle,
)

logger = logging.getLogger("agrisat.services.gemini_parcels")

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TILE_SIZE = 256
EARTH_CIRCUMFERENCE_M = 40_075_016.686
METERS_PER_PIXEL_AT_EQUATOR = EARTH_CIRCUMFERENCE_M / TILE_SIZE
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LNG_AT_EQUATOR = 111_320.0
WEB_MERCATOR_MAX_LAT = 85.05112878

MIN_ZOOM = 12
MAX_ZOOM = 22
# Largeur logique minimale d'une tuile : en dessous, Gemini manque de détails.
MIN_CAPTURE_SIZE_PX = 128
MAX_CAPTURE_SIZE_PX = 640
# Largeur visée de la capture (m). Une tuile de 20 m correspond au détail demandé ;
# comme le zoom Static Maps est plafonné à 21, la capture est une mosaïque de tuiles
# de 20 m (5 × 5 = 100 m) : Gemini voit les parcelles entières sans perdre le détail.
MIN_CAPTURE_SPAN_M = 10.0
MAX_CAPTURE_SPAN_M = 400.0
MAX_MOSAIC_TILES = 5
# La capture large est analysée par fenêtres qui se chevauchent : une seule image
# de plusieurs centaines de parcelles dépasserait la sortie JSON de Gemini et
# perdrait en précision. Le chevauchement dépasse la taille d'une parcelle courante,
# de sorte que toute parcelle coupée par un bord est entière dans la fenêtre voisine.
WINDOW_SPAN_M = 200.0
WINDOW_STRIDE_M = 150.0
WINDOW_UPSCALE_PX = 1280
WINDOW_EDGE_MARGIN = 0.012
MAX_PARALLEL_GEMINI_CALLS = 4
# Un appel Static Maps couvre au plus 640 px de terrain à un zoom donné. `scale=2`
# DOUBLE LE NOMBRE DE PIXELS SANS ÉLARGIR LA COUVERTURE : c'est `size` qui fixe
# l'étendue géographique. Découper l'écran en cellules de `STATIC_MAPS_MAX_SIZE`
# px de terrain est donc la seule façon de couvrir exactement la vue affichée.
STATIC_MAPS_MAX_SIZE = 640
# Résolution de la mosaïque écran, en multiples des pixels de la carte : à 2, une
# limite large d'un pixel à l'écran en occupe deux dans l'image envoyée à Gemini.
SCREEN_CAPTURE_SCALE = 2
# En dessous de 100 m², un contour est plus probablement un artefact qu'un champ.
MIN_AREA_HA = 0.01
MAX_DETECTED_PARCELS = 600

PARCEL_SOURCE = "Détection Gemini"
LABEL_PREFIX = "Parcelle Gemini"
_LABEL_PATTERN = re.compile(rf"^{re.escape(LABEL_PREFIX)}\s+(\d+)$", re.IGNORECASE)

# Seuils de rapprochement : un contour Gemini ne coïncide jamais exactement avec
# une ligne déjà en base (sommets, résolution, date d'image différents).
EXISTING_IOU_THRESHOLD = 0.55
EXISTING_AREA_RATIO = 0.40
CANDIDATE_IOU_THRESHOLD = 0.75


class GeminiParcelsError(RuntimeError):
    """Requête refusée ou résultat inexploitable (réponse 400)."""


class GeminiConfigurationError(RuntimeError):
    """Clé Gemini absente ou refusée (réponse 503)."""


class GeminiUnavailable(RuntimeError):
    """Static Maps ou Gemini injoignable en erreur (réponse 502)."""


@dataclass
class Capture:
    """Image Static Maps et géoréférencement exact de ses quatre bords."""

    center_lat: float
    center_lng: float
    zoom: int
    span_m: float
    size_px: int
    south: float
    west: float
    north: float
    east: float
    image_bytes: bytes
    tiles: int = 1
    tile_span_m: float | None = None
    width_px: int | None = None
    height_px: int | None = None


@dataclass
class Candidate:
    """Contour détecté, validé et projeté dans le repère métrique local."""

    geom: Polygon
    coordinates: list[dict[str, float]]
    center_lat: float
    center_lng: float
    area_ha: float
    confidence: float | None
    label: str | None


@dataclass
class ExistingParcel:
    """Ligne `parcelles` voisine, projetée dans le même repère métrique."""

    id: str
    label: str | None
    surface_ha: float | None
    geom: Polygon


# ── Projections ──


def _mercator_world_px(zoom: int) -> float:
    return TILE_SIZE * (2 ** zoom)


def _lng_to_px(lng: float, world_px: float) -> float:
    return (lng + 180.0) / 360.0 * world_px


def _lat_to_px(lat: float, world_px: float) -> float:
    lat = max(min(lat, WEB_MERCATOR_MAX_LAT), -WEB_MERCATOR_MAX_LAT)
    sin_lat = math.sin(math.radians(lat))
    return (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * world_px


def _px_to_lng(x: float, world_px: float) -> float:
    return x / world_px * 360.0 - 180.0


def _px_to_lat(y: float, world_px: float) -> float:
    n = math.pi - 2 * math.pi * y / world_px
    return math.degrees(math.atan(math.sinh(n)))


def _meters_per_pixel(lat: float, zoom: int) -> float:
    return METERS_PER_PIXEL_AT_EQUATOR * math.cos(math.radians(lat)) / (2 ** zoom)


def _zoom_for_span(lat: float, span_m: float, size_px: int) -> int:
    """Zoom Static Maps dont la largeur d'image s'approche le plus de `span_m`."""
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    ideal = math.log2(METERS_PER_PIXEL_AT_EQUATOR * cos_lat * size_px / max(span_m, 1.0))
    return max(MIN_ZOOM, min(MAX_ZOOM, int(round(ideal))))


def _normalized_to_latlng(x: float, y: float, capture: Capture) -> tuple[float, float]:
    """Point normalisé (0–1, origine en haut à gauche) → (lat, lng)."""
    world_px = _mercator_world_px(capture.zoom)
    left_x = _lng_to_px(capture.west, world_px)
    right_x = _lng_to_px(capture.east, world_px)
    top_y = _lat_to_px(capture.north, world_px)
    bottom_y = _lat_to_px(capture.south, world_px)
    px = left_x + x * (right_x - left_x)
    py = top_y + y * (bottom_y - top_y)
    return _px_to_lat(py, world_px), _px_to_lng(px, world_px)


def _latlng_to_metric(lat: float, lng: float, center_lat: float, center_lng: float) -> tuple[float, float]:
    """Repère métrique local (m) : x vers l'est, y vers le nord."""
    kx = M_PER_DEG_LNG_AT_EQUATOR * max(math.cos(math.radians(center_lat)), 0.1)
    return (lng - center_lng) * kx, (lat - center_lat) * M_PER_DEG_LAT


def _metric_to_latlng(x: float, y: float, center_lat: float, center_lng: float) -> tuple[float, float]:
    kx = M_PER_DEG_LNG_AT_EQUATOR * max(math.cos(math.radians(center_lat)), 0.1)
    return center_lat + y / M_PER_DEG_LAT, center_lng + x / kx


def _viewport_tiling(cover_w: float, cover_h: float) -> tuple[int, int, float, float, int, int]:
    """Grille couvrant exactement `cover_w` × `cover_h` px de terrain à un zoom.

    Un appel Static Maps ne couvre que `STATIC_MAPS_MAX_SIZE` px de terrain à ce
    zoom ; `scale` ne change que le nombre de pixels rendus, jamais l'étendue.
    Le nombre de tuiles se déduit donc de l'étendue, et non de la taille de
    l'écran : c'est l'erreur qui décalait les parcelles d'un facteur deux.

    Retourne (colonnes, lignes, largeur_cellule, hauteur_cellule, largeur_image,
    hauteur_image), les cellules en px de terrain et l'image en px rendus.
    """
    cover_w = max(1.0, cover_w)
    cover_h = max(1.0, cover_h)
    cols = max(1, int(math.ceil(cover_w / STATIC_MAPS_MAX_SIZE)))
    rows = max(1, int(math.ceil(cover_h / STATIC_MAPS_MAX_SIZE)))
    out_w = max(1, int(round(cover_w * SCREEN_CAPTURE_SCALE)))
    out_h = max(1, int(round(cover_h * SCREEN_CAPTURE_SCALE)))
    return cols, rows, cover_w / cols, cover_h / rows, out_w, out_h


def _valid_bounds(bounds: dict[str, float]) -> bool:
    """Bornes exploitables : présentes, finies et ordonnées.

    Une carte tournée ou une vue dégénérée peut envoyer des bornes incohérentes ;
    dans ce cas la capture retombe sur le calcul par centre + zoom.
    """
    try:
        south = float(bounds["south"])
        west = float(bounds["west"])
        north = float(bounds["north"])
        east = float(bounds["east"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (south, west, north, east)):
        return False
    return south < north and west < east


# ── Capture satellite ──


async def _fetch_static_map(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    zoom: int,
    size_px: int,
    maps_key: str,
    *,
    scale: int = 2,
) -> bytes:
    """Télécharge une image Static Maps et vérifie que Google renvoie bien une image."""
    params = {
        "center": f"{latitude:.7f},{longitude:.7f}",
        "zoom": zoom,
        "size": f"{size_px}x{size_px}",
        "scale": scale,
        "maptype": "satellite",
        "format": "png",
        "language": "fr",
        "key": maps_key,
    }
    try:
        response = await client.get(STATIC_MAPS_URL, params=params)
    except httpx.TimeoutException as exc:
        raise GeminiUnavailable("La capture satellite a dépassé le délai imparti.") from exc
    except httpx.HTTPError as exc:
        # Ne pas reprendre str(exc) : l'URL Static Maps contient la clé API.
        raise GeminiUnavailable(
            f"Capture satellite injoignable ({type(exc).__name__})."
        ) from exc

    if response.status_code >= 400 or not response.content:
        detail = _static_maps_error(response)
        raise GeminiUnavailable(f"Capture satellite refusée par Google Maps : {detail}")
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise GeminiUnavailable(
            "La capture satellite n'a pas renvoyé d'image "
            f"(content-type {content_type or 'inconnu'}) : {_static_maps_error(response)}"
        )
    return response.content


def _max_capture_size_px() -> int:
    return max(
        MIN_CAPTURE_SIZE_PX,
        min(MAX_CAPTURE_SIZE_PX, int(settings.gemini_capture_size_px or MAX_CAPTURE_SIZE_PX)),
    )


def _capture_geometry(lat: float, span_m: float, max_size_px: int) -> tuple[int, int]:
    """Choisit le zoom le plus précis dont la tuile couvre `span_m` sans dépasser la taille max."""
    for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        meters_per_px = _meters_per_pixel(lat, zoom)
        size = int(round(span_m / max(meters_per_px, 1e-9)))
        if MIN_CAPTURE_SIZE_PX <= size <= max_size_px:
            return zoom, size
    zoom = _zoom_for_span(lat, span_m, max_size_px)
    meters_per_px = _meters_per_pixel(lat, zoom)
    size = int(round(span_m / max(meters_per_px, 1e-9)))
    size = max(MIN_CAPTURE_SIZE_PX, min(max_size_px, size))
    return zoom, size


async def _capture_satellite(
    latitude: float,
    longitude: float,
    span_m: float,
    zoom: int | None,
    maps_key: str,
) -> Capture:
    max_size_px = _max_capture_size_px()
    if zoom is not None:
        chosen_zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
    else:
        chosen_zoom = _zoom_for_span(latitude, span_m, max_size_px)
        chosen_zoom = max(MIN_ZOOM, min(MAX_ZOOM, chosen_zoom))

    meters_per_px = _meters_per_pixel(latitude, chosen_zoom)
    size_px = int(round(span_m / max(meters_per_px, 1e-9)))
    size_px = max(MIN_CAPTURE_SIZE_PX, min(max_size_px, size_px))

    async with httpx.AsyncClient(timeout=45.0) as client:
        image_bytes = await _fetch_static_map(
            client, latitude, longitude, chosen_zoom, size_px, maps_key, scale=2
        )

    world_px = _mercator_world_px(chosen_zoom)
    center_x = _lng_to_px(longitude, world_px)
    center_y = _lat_to_px(latitude, world_px)
    half = size_px / 2.0
    actual_span_m = _meters_per_pixel(latitude, chosen_zoom) * size_px

    return Capture(
        center_lat=latitude,
        center_lng=longitude,
        zoom=chosen_zoom,
        span_m=actual_span_m,
        size_px=size_px,
        south=_px_to_lat(center_y + half, world_px),
        west=_px_to_lng(center_x - half, world_px),
        north=_px_to_lat(center_y - half, world_px),
        east=_px_to_lng(center_x + half, world_px),
        image_bytes=image_bytes,
        tiles=1,
        tile_span_m=actual_span_m,
    )


async def _capture_mosaic(
    latitude: float,
    longitude: float,
    span_m: float,
    tile_span_m: float,
    zoom: int | None,
    maps_key: str,
) -> Capture:
    """Assemble une grille de tuiles Static Maps, chacune couvrant `tile_span_m`."""
    max_size_px = _max_capture_size_px()
    if zoom is not None:
        tile_zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        meters_per_px = _meters_per_pixel(latitude, tile_zoom)
        tile_size = int(round(tile_span_m / max(meters_per_px, 1e-9)))
        tile_size = max(MIN_CAPTURE_SIZE_PX, min(max_size_px, tile_size))
    else:
        tile_zoom, tile_size = _capture_geometry(latitude, tile_span_m, max_size_px)

    max_tiles = max(1, min(MAX_MOSAIC_TILES, int(settings.gemini_capture_max_tiles or MAX_MOSAIC_TILES)))
    tiles_per_side = min(max_tiles, max(1, int(math.ceil(span_m / tile_span_m))))
    total_size = tiles_per_side * tile_size

    world_px = _mercator_world_px(tile_zoom)
    center_x = _lng_to_px(longitude, world_px)
    center_y = _lat_to_px(latitude, world_px)
    offsets = [
        (index - (tiles_per_side - 1) / 2.0) * tile_size
        for index in range(tiles_per_side)
    ]
    centers: list[tuple[float, float]] = []
    for dy in offsets:  # lignes du haut vers le bas
        for dx in offsets:  # colonnes de gauche à droite
            centers.append(
                (
                    _px_to_lat(center_y + dy, world_px),
                    _px_to_lng(center_x + dx, world_px),
                )
            )

    semaphore = asyncio.Semaphore(5)

    async with httpx.AsyncClient(timeout=45.0) as client:

        async def fetch(tile_lat: float, tile_lng: float) -> bytes:
            async with semaphore:
                return await _fetch_static_map(
                    client, tile_lat, tile_lng, tile_zoom, tile_size, maps_key, scale=1
                )

        images = await asyncio.gather(*(fetch(lat, lng) for lat, lng in centers))

    try:
        tile_images = [Image.open(BytesIO(data)).convert("RGB") for data in images]
    except Exception as exc:
        raise GeminiUnavailable(f"Assemblage de la capture satellite impossible : {exc}") from exc
    expected = (tile_size, tile_size)
    if any(image.size != expected for image in tile_images):
        sizes = ", ".join(f"{image.size[0]}x{image.size[1]}" for image in tile_images[:4])
        raise GeminiUnavailable(
            f"Tuile satellite de taille inattendue ({sizes}) ; attendu {tile_size}x{tile_size}."
        )

    mosaic = Image.new("RGB", (total_size, total_size))
    for index, image in enumerate(tile_images):
        row, column = divmod(index, tiles_per_side)
        mosaic.paste(image, (column * tile_size, row * tile_size))
    buffer = BytesIO()
    mosaic.save(buffer, format="PNG")

    half = total_size / 2.0
    actual_span_m = _meters_per_pixel(latitude, tile_zoom) * total_size
    return Capture(
        center_lat=latitude,
        center_lng=longitude,
        zoom=tile_zoom,
        span_m=actual_span_m,
        size_px=total_size,
        south=_px_to_lat(center_y + half, world_px),
        west=_px_to_lng(center_x - half, world_px),
        north=_px_to_lat(center_y - half, world_px),
        east=_px_to_lng(center_x + half, world_px),
        image_bytes=buffer.getvalue(),
        tiles=tiles_per_side,
        tile_span_m=tile_span_m,
    )


async def _capture_screen_viewport(
    latitude: float,
    longitude: float,
    zoom: int,
    width_px: int,
    height_px: int,
    maps_key: str,
    bounds: dict[str, float] | None = None,
) -> Capture:
    """Capture satellite haute résolution correspondant exactement à l'écran visible."""
    chosen_zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
    world_px = _mercator_world_px(chosen_zoom)
    center_x = _lng_to_px(longitude, world_px)
    center_y = _lat_to_px(latitude, world_px)

    # Bornes géographiques de l'écran. `bounds` vient de la carte : c'est la
    # seule source exacte, car `zoom` est arrondi à l'entier et ne représente donc
    # pas forcément la vue affichée (Google Maps accepte un zoom fractionnaire).
    if bounds and _valid_bounds(bounds):
        west = float(bounds["west"])
        east = float(bounds["east"])
        north = float(bounds["north"])
        south = float(bounds["south"])
    else:
        west = _px_to_lng(center_x - width_px / 2.0, world_px)
        east = _px_to_lng(center_x + width_px / 2.0, world_px)
        north = _px_to_lat(center_y - height_px / 2.0, world_px)
        south = _px_to_lat(center_y + height_px / 2.0, world_px)

    # Étendue de la vue, exprimée en pixels du monde à `chosen_zoom`. C'est cette
    # grandeur — et non `width_px` — qui décrit le terrain réellement couvert.
    x_min = _lng_to_px(west, world_px)
    x_max = _lng_to_px(east, world_px)
    y_min = _lat_to_px(north, world_px)
    y_max = _lat_to_px(south, world_px)
    cover_w = max(1.0, x_max - x_min)
    cover_h = max(1.0, y_max - y_min)

    cols, rows, cell_w, cell_h, out_w, out_h = _viewport_tiling(cover_w, cover_h)
    mosaic = Image.new("RGB", (out_w, out_h))
    tiles_count = cols * rows

    mpp = _meters_per_pixel(latitude, chosen_zoom)
    span_m = max(cover_w, cover_h) * mpp

    semaphore = asyncio.Semaphore(5)

    async def fetch_tile(r: int, c: int) -> tuple[int, int, Image.Image]:
        """Une cellule : `size` couvre exactement son terrain, `scale` le suréchantillonne."""
        tcx = x_min + (c + 0.5) * cell_w
        tcy = y_min + (r + 0.5) * cell_h
        req_w = max(1, min(STATIC_MAPS_MAX_SIZE, int(math.ceil(cell_w))))
        req_h = max(1, min(STATIC_MAPS_MAX_SIZE, int(math.ceil(cell_h))))
        params = {
            "center": f"{_px_to_lat(tcy, world_px):.7f},{_px_to_lng(tcx, world_px):.7f}",
            "zoom": chosen_zoom,
            "size": f"{req_w}x{req_h}",
            "scale": SCREEN_CAPTURE_SCALE,
            "maptype": "satellite",
            "format": "png",
            "language": "fr",
            "key": maps_key,
        }
        tile = None
        for attempt in range(3):
            try:
                async with semaphore:
                    tile = await client.get(STATIC_MAPS_URL, params=params)
                if tile.status_code < 500:
                    break
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt == 2:
                    raise
                await asyncio.sleep(1.0)
        if tile is None or tile.status_code >= 400 or not tile.content:
            raise GeminiUnavailable(
                f"Capture satellite refusée : {_static_maps_error(tile) if tile else 'injoignable'}"
            )
        try:
            image = Image.open(BytesIO(tile.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise GeminiUnavailable(f"Tuile satellite illisible : {exc}") from exc
        # La cellule occupe exactement cette part de la mosaïque ; l'arrondi au
        # pixel supérieur de `size` laisse au plus un pixel d'écart, que ce
        # redimensionnement absorbe.
        left = int(round(c * out_w / cols))
        top = int(round(r * out_h / rows))
        width = int(round((c + 1) * out_w / cols)) - left
        height = int(round((r + 1) * out_h / rows)) - top
        return left, top, image.resize((max(1, width), max(1, height)), Image.LANCZOS)

    async with httpx.AsyncClient(timeout=45.0) as client:
        tile_results = await asyncio.gather(
            *(fetch_tile(r, c) for r in range(rows) for c in range(cols))
        )
    for left, top, image in tile_results:
        mosaic.paste(image, (left, top))

    buffer = BytesIO()
    mosaic.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return Capture(
        center_lat=latitude,
        center_lng=longitude,
        zoom=chosen_zoom,
        span_m=span_m,
        size_px=max(out_w, out_h),
        south=south,
        west=west,
        north=north,
        east=east,
        image_bytes=image_bytes,
        tiles=tiles_count,
        tile_span_m=mpp * cell_w,
        width_px=out_w,
        height_px=out_h,
    )


def _split_windows(capture: Capture) -> list[Capture]:
    """Découpe la capture en fenêtres chevauchantes, chacune géoréférencée."""
    mpp = capture.span_m / capture.size_px
    window_px = int(round(WINDOW_SPAN_M / mpp))
    if window_px >= capture.size_px:
        return [capture]
    stride_px = max(1, int(round(WINDOW_STRIDE_M / mpp)))
    per_side = int(math.ceil((capture.size_px - window_px) / stride_px)) + 1
    starts = [
        int(round(index * (capture.size_px - window_px) / (per_side - 1)))
        for index in range(per_side)
    ]

    try:
        full = Image.open(BytesIO(capture.image_bytes)).convert("RGB")
    except Exception as exc:
        raise GeminiUnavailable(f"Découpage de la capture impossible : {exc}") from exc

    world_px = _mercator_world_px(capture.zoom)
    center_x = _lng_to_px(capture.center_lng, world_px)
    center_y = _lat_to_px(capture.center_lat, world_px)
    out_px = max(window_px, WINDOW_UPSCALE_PX)
    windows: list[Capture] = []
    for top in starts:
        for left in starts:
            crop = full.crop((left, top, left + window_px, top + window_px))
            if out_px != window_px:
                crop = crop.resize((out_px, out_px), Image.LANCZOS)
            buffer = BytesIO()
            crop.save(buffer, format="PNG")
            offset_x = left + window_px / 2.0 - capture.size_px / 2.0
            offset_y = top + window_px / 2.0 - capture.size_px / 2.0
            wx, wy = center_x + offset_x, center_y + offset_y
            half = window_px / 2.0
            windows.append(
                Capture(
                    center_lat=_px_to_lat(wy, world_px),
                    center_lng=_px_to_lng(wx, world_px),
                    zoom=capture.zoom,
                    span_m=mpp * window_px,
                    size_px=window_px,
                    south=_px_to_lat(wy + half, world_px),
                    west=_px_to_lng(wx - half, world_px),
                    north=_px_to_lat(wy - half, world_px),
                    east=_px_to_lng(wx + half, world_px),
                    image_bytes=buffer.getvalue(),
                )
            )
    return windows


def _static_maps_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or f"HTTP {response.status_code}").strip()[:300]
    message = payload.get("error_message") or payload.get("error") or payload.get("status")
    return str(message or f"HTTP {response.status_code}")[:300]


# ── Appel Gemini ──

_SYSTEM_PROMPT = """Tu es un expert en télédétection et cartographie agricole de haute précision.
L'image est une capture satellite directe de la zone de terrain visible à l'écran de l'utilisateur (environ {span:.0f} mètres de large).

Objectif impératif :
Détecte et trace les limites de TOUTES les parcelles agricoles visibles sur l'image, sans en omettre aucune :
- Champs cultivés, rizières, terrasses, parcelles labourées, en jachère ou en sol nu, vergers, vignes, serres et prairies agricoles, y compris les petites parcelles.
- Détecte TOUTES les parcelles individuelles visibles.
- INCLUS également les parcelles visibles qui bordent ou coupent les bords de l'image (délimite fidèlement leur surface visible jusqu'aux limites de l'image).
- Ignore les routes goudronnées principales, les cours d'eau majeurs, les habitations/bâtiments isolés et les mentions/logos de la carte.

Règles de tracé des limites :
- Trace chaque limite au plus près du contour réel visible (bordures physiques, chemins de terre, haies, canaux d'irrigation, fossés, contrastes de cultures ou de texture de terre).
- Conserve fidèlement la forme réelle de la parcelle, sans lisser ni arrondir artificiellement les angles.
- Utilise autant de sommets que nécessaire (de 4 à 80 sommets par parcelle) pour épouser précisément les contours.
- Sépare distinctement deux parcelles voisines mitoyennes : chacune doit former son propre polygone fermé indépendant. Ne les fusionne jamais.
- Chaque contour est un polygone fermé simple avec au moins 3 sommets ordonnés le long de la limite (sans auto-intersection). Ne répète pas le premier point à la fin.

Coordonnées :
- x et y sont normalisés entre 0.0 et 1.0 dans l'image.
- x = 0.0 correspond au bord gauche de l'écran, x = 1.0 au bord droit.
- y = 0.0 correspond au bord supérieur de l'écran, y = 1.0 au bord inférieur.

Réponds strictement avec l'objet JSON demandé par le schéma. Si aucun champ n'est visible, retourne {{"parcels": []}}."""

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "parcels": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                    "ring": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "x": {"type": "NUMBER"},
                                "y": {"type": "NUMBER"},
                            },
                            "required": ["x", "y"],
                        },
                    },
                },
                "required": ["ring"],
            },
        }
    },
    "required": ["parcels"],
}


# ── Appel Gemini : modèle image (limites dessinées en rouge) ──

_GEMINI_IMAGE_PROMPT = """Cette image est une prise de vue satellite d'une zone agricole.
Dessine en ROUGE vif, par-dessus la photo, le contour de CHAQUE parcelle agricole visible.

- Suis les limites réelles : chemins de terre, diguettes, haies, fossés, canaux d'irrigation, bordures de cultures ou changements de texture de terre.
- Trace aussi une ligne rouge ENTRE deux parcelles voisines mitoyennes, même si elles se ressemblent, de sorte que chaque parcelle forme une zone fermée séparée des autres.
- Délimite aussi les parcelles qui touchent le bord de l'image.
- N'oublie aucune parcelle, y compris les petites.
- Ignore les routes goudronnées principales, les habitations et les mentions portées sur la carte.

Ne modifie RIEN d'autre : conserve exactement le cadrage, l'orientation, la résolution et les couleurs de la photo d'origine. Les traits rouges sont la SEULE différence. N'ajoute aucun texte, aucune étiquette, aucun contour colorié autre qu'en rouge.
Renvoie uniquement l'image ainsi annotée."""


def _gemini_image_mode() -> bool:
    """Vrai quand la détection passe par un modèle image (limites dessinées)."""
    return (settings.gemini_parcels_mode or "").strip().lower() in {"image", "dessin", "red"}


async def _post_gemini(
    url: str, payload: dict[str, Any], api_key: str, sujet: str
) -> httpx.Response:
    """POST Gemini avec trois tentatives sur les erreurs réseau et les 5xx."""
    response: httpx.Response | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=float(settings.gemini_timeout_s or 150.0)) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                )
            if response.status_code < 500:
                break
        except httpx.TimeoutException as exc:
            if attempt == 2:
                raise GeminiUnavailable(
                    f"{sujet} n'a pas répondu dans les {int(settings.gemini_timeout_s or 150)} s."
                ) from exc
            await asyncio.sleep(1.0)
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            if attempt == 2:
                raise GeminiUnavailable(f"{sujet} injoignable (échec réseau / DNS) : {exc}") from exc
            await asyncio.sleep(1.0)
        except httpx.HTTPError as exc:
            raise GeminiUnavailable(f"{sujet} injoignable : {exc}") from exc

    if response is None:
        raise GeminiUnavailable(f"{sujet} n'a renvoyé aucune réponse.")
    return response


async def _call_gemini_image(image_bytes: bytes, api_key: str) -> bytes:
    """Demande au modèle image de DESSINER les limites en rouge sur la capture."""
    model = (settings.gemini_image_model or "gemini-nano-banana-2.1").strip()
    url = GEMINI_ENDPOINT.format(model=model)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _GEMINI_IMAGE_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }

    response = await _post_gemini(url, payload, api_key, "Le modèle image Gemini")
    if response.status_code >= 400:
        message = _gemini_error(response)
        if response.status_code in (401, 403) or "API_KEY" in message.upper():
            raise GeminiConfigurationError(
                "Gemini a refusé la clé API pour le modèle image "
                f"({model}). Renseignez GEMINI_API_KEY dans BACKEND_AGRISAT/.env : {message}"
            )
        raise GeminiUnavailable(
            f"Le modèle image Gemini a renvoyé une erreur {response.status_code} : {message}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise GeminiUnavailable("Le modèle image Gemini a renvoyé une réponse illisible.") from exc

    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            try:
                return base64.b64decode(inline["data"])
            except (ValueError, TypeError) as exc:
                raise GeminiUnavailable("Image Gemini illisible (base64 invalide).") from exc

    finish = (data.get("candidates") or [{}])[0].get("finishReason")
    raise GeminiParcelsError(
        "Le modèle image Gemini n'a renvoyé aucune image"
        + (f" (fin : {finish})." if finish else ".")
    )


async def _window_rings(window: Capture, gemini_key: str) -> list[dict[str, Any]]:
    """Contours bruts d'une fenêtre, au format `{"ring": [...]}`, selon le mode.

    Mode « text » (défaut) : Gemini renvoie les contours en coordonnées JSON.
    Mode « image » : Gemini dessine les limites en rouge sur la photo, que
    `parcel_rings_from_red_lines` reconvertit en anneaux normalisés. Le reste de
    la chaîne (validation, géoréférencement, déduplication) est identique.
    """
    if not _gemini_image_mode():
        result = await _call_gemini(window.image_bytes, window.span_m, gemini_key)
        items = result.get("parcels")
        return items if isinstance(items, list) else []

    generated = await _call_gemini_image(window.image_bytes, gemini_key)
    warning = framing_warning(window.image_bytes, generated)
    if warning:
        logger.warning("gemini-parcels (mode image) : %s", warning)
    try:
        rings = parcel_rings_from_red_lines(generated)
    except ValueError as exc:
        logger.warning("gemini-parcels (mode image) : %s", exc)
        return []
    return [{"ring": ring} for ring in rings]


async def _call_gemini(image_bytes: bytes, span_m: float, api_key: str) -> dict[str, Any]:
    model = (settings.gemini_model or "gemini-3.8-flash").strip()
    url = GEMINI_ENDPOINT.format(model=model)
    payload = {
        "systemInstruction": {
            "parts": [{"text": _SYSTEM_PROMPT.format(span=span_m)}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Détecte toutes les parcelles agricoles de cette image satellite."},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 16384,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }

    response = await _post_gemini(url, payload, api_key, "Gemini")

    if response.status_code >= 400:
        message = _gemini_error(response)
        if response.status_code in (401, 403) or "API_KEY" in message.upper():
            raise GeminiConfigurationError(
                "Gemini a refusé la clé API. Renseignez une clé Gemini valide dans "
                f"GEMINI_API_KEY (BACKEND_AGRISAT/.env) : {message}"
            )
        raise GeminiUnavailable(f"Gemini a renvoyé une erreur {response.status_code} : {message}")

    try:
        data = response.json()
    except ValueError as exc:
        raise GeminiUnavailable("Gemini a renvoyé une réponse illisible.") from exc

    text = _gemini_text(data)
    return _parse_json(text)


def _gemini_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or f"HTTP {response.status_code}").strip()[:400]
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("status")
        details = [
            str(item.get("reason"))
            for item in (error.get("details") or [])
            if isinstance(item, dict) and item.get("reason")
        ]
        if details:
            message = f"{message} ({', '.join(details)})"
        return str(message or f"HTTP {response.status_code}")[:400]
    return str(error or f"HTTP {response.status_code}")[:400]


def _gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = data.get("promptFeedback") or {}
        reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        raise GeminiParcelsError(
            "Gemini n'a renvoyé aucun résultat"
            + (f" (blocage : {reason})." if reason else ".")
        )
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ).strip()
    if not text:
        finish = (candidates[0] or {}).get("finishReason")
        raise GeminiParcelsError(
            "Gemini n'a renvoyé aucun contour"
            + (f" (fin : {finish})." if finish else ".")
        )
    return text


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        payload = json.loads(cleaned)
    except ValueError as exc:
        raise GeminiParcelsError("Gemini a renvoyé un JSON invalide.") from exc
    if not isinstance(payload, dict):
        raise GeminiParcelsError("Gemini a renvoyé une structure inattendue.")
    return payload


# ── Validation des contours ──


def _clean_ring(raw: Any) -> list[tuple[float, float]] | None:
    if not isinstance(raw, list):
        return None
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        # Un sommet hors image rend le contour inexploitable : on l'écarte.
        if not (-0.02 <= x <= 1.02 and -0.02 <= y <= 1.02):
            continue
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        if points and abs(points[-1][0] - x) < 1e-7 and abs(points[-1][1] - y) < 1e-7:
            continue
        points.append((x, y))
    if len(points) >= 2 and abs(points[0][0] - points[-1][0]) < 1e-7 and abs(points[0][1] - points[-1][1]) < 1e-7:
        points.pop()
    distinct = {(round(x, 6), round(y, 6)) for x, y in points}
    if len(points) < 3 or len(distinct) < 3:
        return None
    return points


def _largest_polygon(geom: Polygon) -> Polygon | None:
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    parts = [part for part in getattr(geom, "geoms", []) if part.geom_type == "Polygon" and not part.is_empty]
    return max(parts, key=lambda part: part.area) if parts else None


def _candidate_from_ring(
    raw_ring: Any,
    capture: Capture,
    confidence: Any = None,
    label: Any = None,
    frame: Capture | None = None,
    edge_margin: float = 0.0,
) -> Candidate | None:
    ring = _clean_ring(raw_ring)
    if ring is None:
        return None
    if edge_margin and any(
        x < edge_margin or x > 1 - edge_margin or y < edge_margin or y > 1 - edge_margin
        for x, y in ring
    ):
        # Parcelle coupée par le bord de la fenêtre : la fenêtre voisine la voit entière.
        return None

    origin = frame or capture
    metric: list[tuple[float, float]] = []
    for x, y in ring:
        lat, lng = _normalized_to_latlng(x, y, capture)
        metric.append(_latlng_to_metric(lat, lng, origin.center_lat, origin.center_lng))

    geom = _largest_polygon(Polygon(metric))
    if geom is None:
        return None
    area_m2 = geom.area
    if area_m2 < MIN_AREA_HA * 10_000:
        return None
    # Un contour qui couvre presque toute l'image est une erreur de segmentation.
    img_w_px = capture.width_px or capture.size_px
    img_h_px = capture.height_px or capture.size_px
    mpp = _meters_per_pixel(origin.center_lat, capture.zoom)
    max_area_m2 = (img_w_px * mpp) * (img_h_px * mpp) * 0.98
    if area_m2 > max_area_m2:
        return None

    ring_metric = list(geom.exterior.coords)[:-1]
    if len(ring_metric) < 3:
        return None
    coordinates: list[dict[str, float]] = []
    for mx, my in ring_metric:
        lat, lng = _metric_to_latlng(mx, my, origin.center_lat, origin.center_lng)
        if not (math.isfinite(lat) and math.isfinite(lng)):
            return None
        coordinates.append({"lat": lat, "lng": lng})

    centroid = geom.centroid
    center_lat, center_lng = _metric_to_latlng(
        centroid.x, centroid.y, origin.center_lat, origin.center_lng
    )
    trust = confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None
    return Candidate(
        geom=geom,
        coordinates=coordinates,
        center_lat=center_lat,
        center_lng=center_lng,
        area_ha=area_m2 / 10_000.0,
        confidence=min(1.0, max(0.0, float(trust))) if trust is not None and math.isfinite(trust) else None,
        label=label if isinstance(label, str) and label.strip() else None,
    )


def _iou(a: Polygon, b: Polygon) -> float:
    if a.disjoint(b):
        return 0.0
    intersection = a.intersection(b).area
    if intersection <= 0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def _same_candidate(a: Candidate, b: Candidate) -> bool:
    """Deux contours de la même réponse Gemini qui désignent le même champ."""
    if _iou(a.geom, b.geom) >= CANDIDATE_IOU_THRESHOLD:
        return True
    distance = math.hypot(a.geom.centroid.x - b.geom.centroid.x, a.geom.centroid.y - b.geom.centroid.y)
    ratio = min(a.geom.area, b.geom.area) / max(a.geom.area, b.geom.area)
    return distance <= 5.0 and ratio >= 0.6


# ── Rapprochement avec les parcelles déjà en base ──


def _coordinates_to_metric(
    coordinates: Any,
    center_lat: float,
    center_lng: float,
) -> Polygon | None:
    if not isinstance(coordinates, list) or len(coordinates) < 3:
        return None
    points: list[tuple[float, float]] = []
    for point in coordinates:
        if not isinstance(point, dict):
            return None
        try:
            lat = float(point["lat"])
            lng = float(point["lng"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(lat) and math.isfinite(lng)):
            continue
        metric = _latlng_to_metric(lat, lng, center_lat, center_lng)
        if points and abs(points[-1][0] - metric[0]) < 1e-6 and abs(points[-1][1] - metric[1]) < 1e-6:
            continue
        points.append(metric)
    if len(points) < 3:
        return None
    return _largest_polygon(Polygon(points))


def _existing_parcels(rows: list[Parcelle], capture: Capture) -> list[ExistingParcel]:
    existing: list[ExistingParcel] = []
    for row in rows:
        if row.data_source == REJECTED_PARCEL_SOURCE:
            # Marqueur de suppression d'un contour automatique : il ne doit pas
            # empêcher la création d'une délimitation explicite au même endroit.
            continue
        geom = _coordinates_to_metric(row.coordinates, capture.center_lat, capture.center_lng)
        if geom is None:
            continue
        existing.append(
            ExistingParcel(
                id=str(row.id),
                label=row.label,
                surface_ha=row.surface_ha,
                geom=geom,
            )
        )
    return existing


def _match_existing(candidate: Candidate, existing: list[ExistingParcel], tree: STRtree | None) -> ExistingParcel | None:
    if not existing:
        return None
    indexes = (
        [int(index) for index in tree.query(candidate.geom.buffer(30.0))]
        if tree is not None
        else range(len(existing))
    )
    for index in indexes:
        item = existing[index]
        if _iou(candidate.geom, item.geom) >= EXISTING_IOU_THRESHOLD:
            return item
        distance = math.hypot(
            candidate.geom.centroid.x - item.geom.centroid.x,
            candidate.geom.centroid.y - item.geom.centroid.y,
        )
        ratio = min(candidate.geom.area, item.geom.area) / max(candidate.geom.area, item.geom.area)
        tolerance = max(8.0, min(25.0, math.sqrt(max(candidate.geom.area, item.geom.area)) * 0.25))
        if ratio >= EXISTING_AREA_RATIO and distance <= tolerance:
            return item
    return None


# ── Enregistrement ──


async def _next_label_index(session) -> int:
    result = await session.execute(select(Parcelle.label))
    highest = 0
    for (label,) in result.all():
        match = _LABEL_PATTERN.match((label or "").strip())
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def _capture_notes(capture: Capture, candidate: Candidate) -> str:
    notes = (
        f"Détection Gemini — capture de {capture.span_m:.0f} m (zoom {capture.zoom}), "
        f"centre {capture.center_lat:.6f}, {capture.center_lng:.6f}."
    )
    if capture.tiles > 1 and capture.tile_span_m:
        notes += (
            f" Mosaïque {capture.tiles}×{capture.tiles} de tuiles "
            f"de {capture.tile_span_m:.0f} m."
        )
    if candidate.confidence is not None:
        notes += f" Confiance visuelle {candidate.confidence:.2f}."
    if candidate.label:
        notes += f" Libellé Gemini : {candidate.label[:120]}."
    return notes


async def _persist(candidates: list[Candidate], capture: Capture) -> list[dict[str, Any]]:
    async with async_session_maker() as session:
        start_index = await _next_label_index(session)
        rows: list[Parcelle] = []
        for offset, candidate in enumerate(candidates):
            rows.append(
                Parcelle(
                    id=uuid.uuid4(),
                    label=f"{LABEL_PREFIX} {start_index + offset:03d}",
                    coordinates=candidate.coordinates,
                    center_lat=candidate.center_lat,
                    center_lng=candidate.center_lng,
                    surface_ha=candidate.area_ha,
                    # Tous les champs de culture / analyse restent NULL : la parcelle
                    # est une délimitation, elle sera analysée comme un import KML.
                    culture_declared=None,
                    culture_detected=None,
                    ndvi_percentage=None,
                    ndre=None,
                    spectral_bands=None,
                    confidence=None,
                    verdict="Contour détecté par Gemini — à analyser",
                    details=(
                        "Délimitation proposée par Gemini à partir d'une image satellite. "
                        "Aucune culture détectée : lancez l'analyse de la parcelle."
                    ),
                    saison=None,
                    soil_type=None,
                    risk_factors=None,
                    recommendations=None,
                    data_source=PARCEL_SOURCE,
                    owner_name=None,
                    notes=_capture_notes(capture, candidate),
                )
            )
        session.add_all(rows)
        await session.commit()
        for row in rows:
            await session.refresh(row)
        serialized = [serialize_parcelle(row) for row in rows]
    cache.invalidate()
    return serialized


def _preview_row(candidate: Candidate) -> dict[str, Any]:
    """Contour non enregistré (`persist=False`) au même format que l'API."""
    return {
        "id": f"gemini-preview-{uuid.uuid4()}",
        "label": candidate.label or "Parcelle Gemini (aperçu)",
        "coordinates": candidate.coordinates,
        "center_lat": candidate.center_lat,
        "center_lng": candidate.center_lng,
        "surface_ha": candidate.area_ha,
        "culture_detected": None,
        "confidence": None,
        "data_source": PARCEL_SOURCE,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Point d'entrée ──


async def detect_parcels_gemini(
    latitude: float,
    longitude: float,
    span_m: float = 100.0,
    tile_span_m: float | None = None,
    zoom: int | None = None,
    persist: bool = True,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
    bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Capture satellite → détection Gemini → validation → déduplication → base."""
    gemini_key = (settings.gemini_api_key or "").strip()
    maps_key = (settings.google_maps_api_key or "").strip()
    if not gemini_key and not maps_key:
        raise GeminiConfigurationError(
            "Aucune clé configurée : renseignez GEMINI_API_KEY (et GOOGLE_MAPS_API_KEY "
            "pour la capture satellite) dans BACKEND_AGRISAT/.env."
        )
    # Une clé Maps n'a pas accès à l'API Generative Language : GEMINI_API_KEY est
    # la seule voie fiable, la clé Maps n'est qu'un repli explicite.
    if not gemini_key:
        gemini_key = maps_key
    if not maps_key:
        raise GeminiConfigurationError(
            "GOOGLE_MAPS_API_KEY est absente : elle est nécessaire pour capturer l'image satellite."
        )

    is_screen_capture = bool(viewport_width and viewport_height)

    if is_screen_capture:
        # `round` et non `int` : la carte accepte un zoom fractionnaire, et
        # tronquer 16.6 en 16 éloigne la capture des tuiles réellement affichées.
        actual_zoom = max(MIN_ZOOM, min(MAX_ZOOM, int(round(zoom if zoom is not None else 18))))
        capture = await _capture_screen_viewport(
            latitude=latitude,
            longitude=longitude,
            zoom=actual_zoom,
            width_px=int(viewport_width),  # type: ignore[arg-type]
            height_px=int(viewport_height),  # type: ignore[arg-type]
            maps_key=maps_key,
            bounds=bounds,
        )
        raw_parcels = await _window_rings(capture, gemini_key)
        raw_windows = [capture] * len(raw_parcels)
        windows = [capture]
        failed_windows = []
    else:
        requested_span = max(MIN_CAPTURE_SPAN_M, min(MAX_CAPTURE_SPAN_M, float(span_m)))
        requested_tile = (
            float(tile_span_m)
            if tile_span_m is not None
            else float(settings.gemini_capture_tile_span_m or requested_span)
        )
        requested_tile = max(MIN_CAPTURE_SPAN_M, min(requested_span, requested_tile))

        if requested_tile < requested_span:
            capture = await _capture_mosaic(
                latitude, longitude, requested_span, requested_tile, zoom, maps_key
            )
        else:
            capture = await _capture_satellite(latitude, longitude, requested_span, zoom, maps_key)

        windows = _split_windows(capture)
        semaphore = asyncio.Semaphore(MAX_PARALLEL_GEMINI_CALLS)

        async def analyze(window: Capture) -> list[dict[str, Any]]:
            async with semaphore:
                return await _window_rings(window, gemini_key)

        results = await asyncio.gather(*(analyze(w) for w in windows), return_exceptions=True)
        failed_windows = [r for r in results if isinstance(r, Exception)]
        if len(failed_windows) == len(results):
            raise failed_windows[0]

        raw_parcels = []
        raw_windows = []
        for window, result in zip(windows, results):
            if isinstance(result, Exception):
                logger.warning("gemini-parcels : fenêtre en échec : %s", result)
                continue
            raw_parcels.extend(result)
            raw_windows.extend([window] * len(result))

    rejected = 0
    candidates: list[Candidate] = []
    split = len(windows) > 1
    for item, window in list(zip(raw_parcels, raw_windows))[: MAX_DETECTED_PARCELS * 2]:
        if not isinstance(item, dict):
            rejected += 1
            continue
        candidate = _candidate_from_ring(
            item.get("ring"),
            window,
            confidence=item.get("confidence"),
            label=item.get("label"),
            frame=capture,
            edge_margin=WINDOW_EDGE_MARGIN if split else 0.0,
        )
        if candidate is None:
            rejected += 1
            continue
        candidates.append(candidate)
    if len(raw_parcels) > MAX_DETECTED_PARCELS * 2:
        rejected += len(raw_parcels) - MAX_DETECTED_PARCELS * 2

    unique: list[Candidate] = []
    for candidate in candidates:
        if any(_same_candidate(candidate, kept) for kept in unique):
            rejected += 1
            continue
        unique.append(candidate)

    rows = await load_parcels_in_area(latitude, longitude, max(capture.span_m * 1.2, 120.0))
    existing = _existing_parcels(rows, capture)
    tree = STRtree([item.geom for item in existing]) if existing else None

    fresh: list[Candidate] = []
    duplicates: list[dict[str, Any]] = []
    for candidate in unique:
        match = _match_existing(candidate, existing, tree)
        if match is None:
            fresh.append(candidate)
        else:
            duplicates.append(
                {
                    "id": match.id,
                    "label": match.label,
                    "surface_ha": match.surface_ha,
                }
            )

    fresh = fresh[:MAX_DETECTED_PARCELS]
    warnings: list[str] = []
    if not raw_parcels:
        warnings.append("Gemini n'a détecté aucune parcelle sur cette image.")
    if rejected:
        warnings.append(
            f"{rejected} contour(s) ignoré(s) : forme invalide, trop petit ou doublon dans la réponse."
        )
    if failed_windows:
        warnings.append(
            f"{len(failed_windows)} zone(s) sur {len(windows)} n'ont pas pu être analysées par Gemini : "
            "relancez la capture pour compléter le tracé."
        )
    if duplicates:
        warnings.append(
            f"{len(duplicates)} parcelle(s) déjà présente(s) en base : aucune ligne en double n'a été créée."
        )
    if not settings.gemini_api_key:
        warnings.append(
            "GEMINI_API_KEY absent : la clé Google Maps a été utilisée en repli. "
            "Gemini la refuse généralement (API_KEY_SERVICE_BLOCKED)."
        )
    if not is_screen_capture and abs(capture.span_m - requested_span) > requested_span * 0.35:
        warnings.append(
            f"La capture couvre {capture.span_m:.0f} m au lieu de {requested_span:.0f} m "
            f"(zoom {capture.zoom}, {capture.tiles} tuile(s) par côté)."
        )

    if persist and fresh:
        parcels = await _persist(fresh, capture)
    else:
        parcels = [_preview_row(candidate) for candidate in fresh]

    logger.info(
        "gemini-parcels : %d détectée(s), %d nouvelle(s), %d doublon(s), %d rejetée(s) "
        "(zoom %d, %.0f m, %d tuile(s) de %.0f m)",
        len(raw_parcels),
        len(parcels),
        len(duplicates),
        rejected,
        capture.zoom,
        capture.span_m,
        capture.tiles,
        capture.tile_span_m or capture.span_m,
    )

    return {
        "capture": {
            "center": {"lat": latitude, "lng": longitude},
            "zoom": capture.zoom,
            "span_m": round(capture.span_m, 2),
            "size_px": capture.size_px,
            "tiles": capture.tiles,
            "tile_span_m": round(capture.tile_span_m or capture.span_m, 2),
            "bounds": {
                "south": capture.south,
                "west": capture.west,
                "north": capture.north,
                "east": capture.east,
            },
            "image_data_url": "data:image/png;base64," + base64.b64encode(capture.image_bytes).decode("ascii"),
        },
        "parcels": parcels,
        "existing": duplicates,
        "summary": {
            "detected": len(raw_parcels),
            "valid": len(unique),
            "created": len(parcels) if persist else 0,
            "duplicates": len(duplicates),
            "rejected": rejected,
        },
        "warnings": warnings,
    }
