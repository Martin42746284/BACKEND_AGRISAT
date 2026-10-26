"""
Importe les delimitations de parcelles contenues dans les fichiers KML.

Objectif : recuperer uniquement les contours deja dessines (Google Earth, etc.)
sans jamais importer de culture et sans lancer de tracage manuel ou automatique.
Les lignes creees portent `data_source = "Import KML"` et tous les champs de
culture restent NULL.

Le script est additif et idempotent :
  - il ne modifie ni ne supprime aucune parcelle existante ;
  - un contour KML deja present en base (IoU >= --existing-iou) est ignore ;
  - les marqueurs « Trace automatique supprime » ne bloquent pas un contour
    KML explicite : ils ne concernent que les traces automatiques ;
  - deux contours quasi identiques dans les KML (IoU >= --import-iou) sont
    fusionnes en gardant le premier dans l'ordre des fichiers ;
  - une seconde execution n'ajoute donc que les nouveaux contours.

Par defaut le script ne fait qu'un rapport (dry-run). Pour ecrire en base :

    python scripts/import_kml_parcels.py
    python scripts/import_kml_parcels.py --apply

Exemples :
    python scripts/import_kml_parcels.py --kml-dir kml --apply
    python scripts/import_kml_parcels.py --existing-iou 0.7
    python scripts/import_kml_parcels.py --json-report import_kml.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Permet d'executer le script depuis n'importe quel dossier, tout en chargeant
# le .env et les modules du backend (comme scripts/sync_orge_model.py).
BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

from shapely.geometry import MultiPolygon, Polygon  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import async_session_maker, engine  # noqa: E402
from app.models import Parcelle  # noqa: E402

KML = "{http://www.opengis.net/kml/2.2}"
DEFAULT_KML_DIR = BACKEND_ROOT / "kml"
DEFAULT_SOURCE = "Import KML"
DEFAULT_LABEL_PREFIX = "Parcelle KML"
REJECTED_SOURCE = "Tracé automatique supprimé"
DEFAULT_EXISTING_IOU = 0.60
DEFAULT_IMPORT_IOU = 0.80
M_PER_DEG_LAT = 110_574
M_PER_DEG_LNG_AT_EQUATOR = 111_320


@dataclass
class KmlParcel:
    """Contour KML normalise avant insertion."""

    file: str
    placemark_order: int
    part: int
    placemark_name: str
    geom: Polygon
    coordinates: list[dict[str, float]]
    surface_ha: float
    center_lat: float
    center_lng: float
    fingerprint: str

    @property
    def display_order(self) -> str:
        return f"{self.placemark_order:04d}" + (f".{self.part}" if self.part > 1 else "")


@dataclass
class ExistingParcel:
    """Parcelle deja en base, seulement les champs utiles au rapprochement."""

    id: uuid.UUID
    label: str | None
    source: str | None
    geom: Polygon


def _ring_area_ha(coords: list[tuple[float, float]]) -> float:
    """Superficie en ha, projection locale identique a ringAreaHa() du frontend."""
    if len(coords) < 3:
        return 0.0
    lat0, lng0 = coords[0][1], coords[0][0]
    kx = M_PER_DEG_LNG_AT_EQUATOR * math.cos(math.radians(lat0))
    ky = M_PER_DEG_LAT
    pts = [((lng - lng0) * kx, (lat - lat0) * ky) for lng, lat in coords]
    area = 0.0
    for index, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(index + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0 / 10_000.0


def _clean_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """Supprime les points consecutifs dupliques et ferme l'anneau."""
    cleaned: list[tuple[float, float]] = []
    for lng, lat in points:
        if not math.isfinite(lng) or not math.isfinite(lat):
            continue
        if cleaned and cleaned[-1] == (lng, lat):
            continue
        cleaned.append((lng, lat))
    if len(cleaned) < 3:
        return None
    if cleaned[0] != cleaned[-1]:
        cleaned.append(cleaned[0])
    if len(set(cleaned[:-1])) < 3:
        return None
    return cleaned


def _polygon_parts(geom: Polygon | MultiPolygon) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 0 else []
    parts = [part for part in geom.geoms if isinstance(part, Polygon) and part.area > 0]
    return parts


def _read_outer_ring(polygon: ET.Element) -> list[tuple[float, float]] | None:
    outer = polygon.find(f"{KML}outerBoundaryIs/{KML}LinearRing/{KML}coordinates")
    if outer is None or not (outer.text or "").strip():
        return None
    points: list[tuple[float, float]] = []
    for token in outer.text.split():
        values = token.split(",")
        if len(values) < 2:
            continue
        try:
            lng, lat = float(values[0]), float(values[1])
        except ValueError:
            continue
        points.append((lng, lat))
    return _clean_ring(points)


def _fingerprint(coords: list[tuple[float, float]]) -> str:
    normalized = [[round(lng, 7), round(lat, 7)] for lng, lat in coords]
    return hashlib.sha1(json.dumps(normalized, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_file(path: Path) -> tuple[list[KmlParcel], dict[str, int]]:
    """Retourne les contours valides du fichier et quelques compteurs de diagnostic."""
    stats = {"placemarks": 0, "polygons": 0, "invalid_repaired": 0, "skipped": 0, "holes": 0}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise SystemExit(f"Fichier KML illisible : {path} ({error})") from error

    parcels: list[KmlParcel] = []
    for order, placemark in enumerate(root.findall(f".//{KML}Placemark"), start=1):
        stats["placemarks"] += 1
        name = (placemark.findtext(f"{KML}name") or "").strip()
        polygons = placemark.findall(f".//{KML}Polygon")
        for part_index, polygon in enumerate(polygons, start=1):
            stats["polygons"] += 1
            if polygon.find(f"{KML}innerBoundaryIs") is not None:
                # Les trous ne sont pas representables dans le modele Parcelle actuel.
                stats["holes"] += 1
            ring = _read_outer_ring(polygon)
            if ring is None:
                stats["skipped"] += 1
                continue

            geom: Polygon | MultiPolygon = Polygon(ring)
            if not geom.is_valid:
                stats["invalid_repaired"] += 1
                geom = geom.buffer(0)
            parts = _polygon_parts(geom)
            if not parts:
                stats["skipped"] += 1
                continue

            for repaired_index, part in enumerate(parts, start=1):
                part_ring = _clean_ring([(x, y) for x, y, *_ in part.exterior.coords])
                if part_ring is None:
                    stats["skipped"] += 1
                    continue
                surface = _ring_area_ha(part_ring)
                if surface <= 0 or not math.isfinite(surface):
                    stats["skipped"] += 1
                    continue
                centroid = part.centroid
                if not (math.isfinite(centroid.x) and math.isfinite(centroid.y)):
                    stats["skipped"] += 1
                    continue
                parcels.append(
                    KmlParcel(
                        file=path.name,
                        placemark_order=order,
                        part=part_index if len(parts) == 1 else repaired_index,
                        placemark_name=name,
                        geom=part,
                        coordinates=[{"lat": lat, "lng": lng} for lng, lat in part_ring],
                        surface_ha=surface,
                        center_lat=float(centroid.y),
                        center_lng=float(centroid.x),
                        fingerprint=_fingerprint(part_ring),
                    )
                )
    return parcels, stats


def _load_kml(kml_dir: Path) -> tuple[list[KmlParcel], dict[str, dict[str, int]]]:
    files = sorted(kml_dir.glob("*.kml"))
    if not files:
        raise SystemExit(f"Aucun fichier .kml trouve dans {kml_dir}")

    parcels: list[KmlParcel] = []
    stats: dict[str, dict[str, int]] = {}
    for path in files:
        file_parcels, file_stats = _parse_file(path)
        parcels.extend(file_parcels)
        stats[path.name] = file_stats
    return parcels, stats


def _dedupe_import(parcels: list[KmlParcel], iou_threshold: float) -> tuple[list[KmlParcel], set[int]]:
    """Fusionne les contours quasi identiques a l'interieur des KML (garde le premier)."""
    if not parcels:
        return [], set()
    tree = STRtree([parcel.geom for parcel in parcels])
    dropped: set[int] = set()
    seen_fingerprints: set[str] = set()

    for index, parcel in enumerate(parcels):
        if index in dropped:
            continue
        if parcel.fingerprint in seen_fingerprints:
            dropped.add(index)
            continue
        seen_fingerprints.add(parcel.fingerprint)
        for other_index in tree.query(parcel.geom):
            other_index = int(other_index)
            if other_index <= index or other_index in dropped:
                continue
            other = parcels[other_index]
            if parcel.geom.disjoint(other.geom):
                continue
            intersection = parcel.geom.intersection(other.geom).area
            if intersection <= 0:
                continue
            union = parcel.geom.area + other.geom.area - intersection
            if union > 0 and intersection / union >= iou_threshold:
                dropped.add(other_index)

    return [parcel for index, parcel in enumerate(parcels) if index not in dropped], dropped


def _coords_to_geom(coordinates: object) -> Polygon | None:
    if not isinstance(coordinates, list) or len(coordinates) < 3:
        return None
    ring: list[tuple[float, float]] = []
    for point in coordinates:
        if not isinstance(point, dict):
            return None
        try:
            lng = float(point["lng"])
            lat = float(point["lat"])
        except (KeyError, TypeError, ValueError):
            return None
        if math.isfinite(lng) and math.isfinite(lat):
            ring.append((lng, lat))
    cleaned = _clean_ring(ring)
    if cleaned is None:
        return None
    geom = Polygon(cleaned)
    if not geom.is_valid:
        geom = geom.buffer(0)
    parts = _polygon_parts(geom)
    return max(parts, key=lambda part: part.area) if parts else None


async def _load_existing() -> list[ExistingParcel]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(Parcelle.id, Parcelle.label, Parcelle.data_source, Parcelle.coordinates)
        )
        existing: list[ExistingParcel] = []
        for row in result.all():
            if row.data_source == REJECTED_SOURCE:
                # Marqueur de suppression d'un contour automatique : il ne doit
                # pas empecher l'import d'une delimitation KML explicite.
                continue
            geom = _coords_to_geom(row.coordinates)
            if geom is None:
                continue
            existing.append(
                ExistingParcel(id=row.id, label=row.label, source=row.data_source, geom=geom)
            )
        return existing


def _filter_existing(
    parcels: list[KmlParcel], existing: list[ExistingParcel], iou_threshold: float
) -> tuple[list[KmlParcel], Counter[str]]:
    """Retire les KML deja presents (meme champ) et compte les sources concernees."""
    if not parcels or not existing:
        return parcels, Counter()
    tree = STRtree([item.geom for item in existing])
    kept: list[KmlParcel] = []
    skipped_sources: Counter[str] = Counter()

    for parcel in parcels:
        best: ExistingParcel | None = None
        best_iou = 0.0
        for index in tree.query(parcel.geom):
            item = existing[int(index)]
            if parcel.geom.disjoint(item.geom):
                continue
            intersection = parcel.geom.intersection(item.geom).area
            if intersection <= 0:
                continue
            union = parcel.geom.area + item.geom.area - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best = item
        if best is not None and best_iou >= iou_threshold:
            skipped_sources[best.source or "(sans source)"] += 1
        else:
            kept.append(parcel)
    return kept, skipped_sources


def _next_label_index(labels: list[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\s+(\d+)$", re.IGNORECASE)
    highest = 0
    for label in labels:
        match = pattern.match(label.strip())
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


async def _apply_import(
    parcels: list[KmlParcel], source: str, label_prefix: str, existing_iou: float
) -> tuple[int, int]:
    """Recharge la base dans la transaction d'ecriture, puis insere les contours."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(Parcelle.id, Parcelle.label, Parcelle.data_source, Parcelle.coordinates)
        )
        existing: list[ExistingParcel] = []
        existing_labels: list[str] = []
        for row in result.all():
            existing_labels.append(row.label or "")
            if row.data_source == REJECTED_SOURCE:
                continue
            geom = _coords_to_geom(row.coordinates)
            if geom is not None:
                existing.append(
                    ExistingParcel(id=row.id, label=row.label, source=row.data_source, geom=geom)
                )

        # Deuxieme verification dans la transaction pour rester idempotent si le
        # script est relance pendant que la base a change.
        to_insert, _ = _filter_existing(parcels, existing, existing_iou)
        start_index = _next_label_index(existing_labels, label_prefix)
        rows: list[Parcelle] = []
        for offset, parcel in enumerate(to_insert):
            part = "" if parcel.part <= 1 else f" (partie {parcel.part})"
            notes = (
                f"Import KML — {parcel.file} — repère {parcel.display_order}{part}. "
                "Délimitation uniquement, aucune culture importée."
            )
            rows.append(
                Parcelle(
                    id=uuid.uuid4(),
                    label=f"{label_prefix} {start_index + offset:03d}",
                    coordinates=parcel.coordinates,
                    center_lat=parcel.center_lat,
                    center_lng=parcel.center_lng,
                    surface_ha=parcel.surface_ha,
                    # Tous les champs de culture / analyse restent volontairement NULL.
                    culture_declared=None,
                    culture_detected=None,
                    ndvi_percentage=None,
                    ndre=None,
                    spectral_bands=None,
                    confidence=None,
                    verdict="Contour importé depuis un fichier KML",
                    details=f"Délimitation importée depuis {parcel.file} (aucune culture importée).",
                    saison=None,
                    soil_type=None,
                    risk_factors=None,
                    recommendations=None,
                    data_source=source,
                    owner_name=None,
                    notes=notes,
                )
            )

        session.add_all(rows)
        await session.commit()
        return len(rows), len(to_insert)


def _per_file_summary(parcels: list[KmlParcel], files: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name in files:
        selected = [parcel for parcel in parcels if parcel.file == name]
        summary[name] = {
            "parcelles": len(selected),
            "surface_ha": sum(parcel.surface_ha for parcel in selected),
        }
    return summary


async def _run(args: argparse.Namespace) -> int:
    kml_dir = args.kml_dir.resolve()
    all_parcels, file_stats = _load_kml(kml_dir)
    files = sorted(file_stats)

    unique_parcels, dropped = _dedupe_import(all_parcels, args.import_iou)
    existing = await _load_existing()
    to_insert, skipped_sources = _filter_existing(unique_parcels, existing, args.existing_iou)

    report = {
        "kml_dir": str(kml_dir),
        "source": args.source,
        "fichiers": files,
        "contours_lus": len(all_parcels),
        "contours_apres_dedoublonnage": len(unique_parcels),
        "doublons_kml_ignores": len(dropped),
        "parcelles_existantes_examinees": len(existing),
        "kml_deja_presents_en_base": len(unique_parcels) - len(to_insert),
        "sources_existantes_concernees": dict(skipped_sources),
        "a_importer": len(to_insert),
        "surface_a_importer_ha": round(sum(parcel.surface_ha for parcel in to_insert), 4),
        "detail_fichiers": {
            name: {
                **file_stats[name],
                **_per_file_summary(to_insert, [name])[name],
            }
            for name in files
        },
    }

    print(f"Dossier KML : {kml_dir}")
    print(f"Fichiers    : {len(files)}")
    print(f"Contours lus: {len(all_parcels)}")
    print(f"Doublons KML ignores (IoU >= {args.import_iou:.2f}) : {len(dropped)}")
    print(f"KML deja presents en base (IoU >= {args.existing_iou:.2f}) : {len(unique_parcels) - len(to_insert)}")
    if skipped_sources:
        print("  sources concernees :")
        for source, count in skipped_sources.most_common():
            print(f"    {count:>3}  {source}")
    print(f"Parcelles a importer : {len(to_insert)}  ({report['surface_a_importer_ha']:.2f} ha)")
    print("Detail par fichier :")
    for name in files:
        detail = report["detail_fichiers"][name]
        print(
            f"  {name}: {detail['parcelles']} à importer"
            f" / {detail['polygons']} polygones lus"
            f" ({detail['surface_ha']:.2f} ha)"
        )

    if args.json_report:
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Rapport JSON ecrit : {args.json_report}")

    if not args.apply:
        print("Mode rapport uniquement : aucune modification en base (ajouter --apply pour importer).")
        return 0

    inserted, expected = await _apply_import(
        to_insert, args.source, args.label_prefix, args.existing_iou
    )
    if inserted != expected:
        print(f"Attention : {inserted} lignes inserees au lieu de {expected} attendues.")
        return 1
    print(f"{inserted} parcelles importees dans la base (data_source = {args.source!r}).")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Importe les contours KML comme parcelles, sans importer les cultures."
    )
    parser.add_argument(
        "--kml-dir",
        type=Path,
        default=DEFAULT_KML_DIR,
        help=f"Dossier des fichiers KML (defaut : {DEFAULT_KML_DIR}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ecrit en base. Sans cette option, le script ne fait qu'un rapport.",
    )
    parser.add_argument(
        "--existing-iou",
        type=float,
        default=DEFAULT_EXISTING_IOU,
        help="Seuil IoU au-dessus duquel un KML est considere deja present en base.",
    )
    parser.add_argument(
        "--import-iou",
        type=float,
        default=DEFAULT_IMPORT_IOU,
        help="Seuil IoU de fusion de deux contours quasi identiques dans les KML.",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Valeur de data_source.")
    parser.add_argument(
        "--label-prefix",
        default=DEFAULT_LABEL_PREFIX,
        help="Prefixe des libelles (ex. 'Parcelle KML 001').",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Ecrit le rapport detaille dans ce fichier JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    async def _main() -> int:
        try:
            return await _run(args)
        finally:
            # Evite un avertissement asyncpg/SSL a l'arret du processus.
            await engine.dispose()

    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
