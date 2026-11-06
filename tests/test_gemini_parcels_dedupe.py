"""Rapprochement des contours détectés avec les parcelles déjà en base.

Une zone déjà analysée contient des parcelles délimitées et des parcelles qui ne
le sont pas encore : un nouveau tracé ne doit jamais reprendre une limite
existante (aucune ligne en double, aucune limite modifiée), mais doit bien
conserver les parcelles neuves. Les IoU ci-dessous sont ceux d'un tracé réel :
il ne coïncide jamais exactement avec la ligne enregistrée.
"""

import pytest

from shapely.geometry import Polygon
from shapely.strtree import STRtree

from app.services.gemini_parcels import Candidate, ExistingParcel, _match_existing


def _square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _candidate(geom: Polygon) -> Candidate:
    return Candidate(
        geom=geom, coordinates=[], center_lat=0.0, center_lng=0.0,
        area_ha=geom.area / 10_000.0, confidence=None, label=None,
    )


def _existing(parcel_id: str, geom: Polygon) -> ExistingParcel:
    return ExistingParcel(
        id=parcel_id, label=parcel_id, surface_ha=geom.area / 10_000.0, geom=geom
    )


def _match(geom: Polygon, rows: list[ExistingParcel]) -> ExistingParcel | None:
    return _match_existing(_candidate(geom), rows, STRtree([row.geom for row in rows]))


@pytest.fixture()
def existing_parcel() -> ExistingParcel:
    return _existing("PAR-1", _square(0, 0, 100, 100))


def test_same_footprint_is_not_traced_again(existing_parcel):
    assert _match(_square(0, 0, 100, 100), [existing_parcel]) is existing_parcel


def test_close_trace_is_not_traced_again(existing_parcel):
    # Tracé décalé de 12 m et 10 % plus grand : IoU 0,72.
    assert _match(_square(12, 12, 122, 122), [existing_parcel]) is existing_parcel


def test_contour_merging_an_existing_parcel_is_rejected(existing_parcel):
    # Parcelle existante fusionnée avec une voisine neuve : IoU 0,5 et centre à
    # 50 m, donc seuls les critères de recouvrement l'écartent.
    assert _match(_square(0, 0, 200, 100), [existing_parcel]) is existing_parcel


def test_polygon_swallowing_a_small_existing_parcel_is_rejected():
    small = _existing("PAR-2", _square(0, 0, 20, 20))
    assert _match(_square(0, 0, 200, 200), [small]) is small


def test_new_parcel_beside_an_existing_one_is_kept(existing_parcel):
    assert _match(_square(100, 0, 200, 100), [existing_parcel]) is None


def test_new_parcel_between_two_existing_ones_is_kept():
    rows = [
        _existing("PAR-1", _square(0, 0, 100, 100)),
        _existing("PAR-3", _square(100, 0, 200, 100)),
    ]
    assert _match(_square(0, 0, 100, 100), rows) is rows[0]
    assert _match(_square(100, 0, 200, 100), rows) is rows[1]
    assert _match(_square(0, 100, 100, 200), rows) is None
