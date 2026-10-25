"""Garde-fou « ne pas réanalyser » : tests des fonctions pures du registre.

Ces tests ne touchent pas la base : ils vérifient la règle qui protège les
données d'une parcelle déjà analysée.
"""

import uuid

from app.models import Parcelle
from app.services.parcel_registry import (
    ANALYSIS_MATCH_TOLERANCE_M,
    REJECTED_PARCEL_SOURCE,
    bbox_deltas,
    distance_m,
    nearest_analyzed_parcel,
    nearest_parcel,
    row_has_analysis,
    values_have_analysis,
)


def _row(**overrides) -> Parcelle:
    """Ligne `parcelles` hors base (aucune session nécessaire)."""
    defaults = {
        "id": uuid.uuid4(),
        "label": "test",
        "coordinates": [],
        "center_lat": 34.0,
        "center_lng": -6.0,
        "surface_ha": 1.0,
        "culture_declared": "Inconnue",
        "culture_detected": None,
        "ndvi_percentage": None,
        "ndre": None,
        "spectral_bands": None,
        "confidence": None,
        "verdict": None,
        "details": None,
        "saison": None,
        "soil_type": None,
        "risk_factors": [],
        "recommendations": None,
        "data_source": None,
        "owner_name": None,
        "notes": None,
    }
    defaults.update(overrides)
    return Parcelle(**defaults)


def test_traced_outline_without_analysis_is_not_analyzed():
    assert row_has_analysis(_row()) is False


def test_detected_culture_means_analyzed():
    assert row_has_analysis(_row(culture_detected="Orge")) is True


def test_indices_without_culture_mean_analyzed():
    assert row_has_analysis(_row(confidence=88.0)) is True
    assert row_has_analysis(_row(ndvi_percentage=42.5)) is True
    assert row_has_analysis(_row(agro_score=0.7)) is True


def test_rejected_contour_is_never_analyzed():
    row = _row(culture_detected="Orge", data_source=REJECTED_PARCEL_SOURCE)
    assert row_has_analysis(row) is False
    assert values_have_analysis({"data_source": REJECTED_PARCEL_SOURCE, "culture_detected": "Orge"}) is False


def test_values_have_analysis_mirrors_row_has_analysis():
    assert values_have_analysis({}) is False
    assert values_have_analysis({"culture_detected": "Orge"}) is True
    assert values_have_analysis({"culture_detected": "  "}) is False
    assert values_have_analysis({"confidence": 12}) is True
    assert values_have_analysis({"hybrid_score": 0.4}) is True


def test_distance_m_is_symmetric_and_roughly_metric():
    assert distance_m(34.0, -6.0, 34.0, -6.0) == 0
    # ~111 m vers le nord
    assert 100 < distance_m(34.0, -6.0, 34.0009, -6.0) < 120


def test_nearest_analyzed_parcel_ignores_traced_outlines():
    outline = _row()
    analyzed = _row(culture_detected="Orge")
    match = nearest_analyzed_parcel([outline, analyzed], 34.0, -6.0)
    assert match is analyzed


def test_nearest_analyzed_parcel_respects_tolerance():
    # ~55 m plus au nord : hors tolérance de 20 m.
    far = _row(culture_detected="Orge", center_lat=34.0005)
    assert nearest_analyzed_parcel([far], 34.0, -6.0) is None
    # ~5 m plus au nord : même parcelle (contour retracé légèrement différent).
    near = _row(culture_detected="Orge", center_lat=34.00005)
    assert nearest_analyzed_parcel([near], 34.0, -6.0) is near


def test_nearest_parcel_returns_traced_outline_too():
    outline = _row()
    assert nearest_parcel([outline], 34.0, -6.0) is outline


def test_analyzed_row_takes_precedence_over_nearer_outline():
    # Plusieurs lignes partagent parfois le même centroïde (contour FTW + contour
    # JECAM-VHR) : c'est la ligne porteuse de l'analyse qui fait foi, sinon une
    # relance écraserait la ligne voisine au lieu de la parcelle analysée.
    outline = _row()
    analyzed = _row(culture_detected="Orge")
    rows = [outline, analyzed]
    assert (nearest_analyzed_parcel(rows, 34.0, -6.0) or nearest_parcel(rows, 34.0, -6.0)) is analyzed


def test_bbox_deltas_cover_the_disk():
    lat_delta, lng_delta = bbox_deltas(34.0, 100.0)
    assert lat_delta > 0 and lng_delta > 0
    # La boîte englobante doit être plus large que le disque (100 m de rayon).
    assert lat_delta * 110_574 > 100
    assert lng_delta * 111_320 * 0.8 > 100


def test_tolerance_is_positive():
    assert ANALYSIS_MATCH_TOLERANCE_M > 0
