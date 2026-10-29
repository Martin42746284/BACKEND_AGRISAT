"""Règle métier blé/avoine issus des fichiers KML.

Ces tests ne touchent pas la base : ils vérifient la reconnaissance des cultures
dans les repères KML et le forçage « Autre culture » au moment de l'analyse.
"""

from app.services.kml_cultures import (
    FORCED_OTHER_CULTURE,
    is_forced_other_culture,
    is_non_barley_label,
    kml_crop_from_notes,
    kml_crop_label,
    kml_declared_crop,
    marker_text,
)
from app.services.parcel_registry import analysis_to_values


def _analysis(culture: str = "Orge") -> dict:
    return {
        "culture_detected": culture,
        "confidence": 0.83,
        "verdict": f"{culture} confirmée (83%)",
        "saison": "2026-C",
        "data_source": "Modèle RF",
    }


def _values(notes: str | None) -> dict:
    return analysis_to_values(
        _analysis(),
        label="Parcelle KML 001",
        coordinates=[{"lat": -19.8, "lng": 47.0}],
        center_lat=-19.8,
        center_lng=47.0,
        surface_ha=1.0,
        notes=notes,
    )


def test_kml_crop_label_recognizes_wheat_and_oats():
    assert kml_crop_label("Blé") == "Blé"
    assert kml_crop_label("blé Z85") == "Blé"
    assert kml_crop_label("Avoine") == "Avoine"
    assert kml_crop_label("pomme de terre + avoine") == "Avoine"
    assert kml_crop_label("Orge Z89") is None
    assert kml_crop_label("vide") is None


def test_declared_crop_reads_barley_and_other_kml_crops():
    assert kml_declared_crop("Orge Z89") == "Orge"
    assert kml_declared_crop("orge z77") == "Orge"
    assert kml_declared_crop("Blé") == "Blé"
    assert kml_declared_crop("pomme de terre + avoine") == "Avoine"
    assert kml_declared_crop("Vide") is None


def test_non_barley_legacy_label_is_not_a_culture():
    assert is_non_barley_label("Non-orge") is True
    assert is_non_barley_label("NON_ORGE") is True
    assert is_non_barley_label("Autre culture") is False


def test_model_non_orge_result_is_stored_as_autre_culture():
    values = analysis_to_values(
        _analysis("Non-orge"),
        label="Parcelle 001",
        coordinates=[{"lat": -19.8, "lng": 47.0}],
        center_lat=-19.8,
        center_lng=47.0,
        surface_ha=1.0,
    )
    assert values["culture_detected"] == "Autre culture"


def test_analysis_only_knows_barley_and_other_culture():
    from app.services.analyze_parcel import detected_culture_label

    assert detected_culture_label(True) == "Orge"
    assert detected_culture_label(False) == "Autre culture"


def test_other_kml_crops_are_not_forced_to_autre_culture():
    """La règle ne vise que le blé et l'avoine : une orge KML garde le modèle."""
    notes = f"Import KML — AGRI-MADA (1).kml — repère 0003. {marker_text('Orge')}"
    assert is_forced_other_culture(notes) is False
    values = analysis_to_values(
        _analysis("Orge"),
        label="Parcelle KML 003",
        coordinates=[{"lat": -19.8, "lng": 47.0}],
        center_lat=-19.8,
        center_lng=47.0,
        surface_ha=1.0,
        notes=notes,
    )
    assert values["culture_detected"] == "Orge"
    assert values["verdict"] == "Orge confirmée (83%)"


def test_potato_kml_marker_is_not_forced_either():
    notes = f"Import KML — AGRI-MADA (1).kml — repère 0004. {marker_text('Pomme de terre')}"
    assert is_forced_other_culture(notes) is False


def test_marker_is_relu_from_notes():
    notes = f"Import KML — Projet 2.kml — repère 001. {marker_text('Avoine')}"
    assert kml_crop_from_notes(notes) == "Avoine"
    assert is_forced_other_culture(notes) is True
    assert is_forced_other_culture("Import KML — autre repère, aucune culture.") is False


def test_wheat_parcel_is_always_other_culture_even_if_model_says_barley():
    notes = f"Import KML — AGRI-MADA (1).kml — repère 0012. {marker_text('Blé')}"
    values = _values(notes)
    assert values["culture_detected"] == FORCED_OTHER_CULTURE
    assert values["culture_declared"] == "Blé"
    assert "Blé" in values["verdict"]
    # Les textes de la fiche décrivent la règle, pas une classification du modèle.
    assert "Règle métier" in values["details"]
    assert "classifieur orge / autre culture ne s'applique pas" in values["recommendations"]


def test_other_parcels_keep_the_model_result():
    values = _values("Import KML — AGRI-MADA (1).kml — repère 0002. Délimitation uniquement.")
    assert values["culture_detected"] == "Orge"
    assert values["verdict"] == "Orge confirmée (83%)"
