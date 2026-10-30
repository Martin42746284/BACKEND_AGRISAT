"""Blé/avoine importés d'un KML : classés « Autre culture » sans passer par le modèle.

Une analyse de zone classe toutes les parcelles du rayon. Pour les contours KML
identifiés comme blé ou avoine, la règle métier tranche avant tout appel
satellite/IA : le résultat est immédiatement « Autre culture ». Les autres
parcelles continuent d'être analysées par le modèle.
"""

import asyncio

from app.services import automatic_parcels
from app.services.automatic_parcels import KML_FORCED_SOURCE, _analyze_candidate
from app.services.kml_cultures import FORCED_CANDIDATE_TAG, FORCED_OTHER_CULTURE

POLYGON = [
    {"lat": -19.8000, "lng": 47.0000},
    {"lat": -19.8000, "lng": 47.0005},
    {"lat": -19.8005, "lng": 47.0005},
    {"lat": -19.8005, "lng": 47.0000},
]
CENTER = {"lat": -19.80025, "lng": 47.00025}


def _candidate(**overrides) -> dict:
    candidate = {
        "id": "jecam-vhr--19.800250-47.000250",
        "coordinates": POLYGON,
        "center": CENTER,
        "tags": {"source": "jecam-vhr"},
        "persist": False,
    }
    candidate.update(overrides)
    return candidate


def _run(candidate: dict) -> dict:
    return asyncio.run(
        _analyze_candidate(
            candidate,
            {"threshold": 2000, "baseTemperature": 5},
            {"value": None, "error": None},
        )
    )


def test_kml_wheat_candidate_is_classified_autre_culture_without_the_model(monkeypatch):
    async def model_must_not_run(*args, **kwargs):
        raise AssertionError("le modèle de classification ne doit pas être appelé")

    monkeypatch.setattr(automatic_parcels, "_analyze_parcel_handler", model_must_not_run)

    result = _run(_candidate(tags={"source": "jecam-vhr", FORCED_CANDIDATE_TAG: "Blé"}))

    analysis = result["analysis"]
    assert result["analysis_error"] is None
    assert analysis["culture_detected"] == FORCED_OTHER_CULTURE
    assert analysis["is_barley"] is False
    assert analysis["barley_presence"] == "none"
    assert analysis["data_source"] == KML_FORCED_SOURCE
    # Aucune confiance de modèle n'est inventée : la décision vient du KML.
    assert analysis["confidence"] is None
    assert "Blé" in analysis["verdict"]
    # La recommandation rappelle que le classifieur orge / autre culture ne s'applique pas.
    assert "ne s'applique pas" in analysis["recommendations"]


def test_kml_oat_candidate_is_forced_with_the_oat_label(monkeypatch):
    async def model_must_not_run(*args, **kwargs):
        raise AssertionError("le modèle de classification ne doit pas être appelé")

    monkeypatch.setattr(automatic_parcels, "_analyze_parcel_handler", model_must_not_run)

    result = _run(_candidate(tags={"source": "jecam-vhr", FORCED_CANDIDATE_TAG: "Avoine"}))
    assert result["analysis"]["culture_detected"] == FORCED_OTHER_CULTURE
    assert "Avoine" in result["analysis"]["verdict"]


def test_candidate_without_the_rule_still_goes_through_the_model(monkeypatch):
    calls: list[dict] = []

    async def fake_handler(body):
        calls.append(body)
        return 500, {}

    monkeypatch.setattr(automatic_parcels, "_analyze_parcel_handler", fake_handler)

    result = _run(_candidate())

    assert len(calls) == 1
    assert calls[0]["lat"] == CENTER["lat"]
    assert calls[0]["lng"] == CENTER["lng"]
    assert result["analysis"] is None
    assert result["analysis_error"] == "Analyse satellite indisponible."
