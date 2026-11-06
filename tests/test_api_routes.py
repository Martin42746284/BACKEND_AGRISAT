#!/usr/bin/env python3
"""Tests unitaires pour l'API AGRISAT-BACKEND.

Ces tests vérifient :
- GET /api/modeles : présence des champs de métriques et d'entraînement attendus
- PATCH /api/parcelles/{id} : réponse 404 pour une parcelle inexistante (ne nécessite pas
  de connexion à la base de données car l'ID n'existe pas)
- POST /api/audit : création d'une entrée d'audit (nécessite une connexion DB)
- GET /api/audit : lecture des entrées d'audit (nécessite une connexion DB)

NOTE : Les tests qui créent ou lisent des ressources en base de données (POST /api/parcelles,
GET /api/audit, POST /api/audit) peuvent échouer si la connexion à la base de données distante
(Neon) n'est pas disponible ou si le client de test (TestClient) ne gère pas correctement
les boucles d'événements asyncio avec asyncpg.
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/modeles
# ---------------------------------------------------------------------------

def test_api_modeles_contains_classification_orge(client):
    """Vérifie que la route /api/modeles retourne les métriques et l'entraînement attendus."""
    resp = client.get("/api/modeles")
    assert resp.status_code == 200
    data = resp.json()
    assert "classification_orge" in data
    rf = data["classification_orge"]
    assert rf["statut"] == "pret"
    assert rf["present"] is True

    m = rf["metriques"]
    # Champs attendus par le contrat frontend
    for key in ("seuil", "accuracy", "precision_orge", "rappel_orge", "f1_orge",
                "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"):
        assert key in m, f"champ manquant : {key}"
        assert m[key] is not None, f"champ {key} est None"

    e = rf["entrainement"]
    for key in ("n_features", "n_lignes", "n_parcelles", "n_splits",
                "anti_leakage_ok", "validation", "meilleurs_parametres"):
        assert key in e, f"champ manquant : {key}"
        assert e[key] is not None, f"champ {key} est None"


def test_api_modeles_delimitation_vhr_structure(client):
    resp = client.get("/api/modeles")
    assert resp.status_code == 200
    data = resp.json()
    vhr = data["delimitation_vhr"]
    assert "statut" in vhr
    assert "service" in vhr


# ---------------------------------------------------------------------------
# PATCH /api/parcelles/{id}
# ---------------------------------------------------------------------------

def test_patch_parcelle_unknown_id(client):
    """PATCH sur une parcelle inexistante doit retourner 404."""
    resp = client.patch(f"/api/parcelles/{uuid.uuid4()}", json={"culture_detected": "Orge"})
    assert resp.status_code == 404


def test_patch_parcelle_updates_fields(client):
    """PATCH sur une parcelle existante doit mettre à jour les champs."""
    # Créer d'abord une parcelle via POST
    parcelle_payload = {
        "label": "Test parcelle",
        "coordinates": [{"lat": -19.82, "lng": 47.07}, {"lat": -19.82, "lng": 47.08},
                        {"lat": -19.81, "lng": 47.08}, {"lat": -19.81, "lng": 47.07}],
        "center_lat": -19.815,
        "center_lng": 47.075,
        "surface_ha": 1.5,
        "culture_declared": "Inconnue",
        "culture_detected": None,
        "confidence": None,
        "verdict": None,
        "details": None,
        "saison": None,
        "soil_type": None,
        "risk_factors": [],
        "recommendations": None,
        "data_source": "test",
        "owner_name": "test",
        "notes": "test",
        "time_series_s2": [],
        "time_series_s1": [],
        "time_series_rain": [],
        "ndvi_percentage": None,
        "spectral_bands": None,
        "ndre": None,
        "estimated_planting_date": None,
        "estimated_harvest_date": None,
        "days_since_planting": None,
        "growth_stage": None,
        "planting_confidence": None,
        "evi": None,
        "savi": None,
        "ndwi": None,
        "agro_score": None,
        "hybrid_score": None,
        "cnn_prob_barley": None,
        "cnn_prob_non_barley": None,
    }

    create_resp = client.post("/api/parcelles", json=parcelle_payload)
    if create_resp.status_code != 201:
        pytest.skip(f"Impossible de créer une parcelle de test : {create_resp.text}")
        return

    created = create_resp.json()
    parcelle_id = created["id"]

    # Patcher la parcelle
    patch_resp = client.patch(f"/api/parcelles/{parcelle_id}", json={
        "culture_detected": "Orge",
        "confidence": 85.0,
        "verdict": "ORGE CONFIRMÉE",
    })
    assert patch_resp.status_code == 200, patch_resp.text
    updated = patch_resp.json()
    assert updated["culture_detected"] == "Orge"
    assert updated["confidence"] == 85.0
    assert updated["verdict"] == "ORGE CONFIRMÉE"


# ---------------------------------------------------------------------------
# /api/audit (GET et POST)
# ---------------------------------------------------------------------------

def test_get_audit_returns_list(client):
    """GET /api/audit doit retourner une liste (même vide)."""
    resp = client.get("/api/audit")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert isinstance(resp.json(), list)


def test_post_audit_creates_entry(client):
    """POST /api/audit doit créer une entrée et la retourner."""
    payload = {
        "user_name": "test_user",
        "action": "correction",
        "object": "Parcelle test-123",
    }
    resp = client.post("/api/audit", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["user_name"] == "test_user"
    assert data["action"] == "correction"
    assert data["object"] == "Parcelle test-123"
    assert "id" in data
    assert "at" in data


def test_post_audit_rejects_invalid_payload(client):
    """POST /api/audit doit rejeter un payload invalide."""
    resp = client.post("/api/audit", json={"user_name": ""})
    assert resp.status_code == 400