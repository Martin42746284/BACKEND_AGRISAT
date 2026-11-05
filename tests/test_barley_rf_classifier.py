import json
import math

import pandas as pd
import pytest

from app.services.barley_rf_classifier import _verdict, build_features, classify
from app.services.orge_scoring import final_barley_score, is_barley_confirmed


def _series() -> list[dict[str, float | str | None]]:
    """Série mensuelle synthétique d'une parcelle (12 mois, cycle d'orge)."""
    rows: list[dict[str, float | str | None]] = []
    for offset, date in enumerate(pd.date_range("2025-10-01", periods=12, freq="MS")):
        month = date.month
        # Pic de végétation en juin-juillet, senescence en octobre.
        ndvi = 0.18 + 0.55 * math.exp(-((offset - 8) ** 2) / 5.0)
        b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12 = (
            0.10,
            0.13,
            0.16,
            0.22,
            0.30,
            0.36,
            0.38 + ndvi * 0.20,
            0.39 + ndvi * 0.20,
            0.22,
            0.16,
        )
        rows.append(
            {
                "date": date.strftime("%Y-%m"),
                "B2": b2,
                "B3": b3,
                "B4": b4,
                "B5": b5,
                "B6": b6,
                "B7": b7,
                "B8": b8,
                "B8A": b8a,
                "B11": b11,
                "B12": b12,
                "NDVI": ndvi,
                "EVI": 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1),
                "NDRE": (b8 - b5) / (b8 + b5),
                "NDMI": (b8 - b11) / (b8 + b11),
            }
        )
    return rows


def test_features_have_model_shape_and_no_nan():
    features = build_features(_series())

    assert features is not None
    assert features.shape == (12, 64)
    assert not features.isna().any().any()


def test_classify_returns_parcel_probability():
    from app.services.barley_rf_classifier import THRESHOLD_PATH

    result = classify(_series())
    expected_threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["seuil_retenu"]

    assert result is not None
    assert 0.0 <= result["probability"] <= 1.0
    assert result["threshold"] == pytest.approx(expected_threshold)
    assert result["n_features"] == 64
    assert result["n_observations"] == 12
    assert result["verdict"] in {"ORGE", "A_VERIFIER", "AUTRE_CULTURE"}
    assert result["model_version"].startswith("orge_rf")


def test_cloudy_month_is_tolerated():
    series = _series()
    series[2] = {"date": series[2]["date"]}

    result = classify(series)

    assert result is not None
    assert result["n_observations"] == 12


def test_empty_series_returns_none():
    assert classify([]) is None
    assert build_features([]) is None


def test_model_artifacts_are_valid_json():
    """metrics.json et threshold_selection.json alimentent /api/modeles : validés ici, pas au démarrage."""
    from app.services.barley_rf_classifier import METRICS_PATH, THRESHOLD_PATH, load_model

    for path in (METRICS_PATH, THRESHOLD_PATH):
        json.loads(path.read_text(encoding="utf-8"))

    bundle = load_model()
    expected_threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["seuil_retenu"]

    assert bundle is not None
    assert bundle["metrics"].get("generated_at")
    assert bundle["threshold"] == pytest.approx(expected_threshold)


def test_final_barley_score_examples():
    """Règle utilisateur : +10 points sous 70 %, puis confirmation si ≥60 %."""
    assert final_barley_score(56) == pytest.approx(66)
    assert final_barley_score(62) == pytest.approx(72)
    assert final_barley_score(69) == pytest.approx(79)
    assert final_barley_score(75) == pytest.approx(75)
    assert final_barley_score(70) == pytest.approx(70)
    assert final_barley_score(100) == pytest.approx(100)

    assert not is_barley_confirmed(49)  # 59 < 60
    assert is_barley_confirmed(50)      # 60 >= 60 (seuil inclus)
    assert is_barley_confirmed(56)      # 66 >= 60
    assert is_barley_confirmed(62)      # 72 >= 60
    assert is_barley_confirmed(75)      # 75 >= 60
    assert is_barley_confirmed(60)      # 70 >= 60
    assert is_barley_confirmed(65)      # 75 >= 60


def test_verdict_uses_final_score_only():
    # Les anciens garde-fous agro (NDVI d'octobre, date du pic) ne bloquent plus la confirmation.
    assert _verdict(0.90, ndvi_oct=0.20, doy_peak=210) == "ORGE"
    assert _verdict(0.90, ndvi_oct=0.80, doy_peak=210) == "ORGE"
    assert _verdict(0.62, ndvi_oct=0.80, doy_peak=300) == "ORGE"   # 72 >= 60
    assert _verdict(0.60, ndvi_oct=0.20, doy_peak=210) == "ORGE"   # 70 >= 60
    assert _verdict(0.56, ndvi_oct=0.20, doy_peak=210) == "ORGE"   # 66 >= 60
    assert _verdict(0.54, ndvi_oct=0.20, doy_peak=210) == "ORGE"   # 64 >= 60
    assert _verdict(0.49, ndvi_oct=0.20, doy_peak=210) == "A_VERIFIER"  # 59 < 60
    assert _verdict(0.05, ndvi_oct=0.20, doy_peak=210) == "AUTRE_CULTURE"

    # Le seuil « autre culture » du modèle reste celui du RF (et non l'ancien 0.15).
    assert _verdict(0.30, threshold=0.0567) == "A_VERIFIER"
    assert _verdict(0.05, threshold=0.0567) == "AUTRE_CULTURE"


def test_verdict_from_rf_applies_final_score():
    from app.services.analyze_parcel import verdict_from_rf

    for initial, final, confirmed in ((49, 59, False), (56, 66, True), (62, 72, True), (75, 75, True)):
        result = verdict_from_rf({"probability": initial / 100, "is_barley": True, "threshold": 0.2892})
        assert result["hybrid_score"] == final
        assert result["final_confirmed"] is confirmed
        assert result["final_confidence"] == final
        assert f"score final {final}%" in result["final_verdict"]

    # Autre culture : le bonus ne touche pas la confiance de classe (1 - probabilité d'orge).
    autre = verdict_from_rf({"probability": 0.23, "is_barley": False, "threshold": 0.2892})
    assert autre["hybrid_score"] == 23
    assert autre["final_confirmed"] is False
    assert autre["final_confidence"] == 77
    assert "score final" not in autre["final_verdict"]
    assert autre["final_verdict"].startswith("Autre culture")
