"""Géométrie de la capture écran de /api/gemini-parcels.

Régression : la capture demandait `size = largeur_écran / 2` avec `scale=2`.
`scale` ne modifie que le nombre de pixels rendus, pas l'étendue couverte : la
capture ne voyait donc que la moitié du terrain annoncé par le géoréférencement,
et chaque parcelle ressortait deux fois plus loin du centre qu'elle ne l'est.
"""

import math

import pytest

from app.services.gemini_parcels import (
    SCREEN_CAPTURE_SCALE,
    STATIC_MAPS_MAX_SIZE,
    _valid_bounds,
    _viewport_tiling,
)


def test_viewport_wide_enough_for_two_columns():
    # 1000 px d'écran = 1000 px de terrain à couvrir ; un appel en couvre 640.
    cols, rows, cell_w, cell_h, _, _ = _viewport_tiling(1000.0, 1000.0)
    assert cols == 2
    assert rows == 2
    assert cell_w == pytest.approx(500.0)
    assert cell_h == pytest.approx(500.0)


def test_cells_never_exceed_one_static_maps_call():
    for width, height in [(1, 1), (320, 200), (640, 640), (641, 640), (1447, 708), (3840, 2160)]:
        cols, rows, cell_w, cell_h, _, _ = _viewport_tiling(float(width), float(height))
        assert cell_w <= STATIC_MAPS_MAX_SIZE
        assert cell_h <= STATIC_MAPS_MAX_SIZE
        # les cellules recouvrent exactement l'étendue demandée
        assert cols * cell_w == pytest.approx(float(width))
        assert rows * cell_h == pytest.approx(float(height))


def test_image_is_oversampled_but_keeps_the_same_extent():
    cols, rows, cell_w, cell_h, out_w, out_h = _viewport_tiling(1447.0, 708.0)
    # 1447 px de terrain -> 2894 px d'image : plus de détail, même emprise.
    assert out_w == int(round(1447 * SCREEN_CAPTURE_SCALE))
    assert out_h == int(round(708 * SCREEN_CAPTURE_SCALE))
    assert cols * cell_w == pytest.approx(1447.0)
    assert rows * cell_h == pytest.approx(708.0)


def test_small_viewport_needs_a_single_tile():
    cols, rows, cell_w, cell_h, out_w, out_h = _viewport_tiling(640.0, 400.0)
    assert (cols, rows) == (1, 1)
    assert cell_w == pytest.approx(640.0)
    assert cell_h == pytest.approx(400.0)
    assert (out_w, out_h) == (1280, 800)


def test_degenerate_extent_still_requests_one_tile():
    cols, rows, cell_w, _, out_w, _ = _viewport_tiling(0.0, 0.0)
    assert (cols, rows) == (1, 1)
    assert cell_w == pytest.approx(1.0)
    assert math.isfinite(out_w)


def test_valid_bounds_accepts_ordered_box():
    assert _valid_bounds({"south": 48.0, "west": 1.0, "north": 48.1, "east": 1.1})


@pytest.mark.parametrize(
    "bounds",
    [
        {"south": 48.1, "west": 1.0, "north": 48.0, "east": 1.1},  # inversées
        {"south": 48.0, "west": 1.1, "north": 48.1, "east": 1.0},
        {"south": float("nan"), "west": 1.0, "north": 48.1, "east": 1.1},
        {"south": 48.0, "west": 1.0, "north": 48.1},  # incomplètes
        {"south": 48.0, "west": 1.0, "north": 48.1, "east": "est"},
    ],
)
def test_valid_bounds_rejects_unusable_box(bounds):
    assert not _valid_bounds(bounds)
