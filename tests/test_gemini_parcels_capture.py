"""Géométrie de la capture écran de /api/gemini-parcels.

Régression : la capture demandait `size = largeur_écran / 2` avec `scale=2`.
`scale` ne modifie que le nombre de pixels rendus, pas l'étendue couverte : la
capture ne voyait donc que la moitié du terrain annoncé par le géoréférencement,
et chaque parcelle ressortait deux fois plus loin du centre qu'elle ne l'est.
"""

import io
import math

import pytest
from PIL import Image

from app.services.gemini_parcels import (
    MAX_ANALYSIS_WINDOWS,
    SCREEN_CAPTURE_SCALE,
    STATIC_MAPS_MAX_SIZE,
    WINDOW_UPSCALE_PX,
    Capture,
    _lat_to_px,
    _lng_to_px,
    _mercator_world_px,
    _meters_per_pixel,
    _px_to_lat,
    _px_to_lng,
    _split_windows,
    _valid_bounds,
    _viewport_tiling,
    _window_grid,
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


# ── Découpage en fenêtres haute résolution ──

def _screen_capture(width: int = 1200, height: int = 700) -> Capture:
    """Capture d'écran de synthèse, géoréférencée comme le fait `_capture_screen_viewport`."""
    image = Image.new("RGB", (width, height), (90, 110, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    center_lat, center_lng, zoom = -19.85, 47.01, 18
    world_px = _mercator_world_px(zoom)
    # Une capture d'écran rend `SCREEN_CAPTURE_SCALE` pixels par pixel de carte.
    mpp = _meters_per_pixel(center_lat, zoom) / SCREEN_CAPTURE_SCALE
    map_px_per_image_px = 1.0 / SCREEN_CAPTURE_SCALE
    half_x = width / 2.0 * map_px_per_image_px
    half_y = height / 2.0 * map_px_per_image_px
    cx = _lng_to_px(center_lng, world_px)
    cy = _lat_to_px(center_lat, world_px)
    return Capture(
        center_lat=center_lat, center_lng=center_lng, zoom=zoom,
        span_m=max(width, height) * mpp,
        size_px=max(width, height),
        south=_px_to_lat(cy + half_y, world_px),
        west=_px_to_lng(cx - half_x, world_px),
        north=_px_to_lat(cy - half_y, world_px),
        east=_px_to_lng(cx + half_x, world_px),
        image_bytes=buffer.getvalue(), width_px=width, height_px=height,
    )


def test_wide_capture_is_split_into_overlapping_windows():
    capture = _screen_capture()
    windows = _split_windows(capture)

    assert 1 < len(windows) <= MAX_ANALYSIS_WINDOWS
    for window in windows:
        # Fenêtres carrées, rééchantillonnées pour le modèle.
        assert window.image_bytes
        with Image.open(io.BytesIO(window.image_bytes)) as image:
            assert image.width == image.height
            assert image.width == max(WINDOW_UPSCALE_PX, window.size_px)
        # Un pixel de fenêtre couvre bien moins de terrain qu'un pixel de la capture.
        assert window.span_m < capture.span_m


def test_windows_cover_the_whole_capture_and_overlap():
    capture = _screen_capture()
    windows = _split_windows(capture)

    west = min(window.west for window in windows)
    east = max(window.east for window in windows)
    south = min(window.south for window in windows)
    north = max(window.north for window in windows)
    assert west == pytest.approx(capture.west, abs=1e-6)
    assert east == pytest.approx(capture.east, abs=1e-6)
    assert south == pytest.approx(capture.south, abs=1e-6)
    assert north == pytest.approx(capture.north, abs=1e-6)
    for window in windows:
        assert window.west >= capture.west - 1e-9
        assert window.east <= capture.east + 1e-9
        assert window.south >= capture.south - 1e-9
        assert window.north <= capture.north + 1e-9
    # Deux fenêtres voisines se recouvrent : aucune bande de terrain n'est sautée.
    world_px = _mercator_world_px(capture.zoom)
    mpp = _meters_per_pixel(capture.center_lat, capture.zoom)
    window_map_px = windows[0].span_m / mpp
    xs = sorted({round(_lng_to_px(window.west, world_px), 6) for window in windows})
    ys = sorted({round(_lat_to_px(window.north, world_px), 6) for window in windows})
    for previous, current in zip(xs, xs[1:]):
        assert current - previous < window_map_px
    for previous, current in zip(ys, ys[1:]):
        assert current - previous < window_map_px


def test_narrow_capture_is_analysed_in_a_single_call():
    capture = _screen_capture()
    # Vue de 250 m : déjà assez détaillée pour un seul appel.
    capture.span_m = 250.0

    assert _split_windows(capture) == [capture]


def test_very_wide_view_falls_back_to_a_single_call():
    # 4 000 x 2 000 px à 0,5 m/px : plus de 12 fenêtres même en 480 m.
    assert _window_grid(4000, 2000, 0.5) == (0, [], [])
