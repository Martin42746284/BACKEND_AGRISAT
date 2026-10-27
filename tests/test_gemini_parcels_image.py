"""Reconstruction des parcelles depuis une image où Gemini a dessiné les limites.

Ces tests tournent hors ligne : ils fabriquent une image de synthèse (champs de
couleurs + limites rouges) et vérifient la géométrie extraite. L'appel Gemini
lui-même n'est pas couvert (il demande une clé API et un accès réseau).
"""

import io

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app.services.gemini_parcels_image import (  # noqa: E402
    DILATE_MIN_PX,
    _dilate_lines,
    _drop_blobs,
    _red_mask,
    _separating_mask,
    framing_warning,
    parcel_rings_from_red_lines,
)

SIZE = 512
BACKGROUND = (120, 130, 90)
FIELD_COLORS = [(110, 140, 70), (150, 120, 80), (90, 150, 100), (170, 160, 110)]
RED = (230, 20, 20)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _grid_image(
    *,
    columns: int = 3,
    rows: int = 3,
    line_px: int = 3,
    stray_blob: bool = False,
) -> Image.Image:
    """Mosaïque de `columns` x `rows` champs séparés par des limites rouges."""
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    cell_w = SIZE // columns
    cell_h = SIZE // rows
    for row in range(rows):
        for column in range(columns):
            box = (
                column * cell_w + line_px,
                row * cell_h + line_px,
                (column + 1) * cell_w - line_px,
                (row + 1) * cell_h - line_px,
            )
            draw.rectangle(box, fill=FIELD_COLORS[(row * columns + column) % len(FIELD_COLORS)])
    for column in range(columns + 1):
        x = min(SIZE - line_px, column * cell_w)
        draw.rectangle((x, 0, x + line_px - 1, SIZE), fill=RED)
    for row in range(rows + 1):
        y = min(SIZE - line_px, row * cell_h)
        draw.rectangle((0, y, SIZE, y + line_px - 1), fill=RED)
    if stray_blob:
        # Tache rouge compacte au milieu d'un champ : le modèle en laisse parfois.
        center = (cell_w + cell_w // 2, cell_h + cell_h // 2)
        draw.ellipse(
            (center[0] - 12, center[1] - 12, center[0] + 12, center[1] + 12), fill=RED
        )
    return image


def _ring_centers(rings) -> list[tuple[float, float]]:
    """Centre de la boîte englobante de chaque anneau.

    Une moyenne des sommets serait biaisée : un contour tracé au pixel n'a pas
    ses sommets répartis régulièrement (Douglas-Peucker les concentrate aux
    angles), ce qui déplace la moyenne sans que la forme bouge.
    """
    return [
        (
            (min(point["x"] for point in ring) + max(point["x"] for point in ring)) / 2,
            (min(point["y"] for point in ring) + max(point["y"] for point in ring)) / 2,
        )
        for ring in rings
    ]


def test_red_mask_selects_only_the_drawn_lines():
    image = np.asarray(_grid_image(columns=2, rows=2))
    mask = _red_mask(image)

    assert mask.sum() > 0
    # Un champ vert/beige n'est jamais « rouge » au sens du masque.
    assert not mask[64:120, 64:120].any()
    # La limite en est bien.
    assert mask[:, 0:3].any()


def test_grid_yields_one_parcel_per_cell():
    rings = parcel_rings_from_red_lines(_png(_grid_image(columns=3, rows=3)))

    assert len(rings) == 9
    # Chaque anneau reste dans sa cellule (centre proche du tiers attendu).
    centers = sorted(_ring_centers(rings))
    assert centers[0][0] == pytest.approx(1 / 6, abs=0.06)
    assert centers[-1][1] == pytest.approx(5 / 6, abs=0.06)
    for ring in rings:
        assert len(ring) >= 3
        for point in ring:
            assert 0.0 <= point["x"] <= 1.0
            assert 0.0 <= point["y"] <= 1.0


def test_stray_blob_does_not_split_a_field():
    without = parcel_rings_from_red_lines(_png(_grid_image(columns=3, rows=3)))
    with_blob = parcel_rings_from_red_lines(
        _png(_grid_image(columns=3, rows=3, stray_blob=True))
    )

    # Une tache parasite ne doit ni créer de parcelle, ni découper une cellule.
    assert len(with_blob) == len(without) == 9


def _grid_with_gap(gap_px: int, columns: int = 4, rows: int = 4) -> Image.Image:
    """Grille dont une limite interne est interrompue sur `gap_px` pixels.

    Le modèle laisse couramment ce genre de jour entre deux limites qui se
    rejoignent ; c'est la cause directe de deux parcelles fusionnées.
    """
    image = _grid_image(columns=columns, rows=rows)
    draw = ImageDraw.Draw(image)
    cell_w = SIZE // columns
    cell_h = SIZE // rows
    x = cell_w  # limite verticale entre la colonne 0 et la colonne 1
    top = cell_h + 20  # franchement sous la première limite horizontale
    draw.rectangle((x, top, x + 2, top + gap_px - 1), fill=FIELD_COLORS[0])
    return image


def test_gap_in_a_boundary_does_not_merge_neighbours():
    image = _grid_with_gap(7)
    mask = _drop_blobs(_red_mask(np.asarray(image)))

    # Un rayon figé au minimum ne referme pas un jour de 7 pixels : c'était le
    # défaut — les deux cellules voisines ressortaient n'en formant qu'une.
    assert not np.array_equal(_separating_mask(mask), _dilate_lines(mask, DILATE_MIN_PX))
    # La recherche du rayon, elle, rend bien une parcelle par cellule.
    assert len(parcel_rings_from_red_lines(_png(image))) == 16


def test_search_does_not_thicken_more_than_needed():
    # Grille déjà bien fermée : aucune raison d'épaissir au-delà du minimum, ce
    # qui raboterait les parcelles et ferait disparaître les plus étroites.
    mask = _drop_blobs(_red_mask(np.asarray(_grid_image(columns=3, rows=3))))

    assert np.array_equal(_separating_mask(mask), _dilate_lines(mask, DILATE_MIN_PX))


def test_gap_wider_than_the_search_limit_stays_merged():
    """Limite assumée : au-delà de deux fois le rayon maximal, le jour subsiste."""
    assert len(parcel_rings_from_red_lines(_png(_grid_with_gap(60)))) == 15


def _enclosure(with_divider: bool) -> Image.Image:
    """Deux champs dans une même enceinte rouge, avec ou sans leur séparation."""
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 475, 475), fill=FIELD_COLORS[0])
    draw.rectangle((266, 40, 472, 472), fill=FIELD_COLORS[1])
    draw.rectangle((33, 33, 479, 479), outline=RED, width=5)
    if with_divider:
        draw.rectangle((252, 40, 258, 472), fill=RED)
    return image


def test_unclosed_boundary_merges_neighbours():
    """Limite connue : une limite non fermée fait fusionner deux parcelles."""
    assert len(parcel_rings_from_red_lines(_png(_enclosure(with_divider=True)))) == 2
    # Sans la séparation, les deux champs forment une seule zone : c'est
    # exactement le défaut à connaître du mode « image ».
    assert len(parcel_rings_from_red_lines(_png(_enclosure(with_divider=False)))) == 1


def test_no_red_line_is_rejected():
    plain = Image.new("RGB", (SIZE, SIZE), FIELD_COLORS[0])

    with pytest.raises(ValueError):
        parcel_rings_from_red_lines(_png(plain))


def test_unreadable_image_is_rejected():
    with pytest.raises(ValueError):
        parcel_rings_from_red_lines(b"pas-une-image")


def test_border_background_is_dropped_but_cells_are_kept():
    # Anneau rouge à l'intérieur d'une image : le pourtour (relié au bord) est le
    # fond, le carré central est une parcelle.
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    ImageDraw.Draw(image).rectangle((80, 80, SIZE - 80, SIZE - 80), outline=RED, width=4)

    rings = parcel_rings_from_red_lines(_png(image))

    assert len(rings) == 1
    assert _ring_centers(rings)[0] == (pytest.approx(0.5, abs=0.05), pytest.approx(0.5, abs=0.05))


def test_framing_warning_flags_aspect_change():
    source = _png(Image.new("RGB", (256, 256), BACKGROUND))
    same_ratio = _png(Image.new("RGB", (1024, 1024), BACKGROUND))
    changed = _png(Image.new("RGB", (1024, 512), BACKGROUND))

    assert framing_warning(source, same_ratio) is None
    assert framing_warning(source, changed) is not None
