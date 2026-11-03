"""Prétraitement d'image avant l'IA : cadrage conservé, modes réglables.

Les tests sont hors ligne et n'ont pas besoin d'OpenCV : le repli Pillow doit
suffire pour que l'amélioration ne bloque jamais l'analyse.
"""

import io

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.services.image_enhancement import (  # noqa: E402
    enhance_image,
    enhance_image_bytes,
    normalize_mode,
)


def _low_contrast_png(width: int = 96, height: int = 64) -> bytes:
    """Image synthétique proche d'un sol uniforme, avec une limite peu marquée."""
    y, x = np.mgrid[0:height, 0:width]
    base = 118 + (x > width // 2) * 8 + (y > height // 2) * 4
    rgb = np.stack([base, base + 3, base - 2], axis=-1).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_off_returns_the_exact_original_bytes():
    source = _low_contrast_png()
    assert enhance_image_bytes(source, "off") is source


def test_unknown_mode_falls_back_to_light():
    assert normalize_mode(None) == "light"
    assert normalize_mode("") == "light"
    assert normalize_mode("ultra") == "light"
    assert normalize_mode("STRONG") == "strong"


@pytest.mark.parametrize("mode", ["light", "strong"])
def test_enhancement_preserves_dimensions_and_framing(mode: str):
    source = _low_contrast_png()
    enhanced = enhance_image_bytes(source, mode)

    original_image = Image.open(io.BytesIO(source))
    enhanced_image = Image.open(io.BytesIO(enhanced))
    assert enhanced_image.size == original_image.size
    assert enhanced_image.mode == "RGB"


def test_enhancement_increases_local_contrast_without_changing_shape():
    source = _low_contrast_png()
    with Image.open(io.BytesIO(source)) as image:
        before = np.asarray(image.convert("RGB"), dtype=np.float32)
        after = np.asarray(enhance_image(image, "light"), dtype=np.float32)

    assert after.shape == before.shape
    # Une image presque plate doit gagner du contraste, pas perdre du signal.
    assert float(after.std()) > float(before.std())


def test_enhancement_never_raises_on_invalid_image():
    garbage = b"not an image"
    assert enhance_image_bytes(garbage, "light") == garbage
