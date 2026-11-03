"""Amélioration de la lisibilité des captures avant analyse par l'IA.

Objectif
--------
Les limites de parcelles peu contrastées (sol nu, cultures similaires, image
ancienne ou agrandie au-delà de sa résolution native) sont difficiles à voir,
pour l'utilisateur comme pour le modèle. Ce module renforce uniquement les
ruptures **déjà présentes** :

1. CLAHE sur la luminance : contraste local, sans déplacer les couleurs ;
2. masque de netteté : accentue les micro-contours (diguettes, chemins, fossés) ;
3. saturation légère : sépare mieux deux couverts de teintes proches.

Il n'invente aucune limite et ne change **jamais** les dimensions ni le cadrage :
le géoréférencement des contours (coordonnées normalisées) reste exact.

Modes
-----
- ``off``    : image d'origine, aucun octet modifié ;
- ``light``  : réglage par défaut, adapté à la plupart des captures ;
- ``strong`` : pour les images très plates ou peu contrastées, au prix d'un
  bruit légèrement plus visible.

``cv2`` est fourni par ultralytics dans l'environnement du backend. S'il est
absent, un contraste automatique Pillow prend le relais : le service ne tombe
jamais en erreur à cause de l'amélioration d'image.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Final

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger("agrisat.services.image_enhancement")

MODES: Final[tuple[str, ...]] = ("off", "light", "strong")
DEFAULT_MODE: Final[str] = "light"

# Réglages par mode : (CLAHE clipLimit, tuiles, netteté rayon / %, seuil, saturation)
_PRESETS: Final[dict[str, tuple[float, int, float, int, int, float]]] = {
    "light": (1.8, 8, 1.0, 70, 3, 1.08),
    "strong": (2.8, 8, 1.2, 120, 2, 1.18),
}


def normalize_mode(value: str | None) -> str:
    """Mode valide, ou ``light`` pour toute valeur absente / inconnue."""
    mode = (value or DEFAULT_MODE).strip().lower()
    if mode not in MODES:
        logger.warning("Mode d'amélioration d'image inconnu (%r) : %s appliqué.", value, DEFAULT_MODE)
        return DEFAULT_MODE
    return mode


def enhance_image_bytes(image_bytes: bytes, mode: str = DEFAULT_MODE) -> bytes:
    """Renvoie l'image PNG améliorée, ou l'originale si le traitement échoue.

    Le mode ``off`` renvoie exactement les octets reçus (aucune copie, aucun
    réencodage) : les tests et les appels existants ne sont pas modifiés.
    """
    mode = normalize_mode(mode)
    if mode == "off":
        return image_bytes
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except Exception as exc:  # noqa: BLE001 — une image illisible est traitée plus loin
        logger.warning("Amélioration d'image ignorée (image illisible) : %s", exc)
        return image_bytes

    try:
        enhanced = enhance_image(image, mode)
        buffer = BytesIO()
        enhanced.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:  # noqa: BLE001 — l'analyse doit continuer avec l'image d'origine
        logger.exception("Amélioration d'image impossible : image d'origine conservée.")
        return image_bytes


def enhance_image(image: Image.Image, mode: str = DEFAULT_MODE) -> Image.Image:
    """Applique le prétraitement à une image RGB, sans changer sa taille."""
    mode = normalize_mode(mode)
    if mode == "off":
        return image

    clip, grid, radius, percent, threshold, saturation = _PRESETS[mode]
    rgb = np.asarray(image.convert("RGB"))
    rgb = _clahe(rgb, clip, grid)

    result = Image.fromarray(rgb)
    result = result.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    if saturation != 1.0:
        result = ImageEnhance.Color(result).enhance(saturation)
    return result


def _clahe(rgb: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    """CLAHE sur le canal L de LAB ; repli Pillow si OpenCV est indisponible."""
    try:
        import cv2  # fourni par ultralytics ; import local pour rester optionnel

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
        lightness = clahe.apply(lightness)
        return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:  # noqa: BLE001 — repli sans dépendance native
        return np.asarray(ImageOps.autocontrast(Image.fromarray(rgb), cutoff=1))
