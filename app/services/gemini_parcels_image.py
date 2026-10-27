"""Parcelles reconstruites depuis une image où Gemini a DESSINÉ les limites.

Principe
--------
Un modèle Gemini d'image (voir `gemini_parcels._call_gemini_image`) reçoit la
capture satellite et renvoie **la même scène** avec les limites de parcelles
tracées en rouge, au lieu d'une liste de coordonnées. Dessiner une ligne le long
d'un chemin ou d'une diguette est une tâche visuelle, bien plus fiable pour le
modèle que d'énumérer des polygones sous forme de nombres : les contours épousent
donc la structure réelle du parcellaire.

Ce module transforme l'image en polygones géoréférencés :

    1. masque des pixels rouges (la limite dessinée) ;
    2. retrait des taches rouges parasites — le modèle ajoute parfois quelques
       marques compactes à l'intérieur d'un champ, qui découperaient une
       parcelle en deux ; une vraie limite s'étend loin dans au moins une
       direction, une tache non ;
    3. dilatation des traits : le modèle laisse de petits jours entre deux
       limites qui se rejoignent, et un seul jour suffit à faire communiquer
       deux parcelles. Dilater le rouge est l'opération sûre — elle ne peut que
       renforcer une séparation, jamais fusionner deux zones déjà distinctes.
       Le rayon n'est pas fixé : il est augmenté jusqu'à ce que le nombre de
       parcelles cesse de progresser, ce qui ferme les jours les plus larges
       sans raboter les parcelles plus que nécessaire ;
    4. composantes connexes du COMPLÉMENT : chaque zone non rouge enfermée par
       des limites est une parcelle ;
    5. extension de chaque parcelle sur les traits épaissis, jusqu'à la
       médiatrice avec sa voisine : les contours se touchent sans se recouvrir et
       l'épaississement de l'étape 3 ne rabote plus les parcelles ;
    6. mise en polygone, puis passage en coordonnées normalisées (0–1).

Le résultat est une liste d'anneaux normalisés, directement acceptés par
`gemini_parcels._candidate_from_ring`, qui applique la validation (surface
minimale, forme) et le géoréférencement : ce module ne connaît ni la latitude,
ni le zoom, ni le CRS.

Limites connues
---------------
* Une limite non fermée fait communiquer deux parcelles voisines : elles
  ressortent fusionnées. C'est la même faiblesse que le modèle de segmentation
  VHR, mais elle est ici visible et localisée (le trou dans la ligne).
* Le fond (tout ce qui n'est pas une parcelle : bâti, routes, image extérieure)
  est la zone reliée au bord de l'image. Les grandes zones de bord sont
  écartées ; une petite parcelle de bord, elle, est conservée.
* La résolution de l'image renvoyée par Gemini varie (souvent 1024 px). Comme
  tout est ramené en coordonnées normalisées, cela n'a pas d'incidence — à
  condition que le modèle n'ait ni recadré ni déformé la scène (voir
  `gemini_parcels._call_gemini_image`).
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import shape

logger = logging.getLogger("agrisat.services.gemini_parcels_image")

# Un pixel est « limite » quand le rouge domine franchement le vert et le bleu.
# Large volontairement : les traits dessinés sont parfois atténués (survol,
# flou), mais un champ n'est jamais franchement plus rouge que vert et bleu.
RED_MIN = 110
RED_DOMINANCE = 45

# Une tache rouge qui n'atteint cette longueur (px) dans AUCUNE direction est
# considérée comme parasite. Les limites traversent l'image ; les taches du
# modèle mesurent quelques dizaines de pixels au plus.
BLOB_MAX_EXTENT_PX = 45
# Épaississement des traits. Un trait interrompu sur quelques pixels suffit à
# faire communiquer deux parcelles voisines : le rayon n'est donc pas figé, il
# est cherché dans cette plage (voir `_separating_mask`). Comme `_grow_labels`
# rend ensuite à chaque parcelle la surface perdue, épaissir ne coûte rien ; au
# delà de cette plage, seules des séparations parasites apparaissent.
DILATE_MIN_PX = 2
DILATE_MAX_PX = 6

# Part de l'image au-delà de laquelle une zone reliée au bord est le fond
# (bâti, routes, image hors parcelles) et non une parcelle coupée par le bord.
BACKGROUND_BORDER_RATIO = 0.20
# Part minimale de l'image pour qu'une zone non rouge soit une parcelle : en
# dessous, c'est du bruit d'extraction. La surface métrique minimale est, elle,
# appliquée plus tard par `_candidate_from_ring`.
PARCEL_MIN_RATIO = 0.00015

# Un contour issu d'un masque de pixels compte un sommet par marche (jusqu'à
# plus de 1 000 par parcelle). Douglas-Peucker à 1,5 px supprime le bruit
# d'échelle sans déplacer la limite.
SIMPLIFY_PX = 1.5


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels où le rouge domine le vert et le bleu (traits dessinés)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r >= RED_MIN) & ((r - g) >= RED_DOMINANCE) & ((r - b) >= RED_DOMINANCE)


def _drop_blobs(mask: np.ndarray) -> np.ndarray:
    """Retire les taches rouges compactes (parasites) et garde les lignes."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return mask
    keep = mask.copy()
    objects = ndimage.find_objects(labels)
    for index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        if max(height, width) < BLOB_MAX_EXTENT_PX:
            keep[labels == index] = False
    return keep


# Voisinage direct : deux parcelles qui ne se touchent que par un coin restent
# distinctes, comme sur le terrain.
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _dilate_lines(mask: np.ndarray, radius: int) -> np.ndarray:
    """Épaissit les traits de `radius` pixels pour refermer les jours.

    On dilate au lieu de fermer : une fermeture (dilatation puis érosion)
    rouvre les jours fins qu'elle vient de combler et peut faire communiquer
    deux parcelles — mesuré sur une capture réelle, elle ramenait 48 parcelles
    à une seule. La dilatation seule est monotone : elle ne fait qu'ajouter du
    rouge, donc ne peut que séparer davantage.
    """
    radius = max(1, int(radius))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (x * x + y * y) <= radius * radius
    return ndimage.binary_dilation(mask, structure=disk, border_value=0)


def _count_parcels(separating: np.ndarray, min_area: float) -> int:
    """Nombre de parcelles qu'une épaisseur de traits donnée fait ressortir."""
    interior = ~separating
    labels, count = ndimage.label(interior, structure=_FOUR_CONNECTED)
    if count == 0:
        return 0
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    # Le fond est la zone reliée au bord : on l'exclut quand elle est immense,
    # mais une parcelle de bord, elle, doit être conservée.
    on_border = np.unique(
        np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
    )
    is_background = np.zeros(count + 1, dtype=bool)
    is_background[on_border] = True
    is_background &= areas > BACKGROUND_BORDER_RATIO * interior.size

    keep = areas >= min_area
    keep[0] = False  # étiquette 0 = pixels rouges, pas une parcelle
    keep &= ~is_background
    return int(np.count_nonzero(keep))


def _separating_mask(mask: np.ndarray) -> np.ndarray:
    """Épaississement retenu pour séparer le plus de parcelles distinctes.

    Un jour de quelques pixels dans un trait fusionne deux parcelles voisines.
    Le rayon nécessaire dépend du modèle et de la résolution qu'il renvoie : il
    est donc évalué sur la plage `DILATE_MIN_PX`–`DILATE_MAX_PX`, et le plus
    petit rayon qui atteint le meilleur compte est gardé. Épaissir ne rabote
    rien puisque `_grow_labels` rend ensuite sa surface à chaque parcelle.
    """
    min_area = PARCEL_MIN_RATIO * mask.size
    best_radius = DILATE_MIN_PX
    best_count = -1
    for radius in range(DILATE_MIN_PX, DILATE_MAX_PX + 1):
        count = _count_parcels(_dilate_lines(mask, radius), min_area)
        if count > best_count:
            best_radius, best_count = radius, count
    return _dilate_lines(mask, best_radius)


def _grow_labels(labels: np.ndarray, separating: np.ndarray) -> np.ndarray:
    """Rend à chaque parcelle la bande de traits épaissis qu'elle a perdue.

    Chaque pixel de la bande est attribué à la parcelle la plus proche : deux
    voisines se rejoignent sur la médiatrice du trait qui les sépare, c'est-à-dire
    exactement là où passe la limite réelle. Les contours se touchent donc sans
    se recouvrir, et l'épaississement ne réduit plus la surface des parcelles.
    """
    if not separating.any() or not labels.any():
        return labels
    indices = ndimage.distance_transform_edt(
        separating, return_distances=False, return_indices=True
    )
    return labels[tuple(indices)]


def _touches_border(component: np.ndarray) -> bool:
    return bool(
        component[0, :].any()
        or component[-1, :].any()
        or component[:, 0].any()
        or component[:, -1].any()
    )


def _component_ring(component: np.ndarray) -> list[dict[str, float]] | None:
    """Plus grand polygone d'une composante, en coordonnées normalisées 0–1."""
    from rasterio.features import shapes

    height, width = component.shape
    polygons: list[tuple[Any, float]] = []
    for geometry, value in shapes(
        component.astype(np.uint8), mask=component, connectivity=4
    ):
        if not value:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        polygons.append((geom, float(geom.area)))
    if not polygons:
        return None
    geom = max(polygons, key=lambda item: item[1])[0]
    if geom.geom_type == "MultiPolygon":
        parts = sorted(geom.geoms, key=lambda part: part.area, reverse=True)
        if not parts:
            return None
        geom = parts[0]
    if geom.geom_type != "Polygon" or geom.is_empty:
        return None
    geom = geom.simplify(SIMPLIFY_PX, preserve_topology=True)
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        parts = sorted(geom.geoms, key=lambda part: part.area, reverse=True)
        if not parts:
            return None
        geom = parts[0]
    if geom.geom_type != "Polygon":
        return None

    ring: list[dict[str, float]] = []
    for x, y in geom.exterior.coords[:-1]:
        ring.append({"x": float(min(1.0, max(0.0, x / width))),
                     "y": float(min(1.0, max(0.0, y / height)))})
    return ring if len(ring) >= 3 else None


def framing_warning(source_bytes: bytes, generated_bytes: bytes) -> str | None:
    """Avertissement quand l'image renvoyée n'a plus le même cadrage que l'entrée.

    Le géoréférencement repose sur des coordonnées normalisées : il n'est juste
    que si le modèle a gardé la même scène, sans recadrer ni changer le rapport
    largeur/hauteur. Un rendu à une autre résolution est normal (le modèle sort
    souvent du 1024 px) ; un changement de rapport, non.
    """
    try:
        with Image.open(BytesIO(source_bytes)) as source, Image.open(BytesIO(generated_bytes)) as generated:
            source_ratio = source.width / source.height
            generated_ratio = generated.width / generated.height
    except Exception:  # noqa: BLE001
        return None
    if source_ratio <= 0 or generated_ratio <= 0:
        return None
    if abs(generated_ratio - source_ratio) / source_ratio > 0.02:
        return (
            "le modele image a change le rapport largeur/hauteur "
            f"({source_ratio:.3f} -> {generated_ratio:.3f}) : les contours peuvent etre decales"
        )
    return None


def parcel_rings_from_red_lines(image_bytes: bytes) -> list[list[dict[str, float]]]:
    """Anneaux normalisés (x, y dans 0–1) des parcelles d'une image à traits rouges.

    Lève `ValueError` si l'image est illisible ou ne contient aucun trait rouge.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as error:  # noqa: BLE001
        raise ValueError("Image Gemini illisible.") from error
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[0] < 8 or rgb.shape[1] < 8:
        raise ValueError("Image Gemini trop petite ou invalide.")

    mask = _drop_blobs(_red_mask(rgb))
    if not mask.any():
        raise ValueError("Aucun trait rouge détecté dans l'image renvoyée.")
    separating = _separating_mask(mask)
    interior = ~separating

    labels, count = ndimage.label(interior, structure=_FOUR_CONNECTED)
    # Chaque parcelle reprend la bande de traits jusqu'à la médiatrice : sans
    # cela, épaissir les traits raboterait d'autant sa surface.
    labels = _grow_labels(labels, separating)
    total = labels.size
    rings: list[list[dict[str, float]]] = []
    for index in range(1, count + 1):
        component = labels == index
        area = int(component.sum())
        if area < PARCEL_MIN_RATIO * total:
            continue
        if _touches_border(component) and area > BACKGROUND_BORDER_RATIO * total:
            # Zone reliée au bord et énorme : c'est le fond, pas une parcelle.
            continue
        ring = _component_ring(component)
        if ring is not None:
            rings.append(ring)

    logger.info(
        "Lignes rouges : %d parcelle(s) reconstruite(s) (%.2f %% de pixels rouges).",
        len(rings),
        100.0 * float(mask.sum()) / total,
    )
    return rings
