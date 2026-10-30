"""Règles métier des cultures déclarées dans les fichiers KML.

Le blé et l'avoine ne font pas partie du classifieur orge/non-orge : une analyse
de terrain les classerait « Orge » ou « Non-orge » selon le couvert observé.
Pour ces parcelles déjà identifiées dans un fichier KML, la règle métier impose
donc le résultat « Autre culture » à chaque analyse.

La règle est portée par `notes`, qui n'est pas effacé par la suppression de la
culture (contrairement aux colonnes de classification) : elle survit donc à
`DELETE /api/parcelles/{id}/culture` et s'applique de nouveau à la réanalyse.
"""

from __future__ import annotations

import re
import unicodedata

FORCED_OTHER_CULTURE = "Autre culture"

# Étiquette posée sur un candidat d'analyse dont la ligne `parcelles` est un
# contour KML blé/avoine : il est classé « Autre culture » par la règle métier,
# sans passer par le modèle de classification.
FORCED_CANDIDATE_TAG = "kml_forced_culture"

# Marqueur lisible, écrit par scripts/import_kml_parcels.py :
#   « Culture d'origine (KML) : Blé. »
MARKER_RE = re.compile(r"culture d['’]origine\s*\(kml\)\s*:\s*([^.;\n]+)", re.IGNORECASE)


def normalize_label(value: str) -> str:
    """Minuscules sans accents, pour comparer « Blé », « ble » et « BLE »."""
    decomposed = unicodedata.normalize("NFD", value)
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return without_accents.lower()


# Cultures reconnues dans les noms de repères KML, hors blé/avoine. Elles ne
# sont pas reprises à l'import (aucune culture n'est attribuée avant l'analyse),
# mais restent exposées par `kml_declared_crop` pour la lecture des fichiers. La
# règle « Autre culture » ne s'applique qu'au blé et à l'avoine
# (`kml_crop_label`). Les variantes fautives rencontrées dans les KML
# (« raigrasse », « carrotte », « Hricot »…) sont ramenées au libellé canonique.
OTHER_KML_CROPS: tuple[tuple[str, str], ...] = (
    ("riz", "Riz"),
    ("hricot", "Haricot"),
    ("haricot", "Haricot"),
    ("mais", "Maïs"),
    ("manioc", "Manioc"),
    ("carrotte", "Carotte"),
    ("carotte", "Carotte"),
    ("raygrasse", "Ray-grass"),
    ("raigrasse", "Ray-grass"),
    ("raugrasse", "Ray-grass"),
    ("petit pois", "Petit pois"),
    ("petit poid", "Petit pois"),
    ("courgette", "Courgette"),
    ("patate douce", "Patate douce"),
    ("pattate douce", "Patate douce"),
    ("tomate", "Tomate"),
    ("soja", "Soja"),
    ("anana", "Ananas"),
    ("ail", "Ail"),
    ("chou", "Chou"),
    ("cou fleure", "Chou"),
    ("pomme de terre", "Pomme de terre"),
)


def kml_crop_label(placemark_name: str | None) -> str | None:
    """Culture KML reconnue dans un nom de repère, ou None si ce n'est ni blé ni avoine.

    Gère les libellés composés (« pomme de terre + avoine ») : la présence
    d'avoine ou de blé suffit, car la parcelle est alors concernée par la règle.
    """
    normalized = normalize_label(placemark_name or "")
    # « Voine » est la faute de frappe la plus fréquente pour « Avoine ».
    if re.search(r"\b(avoine|voine)\b", normalized):
        return "Avoine"
    if re.search(r"\bble\b", normalized):
        return "Blé"
    return None


def kml_declared_crop(placemark_name: str | None) -> str | None:
    """Culture lisible dans un nom de repère KML, toutes cultures reconnues.

    Lecture informative des fichiers : l'import n'écrit plus `culture_declared`,
    une parcelle importée n'ayant aucune culture avant d'être analysée. Seul le
    blé et l'avoine sont mémorisés (marqueur de `notes`) pour être classés
    « Autre culture » à l'analyse, sans modèle.
    """
    forced = kml_crop_label(placemark_name)
    if forced is not None:
        return forced
    normalized = normalize_label(placemark_name or "")
    # « orgeZ89 » / « orge z 77 » : le préfixe suffit, le numéro de zone suit.
    if re.search(r"\borg", normalized):
        return "Orge"
    # « pomme de terrre », « Pommme de tere », « Pome de terre », « Pmme de terre »…
    if re.search(r"\bpom\w*|\bpmme", normalized) and re.search(r"\bter", normalized):
        return "Pomme de terre"
    for needle, label in OTHER_KML_CROPS:
        if re.search(rf"\b{re.escape(needle)}\b", normalized):
            return label
    return None


def kml_crop_from_notes(notes: str | None) -> str | None:
    """Culture KML relue depuis le marqueur persistant, ou None."""
    if not notes:
        return None
    match = MARKER_RE.search(notes)
    if match is None:
        return None
    return kml_crop_label(match.group(1))


def is_forced_other_culture(notes: str | None) -> bool:
    """Vrai si la parcelle doit toujours être analysée « Autre culture »."""
    return kml_crop_from_notes(notes) is not None


def marker_text(crop: str) -> str:
    """Marqueur ajouté aux notes d'une parcelle importée depuis un KML."""
    return f"Culture d'origine (KML) : {crop}."


# Libellés hérités du classifieur binaire qui ne sont pas des cultures.
_NON_BARLEY_LABELS = {"non-orge", "non orge", "non_orge", "nonorge"}


def is_non_barley_label(value: str | None) -> bool:
    """Vrai pour un ancien statut « NON_ORGE » / « Non-orge » du classifieur."""
    if not value:
        return False
    return normalize_label(value).strip() in _NON_BARLEY_LABELS


def forced_culture_texts(crop: str, model_details: str | None = None) -> dict[str, str]:
    """Textes cohérents d'une parcelle blé/avoine classée « Autre culture ».

    Le modèle orge/non-orge reste mentionné comme mesure indicative, mais la
    décision affichée (verdict, détails, recommandation) est la règle métier.
    """
    lower = crop.lower()
    details = (
        f"Règle métier : parcelle déclarée « {crop} » dans le fichier KML, "
        f"toujours classée « {FORCED_OTHER_CULTURE} »."
    )
    if model_details:
        details += f" Mesures du modèle (indicatives) : {model_details}."
    return {
        "verdict": f"{FORCED_OTHER_CULTURE} — culture d'origine (KML) : {crop}.",
        "details": details,
        "recommendations": (
            f"Parcelle de {lower} identifiée dans le fichier KML : le classifieur "
            f"orge / autre culture ne s'applique pas, le résultat reste « {FORCED_OTHER_CULTURE} »."
        ),
    }
