"""Remet les données de culture enregistrées en cohérence avec l'application.

Corrections appliquées (idempotentes, relançables) :

1. Ancien statut binaire « NON_ORGE » / « Non-orge » → « Autre culture » dans la
   culture détectée, le verdict, les détails, les recommandations et les
   facteurs de risque. L'application ne connaît que « Orge » et « Autre
   culture ».
2. Culture déclarée des parcelles importées depuis un KML : le nom du repère
   (« Orge Z89 », « Blé », « Avoine »…) est relu dans les fichiers KML et
   inscrit dans `culture_declared` quand la colonne est vide, pour que la fiche
   affiche la culture réelle du contour.
3. Parcelles blé/avoine (règle métier « Autre culture ») : verdict, détails et
   recommandations réécrits à partir de la règle, au lieu de conserver les
   textes d'une classification orge/non-orge qui les contredisent.

    python scripts/fix_culture_data.py            # rapport (aucune écriture)
    python scripts/fix_culture_data.py --apply    # écrit en base
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select  # noqa: E402

from app.db import async_session_maker, engine  # noqa: E402
from app.models import Parcelle  # noqa: E402
from app.services.kml_cultures import (  # noqa: E402
    FORCED_OTHER_CULTURE,
    forced_culture_texts,
    is_non_barley_label,
    kml_crop_from_notes,
    kml_declared_crop,
    marker_text,
    normalize_label,
)

KML = "{http://www.opengis.net/kml/2.2}"
DEFAULT_KML_DIR = BACKEND_ROOT / "kml"
IMPORT_NOTE_RE = re.compile(r"Import KML\s*[—-]\s*(?P<file>[^\n]+?)\s*[—-]\s*repère\s*(?P<order>\d{4})", re.IGNORECASE)


def _kml_names(kml_dir: Path) -> dict[tuple[str, int], str]:
    """(nom de fichier, ordre du repère) → nom du repère, pour tous les KML."""
    names: dict[tuple[str, int], str] = {}
    for path in sorted(kml_dir.glob("*.kml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            print(f"  ! {path.name} illisible : {error}")
            continue
        for order, placemark in enumerate(root.findall(f".//{KML}Placemark"), start=1):
            names[(path.name, order)] = (placemark.findtext(f"{KML}name") or "").strip()
    return names


def _declared_from_notes(notes: str | None, names: dict[tuple[str, int], str]) -> str | None:
    match = IMPORT_NOTE_RE.search(notes or "")
    if match is None:
        return None
    name = names.get((match.group("file").strip(), int(match.group("order"))))
    if name is None:
        return None
    return kml_declared_crop(name)


def _import_details(notes: str | None, crop: str) -> str | None:
    """Texte de délimitation écrit par l'import KML, pour un contour non analysé."""
    match = IMPORT_NOTE_RE.search(notes or "")
    if match is None:
        return None
    return f"Délimitation importée depuis {match.group('file').strip()} — culture d'origine : {crop}."


def _model_details(details: str | None) -> str | None:
    """Mesures du classifieur orge/autre culture, sans le texte de la règle métier."""
    if not details or details.startswith("Délimitation importée"):
        return None
    marker = "Mesures du modèle (indicatives) :"
    tail = details.split(marker, 1)[1] if marker in details else details
    # Une exécution précédente a pu imbriquer le texte de la règle : il ne reste
    # alors aucune mesure de modèle exploitable.
    if tail.startswith("Règle métier") or tail.startswith("Délimitation importée"):
        return None
    return _replace_non_barley(tail.strip().rstrip(".")) or None


def _replace_non_barley(text: str | None) -> str | None:
    if not text:
        return text
    return (
        text.replace("❌ NON-ORGE —", f"{FORCED_OTHER_CULTURE} —")
        .replace("⚠️ NON-ORGE PROBABLE", f"{FORCED_OTHER_CULTURE} probable")
        .replace("NON-ORGE", FORCED_OTHER_CULTURE)
        .replace("Non-orge", FORCED_OTHER_CULTURE)
        .replace("non-orge", "autre culture")
    )


def _plan_updates(row: Parcelle, names: dict[tuple[str, int], str]) -> dict[str, object]:
    updates: dict[str, object] = {}
    forced_crop = kml_crop_from_notes(row.notes)
    # Preuves d'une analyse réelle : le libellé « Autre culture » seul ne suffit
    # pas, il a pu être écrit par un ancien import.
    analyzed = (
        row.confidence is not None
        or row.ndvi_percentage is not None
        or row.hybrid_score is not None
        or bool(row.saison)
        or bool(row.growth_stage)
    )
    declared = row.culture_declared or _declared_from_notes(row.notes, names)

    if forced_crop is not None and analyzed:
        texts = forced_culture_texts(forced_crop, _model_details(row.details))
        if row.culture_detected != FORCED_OTHER_CULTURE:
            updates["culture_detected"] = FORCED_OTHER_CULTURE
        if row.verdict != texts["verdict"]:
            updates["verdict"] = texts["verdict"]
        if row.details != texts["details"]:
            updates["details"] = texts["details"]
        if row.recommendations != texts["recommendations"]:
            updates["recommendations"] = texts["recommendations"]
        if row.culture_declared != forced_crop:
            updates["culture_declared"] = forced_crop
        if row.risk_factors:
            kept = [risk for risk in row.risk_factors if "non-orge" not in normalize_label(risk)]
            if kept != list(row.risk_factors):
                updates["risk_factors"] = kept
        declared = forced_crop
    else:
        if forced_crop is not None:
            # Contour non analysé : aucun résultat de modèle à corriger, seule la
            # culture déclarée est renseignée. Un contour simplement recoupé avec
            # un repère KML (tracé corrigé, détection) n'est pas un import KML.
            imported_notes = _import_details(row.notes, forced_crop)
            expected_verdict = (
                "Contour importé depuis un fichier KML"
                if imported_notes
                else f"Contour non analysé — culture d'origine (KML) : {forced_crop}"
            )
            if row.culture_detected == FORCED_OTHER_CULTURE:
                updates["culture_detected"] = None
            if row.verdict != expected_verdict:
                updates["verdict"] = expected_verdict
            if row.recommendations is not None:
                updates["recommendations"] = None
            # Détails écrits par une exécution précédente sur un contour qui
            # n'est pas un import KML : ils ne décrivent aucune donnée réelle.
            if not imported_notes and (row.details or "").startswith("Règle métier"):
                updates["details"] = None
            declared = forced_crop

        if is_non_barley_label(row.culture_detected):
            updates["culture_detected"] = FORCED_OTHER_CULTURE
        for field in ("verdict", "details", "recommendations"):
            current = getattr(row, field)
            cleaned = _replace_non_barley(current)
            if cleaned != current:
                updates[field] = cleaned
        if row.risk_factors:
            cleaned_risks = [risk.replace("Verdict non-orge", "Classement « Autre culture »") for risk in row.risk_factors]
            if cleaned_risks != list(row.risk_factors):
                updates["risk_factors"] = cleaned_risks

        if not row.culture_declared and declared:
            updates["culture_declared"] = declared

    if not analyzed and declared:
        # Textes d'import écrits avant la reconnaissance de cette culture :
        # « aucune culture importée » contredit la culture déclarée affichée.
        imported = _import_details(row.notes, declared)
        if imported and (row.details is None or "aucune culture importée" in (row.details or "")):
            if row.details != imported:
                updates["details"] = imported
        notes = row.notes or ""
        if "aucune culture importée" in notes:
            cleaned_notes = notes.replace(
                "Délimitation uniquement, aucune culture importée.",
                f"{marker_text(declared)} Délimitation importée, culture non classée automatiquement.",
            )
            if cleaned_notes != notes:
                updates["notes"] = cleaned_notes
    return updates


async def main() -> None:
    parser = argparse.ArgumentParser(description="Corrige les statuts de culture et les cultures déclarées KML.")
    parser.add_argument("--apply", action="store_true", help="écrire les corrections en base")
    parser.add_argument("--kml-dir", type=Path, default=DEFAULT_KML_DIR)
    args = parser.parse_args()

    names = _kml_names(args.kml_dir)
    print(f"{len(names)} repères KML lus dans {args.kml_dir}")

    total = 0
    updated = 0
    async with async_session_maker() as session:
        rows = (await session.execute(select(Parcelle))).scalars().all()
        total = len(rows)
        for row in rows:
            updates = _plan_updates(row, names)
            if not updates:
                continue
            updated += 1
            if args.apply:
                for key, value in updates.items():
                    setattr(row, key, value)
        if args.apply:
            await session.commit()

    verb = "corrigées" if args.apply else "à corriger"
    print(f"{updated}/{total} parcelles {verb}.")
    if not args.apply:
        print("Relancer avec --apply pour écrire les corrections.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
