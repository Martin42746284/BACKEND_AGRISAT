"""Supprime les parcelles tracees par Gemini (data_source « Detection Gemini »).

Chaque appel de POST /api/gemini-parcels avec `persist` (defaut) ecrit une ligne
dans `parcelles` portant `data_source = "Detection Gemini"`. Apres plusieurs
essais de reglage, ces lignes s'accumulent sur la carte. Ce script les retire
toutes, et rien d'autre : les traces manuels, les imports KML, les contours VHR
et les marqueurs de suppression ne sont pas touches.

Par defaut le script ne fait qu'un rapport (dry-run). Pour supprimer :

    python scripts/purge_gemini_parcels.py            # rapport
    python scripts/purge_gemini_parcels.py --apply    # suppression reelle

Exemples :
    python scripts/purge_gemini_parcels.py --source "Detection Gemini"
    python scripts/purge_gemini_parcels.py --apply

Apres suppression, un backend en cours d'execution peut servir son cache
parcelles pendant 30 s au plus (app/cache.py) : attendre ce delai ou le
redemarrer pour voir la carte a jour.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Permet d'executer le script depuis n'importe quel dossier, tout en chargeant
# le .env et les modules du backend (comme scripts/import_kml_parcels.py).
BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))

# La console Windows ecrit en cp1252 : sans cela, les accents des sources
# (« Détection Gemini ») s'affichent en caracteres illisibles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import delete, func, select  # noqa: E402

from app.db import async_session_maker, engine  # noqa: E402
from app.models import Parcelle  # noqa: E402
from app.services.gemini_parcels import LABEL_PREFIX, PARCEL_SOURCE  # noqa: E402


def _when(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value is not None else "(date inconnue)"


async def _purge(sources: list[str], apply: bool) -> None:
    async with async_session_maker() as session:
        inventory = (
            await session.execute(
                select(Parcelle.data_source, func.count()).group_by(Parcelle.data_source)
            )
        ).all()

        print("Parcelles par source :")
        for source, count in sorted(inventory, key=lambda row: row[1], reverse=True):
            print(f"  {count:>6}  {source or '(aucune)'}")
        if not inventory:
            print("  (base vide)")

        targets = (
            await session.execute(
                select(Parcelle.label, Parcelle.created_at, Parcelle.surface_ha)
                .where(Parcelle.data_source.in_(sources))
                .order_by(Parcelle.created_at.asc())
            )
        ).all()

        # Lignes portant le libelle Gemini mais une autre source : elles
        # echapperaient a la suppression, il faut les voir avant de conclure.
        other_sources_with_label = (
            await session.execute(
                select(func.count())
                .select_from(Parcelle)
                .where(
                    Parcelle.label.like(f"{LABEL_PREFIX}%"),
                    Parcelle.data_source.notin_(sources),
                )
            )
        ).scalar_one()

        print()
        print("A supprimer (source : " + ", ".join(f"« {s} »" for s in sources) + ") :")
        print(f"  {len(targets)} parcelle(s)")
        if targets:
            area = sum(float(surface or 0.0) for _, _, surface in targets)
            print(f"  surface cumulee   : {area:.2f} ha")
            print(f"  premiere tracee le : {_when(targets[0][1])}")
            print(f"  derniere tracee le : {_when(targets[-1][1])}")
            print("  exemples de libelles : " + ", ".join(label or "(sans libelle)" for label, _, _ in targets[:5]))
        if other_sources_with_label:
            print(
                f"  ATTENTION : {other_sources_with_label} parcelle(s) portent le libelle "
                f"« {LABEL_PREFIX} » avec une autre source et ne sont PAS supprimees "
                "(utiliser --source pour les cibler)."
            )

        if not apply:
            print()
            print("Rapport seul : rien n'a ete supprime. Relancer avec --apply pour supprimer.")
            return

        if not targets:
            print()
            print("Aucune parcelle a supprimer : la base est deja propre.")
            return

        result = await session.execute(delete(Parcelle).where(Parcelle.data_source.in_(sources)))
        await session.commit()
        removed = result.rowcount or 0

    print()
    print(f"{removed} parcelle(s) supprimee(s).")
    print(
        "Un backend en cours d'execution peut servir son cache parcelles jusqu'a 30 s : "
        "attendre ce delai ou le redemarrer."
    )


async def _run(sources: list[str], apply: bool) -> None:
    try:
        await _purge(sources, apply)
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Supprime toutes les parcelles tracees par Gemini. Par defaut le script "
            "n'affiche qu'un rapport ; --apply supprime reellement."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help=(
            f"data_source a supprimer (repetable). Defaut : « {PARCEL_SOURCE} ». "
            "Ne pas cibler « Tracé automatique supprimé » : ces marqueurs empechent "
            "les contours automatiques supprimes de reapparaitre."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Supprime reellement les lignes (sans ce drapeau : rapport seul).",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run(args.sources or [PARCEL_SOURCE], args.apply))
    except Exception as error:  # noqa: BLE001 — message lisible en ligne de commande
        print(f"Echec : {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
