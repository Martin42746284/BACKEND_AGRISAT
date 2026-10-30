"""Backfill du bonus d'orge (+10 points sous 70 %) sur les parcelles déjà analysées.

Le score initial est relu dans le verdict (« probabilité d'orge X % »), puis le
score final est recalculé avec `final_barley_score()` et le seuil de confirmation
courant. Les colonnes `confidence` et `hybrid_score` ainsi que le texte du verdict
sont mis à jour. Les parcelles « Autre culture » ne sont pas touchées.

Le script est idempotent : le score initial reste écrit dans le verdict, donc une
seconde exécution réapplique exactement le même résultat.

Usage (depuis BACKEND_AGRISAT) :
    python scripts/backfill_barley_bonus.py            # simulation, aucune écriture
    python scripts/backfill_barley_bonus.py --apply    # écriture + sauvegarde JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db import async_session_maker, engine  # noqa: E402
from app.models import Parcelle  # noqa: E402
from app.services.orge_scoring import (  # noqa: E402
    CONFIRM_THRESHOLD_PCT,
    final_barley_score,
)

PROB_RE = re.compile(r"probabilit[ée] d'orge\s+(\d+(?:[.,]\d+)?)\s*%", re.I)
PROB_SEGMENT_RE = re.compile(
    r"probabilit[ée] d'orge\s+\d+(?:[.,]\d+)?\s*%"
    r"(?:\s*→\s*score final\s+\d+(?:[.,]\d+)?\s*%)?",
    re.I,
)
STATUS_PREFIX_RE = re.compile(
    r"^\s*(?:✅|⚠️|❌)?\s*(?:ORGE CONFIRM[ÉE]E|ORGE [ÀA] V[ÉE]RIFIER)\s*—\s*",
    re.I,
)


def _num(value: float) -> str:
    """66.0 → « 66 », 62.5 → « 62.5 »."""
    return format(float(value), "g")


def initial_score(verdict: str | None) -> float | None:
    match = PROB_RE.search(verdict or "")
    return float(match.group(1).replace(",", ".")) if match else None


def new_verdict(verdict: str, initial: float, final: float, confirmed: bool) -> str:
    label = "✅ ORGE CONFIRMÉE" if confirmed else "⚠️ ORGE À VÉRIFIER"
    body = STATUS_PREFIX_RE.sub("", verdict, count=1)
    body, replaced = PROB_SEGMENT_RE.subn(
        f"probabilité d'orge {_num(initial)}% → score final {_num(final)}%",
        body,
        count=1,
    )
    if not replaced:  # ne devrait pas arriver : initial_score a déjà matché
        raise ValueError(f"segment de score introuvable dans : {verdict!r}")
    return f"{label} — {body}"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="écrit réellement en base")
    parser.add_argument(
        "--backup",
        default="",
        help="chemin du JSON de sauvegarde (défaut : .cache/backfill_barley_bonus_<date>.json)",
    )
    args = parser.parse_args()

    async with async_session_maker() as session:
        rows = (
            await session.execute(
                select(Parcelle)
                .where(Parcelle.culture_detected == "Orge")
                .order_by(Parcelle.label, Parcelle.id)
            )
        ).scalars().all()

        changes: list[tuple[Parcelle, dict[str, Any], float, float, bool]] = []
        skipped: list[tuple[Parcelle, str]] = []

        for row in rows:
            initial = initial_score(row.verdict)
            if initial is None:
                skipped.append((row, "score initial introuvable dans le verdict"))
                continue
            if row.confidence is None and row.hybrid_score is None:
                skipped.append((row, "parcelle non analysée"))
                continue

            final = final_barley_score(initial)
            confirmed = final >= CONFIRM_THRESHOLD_PCT
            verdict = new_verdict(row.verdict or "", initial, final, confirmed)
            before = {
                "confidence": row.confidence,
                "hybrid_score": row.hybrid_score,
                "verdict": row.verdict,
            }
            changes.append((row, before, final, final, confirmed))

        # Statistiques avant écriture.
        touched = []
        flips = []
        for row, before, confidence, hybrid, confirmed in changes:
            differs = (
                before["confidence"] != confidence
                or before["hybrid_score"] != hybrid
                or before["verdict"] != new_verdict(row.verdict or "", initial_score(row.verdict) or 0, confidence, confirmed)
            )
            if differs:
                touched.append((row, before, confidence, hybrid, confirmed))
            old_status = (before["confidence"] or 0) >= CONFIRM_THRESHOLD_PCT
            if old_status != confirmed:
                flips.append((row, before, confidence, confirmed))

        print(f"parcelles Orge en base : {len(rows)}")
        print(f"à mettre à jour          : {len(touched)}")
        print(f"déjà conformes           : {len(changes) - len(touched)}")
        print(f"ignorées                 : {len(skipped)}")
        print(f"changements de statut    : {len(flips)}")
        print("statuts après backfill   :", dict(Counter("confirmée" if c[4] else "à vérifier" for c in changes)))
        for row, reason in skipped:
            print(f"  IGNORÉE {row.id} ({row.label!r}) : {reason}")
        print()
        print("--- échantillon (10) ---")
        for row, before, confidence, hybrid, confirmed in touched[:10]:
            print(
                f"{(row.label or '')[:28]:28} | "
                f"conf {before['confidence']} → {confidence} | "
                f"hybride {before['hybrid_score']} → {hybrid} | "
                f"{'CONFIRMÉE' if confirmed else 'À VÉRIFIER'}"
            )
            print(f"  avant : {before['verdict']}")
            print(f"  après : {new_verdict(row.verdict or '', initial_score(row.verdict) or 0, confidence, confirmed)}")

        if not args.apply:
            print("\nSimulation uniquement — relancer avec --apply pour écrire en base.")
            await engine.dispose()
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = Path(args.backup) if args.backup else Path(".cache") / f"backfill_barley_bonus_{timestamp}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(
            json.dumps(
                [
                    {
                        "id": str(row.id),
                        "label": row.label,
                        **before,
                    }
                    for row, before, *_ in changes
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSauvegarde avant écriture : {backup.resolve()}")

        for row, before, confidence, hybrid, confirmed in touched:
            initial = initial_score(row.verdict) or 0
            row.confidence = confidence
            row.hybrid_score = hybrid
            row.verdict = new_verdict(row.verdict or "", initial, confidence, confirmed)
        await session.commit()
        print(f"{len(touched)} parcelle(s) mise(s) à jour.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
