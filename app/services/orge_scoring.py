"""Score final de l'orge après analyse (règle de classification du 2026-10-28).

Règle métier :
- si le score initial de l'IA est strictement inférieur à 70 %, ajouter 5 points
  de pourcentage (plafond 100 %) ;
- si le score initial est supérieur ou égal à 70 %, le conserver tel quel ;
- une parcelle est « ORGE CONFIRMÉE » si le score final est strictement
  supérieur à 65 %, sinon « À VÉRIFIER ».
"""

from __future__ import annotations

BONUS_THRESHOLD_PCT = 70.0
BONUS_POINTS = 5.0
CONFIRM_THRESHOLD_PCT = 65.0


def final_barley_score(score_pct: float) -> float:
    """Score d'orge final (%) : +5 points sous 70 %, inchangé à partir de 70 %."""
    score = float(score_pct)
    if score < BONUS_THRESHOLD_PCT:
        score += BONUS_POINTS
    return min(100.0, score)


def is_barley_confirmed(score_pct: float) -> bool:
    """Vrai si le score final dépasse strictement 65 % (ORGE CONFIRMÉE)."""
    return final_barley_score(score_pct) > CONFIRM_THRESHOLD_PCT
