"""Configuration de la journalisation du backend.

`httpx` journalise chaque requête HTTP à INFO avec l'URL complète. La capture
satellite passe la clé Google Maps dans la query string (`...&key=...`) : sans
filtre, la clé apparaît en clair dans le terminal à chaque tuile. On garde donc
`httpx` au niveau WARNING et on masque, par défense en profondeur, toute clé
présente dans un message de log.
"""

from __future__ import annotations

import logging
import re

# `key=...` (Static Maps) et les clés Google `AIza...` nues (URL d'exception).
_API_KEY_PARAM_RE = re.compile(r"(?i)\b(key|api_key|apikey)=([^&\s\"'<>]+)")
_GOOGLE_API_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}")


def redact_api_keys(message: str) -> str:
    """Masque les clés d'API d'un message destiné aux logs."""
    message = _API_KEY_PARAM_RE.sub(r"\1=***", message)
    return _GOOGLE_API_KEY_RE.sub("***", message)


class RedactApiKeysFilter(logging.Filter):
    """Filtre de logging qui retire les clés d'API des messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — un message non formatable ne doit pas être bloqué
            return True
        redacted = redact_api_keys(message)
        if redacted != message:
            # Le message est désormais déjà formaté : ne pas réappliquer args.
            record.msg = redacted
            record.args = ()
        return True


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)

    # httpx loggue « HTTP Request: GET https://...&key=... » à INFO.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    redaction = RedactApiKeysFilter()
    seen: set[int] = set()
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            handler.addFilter(redaction)
