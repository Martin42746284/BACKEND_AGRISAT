"""Masquage des clés d'API dans les logs (fuite Static Maps)."""

import logging

from app.logging_setup import RedactApiKeysFilter, redact_api_keys

_FAKE_KEY = "AIzaSyFAKE123456789012345678901234567"


def test_redacts_key_query_parameter():
    message = f"GET https://maps.googleapis.com/maps/api/staticmap?center=1&key={_FAKE_KEY}&zoom=21"
    redacted = redact_api_keys(message)
    assert _FAKE_KEY not in redacted
    assert "key=***" in redacted


def test_redacts_bare_google_key():
    assert _FAKE_KEY not in redact_api_keys(f"clé refusée : {_FAKE_KEY}")


def test_filter_rewrites_log_record():
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        __file__,
        1,
        "HTTP Request: GET %s",
        (f"https://example.test/?key={_FAKE_KEY}",),
        None,
    )
    assert RedactApiKeysFilter().filter(record) is True
    assert _FAKE_KEY not in record.getMessage()
