"""Progression réelle de la capture / détection des parcelles.

Le frontend n'estime plus le pourcentage dans le temps : la route NDJSON relaie
les paliers émis par le service, et ces paliers correspondent à des unités de
travail terminées (tuiles téléchargées, fenêtres Gemini analysées, etc.).
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from app.routes.gemini_parcels import post_gemini_parcels_stream, router
from app.schemas import GeminiParcelsRequest
from app.services import gemini_parcels as service
from app.services.gemini_parcels import Capture, GeminiParcelsError


def _capture() -> Capture:
    return Capture(
        center_lat=0.0,
        center_lng=0.0,
        zoom=18,
        span_m=200.0,
        size_px=256,
        south=0.0,
        west=0.0,
        north=0.0,
        east=0.0,
        image_bytes=b"",
        tiles=1,
        tile_span_m=200.0,
    )


def test_stream_route_relays_real_progress_then_result(monkeypatch):
    async def fake_detect(**kwargs):
        report = kwargs["progress"]
        await report(17, "Capture satellite — 2/8 tuiles")
        await report(63, "Analyse Gemini — fenêtre 3/6")
        return {"capture": {}, "parcels": [], "existing": [], "warnings": []}

    monkeypatch.setattr(
        "app.routes.gemini_parcels.detect_parcels_gemini", fake_detect
    )

    async def run():
        response = await post_gemini_parcels_stream(
            GeminiParcelsRequest(latitude=-18.9, longitude=47.5)
        )
        assert isinstance(response, StreamingResponse)
        chunks = [chunk async for chunk in response.body_iterator]
        return [json.loads(line) for line in b"".join(chunks).decode().splitlines()]

    events = asyncio.run(run())

    assert [event["type"] for event in events] == ["progress", "progress", "result"]
    assert events[0] == {
        "type": "progress",
        "percent": 17,
        "label": "Capture satellite — 2/8 tuiles",
    }
    assert events[1]["percent"] == 63
    assert events[2]["result"]["parcels"] == []


def test_stream_endpoint_serves_ndjson_over_http(monkeypatch):
    async def fake_detect(**kwargs):
        await kwargs["progress"](28, "Capture satellite — 4/5 tuiles")
        return {"capture": {}, "parcels": [], "existing": [], "warnings": []}

    monkeypatch.setattr(
        "app.routes.gemini_parcels.detect_parcels_gemini", fake_detect
    )
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.post(
            "/api/gemini-parcels/stream",
            json={"latitude": -18.9, "longitude": 47.5},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["percent"] == 28
    assert events[-1]["type"] == "result"


def test_stream_route_reports_service_error(monkeypatch):
    async def fake_detect(**kwargs):
        await kwargs["progress"](40, "Image satellite capturée")
        raise GeminiParcelsError("Gemini a refusé la capture.")

    monkeypatch.setattr(
        "app.routes.gemini_parcels.detect_parcels_gemini", fake_detect
    )

    async def run():
        response = await post_gemini_parcels_stream(
            GeminiParcelsRequest(latitude=-18.9, longitude=47.5)
        )
        assert isinstance(response, StreamingResponse)
        chunks = [chunk async for chunk in response.body_iterator]
        return [json.loads(line) for line in b"".join(chunks).decode().splitlines()]

    events = asyncio.run(run())

    assert [event["type"] for event in events] == ["progress", "error"]
    assert events[-1]["error"] == "Gemini a refusé la capture."


def test_analyze_windows_reports_each_completed_window(monkeypatch):
    async def fake_window_rings(window, gemini_key):
        await asyncio.sleep(0)
        return [{"ring": [[0.1, 0.1]]}]

    monkeypatch.setattr(service, "_window_rings", fake_window_rings)

    progress: list[tuple[int, str]] = []

    async def report(percent: int, label: str) -> None:
        progress.append((percent, label))

    windows = [_capture(), _capture(), _capture()]
    rings, origins, failed = asyncio.run(
        service._analyze_windows(windows, "gemini-key", report)
    )

    assert len(rings) == 3
    assert len(origins) == 3
    assert failed == []
    assert len(progress) == 3
    assert [percent for percent, _ in progress] == sorted(
        percent for percent, _ in progress
    )
    assert progress[-1][0] == 78
    assert progress[-1][1] == "Analyse Gemini — fenêtre 3/3"
