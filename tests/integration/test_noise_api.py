"""Integration tests for Noise Suppression REST API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from hth.api.routers.noise import set_noise_suppressor
from hth.audio.noise.contracts import NoisePreset
from hth.audio.noise.suppressor import NoiseSuppressor
from hth.core.app import create_app
from hth.core.settings import get_settings

_TOKEN = "test-token-fixed-32-chars-abc12345"
_HEADERS = {
    "X-HTH-Token": _TOKEN,
    "Origin": "http://127.0.0.1:5173",
}


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("HTH_SESSION_TOKEN", _TOKEN)
    monkeypatch.setenv("HTH_DB_PATH", ":memory:")
    monkeypatch.setenv("HTH_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("HTH_SINGLE_INSTANCE", "0")
    # Fresh suppressor instance per test
    ns = NoiseSuppressor(preset=NoisePreset.LIGHT)
    set_noise_suppressor(ns)
    yield
    ns.close()
    set_noise_suppressor(None)


@pytest.fixture
async def client() -> AsyncClient:  # type: ignore[override]
    get_settings.cache_clear()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_get_noise_config(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/noise/config", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["preset"] == "LIGHT"
    assert data["sample_rate"] == 48000
    assert data["frame_size"] == 480


@pytest.mark.asyncio
async def test_set_noise_preset(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/noise/preset",
        json={"preset": "STRONG"},
        headers=_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["preset"] == "STRONG"

    # Verify updated via GET
    get_resp = await client.get("/api/v1/noise/config", headers=_HEADERS)
    assert get_resp.json()["preset"] == "STRONG"


@pytest.mark.asyncio
async def test_set_invalid_preset_rejected(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/noise/preset",
        json={"preset": "EXTREME_INVALID"},
        headers=_HEADERS,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_noise_metrics(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/noise/metrics", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "frames_processed" in data
    assert "p50_ms" in data
    assert "p95_ms" in data
    assert data["active_preset"] == "LIGHT"


@pytest.mark.asyncio
async def test_testsink_lifecycle_api(client: AsyncClient) -> None:
    # 1. Check initial status
    status_resp = await client.get("/api/v1/noise/testsink/status", headers=_HEADERS)
    assert status_resp.status_code == 200
    st = status_resp.json()
    assert st["is_test_sink_mode"] is True
    assert st["vb_cable_transmission_enabled"] is False
    assert st["test_harness_banner"] == "TEST HARNESS — LIVE TRANSMISSION DISABLED"

    # 2. Start TestSink
    start_resp = await client.post(
        "/api/v1/noise/testsink/start",
        json={"preset": "STRONG"},
        headers=_HEADERS,
    )
    assert start_resp.status_code == 200
    started = start_resp.json()
    assert started["is_test_sink_mode"] is True
    assert started["active_preset"] == "STRONG"

    # 3. Trigger Panic
    panic_resp = await client.post("/api/v1/noise/testsink/panic", headers=_HEADERS)
    assert panic_resp.status_code == 200
    assert panic_resp.json()["status"] == "muted"

    # 4. Stop TestSink
    stop_resp = await client.post("/api/v1/noise/testsink/stop", headers=_HEADERS)
    assert stop_resp.status_code == 200
    stopped = stop_resp.json()
    assert stopped["is_running"] is False
