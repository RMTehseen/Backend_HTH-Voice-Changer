"""Integration tests for the FastAPI application.

Uses httpx.AsyncClient with ASGITransport so no real network is involved.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from hth.core.app import create_app
from hth.core.settings import get_settings

# A fixed token used throughout these tests
_TOKEN = "test-token-fixed-32-chars-abc12345"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override settings singleton with a test-friendly instance."""
    from hth.api.dependencies import init_engine_service

    get_settings.cache_clear()
    monkeypatch.setenv("HTH_SESSION_TOKEN", _TOKEN)
    monkeypatch.setenv("HTH_DB_PATH", ":memory:")
    monkeypatch.setenv("HTH_LOG_LEVEL", "WARNING")
    init_engine_service("fake")


@pytest.fixture
async def client() -> AsyncClient:  # type: ignore[override]
    get_settings.cache_clear()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as ac:
        yield ac


_HEADERS = {"X-HTH-Token": _TOKEN}


@pytest.mark.asyncio
async def test_health_no_token(client: AsyncClient) -> None:
    """GET /health must be reachable without a token."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_with_token(client: AsyncClient) -> None:
    resp = await client.get("/health", headers=_HEADERS)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_missing_token_returns_401(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/engine/status")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_token_returns_401(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/engine/status", headers={"X-HTH-Token": "wrong-token"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_engine_status(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/engine/status", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "IDLE"


@pytest.mark.asyncio
async def test_engine_start_stop(client: AsyncClient) -> None:
    start = await client.post(
        "/api/v1/engine/start", headers=_HEADERS, json={"sample_rate": 48000, "buffer_size": 512}
    )
    assert start.status_code == 200
    assert start.json()["state"] == "RUNNING"

    stop = await client.post("/api/v1/engine/stop", headers=_HEADERS)
    assert stop.status_code == 200
    assert stop.json()["ok"] is True

    # Idempotent stop when already IDLE must return 200 OK (not 409 Conflict)
    stop_again = await client.post("/api/v1/engine/stop", headers=_HEADERS)
    assert stop_again.status_code == 200
    assert stop_again.json()["ok"] is True
    assert stop_again.json()["state"] == "IDLE"


@pytest.mark.asyncio
async def test_engine_double_start_409(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/engine/start", headers=_HEADERS, json={"sample_rate": 48000, "buffer_size": 512}
    )
    resp = await client.post(
        "/api/v1/engine/start", headers=_HEADERS, json={"sample_rate": 48000, "buffer_size": 512}
    )
    assert resp.status_code == 409
    await client.post("/api/v1/engine/stop", headers=_HEADERS)


@pytest.mark.asyncio
async def test_passthrough_blocked_when_idle(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/engine/passthrough", headers=_HEADERS, json={"enabled": True})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_devices_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/devices/", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2  # FakeEngine returns 2 fake devices


@pytest.mark.asyncio
async def test_system_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/system/", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "cpu_percent" in data
    assert "ram_total_mb" in data


@pytest.mark.asyncio
async def test_settings_get_no_token(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/settings/", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_token"] == "[REDACTED]"
    assert "host" in data


@pytest.mark.asyncio
async def test_openapi_accessible(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
