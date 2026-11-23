"""Integration tests for Phase 4 audio devices, routing validation, and panic controls."""

import pytest
from httpx import ASGITransport, AsyncClient

from hth.audio.devices import FakeDeviceProvider, set_audio_device_provider
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
    set_audio_device_provider(FakeDeviceProvider())


@pytest.fixture
async def client() -> AsyncClient:  # type: ignore[override]
    get_settings.cache_clear()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_get_inventory(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/devices/inventory", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["capture_devices"]) >= 2
    assert len(data["render_devices"]) >= 2
    assert data["vb_cable_installed"] is True
    assert data["default_console_render_id"] is not None


@pytest.mark.asyncio
async def test_validate_routing_success(client: AsyncClient) -> None:
    payload = {
        "input_device_id": "{0.0.1.00000000}.{realtek-mic-guid}",
        "output_device_id": "{0.0.0.00000000}.{vbcable-in-guid}",
        "monitor_device_id": "{0.0.0.00000000}.{realtek-hp-guid}",
        "monitor_enabled": True,
    }
    resp = await client.post("/api/v1/devices/routing/validate", json=payload, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert len(data["errors"]) == 0
    assert len(data["checklist"]) >= 4


@pytest.mark.asyncio
async def test_validate_routing_blocks_feedback_loop(client: AsyncClient) -> None:
    payload = {
        "input_device_id": "{0.0.1.00000000}.{vbcable-out-guid}",  # Feedback loop!
        "output_device_id": "{0.0.0.00000000}.{vbcable-in-guid}",
    }
    resp = await client.post("/api/v1/devices/routing/validate", json=payload, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert any("FEEDBACK LOOP" in e for e in data["errors"])


@pytest.mark.asyncio
async def test_apply_routing_rejects_feedback_loop_with_400(client: AsyncClient) -> None:
    payload = {
        "input_device_id": "{0.0.1.00000000}.{vbcable-out-guid}",
        "output_device_id": "{0.0.0.00000000}.{vbcable-in-guid}",
    }
    resp = await client.post("/api/v1/devices/routing/apply", json=payload, headers=_HEADERS)
    assert resp.status_code == 400
    data = resp.json()
    assert "rejected" in data["detail"]["message"].lower()


@pytest.mark.asyncio
async def test_panic_stop(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/devices/panic-stop", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "muted"

    status_resp = await client.get("/api/v1/devices/routing/status", headers=_HEADERS)
    assert status_resp.status_code == 200
    assert status_resp.json()["is_muted"] is True


@pytest.mark.asyncio
async def test_passthrough_toggle(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/devices/passthrough?enabled=true", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_passthrough"] is True
    assert "PASSTHROUGH ACTIVE" in data["warning"]


@pytest.mark.asyncio
async def test_test_record_endpoint(client: AsyncClient) -> None:
    payload = {"duration_seconds": 5.0}
    resp = await client.post("/api/v1/devices/test-record", json=payload, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recording_id"]
    assert data["duration_seconds"] == 5.0
    assert "retained locally" in data["retention_notice"]


@pytest.mark.asyncio
async def test_get_hotkeys(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/devices/hotkeys", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["panic_hotkey"] == "Ctrl+Shift+X"
    assert data["passthrough_hotkey"] == "Ctrl+Shift+P"
