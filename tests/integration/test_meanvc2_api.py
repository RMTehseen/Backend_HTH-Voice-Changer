"""Integration tests for MeanVC2 REST API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

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
    yield
    get_settings.cache_clear()


@pytest.fixture
async def client() -> AsyncClient:
    get_settings.cache_clear()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_get_meanvc2_status(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/meanvc2/status", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "asset_status" in data
    assert "profiles" in data
    assert "model_variant" in data
    assert data["model_variant"] in ("40ms", "120ms")


@pytest.mark.asyncio
async def test_get_meanvc2_assets(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/meanvc2/assets", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "assets" in data
    assert len(data["assets"]) > 0
    asset_names = [a["name"] for a in data["assets"]]
    assert "meanvc2_40ms_40ms.safetensors" in asset_names
    assert "vocos.pt" in asset_names
    assert "wavlm_large_finetune.pth" in asset_names


@pytest.mark.asyncio
async def test_list_meanvc2_profiles(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/meanvc2/profiles", headers=_HEADERS)
    assert resp.status_code == 200
    profiles = resp.json()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1
    profile_ids = [p["profile_id"] for p in profiles]
    assert "profile-female-eva" in profile_ids


@pytest.mark.asyncio
async def test_set_model_variant(client: AsyncClient) -> None:
    # Valid
    resp = await client.post("/api/v1/meanvc2/variant", json={"variant": "40ms"}, headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["variant"] == "40ms"

    # Invalid
    resp_bad = await client.post("/api/v1/meanvc2/variant", json={"variant": "999ms"}, headers=_HEADERS)
    assert resp_bad.status_code == 400


@pytest.mark.asyncio
async def test_select_voice_profile(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/meanvc2/profiles/profile-female-eva/select", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["selected"] == "profile-female-eva"


@pytest.mark.asyncio
async def test_update_voice_profile(client: AsyncClient) -> None:
    updates = {
        "volume_gain": 1.25,
        "pitch_shift": 2.5,
        "timbre_ratio": 1.05,
        "sound_effect": "room",
        "favorite": True,
    }
    resp = await client.put("/api/v1/meanvc2/profiles/profile-female-eva", json=updates, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["volume_gain"] == 1.25
    assert data["pitch_shift"] == 2.5
    assert data["timbre_ratio"] == 1.05
    assert data["sound_effect"] == "room"
    assert data["favorite"] is True


@pytest.mark.asyncio
async def test_monitor_control(client: AsyncClient) -> None:
    # Enable monitor
    resp = await client.post("/api/v1/meanvc2/monitor", json={"enabled": True, "volume": 0.8}, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["monitor_enabled"] is True

    # Disable monitor
    resp2 = await client.post("/api/v1/meanvc2/monitor", json={"enabled": False}, headers=_HEADERS)
    assert resp2.status_code == 200
    assert resp2.json()["monitor_enabled"] is False


@pytest.mark.asyncio
async def test_volume_control(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/meanvc2/volume", json={"volume": 1.4}, headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["output_volume"] == 1.4


@pytest.mark.asyncio
async def test_stream_mute(client: AsyncClient) -> None:
    # Engage panic mute
    resp = await client.post("/api/v1/meanvc2/stream/mute", json={"muted": True}, headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["is_muted"] is True

    # Release mute
    resp2 = await client.post("/api/v1/meanvc2/stream/mute", json={"muted": False}, headers=_HEADERS)
    assert resp2.status_code == 200
    assert resp2.json()["is_muted"] is False


@pytest.mark.asyncio
async def test_get_profile_reference_audio_and_path_jail(client: AsyncClient, tmp_path) -> None:
    """GET /profiles/{id}/reference enforces ID format, path jail, and returns WAV bytes."""
    import wave

    import numpy as np

    from hth.adapters.meanvc2.adapter import MeanVC2VoiceAdapter
    from hth.adapters.meanvc2.voice_profile import VoiceProfileStore
    from hth.api.routers.meanvc2 import _get_meanvc2_adapter

    # Invalid profile_id -> 400
    resp_bad = await client.get("/api/v1/meanvc2/profiles/bad!id/reference", headers=_HEADERS)
    assert resp_bad.status_code == 400

    profiles_dir = tmp_path / "profiles"
    adapter = MeanVC2VoiceAdapter()
    adapter._profile_store = VoiceProfileStore(str(profiles_dir), auto_seed=True)

    app = client._transport.app  # type: ignore[attr-defined]
    app.dependency_overrides[_get_meanvc2_adapter] = lambda: adapter
    try:
        # No reference WAV attached yet -> 404
        resp_missing = await client.get(
            "/api/v1/meanvc2/profiles/profile-female-eva/reference", headers=_HEADERS
        )
        assert resp_missing.status_code == 404

        # Path outside profile_dir (jail bypass attempt) -> 404
        outside_wav = tmp_path / "outside.wav"
        outside_wav.write_bytes(b"RIFFdemo")
        prof = adapter._profile_store.get("profile-female-eva")
        assert prof is not None
        prof.reference_wav_path = str(outside_wav)
        adapter._profile_store.save(prof)

        resp_jail = await client.get(
            "/api/v1/meanvc2/profiles/profile-female-eva/reference", headers=_HEADERS
        )
        assert resp_jail.status_code == 404

        # Valid WAV inside profile_dir -> 200 audio/wav
        inside_dir = profiles_dir / "references"
        inside_dir.mkdir(parents=True, exist_ok=True)
        inside_wav = inside_dir / "profile-female-eva.wav"
        pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
        with wave.open(str(inside_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        prof.reference_wav_path = str(inside_wav)
        adapter._profile_store.save(prof)

        resp_ok = await client.get(
            "/api/v1/meanvc2/profiles/profile-female-eva/reference", headers=_HEADERS
        )
        assert resp_ok.status_code == 200
        assert resp_ok.headers["content-type"].startswith("audio/wav")
        assert resp_ok.content.startswith(b"RIFF")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_set_meanvc2_processing_mode(client: AsyncClient) -> None:
    """POST /mode switches cleanly between pure_neural, hybrid, and bypass."""
    from hth.adapters.meanvc2.adapter import MeanVC2VoiceAdapter
    from hth.api.routers.meanvc2 import _get_meanvc2_adapter

    adapter = MeanVC2VoiceAdapter()
    app = client._transport.app  # type: ignore[attr-defined]
    app.dependency_overrides[_get_meanvc2_adapter] = lambda: adapter
    try:
        for mode in ("hybrid", "bypass", "pure_neural"):
            resp = await client.post("/api/v1/meanvc2/mode", json={"mode": mode}, headers=_HEADERS)
            assert resp.status_code == 200
            assert resp.json()["processing_mode"] == mode

        status_resp = await client.get("/api/v1/meanvc2/status", headers=_HEADERS)
        assert status_resp.status_code == 200
        assert status_resp.json()["processing_mode"] == "pure_neural"

        bad_resp = await client.post("/api/v1/meanvc2/mode", json={"mode": "invalid"}, headers=_HEADERS)
        assert bad_resp.status_code == 400
    finally:
        app.dependency_overrides.clear()



