"""Integration tests for Voice Note Conversion REST API endpoints (Phase 9)."""

import io
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from hth.core.app import create_app
from hth.core.settings import get_settings

_TOKEN = "test-token-fixed-32-chars-abc12345"
_HEADERS = {"X-HTH-Token": _TOKEN}


def _create_wav_data(duration_sec: float = 0.2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        nframes = int(16000 * duration_sec)
        wf.writeframes(struct.pack(f"<{nframes}h", *([500] * nframes)))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _patch_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("HTH_SESSION_TOKEN", _TOKEN)
    monkeypatch.setenv("HTH_DB_PATH", ":memory:")
    monkeypatch.setenv("HTH_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("HTH_VOICE_NOTES_DIR", str(tmp_path / "voice_notes"))


@pytest.fixture
async def api_client() -> AsyncClient:
    get_settings.cache_clear()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_api_live_guide(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/voice-notes/live-guide", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended_whatsapp_input"] == "CABLE Output (VB-Audio Virtual Cable)"
    assert len(data["steps"]) == 6
    assert "attachment_warning" in data


@pytest.mark.asyncio
async def test_api_capabilities(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/voice-notes/capabilities", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "wav" in data["supported_output_formats"]
    assert data["max_file_size_mb"] == 25


@pytest.mark.asyncio
async def test_api_convert_and_retrieve(api_client: AsyncClient):
    wav_bytes = _create_wav_data(0.2)

    files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
    form_data = {
        "target_format": "wav",
        "model_id": "test_model",
        "model_format": "fake_model",
        "pitch_shift": "0.0",
        "formant_shift": "0.0",
    }

    resp = await api_client.post(
        "/api/v1/voice-notes/convert",
        files=files,
        data=form_data,
        headers=_HEADERS,
    )
    assert resp.status_code == 202
    job = resp.json()
    job_id = job["job_id"]
    assert job["input_format"] == "wav"
    assert job["output_format"] == "wav"

    # Query status
    status_resp = await api_client.get(
        f"/api/v1/voice-notes/jobs/{job_id}",
        headers=_HEADERS,
    )
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_api_cancel_job(api_client: AsyncClient):
    wav_bytes = _create_wav_data(1.0)
    files = {"file": ("cancel_test.wav", wav_bytes, "audio/wav")}
    form_data = {
        "target_format": "wav",
        "model_id": "test_model",
        "model_format": "fake_model",
    }

    resp = await api_client.post(
        "/api/v1/voice-notes/convert",
        files=files,
        data=form_data,
        headers=_HEADERS,
    )
    job_id = resp.json()["job_id"]

    # DELETE /jobs/{job_id}
    del_resp = await api_client.delete(
        f"/api/v1/voice-notes/jobs/{job_id}",
        headers=_HEADERS,
    )
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_api_cleanup_retention(api_client: AsyncClient):
    resp = await api_client.post(
        "/api/v1/voice-notes/cleanup?retention_hours=24",
        headers=_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "deleted_temp_files" in data
    assert "freed_bytes" in data


@pytest.mark.asyncio
async def test_api_reject_unsupported_format(api_client: AsyncClient):
    wav_bytes = _create_wav_data(0.1)
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    form_data = {
        "target_format": "unsupported_codec_xyz",
        "model_id": "test_model",
        "model_format": "fake_model",
    }
    resp = await api_client.post(
        "/api/v1/voice-notes/convert",
        files=files,
        data=form_data,
        headers=_HEADERS,
    )
    assert resp.status_code == 400
    assert "Unsupported target format" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_reject_invalid_audio_file(api_client: AsyncClient):
    corrupt_bytes = b"NOT_A_VALID_AUDIO_HEADER_BYTES"
    files = {"file": ("corrupt.wav", corrupt_bytes, "audio/wav")}
    form_data = {
        "target_format": "wav",
        "model_id": "test_model",
        "model_format": "fake_model",
    }
    resp = await api_client.post(
        "/api/v1/voice-notes/convert",
        files=files,
        data=form_data,
        headers=_HEADERS,
    )
    assert resp.status_code == 400
    assert "does not contain valid audio header magic bytes" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_reject_incompatible_model(api_client: AsyncClient):
    wav_bytes = _create_wav_data(0.1)
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    form_data = {
        "target_format": "wav",
        "model_id": "rvc_model",
        "model_format": "applio_rvc",  # Incompatible with fake converter
    }
    resp = await api_client.post(
        "/api/v1/voice-notes/convert",
        files=files,
        data=form_data,
        headers=_HEADERS,
    )
    assert resp.status_code == 422
    assert "is incompatible with engine" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_stream_audio_cookie_auth(api_client: AsyncClient):
    """Verifies that voice-note audio streaming succeeds with HttpOnly session cookie."""
    # 1. Create a job
    wav_bytes = _create_wav_data(0.1)
    files = {"file": ("test_stream.wav", wav_bytes, "audio/wav")}
    form_data = {
        "target_format": "wav",
        "model_id": "test_model",
        "model_format": "fake_model",
    }
    resp = await api_client.post(
        "/api/v1/voice-notes/convert",
        files=files,
        data=form_data,
        headers=_HEADERS,
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # 2. Access source-audio using cookie only (no Authorization or X-HTH-Token header)
    api_client.cookies.set("hth_session", _TOKEN)
    stream_resp = await api_client.get(
        f"/api/v1/voice-notes/jobs/{job_id}/source-audio",
        headers={"Host": "127.0.0.1:8765"},
    )
    assert stream_resp.status_code == 200
    assert len(stream_resp.content) > 0


@pytest.mark.asyncio
async def test_api_stream_audio_unauthenticated_rejected(api_client: AsyncClient):
    """Requests without cookie or token receive 401 Unauthorized."""
    resp = await api_client.get(
        "/api/v1/voice-notes/jobs/fake-job-id/source-audio",
        headers={"Host": "127.0.0.1:8765"},
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_api_stream_audio_query_token_rejected(api_client: AsyncClient):
    """Query parameter ?token=... is rejected; must use HttpOnly cookie or header."""
    resp = await api_client.get(
        f"/api/v1/voice-notes/jobs/fake-job-id/source-audio?token={_TOKEN}",
        headers={"Host": "127.0.0.1:8765"},
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"
