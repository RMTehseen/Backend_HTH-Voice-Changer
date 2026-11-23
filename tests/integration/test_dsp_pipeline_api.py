"""Integration tests for FastAPI endpoints with production AudioPipelineAdapter."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from hth.adapters.pipeline_adapter import AudioPipelineAdapter
from hth.audio.pipeline import AudioPipeline, AudioPipelineConfig, AudioPipelineMetrics
from hth.core.app import create_app
from hth.core.settings import get_settings

_TOKEN = "test-token-fixed-32-chars-abc12345"
_HEADERS = {"X-HTH-Token": _TOKEN}


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock(spec=AudioPipeline)
    pipeline.is_running = False
    pipeline.is_muted = False
    pipeline.config = AudioPipelineConfig()

    def fake_start(cfg: AudioPipelineConfig) -> None:
        pipeline.is_running = True
        pipeline.config = cfg

    def fake_stop() -> None:
        pipeline.is_running = False

    def fake_panic() -> None:
        pipeline.is_muted = True

    def fake_unmute() -> None:
        pipeline.is_muted = False

    pipeline.start.side_effect = fake_start
    pipeline.stop.side_effect = fake_stop
    pipeline.panic_mute.side_effect = fake_panic
    pipeline.un_mute.side_effect = fake_unmute
    pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=False,
        is_muted=False,
        processing_time_p50_ms=0.38,
        processing_time_p95_ms=1.12,
        render_frames_total=150,
    )
    return pipeline


@pytest.fixture
def _patch_dsp_engine(mock_pipeline: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    from hth.services.engine_service import EngineService

    get_settings.cache_clear()
    monkeypatch.setenv("HTH_SESSION_TOKEN", _TOKEN)
    monkeypatch.setenv("HTH_DB_PATH", ":memory:")
    monkeypatch.setenv("HTH_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("HTH_ENGINE_ADAPTER", "dsp")

    adapter = AudioPipelineAdapter(pipeline=mock_pipeline)
    import hth.api.dependencies as deps

    deps._engine_service = EngineService(adapter)


@pytest.fixture
async def client(_patch_dsp_engine: None) -> AsyncClient:  # type: ignore[override]
    get_settings.cache_clear()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_dsp_engine_health(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/engine/capabilities", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine_id"] == "real-dsp-pipeline"
    assert "acoustic_pitch_formant_dsp" in data["capabilities"]
    assert "rnnoise_noise_suppression" in data["capabilities"]


@pytest.mark.asyncio
async def test_dsp_engine_lifecycle(client: AsyncClient, mock_pipeline: MagicMock) -> None:
    # 1. Check initial status
    resp = await client.get("/api/v1/engine/status", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "IDLE"

    # 2. Start engine
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=False,
        processing_time_p50_ms=0.41,
        processing_time_p95_ms=1.05,
        render_frames_total=400,
    )
    start_payload = {
        "sample_rate": 48000,
        "buffer_size": 480,
        "input_device_id": "18",
        "output_device_id": "14",
    }
    resp = await client.post("/api/v1/engine/start", json=start_payload, headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "RUNNING"
    assert resp.json()["latency_p50_ms"] == 0.41

    # 3. Trigger panic mute
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=True,
        processing_time_p50_ms=0.0,
        render_frames_total=450,
    )
    resp = await client.post("/api/v1/engine/panic-mute", headers=_HEADERS)
    assert resp.status_code == 200

    resp = await client.get("/api/v1/engine/status", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "MUTED"

    # 4. Passthrough must be blocked with 409 Conflict
    resp = await client.post("/api/v1/engine/passthrough", json={"enabled": True}, headers=_HEADERS)
    assert resp.status_code == 409
    assert "Raw microphone passthrough is disabled" in resp.json()["detail"]

    # 5. Stop engine
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=False,
        is_muted=False,
    )
    resp = await client.post("/api/v1/engine/stop", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp = await client.get("/api/v1/engine/status", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "IDLE"


@pytest.mark.asyncio
async def test_dsp_engine_unmute_lifecycle(client: AsyncClient, mock_pipeline: MagicMock) -> None:
    # 1. Start engine
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=False,
        processing_time_p50_ms=0.40,
        render_frames_total=100,
    )
    start_payload = {
        "sample_rate": 48000,
        "buffer_size": 480,
        "input_device_id": "18",
        "output_device_id": "14",
    }
    resp = await client.post("/api/v1/engine/start", json=start_payload, headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "RUNNING"

    # 2. Trigger panic mute
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=True,
        processing_time_p50_ms=0.0,
        render_frames_total=150,
    )
    resp = await client.post("/api/v1/engine/panic-mute", headers=_HEADERS)
    assert resp.status_code == 200

    resp = await client.get("/api/v1/engine/status", headers=_HEADERS)
    assert resp.json()["state"] == "MUTED"

    # 3. Unmute restores RUNNING status
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=False,
        processing_time_p50_ms=0.42,
        render_frames_total=200,
    )
    resp = await client.post("/api/v1/engine/unmute", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "RUNNING"
    mock_pipeline.un_mute.assert_called_once()

    # 4. Repeated unmute is safe and idempotent
    resp = await client.post("/api/v1/engine/unmute", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "RUNNING"
    assert mock_pipeline.un_mute.call_count == 2

    # 5. Stop engine
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=False,
        is_muted=False,
    )
    await client.post("/api/v1/engine/stop", headers=_HEADERS)

    # 6. Unmute when stopped remains IDLE
    resp = await client.post("/api/v1/engine/unmute", headers=_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["state"] == "IDLE"
