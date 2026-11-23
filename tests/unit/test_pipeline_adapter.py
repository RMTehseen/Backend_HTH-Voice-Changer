"""Unit tests for AudioPipelineAdapter (Phase B Production VoiceEngineProtocol Adapter)."""

from unittest.mock import MagicMock

import pytest

from hth.adapters.pipeline_adapter import AudioPipelineAdapter
from hth.audio.conversion.contracts import VoiceProfile
from hth.audio.pipeline import AudioPipeline, AudioPipelineConfig, AudioPipelineMetrics
from hth.domain.api_contracts import EngineState, LiveConfig
from hth.domain.errors import DeviceError


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Mock AudioPipeline instance for deterministic lifecycle testing."""
    pipeline = MagicMock(spec=AudioPipeline)
    pipeline.is_running = False
    pipeline.is_muted = False
    pipeline.converter = MagicMock()
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
        processing_time_p50_ms=0.45,
        processing_time_p95_ms=1.25,
        underrun_count=0,
        render_frames_total=120,
    )
    return pipeline


@pytest.fixture
def adapter(mock_pipeline: MagicMock) -> AudioPipelineAdapter:
    return AudioPipelineAdapter(pipeline=mock_pipeline)


@pytest.mark.asyncio
async def test_ping(adapter: AudioPipelineAdapter) -> None:
    assert await adapter.ping() is True


@pytest.mark.asyncio
async def test_initial_state_idle(adapter: AudioPipelineAdapter) -> None:
    status = await adapter.status()
    assert status.state == EngineState.IDLE
    assert status.frames_processed == 120


@pytest.mark.asyncio
async def test_start_stop_cycle(adapter: AudioPipelineAdapter, mock_pipeline: MagicMock) -> None:
    config = LiveConfig(
        sample_rate=48000, buffer_size=480, input_device_id="18", output_device_id="14"
    )
    await adapter.start(config)

    assert mock_pipeline.is_running is True
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=False,
        processing_time_p50_ms=0.42,
        processing_time_p95_ms=1.10,
        underrun_count=0,
        render_frames_total=500,
    )

    status = await adapter.status()
    assert status.state == EngineState.RUNNING
    assert status.latency_p50_ms == 0.42

    await adapter.stop()
    assert mock_pipeline.is_running is False

    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=False,
        is_muted=False,
    )
    status = await adapter.status()
    assert status.state == EngineState.IDLE


@pytest.mark.asyncio
async def test_idempotent_double_start(
    adapter: AudioPipelineAdapter, mock_pipeline: MagicMock
) -> None:
    config = LiveConfig()
    await adapter.start(config)
    assert mock_pipeline.is_running is True
    assert mock_pipeline.start.call_count == 1

    # Second start call should be safely ignored (no duplicate streams)
    await adapter.start(config)
    assert mock_pipeline.start.call_count == 1


@pytest.mark.asyncio
async def test_idempotent_double_stop(
    adapter: AudioPipelineAdapter, mock_pipeline: MagicMock
) -> None:
    # Stop when idle does nothing
    await adapter.stop()
    assert mock_pipeline.stop.call_count == 0

    await adapter.start(LiveConfig())
    await adapter.stop()
    assert mock_pipeline.stop.call_count == 1

    # Second stop when already idle does not raise
    await adapter.stop()
    assert mock_pipeline.stop.call_count == 1


@pytest.mark.asyncio
async def test_start_failure_fail_closed(
    adapter: AudioPipelineAdapter, mock_pipeline: MagicMock
) -> None:
    mock_pipeline.start.side_effect = RuntimeError("WASAPI Device Disconnected")

    with pytest.raises(DeviceError, match="Failed to start live DSP audio pipeline"):
        await adapter.start(LiveConfig())

    health = await adapter.health()
    assert health.state == EngineState.ERROR
    mock_pipeline.stop.assert_called_once()


@pytest.mark.asyncio
async def test_panic_mute_and_unmute(
    adapter: AudioPipelineAdapter, mock_pipeline: MagicMock
) -> None:
    await adapter.start(LiveConfig())
    await adapter.panic_mute()
    mock_pipeline.panic_mute.assert_called_once()

    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=True,
    )
    status = await adapter.status()
    assert status.state == EngineState.MUTED

    # Unmute restores RUNNING state
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=True,
        is_muted=False,
    )
    await adapter.un_mute()
    mock_pipeline.un_mute.assert_called_once()
    status = await adapter.status()
    assert status.state == EngineState.RUNNING

    # Repeated unmute is safe and idempotent
    await adapter.un_mute()
    assert mock_pipeline.un_mute.call_count == 2
    status = await adapter.status()
    assert status.state == EngineState.RUNNING

    # Unmute when stopped remains IDLE
    await adapter.stop()
    mock_pipeline.get_metrics.return_value = AudioPipelineMetrics(
        is_running=False,
        is_muted=False,
    )
    await adapter.un_mute()
    status = await adapter.status()
    assert status.state == EngineState.IDLE


@pytest.mark.asyncio
async def test_passthrough_strictly_prohibited(adapter: AudioPipelineAdapter) -> None:
    # Rule A-03: Raw passthrough must raise in production Phase B
    with pytest.raises(RuntimeError, match="Raw microphone passthrough is disabled"):
        await adapter.set_passthrough(True)

    # Disabling passthrough is safe
    await adapter.set_passthrough(False)


@pytest.mark.asyncio
async def test_select_model_and_set_params(
    adapter: AudioPipelineAdapter, mock_pipeline: MagicMock
) -> None:
    await adapter.select_model("female_to_male_v1")
    mock_pipeline.set_conversion_profile.assert_called_with(VoiceProfile.FEMALE_TO_MALE)

    await adapter.select_model("male_to_female_v1")
    mock_pipeline.set_conversion_profile.assert_called_with(VoiceProfile.MALE_TO_FEMALE)

    await adapter.set_params({"profile": "to_male", "pitch_shift": -12.0, "formant_shift": 0.85})
    mock_pipeline.set_conversion_profile.assert_called_with(VoiceProfile.FEMALE_TO_MALE)


@pytest.mark.asyncio
async def test_health_reports_truthful_dsp_capabilities(adapter: AudioPipelineAdapter) -> None:
    info = await adapter.health()
    assert info.engine_id == "real-dsp-pipeline"
    assert "rnnoise_noise_suppression" in info.capabilities
    assert "acoustic_pitch_formant_dsp" in info.capabilities
    assert "fail_closed_mute" in info.capabilities
    assert "vb_cable_routing" in info.capabilities


@pytest.mark.asyncio
async def test_list_devices(adapter: AudioPipelineAdapter) -> None:
    devices = await adapter.list_devices()
    assert len(devices) >= 2
    assert any(d.device_type.value in ("INPUT", "input") for d in devices)
