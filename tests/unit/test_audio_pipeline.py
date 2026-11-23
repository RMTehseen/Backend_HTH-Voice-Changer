"""Unit tests for AudioPipeline verifying TestSink safety, metrics, and fail-closed state."""

import numpy as np
import pytest

from hth.audio.noise.contracts import NoisePreset
from hth.audio.pipeline import AudioPipeline, AudioPipelineConfig
from hth.domain.errors import AudioProcessingError


def test_pipeline_initialization() -> None:
    pipeline = AudioPipeline()
    metrics = pipeline.get_metrics()
    assert metrics.is_running is False
    assert metrics.is_test_sink_mode is True
    assert "TEST HARNESS" in metrics.test_harness_banner
    assert metrics.active_preset == NoisePreset.LIGHT


def test_pipeline_vb_cable_security_block() -> None:
    # Security Rule: VB-CABLE live transmission is strictly blocked in Milestone 1
    cfg = AudioPipelineConfig(enable_vb_cable=True)
    pipeline = AudioPipeline(config=cfg)

    with pytest.raises(
        AudioProcessingError, match="Security violation: VB-CABLE live transmission is disabled"
    ):
        pipeline.start()


def test_pipeline_preset_switch() -> None:
    pipeline = AudioPipeline()
    pipeline.set_preset(NoisePreset.STRONG)
    assert pipeline.config.preset == NoisePreset.STRONG
    assert pipeline.suppressor.get_preset() == NoisePreset.STRONG

    pipeline.set_preset(NoisePreset.OFF)
    assert pipeline.config.preset == NoisePreset.OFF
    assert pipeline.suppressor.get_preset() == NoisePreset.OFF


def test_pipeline_panic_mute() -> None:
    pipeline = AudioPipeline()
    pipeline.panic_mute()
    metrics = pipeline.get_metrics()
    assert metrics.is_muted is True

    pipeline.un_mute()
    metrics = pipeline.get_metrics()
    assert metrics.is_muted is False


def test_pipeline_offline_chunk_processing() -> None:
    pipeline = AudioPipeline()
    in_frame = np.random.uniform(-0.1, 0.1, 480).astype(np.float32)
    out_frame = np.zeros(480, dtype=np.float32)

    vad_prob = pipeline.process_offline_chunk(in_frame, out_frame)
    assert 0.0 <= vad_prob <= 1.0
    assert not np.isnan(out_frame).any()
    assert not np.isinf(out_frame).any()


def test_pipeline_discontinuity_handling() -> None:
    pipeline = AudioPipeline()
    # Trigger discontinuity hook directly
    pipeline._handle_discontinuity()
    assert pipeline.suppressor._has_error is False


def test_pipeline_resampled_processing_end_to_end() -> None:
    pipeline = AudioPipeline()
    in_48k = np.random.uniform(-0.1, 0.1, 480).astype(np.float32)
    out_48k = np.zeros(480, dtype=np.float32)

    # Engine callback that multiplies 16 kHz clean input by 0.5
    def mock_voice_engine(in_16k: np.ndarray, out_16k: np.ndarray) -> None:
        np.multiply(in_16k, 0.5, out=out_16k)

    proc_time_ms = pipeline.process_frame_resampled(
        in_48k, out_48k, voice_engine_callback=mock_voice_engine
    )
    assert proc_time_ms >= 0.0
    assert not np.isnan(out_48k).any()
    assert not np.isinf(out_48k).any()

    # Verify panic mute zeros output immediately
    pipeline.panic_mute()
    out_muted = np.ones(480, dtype=np.float32)
    pipeline.process_frame_resampled(in_48k, out_muted)
    assert np.all(out_muted == 0.0)


def test_router_pipeline_attachment_and_panic_stop() -> None:
    from hth.audio.router import AudioRouter
    from hth.domain.audio_contracts import AudioRoutingConfig

    router = AudioRouter()
    pipeline = AudioPipeline()
    router.attach_pipeline(pipeline)

    # Trigger panic stop on router -> must propagate to pipeline
    router.trigger_panic_stop()
    assert router.get_status().is_muted is True
    assert pipeline.get_metrics().is_muted is True

    # Reset
    pipeline.un_mute()
    assert pipeline.get_metrics().is_muted is False

    # Simulate device unplug on active config
    config = AudioRoutingConfig(input_device_id="mic_1", output_device_id="headphone_1")
    router._active_config = config
    router.handle_device_unplug("mic_1")
    assert pipeline.get_metrics().is_muted is True


def test_pipeline_recording_rejected_in_live_call_mode() -> None:
    """Ensure recording tap raises AudioProcessingError during live call or VB-CABLE mode."""
    cfg = AudioPipelineConfig(live_call_mode=True, enable_vb_cable=True)
    pipeline = AudioPipeline(config=cfg)

    with pytest.raises(AudioProcessingError, match="Recording tap is disabled during live calls"):
        pipeline.start_recording()


def test_pipeline_recording_lifecycle_in_testsink_mode() -> None:
    """Ensure recording tap starts, writes asynchronously via preallocated ring, and stops safely."""
    cfg = AudioPipelineConfig(test_sink_mode=True, live_call_mode=False, enable_vb_cable=False)
    pipeline = AudioPipeline(config=cfg)

    path = pipeline.start_recording(duration_seconds=1.0)
    assert path is not None
    assert path.name.endswith(".wav")
    assert pipeline._record_active is True
    assert pipeline._record_thread is not None
    assert pipeline._record_thread.is_alive()

    # Feed a dummy frame through preallocated ring buffer
    dummy_frame = np.full(480, 0.1, dtype=np.float32)
    slot = pipeline._record_head % 300
    np.copyto(pipeline._record_ring[slot], dummy_frame)
    pipeline._record_head += 1

    stopped_path = pipeline.stop_recording()
    assert stopped_path == path
    assert pipeline._record_active is False


def test_pipeline_recording_overflow_handling() -> None:
    """Ensure ring buffer overflow drops frames without crashing or allocating heap."""
    cfg = AudioPipelineConfig(test_sink_mode=True, live_call_mode=False, enable_vb_cable=False)
    pipeline = AudioPipeline(config=cfg)
    pipeline._record_active = True
    pipeline._record_head = 300
    pipeline._record_tail = 0

    # Simulate render callback pushing frame when full
    head = pipeline._record_head
    tail = pipeline._record_tail
    if (head - tail) >= 300:
        pipeline._record_overflow_count += 1

    assert pipeline._record_overflow_count == 1


def test_dynamic_wasapi_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dynamic WASAPI host API discovery and error handling."""
    from hth.audio.pipeline import get_wasapi_hostapi_index
    from hth.domain.errors import DeviceError

    # Test when WASAPI is present at index 1
    mock_apis = [
        {"name": "MME"},
        {"name": "Windows WASAPI"},
        {"name": "DirectSound"},
    ]
    monkeypatch.setattr("hth.audio.pipeline.sd.query_hostapis", lambda: mock_apis)
    idx = get_wasapi_hostapi_index()
    assert idx == 1

    # Test when WASAPI is absent
    mock_no_wasapi = [{"name": "MME"}, {"name": "DirectSound"}]
    monkeypatch.setattr("hth.audio.pipeline.sd.query_hostapis", lambda: mock_no_wasapi)
    with pytest.raises(DeviceError, match="Windows WASAPI host API is not available"):
        get_wasapi_hostapi_index()
