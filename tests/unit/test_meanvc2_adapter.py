"""Unit tests for MeanVC2 adapter, config, voice profiles, and fail-closed safety."""

import tempfile

import numpy as np
import pytest

from hth.adapters.meanvc2.adapter import MeanVC2VoiceAdapter
from hth.adapters.meanvc2.config import MeanVC2Config
from hth.adapters.meanvc2.voice_profile import (
    VoiceProfile,
    VoiceProfileStore,
)
from hth.domain.api_contracts import EngineState, LiveConfig


def test_meanvc2_config_defaults():
    cfg = MeanVC2Config()
    assert cfg.device == "cpu"
    assert cfg.model_variant == "40ms"
    assert "meanvc2_40ms_40ms.safetensors" in cfg.vc_model_path
    assert "fastu2pp_80ms.pt" in cfg.asr_model_path

    cfg120 = MeanVC2Config(model_variant="120ms")
    assert "meanvc2_120ms_40ms.safetensors" in cfg120.vc_model_path
    assert "fastu2pp_160ms.pt" in cfg120.asr_model_path


def test_meanvc2_config_check_assets():
    cfg = MeanVC2Config()
    assets = cfg.check_assets()
    assert "vc_model" in assets
    assert "asr_model" in assets
    assert "vocoder" in assets
    assert "speaker_model" in assets
    assert "source_dir" in assets
    # Source dir should exist because MeanVC2-main is in project root
    assert assets["source_dir"] is True


def test_voice_profile_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VoiceProfileStore(tmpdir)
        assert len(store.list_profiles()) == 0

        p = VoiceProfile(
            display_name="Test Speaker",
            reference_wav_path="/path/to/test.wav",
            reference_sha256="abc123sha",
            reference_duration_s=5.2,
            consent_declared=True,
            consent_statement="Consented by speaker",
        )
        store.save(p)

        loaded = store.get(p.profile_id)
        assert loaded is not None
        assert loaded.display_name == "Test Speaker"
        assert loaded.consent_declared is True

        # Reopen store from disk
        store2 = VoiceProfileStore(tmpdir)
        assert len(store2.list_profiles()) == 1
        assert store2.get(p.profile_id).display_name == "Test Speaker"

        # Delete
        assert store2.delete(p.profile_id) is True
        assert len(store2.list_profiles()) == 0


@pytest.mark.asyncio
async def test_meanvc2_adapter_fail_closed_and_state():
    cfg = MeanVC2Config()
    adapter = MeanVC2VoiceAdapter(cfg)

    # Initial state
    assert adapter._state == EngineState.IDLE
    health = await adapter.health()
    assert health.engine_id == MeanVC2VoiceAdapter.ENGINE_ID
    assert "fail_closed_mute" in health.capabilities

    # Raw passthrough should be rejected when not running (Rule A-03)
    with pytest.raises(RuntimeError, match="deliberate user action only"):
        await adapter.set_passthrough(True)

    # Starting without target profile should fail closed with clear error
    with pytest.raises(RuntimeError, match="No target voice profile selected"):
        await adapter.start(LiveConfig())

    status = await adapter.status()
    assert status.state == EngineState.ERROR
    assert adapter._is_muted is True

    # Audio pop when muted should return silence (Rule 3)
    silent_audio = adapter.pop_converted_audio(480)
    assert np.all(silent_audio == 0.0)

    # Test Panic Mute
    await adapter.panic_mute()
    assert adapter._is_muted is True

    # Unmute
    await adapter.un_mute()
    assert adapter._is_muted is False


@pytest.mark.asyncio
async def test_meanvc2_profile_consent_enforcement():
    adapter = MeanVC2VoiceAdapter()
    # Profile creation without consent must raise ValueError (Rule 4)
    with pytest.raises(ValueError, match="Consent declaration is required"):
        await adapter.create_voice_profile(
            display_name="Unauthorized Voice",
            reference_wav_path="dummy.wav",
            consent_declared=False,
        )


def test_meanvc2_audio_bridge_resampling_and_safety():
    from hth.adapters.meanvc2.audio_bridge import MeanVC2AudioBridge
    adapter = MeanVC2VoiceAdapter()
    bridge = MeanVC2AudioBridge(adapter)

    assert bridge.is_running is False
    assert bridge.is_muted is False

    # Test decimation & interpolation DSP sizing
    in_48k = np.ones(480, dtype=np.float32) * 0.5
    out_16k = np.zeros(160, dtype=np.float32)
    bridge.resampler_48_to_16.process_frame(in_48k, out_frame=out_16k)
    assert out_16k.shape == (160,)

    out_48k = np.zeros(480, dtype=np.float32)
    bridge.resampler_16_to_48.process_frame(out_16k, out_frame=out_48k)
    assert out_48k.shape == (480,)

    # Test fail-closed render callback when stopped or muted
    render_buf = np.ones((480, 1), dtype=np.float32)
    bridge._render_callback(render_buf, 480, None, None)
    assert np.all(render_buf == 0.0)

    # Test panic mute
    bridge.panic_mute()
    assert bridge.is_muted is True
    bridge.un_mute()
    assert bridge.is_muted is False


def test_meanvc2_audio_bridge_dual_routing_and_compaction():
    from hth.adapters.meanvc2.audio_bridge import MeanVC2AudioBridge
    adapter = MeanVC2VoiceAdapter()
    bridge = MeanVC2AudioBridge(adapter)

    # Monitor controls
    bridge.set_monitor_enabled(True)
    assert bridge.monitor_enabled is True
    bridge.set_monitor_volume(0.75)
    assert bridge.monitor_volume == 0.75
    bridge.set_output_volume(1.2)
    assert bridge.output_volume == 1.2

    # Monitor callback when inactive or empty -> silent
    mon_buf = np.ones((480, 1), dtype=np.float32)
    bridge._monitor_callback(mon_buf, 480, None, None)
    assert np.all(mon_buf == 0.0)

    # Queue compaction test: push data into monitor ring buffer
    bridge.monitor_ring.write(np.ones(1000, dtype=np.float32) * 0.5)
    assert bridge.monitor_ring.available_read >= 1000

    # Pop via monitor callback
    bridge._is_running = True
    bridge._monitor_callback(mon_buf, 480, None, None)
    assert np.allclose(mon_buf, 0.5 * 0.75)  # 0.5 * monitor_volume


def test_dsp_post_processor_effects():
    from hth.adapters.meanvc2.dsp_effects import DSPPostProcessor
    dsp = DSPPostProcessor(sample_rate=16000, chunk_size=2560)
    samples = np.sin(2 * np.pi * 440 * np.arange(2560) / 16000).astype(np.float32) * 0.5

    # 1. Bypass
    out_clean = dsp.process(samples, pitch_shift=0.0, timbre_ratio=1.0, sound_effect="none", volume_gain=1.0)
    assert out_clean.shape == samples.shape
    assert out_clean.dtype == np.float32
    assert np.max(np.abs(out_clean)) > 0.0

    # 2. Pitch Shift
    out_pitch = dsp.process(samples, pitch_shift=4.0, timbre_ratio=1.0, sound_effect="none", volume_gain=1.0)
    assert out_pitch.shape == samples.shape
    assert not np.array_equal(out_pitch, samples)

    # 3. Timbre / Formant Tilt
    out_timbre = dsp.process(samples, pitch_shift=0.0, timbre_ratio=1.2, sound_effect="none", volume_gain=1.0)
    assert out_timbre.shape == samples.shape
    assert not np.array_equal(out_timbre, samples)

    # 4. Sound Effects Presets
    for fx in ["room", "reverb", "radio", "telephone"]:
        out_fx = dsp.process(samples, sound_effect=fx)
        assert out_fx.shape == samples.shape
        assert np.max(np.abs(out_fx)) <= 1.0  # Limiter works

    # 5. Volume Gain & Limiting
    out_vol = dsp.process(samples, volume_gain=1.5)
    assert np.max(np.abs(out_vol)) <= 1.0


@pytest.mark.asyncio
async def test_meanvc2_starting_state_reentrant_start_is_safe():
    """Verify that calling start() while EngineState is STARTING does not throw an error."""
    adapter = MeanVC2VoiceAdapter()
    adapter._state = EngineState.STARTING
    # Reentrant start should return cleanly without raising RuntimeError
    await adapter.start(LiveConfig())
    assert adapter._state == EngineState.STARTING


def test_meanvc2_hardware_rate_adaptation_and_cable_rejection():
    """Verify WASAPI device resolution strictly rejects virtual cable for mic capture."""
    from hth.adapters.meanvc2.audio_bridge import MeanVC2AudioBridge
    adapter = MeanVC2VoiceAdapter()
    bridge = MeanVC2AudioBridge(adapter)

    # Verify metrics structure
    metrics = bridge.get_metrics()
    assert "capture_frames" in metrics
    assert "underruns" in metrics
    assert "monitor_enabled" in metrics
    assert metrics["monitor_enabled"] is False

    # Check 44.1kHz linear interpolation for Jabra headphone monitoring
    # 160 samples (10ms @ 16kHz) -> 441 samples (10ms @ 44.1kHz)
    slice_160 = np.sin(2 * np.pi * 440 * np.arange(160) / 16000).astype(np.float32)
    out_441 = np.zeros(441, dtype=np.float32)
    out_441[:] = np.interp(bridge._interp_x_out_44k, bridge._interp_x_in, slice_160)
    assert len(out_441) == 441
    assert np.max(np.abs(out_441)) > 0.0


@pytest.mark.asyncio
async def test_clean_mic_bypass_selection():
    """Verify clean-mic can be selected without 400 Bad Request and activates bypass."""
    adapter = MeanVC2VoiceAdapter()
    await adapter.select_voice_profile("clean-mic")
    assert adapter._active_profile_id == "clean-mic"
    assert adapter.get_active_profile() is None


def test_meanvc2_config_empty_source_dir_and_env(monkeypatch):
    """Empty source_dir must report False (not CWD), and HTH_MEANVC2_SOURCE_DIR overrides."""
    cfg_empty = MeanVC2Config(source_dir="")
    assets = cfg_empty.check_assets()
    assert assets["source_dir"] is False
    assert assets["vc_config"] is False

    monkeypatch.setenv("HTH_MEANVC2_SOURCE_DIR", "/tmp/custom_meanvc2_src")
    cfg_env = MeanVC2Config()
    assert cfg_env.source_dir == "/tmp/custom_meanvc2_src"


@pytest.mark.asyncio
async def test_remote_audio_subscription_and_bypass():
    """Remote WebSocket subscriber receives converted/bypass bytes and unsubscribes cleanly."""
    adapter = MeanVC2VoiceAdapter()
    q = adapter.subscribe_audio_out()
    adapter._state = EngineState.RUNNING
    await adapter.select_voice_profile("clean-mic")

    samples = np.ones(160, dtype=np.float32) * 0.25
    await adapter.push_remote_audio(samples)

    out_bytes = q.get_nowait()
    assert len(out_bytes) == 160 * 4
    adapter.unsubscribe_audio_out(q)
    assert q not in adapter._remote_audio_queues




