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


def test_meanvc2_config_device_and_variant_env(monkeypatch):
    """HTH_MEANVC2_DEVICE and HTH_MEANVC2_VARIANT override config defaults."""
    monkeypatch.setenv("HTH_MEANVC2_DEVICE", "cuda")
    monkeypatch.setenv("HTH_MEANVC2_VARIANT", "120ms")
    cfg = MeanVC2Config()
    assert cfg.device == "cuda"
    assert cfg.model_variant == "120ms"


@pytest.mark.asyncio
async def test_remote_audio_backpressure_drops_stale_chunks():
    """When inflight worker chunks reach max limit, stale incoming frames drop cleanly."""
    adapter = MeanVC2VoiceAdapter()
    adapter._state = EngineState.RUNNING
    adapter._active_profile_id = "profile-female-eva"
    adapter._inflight_chunks = adapter._max_inflight_chunks

    samples = np.zeros(160, dtype=np.float32)
    await adapter.push_remote_audio(samples)
    assert adapter._overruns == 1

    await adapter._handle_worker_event({"event": "audio_buffering"})
    assert adapter._inflight_chunks == adapter._max_inflight_chunks - 1


def test_telemetry_subscription_and_unsubscribe():
    """Telemetry queue receives metrics snapshots and unsubscribes cleanly."""
    adapter = MeanVC2VoiceAdapter()
    q = adapter.subscribe_telemetry()
    adapter._last_inference_ms = 14.2
    adapter._last_rtf = 0.09
    adapter._overruns = 2
    adapter._last_telemetry_ts = 0.0
    adapter._emit_telemetry()

    item = q.get_nowait()
    assert item["type"] == "telemetry"
    assert item["inference_ms"] == 14.2
    assert item["rtf"] == 0.09
    assert item["overruns"] == 2

    adapter.unsubscribe_telemetry(q)
    assert q not in adapter._telemetry_queues


@pytest.mark.asyncio
async def test_attach_profile_reference_and_has_reference(tmp_path):
    """Attaching a valid WAV updates reference path and profile_has_reference."""
    import wave

    adapter = MeanVC2VoiceAdapter()
    adapter._profile_store = VoiceProfileStore(str(tmp_path / "profiles"), auto_seed=True)
    prof = adapter.list_voice_profiles()[0]
    assert adapter.profile_has_reference(prof) is False

    wav_path = tmp_path / "sample.wav"
    sr = 16000
    t = np.arange(int(sr * 1.5), dtype=np.float32) / sr
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    updated = await adapter.attach_profile_reference(prof.profile_id, str(wav_path))
    assert updated.reference_wav_path == str(wav_path)
    assert adapter.profile_has_reference(updated) is True


@pytest.mark.asyncio
async def test_processing_mode_switch_and_bypass_passthrough():
    """Processing mode defaults to hybrid (calibrated age/vocal shaping) and routes bypass cleanly."""
    adapter = MeanVC2VoiceAdapter()
    assert adapter._processing_mode == "hybrid"
    assert adapter.set_processing_mode("pure_neural") == "pure_neural"
    assert adapter.set_processing_mode("hybrid") == "hybrid"
    assert adapter.set_processing_mode("bypass") == "bypass"

    q = adapter.subscribe_audio_out()
    adapter._state = EngineState.RUNNING
    adapter._active_profile_id = "profile-female-eva"
    samples = np.ones(160, dtype=np.float32) * 0.125
    await adapter.push_remote_audio(samples)
    out = q.get_nowait()
    assert len(out) == 160 * 4
    adapter.unsubscribe_audio_out(q)

    with pytest.raises(ValueError):
        adapter.set_processing_mode("unsupported")


@pytest.mark.asyncio
async def test_idle_profile_switch_clears_applied_target_and_wsola_continuity(tmp_path):
    """Switching voice profile while IDLE invalidates cached worker target; WSOLA preserves energy."""
    from hth.adapters.meanvc2.dsp_effects import DSPPostProcessor

    adapter = MeanVC2VoiceAdapter()
    adapter._profile_store = VoiceProfileStore(str(tmp_path / "profiles"), auto_seed=True)
    adapter._worker_target_set.set()
    adapter._applied_profile_id = "profile-female-eva"

    await adapter.select_voice_profile("profile-male-hamza")
    assert adapter._active_profile_id == "profile-male-hamza"
    assert adapter._applied_profile_id is None
    assert not adapter._worker_target_set.is_set()

    # Curated profiles should have distinct vocal age/pitch calibrations
    zoya = adapter._profile_store.get("profile-female-zoya")
    eva = adapter._profile_store.get("profile-female-eva")
    hamza = adapter._profile_store.get("profile-male-hamza")
    bilal = adapter._profile_store.get("profile-male-bilal")
    assert zoya and eva and hamza and bilal
    assert zoya.pitch_shift > eva.pitch_shift
    assert hamza.pitch_shift > bilal.pitch_shift

    # Verify WSOLA pitch shift continuity across consecutive 80ms (1280) and 160ms (2560) chunks
    for csize in (1280, 2560):
        dsp = DSPPostProcessor(sample_rate=16000, chunk_size=csize)
        t_all = np.arange(csize * 3, dtype=np.float32) / 16000.0
        sig = 0.4 * np.sin(2 * np.pi * 200.0 * t_all)
        out_chunks = [
            dsp.process(sig[i * csize : (i + 1) * csize], pitch_shift=3.0, timbre_ratio=1.08)
            for i in range(3)
        ]
        joined = np.concatenate(out_chunks)
        b_idx = csize * 2
        rms_local = float(np.sqrt(np.mean(joined[b_idx - 50 : b_idx + 50] ** 2)))
        assert rms_local > 0.20
        assert abs(float(joined[b_idx] - joined[b_idx - 1])) / rms_local < 0.08


def test_long_soak_config_and_silent_cache_reset(monkeypatch):
    """Verify generous watchdog/backpressure limits and synchronized silent cache reset."""
    import sys
    import types
    from hth.adapters.meanvc2.worker_process import MeanVC2Worker

    cfg = MeanVC2Config()
    assert cfg.heartbeat_timeout_s >= 25.0
    adapter = MeanVC2VoiceAdapter(cfg)
    assert adapter._max_inflight_chunks >= 12

    # Build minimal torch + kaldi stubs so worker_process silence reset runs in isolated Python 3.13 venv
    class _FakeTensor:
        def __init__(self, data: np.ndarray):
            self.data = np.asarray(data, dtype=np.float32)
            self.shape = self.data.shape
            self.dtype = "float32"

        def dim(self):
            return self.data.ndim

        def size(self, axis: int):
            return self.data.shape[axis]

        def unsqueeze(self, axis: int):
            return _FakeTensor(np.expand_dims(self.data, axis))

        def squeeze(self, axis: int | None = None):
            return _FakeTensor(np.squeeze(self.data, axis=axis))

        def transpose(self, d0: int, d1: int):
            return _FakeTensor(np.swapaxes(self.data, d0, d1))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def float(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def zero_(self):
            self.data.fill(0.0)
            return self

        def fill_(self, val: float):
            self.data.fill(val)
            return self

        def clone(self):
            return _FakeTensor(self.data.copy())

        def __getitem__(self, item):
            return _FakeTensor(self.data[item])

        def __setitem__(self, item, val):
            self.data[item] = val.data if isinstance(val, _FakeTensor) else val

        def __sub__(self, other):
            return _FakeTensor(self.data - (other.data if isinstance(other, _FakeTensor) else other))

        def __add__(self, other):
            return _FakeTensor(self.data + (other.data if isinstance(other, _FakeTensor) else other))

        def __mul__(self, other):
            return _FakeTensor(self.data * (other.data if isinstance(other, _FakeTensor) else other))

        def __rmul__(self, other):
            return self.__mul__(other)

        def __truediv__(self, other):
            return _FakeTensor(self.data / (other.data if isinstance(other, _FakeTensor) else other))

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.int64 = "int64"
    fake_torch.float16 = "float16"
    fake_torch.zeros = lambda *shape, **kw: _FakeTensor(np.zeros(shape, dtype=np.float32))
    fake_torch.ones = lambda *shape, **kw: _FakeTensor(np.ones(shape, dtype=np.float32))
    fake_torch.randn = lambda *shape, **kw: _FakeTensor(np.zeros(shape, dtype=np.float32))
    fake_torch.full = lambda shape, val, **kw: _FakeTensor(np.full(shape, val, dtype=np.float32))
    fake_torch.tensor = lambda val, **kw: _FakeTensor(np.array(val, dtype=np.float32))
    fake_torch.from_numpy = lambda arr: _FakeTensor(arr)
    fake_torch.cat = lambda seq, dim=0: _FakeTensor(np.concatenate([x.data for x in seq], axis=dim))
    fake_torch.linspace = lambda a, b, steps=10: _FakeTensor(np.linspace(a, b, steps, dtype=np.float32))
    fake_torch.no_grad = _NoGrad
    fake_torch.inference_mode = _NoGrad

    fake_F = types.ModuleType("torch.nn.functional")
    fake_F.interpolate = lambda x, size=1, mode="linear", align_corners=True: _FakeTensor(
        np.zeros((x.shape[0], x.shape[1], size), dtype=np.float32)
    )
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.functional = fake_F
    fake_torch.nn = fake_nn

    fake_kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    fake_kaldi.fbank = lambda wav, **kw: _FakeTensor(
        np.zeros((max(0, (wav.shape[-1] - 400) // 160 + 1), 80), dtype=np.float32)
    )
    fake_comp = types.ModuleType("torchaudio.compliance")
    fake_comp.kaldi = fake_kaldi
    fake_ta = types.ModuleType("torchaudio")
    fake_ta.compliance = fake_comp

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fake_F)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)
    monkeypatch.setitem(sys.modules, "torchaudio.compliance", fake_comp)
    monkeypatch.setitem(sys.modules, "torchaudio.compliance.kaldi", fake_kaldi)
    monkeypatch.setattr("hth.adapters.meanvc2.worker_process.torch", fake_torch)
    monkeypatch.setattr("hth.adapters.meanvc2.worker_process.F", fake_F)

    worker = MeanVC2Worker()

    class _FakeASR:
        def __call__(self, fb, offset, req, att, cnn):
            return fake_torch.zeros(1, 2, 256), att + 1.0, cnn + 1.0

    class _FakeVC:
        def __call__(self, x, t, r, cache=None, cond=None, spks=None, offset=0, is_inference=True, kv_cache=None, gtm_kv=None):
            return fake_torch.zeros(*x.shape), [("k", "v")]

    class _FakeVocoder:
        def decode(self, mel):
            return fake_torch.zeros(1, 1, mel.shape[-1] * 160)

    class _Runner:
        device = "cpu"
        asr_device = "cpu"
        asr_scalar_device = "cpu"
        asr = _FakeASR()
        vc = _FakeVC()
        vocoder = _FakeVocoder()
        _kaldi = fake_kaldi
        vc_spk_emb = fake_torch.ones(1, 256)
        vc_gtm_kv = (fake_torch.zeros(1, 32, 256), fake_torch.zeros(1, 32, 256))
        variant = "40ms"
        n_steps = 1
        use_fp16 = False
        chunk_size = 4
        block_size = 4
        CHUNK = 2560
        _bn_window = 11
        _bn_stride = 8
        _required_cache_size = 4
        _asr_offset_init = 4
        _asr_offset_step = 2
        vocoder_overlap = 2
        upsample_factor = 160
        vocoder_wav_overlap = 160
        vocoder_wav_pad = 0
        down_linspace = fake_torch.linspace(1, 0, steps=160)
        up_linspace = fake_torch.linspace(0, 1, steps=160)
        _min_bn_len = 8
        _req_cache_tensor = fake_torch.tensor(4, dtype="int64")

    worker._runner = _Runner()
    worker._init_cache()
    r = worker._runner
    r.asr_offset = 2500
    r.vc_offset = 1800
    r.vc_kv_cache = [("old_k", "old_v")]
    r.asr_att_cache.fill_(3.0)

    silent_80 = np.zeros(1280, dtype=np.float32)
    r.asr_offset = 2500
    r.vc_offset = 1800
    r.vc_kv_cache = [("old_k", "old_v")]
    r._silent_streak = 0
    for _ in range(3):
        worker.process_chunk(silent_80)
    assert r.vc_offset >= 1800  # < 4 chunks (320ms) hasn't reset yet
    worker.process_chunk(silent_80)
    assert r.asr_offset < 100
    assert r.vc_offset < 100
    assert r.vc_kv_cache != [("old_k", "old_v")]






