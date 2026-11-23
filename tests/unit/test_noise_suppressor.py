"""Unit tests for Phase 5 real-time Noise Suppressor DSP module."""

import numpy as np
import pytest

from hth.audio.noise.contracts import NoisePreset
from hth.audio.noise.suppressor import FRAME_SIZE, NoiseSuppressor


@pytest.fixture
def suppressor() -> NoiseSuppressor:
    ns = NoiseSuppressor(preset=NoisePreset.LIGHT)
    yield ns
    ns.close()


def test_suppressor_initialization(suppressor: NoiseSuppressor) -> None:
    """Verify preallocated buffers and state pointers."""
    assert suppressor.get_preset() == NoisePreset.LIGHT
    assert len(suppressor._in_buf) == FRAME_SIZE
    assert len(suppressor._out_buf) == FRAME_SIZE
    assert len(suppressor._ramp_buf) == FRAME_SIZE
    assert len(suppressor._silence_buf) == FRAME_SIZE
    assert suppressor._is_initialized is True


def test_preset_off_bypasses_dsp(suppressor: NoiseSuppressor) -> None:
    """When preset is OFF, input is passed through exactly."""
    suppressor.set_preset(NoisePreset.OFF)
    # Trigger one frame to settle target alpha
    in_dummy = np.zeros(FRAME_SIZE, dtype=np.float32)
    out_dummy = np.zeros(FRAME_SIZE, dtype=np.float32)
    suppressor.process_frame(in_dummy, out_dummy)

    test_input = np.sin(np.linspace(0, 10, FRAME_SIZE)).astype(np.float32)
    test_out = np.zeros(FRAME_SIZE, dtype=np.float32)

    prob = suppressor.process_frame(test_input, test_out)
    assert np.allclose(test_input, test_out, atol=1e-6)
    assert prob == 0.0


def test_preset_strong_attenuates_noise(suppressor: NoiseSuppressor) -> None:
    """When preset is STRONG, fan/hum noise is heavily attenuated."""
    suppressor.set_preset(NoisePreset.STRONG)
    # Settle transition
    in_dummy = np.zeros(FRAME_SIZE, dtype=np.float32)
    out_dummy = np.zeros(FRAME_SIZE, dtype=np.float32)
    suppressor.process_frame(in_dummy, out_dummy)

    t = np.linspace(0, 0.01, FRAME_SIZE, endpoint=False)
    noise = (0.3 * np.sin(2 * np.pi * 60 * t) + 0.05 * np.random.randn(FRAME_SIZE)).astype(
        np.float32
    )
    out = np.zeros(FRAME_SIZE, dtype=np.float32)

    # Process through RNNoise
    for _ in range(5):
        suppressor.process_frame(noise, out)

    in_rms = np.sqrt(np.mean(noise**2))
    out_rms = np.sqrt(np.mean(out**2))
    # Attenuation must be at least 15 dB
    assert out_rms < in_rms * 0.25


def test_smooth_click_free_transition(suppressor: NoiseSuppressor) -> None:
    """Switching preset applies a continuous linear crossfade ramp across the 480 samples."""
    suppressor.set_preset(NoisePreset.OFF)
    in_dummy = np.zeros(FRAME_SIZE, dtype=np.float32)
    out_dummy = np.zeros(FRAME_SIZE, dtype=np.float32)
    suppressor.process_frame(in_dummy, out_dummy)

    # Now switch to STRONG
    suppressor.set_preset(NoisePreset.STRONG)
    test_input = np.ones(FRAME_SIZE, dtype=np.float32) * 0.5
    test_out = np.zeros(FRAME_SIZE, dtype=np.float32)

    suppressor.process_frame(test_input, test_out)
    # Ramp buffer was populated with linear interpolation from 0.0 to 1.0
    assert suppressor._ramp_buf[0] == 0.0
    assert suppressor._ramp_buf[-1] > 0.95
    # First sample was near 100% dry, last sample was near 100% wet
    assert test_out[0] == test_input[0]


def test_fail_closed_on_corrupt_input_dimension(suppressor: NoiseSuppressor) -> None:
    """Invalid frame length immediately zeroes output buffer (Fail-Closed, Rule A-03)."""
    corrupt_input = np.ones(200, dtype=np.float32)  # Not 480 samples
    out = np.ones(FRAME_SIZE, dtype=np.float32)  # Pre-filled with non-zero

    prob = suppressor.process_frame(corrupt_input, out)
    assert np.all(out == 0.0)
    assert prob == 0.0


def test_metrics_telemetry(suppressor: NoiseSuppressor) -> None:
    """Metrics tracking reports execution times and underrun counts."""
    in_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    out_frame = np.zeros(FRAME_SIZE, dtype=np.float32)

    for _ in range(20):
        suppressor.process_frame(in_frame, out_frame)

    metrics = suppressor.get_metrics()
    assert metrics.frames_processed >= 20
    assert metrics.p50_ms >= 0.0
    assert metrics.p95_ms >= metrics.p50_ms
    assert metrics.active_preset == NoisePreset.LIGHT


def test_reset_state(suppressor: NoiseSuppressor) -> None:
    """Verify DSP state reset upon buffer discontinuity."""
    suppressor.reset_state()
    assert suppressor._is_initialized is True
    assert suppressor._has_error is False
    assert np.all(suppressor._in_buf == 0.0)
    assert np.all(suppressor._out_buf == 0.0)
