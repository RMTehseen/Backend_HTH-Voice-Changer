"""Unit tests for Zero-Allocation Polyphase Audio Resampler (48 kHz <-> 16 kHz)."""

import tracemalloc

import numpy as np
import pytest

from hth.audio.resampler import (
    AudioResampler,
    PolyphaseResampler16To48,
    PolyphaseResampler48To16,
    design_kaiser_sinc_fir,
)
from hth.domain.errors import AudioProcessingError


def test_kaiser_sinc_design() -> None:
    """Verifies pure NumPy Kaiser-windowed sinc FIR coefficients."""
    taps = design_kaiser_sinc_fir(num_taps=72, cutoff_hz=7200.0, fs_hz=48000.0, gain=1.0)
    assert taps.shape == (72,)
    assert taps.dtype == np.float32

    # Linear phase FIR filter must be symmetric
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-6)

    # DC gain must be 1.0
    assert float(np.sum(taps)) == pytest.approx(1.0, rel=1e-4)

    # Gain 3.0 for interpolation
    taps_up = design_kaiser_sinc_fir(num_taps=72, cutoff_hz=7200.0, fs_hz=48000.0, gain=3.0)
    assert float(np.sum(taps_up)) == pytest.approx(3.0, rel=1e-4)

    # Invalid tap count (not divisible by 3) raises ValueError
    with pytest.raises(ValueError, match="multiple of 3"):
        design_kaiser_sinc_fir(num_taps=70)


def test_downsampler_shapes_and_errors() -> None:
    """Verifies input/output validation for downsampler."""
    downsampler = PolyphaseResampler48To16()

    # Invalid input length
    bad_in = np.zeros(200, dtype=np.float32)
    with pytest.raises(AudioProcessingError, match="Expected 48 kHz frame with 480 samples"):
        downsampler.process_frame(bad_in)

    # Valid input length
    good_in = np.zeros(480, dtype=np.float32)
    out = downsampler.process_frame(good_in)
    assert out.shape == (160,)
    assert out.dtype == np.float32

    # Bad output buffer length
    bad_out = np.zeros(100, dtype=np.float32)
    with pytest.raises(AudioProcessingError, match="Expected 16 kHz output buffer with 160"):
        downsampler.process_frame(good_in, out_frame=bad_out)


def test_upsampler_shapes_and_errors() -> None:
    """Verifies input/output validation for upsampler."""
    upsampler = PolyphaseResampler16To48()

    # Invalid input length
    bad_in = np.zeros(100, dtype=np.float32)
    with pytest.raises(AudioProcessingError, match="Expected 16 kHz frame with 160 samples"):
        upsampler.process_frame(bad_in)

    # Valid input length
    good_in = np.zeros(160, dtype=np.float32)
    out = upsampler.process_frame(good_in)
    assert out.shape == (480,)
    assert out.dtype == np.float32

    # Bad output buffer length
    bad_out = np.zeros(200, dtype=np.float32)
    with pytest.raises(AudioProcessingError, match="Expected 48 kHz output buffer with 480"):
        upsampler.process_frame(good_in, out_frame=bad_out)


def test_dc_signal_preservation() -> None:
    """Verifies that unity DC signals pass through with near-zero ripple."""
    resampler = AudioResampler()

    dc_48k = np.ones(480, dtype=np.float32)
    # Warmup filter state
    for _ in range(5):
        resampler.downsample_48k_to_16k(dc_48k)

    dc_16k = resampler.downsample_48k_to_16k(dc_48k)
    np.testing.assert_allclose(dc_16k, 1.0, atol=0.01)

    # Test upsampler with DC 1.0
    dc_input_16k = np.ones(160, dtype=np.float32)
    for _ in range(5):
        resampler.upsample_16k_to_48k(dc_input_16k)

    reconstructed_48k = resampler.upsample_16k_to_48k(dc_input_16k)
    np.testing.assert_allclose(reconstructed_48k, 1.0, atol=0.01)


def test_frame_boundary_continuity() -> None:
    """Verifies that streaming sine waves across frame boundaries has zero clicks or jumps."""
    downsampler = PolyphaseResampler48To16()
    upsampler = PolyphaseResampler16To48()

    # 1 kHz test sine wave across 5 consecutive 10ms frames
    t_all = np.arange(480 * 5) / 48000.0
    sig_all = np.sin(2 * np.pi * 1000.0 * t_all).astype(np.float32)

    downsampled_chunks = []
    for i in range(5):
        chunk = sig_all[i * 480 : (i + 1) * 480]
        out_chunk = downsampler.process_frame(chunk)
        downsampled_chunks.append(out_chunk.copy())

    concatenated_16k = np.concatenate(downsampled_chunks)
    assert len(concatenated_16k) == 160 * 5

    # Check derivative continuity across frame boundaries (indices 159->160, 319->320, etc.)
    diffs_16k = np.diff(concatenated_16k)
    boundary_steps = np.abs(diffs_16k[159::160])
    max_normal_step = np.max(np.abs(diffs_16k))
    # Boundary step must not exceed normal signal transition
    assert np.max(boundary_steps) <= max_normal_step + 1e-4

    # Now test upsampler continuity
    upsampled_chunks = []
    for i in range(5):
        chunk_16k = concatenated_16k[i * 160 : (i + 1) * 160]
        out_48k = upsampler.process_frame(chunk_16k)
        upsampled_chunks.append(out_48k.copy())

    concatenated_48k = np.concatenate(upsampled_chunks)
    assert len(concatenated_48k) == 480 * 5
    diffs_48k = np.diff(concatenated_48k)
    boundary_steps_48k = np.abs(diffs_48k[479::480])
    max_normal_step_48k = np.max(np.abs(diffs_48k))
    assert np.max(boundary_steps_48k) <= max_normal_step_48k + 1e-4


def test_zero_allocation_in_streaming() -> None:
    """Verifies Rule A-01: Zero runtime heap memory allocation during processing."""
    downsampler = PolyphaseResampler48To16()
    upsampler = PolyphaseResampler16To48()

    in_480 = np.random.randn(480).astype(np.float32)
    in_160 = np.random.randn(160).astype(np.float32)
    out_160 = np.zeros(160, dtype=np.float32)
    out_480 = np.zeros(480, dtype=np.float32)

    # Warmup
    for _ in range(5):
        downsampler.process_frame(in_480, out_frame=out_160)
        upsampler.process_frame(in_160, out_frame=out_480)

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    for _ in range(200):
        downsampler.process_frame(in_480, out_frame=out_160)
        upsampler.process_frame(in_160, out_frame=out_480)

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snap_after.compare_to(snap_before, "lineno")
    # Filter out test loop counter / tracemalloc internal overhead
    streaming_allocs = [
        stat
        for stat in top_stats
        if "resampler.py" in stat.traceback.format()[0] and stat.size_diff > 64
    ]
    assert len(streaming_allocs) == 0, f"Found streaming allocations: {streaming_allocs}"


def test_resampler_reset() -> None:
    """Verifies state reset clears history buffers cleanly."""
    resampler = AudioResampler()
    sig_480 = np.ones(480, dtype=np.float32)
    resampler.downsample_48k_to_16k(sig_480)
    assert np.any(resampler.downsampler._history != 0.0)

    resampler.reset()
    assert np.all(resampler.downsampler._history == 0.0)
    assert np.all(resampler.upsampler._history == 0.0)
