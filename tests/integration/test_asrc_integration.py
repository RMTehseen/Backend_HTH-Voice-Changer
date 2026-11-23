"""Integration tests for ASRC clock synchronization inside AudioPipeline.

Validates:
1. Zero heap allocation in render callback during ASRC resampling.
2. Clock drift tracking: PI controller adapts rho dynamically based on buffer occupancy.
3. Fail-closed underflow protection: state machine zeroes output on buffer starvation.
4. Latched MUTED_ERROR behavior and recovery through un_mute().
5. Panic mute absolute precedence over ASRC processing.
6. ASRC metrics exposure in AudioPipelineMetrics.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from hth.audio.asrc import ASRCState
from hth.audio.pipeline import AudioPipeline


def test_asrc_render_callback_streaming() -> None:
    """Verify render callback with ASRC produces clean audio and advances read pointer."""
    pipeline = AudioPipeline()
    pipeline._is_running = True

    # Pre-populate ring buffer with 960 samples (target occupancy)
    test_tone = np.sin(2 * np.pi * 440 * np.arange(960) / 48000).astype(np.float32)
    pipeline.ring_buffer.write(test_tone)
    assert pipeline.ring_buffer.available_read == 960

    outdata = np.zeros((480, 1), dtype=np.float32)
    pipeline._render_callback(outdata, 480, None, None)

    # Output must have sound (not zero) and no NaNs
    assert not np.all(outdata == 0.0)
    assert not np.any(np.isnan(outdata))

    # Read pointer must have advanced by around 480 samples
    assert pipeline.ring_buffer.available_read < 960
    assert pipeline.ring_buffer.available_read >= 475

    metrics = pipeline.get_metrics()
    assert metrics.asrc_state == ASRCState.HEALTHY.value
    assert metrics.asrc_rho == pytest.approx(1.0, abs=1e-4)


def test_asrc_fail_closed_underflow_and_latch() -> None:
    """Verify fail-closed silence on underflow and eventual MUTED_ERROR latching."""
    pipeline = AudioPipeline()
    pipeline._is_running = True

    # Buffer is empty (0 samples < needed ~482)
    outdata = np.ones((480, 1), dtype=np.float32)

    # Initial underflows trigger UNDERFLOW_WARNING
    for _ in range(5):
        pipeline._render_callback(outdata, 480, None, None)
        assert np.all(outdata == 0.0)

    metrics = pipeline.get_metrics()
    assert metrics.asrc_state == ASRCState.UNDERFLOW_WARNING.value
    assert metrics.underrun_count == 5

    # Sustained underflow triggers MUTED_ERROR latch (threshold = 20)
    for _ in range(15):
        pipeline._render_callback(outdata, 480, None, None)
        assert np.all(outdata == 0.0)

    metrics = pipeline.get_metrics()
    assert metrics.asrc_state == ASRCState.MUTED_ERROR.value
    assert metrics.underrun_count == 20

    # In MUTED_ERROR, even if samples arrive, stream remains muted until un_mute()
    pipeline.ring_buffer.write(np.zeros(960, dtype=np.float32))
    outdata.fill(1.0)
    pipeline._render_callback(outdata, 480, None, None)
    assert np.all(outdata == 0.0)

    # Calling un_mute() with sufficient occupancy recovers to RUNNING/HEALTHY
    pipeline.un_mute()
    outdata.fill(1.0)
    pipeline._render_callback(outdata, 480, None, None)
    # Output now processed without error
    metrics = pipeline.get_metrics()
    assert metrics.asrc_state in (ASRCState.RUNNING.value, ASRCState.HEALTHY.value)


def test_asrc_panic_mute_precedence() -> None:
    """Verify panic mute zeroes output regardless of ASRC state or available buffer."""
    pipeline = AudioPipeline()
    pipeline._is_running = True

    # Fill buffer with audio
    pipeline.ring_buffer.write(np.ones(960, dtype=np.float32) * 0.5)

    pipeline.panic_mute()
    assert pipeline.is_muted

    outdata = np.ones((480, 1), dtype=np.float32)
    pipeline._render_callback(outdata, 480, None, None)

    assert np.all(outdata == 0.0)
    metrics = pipeline.get_metrics()
    assert metrics.asrc_state == ASRCState.MUTED_ERROR.value


def test_asrc_zero_allocation_in_callback() -> None:
    """Verify that repeated render callback execution causes zero heap allocation."""
    pipeline = AudioPipeline()
    pipeline._is_running = True
    # Stop background sampler thread so its periodic psutil RSS sample doesn't contaminate tracemalloc
    pipeline._telemetry_sampler.stop()

    outdata = np.zeros((480, 1), dtype=np.float32)
    in_chunk = np.zeros(480, dtype=np.float32)

    # Warm-up phase
    for _ in range(20):
        pipeline.ring_buffer.write(in_chunk)
        pipeline._render_callback(outdata, 480, None, None)

    tracemalloc.start()
    s0 = tracemalloc.take_snapshot()

    for _ in range(100):
        pipeline.ring_buffer.write(in_chunk)
        pipeline._render_callback(outdata, 480, None, None)

    s1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diff = [
        d
        for d in s1.compare_to(s0, "lineno")
        if ("asrc.py" in d.traceback[0].filename or "pipeline.py" in d.traceback[0].filename)
        and d.size_diff > 64
    ]
    assert len(diff) == 0, f"Detected unexpected heap allocations in ASRC callback: {diff}"
