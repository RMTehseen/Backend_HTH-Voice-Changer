"""Unit and integration tests for Ring Buffer Overflow Remediation (Phase C1 Gate 2).

Covers:
- Occupancy exactly at high-water threshold.
- Occupancy one sample above threshold.
- Maximum legal capture packet and observed bursts.
- Buffer capacity boundary and circular wraparound during overflow.
- Minimal-loss frame-aligned recovery (480 samples vs legacy 1440 bulk discard).
- ASRC phase and controller state preservation after discontinuity.
- Prevention of false underflow following burst events.
- Fail-closed security: no raw microphone passthrough and panic mute precedence.
- Concurrent producer/consumer consistency and zero-allocation hot paths.
"""

from __future__ import annotations

import threading
import time
import tracemalloc

import numpy as np
import pytest

from hth.audio.pipeline import AudioPipeline, AudioPipelineConfig
from hth.audio.ring_buffer import AudioRingBuffer


def test_occupancy_exactly_at_high_water_threshold() -> None:
    """Verify that writing exactly to max_occupancy triggers zero drops."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)
    # Write 6 frames of 480 = 2880 samples
    for i in range(6):
        data = np.full(480, float(i + 1), dtype=np.float32)
        written = rb.write(data)
        assert written == 480

    assert rb.available_read == 2880
    metrics = rb.get_metrics()
    assert metrics["overflow_count"] == 0
    assert metrics["dropped_samples"] == 0


def test_occupancy_one_sample_above_threshold() -> None:
    """Verify that writing one sample above max_occupancy triggers minimal 1-frame drop."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)
    # Fill to 2880
    for _ in range(6):
        rb.write(np.zeros(480, dtype=np.float32))
    assert rb.available_read == 2880

    # Write 1 additional sample: projected = 2881 > 2880
    # Must drop exactly 1 frame (480 samples), leaving 2881 - 480 = 2401 samples
    rb.write(np.array([1.0], dtype=np.float32))

    metrics = rb.get_metrics()
    assert metrics["overflow_count"] == 1
    assert metrics["dropped_samples"] == 480
    assert rb.available_read == 2401


def test_maximum_legal_capture_packet_and_bursts() -> None:
    """Verify handling of legal 480-sample and 960-sample burst packets."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)

    # 480-sample normal frame
    written = rb.write(np.ones(480, dtype=np.float32))
    assert written == 480
    assert rb.available_read == 480

    # 960-sample double-frame burst (WASAPI timing jitter)
    written2 = rb.write(np.ones(960, dtype=np.float32))
    assert written2 == 960
    assert rb.available_read == 1440
    assert rb.get_metrics()["overflow_count"] == 0


def test_buffer_capacity_boundary_headroom() -> None:
    """Verify that with max_occupancy=2880, capacity=4096 maintains 1216-sample safe headroom."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)
    assert rb.capacity == 4096
    assert rb.max_occupancy == 2880
    # Headroom before physical capacity is exhausted
    headroom = rb.capacity - rb.max_occupancy
    assert headroom == 1216  # Over 2.5 full frames (53.3 ms margin)


def test_ring_buffer_wraparound_during_overflow_handling() -> None:
    """Verify data integrity when overflow drops occur across power-of-2 ring wraparound."""
    rb = AudioRingBuffer(capacity=2048, max_occupancy=1440, target_occupancy=480)
    out_buf = np.zeros(480, dtype=np.float32)

    # Cycle 20 times writing 480 samples and occasionally reading
    for i in range(20):
        data = np.full(480, float(i), dtype=np.float32)
        rb.write(data)
        if i % 2 == 0:
            rb.read(out_buf, 480)

    # Ensure buffer remains strictly bounded and functional
    assert rb.available_read <= 1440
    metrics = rb.get_metrics()
    assert metrics["overflow_count"] > 0
    assert metrics["underrun_count"] == 0


def test_minimal_loss_recovery_vs_legacy_bulk_discard() -> None:
    """Verify minimal-loss discard: drops 480 samples instead of legacy 1440 samples."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)
    # Fill to 2400
    for _ in range(5):
        rb.write(np.zeros(480, dtype=np.float32))
    assert rb.available_read == 2400

    # Add 960 burst -> projected = 3360 > 2880
    # excess = 3360 - 2880 = 480 -> drops exactly 1 frame (480 samples)
    rb.write(np.zeros(960, dtype=np.float32))
    metrics = rb.get_metrics()
    assert metrics["overflow_count"] == 1
    assert metrics["dropped_samples"] == 480  # NOT 1440!
    assert rb.available_read == 2880


def test_no_false_underflow_after_overflow_drop() -> None:
    """Verify that after an overflow drop, buffer retains sufficient samples for Catmull-Rom support."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)
    peek_buf = np.zeros(512, dtype=np.float32)

    # Fill to 2880 and push 480 over limit
    for _ in range(7):
        rb.write(np.zeros(480, dtype=np.float32))

    # Peek Catmull-Rom support window (482 samples)
    has_support = rb.peek(peek_buf, num_samples=482)
    assert has_support is True, "Buffer falsely underflowed following overflow drop!"
    assert rb.available_read >= 482


def test_asrc_phase_and_controller_preservation_after_discontinuity() -> None:
    """Verify pipeline discontinuity hook resets resampler phase but preserves learned clock drift."""
    pipe = AudioPipeline()
    # Prime controller with learned drift
    pipe.asrc_controller.integral_err = 500.0
    pipe.asrc_controller.curr_corr_ppm = 250.0

    # Trigger discontinuity
    pipe._handle_discontinuity()

    # Controller learned state must be preserved
    assert pipe.asrc_controller.integral_err == 500.0
    assert pipe.asrc_controller.curr_corr_ppm == 250.0
    # Resampler phase accumulator must be reset
    assert pipe.asrc_resampler.phase_accum == 0.0


def test_fail_closed_no_raw_mic_passthrough() -> None:
    """Enforce strict fail-closed safety: raw mic passthrough is rejected in production mode."""
    from hth.adapters.pipeline_adapter import AudioPipelineAdapter

    adapter = AudioPipelineAdapter()
    with pytest.raises(RuntimeError, match="fail-closed rule A-03"):
        import asyncio

        asyncio.run(adapter.set_passthrough(True))


def test_panic_mute_absolute_precedence() -> None:
    """Verify panic mute unconditionally zeroes output buffer even with full audio in ring buffer."""
    pipe = AudioPipeline(AudioPipelineConfig(input_device_id=0, output_device_id=0))
    # Write loud audio into ring buffer
    pipe.ring_buffer.write(np.full(960, 0.8, dtype=np.float32))
    assert pipe.ring_buffer.available_read == 960

    # Engage panic mute
    pipe.panic_mute()
    assert pipe.is_muted is True

    outdata = np.full((480, 1), 0.5, dtype=np.float32)
    pipe._render_callback(outdata, 480, None, None)

    # Must be strictly zeroed
    np.testing.assert_allclose(outdata, 0.0, atol=1e-7)


def test_concurrent_producer_consumer_correctness() -> None:
    """Verify thread-safe SPSC operation with concurrent producer and consumer threads."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)
    stop_event = threading.Event()
    frames_produced = 0
    frames_consumed = 0

    def producer() -> None:
        nonlocal frames_produced
        while not stop_event.is_set():
            data = np.full(480, 0.1, dtype=np.float32)
            rb.write(data)
            frames_produced += 1
            time.sleep(0.005)

    def consumer() -> None:
        nonlocal frames_consumed
        out = np.zeros(480, dtype=np.float32)
        while not stop_event.is_set():
            if rb.available_read >= 480:
                success = rb.read(out, 480)
                if success:
                    frames_consumed += 1
            time.sleep(0.005)

    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, daemon=True)

    t_prod.start()
    t_cons.start()
    time.sleep(0.5)
    stop_event.set()
    t_prod.join(timeout=1.0)
    t_cons.join(timeout=1.0)

    assert frames_produced > 20
    assert frames_consumed > 10
    assert rb.get_metrics()["underrun_count"] == 0


def test_zero_allocation_during_overflow_drop() -> None:
    """Verify zero runtime heap allocation in write() even when an overflow drop occurs."""
    rb = AudioRingBuffer(capacity=4096, max_occupancy=2880, target_occupancy=960)
    in_data = np.zeros(480, dtype=np.float32)

    # Fill to boundary
    for _ in range(6):
        rb.write(in_data)

    # Warm-up drop
    rb.write(in_data)

    tracemalloc.start()
    s0 = tracemalloc.take_snapshot()

    # Trigger drops in hot loop
    for _ in range(50):
        rb.write(in_data)

    s1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diff = [
        d
        for d in s1.compare_to(s0, "lineno")
        if "ring_buffer.py" in d.traceback[0].filename and d.size_diff > 64
    ]
    assert len(diff) == 0, f"Detected unexpected buffer allocations during overflow: {diff}"
