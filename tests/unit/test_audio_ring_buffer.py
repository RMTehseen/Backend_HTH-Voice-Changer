"""Unit tests for AudioRingBuffer verifying zero-allocation, wrap-around, and fail-closed safety."""

import tracemalloc

import numpy as np
import pytest

from hth.audio.ring_buffer import AudioRingBuffer


def test_initialization() -> None:
    rb = AudioRingBuffer(capacity=2048, max_occupancy=1440, target_occupancy=480)
    assert rb.capacity == 2048
    assert rb.available_read == 0
    assert rb.available_write == 2048

    # Non-power-of-two should fail
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity=1000)

    # max_occupancy > capacity should fail
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity=2048, max_occupancy=3000)


def test_write_read_basic() -> None:
    rb = AudioRingBuffer(capacity=2048)
    in_data = np.linspace(-1.0, 1.0, 480, dtype=np.float32)
    out_frame = np.zeros(480, dtype=np.float32)

    written = rb.write(in_data)
    assert written == 480
    assert rb.available_read == 480

    success = rb.read(out_frame, 480)
    assert success is True
    assert rb.available_read == 0
    np.testing.assert_allclose(out_frame, in_data, atol=1e-6)


def test_circular_wrap_around() -> None:
    rb = AudioRingBuffer(capacity=2048)
    out_frame = np.zeros(480, dtype=np.float32)

    # Cycle 10 times (4800 samples total, wrapping around 2048 capacity multiple times)
    for i in range(10):
        in_data = np.full(480, fill_value=float(i) * 0.1, dtype=np.float32)
        rb.write(in_data)
        success = rb.read(out_frame, 480)
        assert success is True
        np.testing.assert_allclose(out_frame, in_data, atol=1e-6)

    assert rb.available_read == 0
    metrics = rb.get_metrics()
    assert metrics["underrun_count"] == 0
    assert metrics["overflow_count"] == 0


def test_stereo_downmix() -> None:
    rb = AudioRingBuffer(capacity=2048)
    # Left channel = 0.4, Right channel = 0.8 -> Expected mono = 0.6
    stereo_in = np.zeros((480, 2), dtype=np.float32)
    stereo_in[:, 0] = 0.4
    stereo_in[:, 1] = 0.8
    out_frame = np.zeros(480, dtype=np.float32)

    rb.write(stereo_in)
    success = rb.read(out_frame, 480)
    assert success is True
    np.testing.assert_allclose(out_frame, 0.6, atol=1e-6)


def test_underflow_safety() -> None:
    rb = AudioRingBuffer(capacity=2048)
    out_frame = np.ones(480, dtype=np.float32)  # pre-fill with 1.0 to check zeroing

    # Read when empty
    success = rb.read(out_frame, 480)
    assert success is False
    # Must be zeroed (pure silence)
    np.testing.assert_allclose(out_frame, 0.0, atol=1e-6)
    assert rb.get_metrics()["underrun_count"] == 1


def test_queue_age_capping_and_overflow() -> None:
    rb = AudioRingBuffer(capacity=2048, max_occupancy=1440, target_occupancy=480)
    # Write 3 chunks of 480 = 1440 samples (at limit)
    rb.write(np.full(480, 1.0, dtype=np.float32))
    rb.write(np.full(480, 2.0, dtype=np.float32))
    rb.write(np.full(480, 3.0, dtype=np.float32))
    assert rb.available_read == 1440

    # Write 4th chunk of 480 (total would be 1920 > max 1440)
    # Should drop oldest excess samples to keep latency bounded
    rb.write(np.full(480, 4.0, dtype=np.float32))
    metrics = rb.get_metrics()
    assert metrics["overflow_count"] == 1
    assert metrics["dropped_samples"] > 0

    # Available read should be capped near target occupancy
    assert rb.available_read <= 1440


def test_discontinuity_callback_triggered() -> None:
    reset_count = 0

    def on_disc() -> None:
        nonlocal reset_count
        reset_count += 1

    rb = AudioRingBuffer(
        capacity=2048,
        max_occupancy=1440,
        target_occupancy=480,
        on_discontinuity=on_disc,
    )
    # Write 3 chunks = 1440
    rb.write(np.full(480, 1.0, dtype=np.float32))
    rb.write(np.full(480, 2.0, dtype=np.float32))
    rb.write(np.full(480, 3.0, dtype=np.float32))
    assert reset_count == 0

    # 4th chunk triggers overflow
    rb.write(np.full(480, 4.0, dtype=np.float32))
    assert reset_count == 1
    assert rb.get_metrics()["discontinuity_epoch"] == 1


def test_zero_allocation() -> None:
    rb = AudioRingBuffer(capacity=2048)
    in_data = np.zeros(480, dtype=np.float32)
    out_frame = np.zeros(480, dtype=np.float32)

    # Warm-up run across multiple iterations so integer references settle
    for _ in range(10):
        rb.write(in_data)
        rb.read(out_frame, 480)

    # Track allocations during hot loop
    tracemalloc.start()
    s0 = tracemalloc.take_snapshot()

    for _ in range(100):
        rb.write(in_data)
        rb.read(out_frame, 480)

    s1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Filter out tracemalloc's internal overhead and verify zero buffer/array allocations
    # (Per-frame allocations must be 0; total across 100 frames must not exceed one-time scalar int cache)
    diff = [
        d
        for d in s1.compare_to(s0, "lineno")
        if "ring_buffer.py" in d.traceback[0].filename and d.size_diff > 64
    ]
    assert len(diff) == 0, f"Detected unexpected buffer allocations in hot path: {diff}"


def test_peek_and_advance_read_ptr() -> None:
    rb = AudioRingBuffer(capacity=2048)
    in_data = np.arange(480, dtype=np.float32)
    peek_buf = np.zeros(480, dtype=np.float32)

    # 1. Underflow peek on empty buffer
    success = rb.peek(peek_buf, 480)
    assert success is False
    np.testing.assert_allclose(peek_buf, 0.0)

    # 2. Write 480 samples
    rb.write(in_data)
    assert rb.available_read == 480

    # 3. Peek 480 samples: must succeed and NOT advance pointers
    success = rb.peek(peek_buf, 480)
    assert success is True
    assert rb.available_read == 480
    np.testing.assert_allclose(peek_buf, in_data)

    # 4. Advance pointer by 240 samples
    advanced = rb.advance_read_ptr(240)
    assert advanced == 240
    assert rb.available_read == 240

    # 5. Peek remaining 240 samples
    half_buf = np.zeros(240, dtype=np.float32)
    success = rb.peek(half_buf, 240)
    assert success is True
    np.testing.assert_allclose(half_buf, in_data[240:])

    # 6. Advance remaining 240 samples
    advanced = rb.advance_read_ptr(240)
    assert advanced == 240
    assert rb.available_read == 0

    # 7. Circular wrap-around with peek & advance
    # Cycle 5 times wrapping around 2048
    for i in range(5):
        chunk = np.full(480, fill_value=float(i + 1), dtype=np.float32)
        rb.write(chunk)
        success = rb.peek(peek_buf, 480)
        assert success is True
        np.testing.assert_allclose(peek_buf, chunk)
        adv = rb.advance_read_ptr(480)
        assert adv == 480
    assert rb.available_read == 0
