"""Bounded Single-Producer Single-Consumer (SPSC) Audio Ring Buffer.

Adheres strictly to Windows Real-Time Audio Constraints (Rules A-01..A-06):
- Bounded capacity (2048 samples = 42.67 ms @ 48 kHz).
- Preallocated contiguous float32 storage; zero allocation in read/write hot paths.
- Queue age capping: drops oldest frames if latency exceeds 30 ms (1440 samples).
- Frame-aligned sample dropping on overflow with stateful DSP reset callback.
- Fail-closed underflow: fills out_frame with silence on underflow; never repeats audio.
- Automatic channel normalization (stereo-to-mono downmix during capture write).

NOTE: This Python/NumPy implementation provides bounded, preallocated memory management
for the local TestSink audio pipeline. In accordance with project engineering rules,
Python callbacks running under the GIL cannot be described as guaranteed lock-free
atomics; this implementation serves as the verified Stage 1 audio foundation.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class AudioRingBuffer:
    """Bounded, zero-allocation SPSC ring buffer for real-time audio streaming."""

    def __init__(
        self,
        capacity: int = 2048,
        max_occupancy: int = 1440,
        target_occupancy: int = 480,
        on_discontinuity: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the ring buffer.

        Args:
            capacity: Total power-of-two buffer capacity in samples. Default 2048 (42.67ms @ 48kHz).
            max_occupancy: Maximum permitted samples in queue before dropping oldest (default 1440 = 30ms).
            target_occupancy: Desired occupancy when resynchronizing after overflow (default 480 = 10ms).
            on_discontinuity: Callback invoked when a buffer discontinuity occurs to reset DSP state.
        """
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError(f"Capacity must be a positive power of 2, got {capacity}")
        if max_occupancy > capacity:
            raise ValueError(f"max_occupancy ({max_occupancy}) cannot exceed capacity ({capacity})")

        self.capacity: int = capacity
        self._mask: int = capacity - 1
        self.max_occupancy: int = max_occupancy
        self.target_occupancy: int = target_occupancy
        self.on_discontinuity: Optional[Callable[[], None]] = on_discontinuity

        # Preallocated memory storage (Rule A-01: Zero runtime allocation)
        self._buffer: np.ndarray = np.zeros(self.capacity, dtype=np.float32)
        self._silence_buf: np.ndarray = np.zeros(480, dtype=np.float32)
        self._mono_temp: np.ndarray = np.zeros(capacity, dtype=np.float32)

        # Indices strictly bounded within [0, capacity - 1]
        self._write_idx: int = 0
        self._read_idx: int = 0
        self._occupancy: int = 0

        # Telemetry metrics
        self._dropped_samples: int = 0
        self._overflow_count: int = 0
        self._underrun_count: int = 0
        self._discontinuity_epoch: int = 0

    @property
    def available_read(self) -> int:
        """Number of samples currently available for reading."""
        return self._occupancy

    @property
    def available_write(self) -> int:
        """Remaining capacity before the buffer is completely full."""
        return self.capacity - self._occupancy

    def write(self, in_data: np.ndarray) -> int:
        """Write incoming audio samples into the ring buffer.

        Executed by the Capture Callback (Producer thread).
        Guarantees zero heap allocation in steady state.

        Args:
            in_data: Contiguous float32 array (1D mono or 2D [N, channels] multi-channel).

        Returns:
            Number of samples written.
        """
        if in_data.ndim == 2:
            num_samples = in_data.shape[0]
            if num_samples > self.capacity:
                in_data = in_data[-self.capacity :]
                num_samples = self.capacity
            # In-place stereo/multi-channel downmix to mono in preallocated buffer
            if in_data.shape[1] >= 2:
                np.add(in_data[:, 0], in_data[:, 1], out=self._mono_temp[:num_samples])
                self._mono_temp[:num_samples] *= 0.5
            else:
                np.copyto(self._mono_temp[:num_samples], in_data[:, 0])
            src = self._mono_temp[:num_samples]
        elif in_data.ndim == 1:
            num_samples = len(in_data)
            src = in_data
        else:
            return 0

        if num_samples == 0:
            return 0

        # Queue Age Capping: Check if writing these samples exceeds latency threshold
        projected_occupancy = self._occupancy + num_samples
        if projected_occupancy > self.max_occupancy:
            # Worker lagging: drop oldest unread frame(s) to restore conversational delay (minimal-loss discard)
            excess = projected_occupancy - self.max_occupancy
            frames_to_drop = max(1, int(np.ceil(excess / 480.0)))
            drop_samples = min(frames_to_drop * 480, self._occupancy)

            self._read_idx = (self._read_idx + drop_samples) & self._mask
            self._dropped_samples += drop_samples
            self._overflow_count += 1
            self._discontinuity_epoch += 1

            # Trigger stateful DSP recovery (e.g., reset RNNoise recurrent state)
            if self.on_discontinuity:
                try:
                    self.on_discontinuity()
                except Exception:
                    pass
        else:
            drop_samples = 0

        # Circular buffer copy using power-of-two bitwise masking
        w_idx = self._write_idx
        if w_idx + num_samples <= self.capacity:
            self._buffer[w_idx : w_idx + num_samples] = src
        else:
            first_chunk = self.capacity - w_idx
            second_chunk = num_samples - first_chunk
            self._buffer[w_idx:] = src[:first_chunk]
            self._buffer[:second_chunk] = src[first_chunk:]

        self._write_idx = (w_idx + num_samples) & self._mask
        # Consistent atomic SPSC occupancy update: never temporarily dip below safe lookahead
        self._occupancy = self._occupancy - drop_samples + num_samples
        return int(num_samples)

    def read(self, out_frame: np.ndarray, num_samples: int = 480) -> bool:
        """Read audio samples from the ring buffer.

        Executed by the Processing Worker or Render Callback (Consumer thread).
        Guarantees zero heap allocation.

        Args:
            out_frame: Preallocated output buffer of length >= num_samples.
            num_samples: Number of samples to read (typically 480 for 10ms frame).

        Returns:
            True if sufficient samples were read; False if underflow occurred.
        """
        if len(out_frame) < num_samples:
            return False

        if self._occupancy < num_samples:
            # Underflow: fill with silence, increment counter, fail-closed
            out_frame[:num_samples] = 0.0
            self._underrun_count += 1
            return False

        r_idx = self._read_idx
        if r_idx + num_samples <= self.capacity:
            out_frame[:num_samples] = self._buffer[r_idx : r_idx + num_samples]
        else:
            first_chunk = self.capacity - r_idx
            second_chunk = num_samples - first_chunk
            out_frame[:first_chunk] = self._buffer[r_idx:]
            out_frame[first_chunk:num_samples] = self._buffer[:second_chunk]

        self._read_idx = (r_idx + num_samples) & self._mask
        self._occupancy -= num_samples
        return True

    def peek(self, out_frame: np.ndarray, num_samples: int = 480) -> bool:
        """Inspect audio samples from the ring buffer without advancing read pointer.

        Executed by Consumer / ASRC Resampler.
        Guarantees zero heap allocation and no locks.

        Args:
            out_frame: Preallocated output buffer of length >= num_samples.
            num_samples: Number of samples to peek.

        Returns:
            True if sufficient samples were available and copied; False if underflow.
        """
        if len(out_frame) < num_samples:
            return False

        if self._occupancy < num_samples:
            out_frame[:num_samples] = 0.0
            return False

        r_idx = self._read_idx
        if r_idx + num_samples <= self.capacity:
            out_frame[:num_samples] = self._buffer[r_idx : r_idx + num_samples]
        else:
            first_chunk = self.capacity - r_idx
            second_chunk = num_samples - first_chunk
            out_frame[:first_chunk] = self._buffer[r_idx:]
            out_frame[first_chunk:num_samples] = self._buffer[:second_chunk]

        return True

    def advance_read_ptr(self, num_samples: int) -> int:
        """Advance the read pointer by committed integer consumed samples.

        Executed by Consumer / ASRC Resampler after fractional resampling step.

        Args:
            num_samples: Number of samples to advance.

        Returns:
            Number of samples actually advanced (clamped to available).
        """
        if num_samples <= 0:
            return 0
        clamped = min(num_samples, self._occupancy)
        self._read_idx = (self._read_idx + clamped) & self._mask
        self._occupancy -= clamped
        return clamped

    def reset(self) -> None:
        """Reset the buffer state to empty and clear memory."""
        self._buffer.fill(0.0)
        self._write_idx = 0
        self._read_idx = 0
        self._occupancy = 0
        self._dropped_samples = 0
        self._overflow_count = 0
        self._underrun_count = 0
        self._discontinuity_epoch = 0

    def get_metrics(self) -> dict[str, int]:
        """Return snapshot of ring buffer telemetry."""
        return {
            "capacity": self.capacity,
            "occupancy_samples": self.available_read,
            "dropped_samples": self._dropped_samples,
            "overflow_count": self._overflow_count,
            "underrun_count": self._underrun_count,
            "discontinuity_epoch": self._discontinuity_epoch,
        }
