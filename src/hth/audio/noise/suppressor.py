"""Modular real-time Noise Suppressor adhering to Rules A-01..A-06.

Adherence guarantees:
- A-01 (Zero-Allocation): Pre-allocates all float32 memory and ctypes pointers during init.
- A-02 (Lock-Free Callback): No mutex, heap allocation, filesystem, network I/O, or logging in process_frame.
- A-03 (Fail-Closed Mute): Any unexpected C-ABI error immediately zeroes output buffer; never leaks raw mic.
- Click-Free Transitions: Linearly ramps alpha blending across 480 samples during preset switching.
- Standard 48kHz / 480-sample frame architecture.
"""

from __future__ import annotations

import ctypes
import time
from typing import Optional

import numpy as np

from hth.audio.noise.contracts import NoiseMetrics, NoisePreset
from hth.domain.errors import AudioProcessingError

FRAME_SIZE: int = 480
SAMPLE_RATE: int = 48000

# Preset wet/dry blend alphas (0.0 = full dry bypass, 1.0 = full wet denoise)
PRESET_ALPHAS: dict[NoisePreset, float] = {
    NoisePreset.OFF: 0.0,
    NoisePreset.LIGHT: 0.60,  # Preserves consonant warmth, suppresses fan/hum
    NoisePreset.STRONG: 1.0,  # Full neural recurrent suppression
}


class NoiseSuppressor:
    """Preallocated, zero-allocation real-time RNNoise DSP processor."""

    def __init__(self, preset: NoisePreset = NoisePreset.LIGHT) -> None:
        self._preset = preset
        self._current_alpha: float = PRESET_ALPHAS[preset]
        self._target_alpha: float = PRESET_ALPHAS[preset]
        self._state: Optional[ctypes.c_void_p] = None
        self._is_initialized = False

        # Preallocated buffers (Rule A-01: Zero allocation in audio callback)
        self._in_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._out_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._ramp_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._inv_ramp_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._blend_dry_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._blend_wet_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._silence_buf = np.zeros(FRAME_SIZE, dtype=np.float32)

        # Pre-bind ctypes pointers to avoid pointer allocation in hot path
        self._in_ptr = self._in_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._out_ptr = self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Metrics buffer (preallocated numpy ring buffer; zero object allocation in callback)
        self._latencies_buf = np.zeros(200, dtype=np.float32)
        self._latencies_idx: int = 0
        self._latencies_count: int = 0
        self._frames_processed: int = 0
        self._underrun_count: int = 0
        self._last_vad_prob: float = 0.0
        self._has_error: bool = False

        self._init_c_abi()

    def _init_c_abi(self) -> None:
        """Locates and loads the RNNoise C-ABI dynamic library."""
        try:
            from pyrnnoise.rnnoise import lib as rnn_lib

            self._lib = rnn_lib
            self._state = self._lib.rnnoise_create(None)
            if not self._state:
                raise AudioProcessingError("Failed to initialize native RNNoise state pointer.")
            self._is_initialized = True
        except Exception as exc:
            self._has_error = True
            raise AudioProcessingError(f"Failed to load RNNoise C-ABI: {exc}") from exc

    def set_preset(self, preset: NoisePreset) -> None:
        """Atomic target preset switch with smooth linear crossfade."""
        self._preset = preset
        self._target_alpha = PRESET_ALPHAS[preset]

    def get_preset(self) -> NoisePreset:
        return self._preset

    def process_frame(self, in_frame: np.ndarray, out_frame: np.ndarray) -> float:
        """Real-time audio callback execution.

        Args:
            in_frame: Contiguous 1D float32 array with exactly 480 samples [-1.0, 1.0].
            out_frame: Preallocated output buffer of 480 float32 samples.

        Returns:
            Speech probability (0.0 to 1.0) estimated by RNNoise VAD.

        Guarantees:
            - Zero heap allocation.
            - Zero blocking calls or locks.
            - Output muted on error (Fail-Closed, Rule A-03).
        """
        # Fail-closed guard: if engine had fatal error or uninitialized, zero out buffer
        if not self._is_initialized or self._has_error or self._state is None:
            np.copyto(out_frame, self._silence_buf)
            return 0.0

        if len(in_frame) != FRAME_SIZE:
            # Dimension mismatch: fail-closed mute immediately
            np.copyto(out_frame, self._silence_buf)
            return 0.0

        t0 = time.perf_counter()

        try:
            # If preset is OFF and transition is complete, bypass DSP
            if self._current_alpha == 0.0 and self._target_alpha == 0.0:
                np.copyto(out_frame, in_frame)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self._record_metrics(dt_ms, 0.0)
                return 0.0

            # 1. Scale [-1.0, 1.0] samples to RNNoise expected range [-32768, 32767] in-place
            np.copyto(self._in_buf, in_frame)
            self._in_buf *= 32767.0

            # 2. Execute C-ABI neural recurrent inference
            vad_prob = float(
                self._lib.rnnoise_process_frame(self._state, self._out_ptr, self._in_ptr)
            )
            self._last_vad_prob = vad_prob

            # 3. Rescale output back to [-1.0, 1.0]
            self._out_buf *= 1.0 / 32767.0

            # 4. Click-Free Crossfade Ramp:
            curr = self._current_alpha
            target = self._target_alpha

            if curr != target:
                # Interpolate alpha across the frame
                step = (target - curr) / float(FRAME_SIZE)
                # Compute ramp into preallocated buffer without allocation
                for i in range(FRAME_SIZE):
                    r = curr + (i * step)
                    self._ramp_buf[i] = r
                    self._inv_ramp_buf[i] = 1.0 - r
                self._current_alpha = target
                # Apply wet/dry blend in-place: out = (1 - ramp) * raw + ramp * denoised
                np.multiply(self._inv_ramp_buf, in_frame, out=self._blend_dry_buf)
                np.multiply(self._ramp_buf, self._out_buf, out=self._blend_wet_buf)
                np.add(self._blend_dry_buf, self._blend_wet_buf, out=out_frame)
            elif curr == 1.0:
                # Full wet: copy directly into out_frame
                np.copyto(out_frame, self._out_buf)
            else:
                # Constant alpha wet/dry blend in-place
                inv_curr = 1.0 - curr
                np.multiply(in_frame, inv_curr, out=self._blend_dry_buf)
                np.multiply(self._out_buf, curr, out=self._blend_wet_buf)
                np.add(self._blend_dry_buf, self._blend_wet_buf, out=out_frame)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._record_metrics(dt_ms, vad_prob)
            return vad_prob

        except Exception:
            # Rule A-03: Fail-closed mute immediately on error
            self._has_error = True
            np.copyto(out_frame, self._silence_buf)
            return 0.0

    def _record_metrics(self, dt_ms: float, vad_prob: float) -> None:
        """Lock-free metric recording into preallocated ring buffer."""
        self._latencies_buf[self._latencies_idx] = dt_ms
        self._latencies_idx = (self._latencies_idx + 1) % 200
        if self._latencies_count < 200:
            self._latencies_count += 1
        self._frames_processed += 1
        if dt_ms > 10.0:
            self._underrun_count += 1

    def get_metrics(self) -> NoiseMetrics:
        """Snapshot of real-time metrics for UI/API telemetry."""
        if self._latencies_count == 0:
            return NoiseMetrics(
                frames_processed=self._frames_processed,
                p50_ms=0.0,
                p95_ms=0.0,
                max_ms=0.0,
                underrun_count=self._underrun_count,
                last_speech_probability=self._last_vad_prob,
                active_preset=self._preset,
            )

        valid_slice = self._latencies_buf[: self._latencies_count]
        return NoiseMetrics(
            frames_processed=self._frames_processed,
            p50_ms=float(np.percentile(valid_slice, 50)),
            p95_ms=float(np.percentile(valid_slice, 95)),
            max_ms=float(np.max(valid_slice)),
            underrun_count=self._underrun_count,
            last_speech_probability=self._last_vad_prob,
            active_preset=self._preset,
        )

    def reset_state(self) -> None:
        """Stateful DSP reset: destroys and recreates recurrent GRU state to clear history after discontinuity."""
        if hasattr(self, "_lib") and self._lib:
            if self._state:
                try:
                    self._lib.rnnoise_destroy(self._state)
                except Exception:
                    pass
                self._state = None
            try:
                self._state = self._lib.rnnoise_create(None)
                self._has_error = False
            except Exception:
                self._has_error = True
        self._in_buf.fill(0.0)
        self._out_buf.fill(0.0)
        self._current_alpha = PRESET_ALPHAS[self._preset]
        self._target_alpha = PRESET_ALPHAS[self._preset]

    def close(self) -> None:
        """Safely destroy C-ABI context."""
        if self._state and hasattr(self, "_lib") and self._lib:
            try:
                self._lib.rnnoise_destroy(self._state)
            except Exception:
                pass
            self._state = None
        self._is_initialized = False

    def __del__(self) -> None:
        self.close()
