"""Zero-Allocation Polyphase Audio Resampler (48 kHz <-> 16 kHz).

Bridges the 48 kHz / 480-sample (10 ms) RNNoise noise-suppression domain
and the 16 kHz / 160-sample (10 ms) neural voice conversion engine domain.

DSP Architecture:
- Exact 3:1 integer sample-rate ratio.
- Decimation (48 kHz -> 16 kHz): 72-tap Kaiser-windowed sinc FIR anti-aliasing filter
  with cutoff at 7.2 kHz (Nyquist 8 kHz) and >60 dB stopband attenuation.
- Interpolation (16 kHz -> 48 kHz): 72-tap polyphase anti-imaging FIR filter
  with 3 sub-filters (K=24 taps each) evaluated via SIMD matrix multiplication.
- Zero runtime heap allocation: all sliding windows, index matrices, filter taps,
  and state buffers are preallocated during initialization.
- State continuity across frame boundaries: maintains persistent history buffers
  to guarantee zero phonetic clicks, boundary discontinuities, or phase distortion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hth.domain.errors import AudioProcessingError

# Standard real-time 10 ms frame sizes
FS_48K = 48000
FS_16K = 16000
FRAME_SIZE_48K = 480
FRAME_SIZE_16K = 160
RATIO = 3  # 48000 / 16000


def design_kaiser_sinc_fir(
    num_taps: int = 72,
    cutoff_hz: float = 7200.0,
    fs_hz: float = 48000.0,
    beta: float = 6.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Designs a low-pass linear-phase FIR filter using a Kaiser-windowed sinc.

    Implemented in pure NumPy without requiring external SciPy dependencies.

    Args:
        num_taps: Number of filter taps (must be a multiple of 3).
        cutoff_hz: Low-pass cutoff frequency in Hertz.
        fs_hz: Sampling frequency in Hertz.
        beta: Kaiser window shape parameter (beta=6.0 yields >60 dB attenuation).
        gain: Desired DC passband gain (1.0 for decimation, 3.0 for interpolation).

    Returns:
        float32 1D numpy array of filter coefficients.
    """
    if num_taps % RATIO != 0:
        raise ValueError(f"num_taps must be a multiple of {RATIO}, got {num_taps}")

    m = num_taps - 1
    n = np.arange(num_taps, dtype=np.float64)
    alpha = m / 2.0

    # Kaiser window: I0(beta * sqrt(1 - ((n - alpha) / alpha)^2)) / I0(beta)
    ratio = (n - alpha) / alpha
    arg = np.maximum(0.0, 1.0 - ratio * ratio)
    window = np.i0(beta * np.sqrt(arg)) / np.i0(beta)

    # Normalized sinc: sinc(x) = sin(pi*x) / (pi*x)
    fc_norm = 2.0 * cutoff_hz / fs_hz
    sinc = fc_norm * np.sinc(fc_norm * (n - alpha))

    # Windowed sinc filter
    h = (sinc * window).astype(np.float32)

    # Normalize passband gain
    current_sum = float(np.sum(h))
    if current_sum != 0.0:
        h *= gain / current_sum

    return h


class PolyphaseResampler48To16:
    """Downsamples 48 kHz (480 samples) to 16 kHz (160 samples) with anti-aliasing.

    Guarantees zero heap allocation per frame during real-time streaming.
    """

    def __init__(
        self,
        num_taps: int = 72,
        cutoff_hz: float = 7200.0,
        beta: float = 6.0,
    ) -> None:
        self.num_taps = num_taps
        self.hist_len = num_taps - 1
        self.in_len = FRAME_SIZE_48K
        self.out_len = FRAME_SIZE_16K

        # Anti-aliasing filter taps (gain = 1.0)
        self._filter_taps = design_kaiser_sinc_fir(
            num_taps=num_taps,
            cutoff_hz=cutoff_hz,
            fs_hz=float(FS_48K),
            beta=beta,
            gain=1.0,
        )

        # Preallocated memory buffers (Rule A-01)
        self._history = np.zeros(self.hist_len, dtype=np.float32)
        self._combined = np.zeros(self.hist_len + self.in_len, dtype=np.float32)
        self._windows_buf = np.zeros((self.out_len, self.num_taps), dtype=np.float32)
        self._out_buf = np.zeros(self.out_len, dtype=np.float32)

        # Precompute static index matrix: idx_matrix[m, k] = 3 * m + k
        self._idx_matrix = np.zeros((self.out_len, self.num_taps), dtype=np.intp)
        for m in range(self.out_len):
            self._idx_matrix[m] = RATIO * m + np.arange(self.num_taps)

    def process_frame(
        self,
        in_frame: np.ndarray,
        out_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Processes one 10 ms 48 kHz frame (480 samples) to 16 kHz (160 samples).

        Args:
            in_frame: 1D float32 array with exactly 480 samples.
            out_frame: Optional preallocated destination buffer of length 160.

        Returns:
            160-sample float32 array at 16 kHz.
        """
        if in_frame.size != self.in_len:
            raise AudioProcessingError(
                f"Expected 48 kHz frame with {self.in_len} samples, got {in_frame.size}"
            )

        target = self._out_buf if out_frame is None else out_frame
        if target.size != self.out_len:
            raise AudioProcessingError(
                f"Expected 16 kHz output buffer with {self.out_len} samples, got {target.size}"
            )

        # Flatten if 2D (N, 1)
        flat_in = in_frame.ravel()

        # Fill combined buffer: [history (num_taps-1) | current frame (480)]
        self._combined[: self.hist_len] = self._history
        self._combined[self.hist_len :] = flat_in

        # Zero-allocation sliding-window extraction & inner product
        self._combined.take(self._idx_matrix, out=self._windows_buf)
        np.dot(self._windows_buf, self._filter_taps, out=target)

        # Save history for seamless phase continuity across frame boundaries
        self._history[:] = flat_in[-self.hist_len :]

        return target

    def reset(self) -> None:
        """Clears history buffers on stream restart or discontinuity."""
        self._history.fill(0.0)
        self._combined.fill(0.0)
        self._windows_buf.fill(0.0)
        self._out_buf.fill(0.0)


class PolyphaseResampler16To48:
    """Upsamples 16 kHz (160 samples) to 48 kHz (480 samples) with anti-imaging.

    Evaluates 3 polyphase sub-filters in a single matrix multiplication,
    guaranteeing zero heap allocation and high SIMD performance.
    """

    def __init__(
        self,
        num_taps: int = 72,
        cutoff_hz: float = 7200.0,
        beta: float = 6.0,
    ) -> None:
        self.num_taps = num_taps
        self.taps_per_phase = num_taps // RATIO  # K = 24
        self.hist_len = self.taps_per_phase - 1
        self.in_len = FRAME_SIZE_16K
        self.out_len = FRAME_SIZE_48K

        # Anti-imaging filter taps (gain = 3.0 to preserve power across 1:3 upsampling)
        h_up = design_kaiser_sinc_fir(
            num_taps=num_taps,
            cutoff_hz=cutoff_hz,
            fs_hz=float(FS_48K),
            beta=beta,
            gain=float(RATIO),
        )

        # Polyphase decomposition: matrix of shape (K, 3)
        # Column 0: phase 0 (h[0], h[3], h[6]...)
        # Column 1: phase 1 (h[1], h[4], h[7]...)
        # Column 2: phase 2 (h[2], h[5], h[8]...)
        self._polyphase_matrix = np.zeros((self.taps_per_phase, RATIO), dtype=np.float32)
        for phase in range(RATIO):
            self._polyphase_matrix[:, phase] = h_up[phase::RATIO]

        # Preallocated buffers (Rule A-01)
        self._history = np.zeros(self.hist_len, dtype=np.float32)
        self._combined = np.zeros(self.hist_len + self.in_len, dtype=np.float32)
        self._windows_buf = np.zeros((self.in_len, self.taps_per_phase), dtype=np.float32)

        # 2D output matrix (160, 3) where row m has [y[3m], y[3m+1], y[3m+2]]
        self._out_matrix = np.zeros((self.in_len, RATIO), dtype=np.float32)
        # 1D flattened contiguous view pointing to the same data
        self._out_1d = self._out_matrix.reshape(self.out_len)

        # Precompute static index matrix: idx_matrix[m, k] = m + k
        self._idx_matrix = np.zeros((self.in_len, self.taps_per_phase), dtype=np.intp)
        for m in range(self.in_len):
            self._idx_matrix[m] = m + np.arange(self.taps_per_phase)

    def process_frame(
        self,
        in_frame: np.ndarray,
        out_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Processes one 10 ms 16 kHz frame (160 samples) to 48 kHz (480 samples).

        Args:
            in_frame: 1D float32 array with exactly 160 samples.
            out_frame: Optional preallocated destination buffer of length 480.

        Returns:
            480-sample float32 array at 48 kHz.
        """
        if in_frame.size != self.in_len:
            raise AudioProcessingError(
                f"Expected 16 kHz frame with {self.in_len} samples, got {in_frame.size}"
            )

        # Flatten if 2D (N, 1)
        flat_in = in_frame.ravel()

        # Fill combined buffer: [history (K-1) | current frame (160)]
        self._combined[: self.hist_len] = self._history
        self._combined[self.hist_len :] = flat_in

        # Sliding window extraction on 16 kHz input stream
        self._combined.take(self._idx_matrix, out=self._windows_buf)

        # Matrix multiply: (160, K) @ (K, 3) -> (160, 3)
        np.dot(self._windows_buf, self._polyphase_matrix, out=self._out_matrix)

        # Save history for seamless phase continuity
        self._history[:] = flat_in[-self.hist_len :]

        if out_frame is not None:
            if out_frame.size != self.out_len:
                raise AudioProcessingError(
                    f"Expected 48 kHz output buffer with {self.out_len} samples, got {out_frame.size}"
                )
            out_frame[:] = self._out_1d
            return out_frame

        return self._out_1d

    def reset(self) -> None:
        """Clears history buffers on stream restart or discontinuity."""
        self._history.fill(0.0)
        self._combined.fill(0.0)
        self._windows_buf.fill(0.0)
        self._out_matrix.fill(0.0)


class AudioResampler:
    """Unified bidirectional polyphase audio resampler for the HTH pipeline."""

    def __init__(
        self,
        num_taps: int = 72,
        cutoff_hz: float = 7200.0,
        beta: float = 6.0,
    ) -> None:
        self.downsampler = PolyphaseResampler48To16(
            num_taps=num_taps, cutoff_hz=cutoff_hz, beta=beta
        )
        self.upsampler = PolyphaseResampler16To48(num_taps=num_taps, cutoff_hz=cutoff_hz, beta=beta)

    def downsample_48k_to_16k(
        self,
        in_480: np.ndarray,
        out_160: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Downsamples 480 samples @ 48 kHz to 160 samples @ 16 kHz."""
        return self.downsampler.process_frame(in_480, out_frame=out_160)

    def upsample_16k_to_48k(
        self,
        in_160: np.ndarray,
        out_480: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Upsamples 160 samples @ 16 kHz to 480 samples @ 48 kHz."""
        return self.upsampler.process_frame(in_160, out_frame=out_480)

    def reset(self) -> None:
        """Resets history state in both directions."""
        self.downsampler.reset()
        self.upsampler.reset()
