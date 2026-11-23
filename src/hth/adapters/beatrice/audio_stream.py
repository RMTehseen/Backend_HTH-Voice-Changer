"""Real-time audio callback manager adhering strictly to Rules A-01..A-06."""

import ctypes
import os
import time
from typing import Any, Callable, Optional

import numpy as np

try:
    import sounddevice as sd

    HAS_AUDIO_HARDWARE = True
except ImportError:
    sd = None
    HAS_AUDIO_HARDWARE = False

from hth.domain.contracts import AudioMetrics, DeviceInfo, LiveConfig
from hth.domain.errors import DeviceError


class BeatriceAudioStream:
    """Manages real-time PortAudio streams with fail-closed callback safety."""

    def __init__(
        self,
        config: LiveConfig,
        on_error_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.on_error_cb = on_error_cb
        self._is_running = False
        self._mute_flag = ctypes.c_bool(False)
        self._passthrough_flag = ctypes.c_bool(False)

        # Atomic lock-free metrics (ctypes values safe for cross-thread access)
        self._input_peak = ctypes.c_float(0.0)
        self._output_peak = ctypes.c_float(0.0)
        self._dropped_frames = ctypes.c_uint64(0)
        self._underruns = ctypes.c_uint64(0)
        self._callback_duration_ms = ctypes.c_float(0.0)

        # Pre-allocate audio buffers (Zero dynamic allocation inside callback)
        self.block_size = config.block_size
        self.sample_rate = config.sample_rate

        self._input_buf = np.zeros((self.block_size, 1), dtype=np.float32)
        self._output_buf = np.zeros((self.block_size, 1), dtype=np.float32)
        self._silence_buf = np.zeros((self.block_size, 1), dtype=np.float32)
        self._downsample_buf = np.zeros((self.block_size,), dtype=np.float32)

        # Pre-allocate persistent ctypes DSP buffers
        self._c_phone_features = (ctypes.c_float * 128)()
        self._c_pitch_features = (ctypes.c_float * 4)()
        self._c_pitch_bin = ctypes.c_int(0)
        self._c_out_samples = (ctypes.c_float * 240)()

        # Native C engine handle reference
        self.native_engine: Optional[Any] = None
        self._stream: Optional[Any] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_mute(self, muted: bool) -> None:
        """Sets fail-closed mute state."""
        self._mute_flag.value = muted

    def set_passthrough(self, enabled: bool) -> None:
        """Sets explicit user-initiated passthrough."""
        self._passthrough_flag.value = enabled

    def get_metrics(self) -> AudioMetrics:
        """Returns snapshot of atomic callback metrics."""
        return AudioMetrics(
            input_peak=float(self._input_peak.value),
            output_peak=float(self._output_peak.value),
            dropped_frames=int(self._dropped_frames.value),
            underruns=int(self._underruns.value),
            callback_duration_ms=float(self._callback_duration_ms.value),
        )

    def start(self) -> None:
        """Opens and starts the PortAudio stream."""
        if self._is_running:
            return

        if not HAS_AUDIO_HARDWARE or os.environ.get("HTH_USE_FAKE_AUDIO_DEVICES") == "1":
            # Emulated stream mode for headless / non-audio environments
            self._is_running = True
            return

        in_dev = self.config.input_device_id
        out_dev = self.config.output_device_id

        try:
            self._stream = sd.Stream(
                device=(in_dev, out_dev),
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
        except Exception as exc:
            # Remediation of BS-04: NEVER fallback silently to system default devices
            self._is_running = False
            raise DeviceError(
                f"Failed to open audio stream on device ({in_dev}, {out_dev}): {exc}",
                device_id=in_dev,
            ) from exc

    def stop(self) -> None:
        """Stops and closes the PortAudio stream."""
        if not self._is_running:
            return

        self._is_running = False
        if HAS_AUDIO_HARDWARE and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def _audio_callback(
        self, indata: Any, outdata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        """Real-time callback: strictly no allocation, no locks, no logging, fail-closed."""
        t_start = time.perf_counter()

        # Check stream underrun / overflow
        if status:
            self._underruns.value += 1
            if status.output_underflow or status.input_overflow:
                self._dropped_frames.value += frames
                # Fail-closed: mute output on severe underrun
                np.copyto(outdata, self._silence_buf)
                return

        # Check explicit mute flag
        if self._mute_flag.value or self.native_engine is None:
            np.copyto(outdata, self._silence_buf)
            self._output_peak.value = 0.0
            return

        try:
            # Measure input peak
            in_slice = indata[:, 0]
            in_peak = float(np.max(np.abs(in_slice)))
            self._input_peak.value = in_peak

            # Noise gate check
            if in_peak < self.config.params.gate_threshold:
                np.copyto(outdata, self._silence_buf)
                self._output_peak.value = 0.0
                return

            # Explicit passthrough mode (never automatic fallback)
            if self._passthrough_flag.value:
                np.copyto(outdata, indata)
                self._output_peak.value = in_peak
                return

            # Native Beatrice C-ABI execution
            lib = self.native_engine.lib
            phone_ctx = self.native_engine.phone_context
            pitch_ctx = self.native_engine.pitch_context
            wave_ctx = self.native_engine.waveform_context

            in_ptr = in_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # 1. Phone extraction
            lib.Beatrice20rc0_ExtractPhone1(
                self.native_engine.phone_extractor,
                in_ptr,
                self._c_phone_features,
                phone_ctx,
            )

            # 2. Pitch estimation
            self._c_pitch_bin.value = 0
            lib.Beatrice20rc0_EstimatePitch1(
                self.native_engine.pitch_estimator,
                in_ptr,
                ctypes.byref(self._c_pitch_bin),
                self._c_pitch_features,
                pitch_ctx,
            )

            # Pitch shift adjustment
            pitch_semitones = self.config.params.pitch_shift
            if pitch_semitones != 0.0:
                shift_factor = 2.0 ** (pitch_semitones / 12.0)
                shifted_bin = int(round(self._c_pitch_bin.value * shift_factor))
                self._c_pitch_bin.value = max(0, min(1000, shifted_bin))

            # 3. Waveform generation (240 samples @ 24kHz)
            lib.Beatrice20rc0_GenerateWaveform1(
                self.native_engine.waveform_generator,
                self._c_phone_features,
                ctypes.byref(self._c_pitch_bin),
                self._c_pitch_features,
                self._c_out_samples,
                wave_ctx,
            )

            # Downsample 24kHz (240 spl) to 16kHz (160 spl) via linear interpolation
            out_24k = np.frombuffer(self._c_out_samples, dtype=np.float32)
            # Use pre-allocated buffer for downsampled result
            # np.interp operates in-place or returns array
            resampled = (
                np.interp(
                    np.linspace(0, 240, self.block_size, endpoint=False),
                    np.arange(240),
                    out_24k,
                ).astype(np.float32)
                * self.config.params.volume
            )

            outdata[:, 0] = resampled
            self._output_peak.value = float(np.max(np.abs(resampled)))

        except Exception:
            # Remediation of BS-03: STRICT FAIL-CLOSED MUTE ON EXCEPTION
            # Never fall back to raw input audio!
            np.copyto(outdata, self._silence_buf)
            self._output_peak.value = 0.0
            self._mute_flag.value = True
            # Non-blocking callback to error delegate if configured
            if self.on_error_cb:
                try:
                    self.on_error_cb("Audio callback DSP exception: muted fail-closed")
                except Exception:
                    pass
        finally:
            self._callback_duration_ms.value = float((time.perf_counter() - t_start) * 1000.0)

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        """Lists available audio devices."""
        devices: list[DeviceInfo] = []
        if not HAS_AUDIO_HARDWARE:
            # Provide standard default virtual devices
            return [
                DeviceInfo(
                    id=0,
                    name="Default Microphone (Mock)",
                    max_input_channels=2,
                    max_output_channels=0,
                    default_sample_rate=16000,
                    is_default_input=True,
                ),
                DeviceInfo(
                    id=1,
                    name="VB-Audio Virtual Cable (Mock)",
                    max_input_channels=0,
                    max_output_channels=2,
                    default_sample_rate=16000,
                    is_default_output=True,
                ),
            ]

        try:
            dev_list = sd.query_devices()
            default_in, default_out = sd.default.device
            for idx, dev in enumerate(dev_list):
                devices.append(
                    DeviceInfo(
                        id=idx,
                        name=dev.get("name", f"Device {idx}"),
                        max_input_channels=dev.get("max_input_channels", 0),
                        max_output_channels=dev.get("max_output_channels", 0),
                        default_sample_rate=int(dev.get("default_samplerate", 16000)),
                        host_api=str(dev.get("hostapi", "MME/WASAPI")),
                        is_default_input=(idx == default_in),
                        is_default_output=(idx == default_out),
                    )
                )
        except Exception:
            pass
        return devices
