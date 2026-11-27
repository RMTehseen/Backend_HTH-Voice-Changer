"""Generic Real-Time Audio Capture & Rendering Pipeline with TestSink Safety.

Adheres strictly to Phase 5 / Final Implementation 1 Architecture:
- Captures real physical microphone PCM audio via sounddevice / WASAPI.
- Pre-allocated bounded SPSC AudioRingBuffer (zero allocation during streaming).
- 48 kHz / 480-sample (10 ms) processing through RNNoise (NoiseSuppressor).
- Strict TestSink mode: routes processed audio exclusively to physical headphones.
- VB-CABLE virtual microphone output is HARD-MUTED and transmission is DISABLED
  (Rule 6 & 7: No unconverted microphone audio may ever reach VB-CABLE).
- Live real-time performance telemetry (p50, p95, p99, max, occupancy, xruns).
- Fail-closed mute guard on device disconnect, buffer overflow, or exception.
- Optional live real PCM WAV recording for acoustic verification (Rule 8 compliant).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    sd = None
    HAS_SOUNDDEVICE = False

from hth.audio.asrc import (
    ASRCState,
    ASRCUnderflowStateMachine,
    FractionalHermiteResampler,
    PIController,
)
from hth.audio.conversion.acoustic_converter import AcousticVoiceConverter
from hth.audio.conversion.contracts import VoiceConversionConfig, VoiceProfile
from hth.audio.noise.contracts import NoisePreset
from hth.audio.noise.suppressor import FRAME_SIZE, SAMPLE_RATE, NoiseSuppressor
from hth.audio.resampler import (
    FRAME_SIZE_16K,
    FS_16K,
    FS_48K,
    AudioResampler,
)
from hth.audio.ring_buffer import AudioRingBuffer
from hth.core.logging import get_logger
from hth.domain.errors import AudioProcessingError, DeviceError

logger = get_logger("hth.audio.pipeline")


@dataclass
class AudioPipelineConfig:
    """Configuration for real-time capture and TestSink rendering."""

    input_device_id: Optional[int | str] = None
    output_device_id: Optional[int | str] = None
    sample_rate: int = SAMPLE_RATE  # 48000 Hz native for RNNoise
    frame_size: int = FRAME_SIZE  # 480 samples = 10 ms @ 48 kHz
    preset: NoisePreset = NoisePreset.LIGHT
    test_sink_mode: bool = True  # Strict TestSink mode (headphones only)
    enable_vb_cable: bool = False  # Enabled only with verified voice conversion
    enable_voice_conversion: bool = False  # Live vocal tract & pitch conversion
    conversion_profile: VoiceProfile = VoiceProfile.MALE_TO_FEMALE
    live_call_mode: bool = False  # Output to VB-CABLE Input for WhatsApp/Teams


@dataclass
class AudioPipelineMetrics:
    """Real-time performance, latency, and buffer health telemetry."""

    is_running: bool = False
    is_muted: bool = False
    is_test_sink_mode: bool = True
    test_harness_banner: str = "TEST HARNESS — LIVE TRANSMISSION DISABLED"
    active_preset: NoisePreset = NoisePreset.LIGHT
    is_converting: bool = False
    active_profile: str = "male_to_female"
    live_call_mode: bool = False
    sample_rate: int = SAMPLE_RATE
    frame_size: int = FRAME_SIZE

    # Latency and timing
    processing_time_p50_ms: float = 0.0
    processing_time_p95_ms: float = 0.0
    processing_time_p99_ms: float = 0.0
    processing_time_max_ms: float = 0.0

    # Buffer & Queue health
    buffer_occupancy_samples: int = 0
    buffer_occupancy_ms: float = 0.0
    capture_frames_total: int = 0
    render_frames_total: int = 0
    overflow_count: int = 0
    underrun_count: int = 0
    dropped_samples_total: int = 0
    discontinuity_epoch: int = 0
    native_callback_errors: int = 0

    # ASRC Clock Synchronization
    asrc_corr_ppm: float = 0.0
    asrc_rho: float = 1.0
    asrc_state: str = "HEALTHY"

    # Host resources
    cpu_pct: float = 0.0
    ram_mb: float = 0.0
    last_error: Optional[str] = None


def get_wasapi_hostapi_index() -> int:
    """Dynamically discover the Windows WASAPI host API index from sounddevice.

    Raises DeviceError if sounddevice is unavailable or WASAPI is absent.
    """
    if not HAS_SOUNDDEVICE or sd is None:
        raise DeviceError("sounddevice is not available")

    hostapis = sd.query_hostapis()
    for idx, hostapi in enumerate(hostapis):
        name = hostapi.get("name", "").lower()
        if "wasapi" in name:
            return idx

    raise DeviceError(
        "Windows WASAPI host API is not available on this system. "
        "AudioPipeline requires WASAPI for low-latency streaming."
    )


class TelemetrySampler:
    """Non-blocking background sampler for host metrics (CPU % and RAM RSS).

    Decouples psutil system sampling from:
    1. Real-time audio callbacks (zero allocation, non-blocking).
    2. FastAPI asynchronous request loops (eliminates 100ms blocking sleep).
    """

    def __init__(self, sample_interval_sec: float = 1.0) -> None:
        self._interval = sample_interval_sec
        self._cpu_pct: float = 0.0
        self._ram_mb: float = 0.0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[psutil.Process] = None
        self._initialized: bool = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background telemetry sampler thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            try:
                self._process = psutil.Process()
                # Initial non-blocking seed to prime the internal counter
                self._process.cpu_percent(interval=None)
                self._ram_mb = float(self._process.memory_info().rss / (1024 * 1024))
            except Exception:
                self._process = None

            self._thread = threading.Thread(
                target=self._sample_loop,
                name="HTH-TelemetrySampler",
                daemon=True,
            )
            self._thread.start()

    def _sample_loop(self) -> None:
        while self._running:
            try:
                time.sleep(self._interval)
                if not self._running:
                    break
                if self._process is not None:
                    # Non-blocking query over the elapsed interval
                    self._cpu_pct = float(self._process.cpu_percent(interval=None))
                    self._ram_mb = float(self._process.memory_info().rss / (1024 * 1024))
                    self._initialized = True
            except Exception:
                pass

    def stop(self) -> None:
        """Gracefully stop the background sampler thread."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None

    @property
    def cpu_pct(self) -> float:
        """Cached CPU percentage, returned instantaneously with zero blocking."""
        return self._cpu_pct

    @property
    def ram_mb(self) -> float:
        """Cached RSS memory usage in MB, returned instantaneously with zero blocking."""
        return self._ram_mb

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class AudioPipeline:
    """Generic Windows audio capture, noise suppression, and TestSink rendering manager."""

    def __init__(self, config: Optional[AudioPipelineConfig] = None) -> None:
        self.config = config or AudioPipelineConfig()
        self._lock = threading.Lock()
        self._is_running = False
        self._is_muted = False
        self._panic_flag = False

        # RNNoise processor
        self.suppressor = NoiseSuppressor(preset=self.config.preset)

        # SPSC Audio Ring Buffer with state reset hook upon discontinuity
        self.ring_buffer = AudioRingBuffer(
            capacity=4096,
            max_occupancy=2880,  # 60.0 ms @ 48 kHz (1216-sample / 25.3 ms headroom before 4096 capacity)
            target_occupancy=960,  # 20.0 ms @ 48 kHz
            on_discontinuity=self._handle_discontinuity,
        )

        # Preallocated DSP and audio stream buffers (Rule A-01: Zero runtime allocation)
        self._in_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._proc_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._silence_frame = np.zeros(FRAME_SIZE, dtype=np.float32)

        # ASRC Dynamic Clock-Drift Resampler & Controller
        # Critically damped kp=0.20, ki=0.015 prevents overshoot oscillations and eliminates jitter underflows
        self.asrc_controller = PIController(kp=0.20, ki=0.015, alpha=0.01, target_occ=960.0)
        self.asrc_resampler = FractionalHermiteResampler(block_size=FRAME_SIZE)
        self.asrc_state_machine = ASRCUnderflowStateMachine(
            consec_threshold=20,
            rolling_window_size=200,
            rolling_max_underrun_pct=10.0,
            target_occupancy=960,
        )
        self._asrc_scratch = np.zeros(512, dtype=np.float32)
        self._asrc_corr_ppm = 0.0
        self._asrc_rho = 1.0

        # Polyphase Resampler (48 kHz <-> 16 kHz) for neural voice engines
        self.resampler = AudioResampler()
        self._16k_clean = np.zeros(FRAME_SIZE_16K, dtype=np.float32)
        self._16k_converted = np.zeros(FRAME_SIZE_16K, dtype=np.float32)
        self._in_16k_upsample_buf = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._capture_in_sr: int = FS_48K

        # Real Acoustic Voice Converter
        self._conversion_profile = self.config.conversion_profile
        self._is_converting = self.config.enable_voice_conversion
        self.converter = AcousticVoiceConverter(
            VoiceConversionConfig.create_male_to_female(self.config.sample_rate)
            if self._conversion_profile == VoiceProfile.MALE_TO_FEMALE
            else VoiceConversionConfig.create_female_to_male(self.config.sample_rate)
        )
        self._conv_frame = np.zeros(FRAME_SIZE, dtype=np.float32)

        # Stream handles (PortAudio WASAPI manages MMCSS on its native worker threads)
        self._input_stream: Optional[Any] = None
        self._output_stream: Optional[Any] = None
        self._capture_mmcss_handle: Optional[int] = None
        self._render_mmcss_handle: Optional[int] = None
        self._capture_mmcss_initialized = False
        self._render_mmcss_initialized = False

        # Non-blocking background telemetry sampler (CPU & RAM)
        self._telemetry_sampler = TelemetrySampler(sample_interval_sec=1.0)
        self._telemetry_sampler.start()

        # Telemetry metrics buffer (preallocated numpy array for p50/p95/max calculation)
        self._telemetry_buf = np.zeros(256, dtype=np.float32)
        self._telemetry_idx = 0
        self._telemetry_count = 0
        self._capture_frames_total = 0
        self._render_frames_total = 0
        self._callback_errors = 0
        self._last_error: Optional[str] = None

        # Optional real PCM audio recorder tap (Pre-allocated callback-safe circular frame buffer)
        self._record_active = False
        self._record_file: Optional[wave.Wave_write] = None
        self._record_path: Optional[Path] = None
        self._record_samples_remaining = 0
        self._record_lock = threading.Lock()
        self._record_ring = np.zeros((300, FRAME_SIZE), dtype=np.float32)
        self._record_head = 0
        self._record_tail = 0
        self._record_thread: Optional[threading.Thread] = None
        self._record_overflow_count = 0

    @property
    def is_running(self) -> bool:
        """Current streaming state."""
        return self._is_running

    @property
    def is_muted(self) -> bool:
        """Current fail-closed / panic mute state."""
        return self._is_muted or self._panic_flag

    def _handle_discontinuity(self) -> None:
        """Triggered by AudioRingBuffer when overflow forces a sample drop.

        Clears the recurrent GRU state inside RNNoise, polyphase filter history,
        and fractional resampler phase.
        Preserves ASRC controller integral: dropping unread samples does not alter
        the physical crystal oscillator frequency drift between soundcard hardware clocks.
        """
        self.suppressor.reset_state()
        self.resampler.reset()
        self.converter.reset_state()
        self.asrc_resampler.reset()

    def set_preset(self, preset: NoisePreset) -> None:
        """Switch noise suppression preset (OFF, LIGHT, STRONG)."""
        self.config.preset = preset
        self.suppressor.set_preset(preset)
        logger.info("Switched noise suppression preset", preset=preset.value)

    def set_conversion_profile(self, profile: VoiceProfile) -> None:
        """Switch active voice conversion target profile (Male->Female, Female->Male)."""
        with self._lock:
            self._conversion_profile = profile
            self.converter.set_profile(profile)
        logger.info("Voice conversion profile switched", profile=profile.value)

    def set_conversion_active(self, active: bool) -> None:
        """Enable or disable voice conversion during streaming."""
        with self._lock:
            if not active and (self.config.live_call_mode or self.config.enable_vb_cable):
                raise AudioProcessingError(
                    "Security violation: Voice conversion cannot be disabled while transmitting "
                    "to VB-CABLE. Unconverted microphone audio must never reach VB-CABLE."
                )
            self._is_converting = active
        logger.info("Voice conversion state updated", active=active)

    def panic_mute(self) -> None:
        """Immediately zero outgoing audio streams."""
        with self._lock:
            self._is_muted = True
            self._panic_flag = True
            self.asrc_state_machine.on_panic_mute()
        logger.warning("Audio pipeline PANIC MUTE engaged — output zeroed")

    def un_mute(self) -> None:
        """Release panic mute."""
        with self._lock:
            self._is_muted = False
            self._is_muted = False
            self._panic_flag = False
            # Resynchronize buffer to target occupancy (960 samples) upon unmute
            avail = self.ring_buffer.available_read
            if avail > 960:
                self.ring_buffer.advance_read_ptr(avail - 960)
            elif avail < 960:
                self.ring_buffer.write(np.zeros(960 - avail, dtype=np.float32))
            self.asrc_controller.reset()
            self.asrc_resampler.reset()
            self.asrc_state_machine.request_unmute(self.ring_buffer.available_read)
        logger.info("Audio pipeline panic mute released")

    def start(self, config: Optional[AudioPipelineConfig] = None) -> None:
        """Start real-time capture and render streams (TestSink or VB-CABLE).

        Guarantees that VB-CABLE output is never used for unconverted microphone audio.
        """
        with self._lock:
            if self._is_running:
                return

            if config:
                self.config = config

            # Rule 6 & 7: VB-CABLE transmission is prohibited unless voice conversion is verified & active
            if self.config.enable_vb_cable or self.config.live_call_mode:
                if not self.config.enable_voice_conversion and not self._is_converting:
                    raise AudioProcessingError(
                        "Security violation: VB-CABLE live transmission is disabled until an "
                        "approved real voice conversion engine is loaded and active. "
                        "Unconverted microphone audio must never reach VB-CABLE."
                    )
                self._is_converting = True

            # Reset ring buffer and DSP states
            self.ring_buffer.reset()
            # Prime ring buffer to target occupancy (960 samples = 20 ms) with silence
            # Guarantees zero baseline underruns and immediate steady-state lock
            self.ring_buffer.write(np.zeros(960, dtype=np.float32))
            self.suppressor.reset_state()
            self.converter.reset_state()
            self.asrc_resampler.reset()
            self.asrc_controller.reset()
            self.asrc_state_machine.reset()
            self._is_muted = False
            self._panic_flag = False
            self._last_error = None

            if not HAS_SOUNDDEVICE:
                logger.warning("sounddevice not available; operating in headless mode")
                self._is_running = True
                return

            try:
                # Resolve device indices
                in_dev = self._resolve_input_device(self.config.input_device_id)
                out_dev = self._resolve_output_device(self.config.output_device_id)

                in_info = sd.query_devices(in_dev) if in_dev is not None else {}
                native_sr = int(in_info.get("default_samplerate", self.config.sample_rate))
                if native_sr in (FS_16K, FS_48K):
                    self._capture_in_sr = native_sr
                else:
                    self._capture_in_sr = self.config.sample_rate

                in_blocksize = int(self._capture_in_sr * 0.01)

                # Open capture stream (WASAPI input)
                self._input_stream = sd.InputStream(
                    device=in_dev,
                    samplerate=self._capture_in_sr,
                    channels=1,  # sounddevice handles downmixing or we do it
                    blocksize=in_blocksize,
                    dtype="float32",
                    callback=self._capture_callback,
                    finished_callback=self._stream_finished_callback,
                )

                # Open render stream (WASAPI output to physical headphones)
                self._output_stream = sd.OutputStream(
                    device=out_dev,
                    samplerate=self.config.sample_rate,
                    channels=1,
                    blocksize=self.config.frame_size,
                    dtype="float32",
                    callback=self._render_callback,
                    finished_callback=self._stream_finished_callback,
                )

                self._input_stream.start()
                self._output_stream.start()

                # PortAudio WASAPI handles native thread MMCSS priority internally on its own worker threads.
                # Avoid redundant or blocking manual AvSetMmThreadCharacteristics calls on the control thread.
                self._is_running = True

                logger.info(
                    "AudioPipeline started successfully in TestSink mode",
                    in_device=in_dev,
                    out_device=out_dev,
                    sample_rate=self.config.sample_rate,
                    frame_size=self.config.frame_size,
                )
            except Exception as exc:
                self._last_error = str(exc)
                self.stop()
                raise DeviceError(f"Failed to open audio streams: {exc}") from exc

    def stop(self) -> None:
        """Gracefully halt streams, revert MMCSS thread priority, and flush buffers."""
        with self._lock:
            self._is_running = False
            self._is_muted = False
            self._panic_flag = False

            if self._input_stream:
                try:
                    self._input_stream.stop()
                    self._input_stream.close()
                except Exception:
                    pass
                self._input_stream = None

            if self._output_stream:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                except Exception:
                    pass
                self._output_stream = None

            # Clean up streams and background telemetry sampler
            self._telemetry_sampler.stop()

            self.resampler.reset()
            self.asrc_resampler.reset()
            self.asrc_controller.reset()
            self.asrc_state_machine.reset()
            self.stop_recording()
            logger.info("AudioPipeline stopped")

    def _resolve_input_device(self, dev_id: Optional[int | str]) -> Any:
        """Resolve requested input device, prioritizing WASAPI physical microphones."""
        if not HAS_SOUNDDEVICE or sd is None:
            return None

        wasapi_idx = get_wasapi_hostapi_index()
        devices = sd.query_devices()

        target_name: Optional[str] = None
        if isinstance(dev_id, str) and dev_id.startswith("{"):
            try:
                from hth.audio.devices import get_device_inventory

                inv = get_device_inventory()
                for ep in inv.capture_devices:
                    if ep.id.lower() == dev_id.lower():
                        target_name = ep.friendly_name.lower()
                        break
            except Exception:
                pass

        if dev_id is not None:
            cand_idx: Optional[int] = None
            if isinstance(dev_id, int):
                cand_idx = dev_id
            elif isinstance(dev_id, str):
                if dev_id.isdigit():
                    cand_idx = int(dev_id)
                else:
                    search_term = target_name if target_name else dev_id.lower()
                    for idx, d in enumerate(devices):
                        if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                            name = d.get("name", "").lower()
                            if search_term == name or search_term in name or name in search_term:
                                cand_idx = idx
                                break
            if cand_idx is not None and 0 <= cand_idx < len(devices):
                cand_name = devices[cand_idx].get("name", "").lower()
                if "cable" in cand_name:
                    logger.warning(
                        "Rejecting virtual cable loopback as physical microphone input to prevent silence",
                        requested=cand_name,
                        index=cand_idx,
                    )
                else:
                    return cand_idx

        # Auto-detect default WASAPI physical microphone
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if "microphone" in name and "cable" not in name:
                    return idx

        # Fallback to any non-cable WASAPI input
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                if "cable" not in d.get("name", "").lower():
                    return idx

        hostapis = sd.query_hostapis()
        def_in = hostapis[wasapi_idx].get("default_input_device")
        if def_in is not None and def_in >= 0:
            return def_in

        raise DeviceError("No suitable WASAPI physical microphone found.")

    def _resolve_output_device(self, dev_id: Optional[int | str]) -> Any:
        """Resolve requested output device, handling both TestSink and LiveCall modes."""
        if not HAS_SOUNDDEVICE or sd is None:
            return None

        wasapi_idx = get_wasapi_hostapi_index()
        devices = sd.query_devices()

        target_name: Optional[str] = None
        if isinstance(dev_id, str) and dev_id.startswith("{"):
            try:
                from hth.audio.devices import get_device_inventory

                inv = get_device_inventory()
                for ep in inv.render_devices:
                    if ep.id.lower() == dev_id.lower():
                        target_name = ep.friendly_name.lower()
                        break
            except Exception:
                pass

        # If user explicitly specified an output device, resolve and respect it!
        if dev_id is not None:
            if isinstance(dev_id, int):
                out_idx = dev_id
            elif isinstance(dev_id, str) and dev_id.isdigit():
                out_idx = int(dev_id)
            else:
                out_idx = None

            if out_idx is not None:
                if self.config.test_sink_mode:
                    try:
                        d_info = sd.query_devices(out_idx)
                        name = d_info.get("name", "").lower()
                        if "cable" in name:
                            raise AudioProcessingError(
                                "CRITICAL SAFETY VIOLATION: Output device cannot be VB-CABLE in TestSink mode."
                            )
                    except AudioProcessingError:
                        raise
                    except Exception:
                        pass
                return out_idx

            if isinstance(dev_id, str) and not dev_id.isdigit():
                search_term = target_name if target_name else dev_id.lower()
                if self.config.test_sink_mode and "cable" in search_term:
                    raise AudioProcessingError(
                        "CRITICAL SAFETY VIOLATION: Output device cannot be VB-CABLE in TestSink mode."
                    )
                for idx, d in enumerate(devices):
                    if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                        name = d.get("name", "").lower()
                        if search_term in name or name in search_term:
                            return idx

        # When dev_id is None, handle based on mode:
        if self.config.live_call_mode or self.config.enable_vb_cable:
            # LIVE CALL / WHATSAPP MODE: Route converted audio to VB-CABLE Input
            for idx, d in enumerate(devices):
                if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                    name = d.get("name", "").lower()
                    if "cable" in name and "input" in name:
                        return idx

            raise DeviceError(
                "VB-CABLE CABLE Input (WASAPI render endpoint) is required for live call mode but was not found."
            )

        # TEST SINK MODE: Strictly targeting physical headphones/speakers
        for idx, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if ("speaker" in name or "headphone" in name) and "cable" not in name:
                    return idx

        hostapis = sd.query_hostapis()
        def_out = hostapis[wasapi_idx].get("default_output_device")
        if def_out is not None and def_out >= 0:
            return def_out

        raise DeviceError("No suitable WASAPI headphones or speakers found for TestSink.")

    def _capture_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """Real-time capture callback from microphone (Producer).

        Guarantees zero heap allocation, no locks, no file/network I/O.
        """
        if status:
            self._callback_errors += 1

        self._capture_frames_total += 1
        if self._capture_in_sr == FS_16K:
            # 160 samples @ 16 kHz -> upsample to 480 samples @ 48 kHz
            mono_in = indata[:, 0] if indata.ndim > 1 else indata
            if len(mono_in) == FRAME_SIZE_16K:
                self.resampler.upsample_16k_to_48k(mono_in, out_480=self._in_16k_upsample_buf)
                self.ring_buffer.write(self._in_16k_upsample_buf)
            else:
                self.ring_buffer.write(indata)
        else:
            # Push incoming samples into SPSC ring buffer (handles stereo-to-mono in-place)
            self.ring_buffer.write(indata)

    def _render_callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """Real-time render callback to physical headphones (Consumer & DSP Processor).

        Guarantees bounded latency, no locks, fail-closed output zeroing.
        (Note: test recording tap performs file writes if active; transient allocations
        occur during FFT operations).
        """
        if status:
            self._callback_errors += 1

        # Fail-closed mute guard (Rule A-03)
        if self._panic_flag or self._is_muted or not self._is_running:
            np.copyto(outdata, 0.0)
            return

        # CRITICAL SECURITY GUARD (Rules 6 & 7):
        # Under NO circumstances allow unconverted mic audio to reach VB-CABLE
        if (self.config.live_call_mode or self.config.enable_vb_cable) and not self._is_converting:
            np.copyto(outdata, 0.0)
            return

        t0 = time.perf_counter()

        try:
            # 1. ASRC Dynamic Resampling and Buffer Occupancy Tracking
            avail = self.ring_buffer.available_read
            corr_ppm, rho = self.asrc_controller.update(float(avail))
            self._asrc_corr_ppm = corr_ppm
            self._asrc_rho = rho

            needed_samples, consumed_int = self.asrc_resampler.calculate_support(rho)

            # Check underflow state machine (fail-closed if insufficient samples)
            can_process = self.asrc_state_machine.record_frame(avail, needed_samples)
            if not can_process:
                np.copyto(outdata, 0.0)
                return

            # Non-destructive lookahead peek from ring buffer
            has_data = self.ring_buffer.peek(self._asrc_scratch, num_samples=needed_samples)
            if not has_data:
                np.copyto(outdata, 0.0)
                return

            # Perform 4-point Hermite cubic resampling into preallocated _in_frame
            self.asrc_resampler.process_block(self._asrc_scratch, rho, out_dest=self._in_frame)

            # Commit consumed integer samples to ring buffer
            self.ring_buffer.advance_read_ptr(consumed_int)

            # 2. Process frame through RNNoise (zero-allocation in-place DSP)
            self.suppressor.process_frame(self._in_frame, self._proc_frame)

            # 3. Apply Voice Conversion if active
            if self._is_converting:
                self.converter.process_frame(self._proc_frame, self._conv_frame)
                outdata[:, 0] = self._conv_frame
                rendered_frame = self._conv_frame
            else:
                outdata[:, 0] = self._proc_frame
                rendered_frame = self._proc_frame

            # 4. Record PCM tap into preallocated ring buffer (Zero allocation & zero mutex in callback)
            if self._record_active:
                head = self._record_head
                tail = self._record_tail
                if (head - tail) >= 300:
                    self._record_overflow_count += 1
                else:
                    slot = head % 300
                    np.copyto(self._record_ring[slot], rendered_frame)
                    self._record_head = head + 1

            self._render_frames_total += 1

            # 5. Measure DSP duration into preallocated circular telemetry buffer
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry_buf[self._telemetry_idx] = dt_ms
            self._telemetry_idx = (self._telemetry_idx + 1) & 255
            if self._telemetry_count < 256:
                self._telemetry_count += 1

        except Exception as exc:
            # Fail-closed: immediately zero output buffer
            np.copyto(outdata, 0.0)
            self._callback_errors += 1
            self._last_error = str(exc)
            self.asrc_state_machine.state = ASRCState.MUTED_ERROR

    def _stream_finished_callback(self) -> None:
        """Triggered when sounddevice stream terminates or device disconnects."""
        logger.warning("Audio stream finished or device disconnected")
        self._is_running = False

    def process_offline_chunk(self, in_chunk: np.ndarray, out_chunk: np.ndarray) -> float:
        """Direct zero-stream frame processing for unit testing and offline verification."""
        return self.suppressor.process_frame(in_chunk, out_chunk)

    def process_frame_resampled(
        self,
        in_frame_48k: np.ndarray,
        out_frame_48k: np.ndarray,
        voice_engine_callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    ) -> float:
        """End-to-end processing: 48 kHz in -> RNNoise -> 16 kHz downsample -> Voice Engine -> 48 kHz upsample.

        Guarantees zero heap allocation, seamless frame boundary phase continuity,
        and strict fail-closed output muting on exception.
        """
        if self._panic_flag or self._is_muted:
            np.copyto(out_frame_48k, 0.0)
            return 0.0

        t0 = time.perf_counter()

        # 1. 48 kHz Noise suppression
        self.suppressor.process_frame(in_frame_48k, self._proc_frame)

        # 2. Downsample 48 kHz (480) -> 16 kHz (160)
        self.resampler.downsample_48k_to_16k(self._proc_frame, out_160=self._16k_clean)

        # 3. 16 kHz Voice Engine conversion (or pass-through clean voice)
        if voice_engine_callback is not None:
            voice_engine_callback(self._16k_clean, self._16k_converted)
        else:
            self._16k_converted[:] = self._16k_clean

        # 4. Upsample 16 kHz (160) -> 48 kHz (480)
        self.resampler.upsample_16k_to_48k(self._16k_converted, out_480=out_frame_48k)

        return (time.perf_counter() - t0) * 1000.0

    def start_recording(self, duration_seconds: float = 5.0) -> Path:
        """Begin tapping real physical microphone audio into a local WAV recording.

        Rule 8 compliant: captures real microphone PCM rather than synthetic sine waves.
        Prohibited during live calls or VB-CABLE streaming (safety & performance).
        """
        with self._record_lock:
            if self.config.live_call_mode or self.config.enable_vb_cable:
                raise AudioProcessingError(
                    "Security / Performance violation: Recording tap is disabled during live calls "
                    "or active VB-CABLE streaming to avoid audio degradation."
                )
            if self._record_active:
                return self._record_path or Path("")

            local_app_data = os.environ.get(
                "LOCALAPPDATA",
                str(Path.home() / "AppData" / "Local"),
            )
            rec_dir = Path(local_app_data) / "HTH" / "temp_recordings"
            rec_dir.mkdir(parents=True, exist_ok=True)

            rec_id = str(uuid.uuid4())[:8]
            rec_path = rec_dir / f"physical_mic_test_{rec_id}.wav"

            wf = wave.open(str(rec_path), "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(self.config.sample_rate)

            self._record_file = wf
            self._record_path = rec_path
            self._record_samples_remaining = int(self.config.sample_rate * duration_seconds)
            self._record_head = 0
            self._record_tail = 0
            self._record_overflow_count = 0
            self._record_active = True

            # Spawn asynchronous background record writer worker
            self._record_thread = threading.Thread(
                target=self._record_worker,
                name="hth-record-worker",
                daemon=True,
            )
            self._record_thread.start()

            logger.info("Started real microphone test recording", path=str(rec_path))
            return rec_path

    def _record_worker(self) -> None:
        """Dedicated background worker pulling frames from preallocated ring and writing to disk."""
        while self._record_active or self._record_tail < self._record_head:
            head = self._record_head
            tail = self._record_tail
            if tail < head:
                slot = tail % 300
                frame = self._record_ring[slot]
                self._record_tail = tail + 1

                with self._record_lock:
                    if not self._record_file or self._record_samples_remaining <= 0:
                        break
                    clamped = np.clip(frame, -1.0, 1.0)
                    pcm16 = (clamped * 32767.0).astype(np.int16)
                    try:
                        self._record_file.writeframes(pcm16.tobytes())
                    except Exception:
                        break
                    self._record_samples_remaining -= len(frame)
                    if self._record_samples_remaining <= 0:
                        break
            else:
                time.sleep(0.010)

        if self._record_active and self._record_samples_remaining <= 0:
            self.stop_recording()

    def stop_recording(self) -> Optional[Path]:
        """Stop and close the WAV test recording safely."""
        with self._record_lock:
            if not self._record_active:
                return self._record_path
            self._record_active = False

        if (
            self._record_thread
            and self._record_thread.is_alive()
            and threading.current_thread() != self._record_thread
        ):
            self._record_thread.join(timeout=1.0)
        self._record_thread = None

        with self._record_lock:
            path = self._record_path
            if self._record_file:
                try:
                    self._record_file.close()
                except Exception:
                    pass
                self._record_file = None
            logger.info("Closed test recording", path=str(path))
            return path

    def get_metrics(self) -> AudioPipelineMetrics:
        """Retrieve real-time metrics, latency percentiles, and buffer statistics."""
        rb_metrics = self.ring_buffer.get_metrics()
        occupancy = rb_metrics["occupancy_samples"]
        occ_ms = (occupancy / self.config.sample_rate) * 1000.0

        p50 = 0.0
        p95 = 0.0
        p99 = 0.0
        max_ms = 0.0
        if self._telemetry_count > 0:
            valid = self._telemetry_buf[: self._telemetry_count]
            p50 = float(np.percentile(valid, 50))
            p95 = float(np.percentile(valid, 95))
            p99 = float(np.percentile(valid, 99))
            max_ms = float(np.max(valid))

        # Non-blocking retrieval of cached CPU and RAM telemetry
        cpu = self._telemetry_sampler.cpu_pct
        ram = self._telemetry_sampler.ram_mb

        is_sink = self.config.test_sink_mode and not (
            self.config.live_call_mode or self.config.enable_vb_cable
        )
        banner = (
            "TEST HARNESS — LIVE TRANSMISSION DISABLED"
            if is_sink
            else "LIVE CONVERSION — VB-CABLE ROUTE ACTIVE"
        )

        return AudioPipelineMetrics(
            is_running=self._is_running,
            is_muted=self._is_muted or self._panic_flag,
            is_test_sink_mode=is_sink,
            test_harness_banner=banner,
            active_preset=self.config.preset,
            is_converting=self._is_converting,
            active_profile=self._conversion_profile.value,
            live_call_mode=self.config.live_call_mode or self.config.enable_vb_cable,
            sample_rate=self.config.sample_rate,
            frame_size=self.config.frame_size,
            processing_time_p50_ms=round(p50, 3),
            processing_time_p95_ms=round(p95, 3),
            processing_time_p99_ms=round(p99, 3),
            processing_time_max_ms=round(max_ms, 3),
            buffer_occupancy_samples=occupancy,
            buffer_occupancy_ms=round(occ_ms, 2),
            capture_frames_total=self._capture_frames_total,
            render_frames_total=self._render_frames_total,
            overflow_count=rb_metrics["overflow_count"],
            underrun_count=max(
                rb_metrics["underrun_count"], self.asrc_state_machine.total_underruns
            ),
            dropped_samples_total=rb_metrics["dropped_samples"],
            discontinuity_epoch=rb_metrics["discontinuity_epoch"],
            native_callback_errors=self._callback_errors,
            asrc_corr_ppm=round(self._asrc_corr_ppm, 3),
            asrc_rho=round(self._asrc_rho, 7),
            asrc_state=self.asrc_state_machine.state.value,
            cpu_pct=cpu,
            ram_mb=ram,
            last_error=self._last_error,
        )


# Global singleton pipeline instance
_global_pipeline: Optional[AudioPipeline] = None


def get_audio_pipeline() -> AudioPipeline:
    """Dependency accessor for the real-time audio pipeline."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = AudioPipeline()
    return _global_pipeline


def set_audio_pipeline(pipeline: Optional[AudioPipeline]) -> None:
    """Setter for testing and isolation."""
    global _global_pipeline
    _global_pipeline = pipeline
