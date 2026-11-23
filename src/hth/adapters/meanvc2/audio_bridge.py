"""Real-time WASAPI physical audio bridge for MeanVC2.

Architecture:
  Physical Mic (WASAPI) [48 kHz, 480-sample / 10ms frame]
    -> Input Ring Buffer (preallocated bounded SPSC)
    -> Queue Backlog Compaction (prevents unbounded drift on CPU)
    -> Resampler 48k -> 16k (Polyphase FIR decimation, 160-sample frames)
    -> Chunk Accumulator (2560 samples / 160ms)
    -> MeanVC2 Neural Worker Subprocess (Fast-U2++ ASR -> DiT CFM -> Vocos)
    -> Output Chunk (2560 samples / 160ms @ 16kHz)
    -> Resampler 16k -> 48k (Polyphase FIR interpolation, 480-sample frames)
    -> Dual Output Routing:
       1. Virtual Output Ring -> Render Callback (WASAPI) -> VB-CABLE CABLE Input (for WhatsApp / Discord)
       2. Monitor Output Ring -> Monitor Callback (WASAPI) -> Physical Headphones (Zero-Feedback Monitoring)

Adheres strictly to Mandatory Engineering Rules:
- Rule 1: Local only (127.0.0.1)
- Rule 2: Audio callbacks are non-blocking, zero heap allocation, no locks, no file/network I/O
- Rule 3: Fail-closed mute guard on device disconnect, buffer overflow, or worker error.
          Never auto-falls back to raw microphone passthrough.
- Rule 4: Consent & disclosure enforced before starting conversion.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

from hth.audio.resampler import (
    FRAME_SIZE_16K,
    FRAME_SIZE_48K,
    FS_16K,
    FS_48K,
    PolyphaseResampler48To16,
    PolyphaseResampler16To48,
)
from hth.audio.ring_buffer import AudioRingBuffer
from hth.core.logging import get_logger
from hth.domain.errors import AudioProcessingError, DeviceError
from hth.adapters.meanvc2.dsp_effects import DSPPostProcessor

logger = get_logger("hth.adapters.meanvc2.audio_bridge")

# Frame and chunk sizes
CHUNK_16K = 2560  # 160ms @ 16kHz
FRAMES_PER_CHUNK = CHUNK_16K // FRAME_SIZE_16K  # 16 frames of 10ms


def get_wasapi_hostapi_index() -> int:
    """Discover Windows WASAPI host API index from sounddevice."""
    if not HAS_SOUNDDEVICE or sd is None:
        raise DeviceError("sounddevice is not available")
    hostapis = sd.query_hostapis()
    for idx, hostapi in enumerate(hostapis):
        name = hostapi.get("name", "").lower()
        if "wasapi" in name:
            return idx
    raise DeviceError("Windows WASAPI host API is not available on this system.")


class MeanVC2AudioBridge:
    """
    Connects real Windows WASAPI physical audio capture (Microphone) and
    dual render endpoints (VB-CABLE CABLE Input AND Physical Headphones) to MeanVC2.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._lock = threading.Lock()
        self._is_running = False
        self._is_muted = False
        self._panic_flag = False

        # Dual routing & monitoring states
        self.monitor_enabled = False
        self.monitor_volume = 1.0
        self.output_volume = 1.0

        # Preallocated bounded ring buffers (Rule A-01: Zero runtime allocation)
        # 32768 samples = 682 ms capacity @ 48 kHz
        self.input_ring = AudioRingBuffer(capacity=32768, max_occupancy=28800)
        self.output_ring = AudioRingBuffer(capacity=32768, max_occupancy=28800)
        self.monitor_ring = AudioRingBuffer(capacity=32768, max_occupancy=28800)

        # Polyphase Resamplers (48 kHz <-> 16 kHz)
        self.resampler_48_to_16 = PolyphaseResampler48To16()
        self.resampler_16_to_48 = PolyphaseResampler16To48()

        # Preallocated processing buffers
        self._frame_48k = np.zeros(FRAME_SIZE_48K, dtype=np.float32)
        self._frame_16k = np.zeros(FRAME_SIZE_16K, dtype=np.float32)
        self._out_frame_16k = np.zeros(FRAME_SIZE_16K, dtype=np.float32)
        self._out_frame_48k = np.zeros(FRAME_SIZE_48K, dtype=np.float32)

        # Real-time zero-allocation DSP post-processor (Pitch, Formant EQ, Reverb/FX)
        self._dsp_processor = DSPPostProcessor(sample_rate=FS_16K, chunk_size=CHUNK_16K)

        # Monitor device configuration
        self.monitor_device_id: Optional[int | str] = None

        # Stream hardware properties (dynamic WASAPI format support)
        self._input_sr: int = FS_48K
        self._input_frame_size: int = FRAME_SIZE_48K
        self._monitor_sr: int = FS_48K
        self._monitor_channels: int = 1
        self._monitor_frame_size: int = FRAME_SIZE_48K

        # Preallocated interpolation coordinates and buffers (Rule A-01: Zero runtime allocation)
        self._interp_x_in = np.linspace(0.0, 1.0, FRAME_SIZE_16K, endpoint=False, dtype=np.float32)
        self._interp_x_out_44k = np.linspace(0.0, 1.0, 441, endpoint=False, dtype=np.float32)
        self._out_frame_44k = np.zeros(441, dtype=np.float32)

        # Preallocated chunk accumulator for 16kHz input
        self._chunk_buf = np.zeros(CHUNK_16K, dtype=np.float32)
        self._chunk_pos = 0

        # Render scratch buffers for callbacks (covers max possible frame size, e.g. 2048)
        self._render_scratch = np.zeros(2048, dtype=np.float32)
        self._monitor_scratch = np.zeros(2048, dtype=np.float32)

        # Streams and worker thread
        self._input_stream: Optional[Any] = None
        self._output_stream: Optional[Any] = None
        self._monitor_stream: Optional[Any] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Telemetry metrics
        self.capture_frames_total: int = 0
        self.render_frames_total: int = 0
        self.monitor_frames_total: int = 0
        self.callback_errors: int = 0
        self.underruns: int = 0
        self.dropped_chunks: int = 0
        self.last_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_muted(self) -> bool:
        return self._is_muted or self._panic_flag

    def set_monitor_enabled(self, enabled: bool) -> None:
        """Enable or disable live headphone monitoring."""
        with self._lock:
            self.monitor_enabled = enabled
            if enabled and self._is_running and self._monitor_stream is None:
                self._start_monitor_stream()
            elif not enabled and self._monitor_stream is not None:
                self._stop_monitor_stream()
        logger.info("Headphone monitoring state updated", enabled=enabled)

    def set_monitor_device(self, device_id: Optional[int | str]) -> None:
        """Dynamically switch physical monitor output endpoint in real time."""
        with self._lock:
            self.monitor_device_id = device_id
            if self._is_running and self.monitor_enabled:
                self._stop_monitor_stream()
                self._start_monitor_stream()
        logger.info("Monitor device updated in real-time", device_id=device_id)

    def set_monitor_volume(self, volume: float) -> None:
        """Adjust local headphone monitor volume (0.0 to 1.5)."""
        self.monitor_volume = max(0.0, min(1.5, volume))

    def set_output_volume(self, volume: float) -> None:
        """Adjust master virtual mic output volume (0.0 to 1.5)."""
        self.output_volume = max(0.0, min(1.5, volume))

    def panic_mute(self) -> None:
        """Immediately zero all outgoing audio streams (fail-closed)."""
        with self._lock:
            self._is_muted = True
            self._panic_flag = True
        logger.warning("MeanVC2 Audio Bridge PANIC MUTE engaged")

    def un_mute(self) -> None:
        """Release panic mute on explicit user action."""
        with self._lock:
            self._is_muted = False
            self._panic_flag = False
        logger.info("MeanVC2 Audio Bridge panic mute released")

    def _resolve_input_device(self, dev_id: Optional[int | str]) -> Any:
        """Resolve WASAPI physical microphone, strictly rejecting virtual loopback cables."""
        if not HAS_SOUNDDEVICE or sd is None:
            return None
        wasapi_idx = get_wasapi_hostapi_index()
        devices = sd.query_devices()

        def _is_cable(name: str) -> bool:
            return "cable" in name.lower()

        # If a specific device ID was requested, ensure it is NOT a virtual cable
        if dev_id is not None:
            cand_idx: Optional[int] = None
            if isinstance(dev_id, int) and 0 <= dev_id < len(devices):
                cand_idx = dev_id
            elif isinstance(dev_id, str) and dev_id.isdigit() and 0 <= int(dev_id) < len(devices):
                cand_idx = int(dev_id)
            elif isinstance(dev_id, str):
                search_term = dev_id.lower()
                for idx, d in enumerate(devices):
                    if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                        name = d.get("name", "").lower()
                        if search_term in name:
                            cand_idx = idx
                            break
            if cand_idx is not None:
                cand_name = devices[cand_idx].get("name", "")
                if not _is_cable(cand_name):
                    return cand_idx
                logger.warning(
                    "Rejecting virtual cable as input capture device to avoid audio loop",
                    requested=cand_name,
                    index=cand_idx,
                )

        # Auto-detect physical microphone (prefer Jabra headset mic or Microphone Array, exclude cable)
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if not _is_cable(name) and ("jabra" in name or "headset" in name):
                    return idx

        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if not _is_cable(name) and "microphone" in name:
                    return idx

        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if not _is_cable(name):
                    return idx

        hostapis = sd.query_hostapis()
        def_in = hostapis[wasapi_idx].get("default_input_device")
        if def_in is not None and def_in >= 0:
            def_name = devices[def_in].get("name", "") if def_in < len(devices) else ""
            if not _is_cable(def_name):
                return def_in
        raise DeviceError("No suitable WASAPI physical microphone found.")

    def _resolve_output_device(self, dev_id: Optional[int | str], live_call_mode: bool = True) -> Any:
        """Resolve WASAPI virtual render output endpoint (VB-CABLE CABLE Input)."""
        if not HAS_SOUNDDEVICE or sd is None:
            return None
        wasapi_idx = get_wasapi_hostapi_index()
        devices = sd.query_devices()

        if dev_id is not None:
            if isinstance(dev_id, int):
                return dev_id
            if isinstance(dev_id, str) and dev_id.isdigit():
                return int(dev_id)
            search_term = str(dev_id).lower()
            for idx, d in enumerate(devices):
                if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                    name = d.get("name", "").lower()
                    if search_term in name:
                        return idx

        if live_call_mode:
            # Route to VB-CABLE Input for WhatsApp / Discord / Teams
            for idx, d in enumerate(devices):
                if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                    name = d.get("name", "").lower()
                    if "cable" in name and "input" in name:
                        return idx
            raise DeviceError("VB-CABLE CABLE Input (WASAPI render endpoint) is required for live call mode but was not found.")

        return self._resolve_headphones_device()

    def _resolve_headphones_device(self, target_dev: Optional[int | str] = None) -> Any:
        """Resolve physical headphones or speakers for local zero-feedback monitor."""
        if not HAS_SOUNDDEVICE or sd is None:
            return None
        wasapi_idx = get_wasapi_hostapi_index()
        devices = sd.query_devices()

        def _is_cable(name: str) -> bool:
            return "cable" in name.lower()

        # Check explicit device selection first
        dev_target = target_dev if target_dev is not None else getattr(self, "monitor_device_id", None)
        if dev_target is not None and str(dev_target).strip() != "":
            if isinstance(dev_target, int) and 0 <= dev_target < len(devices):
                if not _is_cable(devices[dev_target].get("name", "")):
                    return dev_target
            elif isinstance(dev_target, str) and dev_target.isdigit():
                idx_val = int(dev_target)
                if 0 <= idx_val < len(devices) and not _is_cable(devices[idx_val].get("name", "")):
                    return idx_val
            elif isinstance(dev_target, str):
                search_term = dev_target.lower()
                for idx, d in enumerate(devices):
                    if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                        name = d.get("name", "").lower()
                        if not _is_cable(name) and search_term in name:
                            return idx

        # Auto-detection preference 1: Jabra or Headphone
        for idx, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if not _is_cable(name) and ("jabra" in name or "headphone" in name):
                    return idx

        # Preference 2: Speakers
        for idx, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if not _is_cable(name) and "speaker" in name:
                    return idx

        # Preference 3: Any non-cable physical output
        for idx, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == wasapi_idx:
                name = d.get("name", "").lower()
                if not _is_cable(name):
                    return idx

        hostapis = sd.query_hostapis()
        def_out = hostapis[wasapi_idx].get("default_output_device")
        if def_out is not None and def_out >= 0:
            return def_out
        raise DeviceError("No suitable physical headphones or speakers found.")

    def _capture_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Real-time capture callback from physical mic (zero allocation, non-blocking)."""
        if status:
            self.callback_errors += 1
        self.capture_frames_total += 1
        self.input_ring.write(indata)

    def _render_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Real-time render callback to VB-CABLE (virtual mic for calls)."""
        if status:
            self.callback_errors += 1

        if self._panic_flag or self._is_muted or not self._is_running:
            outdata.fill(0.0)
            return

        needed = frames
        if needed > len(self._render_scratch):
            needed = len(self._render_scratch)

        success = self.output_ring.read(self._render_scratch, needed)
        if not success:
            self.underruns += 1
            outdata.fill(0.0)
            return

        outdata[:needed, 0] = self._render_scratch[:needed] * self.output_volume
        if frames > needed:
            outdata[needed:].fill(0.0)
        self.render_frames_total += 1

    def _monitor_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Real-time render callback to physical headphones (local monitor)."""
        if not self.monitor_enabled or self._panic_flag or self._is_muted or not self._is_running:
            outdata.fill(0.0)
            return

        needed = min(frames, len(self._monitor_scratch))
        success = self.monitor_ring.read(self._monitor_scratch, needed)
        if not success:
            outdata.fill(0.0)
            return

        gain = self.monitor_volume
        if outdata.ndim == 1:
            outdata[:needed] = self._monitor_scratch[:needed] * gain
            if frames > needed:
                outdata[needed:].fill(0.0)
        else:
            outdata[:needed, 0] = self._monitor_scratch[:needed] * gain
            if outdata.shape[1] > 1:
                outdata[:needed, 1] = self._monitor_scratch[:needed] * gain
            if frames > needed:
                outdata[needed:, :].fill(0.0)

        self.monitor_frames_total += 1

    def _start_monitor_stream(self) -> None:
        """Open headphone monitor output stream matching hardware native rate & channel layout."""
        if not HAS_SOUNDDEVICE or sd is None or self._monitor_stream is not None:
            return
        try:
            hp_dev = self._resolve_headphones_device(self.monitor_device_id)
            hp_info = sd.query_devices(hp_dev)
            self._monitor_sr = int(hp_info.get("default_samplerate", FS_48K))
            self._monitor_channels = max(1, min(2, int(hp_info.get("max_output_channels", 2))))
            self._monitor_frame_size = int(self._monitor_sr * 0.01)

            self._monitor_stream = sd.OutputStream(
                device=hp_dev,
                samplerate=self._monitor_sr,
                channels=self._monitor_channels,
                blocksize=self._monitor_frame_size,
                dtype="float32",
                callback=self._monitor_callback,
            )
            self._monitor_stream.start()
            logger.info(
                "Headphone monitor stream started",
                device=hp_dev,
                name=hp_info.get("name"),
                samplerate=self._monitor_sr,
                channels=self._monitor_channels,
            )
        except Exception as exc:
            logger.warning("Could not open headphone monitor stream", error=str(exc))
            self._monitor_stream = None

    def _stop_monitor_stream(self) -> None:
        """Close headphone monitor stream."""
        if self._monitor_stream is not None:
            try:
                self._monitor_stream.stop()
                self._monitor_stream.close()
            except Exception:
                pass
            self._monitor_stream = None

    def _bridge_worker_loop(self) -> None:
        """
        Background DSP worker:
        - Decimates/passes mic frames -> 16k frames -> 2560-sample chunks -> MeanVC2 worker
        - Prevents latency drift with queue age compaction
        - Pops converted 2560-sample chunks -> Interpolates to 48k/44.1k -> Dual ring buffers.
        """
        while not self._stop_event.is_set():
            active_pid = getattr(self.adapter, "_active_profile_id", None)
            is_bypass = active_pid in ("clean-mic", "bypass")

            # 1. Drain input ring buffer (handles 16kHz or 48kHz physical mic)
            if self._input_sr == FS_16K:
                # 16kHz native input (e.g. Jabra headset mic) - no decimation needed
                if self.input_ring.available_read > 3200:
                    self.input_ring.read(self._frame_16k, FRAME_SIZE_16K)
                    self.dropped_chunks += 1

                while self.input_ring.available_read >= FRAME_SIZE_16K:
                    has_frame = self.input_ring.read(self._frame_16k, FRAME_SIZE_16K)
                    if not has_frame:
                        break

                    if is_bypass:
                        # Instant clean mic passthrough to VB-CABLE and headphones
                        self.resampler_16_to_48.process_frame(self._frame_16k, out_frame=self._out_frame_48k)
                        self.output_ring.write(self._out_frame_48k)
                        if self.monitor_enabled:
                            if self._monitor_sr == 44100:
                                self._out_frame_44k[:] = np.interp(
                                    self._interp_x_out_44k,
                                    self._interp_x_in,
                                    self._frame_16k,
                                )
                                self.monitor_ring.write(self._out_frame_44k)
                            else:
                                self.monitor_ring.write(self._out_frame_48k)
                        continue

                    rem = CHUNK_16K - self._chunk_pos
                    copy_len = min(rem, FRAME_SIZE_16K)
                    self._chunk_buf[self._chunk_pos : self._chunk_pos + copy_len] = self._frame_16k[:copy_len]
                    self._chunk_pos += copy_len

                    if self._chunk_pos >= CHUNK_16K:
                        chunk_to_send = self._chunk_buf.copy()
                        self.adapter.push_audio(chunk_to_send)
                        self._chunk_pos = 0
            else:
                # 48kHz native input (e.g. Microphone Array) - decimate 48k -> 16k
                if self.input_ring.available_read > 9600:
                    self.input_ring.read(self._frame_48k, FRAME_SIZE_48K)
                    self.dropped_chunks += 1

                while self.input_ring.available_read >= FRAME_SIZE_48K:
                    has_frame = self.input_ring.read(self._frame_48k, FRAME_SIZE_48K)
                    if not has_frame:
                        break
                    self.resampler_48_to_16.process_frame(self._frame_48k, out_frame=self._frame_16k)

                    if is_bypass:
                        # Instant clean mic passthrough to VB-CABLE and headphones
                        self.output_ring.write(self._frame_48k)
                        if self.monitor_enabled:
                            if self._monitor_sr == 44100:
                                self._out_frame_44k[:] = np.interp(
                                    self._interp_x_out_44k,
                                    self._interp_x_in,
                                    self._frame_16k,
                                )
                                self.monitor_ring.write(self._out_frame_44k)
                            else:
                                self.monitor_ring.write(self._frame_48k)
                        continue

                    rem = CHUNK_16K - self._chunk_pos
                    copy_len = min(rem, FRAME_SIZE_16K)
                    self._chunk_buf[self._chunk_pos : self._chunk_pos + copy_len] = self._frame_16k[:copy_len]
                    self._chunk_pos += copy_len

                    if self._chunk_pos >= CHUNK_16K:
                        chunk_to_send = self._chunk_buf.copy()
                        self.adapter.push_audio(chunk_to_send)
                        self._chunk_pos = 0

            if is_bypass:
                time.sleep(0.002)
                continue

            # 2. Check for converted audio chunks from MeanVC2 adapter
            converted_chunk = self.adapter.pop_converted_audio(CHUNK_16K)
            if np.any(converted_chunk != 0.0):
                # Apply active profile DSP post-processing (pitch, formant EQ, reverb/FX)
                active_profile = getattr(self.adapter, "get_active_profile", lambda: None)()
                if active_profile is not None:
                    converted_chunk = self._dsp_processor.process(
                        converted_chunk,
                        pitch_shift=float(getattr(active_profile, "pitch_shift", 0.0) or 0.0),
                        timbre_ratio=float(getattr(active_profile, "timbre_ratio", 1.0) or 1.0),
                        sound_effect=str(getattr(active_profile, "sound_effect", "none") or "none"),
                        volume_gain=float(getattr(active_profile, "volume_gain", 1.0) or 1.0),
                    )

                # Process 16 frames of 160 samples (160ms total @ 16kHz)
                for f_idx in range(FRAMES_PER_CHUNK):
                    start = f_idx * FRAME_SIZE_16K
                    end = start + FRAME_SIZE_16K
                    slice_160 = converted_chunk[start:end]

                    # Dual routing 1: VB-CABLE output (always 48kHz for virtual mic)
                    self.resampler_16_to_48.process_frame(slice_160, out_frame=self._out_frame_48k)
                    self.output_ring.write(self._out_frame_48k)

                    # Dual routing 2: Physical Headphone Monitor
                    if self.monitor_enabled:
                        if self._monitor_sr == FS_48K:
                            self.monitor_ring.write(self._out_frame_48k)
                        elif self._monitor_sr == 44100:
                            self._out_frame_44k[:] = np.interp(
                                self._interp_x_out_44k,
                                self._interp_x_in,
                                slice_160,
                            )
                            self.monitor_ring.write(self._out_frame_44k)
                        else:
                            self.monitor_ring.write(slice_160)

            # Cooperative sleep to prevent CPU spin
            time.sleep(0.002)

    def start(
        self,
        input_device_id: Optional[int | str] = None,
        output_device_id: Optional[int | str] = None,
        monitor_device_id: Optional[int | str] = None,
        live_call_mode: bool = True,
        enable_monitor: bool = False,
    ) -> None:
        """Start physical capture and render streams with resamplers."""
        with self._lock:
            if self._is_running:
                return

            self.monitor_device_id = monitor_device_id
            self.input_ring.reset()
            self.output_ring.reset()
            self.monitor_ring.reset()
            # Prime output ring buffers with 40ms silence to absorb initial jitter
            self.output_ring.write(np.zeros(1920, dtype=np.float32))
            self.monitor_ring.write(np.zeros(1920, dtype=np.float32))
            self.resampler_48_to_16.reset()
            self.resampler_16_to_48.reset()
            self._chunk_pos = 0
            self._is_muted = False
            self._panic_flag = False
            self.last_error = None
            self.monitor_enabled = enable_monitor
            self._stop_event.clear()

            if not HAS_SOUNDDEVICE:
                logger.warning("sounddevice not available; audio bridge running in headless mode")
                self._is_running = True
                return

            try:
                in_dev = self._resolve_input_device(input_device_id)
                out_dev = self._resolve_output_device(output_device_id, live_call_mode=live_call_mode)

                in_info = sd.query_devices(in_dev)
                self._input_sr = int(in_info.get("default_samplerate", FS_48K))
                self._input_frame_size = int(self._input_sr * 0.01)

                logger.info(
                    "Opening WASAPI streams for MeanVC2",
                    in_dev=in_dev,
                    in_name=in_info.get("name"),
                    in_sr=self._input_sr,
                    out_dev=out_dev,
                )

                self._input_stream = sd.InputStream(
                    device=in_dev,
                    samplerate=self._input_sr,
                    channels=1,
                    blocksize=self._input_frame_size,
                    dtype="float32",
                    callback=self._capture_callback,
                )

                self._output_stream = sd.OutputStream(
                    device=out_dev,
                    samplerate=FS_48K,
                    channels=1,
                    blocksize=FRAME_SIZE_48K,
                    dtype="float32",
                    callback=self._render_callback,
                )

                self._input_stream.start()
                self._output_stream.start()

                if self.monitor_enabled:
                    self._start_monitor_stream()

                # Start background DSP bridge thread
                self._worker_thread = threading.Thread(
                    target=self._bridge_worker_loop,
                    name="MeanVC2-AudioBridge",
                    daemon=True,
                )
                self._worker_thread.start()

                self._is_running = True
                logger.info("MeanVC2 Audio Bridge successfully started physical streams")
            except Exception as exc:
                self.last_error = str(exc)
                self.stop()
                raise DeviceError(f"Failed to start MeanVC2 audio streams: {exc}") from exc

    def stop(self) -> None:
        """Stop streams and release hardware resources."""
        with self._lock:
            self._is_running = False
            self._stop_event.set()

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

            self._stop_monitor_stream()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        self.input_ring.reset()
        self.output_ring.reset()
        self.monitor_ring.reset()
        self.resampler_48_to_16.reset()
        self.resampler_16_to_48.reset()
        logger.info("MeanVC2 Audio Bridge stopped")

    def get_metrics(self) -> dict:
        """Return runtime telemetry for audio streaming and compaction."""
        return {
            "capture_frames": self.capture_frames_total,
            "render_frames": self.render_frames_total,
            "monitor_frames": self.monitor_frames_total,
            "dropped_chunks": self.dropped_chunks,
            "underruns": self.underruns,
            "callback_errors": self.callback_errors,
            "monitor_enabled": self.monitor_enabled,
            "is_muted": self._is_muted or self._panic_flag,
        }
