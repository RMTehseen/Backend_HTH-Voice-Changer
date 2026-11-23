"""
MeanVC2VoiceAdapter — HTH VoiceEngineProtocol adapter for MeanVC2 neural streaming VC.

Architecture:
  Mic → WASAPI capture callback → input ring buffer
      → MeanVC2 inference worker (subprocess, Python 3.11)
      → output ring buffer → WASAPI render callback → VB-CABLE

The inference worker runs in a separate Python 3.11 subprocess with PyTorch/s3prl.
IPC: subprocess stdin/stdout JSON + asyncio subprocess streams.
Audio: PCM float32 16kHz base64 chunks (kept small; real WASAPI I/O in HTH audio pipeline).
"""
from __future__ import annotations

import asyncio
import base64
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Deque, Optional

import numpy as np

from hth.core.logging import get_logger
from hth.domain.api_contracts import (
    DeviceInfo,
    DeviceType,
    EngineInfo,
    EngineState,
    LiveConfig,
    LiveStatus,
)
from .config import MeanVC2Config
from .voice_profile import VoiceProfile, VoiceProfileStore, validate_reference_audio, compute_file_sha256

logger = get_logger("hth.adapters.meanvc2")


class MeanVC2VoiceAdapter:
    """
    Production VoiceEngineProtocol adapter for MeanVC2 neural streaming voice conversion.
    Fail-closed: errors → muted, never raw passthrough.
    """

    ENGINE_ID = "meanvc2-neural-streaming"
    ENGINE_NAME = "MeanVC2 Neural Streaming Voice Conversion (ASLP-lab)"
    ENGINE_VERSION = "2.0.0"

    def __init__(self, config: Optional[MeanVC2Config] = None) -> None:
        self._cfg = config or MeanVC2Config()
        self._lock = asyncio.Lock()
        self._state: EngineState = EngineState.IDLE
        self._last_error: Optional[str] = None
        self._active_config: Optional[LiveConfig] = None
        self._active_profile_id: Optional[str] = None

        # Worker subprocess
        self._worker_proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}
        self._worker_ready = asyncio.Event()
        self._worker_loaded = asyncio.Event()
        self._worker_target_set = asyncio.Event()

        # Audio I/O queues (bounded)
        self._input_queue: Deque[np.ndarray] = collections.deque(maxlen=self._cfg.max_input_queue)
        self._output_queue: Deque[np.ndarray] = collections.deque(maxlen=self._cfg.max_output_queue)
        self._processing_task: Optional[asyncio.Task] = None

        # Metrics
        self._frames_processed: int = 0
        self._underruns: int = 0
        self._overruns: int = 0
        self._last_inference_ms: float = 0.0
        self._last_rtf: float = 0.0
        self._is_muted: bool = False
        self._passthrough_enabled: bool = False

        # Profile store
        local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        profile_dir = str(Path(local_app_data) / "HTH" / "MeanVC2Profiles")
        self._profile_store = VoiceProfileStore(profile_dir)

        # WASAPI physical audio bridge
        from .audio_bridge import MeanVC2AudioBridge
        self._audio_bridge = MeanVC2AudioBridge(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Start the MeanVC2 worker subprocess."""
        async with self._lock:
            if self._state not in (EngineState.IDLE,):
                raise RuntimeError(f"Cannot initialize in state {self._state}")
            await self._start_worker()
            self._state = EngineState.IDLE
            logger.info("MeanVC2 adapter initialized")

    async def _start_worker(self) -> None:
        """Launch MeanVC2 inference worker subprocess."""
        worker_script = Path(__file__).parent / "worker_process.py"

        # Find Python 3.11 venv for MeanVC2
        meanvc2_python = self._find_meanvc2_python()
        cmd = [meanvc2_python, str(worker_script)]

        logger.info("Starting MeanVC2 worker", python=meanvc2_python, script=str(worker_script))

        self._worker_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._worker_ready.clear()
        self._worker_loaded.clear()
        self._worker_target_set.clear()

        # Start reader
        self._reader_task = asyncio.create_task(self._read_worker_output())

        # Wait for ready signal
        try:
            await asyncio.wait_for(self._worker_ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            await self._kill_worker()
            raise RuntimeError("MeanVC2 worker did not send ready signal within 30s")

    def _find_meanvc2_python(self) -> str:
        """Find Python interpreter for MeanVC2 worker. Checks for dedicated venv."""
        project_root = Path(self._cfg.source_dir).parent if self._cfg.source_dir else Path.cwd()
        custom = os.environ.get("MEANVC2_PYTHON", "").strip()
        candidates = [
            project_root / "meanvc2-venv" / "Scripts" / "python.exe",
            project_root / "meanvc2-venv" / "bin" / "python",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe",
            Path("C:/Program Files/Python311/python.exe"),
            Path(sys.executable),
        ]
        if custom:
            candidates.insert(0, Path(custom))

        for c in candidates:
            if c.is_file():
                return str(c)
        # Fall back to current Python (may not have PyTorch/s3prl)
        logger.warning("MeanVC2 dedicated venv not found, using current Python", python=sys.executable)
        return sys.executable

    async def _read_worker_output(self) -> None:
        """Background task: read JSON events from worker stdout."""
        proc = self._worker_proc
        if not proc or not proc.stdout:
            return
        try:
            async for line in proc.stdout:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Worker sent non-JSON", line=line[:100])
                    continue
                await self._handle_worker_event(msg)
        except Exception as e:
            logger.error("Worker reader task error", error=str(e))
            self._state = EngineState.ERROR
            self._last_error = str(e)

    async def _handle_worker_event(self, msg: dict) -> None:
        event = msg.get("event", "")
        if event == "ready":
            self._worker_ready.set()
        elif event == "loaded":
            self._worker_loaded.set()
            logger.info("MeanVC2 model loaded", variant=msg.get("variant"), device=msg.get("device"))
        elif event == "target_set":
            self._worker_target_set.set()
            logger.info("Target speaker set", profile_id=msg.get("profile_id"))
        elif event == "warmup_done":
            logger.info("MeanVC2 warmup complete")
        elif event == "audio_out":
            if not self._is_muted and not self._passthrough_enabled:
                raw = base64.b64decode(msg["data_b64"])
                samples = np.frombuffer(raw, dtype=np.float32).copy()
                self._output_queue.append(samples)
                self._frames_processed += 1
            self._last_inference_ms = msg.get("inference_ms", 0.0)
            self._last_rtf = msg.get("rtf", 0.0)
        elif event == "audio_buffering":
            pass  # Normal — model is buffering
        elif event == "error":
            logger.error("Worker error", message=msg.get("message", ""), traceback=msg.get("traceback", ""))
            self._last_error = msg.get("message", "unknown worker error")
            self._state = EngineState.ERROR
            self._is_muted = True
            self._worker_ready.set()
            self._worker_loaded.set()
            self._worker_target_set.set()
        elif event == "pong":
            pass
        elif event == "shutdown_ok":
            logger.info("Worker shutdown acknowledged")
        elif event == "progress":
            logger.info("Worker progress", step=msg.get("step"))
        elif event == "cache_reset":
            logger.debug("Worker cache reset")

    async def _send_to_worker(self, msg: dict) -> None:
        """Send JSON command to worker stdin."""
        if not self._worker_proc or not self._worker_proc.stdin:
            raise RuntimeError("Worker not running")
        line = (json.dumps(msg) + "\n").encode("utf-8")
        self._worker_proc.stdin.write(line)
        await self._worker_proc.stdin.drain()

    async def _kill_worker(self) -> None:
        """Kill the worker subprocess."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._worker_proc:
            try:
                self._worker_proc.kill()
                await self._worker_proc.wait()
            except Exception:
                pass
            self._worker_proc = None

    # ------------------------------------------------------------------
    # VoiceEngineProtocol implementation
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        try:
            await self._send_to_worker({"cmd": "ping"})
            return True
        except Exception:
            return False

    async def list_devices(self) -> list[DeviceInfo]:
        try:
            from hth.audio.devices import get_device_inventory
            inventory = get_device_inventory()
            devices = []
            for ep in inventory.capture_devices + inventory.render_devices:
                dev_type = DeviceType.INPUT if ep.data_flow == "capture" else DeviceType.OUTPUT
                devices.append(DeviceInfo(
                    id=ep.id,
                    friendly_name=ep.friendly_name,
                    device_type=dev_type,
                    sample_rates=[48000, 16000],
                    default_sample_rate=48000,
                    is_default=ep.is_default_console,
                ))
            return devices
        except Exception as e:
            logger.warning("Device enumeration failed", error=str(e))
            return []

    async def start(self, config: LiveConfig) -> None:
        """Start neural voice conversion streaming."""
        async with self._lock:
            if self._state == EngineState.RUNNING:
                logger.warning("Already running")
                return
            if self._state == EngineState.STARTING:
                logger.info("Engine is already starting, ignoring redundant start request")
                return
            if self._state not in (EngineState.IDLE, EngineState.ERROR, EngineState.MUTED):
                raise RuntimeError(f"Cannot start from state {self._state}")

            self._state = EngineState.STARTING
            self._active_config = config
            self._is_muted = False
            self._last_error = None

        try:
            # 1. Target profile check (fail-closed immediately if unset)
            if not self._active_profile_id:
                raise RuntimeError("No target voice profile selected. Set a profile before starting.")

            # 2. Assets readiness check
            if not self._cfg.all_assets_ready():
                missing = [k for k, v in self._cfg.check_assets().items() if not v]
                raise RuntimeError(f"MeanVC2 assets missing: {', '.join(missing)}. Download weights first.")

            # 3. Ensure worker is running
            if not self._worker_proc:
                await self._start_worker()

            # 4. Load model if not loaded
            if not self._worker_loaded.is_set():
                await self._load_model()

            # 5. Apply target speaker
            if self._active_profile_id not in ("clean-mic", "bypass"):
                if not self._worker_target_set.is_set():
                    await self._apply_profile(self._active_profile_id)

            # Reset caches
            await self._send_to_worker({"cmd": "init_cache"})

            # Start processing loop
            self._processing_task = asyncio.create_task(self._processing_loop())

            async with self._lock:
                self._state = EngineState.RUNNING

            # Start WASAPI physical capture & render streams
            try:
                if getattr(config, "enable_audio_streams", True):
                    self._audio_bridge.start(
                        input_device_id=config.input_device_id,
                        output_device_id=config.output_device_id,
                        monitor_device_id=getattr(config, "monitor_device_id", None),
                        live_call_mode=True,
                        enable_monitor=getattr(config, "monitor_enabled", True) if getattr(config, "monitor_enabled", None) is not None else True,
                    )
            except Exception as e:
                logger.warning("WASAPI audio bridge streams not started", error=str(e))

            logger.info("MeanVC2 neural streaming started")
        except Exception as e:
            async with self._lock:
                self._state = EngineState.ERROR
                self._is_muted = True
                self._last_error = str(e)
            logger.error("MeanVC2 start failed", error=str(e))
            raise

    async def _load_model(self) -> None:
        """Send load command to worker."""
        cfg = self._cfg
        load_msg = {
            "cmd": "load",
            "config": {
                "source_dir": cfg.source_dir,
                "model_variant": cfg.model_variant,
                "device": cfg.device,
                "vc_model_path": cfg.vc_model_path,
                "vc_config_path": cfg.vc_config_path,
                "asr_model_path": cfg.asr_model_path,
                "vocoder_path": cfg.vocoder_path,
                "speaker_model_path": cfg.speaker_model_path,
                "wavlm_config_path": cfg.wavlm_config_path,
            }
        }
        await self._send_to_worker(load_msg)
        try:
            await asyncio.wait_for(self._worker_loaded.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            raise RuntimeError("MeanVC2 model loading timed out (120s)")
        if self._state == EngineState.ERROR:
            raise RuntimeError(f"MeanVC2 model loading failed: {self._last_error}")

        # Warmup
        await self._send_to_worker({"cmd": "warmup"})

    async def _apply_profile(self, profile_id: str) -> None:
        """Set target speaker from profile."""
        profile = self._profile_store.get(profile_id)
        if not profile:
            raise RuntimeError(f"Profile not found: {profile_id}")
        emb_path = self._profile_store.get_embedding_path(profile_id)
        self._worker_target_set.clear()
        await self._send_to_worker({
            "cmd": "set_target",
            "wav_path": profile.reference_wav_path,
            "profile_id": profile_id,
            "embedding_path": emb_path,
        })
        try:
            await asyncio.wait_for(self._worker_target_set.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Target speaker embedding extraction timed out (60s)")
        if self._state == EngineState.ERROR:
            raise RuntimeError(f"Setting target speaker failed: {self._last_error}")

    async def _processing_loop(self) -> None:
        """Main audio processing loop — feeds input chunks to worker."""
        CHUNK = 2560  # 160ms at 16kHz — same as upstream CHUNK
        while self._state == EngineState.RUNNING:
            if self._input_queue:
                chunk = self._input_queue.popleft()
                # Send to worker
                try:
                    b64 = base64.b64encode(chunk.astype(np.float32).tobytes()).decode()
                    await self._send_to_worker({"cmd": "process_chunk", "data_b64": b64})
                except Exception as e:
                    logger.error("Failed to send chunk to worker", error=str(e))
                    self._state = EngineState.ERROR
                    self._last_error = str(e)
                    self._is_muted = True
                    break
            else:
                await asyncio.sleep(0.005)  # 5ms polling

    def push_audio(self, samples: np.ndarray) -> bool:
        """Thread-safe: push captured mic audio (float32, 16kHz) to input queue."""
        if self._state != EngineState.RUNNING or self._is_muted:
            return False
        if len(self._input_queue) >= self._cfg.max_input_queue:
            self._overruns += 1
            return False
        self._input_queue.append(samples.astype(np.float32))
        return True

    def pop_converted_audio(self, n_samples: int) -> np.ndarray:
        """Thread-safe: pop converted audio for output callback."""
        if self._is_muted or self._passthrough_enabled:
            return np.zeros(n_samples, dtype=np.float32)
        accumulated = []
        total = 0
        while total < n_samples and self._output_queue:
            chunk = self._output_queue.popleft()
            accumulated.append(chunk)
            total += len(chunk)
        if not accumulated:
            self._underruns += 1
            return np.zeros(n_samples, dtype=np.float32)
        combined = np.concatenate(accumulated)
        if len(combined) > n_samples:
            remainder = combined[n_samples:]
            self._output_queue.appendleft(remainder)
            return combined[:n_samples]
        elif len(combined) < n_samples:
            pad = np.zeros(n_samples - len(combined), dtype=np.float32)
            return np.concatenate([combined, pad])
        return combined

    async def stop(self) -> None:
        """Stop conversion stream."""
        async with self._lock:
            self._state = EngineState.STOPPING

        self._audio_bridge.stop()

        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        self._input_queue.clear()
        self._output_queue.clear()

        async with self._lock:
            self._state = EngineState.IDLE
            self._active_config = None
        logger.info("MeanVC2 stream stopped")

    async def shutdown(self) -> None:
        """Full shutdown: stop stream, kill worker."""
        try:
            if self._state == EngineState.RUNNING:
                await self.stop()
        except Exception:
            pass
        try:
            await self._send_to_worker({"cmd": "shutdown"})
            await asyncio.sleep(0.5)
        except Exception:
            pass
        await self._kill_worker()
        logger.info("MeanVC2 adapter shut down")

    async def set_passthrough(self, enabled: bool) -> None:
        """Raw passthrough — user-initiated only. Never auto-fallback."""
        if enabled and self._state != EngineState.RUNNING:
            raise RuntimeError("Passthrough requires RUNNING engine — deliberate user action only.")
        self._passthrough_enabled = enabled
        logger.warning("MeanVC2 passthrough", enabled=enabled)

    async def set_params(self, params: dict[str, Any]) -> None:
        """Update runtime parameters."""
        if "muted" in params:
            self._is_muted = bool(params["muted"])
        if "model_variant" in params:
            # Requires restart
            self._cfg.model_variant = params["model_variant"]

    async def select_model(self, manifest: Any) -> None:
        """Switch voice profile (target speaker)."""
        profile_id = getattr(manifest, "model_id", str(manifest))
        await self.select_voice_profile(profile_id)

    async def select_voice_profile(self, profile_id: str) -> None:
        """Switch to a different voice profile, or clean-mic bypass."""
        self._active_profile_id = profile_id
        if profile_id in ("clean-mic", "bypass"):
            logger.info("Clean mic / bypass mode selected")
            return

        profile = self._profile_store.get(profile_id)
        if not profile:
            raise RuntimeError(f"Voice profile {profile_id} not found")
        if profile.volume_gain is not None:
            self._audio_bridge.output_volume = float(profile.volume_gain)
        if self._state == EngineState.RUNNING and self._worker_loaded.is_set():
            self._worker_target_set.clear()
            await self._apply_profile(profile_id)
        logger.info("Voice profile selected", profile_id=profile_id)

    async def panic_mute(self) -> None:
        """Immediate mute — fail-closed."""
        self._is_muted = True
        self._output_queue.clear()
        self._audio_bridge.panic_mute()
        async with self._lock:
            if self._state == EngineState.RUNNING:
                self._state = EngineState.MUTED
        logger.warning("MeanVC2 PANIC MUTE engaged")

    async def un_mute(self) -> None:
        """Release panic mute on explicit user action."""
        self._is_muted = False
        self._audio_bridge.un_mute()
        async with self._lock:
            if self._state == EngineState.MUTED:
                self._state = EngineState.RUNNING
        logger.info("MeanVC2 panic mute released")

    unmute = un_mute
    stop_stream = stop
    start_stream = start

    async def status(self) -> LiveStatus:
        actual_state = self._state
        if (self._is_muted or self._audio_bridge.is_muted) and actual_state == EngineState.RUNNING:
            actual_state = EngineState.MUTED
        return LiveStatus(
            state=actual_state,
            latency_p50_ms=self._last_inference_ms,
            latency_p95_ms=self._last_inference_ms * 1.5,
            underruns=self._underruns + self._audio_bridge.underruns,
            frames_processed=self._frames_processed + self._audio_bridge.render_frames_total,
            active_config=self._active_config,
        )

    async def health(self) -> EngineInfo:
        worker_alive = (self._worker_proc is not None
                        and self._worker_proc.returncode is None)
        caps = [
            "neural_voice_conversion",
            "zero_shot",
            "streaming_40ms",
            "streaming_120ms",
            "wasapi_integration",
            "fail_closed_mute",
        ]
        if not worker_alive:
            caps.append("worker_not_running")
        return EngineInfo(
            engine_id=self.ENGINE_ID,
            name=self.ENGINE_NAME,
            version=self.ENGINE_VERSION,
            capabilities=caps,
            state=self._state,
        )

    def get_capabilities(self) -> dict:
        return {
            "supports_neural_vc": True,
            "supports_zero_shot": True,
            "model_variants": ["40ms", "120ms"],
            "sample_rate": 16000,
            "is_production": True,
            "requires_assets": True,
            "license": "apache-2.0",
        }

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    async def create_voice_profile(
        self,
        display_name: str,
        reference_wav_path: str,
        consent_declared: bool,
        consent_statement: str = "",
        language_hint: str = "",
        avatar_id: str = "female_eva",
        category: str = "Female",
        badge: str = "AI",
    ) -> VoiceProfile:
        """Create and save a new voice profile from a reference WAV."""
        if not consent_declared:
            raise ValueError(
                "Consent declaration is required (Rule 4 — Voice Model Consent & Disclosure). "
                "You must attest that you own or have explicit permission to use this voice."
            )

        validation = validate_reference_audio(reference_wav_path)
        if not validation["valid"]:
            raise ValueError(f"Invalid reference audio: {validation['error']}")

        sha256 = compute_file_sha256(reference_wav_path)
        profile = VoiceProfile(
            display_name=display_name,
            reference_wav_path=reference_wav_path,
            reference_sha256=sha256,
            reference_duration_s=validation["duration_s"],
            reference_sample_rate=validation["sample_rate"],
            model_variant=self._cfg.model_variant,
            consent_declared=True,
            consent_statement=consent_statement,
            language_hint=language_hint,
            avatar_id=avatar_id,
            category=category,
            badge=badge,
        )
        self._profile_store.save(profile)
        logger.info("Voice profile created", profile_id=profile.profile_id, name=display_name)
        return profile

    def update_voice_profile(self, profile_id: str, updates: dict) -> Optional[VoiceProfile]:
        """Update tuning parameters or metadata for an existing profile."""
        profile = self._profile_store.get(profile_id)
        if not profile:
            return None
        allowed_keys = {
            "display_name", "avatar_id", "category", "badge",
            "pitch_shift", "timbre_ratio", "volume_gain", "sound_effect", "favorite", "description"
        }
        for k, v in updates.items():
            if k in allowed_keys and hasattr(profile, k):
                setattr(profile, k, v)
        if profile.profile_id == self._active_profile_id and profile.volume_gain is not None:
            self._audio_bridge.output_volume = float(profile.volume_gain)
        self._profile_store.save(profile)
        return profile

    def set_monitor_enabled(self, enabled: bool) -> None:
        self._audio_bridge.set_monitor_enabled(enabled)

    def set_monitor_device(self, device_id: Optional[int | str]) -> None:
        if self._audio_bridge is not None:
            self._audio_bridge.set_monitor_device(device_id)

    def set_monitor_volume(self, volume: float) -> None:
        self._audio_bridge.set_monitor_volume(volume)

    def set_output_volume(self, volume: float) -> None:
        self._audio_bridge.set_output_volume(volume)

    def get_active_profile(self) -> Optional[VoiceProfile]:
        """Return the currently selected VoiceProfile instance or None."""
        if not self._active_profile_id:
            return None
        return self._profile_store.get(self._active_profile_id)

    def list_voice_profiles(self) -> list[VoiceProfile]:
        return self._profile_store.list_profiles()

    def delete_voice_profile(self, profile_id: str) -> bool:
        return self._profile_store.delete(profile_id)

    def get_asset_status(self) -> dict:
        """Return asset readiness info for the UI."""
        checks = self._cfg.check_assets()
        return {
            "all_ready": all(checks.values()),
            "assets": checks,
            "model_dir": str(self._cfg.model_dir),
            "model_variant": self._cfg.model_variant,
            "source_dir": self._cfg.source_dir,
        }
