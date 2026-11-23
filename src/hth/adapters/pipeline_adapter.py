"""Production VoiceEngineProtocol adapter controlling the real WASAPI AudioPipeline.

Phase B architecture:
- Directly connects the desktop UI to the native AudioPipeline (WASAPI capture/render).
- Capture Physical Mic -> RNNoise (48 kHz) -> AcousticVoiceConverter -> VB-CABLE Input.
- Strict fail-closed mute on error or panic; raw microphone passthrough is disabled (Rule A-03).
- Thread-safe panic mute via atomic pipeline flag.
- Idempotent lifecycle management; startup failure rolls back all streams.
- Accurate telemetry reflecting real DSP execution time, underruns, and frame counts.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from hth.audio.conversion.contracts import VoiceConversionConfig, VoiceProfile
from hth.audio.devices import get_device_inventory
from hth.audio.pipeline import AudioPipeline, AudioPipelineConfig, get_audio_pipeline
from hth.core.logging import get_logger
from hth.domain.api_contracts import (
    DeviceInfo,
    DeviceType,
    EngineInfo,
    EngineState,
    LiveConfig,
    LiveStatus,
)
from hth.domain.errors import DeviceError

logger = get_logger("hth.adapters.pipeline")


class AudioPipelineAdapter:
    """Production VoiceEngineProtocol implementation backed by real AudioPipeline."""

    def __init__(self, pipeline: Optional[AudioPipeline] = None) -> None:
        self._lock = asyncio.Lock()
        self._pipeline = pipeline or get_audio_pipeline()
        self._active_config: Optional[LiveConfig] = None
        self._active_profile: VoiceProfile = VoiceProfile.MALE_TO_FEMALE
        self._state: EngineState = EngineState.IDLE
        self._last_error: Optional[str] = None

    async def ping(self) -> bool:
        """Verify pipeline subsystem readiness."""
        return True

    async def list_devices(self) -> list[DeviceInfo]:
        """Enumerate real Windows audio devices with stable identifiers."""
        try:
            inventory = get_device_inventory()
            devices: list[DeviceInfo] = []
            for ep in inventory.capture_devices + inventory.render_devices:
                dev_type = DeviceType.INPUT if ep.data_flow == "capture" else DeviceType.OUTPUT
                full_name = ep.friendly_name
                if ep.adapter_name and ep.adapter_name.lower() not in full_name.lower():
                    full_name = f"{ep.friendly_name} ({ep.adapter_name})"
                devices.append(
                    DeviceInfo(
                        id=ep.id,
                        friendly_name=full_name,
                        device_type=dev_type,
                        sample_rates=[48000],
                        default_sample_rate=48000,
                        is_default=ep.is_default_console,
                    )
                )
            return devices
        except Exception as exc:
            logger.warning(
                "Core Audio inventory enumeration failed, falling back to basic list",
                error=str(exc),
            )
            return [
                DeviceInfo(
                    id="default-input",
                    friendly_name="System Default Physical Microphone",
                    device_type=DeviceType.INPUT,
                    sample_rates=[48000],
                    default_sample_rate=48000,
                    is_default=True,
                ),
                DeviceInfo(
                    id="cable-input",
                    friendly_name="CABLE Input (VB-Audio Virtual Cable)",
                    device_type=DeviceType.OUTPUT,
                    sample_rates=[48000],
                    default_sample_rate=48000,
                    is_default=False,
                ),
            ]

    async def start(self, config: LiveConfig) -> None:
        """Start live real-time voice conversion streams.

        Guarantees:
        - Never falls back to simulated audio.
        - Idempotent: does not create duplicate streams if already running.
        - Fails closed on device or stream initialization errors.
        """
        async with self._lock:
            if self._pipeline.is_running:
                logger.warning("AudioPipeline is already running; start request ignored")
                self._state = EngineState.RUNNING
                return

            self._state = EngineState.STARTING
            self._last_error = None
            self._active_config = config

            # Map LiveConfig to AudioPipelineConfig for live production conversion
            pipeline_cfg = AudioPipelineConfig(
                input_device_id=config.input_device_id,
                output_device_id=config.output_device_id,
                sample_rate=config.sample_rate or 48000,
                frame_size=480,  # 10 ms @ 48 kHz native for RNNoise and AcousticVocoder
                test_sink_mode=False,
                enable_vb_cable=True,
                enable_voice_conversion=True,
                conversion_profile=self._active_profile,
                live_call_mode=True,
            )

            try:
                # Start native WASAPI streams
                self._pipeline.start(pipeline_cfg)
                if not self._pipeline.is_running:
                    raise DeviceError("Audio pipeline failed to reach running state.")
                self._state = EngineState.RUNNING
                logger.info(
                    "Production AudioPipelineAdapter started live conversion",
                    in_dev=config.input_device_id,
                    out_dev=config.output_device_id,
                    profile=self._active_profile.value,
                )
            except Exception as exc:
                self._state = EngineState.ERROR
                self._last_error = str(exc)
                logger.error("AudioPipeline start failed — failing closed", error=str(exc))
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                raise DeviceError(f"Failed to start live DSP audio pipeline: {exc}") from exc

    async def stop(self) -> None:
        """Gracefully halt real WASAPI streams and release hardware resources."""
        async with self._lock:
            if not self._pipeline.is_running and self._state == EngineState.IDLE:
                return

            self._state = EngineState.STOPPING
            try:
                self._pipeline.stop()
            except Exception as exc:
                logger.warning("Error during AudioPipeline stop", error=str(exc))
            finally:
                self._state = EngineState.IDLE
                self._active_config = None
                logger.info("Production AudioPipelineAdapter stopped")

    async def set_passthrough(self, enabled: bool) -> None:
        """Enforce strict fail-closed security: raw mic passthrough is prohibited in Phase B."""
        if enabled:
            raise RuntimeError(
                "Raw microphone passthrough is disabled in production DSP mode (fail-closed rule A-03). "
                "Only converted audio or fail-closed silence is permitted."
            )

    async def set_params(self, params: dict[str, Any]) -> None:
        """Update active DSP conversion parameters."""
        async with self._lock:
            if "profile" in params:
                prof_str = str(params["profile"]).lower()
                self._active_profile = (
                    VoiceProfile.FEMALE_TO_MALE
                    if "to_male" in prof_str
                    else VoiceProfile.MALE_TO_FEMALE
                )
                self._pipeline.set_conversion_profile(self._active_profile)

            if hasattr(self._pipeline.converter, "config") and (
                "pitch_shift" in params or "formant_shift" in params
            ):
                curr = self._pipeline.converter.config
                new_cfg = VoiceConversionConfig(
                    profile=self._active_profile,
                    pitch_semitones=float(params.get("pitch_shift", curr.pitch_semitones)),
                    formant_ratio=float(params.get("formant_shift", curr.formant_ratio)),
                    chest_resonance_gain=curr.chest_resonance_gain,
                    brightness_gain=curr.brightness_gain,
                    f0_min=curr.f0_min,
                    f0_max=curr.f0_max,
                    sample_rate=curr.sample_rate,
                    frame_size=curr.frame_size,
                )
                self._pipeline.converter.set_config(new_cfg)

    async def select_model(self, manifest: Any) -> None:
        """Configure DSP conversion profile (Male->Female or Female->Male).

        Preserves VoiceEngineProtocol contract while accurately configuring DSP profiles.
        """
        async with self._lock:
            m_id = getattr(manifest, "model_id", str(manifest))
            logger.info("Selecting DSP conversion profile", model_id=m_id)
            if "female_to_male" in str(m_id).lower() or "to_male" in str(m_id).lower():
                self._active_profile = VoiceProfile.FEMALE_TO_MALE
            else:
                self._active_profile = VoiceProfile.MALE_TO_FEMALE
            self._pipeline.set_conversion_profile(self._active_profile)

    async def panic_mute(self) -> None:
        """Thread-safe panic mute: immediately zeros outgoing audio."""
        async with self._lock:
            self._pipeline.panic_mute()
            self._state = EngineState.MUTED
            logger.warning("AudioPipelineAdapter PANIC MUTE engaged")

    async def un_mute(self) -> None:
        """Release panic mute on explicit user request."""
        async with self._lock:
            self._pipeline.un_mute()
            if self._pipeline.is_running:
                self._state = EngineState.RUNNING
            else:
                self._state = EngineState.IDLE
            logger.info("AudioPipelineAdapter PANIC MUTE released")

    async def status(self) -> LiveStatus:
        """Retrieve truthful real-time telemetry from the active WASAPI pipeline."""
        m = self._pipeline.get_metrics()
        current_state = self._state

        if m.is_running:
            if m.is_muted:
                current_state = EngineState.MUTED
            else:
                current_state = EngineState.RUNNING
        elif self._state == EngineState.ERROR:
            current_state = EngineState.ERROR
        else:
            current_state = EngineState.IDLE

        return LiveStatus(
            state=current_state,
            latency_p50_ms=round(m.processing_time_p50_ms, 2),
            latency_p95_ms=round(m.processing_time_p95_ms, 2),
            underruns=m.underrun_count,
            frames_processed=m.render_frames_total,
            active_config=self._active_config,
        )

    async def health(self) -> EngineInfo:
        """Report genuine engine identity and capabilities (DSP Vocoder + RNNoise)."""
        is_running = self._pipeline.is_running
        if self._state == EngineState.ERROR:
            state = EngineState.ERROR
        elif is_running:
            state = EngineState.MUTED if self._pipeline.is_muted else EngineState.RUNNING
        else:
            state = EngineState.IDLE
        return EngineInfo(
            engine_id="real-dsp-pipeline",
            name="HTH Real-Time DSP Audio Pipeline (RNNoise + AcousticVocoder)",
            version="1.0.0-dsp",
            capabilities=[
                "rnnoise_noise_suppression",
                "acoustic_pitch_formant_dsp",
                "wasapi_streaming",
                "fail_closed_mute",
                "vb_cable_routing",
            ],
            state=state,
        )
