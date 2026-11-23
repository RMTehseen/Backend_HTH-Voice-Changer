"""Noise suppression configuration and telemetry REST router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from hth.audio.conversion.contracts import VoiceProfile
from hth.audio.noise.contracts import (
    NoiseConfig,
    NoiseMetrics,
    NoisePreset,
    StartLiveCallRequest,
    StartTestSinkRequest,
    TestSinkStatusResponse,
)
from hth.audio.noise.suppressor import FRAME_SIZE, SAMPLE_RATE, NoiseSuppressor
from hth.audio.pipeline import AudioPipelineConfig, AudioPipelineMetrics, get_audio_pipeline

router = APIRouter(prefix="/api/v1/noise", tags=["noise"])


def _build_status_response(m: AudioPipelineMetrics) -> TestSinkStatusResponse:
    return TestSinkStatusResponse(
        is_running=m.is_running,
        is_muted=m.is_muted,
        is_test_sink_mode=m.is_test_sink_mode,
        test_harness_banner=m.test_harness_banner,
        vb_cable_transmission_enabled=m.live_call_mode,
        is_converting=m.is_converting,
        active_profile=m.active_profile,
        live_call_mode=m.live_call_mode,
        active_preset=m.active_preset,
        sample_rate=m.sample_rate,
        frame_size=m.frame_size,
        processing_time_p50_ms=m.processing_time_p50_ms,
        processing_time_p95_ms=m.processing_time_p95_ms,
        processing_time_p99_ms=m.processing_time_p99_ms,
        processing_time_max_ms=m.processing_time_max_ms,
        buffer_occupancy_samples=m.buffer_occupancy_samples,
        buffer_occupancy_ms=m.buffer_occupancy_ms,
        capture_frames_total=m.capture_frames_total,
        render_frames_total=m.render_frames_total,
        overflow_count=m.overflow_count,
        underrun_count=m.underrun_count,
        dropped_samples_total=m.dropped_samples_total,
        discontinuity_epoch=m.discontinuity_epoch,
        native_callback_errors=m.native_callback_errors,
        asrc_corr_ppm=m.asrc_corr_ppm,
        asrc_rho=m.asrc_rho,
        asrc_state=m.asrc_state,
        cpu_pct=m.cpu_pct,
        ram_mb=m.ram_mb,
        last_error=m.last_error,
    )


# Singleton suppressor instance
_suppressor: NoiseSuppressor | None = None


def get_noise_suppressor() -> NoiseSuppressor:
    """Dependency / accessor for the global noise suppressor."""
    global _suppressor
    if _suppressor is None:
        _suppressor = NoiseSuppressor(preset=NoisePreset.LIGHT)
    return _suppressor


def set_noise_suppressor(suppressor: NoiseSuppressor | None) -> None:
    """Setter for testing."""
    global _suppressor
    _suppressor = suppressor


class SetPresetRequest(BaseModel):
    preset: NoisePreset = Field(
        ...,
        description="Target noise suppression preset: OFF, LIGHT, or STRONG.",
    )


@router.get("/config", response_model=NoiseConfig)
async def get_noise_config() -> NoiseConfig:
    """Retrieve active noise suppression preset and DSP parameters."""
    ns = get_noise_suppressor()
    return NoiseConfig(
        preset=ns.get_preset(),
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
    )


@router.post("/preset")
async def set_noise_preset(req: SetPresetRequest) -> dict[str, Any]:
    """Switch noise suppression preset with smooth crossfade interpolation."""
    ns = get_noise_suppressor()
    ns.set_preset(req.preset)
    return {
        "ok": True,
        "preset": req.preset.value,
        "message": f"Noise suppression preset set to {req.preset.value}",
    }


@router.get("/metrics", response_model=NoiseMetrics)
async def get_noise_metrics() -> NoiseMetrics:
    """Retrieve lock-free DSP performance and latency telemetry."""
    ns = get_noise_suppressor()
    return ns.get_metrics()


@router.post("/testsink/start", response_model=TestSinkStatusResponse)
async def start_testsink(req: StartTestSinkRequest) -> TestSinkStatusResponse:
    """Start the real-time audio pipeline in TestSink mode (headphones only).

    VB-CABLE output is hard-muted to guarantee zero unconverted voice transmission.
    """
    pipeline = get_audio_pipeline()
    prof = (
        VoiceProfile.FEMALE_TO_MALE
        if "to_male" in req.profile.lower()
        else VoiceProfile.MALE_TO_FEMALE
    )
    cfg = AudioPipelineConfig(
        input_device_id=req.input_device_id,
        output_device_id=req.output_device_id,
        preset=req.preset,
        test_sink_mode=True,
        enable_vb_cable=False,
        enable_voice_conversion=req.enable_voice_conversion,
        conversion_profile=prof,
        live_call_mode=False,
    )
    pipeline.start(cfg)
    return _build_status_response(pipeline.get_metrics())


@router.post("/live/start", response_model=TestSinkStatusResponse)
async def start_live_call(req: StartLiveCallRequest) -> TestSinkStatusResponse:
    """Start live real-time voice conversion routed to VB-CABLE Input for WhatsApp Desktop/Teams.

    Fail-closed guard: VB-CABLE output is strictly ZEROED unless conversion is active.
    """
    pipeline = get_audio_pipeline()
    prof = (
        VoiceProfile.FEMALE_TO_MALE
        if "to_male" in req.profile.lower()
        else VoiceProfile.MALE_TO_FEMALE
    )
    cfg = AudioPipelineConfig(
        input_device_id=req.input_device_id,
        output_device_id=req.output_device_id,
        preset=req.preset,
        test_sink_mode=False,
        enable_vb_cable=True,
        enable_voice_conversion=True,
        conversion_profile=prof,
        live_call_mode=True,
    )
    pipeline.start(cfg)
    return _build_status_response(pipeline.get_metrics())


@router.post("/conversion/profile", response_model=TestSinkStatusResponse)
async def set_conversion_profile(profile: str) -> TestSinkStatusResponse:
    """Switch active voice conversion target profile (Male->Female or Female->Male)."""
    pipeline = get_audio_pipeline()
    prof = (
        VoiceProfile.FEMALE_TO_MALE if "to_male" in profile.lower() else VoiceProfile.MALE_TO_FEMALE
    )
    pipeline.set_conversion_profile(prof)
    return _build_status_response(pipeline.get_metrics())


@router.post("/testsink/stop", response_model=TestSinkStatusResponse)
async def stop_testsink() -> TestSinkStatusResponse:
    """Halt real-time TestSink / LiveCall streams and mute output."""
    pipeline = get_audio_pipeline()
    pipeline.stop()
    return _build_status_response(pipeline.get_metrics())


@router.get("/testsink/status", response_model=TestSinkStatusResponse)
async def get_testsink_status() -> TestSinkStatusResponse:
    """Retrieve full real-time telemetry for the audio pipeline."""
    pipeline = get_audio_pipeline()
    return _build_status_response(pipeline.get_metrics())


@router.post("/testsink/panic")
async def testsink_panic() -> dict[str, Any]:
    """Engage immediate panic mute on the real-time TestSink audio stream."""
    pipeline = get_audio_pipeline()
    pipeline.panic_mute()
    return {"status": "muted", "message": "TestSink audio output muted immediately"}


@router.post("/testsink/record")
async def testsink_record(duration_seconds: float = 5.0) -> dict[str, Any]:
    """Start tapping real physical microphone audio into a local WAV recording."""
    pipeline = get_audio_pipeline()
    path = pipeline.start_recording(duration_seconds=duration_seconds)
    return {
        "status": "recording",
        "file_path": str(path),
        "duration_seconds": duration_seconds,
        "message": "Capturing real physical microphone PCM audio",
    }
