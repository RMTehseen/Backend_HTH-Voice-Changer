"""Domain contracts and data types for the real-time Noise Suppression (NS) stage."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class NoisePreset(str, Enum):
    """Supported noise suppression presets."""

    OFF = "OFF"
    LIGHT = "LIGHT"  # Recommended default: gentle suppression, preserves consonant warmth
    STRONG = "STRONG"  # Aggressive suppression for noisy environments


class NoiseConfig(BaseModel):
    """Configuration for the real-time noise suppressor."""

    preset: NoisePreset = Field(
        default=NoisePreset.LIGHT,
        description="Active noise suppression preset: OFF, LIGHT, or STRONG.",
    )
    sample_rate: int = Field(
        default=48000, description="Sampling rate in Hz (RNNoise requires 48kHz)."
    )
    frame_size: int = Field(default=480, description="Block size in samples (10.0 ms @ 48kHz).")


class NoiseMetrics(BaseModel):
    """Telemetry and timing metrics for the real-time noise suppression callback."""

    frames_processed: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    underrun_count: int = 0
    last_speech_probability: float = 0.0
    active_preset: NoisePreset = NoisePreset.LIGHT


class TestSinkStatusResponse(BaseModel):
    """Complete real-time telemetry for the TestSink audio harness."""

    is_running: bool = False
    is_muted: bool = False
    is_test_sink_mode: bool = True
    test_harness_banner: str = "TEST HARNESS — LIVE TRANSMISSION DISABLED"
    vb_cable_transmission_enabled: bool = False
    is_converting: bool = False
    active_profile: str = "male_to_female"
    live_call_mode: bool = False
    active_preset: NoisePreset = NoisePreset.LIGHT
    sample_rate: int = 48000
    frame_size: int = 480
    processing_time_p50_ms: float = 0.0
    processing_time_p95_ms: float = 0.0
    processing_time_p99_ms: float = 0.0
    processing_time_max_ms: float = 0.0
    buffer_occupancy_samples: int = 0
    buffer_occupancy_ms: float = 0.0
    capture_frames_total: int = 0
    render_frames_total: int = 0
    overflow_count: int = 0
    underrun_count: int = 0
    dropped_samples_total: int = 0
    discontinuity_epoch: int = 0
    native_callback_errors: int = 0
    asrc_corr_ppm: float = 0.0
    asrc_rho: float = 1.0
    asrc_state: str = "HEALTHY"
    cpu_pct: float = 0.0
    ram_mb: float = 0.0
    last_error: str | None = None


class StartTestSinkRequest(BaseModel):
    """Request parameters to start local TestSink audio monitoring."""

    input_device_id: int | str | None = None
    output_device_id: int | str | None = None
    preset: NoisePreset = NoisePreset.LIGHT
    enable_voice_conversion: bool = False
    profile: str = "male_to_female"


class StartLiveCallRequest(BaseModel):
    """Request parameters to start live conversion routed to VB-CABLE for WhatsApp/Teams."""

    input_device_id: int | str | None = None
    output_device_id: int | str | None = None
    preset: NoisePreset = NoisePreset.LIGHT
    profile: str = "male_to_female"
