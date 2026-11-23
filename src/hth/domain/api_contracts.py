"""FastAPI-layer Pydantic contracts for Phase 2+.

These types are used by the FastAPI routers, services, and FakeEngine.
The existing contracts.py (dataclass-based) is kept for the Phase 1 Beatrice adapter.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel


class EngineState(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"
    MUTED = "MUTED"


class DeviceType(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    VIRTUAL = "VIRTUAL"


class LicenseType(str, Enum):
    MIT = "MIT"
    PERSONAL_POC_ONLY = "PERSONAL_POC_ONLY"
    COMMERCIAL = "COMMERCIAL"
    UNKNOWN = "UNKNOWN"


class DeviceInfo(BaseModel):
    id: str
    friendly_name: str
    device_type: DeviceType
    sample_rates: list[int]
    default_sample_rate: int
    is_default: bool


class EngineInfo(BaseModel):
    engine_id: str
    name: str
    version: str
    capabilities: list[str]
    state: EngineState


class LiveConfig(BaseModel):
    sample_rate: int = 48000
    buffer_size: int = 512
    input_device_id: str | None = None
    output_device_id: str | None = None
    monitor_device_id: str | None = None
    monitor_enabled: bool = False
    monitor_volume: float = 1.0
    enable_audio_streams: bool = True


class LiveStatus(BaseModel):
    state: EngineState
    latency_p50_ms: float
    latency_p95_ms: float
    underruns: int
    frames_processed: int
    active_config: LiveConfig | None = None


class AudioMetrics(BaseModel):
    frame_count: int
    dropped_frames: int
    cpu_pct: float
    ram_mb: float


class ModelManifest(BaseModel):
    model_id: str
    engine_id: str
    name: str
    format: str
    sha256: str
    size_bytes: int
    license_type: LicenseType
    consent_required: bool


class ConsentRecord(BaseModel):
    model_id: str
    attested_by: str
    attested_at: datetime.datetime
    license_type: LicenseType
    notes: str = ""


class JobState(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    progress: float
    error: str | None = None


class SupportedModelFormat(str, Enum):
    """Voice model format identifiers enforced for strict engine isolation."""

    BEATRICE_PARAPHERNALIA = "beatrice_paraphernalia"  # .bin + .toml
    APPLIO_RVC = "applio_rvc"  # .pth + .index
    FAKE_MODEL = "fake_model"  # deterministic in-memory mock
    ACOUSTIC_PROFILE = "acoustic_profile"  # real-time vocal tract resonance profile


class AudioContainerFormat(str, Enum):
    """Allowed container formats for voice-note import/export."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    M4A = "m4a"


class VoiceNoteJobStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class VoiceNoteJobCreate(BaseModel):
    target_format: AudioContainerFormat = AudioContainerFormat.WAV
    model_id: str = "default"
    model_format: SupportedModelFormat = SupportedModelFormat.FAKE_MODEL
    pitch_shift: float = 0.0
    formant_shift: float = 0.0


class VoiceNoteJobStatus(BaseModel):
    job_id: str
    status: VoiceNoteJobStatusEnum
    progress: float  # 0.0 to 1.0
    input_filename: str
    input_format: str
    output_format: str
    duration_seconds: float = 0.0
    processed_frames: int = 0
    total_frames: int = 0
    error: str | None = None
    has_source_audio: bool = False
    has_converted_audio: bool = False
    created_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class VoiceNoteCapabilities(BaseModel):
    ffmpeg_detected: bool
    ffmpeg_version: str | None = None
    available_encoders: list[str] = []
    supported_input_formats: list[str] = ["wav", "mp3", "ogg", "flac", "m4a"]
    supported_output_formats: list[str] = ["wav"]
    max_file_size_mb: int = 25
    max_duration_seconds: int = 300
    live_conversion_active: bool = False
    offline_job_active: bool = False


class LiveRouteGuideResponse(BaseModel):
    vb_cable_installed: bool
    live_engine_running: bool
    current_output_device: str | None = None
    recommended_whatsapp_input: str = "CABLE Output (VB-Audio Virtual Cable)"
    steps: list[str]
    whatsapp_bubble_notice: str
    attachment_warning: str


class RetentionCleanupResponse(BaseModel):
    deleted_temp_files: int
    deleted_export_files: int
    freed_bytes: int
    timestamp: datetime.datetime


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    detail: str | None = None
    timestamp: datetime.datetime


class VoiceEngineProtocol(Protocol):
    """Async protocol implemented structurally by all Phase 2+ engine adapters."""

    async def ping(self) -> bool: ...

    async def list_devices(self) -> list[DeviceInfo]: ...

    async def start(self, config: LiveConfig) -> None: ...

    async def stop(self) -> None: ...

    async def set_passthrough(self, enabled: bool) -> None: ...

    async def set_params(self, params: dict[str, Any]) -> None: ...

    async def select_model(self, manifest: Any) -> None: ...

    async def status(self) -> LiveStatus: ...

    async def health(self) -> EngineInfo: ...
