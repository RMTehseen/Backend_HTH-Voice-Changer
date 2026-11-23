"""Domain contracts, types, and the VoiceEngine async Protocol."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class EngineState(str, Enum):
    """Lifecycle states of the voice engine."""

    UNINITIALIZED = "UNINITIALIZED"
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    MUTED_ERROR = "MUTED_ERROR"
    CRASH_LOOP = "CRASH_LOOP"
    SHUTDOWN = "SHUTDOWN"


@dataclass(frozen=True)
class EngineCapabilities:
    """Declared capabilities of a voice engine adapter."""

    supports_pitch_shift: bool = True
    supports_formant_shift: bool = True
    supports_noise_gate: bool = True
    supports_atomic_model_switching: bool = False
    commercial_distribution_allowed: bool = False
    is_poc_only: bool = True
    pitch_range: tuple[float, float] = (-12.0, 12.0)
    formant_range: tuple[float, float] = (-4.0, 4.0)
    default_sample_rate: int = 16000
    supported_sample_rates: tuple[int, ...] = (16000, 24000, 48000)
    default_block_size: int = 160

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports_pitch_shift": self.supports_pitch_shift,
            "supports_formant_shift": self.supports_formant_shift,
            "supports_noise_gate": self.supports_noise_gate,
            "supports_atomic_model_switching": self.supports_atomic_model_switching,
            "commercial_distribution_allowed": self.commercial_distribution_allowed,
            "is_poc_only": self.is_poc_only,
            "pitch_range": list(self.pitch_range),
            "formant_range": list(self.formant_range),
            "default_sample_rate": self.default_sample_rate,
            "supported_sample_rates": list(self.supported_sample_rates),
            "default_block_size": self.default_block_size,
        }


@dataclass
class DeviceInfo:
    """Audio hardware or virtual endpoint descriptor."""

    id: int | str
    name: str
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_sample_rate: int = 16000
    host_api: str = "Windows Audio"
    is_default_input: bool = False
    is_default_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "max_input_channels": self.max_input_channels,
            "max_output_channels": self.max_output_channels,
            "default_sample_rate": self.default_sample_rate,
            "host_api": self.host_api,
            "is_default_input": self.is_default_input,
            "is_default_output": self.is_default_output,
        }


@dataclass
class EngineParams:
    """Real-time tunable engine DSP parameters."""

    pitch_shift: float = 0.0
    formant_shift: float = 0.0
    volume: float = 1.0
    gate_threshold: float = 0.0
    hear_yourself: bool = False
    speaker_index: int = 0
    muted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch_shift": self.pitch_shift,
            "formant_shift": self.formant_shift,
            "volume": self.volume,
            "gate_threshold": self.gate_threshold,
            "hear_yourself": self.hear_yourself,
            "speaker_index": self.speaker_index,
            "muted": self.muted,
        }


@dataclass
class LiveConfig:
    """Configuration requested when starting an active audio conversion stream."""

    input_device_id: Optional[int | str] = None
    output_device_id: Optional[int | str] = None
    monitor_device_id: Optional[int | str] = None
    sample_rate: int = 16000
    block_size: int = 160
    model_id: str = "jvs"
    params: EngineParams = field(default_factory=EngineParams)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_device_id": self.input_device_id,
            "output_device_id": self.output_device_id,
            "monitor_device_id": self.monitor_device_id,
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "model_id": self.model_id,
            "params": self.params.to_dict(),
        }


@dataclass
class AudioMetrics:
    """Lock-free metrics reported by the real-time audio callback."""

    input_peak: float = 0.0
    output_peak: float = 0.0
    dropped_frames: int = 0
    underruns: int = 0
    callback_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_peak": self.input_peak,
            "output_peak": self.output_peak,
            "dropped_frames": self.dropped_frames,
            "underruns": self.underruns,
            "callback_duration_ms": self.callback_duration_ms,
        }


@dataclass
class EngineStatus:
    """Current snapshot of engine execution and state."""

    state: EngineState
    is_passthrough: bool
    current_model: str
    capabilities: EngineCapabilities
    metrics: AudioMetrics = field(default_factory=AudioMetrics)
    active_config: Optional[LiveConfig] = None
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "is_passthrough": self.is_passthrough,
            "current_model": self.current_model,
            "capabilities": self.capabilities.to_dict(),
            "metrics": self.metrics.to_dict(),
            "active_config": self.active_config.to_dict() if self.active_config else None,
            "last_error": self.last_error,
        }


@dataclass
class EngineHealth:
    """Diagnostic health information from supervisor and child process."""

    healthy: bool
    process_alive: bool
    process_pid: Optional[int]
    last_heartbeat_age_s: float
    restart_count: int
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "process_alive": self.process_alive,
            "process_pid": self.process_pid,
            "last_heartbeat_age_s": self.last_heartbeat_age_s,
            "restart_count": self.restart_count,
            "error_message": self.error_message,
        }


@dataclass
class EngineInfo:
    """Static metadata identifying the engine adapter."""

    name: str
    version: str
    license_id: str
    is_poc_only: bool
    capabilities: EngineCapabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "license_id": self.license_id,
            "is_poc_only": self.is_poc_only,
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass
class ModelManifest:
    """Metadata and integrity descriptors for a voice model."""

    model_id: str
    name: str
    speaker_count: int
    license_id: str
    is_commercial_allowed: bool
    sha256_hash: str
    paraphernalia_path: str


@runtime_checkable
class VoiceEngine(Protocol):
    """Generic asynchronous interface implemented by all HTH voice engine adapters."""

    async def initialize(self) -> None:
        """Start engine supervisor and spawn worker process."""
        ...

    async def shutdown(self) -> None:
        """Gracefully terminate stream, audio engine, and kill child process tree."""
        ...

    async def ping(self) -> float:
        """Perform a round-trip IPC latency measurement in milliseconds."""
        ...

    async def list_devices(self) -> list[DeviceInfo]:
        """Enumerate host audio devices discovered by the engine backend."""
        ...

    async def start(self, config: LiveConfig) -> None:
        """Begin live real-time voice conversion stream."""
        ...

    async def stop(self) -> None:
        """Stop active audio streams and mute output."""
        ...

    async def set_passthrough(self, enabled: bool) -> None:
        """Toggle raw mic passthrough. Must be explicit and user-initiated."""
        ...

    async def set_params(self, params: EngineParams) -> None:
        """Update real-time DSP parameters (pitch, formant, volume, gate)."""
        ...

    async def select_model(self, model_id: str) -> None:
        """Switch active acoustic voice model."""
        ...

    async def status(self) -> EngineStatus:
        """Query the current state and audio metrics."""
        ...

    async def health(self) -> EngineHealth:
        """Query supervisor and child process health."""
        ...

    def get_capabilities(self) -> EngineCapabilities:
        """Retrieve static capabilities and constraints supported by this engine."""
        ...
