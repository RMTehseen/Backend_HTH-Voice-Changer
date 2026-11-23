"""Domain contracts and Pydantic models for Windows Core Audio routing and controls.

Phase 4 contracts for:
- Stable audio endpoint enumeration (MMDevice persistent GUIDs)
- Default role detection (Console, Communications)
- VB-CABLE detection and feedback loop prevention
- Global panic stop and deliberate natural-voice passthrough controls
- Opt-in 5-second test sample recording
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EndpointFormFactor(str, Enum):
    """Windows Core Audio endpoint form factor classification."""

    UNKNOWN = "Unknown"
    SPEAKERS = "Speakers"
    HEADPHONES = "Headphones"
    HEADSET = "Headset"
    MICROPHONE = "Microphone"
    LINE_LEVEL = "LineLevel"
    DIGITAL_AUDIO = "DigitalAudio"
    HANDSET = "Handset"


class AudioEndpointInfo(BaseModel):
    """Detailed audio hardware or virtual endpoint descriptor with stable GUID."""

    id: str = Field(description="Persistent Windows Core Audio endpoint ID string (GUID-based)")
    friendly_name: str = Field(description="User-facing friendly device name")
    adapter_name: str = Field(default="", description="Underlying audio adapter / driver name")
    data_flow: Literal["capture", "render"] = Field(
        description="'capture' for inputs (microphones), 'render' for outputs (speakers/headphones)"
    )
    form_factor: EndpointFormFactor = Field(
        default=EndpointFormFactor.UNKNOWN, description="Hardware form factor"
    )
    is_default_console: bool = Field(
        default=False, description="Whether this is the Windows Console default endpoint"
    )
    is_default_communications: bool = Field(
        default=False, description="Whether this is the Windows Communications default endpoint"
    )
    is_vb_cable_input: bool = Field(
        default=False, description="Whether this is 'CABLE Input (VB-Audio Virtual Cable)'"
    )
    is_vb_cable_output: bool = Field(
        default=False, description="Whether this is 'CABLE Output (VB-Audio Virtual Cable)'"
    )
    is_physical: bool = Field(
        default=True, description="True if physical hardware; False if virtual driver like VB-CABLE"
    )


class AudioDeviceInventory(BaseModel):
    """Full inventory of host audio capture and render endpoints."""

    capture_devices: list[AudioEndpointInfo] = Field(default_factory=list)
    render_devices: list[AudioEndpointInfo] = Field(default_factory=list)
    default_console_render_id: str | None = None
    default_communications_render_id: str | None = None
    default_console_capture_id: str | None = None
    default_communications_capture_id: str | None = None
    vb_cable_installed: bool = False
    physical_microphone_detected: bool = False
    physical_headphones_detected: bool = False


class AudioRoutingConfig(BaseModel):
    """Audio routing configuration requested by the user or UI."""

    input_device_id: str = Field(description="Physical microphone capture endpoint ID")
    output_device_id: str = Field(description="Engine output endpoint ID (target: CABLE Input)")
    monitor_device_id: str | None = Field(
        default=None, description="Headphones render endpoint ID for live monitoring"
    )
    monitor_enabled: bool = Field(
        default=False, description="Whether live headphone monitoring is active"
    )


class RoutingValidationResult(BaseModel):
    """Result of safe routing and feedback loop detection."""

    valid: bool
    errors: list[str] = Field(
        default_factory=list, description="Fatal routing conflicts that block stream"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings or configuration guidance"
    )
    checklist: list[str] = Field(
        default_factory=list,
        description="Step-by-step instructions for Windows & Calling App settings",
    )


class AudioRoutingStatus(BaseModel):
    """Live status of the audio routing layer and safety controls."""

    is_running: bool = False
    is_muted: bool = False
    is_passthrough: bool = False
    active_config: AudioRoutingConfig | None = None
    input_device: AudioEndpointInfo | None = None
    output_device: AudioEndpointInfo | None = None
    monitor_device: AudioEndpointInfo | None = None
    vb_cable_detected: bool = False
    feedback_loop_detected: bool = False
    device_error: str | None = None


class HotkeyStatus(BaseModel):
    """Status and conflict report for global native hotkeys."""

    panic_hotkey: str = "Ctrl+Shift+X"
    passthrough_hotkey: str = "Ctrl+Shift+P"
    panic_registered: bool = False
    passthrough_registered: bool = False
    conflicts: list[str] = Field(default_factory=list)


class TestRecordRequest(BaseModel):
    """Request to record a short self-verification test sample."""

    __test__ = False

    device_id: str | None = Field(
        default=None, description="Endpoint ID to record from. Defaults to configured input device."
    )
    duration_seconds: float = Field(
        default=5.0, ge=1.0, le=10.0, description="Recording duration (fixed 5s recommended)"
    )


class TestRecordResponse(BaseModel):
    """Metadata for an opt-in test recording sample."""

    __test__ = False

    recording_id: str
    duration_seconds: float
    file_path: str
    created_at: datetime
    expires_in_seconds: int = 60
    retention_notice: str = (
        "Audio sample retained locally for self-verification only. "
        "Automatically purged from disk after 60 seconds."
    )
