"""Audio device inventory, routing validation, and panic controls router.

Phase 4 endpoints:
- Stable MMDevice inventory with GUIDs and Windows default roles.
- Feedback loop validation & safe routing application.
- Global panic stop & natural voice passthrough controls.
- 5-second opt-in test recording service.
- Hotkey status and conflict reporting.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from hth.api.dependencies import get_engine_service
from hth.audio.devices import get_device_inventory
from hth.audio.hotkeys import get_hotkey_manager
from hth.audio.recorder import get_test_recorder
from hth.audio.router import get_audio_router
from hth.domain.api_contracts import DeviceInfo
from hth.domain.audio_contracts import (
    AudioDeviceInventory,
    AudioRoutingConfig,
    AudioRoutingStatus,
    HotkeyStatus,
    RoutingValidationResult,
    TestRecordRequest,
    TestRecordResponse,
)
from hth.services.engine_service import EngineService

router = APIRouter(prefix="/api/v1/devices", tags=["devices"])


@router.get("/", response_model=list[DeviceInfo])
async def list_devices(svc: EngineService = Depends(get_engine_service)) -> list[DeviceInfo]:
    """List audio devices (legacy / simplified endpoint)."""
    return await svc._engine.list_devices()


@router.get("/inventory", response_model=AudioDeviceInventory)
async def get_inventory() -> AudioDeviceInventory:
    """Get full Windows Core Audio device inventory with persistent GUIDs and default roles."""
    return get_device_inventory()


@router.get("/routing/status", response_model=AudioRoutingStatus)
async def get_routing_status() -> AudioRoutingStatus:
    """Get live audio routing configuration, endpoint statuses, and feedback loop flags."""
    return get_audio_router().get_status()


@router.post("/routing/validate", response_model=RoutingValidationResult)
async def validate_routing(config: AudioRoutingConfig) -> RoutingValidationResult:
    """Validate proposed routing configuration against accidental feedback loops."""
    return get_audio_router().validate_routing(config)


@router.post("/routing/apply", response_model=RoutingValidationResult)
async def apply_routing(config: AudioRoutingConfig) -> RoutingValidationResult:
    """Validate and apply audio routing configuration."""
    result = get_audio_router().apply_routing(config)
    if not result.valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Audio routing configuration rejected due to feedback loop or hardware conflict"
                ),
                "errors": result.errors,
                "warnings": result.warnings,
            },
        )
    return result


@router.post("/panic-stop")
async def panic_stop() -> dict[str, str]:
    """Trigger immediate audio panic stop (mute output stream)."""
    get_audio_router().trigger_panic_stop()
    return {"status": "muted", "message": "Audio output stream muted immediately"}


@router.post("/passthrough")
async def toggle_passthrough(enabled: bool | None = None) -> dict[str, Any]:
    """Toggle intentional natural voice passthrough."""
    state = get_audio_router().toggle_passthrough(enabled)
    return {
        "status": "ok",
        "is_passthrough": state,
        "warning": "RAW VOICE PASSTHROUGH ACTIVE" if state else "AI voice conversion active",
    }


@router.post("/test-record", response_model=TestRecordResponse)
async def create_test_recording(req: TestRecordRequest) -> TestRecordResponse:
    """Create a 5-second opt-in test audio sample for self-verification."""
    return get_test_recorder().record_test_sample(req)


@router.get("/hotkeys", response_model=HotkeyStatus)
async def get_hotkeys_status() -> HotkeyStatus:
    """Get status of native background global hotkeys and detect any conflicts."""
    return get_hotkey_manager().get_status()
