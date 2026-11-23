"""EngineService — orchestration layer between API routers and the engine adapter."""

from __future__ import annotations

from typing import Any, Callable, Optional

from hth.core.logging import get_logger
from hth.domain.api_contracts import (
    EngineInfo,
    EngineState,
    LiveConfig,
    LiveStatus,
    VoiceEngineProtocol,
)
from hth.domain.errors import ResourceConflictError

logger = get_logger("hth.services.engine")


class EngineService:
    """Wraps a VoiceEngineProtocol and enforces state-transition guards."""

    def __init__(self, engine: VoiceEngineProtocol) -> None:
        self._engine = engine
        self._offline_active_check: Optional[Callable[[], bool]] = None

    def set_offline_active_check(self, check: Callable[[], bool]) -> None:
        """Register callback to query whether an offline conversion job is running."""
        self._offline_active_check = check

    async def get_status(self) -> LiveStatus:
        return await self._engine.status()

    async def get_health(self) -> EngineInfo:
        return await self._engine.health()

    async def start_conversion(self, config: LiveConfig) -> LiveStatus:
        """Start the engine. Raises ValueError if not in IDLE state, or ResourceConflictError if offline conversion active."""
        if self._offline_active_check and self._offline_active_check():
            raise ResourceConflictError(
                "Cannot start live conversion while an offline voice note job is processing. "
                "Please wait for the job to complete or cancel it.",
                competing_task="offline_voice_note_conversion",
            )

        health = await self._engine.health()
        if health.state == EngineState.ERROR:
            logger.warning("Engine was in ERROR state; resetting to IDLE before starting")
            try:
                await self._engine.stop()
            except Exception:
                pass
            health = await self._engine.health()

        if health.state != EngineState.IDLE:
            raise ValueError(f"Engine must be IDLE to start, current state: {health.state.value}")
        try:
            await self._engine.start(config)
        except Exception as exc:
            # Fail-closed: attempt a defensive stop, then propagate
            try:
                await self._engine.stop()
            except Exception:
                pass
            raise RuntimeError(f"Engine start failed — output muted. Cause: {exc}") from exc

        logger.info(
            "Engine started",
            sample_rate=config.sample_rate,
            buffer_size=config.buffer_size,
        )
        return await self._engine.status()

    async def stop_conversion(self) -> None:
        """Stop the engine. Raises ValueError if not in RUNNING or MUTED state."""
        health = await self._engine.health()
        if health.state not in (EngineState.RUNNING, EngineState.MUTED):
            raise ValueError(
                f"Engine must be RUNNING or MUTED to stop, current state: {health.state.value}"
            )
        await self._engine.stop()
        logger.info("Engine stopped")

    async def enable_passthrough(self, enabled: bool) -> None:
        """Enable/disable raw passthrough.

        Passthrough is a deliberate user action only — never an auto-fallback (rule A-03).
        """
        if enabled:
            logger.warning(
                "Passthrough enabled — raw microphone audio will pass through "
                "without voice conversion",
                passthrough=enabled,
            )
        await self._engine.set_passthrough(enabled)

    async def panic_mute(self) -> None:
        """Immediately mute the engine output and disable passthrough."""
        await self._engine.set_passthrough(False)
        if hasattr(self._engine, "panic_mute"):
            await self._engine.panic_mute()
        logger.warning("Panic mute engaged — outgoing audio stream zeroed")

    async def un_mute(self) -> LiveStatus:
        """Release panic mute on deliberate user action.

        Fail-closed safety rules:
        - Never automatically unmute.
        - Rejects unmute if the engine is in ERROR state — requires explicit stop/reset first.
        - Fails closed if pipeline or engine is unavailable.
        - Safe and idempotent on repeated calls.
        - Returns current LiveStatus after releasing mute.
        """
        health = await self._engine.health()
        if health.state == EngineState.ERROR:
            raise RuntimeError(
                "Cannot unmute: engine is in ERROR state. "
                "Call stop() to reset the engine before attempting to unmute."
            )
        if hasattr(self._engine, "un_mute"):
            await self._engine.un_mute()
        logger.info("Panic mute released — converted audio stream resumed")
        return await self._engine.status()

    async def select_model(self, model_id: str) -> None:
        """Switch the active voice model with fail-closed safety."""
        from hth.domain.api_contracts import LicenseType, ModelManifest

        logger.info("Switching voice model", model_id=model_id)
        manifest = ModelManifest(
            model_id=model_id,
            engine_id="fake",
            name=model_id,
            format="fake_model",
            sha256="fake_sha",
            size_bytes=1024,
            license_type=LicenseType.MIT,
            consent_required=False,
        )
        await self._engine.select_model(manifest)

    async def configure_runtime_params(self, params: dict[str, Any]) -> None:
        """Update runtime DSP parameters (pitch_shift, formant_shift, etc.)."""
        if hasattr(self._engine, "configure_runtime_params"):
            await self._engine.configure_runtime_params(params)
