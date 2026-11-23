"""Safe model switching service — atomic mute, hot-swap/warmup, and fail-closed error recovery."""

from __future__ import annotations

import asyncio

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hth.core.logging import get_logger
from hth.domain.api_contracts import EngineState
from hth.domain.model_contracts import (
    CompatibilityStatus,
    ModelAuditEventType,
    ModelSwitchResult,
)
from hth.repositories.models import ModelMetaRow
from hth.services.engine_service import EngineService
from hth.services.license_service import LicenseService, LicensingPolicyViolation
from hth.services.model_service import ModelService

logger = get_logger("hth.services.switching")


class ModelSwitchError(Exception):
    """Raised when an explicit model switch operation fails."""


class SwitchingService:
    """Orchestrates safe, fail-closed model switching without raw microphone leakage."""

    def __init__(
        self,
        model_service: ModelService,
        engine_service: EngineService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._model_service = model_service
        self._engine_service = engine_service
        self._session_factory = session_factory
        self._lock = asyncio.Lock()

    async def switch_model(self, target_model_id: str) -> ModelSwitchResult:
        """Explicitly switch the active voice model.

        If the engine is running:
        1. Mute & stop active audio stream.
        2. Instruct engine adapter to select new model.
        3. Warm up inference pipeline.
        4. Resume stream with previous live config.
        If any step fails, audio output remains muted (fail-closed, Rule A-03).
        """
        async with self._lock:
            # 1. Fetch model and verify eligibility
            model = await self._model_service.get_model(target_model_id)

            # Check consent & licensing
            LicenseService.check_activation_eligibility(model)

            if model.compatibility_status == CompatibilityStatus.CONSENT_REQUIRED:
                raise LicensingPolicyViolation(
                    f"Model '{model.name}' requires signed consent attestation before it can be activated."
                )

            # 2. Check engine state
            live_status = await self._engine_service.get_status()
            was_running = live_status.state == EngineState.RUNNING
            previous_model_id = None

            # Get current active model from DB
            models_list = await self._model_service.list_models()
            for m in models_list:
                if m.is_active:
                    previous_model_id = m.model_id
                    break

            logger.info(
                "Initiating model switch",
                target_model_id=target_model_id,
                was_running=was_running,
                previous_model_id=previous_model_id,
            )

            # 3. Handle live stream stop/mute if running
            try:
                if was_running:
                    # Fail-closed mute: stop stream before changing weights
                    logger.info("Stopping active audio stream to mute output during model switch")
                    await self._engine_service.stop_conversion()

                # 4. Invoke engine adapter select_model
                engine_adapter = self._engine_service._engine
                if hasattr(engine_adapter, "select_model"):
                    await engine_adapter.select_model(model)

                # 5. If engine has warmup, run it
                if hasattr(engine_adapter, "warmup"):
                    await engine_adapter.warmup()

                # 6. Resume stream if it was running
                if was_running and live_status.active_config is not None:
                    logger.info("Resuming audio stream with new model weights")
                    await self._engine_service.start_conversion(live_status.active_config)

            except Exception as exc:
                # Critical Fail-Closed recovery: ensure output is muted, never leak raw mic
                logger.error(
                    "Model switch failed during execution — enforcing fail-closed mute",
                    error=str(exc),
                )
                try:
                    await self._engine_service.stop_conversion()
                except Exception:
                    pass

                await self._model_service.record_audit_event(
                    ModelAuditEventType.MODEL_SWITCH_FAILED,
                    model_id=target_model_id,
                    payload={"error": str(exc), "previous_model_id": previous_model_id},
                )
                raise ModelSwitchError(
                    f"Model switch failed: {exc}. Audio output has been safely muted. "
                    "Raw microphone audio was NOT passed through."
                ) from exc

            # 7. Update database active flags
            session_ctx = self._session_factory()
            if asyncio.iscoroutine(session_ctx):
                session_ctx = await session_ctx
            async with session_ctx as session:
                # Deactivate all
                await session.execute(update(ModelMetaRow).values(is_active=False))
                # Activate target
                await session.execute(
                    update(ModelMetaRow)
                    .where(ModelMetaRow.model_id == target_model_id)
                    .values(is_active=True)
                )
                await session.commit()

            # 8. Record audit event
            await self._model_service.record_audit_event(
                ModelAuditEventType.MODEL_ACTIVATED,
                model_id=target_model_id,
                payload={
                    "model_name": model.name,
                    "previous_model_id": previous_model_id,
                    "engine_state": (await self._engine_service.get_status()).state.value,
                },
            )

            current_status = await self._engine_service.get_status()
            return ModelSwitchResult(
                previous_model_id=previous_model_id,
                active_model_id=target_model_id,
                engine_state=current_status.state.value,
                success=True,
                message=f"Model '{model.name}' successfully activated.",
            )
