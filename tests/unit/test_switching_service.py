"""Unit tests for safe model switching: stop/mute, hot-swap, warmup, and fail-closed error recovery."""

from unittest.mock import AsyncMock

import pytest

from hth.adapters.fake_engine import FakeEngine
from hth.domain.api_contracts import EngineState, LiveConfig
from hth.domain.model_contracts import (
    CompatibilityStatus,
    ModelAuditEventType,
    ModelFormat,
    ModelManifestMetadata,
    ModelRecord,
)
from hth.services.engine_service import EngineService
from hth.services.model_service import ModelService
from hth.services.switching_service import ModelSwitchError, SwitchingService


@pytest.fixture
def mock_model():
    return ModelRecord(
        model_id="test_model_123",
        engine_id="fake",
        model_format=ModelFormat.SYNTHETIC_TEST,
        name="Test Voice",
        manifest=ModelManifestMetadata(name="Test Voice"),
        primary_hash="abc",
        size_bytes=1000,
        license_identifier="MIT",
        commercial_eligible=True,
        compatibility_status=CompatibilityStatus.COMPATIBLE,
        install_path="/tmp/test",
    )


@pytest.mark.asyncio
async def test_switch_model_when_engine_idle(mock_model):
    engine = FakeEngine()
    engine_svc = EngineService(engine)

    mock_model_svc = AsyncMock(spec=ModelService)
    mock_model_svc.get_model.return_value = mock_model
    mock_model_svc.list_models.return_value = []

    mock_session = AsyncMock()
    mock_session_factory = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session

    switching_svc = SwitchingService(mock_model_svc, engine_svc, mock_session_factory)

    result = await switching_svc.switch_model("test_model_123")
    assert result.success is True
    assert result.active_model_id == "test_model_123"
    assert result.engine_state == EngineState.IDLE.value
    mock_model_svc.record_audit_event.assert_called_with(
        ModelAuditEventType.MODEL_ACTIVATED,
        model_id="test_model_123",
        payload={"model_name": "Test Voice", "previous_model_id": None, "engine_state": "IDLE"},
    )


@pytest.mark.asyncio
async def test_switch_model_when_engine_running_mutes_and_resumes(mock_model):
    """When engine is actively streaming, model switch must mute/stop, swap, warmup, and resume."""
    engine = FakeEngine()
    engine_svc = EngineService(engine)

    # Start engine
    cfg = LiveConfig(sample_rate=48000, buffer_size=512)
    await engine_svc.start_conversion(cfg)
    assert (await engine_svc.get_status()).state == EngineState.RUNNING

    mock_model_svc = AsyncMock(spec=ModelService)
    mock_model_svc.get_model.return_value = mock_model
    mock_model_svc.list_models.return_value = []

    mock_session = AsyncMock()
    mock_session_factory = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session

    switching_svc = SwitchingService(mock_model_svc, engine_svc, mock_session_factory)

    result = await switching_svc.switch_model("test_model_123")
    assert result.success is True
    # Should have resumed running
    assert (await engine_svc.get_status()).state == EngineState.RUNNING


@pytest.mark.asyncio
async def test_switch_model_fail_closed_on_error(mock_model):
    """If model switch fails during load, output must remain muted (Fail-closed)."""
    engine = FakeEngine()
    engine_svc = EngineService(engine)

    cfg = LiveConfig(sample_rate=48000, buffer_size=512)
    await engine_svc.start_conversion(cfg)

    # Cause engine.select_model to fail
    async def broken_select(m):
        raise RuntimeError("DSP weights corrupt")

    engine.select_model = broken_select

    mock_model_svc = AsyncMock(spec=ModelService)
    mock_model_svc.get_model.return_value = mock_model
    mock_model_svc.list_models.return_value = []

    mock_session = AsyncMock()
    mock_session_factory = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session

    switching_svc = SwitchingService(mock_model_svc, engine_svc, mock_session_factory)

    with pytest.raises(ModelSwitchError, match="Audio output has been safely muted"):
        await switching_svc.switch_model("test_model_123")

    # Output remains stopped/muted — no raw mic passthrough
    status = await engine_svc.get_status()
    assert status.state in (EngineState.IDLE, EngineState.MUTED, EngineState.ERROR)
    assert not engine._passthrough

    # Audit event should record failure
    mock_model_svc.record_audit_event.assert_called_with(
        ModelAuditEventType.MODEL_SWITCH_FAILED,
        model_id="test_model_123",
        payload={"error": "DSP weights corrupt", "previous_model_id": None},
    )
