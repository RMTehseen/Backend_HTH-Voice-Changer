"""Unit tests for FakeEngine."""

import pytest

from hth.adapters.fake_engine import FakeEngine
from hth.domain.api_contracts import EngineState, LiveConfig


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.mark.asyncio
async def test_ping(engine: FakeEngine) -> None:
    assert await engine.ping() is True


@pytest.mark.asyncio
async def test_initial_state_idle(engine: FakeEngine) -> None:
    status = await engine.status()
    assert status.state == EngineState.IDLE


@pytest.mark.asyncio
async def test_start_stop_cycle(engine: FakeEngine) -> None:
    config = LiveConfig()
    await engine.start(config)
    status = await engine.status()
    assert status.state == EngineState.RUNNING

    await engine.stop()
    status = await engine.status()
    assert status.state == EngineState.IDLE


@pytest.mark.asyncio
async def test_double_start_raises(engine: FakeEngine) -> None:
    config = LiveConfig()
    await engine.start(config)
    with pytest.raises(RuntimeError, match="already running"):
        await engine.start(config)


@pytest.mark.asyncio
async def test_stop_when_idle_raises(engine: FakeEngine) -> None:
    with pytest.raises(RuntimeError, match="not running"):
        await engine.stop()


@pytest.mark.asyncio
async def test_passthrough_blocked_when_idle(engine: FakeEngine) -> None:
    with pytest.raises(RuntimeError, match="Passthrough requires a running engine"):
        await engine.set_passthrough(True)


@pytest.mark.asyncio
async def test_passthrough_allowed_when_running(engine: FakeEngine) -> None:
    await engine.start(LiveConfig())
    await engine.set_passthrough(True)  # should not raise


@pytest.mark.asyncio
async def test_frames_increment_when_running(engine: FakeEngine) -> None:
    await engine.start(LiveConfig())
    s1 = await engine.status()
    s2 = await engine.status()
    assert s2.frames_processed > s1.frames_processed


@pytest.mark.asyncio
async def test_list_devices_returns_two(engine: FakeEngine) -> None:
    devices = await engine.list_devices()
    assert len(devices) == 2


@pytest.mark.asyncio
async def test_health_returns_engine_info(engine: FakeEngine) -> None:
    info = await engine.health()
    assert "fake" in info.engine_id.lower()
    assert "FakeEngine" in info.name
