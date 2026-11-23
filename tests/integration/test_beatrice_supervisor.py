"""Integration tests for BeatriceProcessSupervisor process lifecycle and recovery."""

import asyncio
import time

import pytest

from hth.adapters.beatrice.supervisor import BeatriceProcessSupervisor
from hth.adapters.ipc.messages import CommandType
from hth.domain.contracts import EngineState


@pytest.mark.asyncio
async def test_supervisor_lifecycle_and_ping() -> None:
    """Verifies that supervisor spawns worker, authenticates, and executes ping."""
    sup = BeatriceProcessSupervisor()
    try:
        await sup.start()
        assert sup.state == EngineState.IDLE
        assert sup.pid is not None

        # Ping across process boundary
        res = await sup.send_command(CommandType.PING)
        assert "pong" in res

        # Check health
        health = sup.get_health()
        assert health.healthy is True
        assert health.process_alive is True
        assert health.process_pid == sup.pid
    finally:
        await sup.shutdown()

    assert sup.state == EngineState.SHUTDOWN
    assert sup.pid is None


@pytest.mark.asyncio
async def test_supervisor_child_process_crash_detection() -> None:
    """Verifies supervisor detects an abrupt child process crash and fails closed."""
    sup = BeatriceProcessSupervisor()
    try:
        await sup.start()
        child_pid = sup.pid
        assert child_pid is not None

        # Terminate child abruptly
        if sup._process:
            sup._process.kill()

        # Allow monitor loop to detect death
        await asyncio.sleep(0.8)

        # Supervisor must detect termination and fail closed
        health = sup.get_health()
        assert health.healthy is False
        assert sup.state in (EngineState.MUTED_ERROR, EngineState.IDLE, EngineState.STARTING)
    finally:
        await sup.shutdown()


@pytest.mark.asyncio
async def test_supervisor_crash_loop_policy() -> None:
    """Verifies that supervisor enters CRASH_LOOP after exceeding restart limit."""
    sup = BeatriceProcessSupervisor()
    try:
        await sup.start()

        # Simulate 3 consecutive child failures rapidly
        for _ in range(3):
            sup._restart_timestamps.append(time.time())

        # Trigger child failure handler
        await sup._handle_child_failure("Simulated crash")

        # Must halt in CRASH_LOOP and not spawn further
        assert sup.state == EngineState.CRASH_LOOP
        assert "Crash loop detected" in (sup._last_error or "")
    finally:
        await sup.shutdown()
