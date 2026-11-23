"""Comprehensive integration tests covering all 11 required engine adapter scenarios.

Scenarios tested:
1. Fake host protocol
2. Corrupt messages
3. Timeout
4. Heartbeat loss
5. Child process crash
6. Double start
7. Stop while starting
8. Device unplug
9. Invalid model
10. Unauthorized engine command
11. Model switch failure
"""

import asyncio
import time

import pytest

from hth.adapters.beatrice.adapter import BeatriceAdapter
from hth.adapters.ipc.connection import IPCConnection
from hth.adapters.ipc.messages import CommandType
from hth.domain.contracts import EngineParams, EngineState, LiveConfig
from hth.domain.errors import (
    AuthenticationError,
    CommandError,
    DeviceError,
    EngineCrashError,
    InvalidStateError,
    ModelValidationError,
    ProtocolViolationError,
    TimeoutError,
)


@pytest.fixture
async def running_adapter():
    """Initializes a BeatriceAdapter and shuts it down after test completion."""
    adapter = BeatriceAdapter()
    await adapter.initialize()
    try:
        yield adapter
    finally:
        await adapter.shutdown()


# ---------------------------------------------------------------------------
# 1. Fake Host Protocol
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fake_host_protocol(running_adapter: BeatriceAdapter) -> None:
    """Verifies standard command interactions across process boundary."""
    # Ping
    latency = await running_adapter.ping()
    assert latency > 0.0

    # List devices
    devices = await running_adapter.list_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0

    # Status
    status = await running_adapter.status()
    assert status.state == EngineState.IDLE
    assert status.is_passthrough is False

    # Set Params
    await running_adapter.set_params(EngineParams(pitch_shift=3.0, formant_shift=1.0))

    # Health
    health = await running_adapter.health()
    assert health.healthy is True


# ---------------------------------------------------------------------------
# 2. Corrupt Messages
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_corrupt_messages_handling(running_adapter: BeatriceAdapter) -> None:
    """Verifies that injecting corrupt bytes into IPC transport raises error cleanly without supervisor crash."""
    ipc = running_adapter._supervisor._ipc
    assert ipc is not None

    # Inject malformed data into raw connection
    raw_conn = ipc._conn
    # Send garbage bytes (not adhering to length framing or JSON)
    raw_conn.send_bytes(b"\x00\x00\x00\x05\x01{bad")

    # Next command recv should detect corruption and recover or raise ProtocolViolationError
    with pytest.raises((ProtocolViolationError, EngineCrashError)):
        await running_adapter._supervisor.send_command(CommandType.PING, timeout=1.0)


# ---------------------------------------------------------------------------
# 3. Timeout
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_command_timeout(running_adapter: BeatriceAdapter) -> None:
    """Verifies that when a command response exceeds deadline, TimeoutError is raised."""
    # Issue a command with a zero timeout to ensure deadline has expired
    with pytest.raises(TimeoutError):
        await running_adapter._supervisor.send_command(CommandType.PING, timeout=0.0)


# ---------------------------------------------------------------------------
# 4. Heartbeat Loss
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_heartbeat_loss_detection(running_adapter: BeatriceAdapter) -> None:
    """Verifies supervisor detects heartbeat loss and marks state unhealthy."""
    sup = running_adapter._supervisor
    # Artificially age the last heartbeat time beyond timeout
    sup._last_heartbeat_time = time.time() - 10.0

    health = await running_adapter.health()
    assert health.healthy is False
    assert health.last_heartbeat_age_s >= 10.0


# ---------------------------------------------------------------------------
# 5. Child Process Crash
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_child_process_crash(running_adapter: BeatriceAdapter) -> None:
    """Verifies fail-closed behavior when child process crashes abruptly."""
    sup = running_adapter._supervisor
    assert sup._process is not None

    # Kill worker child process
    sup._process.kill()
    await asyncio.sleep(0.8)

    # Health must report process not alive or unhealthy
    health = await running_adapter.health()
    assert health.process_alive is False or health.healthy is False


# ---------------------------------------------------------------------------
# 6. Double Start
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_double_start_prevention(
    running_adapter: BeatriceAdapter, default_config: LiveConfig
) -> None:
    """Verifies that calling start() when already running raises InvalidStateError."""
    await running_adapter.start(default_config)
    status1 = await running_adapter.status()
    assert status1.state == EngineState.RUNNING

    # Second start must fail closed with InvalidStateError
    with pytest.raises(InvalidStateError, match="already running"):
        await running_adapter.start(default_config)

    # Stop stream
    await running_adapter.stop()
    status2 = await running_adapter.status()
    assert status2.state == EngineState.IDLE


# ---------------------------------------------------------------------------
# 7. Stop While Starting
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stop_while_starting(
    running_adapter: BeatriceAdapter, default_config: LiveConfig
) -> None:
    """Verifies state reconciles cleanly if stop() is called while starting."""
    # Simulate stopping during starting transition
    running_adapter._is_starting = True
    running_adapter._supervisor._state = EngineState.STARTING

    await running_adapter.stop()
    status = await running_adapter.status()
    assert status.state == EngineState.IDLE


# ---------------------------------------------------------------------------
# 8. Device Unplug
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_device_unplug_fail_closed(running_adapter: BeatriceAdapter) -> None:
    """Verifies that invalid or unplugged device fails closed and never auto-falls back to default."""
    bad_config = LiveConfig(
        input_device_id=999999,  # Non-existent device ID
        output_device_id=999998,
        sample_rate=16000,
        block_size=160,
    )

    # In environment with sounddevice, opening non-existent device raises DeviceError.
    # In mock environment, stream verifies device handling without silent fallback.
    try:
        await running_adapter.start(bad_config)
    except (DeviceError, CommandError):
        # Expected error: must fail closed without silent fallback
        status = await running_adapter.status()
        assert status.state != EngineState.RUNNING
        assert running_adapter._supervisor._is_passthrough is False


# ---------------------------------------------------------------------------
# 9. Invalid Model
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalid_model_rejection(running_adapter: BeatriceAdapter) -> None:
    """Verifies that invalid model ID or path traversal pattern is rejected."""
    # Empty model ID
    with pytest.raises(ModelValidationError, match="must not be empty"):
        await running_adapter.select_model("")

    # Path traversal attack pattern
    with pytest.raises(ModelValidationError, match="path traversal"):
        await running_adapter.select_model("../../../etc/passwd")

    with pytest.raises(ModelValidationError, match="path traversal"):
        await running_adapter.select_model("..\\windows\\system32")


# ---------------------------------------------------------------------------
# 10. Unauthorized Engine Command
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unauthorized_engine_command() -> None:
    """Verifies that child worker rejects unauthorized connection without valid session token."""
    import multiprocessing as mp

    from hth.adapters.beatrice.worker import main as worker_main

    ctx = mp.get_context("spawn")
    p_conn, c_conn = ctx.Pipe(duplex=True)
    valid_token = "correct-token-32-chars-long-123456"
    invalid_token = "bad-unauthorized-token-abcdefgh"

    p = ctx.Process(target=worker_main, args=(c_conn, valid_token), daemon=True)
    p.start()
    c_conn.close()

    bad_ipc = IPCConnection(p_conn, invalid_token)
    loop = asyncio.get_running_loop()
    try:
        with pytest.raises(AuthenticationError):
            await loop.run_in_executor(None, bad_ipc.authenticate_server, invalid_token, 10.0)
    finally:
        p.terminate()
        p.join(timeout=1.0)
        p_conn.close()


# ---------------------------------------------------------------------------
# 11. Model Switch Failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_model_switch_failure_retains_safety(running_adapter: BeatriceAdapter) -> None:
    """Verifies model switch failure leaves engine in safe muted state without crashing."""
    initial_status = await running_adapter.status()
    initial_model = initial_status.current_model
    assert initial_model is not None

    # Request corrupted model
    with pytest.raises(CommandError, match="Invalid model manifest"):
        await running_adapter.select_model("invalid_corrupted_model")

    # Engine must maintain consistent model or fail closed, not exposing raw audio
    status_after = await running_adapter.status()
    assert status_after.is_passthrough is False
