"""Process supervisor managing child worker lifecycle, heartbeats, and restarts."""

import asyncio
import multiprocessing as mp
import os
import secrets
import subprocess
import time
from collections import deque
from typing import Any, Optional, cast

from hth.adapters.beatrice.capabilities import BEATRICE_CAPABILITIES
from hth.adapters.beatrice.worker import main as worker_main
from hth.adapters.ipc.connection import IPCConnection
from hth.adapters.ipc.messages import CommandType, IPCMessage, MessageType
from hth.core.logging import get_logger
from hth.domain.contracts import (
    AudioMetrics,
    EngineHealth,
    EngineState,
    EngineStatus,
    LiveConfig,
)
from hth.domain.errors import (
    AuthenticationError,
    CommandError,
    EngineCrashError,
    TimeoutError,
)

logger = get_logger("hth.adapters.beatrice.supervisor")

# Supervisor policy constants
HEARTBEAT_INTERVAL_SEC: float = 1.0
HEARTBEAT_TIMEOUT_SEC: float = 3.0
COMMAND_TIMEOUT_SEC: float = 2.0
MAX_RESTARTS_IN_WINDOW: int = 3
RESTART_WINDOW_SEC: float = 60.0
RESTART_BACKOFF_DELAYS: list[float] = [0.5, 1.5, 4.5]


class BeatriceProcessSupervisor:
    """Supervises the isolated Beatrice child process."""

    def __init__(self) -> None:
        self._process: Optional[Any] = None
        self._parent_conn: Optional[Any] = None
        self._ipc: Optional[IPCConnection] = None
        self._session_token: str = secrets.token_hex(32)

        self._state: EngineState = EngineState.UNINITIALIZED
        self._is_passthrough: bool = False
        self._current_model: str = "jvs"
        self._active_config: Optional[LiveConfig] = None
        self._last_error: Optional[str] = None

        self._restart_timestamps: deque[float] = deque()
        self._last_heartbeat_time: float = 0.0
        self._metrics: AudioMetrics = AudioMetrics()

        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        self._closing = False

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process and self._process.is_alive() else None

    async def start(self) -> None:
        """Launches the worker child process and initiates monitoring."""
        async with self._lock:
            if self._process and self._process.is_alive():
                return
            await self._spawn_worker()
            self._closing = False
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _spawn_worker(self) -> None:
        """Spawns an isolated child process via multiprocessing spawn context."""
        ctx = mp.get_context("spawn")
        p_conn, c_conn = ctx.Pipe(duplex=True)
        self._parent_conn = p_conn
        self._ipc = IPCConnection(cast(Any, p_conn), self._session_token)

        self._process = ctx.Process(
            target=worker_main,
            args=(
                c_conn,
                self._session_token,
                os.environ.get("HTH_USE_FAKE_AUDIO_DEVICES") == "1",
            ),
            name="HTH-BeatriceWorker",
            daemon=True,
        )
        if self._process is not None:
            self._process.start()
        # Close child side in parent
        try:
            c_conn.close()
        except Exception:
            pass

        # Perform handshake authentication in background thread with early-exit detection
        handshake_timeout = float(os.environ.get("HTH_BEATRICE_HANDSHAKE_TIMEOUT", "20.0"))
        loop = asyncio.get_running_loop()
        try:
            auth_future = loop.run_in_executor(
                None, self._ipc.authenticate_server, self._session_token, handshake_timeout
            )
            start_wait = time.time()
            while not auth_future.done():
                if self._process is not None and not self._process.is_alive():
                    exit_code = self._process.exitcode
                    logger.error(
                        "Beatrice worker process terminated prematurely before handshake",
                        exit_code=exit_code,
                    )
                    raise ProcessLookupError(
                        f"Worker process exited prematurely with code {exit_code}"
                    )
                await asyncio.sleep(0.05)
                if time.time() - start_wait > handshake_timeout + 1.0:
                    break

            await auth_future
        except Exception as exc:
            self._state = EngineState.MUTED_ERROR
            self._last_error = f"Authentication handshake failed: {exc}"
            await self._terminate_process_tree()
            raise AuthenticationError(f"Worker handshake failed: {exc}") from exc

        self._state = EngineState.IDLE
        self._last_heartbeat_time = time.time()

    async def shutdown(self) -> None:
        """Gracefully shuts down worker and kills entire process tree."""
        self._closing = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        if self._ipc and self._process and self._process.is_alive():
            try:
                # Issue graceful shutdown command
                await self.send_command(CommandType.SHUTDOWN, timeout=1.0)
            except Exception:
                pass

        await self._terminate_process_tree()
        self._state = EngineState.SHUTDOWN

    async def _terminate_process_tree(self) -> None:
        """Ensures worker process and all descendant processes are killed cleanly."""
        if not self._process:
            return

        pid = self._process.pid
        if pid and self._process.is_alive():
            # On Windows, use taskkill /F /T /PID to terminate the whole tree
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        check=False,
                        timeout=3.0,
                    )
                except Exception:
                    pass

            # Standard process terminate
            try:
                self._process.terminate()
                self._process.join(timeout=1.0)
            except Exception:
                pass

            if self._process.is_alive():
                try:
                    self._process.kill()
                    self._process.join(timeout=1.0)
                except Exception:
                    pass

        if self._ipc:
            self._ipc.close()
            self._ipc = None
        self._process = None

    async def send_command(
        self,
        command: CommandType,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = COMMAND_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        """Sends an authenticated command to child worker and awaits typed response."""
        if not self._ipc or not self._process or not self._process.is_alive():
            raise EngineCrashError("Engine child process is not running")

        msg = IPCMessage(
            message_type=MessageType.COMMAND,
            command=command,
            payload=payload or {},
            token=self._session_token,
        )

        try:
            await self._ipc.async_send_message(msg)
            resp = await self._ipc.async_recv_message(timeout=timeout)
        except TimeoutError:
            raise TimeoutError(
                f"Command '{command.value}' timed out after {timeout}s",
                timeout_seconds=timeout,
            )
        except Exception as exc:
            raise EngineCrashError(
                f"IPC communication failure during '{command.value}': {exc}"
            ) from exc

        if resp.message_type == MessageType.ERROR:
            err_msg = resp.payload.get("error", "Unknown engine error")
            raise CommandError(err_msg, command=command.value)

        return resp.payload

    async def _heartbeat_loop(self) -> None:
        """Periodic background task sending heartbeat pings."""
        while not self._closing:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            if self._closing or not self._ipc or not self._process or not self._process.is_alive():
                continue

            try:
                ping_msg = IPCMessage(message_type=MessageType.HEARTBEAT_PING)
                await self._ipc.async_send_message(ping_msg)
                pong = await self._ipc.async_recv_message(timeout=HEARTBEAT_TIMEOUT_SEC)
                if pong.message_type == MessageType.HEARTBEAT_PONG:
                    self._last_heartbeat_time = time.time()
                    metrics_data = pong.payload.get("metrics")
                    if metrics_data:
                        self._metrics = AudioMetrics(**metrics_data)
            except Exception:
                # Check for heartbeat loss
                time_since_last = time.time() - self._last_heartbeat_time
                if time_since_last > HEARTBEAT_TIMEOUT_SEC:
                    self._state = EngineState.MUTED_ERROR
                    self._last_error = f"Heartbeat lost ({time_since_last:.1f}s elapsed)"
                    # Trigger restart if not closing
                    if not self._closing:
                        asyncio.create_task(self._handle_child_failure("Heartbeat timeout"))

    async def _monitor_loop(self) -> None:
        """Monitors child process exit code and health."""
        while not self._closing:
            await asyncio.sleep(0.5)
            if self._closing:
                break

            if self._process and not self._process.is_alive():
                exit_code = getattr(self._process, "exitcode", None)
                self._state = EngineState.MUTED_ERROR
                self._last_error = f"Child process exited unexpectedly with code {exit_code}"
                await self._handle_child_failure(f"Exit code: {exit_code}")

    async def _handle_child_failure(self, reason: str) -> None:
        """Applies bounded restart policy with exponential backoff."""
        async with self._lock:
            if self._closing:
                return

            now = time.time()
            # Clean old restart timestamps outside window
            while self._restart_timestamps and (
                now - self._restart_timestamps[0] > RESTART_WINDOW_SEC
            ):
                self._restart_timestamps.popleft()

            if len(self._restart_timestamps) >= MAX_RESTARTS_IN_WINDOW:
                # Bounded policy reached: Halt in CRASH_LOOP, fail closed, mute stream
                self._state = EngineState.CRASH_LOOP
                self._last_error = f"Crash loop detected: exceeded {MAX_RESTARTS_IN_WINDOW} restarts in {RESTART_WINDOW_SEC}s"
                await self._terminate_process_tree()
                return

            retry_count = len(self._restart_timestamps)
            delay = RESTART_BACKOFF_DELAYS[min(retry_count, len(RESTART_BACKOFF_DELAYS) - 1)]
            self._restart_timestamps.append(now)

            await self._terminate_process_tree()
            await asyncio.sleep(delay)

            try:
                await self._spawn_worker()
                # If an active config was running, attempt restoring it
                if self._active_config:
                    await self.send_command(
                        CommandType.START,
                        {"config": self._active_config.to_dict()},
                        timeout=3.0,
                    )
                    self._state = EngineState.RUNNING
            except Exception as exc:
                self._state = EngineState.MUTED_ERROR
                self._last_error = f"Failed to restart worker: {exc}"

    def get_status(self) -> EngineStatus:
        return EngineStatus(
            state=self._state,
            is_passthrough=self._is_passthrough,
            current_model=self._current_model,
            capabilities=BEATRICE_CAPABILITIES,
            metrics=self._metrics,
            active_config=self._active_config,
            last_error=self._last_error,
        )

    def get_health(self) -> EngineHealth:
        alive = bool(self._process and self._process.is_alive())
        age = time.time() - self._last_heartbeat_time if self._last_heartbeat_time else 0.0
        healthy = (
            alive
            and self._state != EngineState.MUTED_ERROR
            and self._state != EngineState.CRASH_LOOP
            and age <= HEARTBEAT_TIMEOUT_SEC
        )
        return EngineHealth(
            healthy=healthy,
            process_alive=alive,
            process_pid=self.pid,
            last_heartbeat_age_s=age,
            restart_count=len(self._restart_timestamps),
            error_message=self._last_error,
        )
