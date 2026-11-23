"""FakeEngine (SIMULATION) — a fully in-memory simulated engine for development and testing.

LABELED STRICTLY AS: SIMULATION (Rule L-02 / Engine Clearance).
Not for production deployment. Structurally implements VoiceEngineProtocol.
Fail-closed: errors leave the engine in ERROR state with a muted status.
Passthrough is only allowed when RUNNING (never an auto-fallback).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hth.domain.api_contracts import (
    DeviceInfo,
    DeviceType,
    EngineInfo,
    EngineState,
    LiveConfig,
    LiveStatus,
)


class FakeEngine:
    """In-memory fake engine. Thread-safe via asyncio.Lock."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state = EngineState.IDLE
        self._passthrough = False
        self._params: dict[str, Any] = {}
        self._model: Any = None
        self._config: LiveConfig | None = None
        self._start_time: float | None = None
        self._frames = 0

    async def ping(self) -> bool:
        return True

    async def list_devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(
                id="fake-input-0",
                friendly_name="Fake Microphone (HTH Test)",
                device_type=DeviceType.INPUT,
                sample_rates=[16000, 44100, 48000],
                default_sample_rate=48000,
                is_default=True,
            ),
            DeviceInfo(
                id="fake-output-0",
                friendly_name="Fake Speaker (HTH Test)",
                device_type=DeviceType.OUTPUT,
                sample_rates=[44100, 48000],
                default_sample_rate=48000,
                is_default=True,
            ),
        ]

    async def start(self, config: LiveConfig) -> None:
        # Acquire lock only for state read/write; release before any await
        async with self._lock:
            if self._state != EngineState.IDLE:
                raise RuntimeError(f"FakeEngine already running (state={self._state.value})")
            self._state = EngineState.STARTING
            self._config = config

        # Simulate startup outside the lock
        await asyncio.sleep(0.05)

        async with self._lock:
            self._state = EngineState.RUNNING
            self._start_time = time.monotonic()
            self._frames = 0

    async def stop(self) -> None:
        async with self._lock:
            if self._state != EngineState.RUNNING:
                raise RuntimeError(f"FakeEngine not running (state={self._state.value})")
            self._state = EngineState.STOPPING

        await asyncio.sleep(0.02)

        async with self._lock:
            self._state = EngineState.IDLE
            self._start_time = None
            self._config = None

    async def set_passthrough(self, enabled: bool) -> None:
        """Passthrough requires RUNNING state — never an auto-fallback (rule A-03)."""
        async with self._lock:
            if enabled and self._state != EngineState.RUNNING:
                raise RuntimeError(
                    "Passthrough requires a running engine — start the engine first. "
                    "Raw passthrough is a deliberate user action only."
                )
            self._passthrough = enabled

    async def set_params(self, params: dict[str, Any]) -> None:
        async with self._lock:
            self._params.update(params)

    async def configure_runtime_params(self, params: dict[str, Any]) -> None:
        await self.set_params(params)

    async def select_model(self, model: Any) -> None:
        async with self._lock:
            self._model = model

    async def warmup(self) -> None:
        await asyncio.sleep(0.01)

    async def panic_mute(self) -> None:
        async with self._lock:
            self._state = EngineState.MUTED
            self._passthrough = False

    async def un_mute(self) -> None:
        async with self._lock:
            if self._state == EngineState.MUTED:
                self._state = EngineState.RUNNING if self._config is not None else EngineState.IDLE

    async def status(self) -> LiveStatus:
        async with self._lock:
            if self._state == EngineState.RUNNING:
                self._frames += 1
            return LiveStatus(
                state=self._state,
                latency_p50_ms=1.0,
                latency_p95_ms=2.0,
                underruns=0,
                frames_processed=self._frames,
                active_config=self._config,
            )

    async def health(self) -> EngineInfo:
        async with self._lock:
            return EngineInfo(
                engine_id="simulation-fake-0.1.0",
                name="FakeEngine (SIMULATION)",
                version="0.1.0-sim",
                capabilities=["simulation", "passthrough"],
                state=self._state,
            )
