"""Beatrice V2 VoiceEngine Protocol adapter with isolated process supervision."""

import time
from typing import Optional

from hth.adapters.beatrice.capabilities import BEATRICE_CAPABILITIES
from hth.adapters.beatrice.supervisor import BeatriceProcessSupervisor
from hth.adapters.ipc.messages import CommandType
from hth.domain.contracts import (
    DeviceInfo,
    EngineCapabilities,
    EngineHealth,
    EngineParams,
    EngineState,
    EngineStatus,
    LiveConfig,
)
from hth.domain.errors import (
    CommandError,
    InvalidStateError,
    ModelValidationError,
)

# PERSONAL_POC_ONLY marker and commercial distribution flag
PERSONAL_POC_ONLY: bool = True
COMMERCIAL_DISTRIBUTION_ENABLED: bool = False


class BeatriceAdapter:
    """VoiceEngine protocol implementation managing Beatrice V2 via supervisor."""

    def __init__(self, supervisor: Optional[BeatriceProcessSupervisor] = None) -> None:
        self._supervisor = supervisor or BeatriceProcessSupervisor()
        self._is_starting = False

    def get_capabilities(self) -> EngineCapabilities:
        """Returns Beatrice engine capabilities."""
        return BEATRICE_CAPABILITIES

    async def initialize(self) -> None:
        """Initializes supervisor and spawns child process."""
        await self._supervisor.start()

    async def shutdown(self) -> None:
        """Gracefully halts stream and terminates child worker process."""
        await self._supervisor.shutdown()

    async def ping(self) -> float:
        """Measures round-trip IPC latency in milliseconds."""
        t0 = time.perf_counter()
        await self._supervisor.send_command(CommandType.PING)
        return (time.perf_counter() - t0) * 1000.0

    async def list_devices(self) -> list[DeviceInfo]:
        """Lists available audio devices discovered in child process."""
        resp = await self._supervisor.send_command(CommandType.LIST_DEVICES)
        devices_data = resp.get("devices", [])
        return [DeviceInfo(**d) for d in devices_data]

    async def start(self, config: LiveConfig) -> None:
        """Starts real-time voice conversion stream."""
        current_state = self._supervisor.state
        if current_state == EngineState.RUNNING:
            raise InvalidStateError("Engine is already running", current_state=current_state.value)

        # Validate params against capabilities
        self._validate_params(config.params)

        self._is_starting = True
        try:
            self._supervisor._state = EngineState.STARTING
            self._supervisor._active_config = config
            await self._supervisor.send_command(
                CommandType.START,
                {"config": config.to_dict()},
            )
            self._supervisor._state = EngineState.RUNNING
        finally:
            self._is_starting = False

    async def stop(self) -> None:
        """Stops active conversion stream and mutes output."""
        if self._is_starting:
            # Stop while starting: reconcile gracefully
            self._supervisor._state = EngineState.STOPPING

        await self._supervisor.send_command(CommandType.STOP)
        self._supervisor._state = EngineState.IDLE
        self._supervisor._active_config = None

    async def set_passthrough(self, enabled: bool) -> None:
        """Toggles user-initiated natural voice passthrough."""
        resp = await self._supervisor.send_command(
            CommandType.SET_PASSTHROUGH,
            {"enabled": enabled},
        )
        self._supervisor._is_passthrough = bool(resp.get("is_passthrough", False))

    async def set_params(self, params: EngineParams) -> None:
        """Updates DSP parameters (pitch, formant, volume, gate)."""
        self._validate_params(params)
        await self._supervisor.send_command(
            CommandType.SET_PARAMS,
            {"params": params.to_dict()},
        )
        if self._supervisor._active_config:
            self._supervisor._active_config.params = params

    async def select_model(self, model_id: str) -> None:
        """Switches active voice model atomically with fail-closed muting."""
        if not model_id or not model_id.strip():
            raise ModelValidationError("Model ID must not be empty", model_id=model_id)

        # Basic path traversal and character check
        if ".." in model_id or "/" in model_id or "\\" in model_id:
            raise ModelValidationError(f"Invalid model path traversal pattern: {model_id}")

        await self._supervisor.send_command(
            CommandType.SELECT_MODEL,
            {"model_id": model_id},
        )
        self._supervisor._current_model = model_id

    async def status(self) -> EngineStatus:
        """Returns engine state, passthrough flag, and audio metrics."""
        return self._supervisor.get_status()

    async def health(self) -> EngineHealth:
        """Returns diagnostic process and heartbeat health."""
        return self._supervisor.get_health()

    def _validate_params(self, params: EngineParams) -> None:
        """Validates parameter bounds against engine capabilities."""
        caps = self.get_capabilities()
        p_min, p_max = caps.pitch_range
        if not (p_min <= params.pitch_shift <= p_max):
            raise CommandError(
                f"Pitch shift {params.pitch_shift} out of supported range [{p_min}, {p_max}]",
                command="set_params",
            )

        f_min, f_max = caps.formant_range
        if not (f_min <= params.formant_shift <= f_max):
            raise CommandError(
                f"Formant shift {params.formant_shift} out of supported range [{f_min}, {f_max}]",
                command="set_params",
            )
